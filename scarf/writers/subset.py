import os
from typing import Any

import numpy as np
import zarr

from ..storage.types import as_zarr_array, as_zarr_group
from ..storage.arrays import create_numeric_array, create_zarr_obj_array
from ..storage.count_matrix import CountMatrixPolicy
from ..storage.io_policy import StorageIoPolicy
from ..storage.layout import count_array_spec
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    is_local_zarr_path,
    resolve_storage_profile,
)
from ..storage.schema import create_zarr_count_assay
from ..storage.sharding import write_dense_in_shard_rows
from ..storage.stores import load_zarr
from ..utils.logging import logger


def _source_assay_types(assay: Any, _in_workspace: str | None = None) -> dict[str, str]:
    """Read persisted ``assayTypes`` from the source store when available.

    ``assay.z`` is the assay group. For legacy stores its parent is the Zarr
    root; for workspace stores the parent is the workspace group. Both places
    hold ``assayTypes``.
    """
    try:
        parent = assay.z.parent
    except Exception:
        return {}
    if parent is None:
        return {}
    raw = parent.attrs.get("assayTypes", {})
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def subset_assay_zarr(
    zarr_loc: ZarrLocation,
    in_grp: str,
    out_grp: str,
    cells_idx: np.ndarray,
    feat_idx: np.ndarray,
    storage_options: dict[str, Any] | None = None,
    mem_budget: int | str | None = None,
    nthreads: int | None = None,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
    io: StorageIoPolicy | None = None,
) -> None:
    """Selects a subset of the data in an assay in the specified Zarr
    hierarchy.

    For the arguments `cells_idx` and `feat_idx`, refer to the documentation for numpy.split:
    https://numpy.org/doc/stable/reference/generated/numpy.split.html

    Args:
        zarr_loc: The file name for the Zarr hierarchy.
        in_grp: Group in Zarr hierarchy to subset.
        out_grp: Group name in Zarr hierarchy to write subsetted assay to.
        cells_idx: Indices of cells to keep in the subset.
        feat_idx: Indices of features to keep in the subset.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy. When None, the default
                unitBytes and chunkBytes plan is used.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
    Returns:
        None
    """
    from ..storage.budget import resolve_budget

    resources = resolve_budget(mem_budget, nthreads)
    resolved_profile = resolve_storage_profile(zarr_loc, profile)
    z = load_zarr(zarr_loc, "r+", storage_options=storage_options)
    ig = as_zarr_array(z[in_grp], name=in_grp)
    spec = count_array_spec(
        len(cells_idx),
        len(feat_idx),
        dtype="uint32",
        profile=resolved_profile,
        policy=policy,
    )
    og = create_numeric_array(z, out_grp, spec)
    write_dense_in_shard_rows(
        og,
        lambda start, end: np.asarray(
            ig.get_orthogonal_selection((cells_idx[start:end], feat_idx))
        ),
        msg="Subsetting assay",
        resources=resources,
        io=io,
    )
    return None


class SubsetZarr:
    """Split Zarr file using a subset of cells.

    Args:
        zarr_loc: Path for the output (subsetted) Zarr file
        assays: Source assays to be subsetted. These assays must be from the same dataset
        in_workspace: Source workspace name (None for legacy layout).
        out_workspace: Target workspace name in the output Zarr file.
        cell_key: Name of a boolean column in cell metadata. The cells with value True are included in the
                  subset. Only used when cell_idx is None.
        cell_idx: Explicit indices of cells to include in the subset.
        reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                           are set as True as in the 'I' column. To keep the filtering information set the value for
                           this parameter to False. (Default value: True)
        overwrite_existing_file: If True, then overwrites the existing data. (Default value: False)
        overwrite_cell_data: If True, then overwrites cell data (Default value: False)
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy. When None, the default
                unitBytes and chunkBytes plan is used.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.
    """

    def __init__(
        self,
        zarr_loc: ZarrLocation,
        assays: list[Any],
        in_workspace: str | None = None,
        out_workspace: str | None = None,
        cell_key: str | None = None,
        cell_idx: np.ndarray | None = None,
        reset_cell_filter: bool = True,
        overwrite_existing_file: bool = False,
        overwrite_cell_data: bool = False,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
        io: StorageIoPolicy | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget

        self.resetCells = reset_cell_filter
        self.overFn = overwrite_existing_file
        self.overCells = overwrite_cell_data
        self.inWorkspace = in_workspace
        self.outWorkspace = out_workspace
        self.storage_options = storage_options
        assay_resources = [
            assay.resources for assay in assays if hasattr(assay, "resources")
        ]
        self.resources = resolve_budget(
            (
                mem_budget
                if mem_budget is not None
                else (
                    min(resource.memoryBytes for resource in assay_resources)
                    if assay_resources
                    else None
                )
            ),
            (
                nthreads
                if nthreads is not None
                else (
                    min(resource.workers for resource in assay_resources)
                    if assay_resources
                    else None
                )
            ),
        )
        self.profile = resolve_storage_profile(zarr_loc, profile)
        self.policy = policy
        self.io = io
        self.z = self._check_files(zarr_loc)
        self.assays = self._check_assays(assays)
        self.cellIdx = self._check_idx(cell_key, cell_idx)

    def _check_files(self, zarr_loc: ZarrLocation) -> zarr.Group:
        if (
            is_local_zarr_path(zarr_loc)
            and isinstance(zarr_loc, str)
            and os.path.isdir(zarr_loc)
            and self.overFn is False
        ):
            raise ValueError(
                f"Zarr file with name: {zarr_loc} already exists.\n"
                f"If you want to overwrite it then please set  overwrite_existing_file to True. "
                f"No subsetting was performed."
            )
        return load_zarr(
            zarr_loc=zarr_loc, mode="w", storage_options=self.storage_options
        )

    @staticmethod
    def _check_assays(assays: list[Any]) -> list[Any]:
        # if type(assays) != list:
        if isinstance(assays, list) is False:
            raise TypeError(
                "Value for parameter `assays` should be a list. For example, `[ds.RNA]`"
            )
        n = []
        for assay in assays:
            try:
                n.append(assay.cells.N)
            except AttributeError:
                raise ValueError(
                    "Please make sure you are passing actual assay objects and not assay names. "
                    "For example, `[ds.RNA]`"
                )
        if len(set(n)) != 1:
            raise ValueError(
                f"ERROR: Provided assays do not have the same numer of cells. Please make "  # noqa: F541
                f"sure that the assays are from the same DataStore."  # noqa: F541
            )
        return assays

    def _check_idx(
        self, cell_key: str | None, cell_idx: np.ndarray | None
    ) -> np.ndarray:
        if cell_key is None and cell_idx is None:
            raise ValueError("Both `cell_key` and `cell_idx` parameters cannot be None")
        if cell_idx is None:
            resolved: np.ndarray | None = None
            for assay in self.assays:
                try:
                    idx = assay.cells.fetch_all(cell_key)
                except KeyError:
                    raise ValueError(
                        f"ERROR: Provided cell_key {cell_key} was not found in the assay: {assay.name}"
                    )
                if idx.dtype != bool:
                    raise ValueError(
                        f"ERROR: {cell_key} is not of boolean type. Cannot perform subsetting"
                    )
                if resolved is None:
                    resolved = idx
                elif np.all(resolved == idx) is False:
                    raise ValueError(
                        f"ERROR: Provided cell_key {cell_key} is not consistent across the assays. "
                        f"Please make sure that the assays are from the same DataStore."
                    )
            if resolved is None:
                raise ValueError("No assays available for cell index resolution")
            cell_idx = np.where(resolved)[0]
        else:
            cell_idx = np.array(cell_idx)
            if np.issubdtype(cell_idx.dtype, np.integer) is False:
                raise ValueError(
                    f"ERROR: `cell_idx` must be of integer type. Provided array has a dtype: {cell_idx.dtype}"
                )
            if max(cell_idx) >= self.assays[0].cells.N:
                raise ValueError(
                    f"ERROR: `cell_idx` max value is larger than the number of cells in the data."  # noqa: F541
                )
        return cell_idx

    def _prep_cell_data(self) -> None:
        if self.outWorkspace is None:
            cell_slot = "cellData"
        else:
            cell_slot = f"{self.outWorkspace}/cellData"
        if cell_slot in self.z:
            cell_group = as_zarr_group(self.z[cell_slot], name=cell_slot)
        else:
            cell_group = self.z.create_group(cell_slot)

        cell_data = self.assays[0].cells.locations["primary"]

        n_cells = len(self.cellIdx)
        for i in cell_data.keys():
            if i in cell_group and self.overCells is False:
                continue
            if i in ["I"] and self.resetCells:
                create_zarr_obj_array(
                    cell_group, "I", [True for _ in range(n_cells)], "bool"
                )
                continue
            v = cell_data[i][:][self.cellIdx]
            create_zarr_obj_array(cell_group, i, v, dtype=v.dtype)

    def _prep_counts(self) -> None:
        n_cells = len(self.cellIdx)
        for assay in self.assays:
            create_zarr_count_assay(
                z=self.z,
                assay_name=assay.name,
                workspace=self.outWorkspace,
                n_cells=n_cells,
                feat_ids=assay.feats.fetch_all("ids"),
                feat_names=assay.feats.fetch_all("names"),
                dtype=assay.rawData.dtype,
                profile=self.profile,
                policy=self.policy,
            )

    def dump(self) -> None:
        """Write subsetted cell metadata and count matrices, including RNA ``countsT``.

        Returns:
            None
        """
        self._prep_cell_data()
        self._prep_counts()
        for assay in self.assays:
            raw_data = assay.rawData[self.cellIdx]
            if self.outWorkspace is None:
                store = as_zarr_array(
                    self.z[f"{assay.name}/counts"],
                    name=f"{assay.name}/counts",
                )
            else:
                store = as_zarr_array(
                    self.z[f"matrices/{assay.name}/counts"],
                    name=f"matrices/{assay.name}/counts",
                )
            write_dense_in_shard_rows(
                store,
                lambda start, end: raw_data[start:end, :].compute(),
                msg=f"Subsetting assay: {assay.name}",
                resources=self.resources,
                io=self.io,
            )
            from ..assay.classification import (
                is_rna_assay_type,
                lookup_persisted_assay_type,
            )
            from .counts_t import finalize_writer_counts_t

            if is_rna_assay_type(assay):
                source_types = _source_assay_types(assay, self.inWorkspace)
                finalize_writer_counts_t(
                    self.z,
                    assay.name,
                    self.outWorkspace,
                    assay_type=lookup_persisted_assay_type(
                        assay.name,
                        source_types,
                    ),
                    resources=self.resources,
                    profile=self.profile,
                    policy=self.policy,
                    io=self.io,
                )
        logger.info(
            f"Wrote a subset of {len(self.cellIdx)} cells across "
            f"{len(self.assays)} assay(s)"
        )
