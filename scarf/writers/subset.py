from typing import Any

import numpy as np

from ..storage.types import as_zarr_array, as_zarr_group
from ..storage.arrays import create_zarr_obj_array
from ..storage.count_matrix import (
    CountMatrixPolicy,
    create_count_matrix_array,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from ..storage.copy import copy_zarr_group_tree
from ..storage.identity import (
    GENERATED_FEATURE_COLUMNS,
    CountSummary,
    finalize_counts,
    generated_cell_columns,
)
from ..storage.io_policy import StorageIoPolicy
from ..storage.layout import array_shard_rows
from ..storage.partition import checked_indices
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.schema import create_zarr_count_assay
from ..storage.sharding import (
    dense_counts_admission,
    fit_count_layout,
    write_dense_in_shard_rows,
)
from ..storage.stores import load_zarr, zarr_location_has_content
from ..utils.arrays import has_duplicates
from ..utils.logging import logger


def _source_assay_types(assay: Any) -> dict[str, str]:
    """Read persisted assay types from the source root or workspace."""
    raw = assay._artifact_root.attrs.get("assayTypes", {})
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

    Args:
        zarr_loc: The file name for the Zarr hierarchy.
        in_grp: Group in Zarr hierarchy to subset.
        out_grp: Group name in Zarr hierarchy to write subsetted assay to.
        cells_idx: Distinct non-negative indices of the cells to keep, in the
                   order of the subset's rows.
        feat_idx: Distinct non-negative indices of the features to keep, at
                  least one, in the order of the subset's columns.
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy, used exactly. When None, the
                default policy is used with unitBytes and chunkBytes halved
                together until the subset write fits ``mem_budget``. Either
                way, a subset that does not fit raises MemoryError before
                ``out_grp`` is created.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.

    The subset keeps the dtype of the source counts.

    Returns:
        None
    """
    from ..storage.budget import resolve_budget

    resources = resolve_budget(mem_budget, nthreads)
    resolved_profile = resolve_storage_profile(zarr_loc, profile)
    z = load_zarr(zarr_loc, "r+", storage_options=storage_options)
    ig = as_zarr_array(z[in_grp], name=in_grp)
    cells_idx = checked_indices(cells_idx, limit=ig.shape[0], name="cells_idx")
    feat_idx = checked_indices(feat_idx, limit=ig.shape[1], name="feat_idx")
    if feat_idx.size == 0:
        raise ValueError("feat_idx must select at least one feature")
    resident = (
        cells_idx.nbytes
        + feat_idx.nbytes
        + CountSummary.nbytes_for(len(cells_idx), len(feat_idx))
    )
    producer = int(np.prod(ig.chunks)) * ig.dtype.itemsize
    layout = fit_count_layout(
        {out_grp: (len(feat_idx), ig.dtype)},
        nCells=len(cells_idx),
        profile=resolved_profile,
        memoryBytes=resources.memoryBytes,
        transposed=(),
        admitCounts=dense_counts_admission(resident, lambda _rows: producer),
        requested=policy,
    )
    plan = plan_count_matrix_pair(
        len(cells_idx),
        len(feat_idx),
        ig.dtype,
        policy=layout,
        profile=resolved_profile,
    )
    og = create_count_matrix_array(z, out_grp, plan.counts)
    # The counts and their parent group carry the layout the count contract checks.
    parent = out_grp.rpartition("/")[0]
    persist_count_matrix_plan(z if not parent else as_zarr_group(z[parent]), plan)
    persist_count_matrix_plan(og, plan)
    summary = CountSummary(og)
    write_dense_in_shard_rows(
        og,
        lambda start, end: np.asarray(
            ig.get_orthogonal_selection((cells_idx[start:end], feat_idx))
        ),
        msg="Subsetting assay",
        resources=resources,
        io=io,
        producerBytes=producer,
        residentBytes=resident,
        countSummary=summary,
    )
    finalize_counts(og, summary=summary)
    return None


class SubsetZarr:
    """Split Zarr file using a subset of cells.

    Args:
        zarr_loc: Path for the output (subsetted) Zarr file
        assays: Source assays to be subsetted. These assays must be from the same dataset
        in_workspace: Source workspace name (None for legacy layout).
        out_workspace: Target workspace name in the output Zarr file.
        cell_key: Name of a boolean column in cell metadata. The cells with value True are included in the
                  subset. Only used when cell_idx is None. A column without True values writes a subset
                  without cells.
        cell_idx: Explicit indices of cells to include in the subset: at least one, each a distinct
                  non-negative integer smaller than the number of cells.
        reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                           are set as True as in the 'I' column. To keep the filtering information set the value for
                           this parameter to False. (Default value: True)
        overwrite_existing_file: If True, then overwrites the existing data. (Default value: False)
        storage_options: Backend options passed when opening the Zarr store.
        mem_budget: Memory available to the conversion. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        profile: Zarr encoding profile (``fast_local`` or ``cloud``). When
                 None, chosen from the destination location.
        policy: Count-matrix geometry policy, used exactly for every assay.
                When None, each assay uses the default policy with unitBytes
                and chunkBytes halved together until its subset write and
                countsT transpose fit ``mem_budget``. Either way, a subset
                that does not fit raises MemoryError before the destination
                is created.
        io: Optional explicit read, compute, and write widths. Unset values
            stay under automatic planning.

    Each assay keeps the dtype of its source counts, so the subset holds the
    source values exactly.
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
        self.inWorkspace = in_workspace
        self.outWorkspace = out_workspace
        self.storage_options = storage_options
        self.assays = self._check_assays(assays)
        self.cellIdx = self._check_idx(cell_key, cell_idx)
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
        self.io = io
        self._check_files(zarr_loc)
        # A layout that does not fit fails here, before the destination exists.
        self._layouts = {
            assay.name: self._fit_count_layout(assay, policy) for assay in self.assays
        }
        self.z = load_zarr(
            zarr_loc=zarr_loc, mode="w", storage_options=self.storage_options
        )

    def _fit_count_layout(
        self, assay: Any, requested: CountMatrixPolicy | None
    ) -> CountMatrixPolicy:
        """Return the layout whose subset counts and ``countsT`` fit the budget."""
        from ..assay.classification import lookup_persisted_assay_type
        from .counts_t import counts_t_assays

        raw_data = assay.rawData[self.cellIdx]
        n_features = int(assay.rawData.shape[1])
        resident = raw_data._resident_bytes() + CountSummary.nbytes_for(
            len(self.cellIdx), n_features
        )
        assay_type = lookup_persisted_assay_type(assay.name, _source_assay_types(assay))
        return fit_count_layout(
            {assay.name: (n_features, assay.rawData.dtype)},
            nCells=len(self.cellIdx),
            profile=self.profile,
            memoryBytes=self.resources.memoryBytes,
            transposed=counts_t_assays((assay.name,), {assay.name: assay_type}),
            admitCounts=dense_counts_admission(
                resident,
                lambda rows: raw_data._with_block_size(rows)._block_task_bytes(),
            ),
            requested=requested,
        )

    def _check_files(self, zarr_loc: ZarrLocation) -> None:
        from ..storage.stores import locations_overlap, zarr_root_path

        for assay in self.assays:
            for group in (assay.z, assay.matrixGroup):
                path = zarr_root_path(group)
                if zarr_loc is group.store or (
                    isinstance(zarr_loc, str)
                    and path is not None
                    and locations_overlap(path, zarr_loc)
                ):
                    raise ValueError("Subset destination overlaps a source store")
        if self.overFn is False and zarr_location_has_content(
            zarr_loc, storage_options=self.storage_options
        ):
            raise ValueError(
                f"Zarr file with name: {zarr_loc} already exists.\n"
                f"If you want to overwrite it then please set  overwrite_existing_file to True. "
                f"No subsetting was performed."
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
                elif not np.array_equal(resolved, idx):
                    raise ValueError(
                        f"ERROR: Provided cell_key {cell_key} is not consistent across the assays. "
                        f"Please make sure that the assays are from the same DataStore."
                    )
            # _check_assays admits at least one assay.
            assert resolved is not None
            cell_idx = np.where(resolved)[0]
        else:
            cell_idx = np.array(cell_idx)
            if cell_idx.size == 0:
                raise ValueError("ERROR: `cell_idx` cannot be empty.")
            if np.issubdtype(cell_idx.dtype, np.integer) is False:
                raise ValueError(
                    f"ERROR: `cell_idx` must be of integer type. Provided array has a dtype: {cell_idx.dtype}"
                )
            # A negative index would wrap around to a cell counted from the end.
            if np.any(cell_idx < 0):
                raise ValueError("ERROR: `cell_idx` cannot contain negative indices.")
            if max(cell_idx) >= self.assays[0].cells.N:
                raise ValueError(
                    f"ERROR: `cell_idx` max value is larger than the number of cells in the data."  # noqa: F541
                )
            # A repeated cell would repeat its ID in the subset.
            if has_duplicates(cell_idx):
                raise ValueError("ERROR: `cell_idx` cannot contain duplicate indices.")
        return cell_idx

    def _prep_cell_data(self) -> None:
        if self.outWorkspace is None:
            cell_slot = "cellData"
        else:
            cell_slot = f"{self.outWorkspace}/cellData"
        # The constructor opened the destination empty.
        cell_group = self.z.create_group(cell_slot)

        cell_data = self.assays[0].cells.locations["primary"]

        source_root = self.assays[0]._artifact_root
        names = set(source_root.attrs.get("assayTypes", {})) | {
            assay.name for assay in self.assays
        }
        generated = set().union(
            *(
                generated_cell_columns(
                    name, source_root[name].attrs.get("percentFeatures")
                )
                for name in names
            )
        )
        if self.resetCells:
            generated.add("I")
        copy_zarr_group_tree(
            cell_data,
            cell_group,
            row_indices=self.cellIdx,
            exclude_members=generated,
            profile=self.profile,
        )
        if self.resetCells:
            create_zarr_obj_array(
                cell_group, "I", np.ones(len(self.cellIdx), dtype=bool), "bool"
            )

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
                policy=self._layouts[assay.name],
            )
            path = (
                assay.name
                if self.outWorkspace is None
                else f"{self.outWorkspace}/{assay.name}"
            )
            destination = as_zarr_group(self.z[path], name=path)
            copy_zarr_group_tree(
                assay.feats.locations["primary"],
                as_zarr_group(destination["featureData"], name="featureData"),
                exclude_members={"ids", "names", *GENERATED_FEATURE_COLUMNS},
                profile=self.profile,
            )
            if "size_factor" in assay.attrs:
                destination.attrs["size_factor"] = assay.attrs["size_factor"]

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
            summary = CountSummary(store)
            write_dense_in_shard_rows(
                store,
                lambda start, end: raw_data[start:end, :].compute(),
                msg=f"Subsetting assay: {assay.name}",
                resources=self.resources,
                io=self.io,
                producerBytes=raw_data._with_block_size(
                    array_shard_rows(store)
                )._block_task_bytes(),
                residentBytes=raw_data._resident_bytes() + summary.nbytes,
                countSummary=summary,
            )
            finalize_counts(store, summary=summary)
            from ..assay.classification import lookup_persisted_assay_type
            from .counts_t import finalize_writer_counts_t

            source_types = _source_assay_types(assay)
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
                io=self.io,
            )
        logger.info(
            f"Wrote a subset of {len(self.cellIdx)} cells across "
            f"{len(self.assays)} assay(s)"
        )
