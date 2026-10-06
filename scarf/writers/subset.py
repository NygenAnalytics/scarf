from dataclasses import replace
from typing import Any

import numpy as np
import zarr

from ..storage.types import as_zarr_array, as_zarr_group
from ..storage.arrays import create_zarr_obj_array
from ..storage.count_matrix import (
    COUNT_MATRIX_LAYOUT_KEY,
    CountMatrixPolicy,
    create_count_matrix_array,
    persist_count_matrix_plan,
    plan_count_matrix_pair,
)
from ..storage.copy import copy_zarr_group_tree
from ..storage.destinations import create_destination, refuse_pending_assays
from ..storage.identity import (
    GENERATED_FEATURE_COLUMNS,
    CountSummary,
    finalize_counts,
    generated_cell_columns,
)
from ..storage.io_policy import StorageIoPolicy
from ..storage.layout import array_shard_rows
from ..storage.metadata_keys import assay_membership_column
from ..storage.partition import checked_indices
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    resolve_storage_profile,
)
from ..storage.schema import create_zarr_count_assay, validate_workspace_name
from ..storage.sharding import (
    dense_counts_admission,
    fit_count_layout,
    write_dense_in_shard_rows,
)
from ..storage.stores import load_zarr
from ..utils.arrays import has_duplicates
from ..utils.logging import logger


def _node_parts(path: str) -> tuple[str, ...]:
    """Return the segments of a node path as Zarr resolves it.

    Zarr reads ``\\`` as ``/`` and ignores leading, trailing, and repeated
    separators, so ``/RNA//counts`` and ``RNA/counts`` name one node.
    """
    parts = tuple(part for part in path.replace("\\", "/").split("/") if part)
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"Node path {path!r} must not contain '.' or '..' segments")
    return parts


def _check_new_output(z: zarr.Group, out_grp: str, parts: tuple[str, ...]) -> None:
    """Raise unless writing ``out_grp`` and its layout record replaces nothing."""
    if "/".join(parts) in z:
        raise FileExistsError(
            f"out_grp {out_grp!r} already exists in the store; choose a new out_grp"
        )
    parent_path = "/".join(parts[:-1])
    parent = z.get(parent_path) if parent_path else z
    if isinstance(parent, zarr.Group) and COUNT_MATRIX_LAYOUT_KEY in parent.attrs:
        # The record describes the counts that the group already holds.
        raise FileExistsError(
            f"Group {parent_path or '/'!r} already records the layout of a count "
            f"matrix, which a subset at {out_grp!r} would replace. Write the "
            "subset into a group that holds no counts."
        )


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
        in_grp: Array in Zarr hierarchy to subset.
        out_grp: Path of a new array in Zarr hierarchy to write subsetted assay to.
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

    Raises:
        FileExistsError: If ``out_grp`` exists or its parent group holds counts.
    """
    from ..storage.budget import resolve_budget

    out_parts = _node_parts(out_grp)
    in_parts = _node_parts(in_grp)
    shared = min(len(in_parts), len(out_parts))
    if out_parts[:shared] == in_parts[:shared]:
        raise ValueError(
            f"out_grp {out_grp!r} must not be, contain, or lie inside in_grp {in_grp!r}"
        )
    resources = resolve_budget(mem_budget, nthreads)
    resolved_profile = resolve_storage_profile(zarr_loc, profile)
    z = load_zarr(zarr_loc, "r+", storage_options=storage_options)
    ig = as_zarr_array(z[in_grp], name=in_grp)
    _check_new_output(z, out_grp, out_parts)
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
    # Zarr refuses to replace a node that appeared since the check.
    og = create_count_matrix_array(
        z, "/".join(out_parts), replace(plan.counts, overwrite=False)
    )
    # The counts and their parent group carry the layout the count contract checks.
    parent = "/".join(out_parts[:-1])
    persist_count_matrix_plan(as_zarr_group(z[parent]) if parent else z, plan)
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
        assays: Source assays to be subsetted. These assays must be from the same DataStore
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
        overwrite_existing_file: If True, replaces an existing Scarf store that no ``DataStore``
                                 has opened. (Default value: False)
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

    Raises:
        FileExistsError: If ``zarr_loc`` is not empty and may not be replaced.
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

        validate_workspace_name(out_workspace)
        self.resetCells = reset_cell_filter
        self.overFn = overwrite_existing_file
        self.inWorkspace = in_workspace
        self.outWorkspace = out_workspace
        self.storage_options = storage_options
        self.assays = self._check_assays(assays)
        # A pending derived assay is incomplete; its membership column would
        # be copied with the cell metadata as finished data.
        source = self.assays[0].z
        refuse_pending_assays(
            zarr.open_group(store=source.store, mode="r" if source.read_only else "r+"),
            operation="subset",
        )
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
        # A layout that does not fit fails here, before the destination exists.
        self._layouts = {
            assay.name: self._fit_count_layout(assay, policy) for assay in self.assays
        }
        self._check_overlap(zarr_loc)
        self.z = create_destination(
            zarr_loc,
            overwrite=overwrite_existing_file,
            storage_options=self.storage_options,
        )

    def _check_overlap(self, zarr_loc: ZarrLocation) -> None:
        """Raise when the destination is, or overlaps, a source store."""
        from ..storage.stores import locations_overlap, zarr_root_path

        target = (
            zarr_loc if isinstance(zarr_loc, str) else getattr(zarr_loc, "root", None)
        )
        for assay in self.assays:
            for group in (assay.z, assay.matrixGroup):
                path = zarr_root_path(group)
                if zarr_loc is group.store or (
                    target is not None
                    and path is not None
                    and locations_overlap(path, str(target))
                ):
                    raise ValueError("Subset destination overlaps a source store")

    def _fit_count_layout(
        self, assay: Any, requested: CountMatrixPolicy | None
    ) -> CountMatrixPolicy:
        """Return the layout whose subset counts and ``countsT`` fit the budget."""
        from ..assay.classification import declared_assay_type
        from .counts_t import counts_t_assays

        raw_data = assay.rawData[self.cellIdx]
        n_features = int(assay.rawData.shape[1])
        resident = raw_data._resident_bytes() + CountSummary.nbytes_for(
            len(self.cellIdx), n_features
        )
        assay_type = declared_assay_type(assay)
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

    @staticmethod
    def _check_assays(assays: list[Any]) -> list[Any]:
        # if type(assays) != list:
        if isinstance(assays, list) is False:
            raise TypeError(
                "Value for parameter `assays` should be a list. For example, `[ds.RNA]`"
            )
        if not assays:
            raise ValueError(
                "A subset needs at least one assay. For example, `[ds.RNA]`"
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
        # The assays of one DataStore share its cell table. An assay of another
        # DataStore with as many cells would pair its counts with these cells.
        if any(assay.cells is not assays[0].cells for assay in assays):
            raise ValueError(
                "ERROR: Provided assays are not from the same DataStore. Please pass "
                "assays of one DataStore, such as `[ds.RNA, ds.ADT]`."
            )
        if len({assay.name for assay in assays}) != len(assays):
            raise ValueError("ERROR: Provided assays must not repeat an assay.")
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
        kept = {assay.name for assay in self.assays}
        # Every assay of the source, read from its groups: the assayTypes
        # record of this session's group object can predate assays that the
        # session registered.
        names = kept | {
            name for name, group in source_root.groups() if "is_assay" in group.attrs
        }
        excluded = set().union(
            *(
                generated_cell_columns(
                    name, source_root[name].attrs.get("percentFeatures")
                )
                for name in names
            )
        )
        # The membership column of an assay goes with the assay, so the subset
        # keeps it, with its rows subset, only for the assays it writes.
        excluded.update(assay_membership_column(name) for name in names - kept)
        if self.resetCells:
            excluded.add("I")
        copy_zarr_group_tree(
            cell_data,
            cell_group,
            row_indices=self.cellIdx,
            exclude_members=excluded,
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
            from ..assay.classification import declared_assay_type
            from .counts_t import finalize_writer_counts_t

            finalize_writer_counts_t(
                self.z,
                assay.name,
                self.outWorkspace,
                assay_type=declared_assay_type(assay),
                resources=self.resources,
                profile=self.profile,
                io=self.io,
            )
        logger.info(
            f"Wrote a subset of {len(self.cellIdx)} cells across "
            f"{len(self.assays)} assay(s)"
        )
