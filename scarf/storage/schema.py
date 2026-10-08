from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import zarr
from zarr.errors import ContainsArrayError

from .types import as_zarr_array, as_zarr_group, read_fresh_group
from .arrays import (
    MetadataBlock,
    create_metadata_column,
    create_streamed_metadata_column,
    create_zarr_obj_array,
)
from .count_matrix import CountMatrixPolicy, create_product_counts_array
from .layout import _group_zarr_format
from .metadata_keys import (
    ASSAY_MEMBERSHIP_ROLE,
    assay_membership_attributes,
    assay_membership_column,
)
from .profiles import StorageProfile, resolve_storage_profile
from ..utils.logging import logger

_ASSAY_NAME_OWNERS = {
    "artifacts": "datastore artifact storage",
    "cellData": "datastore cell metadata",
    "matrices": "workspace matrix storage",
    "pipeline": "DataStore.pipeline",
    "plots": "DataStore.plots",
    "summary": "DataStore.summary",
}
RESERVED_ASSAY_NAMES = frozenset(_ASSAY_NAME_OWNERS)
# Marks a derived assay whose counts are still being written. The group has no
# ``is_assay`` marker until the write is complete, so assay scans skip it.
PENDING_ASSAY_ATTR = "scarf:pending_assay"


def _assay_paths(assay_name: str, workspace: str | None) -> tuple[str, str | None]:
    """Return the logical assay path and the separate matrix path, if any."""
    if workspace is None:
        return assay_name, None
    return f"{workspace}/{assay_name}", f"matrices/{assay_name}"


def pending_assay_message(
    assay_name: str, workspace: str | None, operation: object
) -> str:
    """Explain how to remove a derived assay left pending by an interruption."""
    place = "" if workspace is None else f" of workspace {workspace!r}"
    store = (
        "a DataStore"
        if workspace is None
        else f"a DataStore opened with workspace={workspace!r}"
    )
    return (
        f"Assay {assay_name!r}{place} is pending: another process may still be "
        f"running {operation}, or an interrupted {operation} left it. If no "
        "process is writing it, remove it with "
        f"discard_interrupted_assay({assay_name!r}) on {store}, then retry."
    )


def validate_new_assay(z: zarr.Group, assay_name: str, workspace: str | None) -> None:
    """Raise ``ValueError`` unless ``assay_name`` can be created in ``workspace``."""
    validate_assay_name(assay_name)
    validate_workspace_name(workspace)
    logical, matrix = _assay_paths(assay_name, workspace)
    physical = logical if matrix is None else matrix
    manifest = read_fresh_group(z).attrs.get("matrixSource", {})
    mounted = manifest.get("assays", {}) if isinstance(manifest, dict) else {}
    existing = z.get(logical)
    if isinstance(existing, zarr.Group):
        operation = existing.attrs.get(PENDING_ASSAY_ATTR)
        if operation is not None:
            raise ValueError(pending_assay_message(assay_name, workspace, operation))
    if matrix is not None:
        # Workspaces share matrices/<assay>, and discarding a pending assay
        # deletes it, so a name pending in any workspace is taken in all.
        for name, holder, operation in pending_assays(z):
            if name == assay_name:
                raise ValueError(pending_assay_message(name, holder, operation))
    if existing is not None or physical in z or assay_name in mounted:
        raise ValueError(
            f"Assay {assay_name!r} already has metadata or a count matrix; choose a new name"
        )


def pending_assays(root: zarr.Group) -> list[tuple[str, str | None, str]]:
    """List ``(assay, workspace, operation)`` for every pending derived assay."""
    found: list[tuple[str, str | None, str]] = []
    for name, group in root.groups():
        operation = group.attrs.get(PENDING_ASSAY_ATTR)
        if operation is not None:
            found.append((name, None, str(operation)))
        elif name not in RESERVED_ASSAY_NAMES and group.attrs.get("is_assay") is None:
            # Any other group can be a workspace that holds assays.
            found.extend(
                (child_name, name, str(child.attrs[PENDING_ASSAY_ATTR]))
                for child_name, child in group.groups()
                if PENDING_ASSAY_ATTR in child.attrs
            )
    return sorted(found, key=lambda item: (item[1] or "", item[0]))


def discard_pending_assay(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    missing_ok: bool = False,
    keep_matrix: bool = False,
) -> bool:
    """Delete a derived assay that an interrupted write left pending.

    Only a group that still carries the pending marker is removed. The matrix
    group and the recorded assay type go first and the marked group last, so
    an interrupted discard can be repeated.

    Returns:
        True when a pending assay was removed.
    """
    validate_assay_name(assay_name)
    validate_workspace_name(workspace)
    logical, matrix = _assay_paths(assay_name, workspace)
    try:
        existing = read_fresh_group(root, logical, missing_ok=True)
    except ContainsArrayError:
        existing = None
    if existing is None or PENDING_ASSAY_ATTR not in existing.attrs:
        if missing_ok:
            return False
        raise ValueError(
            f"Assay {assay_name!r} is not an interrupted derived assay; "
            "nothing was removed"
        )
    if matrix is not None and not keep_matrix and matrix in root:
        del root[matrix]
    workspace_root = read_fresh_group(root, workspace or "", mode="r+", missing_ok=True)
    raw_types = (
        None if workspace_root is None else workspace_root.attrs.get("assayTypes")
    )
    if (
        workspace_root is not None
        and isinstance(raw_types, dict)
        and assay_name in raw_types
    ):
        workspace_root.attrs["assayTypes"] = {
            key: value for key, value in raw_types.items() if key != assay_name
        }
    _discard_membership_column(root, assay_name, workspace)
    del root[logical]
    return True


def _discard_membership_column(
    root: zarr.Group, assay_name: str, workspace: str | None
) -> None:
    """Delete ``<assay>_I`` when its stored attributes name ``assay_name``.

    A column of that name without them is not the assay's membership, and is
    kept.
    """
    cells = read_fresh_group(
        root, _cell_data_path(workspace), mode="r+", missing_ok=True
    )
    if cells is None:
        return
    name = assay_membership_column(assay_name)
    column = cells.get(name)
    if (
        isinstance(column, zarr.Array)
        and column.attrs.get("role") == ASSAY_MEMBERSHIP_ROLE
        and column.attrs.get("assay") == assay_name
    ):
        del cells[name]


def _require_free_membership_name(
    root: zarr.Group, assay_name: str, workspace: str | None
) -> None:
    """Raise when the cell table holds a column named for a new assay's membership."""
    name = assay_membership_column(assay_name)
    cells = root.get(_cell_data_path(workspace))
    if isinstance(cells, zarr.Group) and name in cells:
        raise ValueError(
            f"Cell column {name!r} already exists, and it is the name of the "
            f"column that records which cells a new assay {assay_name!r} "
            f"measures. Drop the column with cells.drop({name!r}), or choose "
            "another assay name."
        )


class DerivedAssayTransaction:
    """Build one derived assay that scans see only after it is complete.

    Use it through :func:`derived_assay_transaction`. The logical group carries
    a pending marker, and no ``is_assay`` marker, until the context exits
    normally with finalized counts.
    """

    def __init__(
        self,
        root: zarr.Group,
        assay_name: str,
        workspace: str | None,
        operation: str,
        membership: str | None = None,
    ) -> None:
        self.root = root
        self.assay_name = assay_name
        self.workspace = workspace
        self.operation = operation
        self.membership = membership
        self._logical, self._matrix = _assay_paths(assay_name, workspace)
        # Set right after each create_group, so cleanup deletes only the
        # groups that this call created.
        self._created_group = False
        self._created_matrix = False
        self._counts: zarr.Array | None = None

    @property
    def group(self) -> zarr.Group:
        """The pending logical assay group, for provenance attributes."""
        return as_zarr_group(self.root[self._logical], name=self._logical)

    def create_counts(
        self,
        n_cells: int,
        feat_ids: np.ndarray | list[str],
        feat_names: np.ndarray | list[str],
        dtype: Any,
        *,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
    ) -> zarr.Array:
        """Create the pending assay with incomplete counts in ``dtype``."""
        if self._created_group:
            raise RuntimeError("The derived assay counts were already created")
        _validate_dimensions(n_cells, len(feat_ids))
        # These raise before this call creates anything, so its cleanup never
        # touches an assay that another writer created.
        validate_new_assay(self.root, self.assay_name, self.workspace)
        _require_free_membership_name(self.root, self.assay_name, self.workspace)
        group = self.root.create_group(
            self._logical,
            attributes={
                PENDING_ASSAY_ATTR: self.operation,
                "prepared": False,
                "misc": {},
            },
        )
        self._created_group = True
        matrix_group = group
        if self._matrix is not None:
            matrix_group = self.root.create_group(self._matrix)
            self._created_matrix = True
        counts, feature_group, resolved_profile = _create_assay_counts(
            group,
            matrix_group,
            n_cells,
            len(feat_ids),
            dtype,
            profile=profile,
            policy=policy,
        )
        self._counts = counts
        _write_feature_columns(feature_group, feat_ids, feat_names, resolved_profile)
        if self.membership is not None:
            self._write_membership(self.membership, n_cells, resolved_profile)
        return counts

    def _write_membership(
        self, source: str, n_cells: int, profile: StorageProfile
    ) -> None:
        """Copy cell column ``source`` into the pending assay's membership column."""
        cells = as_zarr_group(
            self.root[_cell_data_path(self.workspace)], name="cellData"
        )
        values = as_zarr_array(cells[source], name=source)
        rows = max(1, int(values.chunks[0]))
        # The copy streams one source chunk at a time and never replaces a
        # column.
        create_streamed_metadata_column(
            cells,
            assay_membership_column(self.assay_name),
            shape=n_cells,
            dtype=bool,
            blocks=(
                MetadataBlock(start, np.asarray(values[start : start + rows]))
                for start in range(0, n_cells, rows)
            ),
            overwrite=False,
            chunkSize=rows,
            profile=profile,
            attributes=assay_membership_attributes(self.assay_name),
        )

    def _ready_group(self) -> zarr.Group:
        """Return the stored pending group once its counts can be published."""
        if self._counts is None:
            raise RuntimeError("The derived assay has no counts to publish")
        counts = zarr.open_array(
            store=self._counts.store,
            path=self._counts.path,
            mode="r",
            zarr_format=self._counts.metadata.zarr_format,
        )
        if counts.attrs.get("complete") is not True:
            raise RuntimeError(
                f"Derived assay {self.assay_name!r} counts were not finalized"
            )
        return read_fresh_group(self.root, self._logical)

    def _discard(self, error: BaseException) -> None:
        if not self._created_group:
            # This call created nothing, so it deletes nothing.
            return
        if not isinstance(error, Exception):
            # An interrupted write may still land, so the pending assay stays.
            logger.warning(
                pending_assay_message(self.assay_name, self.workspace, self.operation)
            )
            return
        try:
            discard_pending_assay(
                self.root,
                self.assay_name,
                self.workspace,
                missing_ok=True,
                keep_matrix=not self._created_matrix,
            )
        except Exception as exc:
            logger.warning(
                f"Could not remove the incomplete assay {self.assay_name!r}: {exc}. "
                + pending_assay_message(self.assay_name, self.workspace, self.operation)
            )


@contextmanager
def derived_assay_transaction(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
    *,
    operation: str,
    membership: str | None = None,
) -> Iterator[DerivedAssayTransaction]:
    """Create a derived assay atomically with respect to assay scans.

    The body creates counts with :meth:`DerivedAssayTransaction.create_counts`,
    writes and finalizes them, and records provenance on ``group``. A normal
    exit publishes the ``is_assay`` marker last. An exception before that
    deletes the groups this call created; an interruption, such as
    ``KeyboardInterrupt``, keeps the pending assay and logs how to remove it.

    Args:
        root: Root Zarr group of the store.
        assay_name: Name of the assay to create.
        workspace: Workspace name. None uses the legacy layout.
        operation: Name of the operation recorded in the pending marker.
        membership: Optional name of the boolean cell column that marks the
            cells the new assay measured.
    """
    validate_new_assay(root, assay_name, workspace)
    _require_free_membership_name(root, assay_name, workspace)
    transaction = DerivedAssayTransaction(
        root, assay_name, workspace, operation, membership=membership
    )
    try:
        yield transaction
        group = transaction._ready_group()
    except BaseException as error:
        transaction._discard(error)
        raise
    # The publication write follows the try, so nothing is deleted once it is
    # issued. One attribute write swaps the pending marker for the assay marker.
    attributes = dict(group.attrs)
    attributes.pop(PENDING_ASSAY_ATTR, None)
    group.attrs.put({**attributes, "is_assay": True})


def validate_assay_name(assay_name: str) -> None:
    """Reject invalid assay names and names reserved by the datastore layout."""
    if not assay_name or not assay_name.strip():
        raise ValueError("Assay names must be non-empty")
    if "/" in assay_name or "\\" in assay_name:
        raise ValueError(f"Assay name {assay_name!r} must not contain path separators")
    if assay_name in RESERVED_ASSAY_NAMES:
        owner = _ASSAY_NAME_OWNERS[assay_name]
        raise ValueError(
            f"Assay name {assay_name!r} is reserved for {owner}. "
            "Choose another name, or explicitly migrate an existing assay before "
            "opening the store with Scarf."
        )


def validate_workspace_name(workspace: str | None) -> None:
    """Reject invalid workspace names and names reserved by the datastore layout."""
    if workspace is None:
        return
    if not workspace or not workspace.strip():
        raise ValueError("Workspace names must be non-empty")
    if "/" in workspace or "\\" in workspace:
        raise ValueError(
            f"Workspace name {workspace!r} must not contain path separators"
        )
    if workspace in RESERVED_ASSAY_NAMES:
        owner = _ASSAY_NAME_OWNERS[workspace]
        raise ValueError(
            f"Workspace name {workspace!r} is reserved for {owner}. "
            "Choose another workspace name."
        )


def _validate_dimensions(n_cells: int, n_features: int) -> None:
    if n_cells < 0 or n_features < 0:
        raise ValueError("Assay dimensions must be non-negative")


def _create_assay_counts(
    group: zarr.Group,
    matrix_group: zarr.Group,
    n_cells: int,
    n_features: int,
    dtype: Any,
    *,
    profile: StorageProfile | None,
    policy: CountMatrixPolicy | None,
) -> tuple[zarr.Array, zarr.Group, StorageProfile]:
    """Create the feature group of a new assay group and incomplete counts.

    ``matrix_group`` holds the counts: the assay group itself in the legacy
    layout, or its ``matrices/<assay>`` group in a workspace layout.
    """
    resolved_profile = profile or resolve_storage_profile(group.store)
    feature_group = group.create_group("featureData")
    counts = create_product_counts_array(
        matrix_group,
        n_cells,
        n_features,
        dtype,
        profile=resolved_profile,
        policy=policy,
        zarrFormat=_group_zarr_format(matrix_group),
    )
    counts.attrs["complete"] = False
    return counts, feature_group, resolved_profile


def _create_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    n_features: int,
    dtype: Any,
    *,
    profile: StorageProfile | None,
    policy: CountMatrixPolicy | None,
) -> tuple[zarr.Array, zarr.Group, StorageProfile]:
    """Create a complete-marked assay group with incomplete paired counts."""
    validate_new_assay(z, assay_name, workspace)
    _validate_dimensions(n_cells, n_features)
    logical, matrix = _assay_paths(assay_name, workspace)
    group = z.create_group(
        logical, attributes={"is_assay": True, "prepared": False, "misc": {}}
    )
    return _create_assay_counts(
        group,
        group if matrix is None else z.create_group(matrix),
        n_cells,
        n_features,
        dtype,
        profile=profile,
        policy=policy,
    )


def _write_feature_columns(
    feature_group: zarr.Group,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    profile: StorageProfile,
) -> None:
    for name, values, column_dtype in (
        ("ids", feat_ids, None),
        ("names", feat_names, None),
        ("I", np.ones(len(feat_ids), dtype=bool), "bool"),
    ):
        create_zarr_obj_array(
            feature_group, name, values, column_dtype, profile=profile
        )


def create_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    dtype: Any,
    *,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
) -> zarr.Array:
    """Create an assay group and its incomplete ``counts`` array in ``dtype``.

    Import writers pass the dtype that
    :func:`~scarf.storage.count_dtype.count_storage_dtype` resolves. The
    counts carry ``complete=False`` until the writer calls
    ``finalize_counts``. Derived assays use :func:`derived_assay_transaction`
    instead.
    """
    counts, feature_group, resolved_profile = _create_count_assay(
        z,
        assay_name,
        workspace,
        n_cells,
        len(feat_ids),
        dtype,
        profile=profile,
        policy=policy,
    )
    _write_feature_columns(feature_group, feat_ids, feat_names, resolved_profile)
    return counts


def create_empty_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    n_features: int,
    feature_id_dtype: Any,
    feature_name_dtype: Any,
    dtype: Any,
    *,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
) -> tuple[zarr.Array, zarr.Group]:
    """Create an assay with counts in ``dtype`` and blockwise feature metadata."""
    counts, feature_group, resolved_profile = _create_count_assay(
        z,
        assay_name,
        workspace,
        n_cells,
        n_features,
        dtype,
        profile=profile,
        policy=policy,
    )
    _create_empty_columns(
        feature_group,
        n_features,
        ids_dtype=feature_id_dtype,
        names_dtype=feature_name_dtype,
        profile=resolved_profile,
    )
    return counts, feature_group


def load_count_array(
    root: zarr.Group,
    assay_name: str,
    workspace: str | None,
) -> zarr.Array:
    if workspace is None:
        return as_zarr_array(
            root[f"{assay_name}/counts"],
            name=f"{assay_name}/counts",
        )
    return as_zarr_array(
        root[f"matrices/{assay_name}/counts"],
        name=f"matrices/{assay_name}/counts",
    )


def _cell_data_path(workspace: str | None) -> str:
    return "cellData" if workspace is None else f"{workspace}/cellData"


def _create_empty_columns(
    group: zarr.Group,
    n_rows: int,
    *,
    ids_dtype: Any,
    names_dtype: Any,
    profile: StorageProfile | None,
) -> None:
    """Create blockwise-fillable ``ids`` and ``names`` and an all-true ``I``."""
    for name, dtype in (("ids", ids_dtype), ("names", names_dtype), ("I", bool)):
        column = create_metadata_column(
            group,
            name,
            dtype=dtype,
            shape=n_rows,
            chunkSize=100_000,
            profile=profile,
        )
        if name == "I":
            column[:] = True


def create_cell_data(
    root: zarr.Group,
    workspace: str | None,
    ids: np.ndarray,
    names: np.ndarray,
    profile: StorageProfile | None = None,
) -> zarr.Group:
    group = root.create_group(_cell_data_path(workspace))
    create_zarr_obj_array(group, "ids", ids, ids.dtype, profile=profile)
    create_zarr_obj_array(group, "names", names, names.dtype, profile=profile)
    create_zarr_obj_array(
        group, "I", np.ones(len(ids), dtype=bool), "bool", profile=profile
    )
    return group


def create_empty_cell_data(
    root: zarr.Group,
    workspace: str | None,
    n_cells: int,
    id_dtype: Any,
    name_dtype: Any,
    profile: StorageProfile | None = None,
) -> zarr.Group:
    """Create cell metadata columns that can be filled blockwise."""
    group = root.create_group(_cell_data_path(workspace))
    _create_empty_columns(
        group, n_cells, ids_dtype=id_dtype, names_dtype=name_dtype, profile=profile
    )
    return group
