from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import zarr

from .types import array_metadata_shards, as_zarr_array, as_zarr_group
from .arrays import (
    create_metadata_column,
    create_numeric_array,
    create_zarr_obj_array,
)
from .count_matrix import CountMatrixPolicy, create_product_counts_array
from .layout import _group_zarr_format, count_array_spec
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
    store = (
        "a DataStore"
        if workspace is None
        else f"a DataStore opened with workspace={workspace!r}"
    )
    return (
        f"Assay {assay_name!r} was left incomplete by an interrupted {operation}. "
        f"Remove it with discard_interrupted_assay({assay_name!r}) on {store}, "
        "then retry."
    )


def validate_new_assay(z: zarr.Group, assay_name: str, workspace: str | None) -> None:
    from .identity import fresh_group

    validate_assay_name(assay_name)
    validate_workspace_name(workspace)
    logical, matrix = _assay_paths(assay_name, workspace)
    physical = logical if matrix is None else matrix
    manifest = fresh_group(z).attrs.get("matrixSource", {})
    mounted = manifest.get("assays", {}) if isinstance(manifest, dict) else {}
    existing = z.get(logical)
    if isinstance(existing, zarr.Group):
        operation = existing.attrs.get(PENDING_ASSAY_ATTR)
        if operation is not None:
            raise ValueError(pending_assay_message(assay_name, workspace, operation))
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
    existing = root.get(logical)
    if not isinstance(existing, zarr.Group) or PENDING_ASSAY_ATTR not in (
        existing.attrs
    ):
        if missing_ok:
            return False
        raise ValueError(
            f"Assay {assay_name!r} is not an interrupted derived assay; "
            "nothing was removed"
        )
    if matrix is not None and matrix in root:
        del root[matrix]
    workspace_root = (
        root if workspace is None else as_zarr_group(root[workspace], name=workspace)
    )
    raw_types = workspace_root.attrs.get("assayTypes")
    if isinstance(raw_types, dict) and assay_name in raw_types:
        workspace_root.attrs["assayTypes"] = {
            key: value for key, value in raw_types.items() if key != assay_name
        }
    del root[logical]
    return True


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
    ) -> None:
        self.root = root
        self.assay_name = assay_name
        self.workspace = workspace
        self.operation = operation
        self._logical, _ = _assay_paths(assay_name, workspace)
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
        dtype: str = "float",
        *,
        profile: StorageProfile | None = None,
        policy: CountMatrixPolicy | None = None,
    ) -> zarr.Array:
        """Create the pending assay with incomplete counts and feature names."""
        if self._counts is not None:
            raise RuntimeError("The derived assay counts were already created")
        self._counts = create_zarr_count_assay(
            self.root,
            self.assay_name,
            self.workspace,
            n_cells,
            feat_ids,
            feat_names,
            dtype,
            profile=profile,
            policy=policy,
            pending_operation=self.operation,
        )
        return self._counts

    def _publish(self) -> None:
        from .identity import fresh_group

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
        group = fresh_group(self.group)
        attributes = dict(group.attrs)
        attributes.pop(PENDING_ASSAY_ATTR, None)
        # One attribute write swaps the pending marker for the assay marker.
        group.attrs.put({**attributes, "is_assay": True})

    def _discard(self) -> None:
        try:
            discard_pending_assay(
                self.root, self.assay_name, self.workspace, missing_ok=True
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
) -> Iterator[DerivedAssayTransaction]:
    """Create a derived assay atomically with respect to assay scans.

    The body creates counts with :meth:`DerivedAssayTransaction.create_counts`,
    writes and finalizes them, and records provenance on ``group``. A normal
    exit publishes the ``is_assay`` marker last. Any exception, including
    ``KeyboardInterrupt``, deletes the pending assay before it propagates.

    Args:
        root: Root Zarr group of the store.
        assay_name: Name of the assay to create.
        workspace: Workspace name. None uses the legacy layout.
        operation: Name of the operation recorded in the pending marker.
    """
    validate_new_assay(root, assay_name, workspace)
    transaction = DerivedAssayTransaction(root, assay_name, workspace, operation)
    try:
        yield transaction
        transaction._publish()
    except BaseException:
        transaction._discard()
        raise


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


def create_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    dtype: str = "uint32",
    *,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
    pending_operation: str | None = None,
) -> zarr.Array:
    """Create an assay group and its incomplete ``counts`` array.

    The counts carry ``complete=False`` until the writer calls
    ``finalize_counts``. With ``pending_operation``, the group carries the
    pending marker instead of ``is_assay``; use
    :func:`derived_assay_transaction` rather than passing it directly.
    """
    validate_new_assay(z, assay_name, workspace)
    marker: dict[str, Any] = (
        {"is_assay": True}
        if pending_operation is None
        else {PENDING_ASSAY_ATTR: pending_operation}
    )
    group = z.create_group(
        assay_name if workspace is None else f"{workspace}/{assay_name}",
        attributes={**marker, "prepared": False, "misc": {}},
    )
    resolved_profile = profile or resolve_storage_profile(group.store)
    create_zarr_obj_array(
        group,
        "featureData/ids",
        feat_ids,
        profile=resolved_profile,
    )
    create_zarr_obj_array(
        group,
        "featureData/names",
        feat_names,
        profile=resolved_profile,
    )
    create_zarr_obj_array(
        group,
        "featureData/I",
        [True for _ in range(len(feat_ids))],
        "bool",
        profile=resolved_profile,
    )
    if workspace is not None:
        group = z.create_group(f"matrices/{assay_name}")
    n_feats = len(feat_ids)
    zarr_format = _group_zarr_format(group)
    if zarr_format >= 3:
        counts = create_product_counts_array(
            group,
            n_cells,
            n_feats,
            dtype,
            profile=resolved_profile,
            policy=policy,
        )
    else:
        spec = count_array_spec(
            n_cells,
            n_feats,
            dtype=dtype,
            profile=resolved_profile,
            policy=policy,
            zarrFormat=zarr_format,
        )
        counts = create_numeric_array(group, "counts", spec)
    counts.attrs["complete"] = False
    stored_shards = array_metadata_shards(counts)
    group.attrs["scarf:zarr_spec"] = {
        "profile": resolved_profile,
        "dtype": np.dtype(dtype).str,
        "chunks": list(counts.chunks),
        "shards": None if stored_shards is None else list(stored_shards),
        "zarr_format": zarr_format,
    }
    return counts


def create_empty_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    n_cells: int,
    n_features: int,
    feature_id_dtype: Any,
    feature_name_dtype: Any,
    dtype: Any = "uint32",
    *,
    profile: StorageProfile | None = None,
    policy: CountMatrixPolicy | None = None,
) -> tuple[zarr.Array, zarr.Group]:
    """Create an assay whose feature metadata can be filled blockwise."""
    validate_new_assay(z, assay_name, workspace)
    if n_cells < 0 or n_features < 0:
        raise ValueError("Assay dimensions must be non-negative")
    assay_path = assay_name if workspace is None else f"{workspace}/{assay_name}"
    assay_group = z.create_group(assay_path)
    assay_group.attrs["is_assay"] = True
    assay_group.attrs["prepared"] = False
    assay_group.attrs["misc"] = {}
    resolved_profile = profile or resolve_storage_profile(assay_group.store)
    feature_group = assay_group.create_group("featureData")
    create_metadata_column(
        feature_group,
        "ids",
        dtype=feature_id_dtype,
        shape=n_features,
        chunkSize=100_000,
        profile=resolved_profile,
    )
    create_metadata_column(
        feature_group,
        "names",
        dtype=feature_name_dtype,
        shape=n_features,
        chunkSize=100_000,
        profile=resolved_profile,
    )
    included = create_metadata_column(
        feature_group,
        "I",
        dtype=bool,
        shape=n_features,
        chunkSize=100_000,
        profile=resolved_profile,
    )
    included[:] = True

    matrix_group = (
        assay_group if workspace is None else z.create_group(f"matrices/{assay_name}")
    )
    zarr_format = _group_zarr_format(matrix_group)
    if zarr_format >= 3:
        counts = create_product_counts_array(
            matrix_group,
            n_cells,
            n_features,
            dtype,
            profile=resolved_profile,
            policy=policy,
        )
    else:
        spec = count_array_spec(
            n_cells,
            n_features,
            dtype=dtype,
            profile=resolved_profile,
            policy=policy,
            zarrFormat=zarr_format,
        )
        counts = create_numeric_array(matrix_group, "counts", spec)
    counts.attrs["complete"] = False
    stored_shards = array_metadata_shards(counts)
    matrix_group.attrs["scarf:zarr_spec"] = {
        "profile": resolved_profile,
        "dtype": np.dtype(dtype).str,
        "chunks": list(counts.chunks),
        "shards": None if stored_shards is None else list(stored_shards),
        "zarr_format": zarr_format,
    }
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


def create_cell_data(
    root: zarr.Group,
    workspace: str | None,
    ids: np.ndarray,
    names: np.ndarray,
    profile: StorageProfile | None = None,
) -> zarr.Group:
    if workspace is None:
        group = root.create_group("cellData")
    else:
        group = root.create_group(f"{workspace}/cellData")
    create_zarr_obj_array(group, "ids", ids, ids.dtype, profile=profile)
    create_zarr_obj_array(group, "names", names, names.dtype, profile=profile)
    create_zarr_obj_array(
        group,
        "I",
        [True for _ in range(len(ids))],
        "bool",
        profile=profile,
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
    if n_cells < 0:
        raise ValueError("n_cells must be non-negative")
    path = "cellData" if workspace is None else f"{workspace}/cellData"
    group = root.create_group(path)
    create_metadata_column(
        group,
        "ids",
        dtype=id_dtype,
        shape=n_cells,
        chunkSize=100_000,
        profile=profile,
    )
    create_metadata_column(
        group,
        "names",
        dtype=name_dtype,
        shape=n_cells,
        chunkSize=100_000,
        profile=profile,
    )
    included = create_metadata_column(
        group,
        "I",
        dtype=bool,
        shape=n_cells,
        chunkSize=100_000,
        profile=profile,
    )
    included[:] = True
    return group
