from typing import Any

import numpy as np
import zarr

from .types import array_metadata_shards, as_zarr_array
from .arrays import (
    create_metadata_column,
    create_numeric_array,
    create_zarr_obj_array,
)
from .layout import _group_zarr_format, count_array_spec
from .profiles import StorageProfile, resolve_storage_profile

_ASSAY_NAME_OWNERS = {
    "artifacts": "datastore artifact storage",
    "cellData": "datastore cell metadata",
    "matrices": "workspace matrix storage",
    "pipeline": "DataStore.pipeline",
    "plots": "DataStore.plots",
    "summary": "DataStore.summary",
}
RESERVED_ASSAY_NAMES = frozenset(_ASSAY_NAME_OWNERS)


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
    targetChunkBytes: int | None = None,
    targetShardBytes: int | None = None,
) -> zarr.Array:
    validate_assay_name(assay_name)
    if workspace is None:
        group = z.create_group(assay_name, overwrite=True)
    else:
        group = z.create_group(f"{workspace}/{assay_name}", overwrite=True)
    group.attrs["is_assay"] = True
    group.attrs["misc"] = {}
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
        group = z.create_group(f"matrices/{assay_name}", overwrite=True)
    n_feats = len(feat_ids)
    zarr_format = _group_zarr_format(group)
    spec = count_array_spec(
        n_cells,
        n_feats,
        dtype=dtype,
        profile=resolved_profile,
        targetChunkBytes=targetChunkBytes,
        targetShardBytes=targetShardBytes,
        zarrFormat=zarr_format,
    )
    counts = create_numeric_array(group, "counts", spec)
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
    targetChunkBytes: int | None = None,
    targetShardBytes: int | None = None,
) -> tuple[zarr.Array, zarr.Group]:
    """Create an assay whose feature metadata can be filled blockwise."""
    validate_assay_name(assay_name)
    if n_cells < 0 or n_features < 0:
        raise ValueError("Assay dimensions must be non-negative")
    assay_path = assay_name if workspace is None else f"{workspace}/{assay_name}"
    assay_group = z.create_group(assay_path, overwrite=True)
    assay_group.attrs["is_assay"] = True
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
        assay_group
        if workspace is None
        else z.create_group(f"matrices/{assay_name}", overwrite=True)
    )
    zarr_format = _group_zarr_format(matrix_group)
    spec = count_array_spec(
        n_cells,
        n_features,
        dtype=dtype,
        profile=resolved_profile,
        targetChunkBytes=targetChunkBytes,
        targetShardBytes=targetShardBytes,
        zarrFormat=zarr_format,
    )
    counts = create_numeric_array(matrix_group, "counts", spec)
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
