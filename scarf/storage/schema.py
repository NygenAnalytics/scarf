import numpy as np
import zarr

from .types import array_metadata_shards, as_zarr_array
from .arrays import (
    create_numeric_array,
    create_zarr_obj_array,
)
from .layout import _group_zarr_format, count_array_spec
from .profiles import StorageProfile, resolve_storage_profile

RESERVED_ASSAY_NAMES = frozenset({"artifacts", "pipeline", "plots"})


def validate_assay_name(assay_name: str) -> None:
    """Reject assay names reserved by the datastore API."""
    if assay_name in RESERVED_ASSAY_NAMES:
        owner = (
            "datastore artifact storage"
            if assay_name == "artifacts"
            else f"DataStore.{assay_name}"
        )
        raise ValueError(
            f"Assay name {assay_name!r} is reserved for {owner}. "
            "Choose another name, or rename the assay in the Zarr store before "
            "opening it with Scarf."
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
