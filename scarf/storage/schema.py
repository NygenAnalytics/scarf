import numpy as np
import zarr

from .types import as_zarr_array, as_zarr_group
from .arrays import (
    create_numeric_array,
    create_zarr_dataset,
    create_zarr_obj_array,
)
from .layout import _group_zarr_format, count_array_spec
from .profiles import get_storage_profile
from .sharding import finalize_sharded_counts, write_counts_t

PSEUDOTIME_AGGREGATION_SCHEMA_VERSION = 2


def create_zarr_count_assay(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    chunk_size: tuple[int, int],
    n_cells: int,
    feat_ids: np.ndarray | list[str],
    feat_names: np.ndarray | list[str],
    dtype: str = "uint32",
    *,
    targetChunkBytes: int | None = None,
    minFeatureChunk: int | None = None,
    maxFeatureChunk: int | None = None,
) -> zarr.Array:
    if workspace is None:
        group = z.create_group(assay_name, overwrite=True)
    else:
        group = z.create_group(f"{workspace}/{assay_name}", overwrite=True)
    group.attrs["is_assay"] = True
    group.attrs["misc"] = {}
    create_zarr_obj_array(group, "featureData/ids", feat_ids)
    create_zarr_obj_array(group, "featureData/names", feat_names)
    create_zarr_obj_array(
        group,
        "featureData/I",
        [True for _ in range(len(feat_ids))],
        "bool",
    )
    if workspace is not None:
        group = z.create_group(f"matrices/{assay_name}", overwrite=True)
    n_feats = len(feat_ids)
    if _group_zarr_format(group) >= 3:
        spec = count_array_spec(
            n_cells,
            n_feats,
            dtype=dtype,
            remote=get_storage_profile() == "cloud",
            targetChunkBytes=targetChunkBytes,
            minFeatureChunk=minFeatureChunk,
            maxFeatureChunk=maxFeatureChunk,
        )
        return create_numeric_array(group, "counts", spec)
    return create_zarr_dataset(
        group,
        "counts",
        chunk_size,
        dtype,
        (n_cells, n_feats),
        overwrite=True,
    )


def finalize_counts(
    store: zarr.Group,
    assay_name: str,
    workspace: str | None = None,
) -> zarr.Array:
    counts = finalize_sharded_counts(store, assay_name, workspace)
    if workspace is None:
        group = as_zarr_group(store[assay_name], name=assay_name)
    else:
        group = as_zarr_group(
            store[f"matrices/{assay_name}"],
            name=f"matrices/{assay_name}",
        )
    write_counts_t(counts, group)
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
) -> zarr.Group:
    if workspace is None:
        group = root.create_group("cellData")
    else:
        group = root.create_group(f"{workspace}/cellData")
    create_zarr_obj_array(group, "ids", ids, ids.dtype)
    create_zarr_obj_array(group, "names", names, names.dtype)
    create_zarr_obj_array(group, "I", [True for _ in range(len(ids))], "bool")
    return group
