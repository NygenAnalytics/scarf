import math
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import zarr
from zarr.abc.store import Store
from zarr.codecs import BloscCodec, ZstdCodec

from .._types import ZarrMode, as_zarr_array, as_zarr_group, array_metadata_shards
from .budget import (
    ResourceBudget,
    concurrency_for_chunk,
    get_resource_budget,
)

StorageProfile = Literal["fast_local", "cloud"]

type ZarrLocation = str | Store

PROFILE_COUNT_CHUNKS: dict[StorageProfile, tuple[int, int]] = {
    "fast_local": (512, 512),
    "cloud": (256, 256),
}
PROFILE_COUNT_SHARDS: dict[StorageProfile, tuple[int, int]] = {
    "fast_local": (4096, 4096),
    "cloud": (8192, 8192),
}
PROFILE_METADATA_CHUNK = 100_000
PROFILE_NORMED_TARGET_SHARD_BYTES = 256 * 1024 * 1024
PROFILE_STREAM_TARGET_BYTES: dict[StorageProfile, int] = {
    "fast_local": 64 * 1024 * 1024,
    "cloud": 192 * 1024 * 1024,
}
PROFILE_ASYNC_CONCURRENCY: dict[StorageProfile, int] = {
    "fast_local": 10,
    "cloud": 64,
}
PROFILE_PREFETCH_DEPTH: dict[StorageProfile, int] = {
    "fast_local": 1,
    "cloud": 4,
}


@dataclass(frozen=True)
class ZarrLayout:
    """Dimension-aware Zarr IO layout (chunks, shards, streaming, concurrency)."""

    countChunks: tuple[int, int]
    countShards: tuple[int, int]
    normedRowChunk: int
    normedColChunk: int
    metadataChunk: int
    streamTargetBytes: int
    asyncConcurrency: int
    prefetchDepth: int
    remote: bool = False


def _log_interp(
    value: float, lo: float, hi: float, outLo: float, outHi: float
) -> float:
    value = max(lo, min(hi, value))
    if hi <= lo:
        return outLo
    t = (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return outLo + t * (outHi - outLo)


def compute_zarr_layout(
    nCells: int,
    nFeatures: int,
    *,
    remote: bool = False,
    budget: ResourceBudget | None = None,
) -> ZarrLayout:
    """Derive chunk, shard, and IO settings from matrix dimensions.

    Memory-driven knobs (streaming bytes, async concurrency, prefetch depth)
    are capped by ``budget`` so peak memory tracks ``workers * tileBytes``.
    """
    nCells = max(int(nCells), 1)
    nFeatures = max(int(nFeatures), 1)
    matrixSize = nCells * nFeatures
    budget = budget or get_resource_budget()

    rowChunk = int(_log_interp(nCells, 1e3, 1e6, 256, 4096))
    # Narrow feature chunks keep scattered-feature reads from dragging full
    # wide chunks; row scans still read every chunk regardless of width.
    colChunk = int(_log_interp(nFeatures, 1e3, 5e4, 128, 512))
    rowChunk = min(rowChunk, nCells)
    colChunk = min(colChunk, nFeatures)

    shardRows = int(_log_interp(nCells, 1e3, 1e6, 2048, 16384))
    shardCols = int(_log_interp(nFeatures, 1e3, 5e4, 2048, 8192))
    shardRows = min(shardRows, nCells)
    shardCols = min(shardCols, nFeatures)
    if remote:
        shardRows = min(nCells, max(shardRows, rowChunk * 4))
        # Wide feature shards span many features per object (fewer R2 objects).
        shardCols = min(nFeatures, max(shardCols, colChunk * 16))
    # Keep shard edges as whole multiples of the chunk size.
    shardCols = max(colChunk, (shardCols // colChunk) * colChunk)
    shardRows = max(rowChunk, (shardRows // rowChunk) * rowChunk)
    shardCols = min(shardCols, nFeatures)
    shardRows = min(shardRows, nCells)

    normedRow = min(2048, nCells)
    normedCol = min(
        nFeatures,
        int(_log_interp(nFeatures, 500, 5e4, 512, 2048)),
    )
    streamBytes = int(
        _log_interp(matrixSize, 1e7, 5e10, 64 * 1024 * 1024, 512 * 1024 * 1024)
    )
    if remote:
        streamBytes = min(streamBytes * 2, 1024 * 1024 * 1024)
    streamBytes = min(streamBytes, budget.perWorkerBytes)

    asyncConcurrency = int(_log_interp(nCells, 1e3, 1e6, 8, 64 if remote else 32))
    chunkBytes = rowChunk * colChunk * 4
    asyncConcurrency = min(asyncConcurrency, concurrency_for_chunk(chunkBytes, budget))
    prefetchDepth = max(1, int(_log_interp(nCells, 1e3, 1e6, 1, 6 if remote else 2)))
    prefetchDepth = max(
        1, min(prefetchDepth, budget.memoryBytes // max(1, streamBytes))
    )
    metadataChunk = min(PROFILE_METADATA_CHUNK, nCells)

    return ZarrLayout(
        countChunks=(rowChunk, colChunk),
        countShards=(shardRows, shardCols),
        normedRowChunk=normedRow,
        normedColChunk=normedCol,
        metadataChunk=metadataChunk,
        streamTargetBytes=streamBytes,
        asyncConcurrency=asyncConcurrency,
        prefetchDepth=prefetchDepth,
        remote=remote,
    )


def marker_batch_size(
    nCells: int,
    nFeatures: int,
    layout: ZarrLayout,
    itemsize: int = 4,
) -> int:
    """Number of feature columns to load per marker batch.

    Bounded by ``layout.streamTargetBytes`` (assuming float32 normalized
    values) and aligned to the count column chunk so cloud reads do not
    amplify across partial chunks.
    """
    nCells = max(int(nCells), 1)
    nFeatures = max(int(nFeatures), 1)
    colChunk = max(1, layout.countChunks[1])
    cols = max(1, layout.streamTargetBytes // (nCells * itemsize))
    if cols >= colChunk:
        cols = (cols // colChunk) * colChunk
    else:
        # The budget cannot fit a whole column chunk. Snap down to the largest
        # divisor of the chunk so each batch stays inside one column chunk and
        # batches do not straddle chunk boundaries (which would re-read chunks).
        while colChunk % cols != 0 and cols > 1:
            cols -= 1
    cols = min(cols, nFeatures)
    return int(max(1, cols))


_activeLayout: ZarrLayout | None = None


def set_zarr_layout(layout: ZarrLayout | None) -> None:
    """Apply dimension-aware layout overrides for this process."""
    global _activeLayout
    _activeLayout = layout
    if layout is not None:
        zarr.config.set({"async.concurrency": layout.asyncConcurrency})


def get_zarr_layout() -> ZarrLayout | None:
    return _activeLayout


@dataclass
class ZarrArraySpec:
    """Specification for creating a numeric Zarr array."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: Any
    shards: tuple[int, ...] | None = None
    zarrFormat: int = 3
    compressors: list | None = None
    fillValue: Any | None = None
    overwrite: bool = True


def get_compressors(
    profile: StorageProfile = "fast_local", zarrFormat: int = 3
) -> list:
    """Return codec list for the given storage profile and Zarr format."""
    if zarrFormat == 2:
        from numcodecs import Blosc

        return [Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)]
    if profile == "cloud":
        return [ZstdCodec(level=3)]
    return [BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")]


def _group_zarr_format(group: zarr.Group) -> int:
    metadata = getattr(group, "metadata", None)
    if metadata is not None and getattr(metadata, "zarr_format", None) is not None:
        return int(metadata.zarr_format)
    return 3


def zarr_group_root(group: zarr.Group, mode: ZarrMode = "r+") -> zarr.Group:
    """Open the root Zarr group sharing the same store as ``group``."""
    return zarr.open_group(store=group.store, mode=mode)


def zarr_root_path(group: zarr.Group) -> str | None:
    """Return filesystem path for a Zarr group, if available."""
    store = group.store
    root = getattr(store, "root", None)
    if root is not None:
        return str(root)
    storePath = getattr(group, "store_path", None)
    if storePath and str(storePath).startswith("file://"):
        return str(storePath)[7:]
    return None


def normalize_chunks(
    chunks: tuple[int, ...] | int, shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Map chunk specification to a valid per-dimension chunk tuple for ``shape``."""
    if isinstance(chunks, int):
        chunks = (chunks,)
    if len(chunks) == len(shape):
        return tuple(min(chunk, dim) for chunk, dim in zip(chunks, shape))
    if len(chunks) == 1 and len(shape) > 1:
        return (min(chunks[0], shape[0]),) + tuple(shape[1:])
    if len(chunks) == 1:
        return (min(chunks[0], shape[0]),)
    raise ValueError(
        f"Cannot map chunks {chunks} to array shape {shape}. "
        "Provide one chunk size per dimension."
    )


_activeProfile: StorageProfile | None = None


def set_storage_profile(profile: StorageProfile | None) -> None:
    """Override the active Zarr storage profile for this process."""
    global _activeProfile
    _activeProfile = profile


def get_storage_profile() -> StorageProfile:
    """Return active profile from override or ``SCARF_ZARR_PROFILE`` env var."""
    if _activeProfile is not None:
        return _activeProfile
    envProfile = os.environ.get("SCARF_ZARR_PROFILE", "fast_local")
    if envProfile == "cloud":
        return "cloud"
    if envProfile == "fast_local":
        return "fast_local"
    return "fast_local"


def profile_prefetch_depth() -> int:
    """Return bounded read-ahead depth for the active storage profile."""
    if _activeLayout is not None:
        return _activeLayout.prefetchDepth
    budget = get_resource_budget()
    return max(1, min(PROFILE_PREFETCH_DEPTH[get_storage_profile()], budget.workers))


def configure_zarr_io_for_profile() -> None:
    """Apply zarr async IO settings for the active storage profile."""
    if _activeLayout is not None:
        zarr.config.set({"async.concurrency": _activeLayout.asyncConcurrency})
        return
    budget = get_resource_budget()
    concurrency = min(
        PROFILE_ASYNC_CONCURRENCY[get_storage_profile()],
        concurrency_for_chunk(8 * 1024 * 1024, budget),
    )
    zarr.config.set({"async.concurrency": concurrency})


def count_array_spec(
    nCells: int,
    nFeats: int,
    dtype: Any = "uint32",
    profile: StorageProfile | None = None,
    sharded: bool = False,
    layout: ZarrLayout | None = None,
) -> ZarrArraySpec:
    """Build array spec for assay count matrices."""
    profile = profile or get_storage_profile()
    layout = layout or _activeLayout
    if layout is not None:
        chunks = tuple(min(c, d) for c, d in zip(layout.countChunks, (nCells, nFeats)))
        shards = (
            tuple(min(s, d) for s, d in zip(layout.countShards, (nCells, nFeats)))
            if sharded
            else None
        )
    else:
        chunks = tuple(
            min(c, d) for c, d in zip(PROFILE_COUNT_CHUNKS[profile], (nCells, nFeats))
        )
        shards = (
            tuple(
                min(s, d)
                for s, d in zip(PROFILE_COUNT_SHARDS[profile], (nCells, nFeats))
            )
            if sharded
            else None
        )
    return ZarrArraySpec(
        shape=(nCells, nFeats),
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        compressors=get_compressors("cloud" if (layout and layout.remote) else profile),
        fillValue=0,
    )


def metadata_array_spec(
    length: int,
    dtype: Any,
    profile: StorageProfile | None = None,
    layout: ZarrLayout | None = None,
) -> ZarrArraySpec:
    """Build array spec for 1D metadata columns."""
    layout = layout or _activeLayout
    if layout is not None:
        chunkSize = min(layout.metadataChunk, max(length, 1))
    else:
        chunkSize = min(PROFILE_METADATA_CHUNK, max(length, 1))
    return ZarrArraySpec(
        shape=(length,),
        chunks=(chunkSize,),
        dtype=dtype,
        compressors=get_compressors(profile or get_storage_profile()),
    )


def _aligned_shard_dim(shard_target: int, chunk: int, dim: int) -> int:
    """Snap a shard edge to a multiple of ``chunk``, capped at ``dim``."""
    if dim <= 0:
        return 1
    chunk = min(chunk, dim)
    shard_target = min(shard_target, dim)
    if shard_target <= chunk:
        return chunk
    aligned = (shard_target // chunk) * chunk
    if aligned < chunk:
        return chunk
    return aligned


def normed_array_spec(
    nCells: int,
    nFeats: int,
    profile: StorageProfile | None = None,
    layout: ZarrLayout | None = None,
) -> ZarrArraySpec:
    """Build array spec for normalized expression matrices (graph-building slot)."""
    profile = profile or get_storage_profile()
    layout = layout or _activeLayout
    if layout is not None:
        n_cols = min(nFeats, layout.normedColChunk) if nFeats > 0 else 1
        row_chunk = min(layout.normedRowChunk, max(nCells, 1))
    else:
        n_cols = min(nFeats, 2048) if nFeats > 0 else 1
        row_chunk = min(2048, max(nCells, 1))
    chunks = (row_chunk, n_cols)
    shards = None
    useCloudShards = layout.remote if layout is not None else profile == "cloud"
    if useCloudShards and nCells > 0 and nFeats > 0:
        col_shard = _aligned_shard_dim(min(8192, nFeats), n_cols, nFeats)
        shard_rows = min(16384, nCells)
        bytes_per_row = col_shard * 4
        if bytes_per_row * shard_rows > PROFILE_NORMED_TARGET_SHARD_BYTES:
            shard_rows = max(
                row_chunk,
                PROFILE_NORMED_TARGET_SHARD_BYTES // max(bytes_per_row, 1),
            )
        shard_rows = _aligned_shard_dim(shard_rows, row_chunk, nCells)
        shards = (shard_rows, col_shard)
    return ZarrArraySpec(
        shape=(nCells, nFeats),
        chunks=chunks,
        shards=shards,
        dtype="float32",
        compressors=get_compressors("cloud" if useCloudShards else profile),
        fillValue=0.0,
    )


def streaming_block_size(
    backing: zarr.Array,
    profile: StorageProfile | None = None,
    target_bytes: int | None = None,
    layout: ZarrLayout | None = None,
) -> int:
    """Pick a row block size for streaming reads that targets a byte budget."""
    import numpy as np

    profile = profile or get_storage_profile()
    layout = layout or _activeLayout
    if target_bytes is None:
        if layout is not None:
            target_bytes = layout.streamTargetBytes
        else:
            target_bytes = PROFILE_STREAM_TARGET_BYTES[profile]
    # Bound by a single worker's memory slice. The backing is typically uint32
    # but reductions materialize float64 plus transients, so reserve headroom.
    budget = get_resource_budget()
    expansion_factor = 4
    target_bytes = min(target_bytes, budget.perWorkerBytes // expansion_factor)
    n_rows, n_cols = backing.shape
    if n_rows == 0:
        return 1
    itemsize = int(np.dtype(backing.dtype).itemsize)
    chunk_rows = int(backing.chunks[0]) if getattr(backing, "chunks", None) else n_rows
    block_rows = max(chunk_rows, target_bytes // max(n_cols * itemsize, 1))
    # Keep blocks aligned to whole row chunks for predictable, minimal reads.
    block_rows = max(chunk_rows, (block_rows // chunk_rows) * chunk_rows)
    metadata = getattr(backing, "metadata", None)
    shard_rows = getattr(metadata, "shards", None)
    if shard_rows is not None and len(shard_rows) > 0:
        shard_rows = int(shard_rows[0])
        if shard_rows > chunk_rows and block_rows >= shard_rows:
            block_rows = (block_rows // shard_rows) * shard_rows
            if block_rows < shard_rows:
                block_rows = shard_rows
    return min(max(int(block_rows), 1), n_rows)


def is_remote_zarr_location(location: str) -> bool:
    """Return True when ``location`` is a non-local URI (e.g. s3://, gs://)."""
    if "://" not in location:
        return False
    return not location.startswith("file://")


def is_local_zarr_path(location: ZarrLocation) -> bool:
    """Return True when ``location`` is a plain local filesystem path string."""
    return isinstance(location, str) and not is_remote_zarr_location(location)


def is_remote_datastore(zarr_loc: ZarrLocation, group: zarr.Group) -> bool:
    """Return True when the datastore primary store is a remote/object backend."""
    if isinstance(zarr_loc, str):
        return is_remote_zarr_location(zarr_loc)
    if zarr_root_path(group) is not None:
        return False
    store_name = type(group.store).__name__
    if store_name in ("MemoryStore", "LocalStore"):
        return False
    return True


def copy_zarr_array(
    src: zarr.Array,
    dst: zarr.Array,
    block_rows: int | None = None,
    msg: str | None = None,
) -> None:
    """Stream-copy a 2D Zarr array in row blocks."""
    from ..utils import tqdmbar

    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: src {src.shape} vs dst {dst.shape}")
    if len(src.shape) != 2:
        raise ValueError("copy_zarr_array only supports 2D arrays")
    n_rows = src.shape[0]
    if n_rows == 0:
        return None
    if block_rows is None:
        block_rows = streaming_block_size(src, profile="fast_local")
    block_rows = max(int(block_rows), 1)
    if msg is None:
        msg = "Copying Zarr array"
    n_blocks = int(np.ceil(n_rows / block_rows))
    for start in tqdmbar(range(0, n_rows, block_rows), desc=msg, total=n_blocks):
        end = min(start + block_rows, n_rows)
        dst[start:end, :] = np.asarray(src[start:end, :])
    return None


def open_or_create_staged_normed_array(
    cache_path: str,
    src: zarr.Array,
) -> zarr.Array:
    """Open a reusable local scratch array for staged normalized data."""
    import os

    if os.path.exists(os.path.join(cache_path, "zarr.json")):
        root = zarr.open_group(cache_path, mode="r+")
        if "data" in root:
            arr = as_zarr_array(root["data"], name="data")
            if arr.shape == src.shape:
                return arr
    root = zarr.open_group(cache_path, mode="w")
    spec = normed_array_spec(src.shape[0], src.shape[1], profile="fast_local")
    return create_numeric_array(root, "data", spec)


def _is_obstore_native_store(obj: object) -> bool:
    return type(obj).__module__.startswith("obstore.")


def _maybe_auto_cloud_profile(location: ZarrLocation) -> None:
    """Select the cloud storage profile when opening a remote store without an explicit profile."""
    if _activeProfile is not None:
        return
    if isinstance(location, str) and not is_remote_zarr_location(location):
        return
    if isinstance(location, Store):
        return
    set_storage_profile("cloud")


def make_store(
    location: ZarrLocation,
    *,
    storage_options: dict[str, Any] | None = None,
    read_only: bool = False,
) -> str | Store:
    """Resolve a path, URI, or store object into something ``zarr.open_group`` accepts."""
    if isinstance(location, Store):
        return location

    if _is_obstore_native_store(location):
        from zarr.storage import ObjectStore

        return ObjectStore(store=location, read_only=read_only)  # type: ignore[type-var]

    if isinstance(location, str):
        if is_remote_zarr_location(location):
            try:
                from obstore.store import from_url as obstore_from_url
                from zarr.storage import ObjectStore
            except ImportError as exc:
                raise ImportError("Remote Zarr stores require obstore.") from exc
            _maybe_auto_cloud_profile(location)
            obstore = obstore_from_url(location, **(storage_options or {}))
            return ObjectStore(store=obstore, read_only=read_only)  # type: ignore[type-var]
        return location

    raise TypeError(
        f"zarr location must be a path string or zarr Store, got {type(location)!r}"
    )


def open_store(
    path: ZarrLocation,
    mode: ZarrMode = "r",
    storage_options: dict[str, Any] | None = None,
) -> zarr.Group:
    """Open a Zarr group at ``path`` or from a store object."""
    store = make_store(path, storage_options=storage_options, read_only=(mode == "r"))
    configure_zarr_io_for_profile()
    if isinstance(store, str):
        return zarr.open_group(store, mode=mode)
    return zarr.open_group(store=store, mode=mode)


def create_numeric_array(
    group: zarr.Group,
    name: str,
    spec: ZarrArraySpec,
) -> zarr.Array:
    """Create a numeric Zarr array from a ``ZarrArraySpec``."""
    zarrFormat = _group_zarr_format(group)
    chunks = normalize_chunks(spec.chunks, spec.shape)
    kwargs: dict[str, Any] = {
        "shape": spec.shape,
        "chunks": chunks,
        "dtype": spec.dtype,
        "compressors": get_compressors(zarrFormat=zarrFormat),
        "overwrite": spec.overwrite,
    }
    if spec.shards is not None:
        kwargs["shards"] = spec.shards
    if spec.fillValue is not None:
        kwargs["fill_value"] = spec.fillValue
    return group.create_array(name, **kwargs)


def dtype_fix(dtype: Any, data: np.ndarray) -> Any:
    """Infer or adjust dtype for metadata arrays from sample values."""
    if dtype is None or np.dtype(dtype).kind == "O":
        return "U" + str(max([len(str(x)) for x in data]))
    if np.issubdtype(data.dtype, np.dtype("S")):
        try:
            adata = data.astype("U")
        except UnicodeDecodeError:
            adata = np.array([x.decode("UTF-8") for x in data]).astype("U")
        return adata.dtype
    return dtype


def create_metadata_column(
    group: zarr.Group,
    name: str,
    data: np.ndarray | list | None = None,
    dtype: Any = None,
    overwrite: bool = True,
    chunkSize: int | bool | None = None,
    shape: int | None = None,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    """Create a 1D metadata column, optionally from provided data."""
    if chunkSize is None or chunkSize is False:
        chunks: tuple[int, ...] | bool = False
    else:
        chunks = (chunkSize,)

    compressors = get_compressors(
        profile or get_storage_profile(),
        zarrFormat=_group_zarr_format(group),
    )

    if data is not None:
        data = np.array(data)
        data = np.asarray(data, dtype=dtype_fix(dtype, data))
        if chunks is False:
            chunks = (len(data),)
        return group.create_array(
            name,
            data=data,
            chunks=chunks,
            overwrite=overwrite,
            compressors=compressors,
        )

    if shape is None:
        raise ValueError("shape is required when data is None")
    if chunks is False:
        chunks = (shape,)
    return group.create_array(
        name,
        shape=(shape,),
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        compressors=compressors,
    )


def repack_to_sharded(
    srcArray: zarr.Array,
    dstGroup: zarr.Group,
    name: str,
    shards: tuple[int, ...],
    chunks: tuple[int, ...] | None = None,
    profile: StorageProfile | None = None,
    overwrite: bool = True,
) -> zarr.Array:
    """Copy ``srcArray`` into a new sharded array in ``dstGroup``."""
    chunks = chunks or tuple(srcArray.chunks)
    spec = ZarrArraySpec(
        shape=srcArray.shape,
        chunks=chunks,
        shards=shards,
        dtype=srcArray.dtype,
        compressors=get_compressors(profile or get_storage_profile()),
        overwrite=overwrite,
    )
    dstArray = create_numeric_array(dstGroup, name, spec)
    shardRows, shardCols = shards[0], shards[1]
    nRows, nCols = srcArray.shape
    for rowStart in range(0, nRows, shardRows):
        rowEnd = min(rowStart + shardRows, nRows)
        for colStart in range(0, nCols, shardCols):
            colEnd = min(colStart + shardCols, nCols)
            dstArray[rowStart:rowEnd, colStart:colEnd] = srcArray[
                rowStart:rowEnd, colStart:colEnd
            ]
    return dstArray


def finalize_sharded_counts(
    store: zarr.Group,
    assayName: str,
    workspace: str | None = None,
    profile: StorageProfile | None = None,
    layout: ZarrLayout | None = None,
) -> zarr.Array:
    """Repack assay counts to sharded layout when not already sharded."""
    profile = profile or get_storage_profile()
    layout = layout or _activeLayout
    if workspace is None:
        countsPath = f"{assayName}/counts"
        assayGroup = as_zarr_group(store[assayName], name=assayName)
    else:
        countsPath = f"matrices/{assayName}/counts"
        assayGroup = as_zarr_group(store[f"matrices/{assayName}"], name=assayName)

    srcArray = as_zarr_array(store[countsPath], name=countsPath)
    if array_metadata_shards(srcArray) is not None:
        return srcArray
    if _group_zarr_format(store) == 2:
        return srcArray

    if layout is not None:
        shards = tuple(min(s, d) for s, d in zip(layout.countShards, srcArray.shape))
        chunks = tuple(min(c, d) for c, d in zip(layout.countChunks, srcArray.shape))
    else:
        shards = tuple(
            min(s, d) for s, d in zip(PROFILE_COUNT_SHARDS[profile], srcArray.shape)
        )
        chunks = tuple(
            min(c, d) for c, d in zip(PROFILE_COUNT_CHUNKS[profile], srcArray.shape)
        )
    normalized: list[tuple[int, int]] = []
    for shardDim, chunkDim in zip(shards, chunks):
        if shardDim <= chunkDim or shardDim % chunkDim != 0:
            normalized.append((shardDim, shardDim))
        else:
            normalized.append((shardDim, chunkDim))
    shards = tuple(dim[0] for dim in normalized)
    chunks = tuple(dim[1] for dim in normalized)
    if shards == srcArray.shape:
        return srcArray
    tmpName = "counts__sharded_tmp"
    if tmpName in assayGroup:
        del assayGroup[tmpName]
    repack_to_sharded(
        srcArray,
        assayGroup,
        tmpName,
        shards=shards,
        chunks=chunks,
        profile=profile,
    )
    del assayGroup["counts"]
    repack_to_sharded(
        as_zarr_array(assayGroup[tmpName], name=tmpName),
        assayGroup,
        "counts",
        shards=shards,
        chunks=chunks,
        profile=profile,
    )
    del assayGroup[tmpName]

    if workspace is None:
        assayGroup = as_zarr_group(store[assayName], name=assayName)
    else:
        assayGroup = as_zarr_group(store[f"matrices/{assayName}"], name=assayName)
    assayGroup.attrs["scarf:zarr_spec"] = {
        "profile": profile,
        "chunks": list(chunks),
        "shards": list(shards),
        "zarr_format": 3,
    }
    return as_zarr_array(store[countsPath], name=countsPath)


def array_info(array: zarr.Array) -> str:
    """Return a short summary string for a Zarr array."""
    parts = [f"shape={array.shape}", f"dtype={array.dtype}", f"chunks={array.chunks}"]
    shards_meta = array_metadata_shards(array)
    if shards_meta is not None:
        parts.append(f"shards={shards_meta}")
    return ", ".join(parts)


ANN_INDEX_ARRAY = "ann_idx_bytes"
ANN_INDEX_FORMAT = "zarr-uint8-v1"
ANN_INDEX_CHUNK_BYTES = 8 * 1024 * 1024


def has_ann_index(group: zarr.Group, name: str = ANN_INDEX_ARRAY) -> bool:
    """Return True when an in-zarr ANN index byte array exists."""
    return name in group


def legacy_ann_index_path(zw_root: str | None, ann_loc: str) -> str | None:
    """Return filesystem path to a legacy hnswlib sibling index file, if applicable."""
    if zw_root is None:
        return None
    return os.path.join(zw_root, ann_loc, "ann_idx")


def save_ann_index(
    group: zarr.Group,
    ann_idx: Any,
    name: str = ANN_INDEX_ARRAY,
    profile: StorageProfile | None = None,
) -> None:
    """Persist an hnswlib index as a chunked uint8 array inside ``group``."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        ann_idx.save_index(path)
        data = np.fromfile(path, dtype=np.uint8)
    finally:
        os.unlink(path)

    if name in group:
        del group[name]
    chunk_size = min(ANN_INDEX_CHUNK_BYTES, max(int(data.shape[0]), 1))
    zarr_format = _group_zarr_format(group)
    arr = group.create_array(
        name,
        shape=data.shape,
        chunks=(chunk_size,),
        dtype="uint8",
        overwrite=True,
        compressors=get_compressors(
            profile or get_storage_profile(),
            zarrFormat=zarr_format,
        ),
    )
    arr[:] = data
    group.attrs["annIndexFormat"] = ANN_INDEX_FORMAT
    arr.attrs["byteLength"] = int(data.shape[0])


def load_ann_index(
    group: zarr.Group,
    space: str,
    dim: int,
    name: str = ANN_INDEX_ARRAY,
) -> Any:
    """Load an hnswlib index from an in-zarr uint8 byte array."""
    import tempfile

    import hnswlib

    if name not in group:
        raise FileNotFoundError(f"ANN index array {name!r} not found in group")
    data = np.asarray(as_zarr_array(group[name], name=name)[:], dtype=np.uint8)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        data.tofile(path)
        idx = hnswlib.Index(space=space, dim=dim)
        idx.load_index(path)
        return idx
    finally:
        os.unlink(path)


def load_ann_index_from_path(path: str, space: str, dim: int) -> Any:
    """Load an hnswlib index from a legacy filesystem path."""
    import hnswlib

    idx = hnswlib.Index(space=space, dim=dim)
    idx.load_index(path)
    return idx
