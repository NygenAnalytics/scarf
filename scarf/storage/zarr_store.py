import math
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import zarr
from zarr.abc.store import Store
from zarr.codecs import BloscCodec, ZstdCodec

from .._types import ZarrMode, as_zarr_array, as_zarr_group, array_metadata_shards
from .budget import ResourceBudget, get_resource_budget

StorageProfile = Literal["fast_local", "cloud"]

type ZarrLocation = str | Store

PROFILE_METADATA_CHUNK = 100_000
# numcodecs compression codecs reject buffers larger than a signed 32-bit byte count.
_CODEC_MAX_BYTES = 2_147_483_647
DEFAULT_CLOUD_TARGET_CHUNK_BYTES = 128 * 1024 * 1024
DEFAULT_MIN_FEATURE_CHUNK = 500
DEFAULT_MAX_FEATURE_CHUNK = 10_000


def _ceil_pad(n: int, chunk: int) -> int:
    chunk = max(1, int(chunk))
    n = max(1, int(n))
    return math.ceil(n / chunk) * chunk


def _fit_shard_to_byte_limit(
    row_shard: int,
    feature_chunk: int,
    shard_cols: int,
    n_features: int,
    *,
    itemsize: int,
    max_bytes: int,
) -> tuple[int, int, int]:
    """Shrink shard geometry so ``row_shard * shard_cols * itemsize <= max_bytes``."""

    def shard_bytes(rs: int, sc: int) -> int:
        return rs * sc * itemsize

    while shard_bytes(row_shard, shard_cols) > max_bytes:
        if row_shard > 1:
            row_shard = max(1, max_bytes // (shard_cols * itemsize))
            while row_shard > 1 and shard_bytes(row_shard, shard_cols) > max_bytes:
                row_shard -= 1
        elif shard_cols > feature_chunk:
            shard_cols = max(
                feature_chunk,
                (max_bytes // (row_shard * itemsize) // feature_chunk) * feature_chunk,
            )
            if shard_bytes(row_shard, shard_cols) > max_bytes:
                shard_cols = max(feature_chunk, shard_cols - feature_chunk)
        elif feature_chunk > 1:
            feature_chunk = max(1, max_bytes // (row_shard * itemsize))
            shard_cols = _ceil_pad(n_features, feature_chunk)
            if shard_bytes(row_shard, shard_cols) > max_bytes:
                shard_cols = max(
                    feature_chunk,
                    (max_bytes // (row_shard * itemsize) // feature_chunk)
                    * feature_chunk,
                )
        else:
            feature_chunk = max(1, max_bytes // itemsize)
            shard_cols = feature_chunk
            row_shard = 1
            break
    return row_shard, feature_chunk, shard_cols


def matrix_layout(
    n_cells: int,
    n_features: int,
    *,
    budget: ResourceBudget | None = None,
    itemsize: int = 4,
    width: int | None = None,
    targetChunkBytes: int | None = None,
    minFeatureChunk: int = 1,
    maxFeatureChunk: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Return ``(chunks, shards)`` from the memory-first layout rule.

    Geometry is driven by ``memoryBytes // workingCopies`` so a single chunk
    (or shard) band is a budget-sized read unit, capped at the codec's maximum
    input buffer size. When ``width`` is set (normalized/derived), returns plain
    chunks ``(rowShard, width)`` and ``shards=None``. Otherwise returns
    count-matrix geometry with ceil-padded feature shards.

    When ``targetChunkBytes`` is set, feature-chunk width is chosen so
    ``row_shard * feature_chunk * itemsize`` stays near that target, then
    clamped to ``[minFeatureChunk, maxFeatureChunk]``.
    """
    budget = budget or get_resource_budget()
    n_cells = max(int(n_cells), 1)
    n_features = max(int(n_features), 1)
    itemsize = max(1, int(itemsize))
    work = budget.memoryBytes // max(1, budget.workingCopies)

    if width is not None:
        w = max(1, int(width))
        row_bytes = w * itemsize
        if row_bytes > _CODEC_MAX_BYTES:
            raise ValueError(
                f"One full-width row requires {row_bytes} bytes, exceeding "
                f"the codec limit of {_CODEC_MAX_BYTES} bytes"
            )
        max_chunk_bytes = min(work, _CODEC_MAX_BYTES)
        row_shard = max(1, min(n_cells, max_chunk_bytes // row_bytes))
        chunks = (row_shard, w)
        return chunks, None

    row_shard = max(1, min(n_cells, work // (n_features * itemsize)))
    feature_chunk = max(1, min(n_features, work // (n_cells * itemsize)))
    if targetChunkBytes is not None:
        target = max(itemsize, int(targetChunkBytes))
        feature_chunk = max(1, target // (row_shard * itemsize))
        feature_chunk = min(n_features, feature_chunk)
        lo = max(1, int(minFeatureChunk))
        hi = n_features if maxFeatureChunk is None else max(1, int(maxFeatureChunk))
        if lo > hi:
            lo, hi = hi, lo
        feature_chunk = max(lo, min(hi, feature_chunk))
        feature_chunk = min(n_features, feature_chunk)
    shard_cols = _ceil_pad(n_features, feature_chunk)
    max_shard_bytes = min(work, _CODEC_MAX_BYTES)
    row_shard, feature_chunk, shard_cols = _fit_shard_to_byte_limit(
        row_shard,
        feature_chunk,
        shard_cols,
        n_features,
        itemsize=itemsize,
        max_bytes=max_shard_bytes,
    )
    chunks = (row_shard, feature_chunk)
    shards = (row_shard, shard_cols)
    return chunks, shards


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


def configure_zarr_io_for_profile() -> None:
    """Set zarr async IO parallelism to the budget's worker count."""
    budget = get_resource_budget()
    zarr.config.set({"async.concurrency": max(1, budget.workers)})


def count_array_spec(
    nCells: int,
    nFeats: int,
    dtype: Any = "uint32",
    profile: StorageProfile | None = None,
    *,
    remote: bool | None = None,
    budget: ResourceBudget | None = None,
    targetChunkBytes: int | None = None,
    minFeatureChunk: int | None = None,
    maxFeatureChunk: int | None = None,
) -> ZarrArraySpec:
    """Build array spec for assay count matrices."""
    profile = profile or get_storage_profile()
    budget = budget or get_resource_budget()
    if remote is None:
        remote = profile == "cloud"
    itemsize = int(np.dtype(dtype).itemsize)

    layout_kwargs: dict[str, Any] = {}
    resolved_target = targetChunkBytes
    if resolved_target is None and remote:
        resolved_target = DEFAULT_CLOUD_TARGET_CHUNK_BYTES
    if resolved_target is not None:
        layout_kwargs["targetChunkBytes"] = resolved_target
        layout_kwargs["minFeatureChunk"] = (
            DEFAULT_MIN_FEATURE_CHUNK if minFeatureChunk is None else minFeatureChunk
        )
        layout_kwargs["maxFeatureChunk"] = (
            DEFAULT_MAX_FEATURE_CHUNK if maxFeatureChunk is None else maxFeatureChunk
        )

    chunks, shards_raw = matrix_layout(
        nCells,
        nFeats,
        budget=budget,
        itemsize=itemsize,
        **layout_kwargs,
    )
    assert shards_raw is not None
    # shards_raw's feature dimension is deliberately ceil-padded to a multiple
    # of chunks[1] (see matrix_layout); clipping it down to nFeats here would
    # make the shard size no longer divisible by the chunk size, which Zarr
    # v3 rejects. matrix_layout already bounds both chunk dims by (nCells,
    # nFeats) and the row shard dim by nCells, so no further clipping needed.
    shards = shards_raw

    return ZarrArraySpec(
        shape=(nCells, nFeats),
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        compressors=get_compressors("cloud" if remote else profile),
        fillValue=0,
    )


def metadata_array_spec(
    length: int,
    dtype: Any,
    profile: StorageProfile | None = None,
) -> ZarrArraySpec:
    """Build array spec for 1D metadata columns."""
    chunkSize = min(PROFILE_METADATA_CHUNK, max(length, 1))
    return ZarrArraySpec(
        shape=(length,),
        chunks=(chunkSize,),
        dtype=dtype,
        compressors=get_compressors(profile or get_storage_profile()),
    )


def normed_array_spec(
    nCells: int,
    nFeats: int,
    profile: StorageProfile | None = None,
    *,
    remote: bool | None = None,
    budget: ResourceBudget | None = None,
) -> ZarrArraySpec:
    """Build array spec for normalized expression matrices (graph-building slot)."""
    profile = profile or get_storage_profile()
    budget = budget or get_resource_budget()
    if remote is None:
        remote = profile == "cloud"
    itemsize = 4

    chunks, _ = matrix_layout(
        nCells,
        nFeats,
        budget=budget,
        itemsize=itemsize,
        width=max(nFeats, 1),
    )
    row_chunk, n_cols = chunks

    return ZarrArraySpec(
        shape=(nCells, nFeats),
        chunks=(row_chunk, n_cols),
        shards=None,
        dtype="float32",
        compressors=get_compressors("cloud" if remote else profile),
        fillValue=0.0,
    )


def array_shard_rows(array: zarr.Array) -> int:
    """Row extent of one shard, or row chunk, or the full height."""
    shards_meta = array_metadata_shards(array)
    if shards_meta is not None and len(shards_meta) > 0:
        return int(shards_meta[0])
    chunks = getattr(array, "chunks", None)
    if chunks is not None and len(chunks) > 0:
        return int(chunks[0])
    return max(int(array.shape[0]), 1)


def iter_shard_row_slices(n_rows: int, shard_rows: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` row slices aligned to ``shard_rows``."""
    shard_rows = max(1, int(shard_rows))
    n_rows = max(0, int(n_rows))
    for start in range(0, n_rows, shard_rows):
        yield start, min(start + shard_rows, n_rows)


def write_dense_from_row_batches(
    dst: zarr.Array,
    batches: Iterator[np.ndarray],
    *,
    dtype: Any | None = None,
    msg: str | None = None,
) -> int:
    """Write sequential dense row batches, flushing at destination row-shard edges."""
    from ..utils import tqdmbar

    shard_rows = array_shard_rows(dst)
    pending: list[np.ndarray] = []
    pending_rows = 0
    out_pos = 0
    total_rows = 0

    def flush_partial(final: bool = False) -> None:
        nonlocal pending, pending_rows, out_pos, total_rows
        while pending_rows >= shard_rows or (final and pending_rows > 0):
            block = np.vstack(pending) if len(pending) > 1 else pending[0]
            take = block.shape[0] if final and pending_rows < shard_rows else shard_rows
            piece = block[:take]
            if dtype is not None:
                piece = piece.astype(dtype, copy=False)
            dst[out_pos : out_pos + piece.shape[0], :] = piece
            out_pos += piece.shape[0]
            total_rows += piece.shape[0]
            if block.shape[0] > take:
                pending = [block[take:]]
                pending_rows = block.shape[0] - take
            else:
                pending = []
                pending_rows = 0
            if not final and pending_rows < shard_rows:
                break

    for batch in tqdmbar(batches, desc=msg or "Writing Zarr array"):
        batch = np.asarray(batch)
        if batch.size == 0:
            continue
        pending.append(batch)
        pending_rows += batch.shape[0]
        flush_partial()
    flush_partial(final=True)
    return total_rows


def write_dense_in_shard_rows(
    dst: zarr.Array,
    produce: Callable[[int, int], np.ndarray],
    *,
    msg: str | None = None,
    shard_rows: int | None = None,
    also_write_to: zarr.Array | None = None,
) -> None:
    """Write a 2D array in row-shard bands via ``produce(start, end)``.

    The produce side (read + compute) runs in order with a shallow read-ahead so
    the next band is fetched while the current one is written; the writes stay
    single-threaded so we never race on a shared chunk/shard file regardless of
    how the caller's band size aligns to the on-disk geometry.

    ``also_write_to`` mirrors each produced band into a second array of the same
    shape during the same pass. This populates a local staging cache while
    writing to a remote store, so the normalized matrix never has to be read
    back over the network.
    """
    from ..utils import tqdmbar
    from ..parallel import stream_shards
    from .budget import shard_parallelism

    n_rows = int(dst.shape[0])
    if n_rows == 0:
        return None
    if shard_rows is None:
        shard_rows = array_shard_rows(dst)
    slices = list(iter_shard_row_slices(n_rows, shard_rows))
    if msg is None:
        msg = "Writing Zarr array"

    plan = shard_parallelism(n_shards=len(slices))

    def produce_band(bounds: tuple[int, int]) -> tuple[int, int, np.ndarray]:
        start, end = bounds
        return start, end, np.asarray(produce(start, end))

    produced = stream_shards(
        slices,
        produce_band,
        workers=plan.readAhead,
        within_block_threads=plan.withinBlockThreads,
        io_concurrency=plan.ioConcurrency,
    )
    for start, end, block in tqdmbar(produced, desc=msg, total=len(slices)):
        dst[start:end, :] = block
        if also_write_to is not None:
            also_write_to[start:end, :] = block
    return None


def accumulate_sparse_to_shards(
    dst: zarr.Array,
    data_stream: Iterator[Any],
    *,
    shard_rows: int | None = None,
    dtype: Any | None = None,
) -> int:
    """Buffer sparse COO batches and write one coordinate selection per row shard."""
    from scipy.sparse import coo_matrix

    if shard_rows is None:
        shard_rows = array_shard_rows(dst)
    shard_rows = max(1, int(shard_rows))
    if dtype is None:
        dtype = dst.dtype

    s = 0
    buf_row: list[np.ndarray] = []
    buf_col: list[np.ndarray] = []
    buf_data: list[np.ndarray] = []
    next_flush = shard_rows

    def _concat_or_empty(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.array([], dtype=np.int64)
        return np.concatenate(parts)

    def flush_through(end_row: int) -> None:
        nonlocal next_flush, buf_row, buf_col, buf_data
        while next_flush <= end_row:
            band_start = next_flush - shard_rows
            row = _concat_or_empty(buf_row)
            col = _concat_or_empty(buf_col)
            data = _concat_or_empty(buf_data)
            if row.size:
                mask = (row >= band_start) & (row < next_flush)
                if mask.any():
                    dst.set_coordinate_selection(
                        (row[mask], col[mask]),
                        data[mask].astype(dtype, copy=False),
                    )
                keep = row >= next_flush
                if keep.any():
                    buf_row = [row[keep]]
                    buf_col = [col[keep]]
                    buf_data = [data[keep]]
                else:
                    buf_row = []
                    buf_col = []
                    buf_data = []
            next_flush += shard_rows

    for batch in data_stream:
        coo = batch if hasattr(batch, "row") else coo_matrix(batch)
        if coo.shape[0] == 0:
            continue
        if coo.nnz:
            buf_row.append(np.asarray(coo.row, dtype=np.int64) + s)
            buf_col.append(np.asarray(coo.col, dtype=np.int64))
            buf_data.append(np.asarray(coo.data))
        s += coo.shape[0]
        flush_through(s)

    if buf_row:
        row = _concat_or_empty(buf_row)
        col = _concat_or_empty(buf_col)
        data = _concat_or_empty(buf_data)
        if row.size:
            dst.set_coordinate_selection((row, col), data.astype(dtype, copy=False))
    return s


def is_remote_zarr_location(location: str) -> bool:
    """Return True when ``location`` is a non-local URI (e.g. s3://, gs://)."""
    if "://" not in location:
        return False
    return not location.startswith("file://")


def is_local_zarr_path(location: ZarrLocation) -> bool:
    """Return True when ``location`` is a plain local filesystem path string."""
    return isinstance(location, str) and not is_remote_zarr_location(location)


def is_remote_datastore(zarr_loc: ZarrLocation | None, group: zarr.Group) -> bool:
    """Return True when the datastore primary store is a remote/object backend."""
    if isinstance(zarr_loc, str) and zarr_loc:
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
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: src {src.shape} vs dst {dst.shape}")
    if len(src.shape) != 2:
        raise ValueError("copy_zarr_array only supports 2D arrays")
    if block_rows is None:
        block_rows = array_shard_rows(dst)
    write_dense_in_shard_rows(
        dst,
        lambda start, end: np.asarray(src[start:end, :]),
        msg=msg or "Copying Zarr array",
        shard_rows=block_rows,
    )
    return None


def copy_zarr_group_tree(
    src: zarr.Group,
    dst: zarr.Group,
    *,
    overwrite: bool = True,
) -> None:
    """Recursively copy a Zarr group tree from ``src`` into ``dst``."""
    from ..writers import create_zarr_obj_array

    for name, node in src.members():
        if isinstance(node, zarr.Group):
            child = dst.create_group(name, overwrite=overwrite)
            copy_zarr_group_tree(node, child, overwrite=overwrite)
        else:
            arr = as_zarr_array(node, name=name)
            create_zarr_obj_array(
                dst,
                name,
                np.asarray(arr[:]),
                dtype=arr.dtype,
                overwrite=overwrite,
            )


def create_or_open_staged_normed_array(
    cache_path: str,
    shape: tuple[int, int],
) -> zarr.Array:
    """Open (reusing when shape matches) a local scratch array of ``shape``."""
    import os

    if os.path.exists(os.path.join(cache_path, "zarr.json")):
        root = zarr.open_group(cache_path, mode="r+")
        if "data" in root:
            arr = as_zarr_array(root["data"], name="data")
            if tuple(arr.shape) == tuple(shape):
                return arr
    root = zarr.open_group(cache_path, mode="w")
    spec = normed_array_spec(shape[0], shape[1], profile="fast_local")
    return create_numeric_array(root, "data", spec)


def open_or_create_staged_normed_array(
    cache_path: str,
    src: zarr.Array,
) -> zarr.Array:
    """Open a reusable local scratch array matching ``src``'s shape."""
    return create_or_open_staged_normed_array(
        cache_path, (int(src.shape[0]), int(src.shape[1]))
    )


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
    profile = get_storage_profile()
    compressors = get_compressors(profile, zarrFormat=zarrFormat)
    if spec.compressors is not None and zarrFormat >= 3:
        compressors = spec.compressors
    kwargs: dict[str, Any] = {
        "shape": spec.shape,
        "chunks": chunks,
        "dtype": spec.dtype,
        "compressors": compressors,
        "overwrite": spec.overwrite,
    }
    if spec.shards is not None and zarrFormat >= 3:
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
    shardRows = int(shards[0])
    write_dense_in_shard_rows(
        dstArray,
        lambda start, end: np.asarray(srcArray[start:end, :]),
        msg="Repacking to sharded layout",
        shard_rows=shardRows,
    )
    return dstArray


def write_counts_t(counts: zarr.Array, group: zarr.Group) -> zarr.Array | None:
    """Write feature-major ``countsT`` next to a finalized ``counts`` array.

    ``countsT`` has shape ``(nFeatures, nCells)``. Returns ``None`` for Zarr v2
    stores, which keep the ``counts``-only layout.
    """
    from ..utils import logger, tqdmbar

    if _group_zarr_format(group) < 3:
        return None

    n_cells = int(counts.shape[0])
    n_feats = int(counts.shape[1])
    itemsize = max(1, int(np.dtype(counts.dtype).itemsize))
    source_row_chunk = max(1, int(counts.chunks[0]))
    feature_chunk = max(1, min(n_feats, int(counts.chunks[1])))

    bytes_per_row_band = feature_chunk * source_row_chunk * itemsize
    if bytes_per_row_band >= DEFAULT_CLOUD_TARGET_CHUNK_BYTES:
        cell_chunk = source_row_chunk
    else:
        cell_chunk = (
            DEFAULT_CLOUD_TARGET_CHUNK_BYTES // bytes_per_row_band
        ) * source_row_chunk
    cell_chunk = max(source_row_chunk, min(n_cells, cell_chunk))
    while (
        cell_chunk > source_row_chunk
        and feature_chunk * cell_chunk * itemsize > _CODEC_MAX_BYTES
    ):
        cell_chunk -= source_row_chunk
    cell_chunk = max(1, min(n_cells, cell_chunk))

    if "countsT" in group:
        del group["countsT"]
    counts_t = create_numeric_array(
        group,
        "countsT",
        ZarrArraySpec(
            shape=(n_feats, n_cells),
            chunks=(feature_chunk, cell_chunk),
            dtype=counts.dtype,
            shards=None,
            compressors=get_compressors(get_storage_profile()),
            fillValue=0,
            overwrite=True,
        ),
    )
    counts_t.attrs["complete"] = False
    logger.info(
        f"Writing countsT shape=({n_feats}, {n_cells}) "
        f"chunks=({feature_chunk}, {cell_chunk})"
    )

    feat_starts = list(range(0, n_feats, feature_chunk))
    cell_starts = list(range(0, n_cells, cell_chunk))
    total_tiles = max(1, len(feat_starts) * len(cell_starts))
    with tqdmbar(
        total=total_tiles,
        desc="Writing countsT",
    ) as progress:
        for feat_start in feat_starts:
            feat_end = min(feat_start + feature_chunk, n_feats)
            for cell_start in cell_starts:
                cell_end = min(cell_start + cell_chunk, n_cells)
                block = np.asarray(counts[cell_start:cell_end, feat_start:feat_end])
                counts_t[feat_start:feat_end, cell_start:cell_end] = block.T
                progress.update(1)

    counts_t.attrs["complete"] = True
    return counts_t


def finalize_sharded_counts(
    store: zarr.Group,
    assayName: str,
    workspace: str | None = None,
    profile: StorageProfile | None = None,
) -> zarr.Array:
    """Repack assay counts to sharded layout when not already sharded."""
    profile = profile or get_storage_profile()
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

    spec = count_array_spec(
        srcArray.shape[0],
        srcArray.shape[1],
        dtype=srcArray.dtype,
        profile=profile,
    )
    shards = spec.shards
    chunks = spec.chunks
    assert shards is not None and chunks is not None

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
