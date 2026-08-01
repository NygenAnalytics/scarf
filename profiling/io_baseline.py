"""R2 I/O baseline for a completed profiling store.

Streams the same access patterns as HVG, markers, and graph construction without
compute.

Spawn via the deployed app (survives local network drop):

  uv run --group profiling modal run --env scarf_profiling -m profiling.modal_app -- \\
    io-baseline --config profiling/config.toml --size 1000000
"""

import os
import sys
import time
from typing import Any

import numpy as np

from profiling.config import ProfilingConfig, StageResources
from profiling.metrics import ResourceSampler
from profiling.r2 import put_json, storage_options
from scarf import DataStore
from scarf.assay import _read_block
from scarf.storage.budget import ResourceBudget
from scarf.storage.feature_stream import FeatureStreamPlan, plan_feature_stream
from scarf.storage.types import as_zarr_array
from scarf.utils import iter_column_blocks

_LOG_EVERY_BLOCKS = 25


def _log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024**3):.2f} GiB"


def _fmt_rss(peakRssBytes: int | None) -> str:
    if peakRssBytes is None:
        return "?"
    return f"{peakRssBytes / (1024**3):.2f} GiB"


def _open_store(storeUri: str, resources: StageResources) -> DataStore:
    return DataStore(
        storeUri,
        nthreads=resources.workers,
        zarr_mode="r",
        zarrProfile="cloud" if storeUri.startswith("s3://") else None,
        storage_options=storage_options(storeUri),
        mem_budget=resources.scarfMemoryBudget,
    )


def _measure(label: str, fn: Any) -> dict[str, Any]:
    _log(f"[start] {label}")
    sampler = ResourceSampler(sampleIntervalSeconds=0.25)
    sampler.start()
    t0 = time.perf_counter()
    detail: dict[str, Any]
    try:
        detail = fn()
        status = "ok"
        error = None
    except Exception as exc:
        detail = {}
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - t0
    measurement = sampler.stop()
    result = {
        "pattern": label,
        "status": status,
        "seconds": round(seconds, 3),
        "peakRssBytes": measurement.processTreeRssPeakBytes,
        "peakCgroupBytes": measurement.cgroupMemoryCurrentPeakBytes
        or measurement.operationPeakBytes,
        "error": error,
        **detail,
    }
    bytes_read = detail.get("bytesRead")
    bytes_part = (
        f" bytes={_fmt_bytes(bytes_read)}" if isinstance(bytes_read, int) else ""
    )
    blocks = detail.get("nBlocks")
    blocks_part = f" blocks={blocks}" if blocks is not None else ""
    _log(
        f"[done] {label} status={status} wall={seconds:.1f}s "
        f"peakRss={_fmt_rss(result['peakRssBytes'])}{blocks_part}{bytes_part}"
        + (f" error={error}" if error else "")
    )
    return result


def _progress(
    label: str, blockIdx: int, nBlocks: int, bytesRead: int, t0: float
) -> None:
    if (
        blockIdx == 0
        or (blockIdx + 1) % _LOG_EVERY_BLOCKS != 0
        and (blockIdx + 1) != nBlocks
    ):
        return
    elapsed = time.perf_counter() - t0
    rate = bytesRead / elapsed if elapsed > 0 else 0.0
    _log(
        f"[progress] {label} {blockIdx + 1}/{nBlocks} "
        f"bytes={_fmt_bytes(bytesRead)} wall={elapsed:.1f}s "
        f"rate={rate / (1024**2):.1f} MiB/s"
    )


def _stream_hvg_tiles(
    assay: Any,
    cellIdx: np.ndarray,
    featIdx: np.ndarray,
) -> dict[str, Any]:
    """Physical chunk tiles, same layout walk as HVG feature stats."""
    use_counts_t = getattr(assay, "rawDataT", None) is not None
    if use_counts_t:
        zarr_arr = assay.rawDataT
        chunks = getattr(zarr_arr, "chunks", None)
        feat_chunk = int(chunks[0]) if chunks and len(chunks) > 0 else len(featIdx)
        cell_chunk = int(chunks[1]) if chunks and len(chunks) > 1 else len(cellIdx)
    else:
        zarr_arr = assay.rawData._backing
        chunks = getattr(zarr_arr, "chunks", None)
        cell_chunk = int(chunks[0]) if chunks and len(chunks) > 0 else len(cellIdx)
        feat_chunk = int(chunks[1]) if chunks and len(chunks) > 1 else len(featIdx)
    cell_chunk = max(1, cell_chunk)
    feat_chunk = max(1, feat_chunk)

    cell_bins = np.asarray(cellIdx // cell_chunk, dtype=np.intp)
    feat_bins = np.asarray(featIdx // feat_chunk, dtype=np.intp)
    tiles: list[tuple[np.ndarray, np.ndarray]] = []
    if use_counts_t:
        for feat_bin in np.unique(feat_bins):
            cols = featIdx[feat_bins == feat_bin]
            for cell_bin in np.unique(cell_bins):
                rows = cellIdx[cell_bins == cell_bin]
                tiles.append((rows, cols))
    else:
        for cell_bin in np.unique(cell_bins):
            rows = cellIdx[cell_bins == cell_bin]
            for feat_bin in np.unique(feat_bins):
                cols = featIdx[feat_bins == feat_bin]
                tiles.append((rows, cols))

    n_blocks = len(tiles)
    _log(
        f"[plan] hvgTiles cells={len(cellIdx)} feats={len(featIdx)} "
        f"chunks=({cell_chunk},{feat_chunk}) tiles={n_blocks} "
        f"source={'countsT' if use_counts_t else 'counts'}"
    )
    bytes_read = 0
    chunks_read = 0
    t0 = time.perf_counter()

    def read_block(block_idx: int) -> np.ndarray:
        rows, cols = tiles[block_idx]
        if use_counts_t:
            return _read_block(zarr_arr, cols, rows).T
        return _read_block(zarr_arr, rows, cols)

    for block_idx, raw, read_sec, source in iter_column_blocks(
        n_blocks,
        read_block,
        workers=assay.resources.workers,
    ):
        bytes_read += int(raw.nbytes)
        chunks_read += 1
        if (block_idx + 1) % _LOG_EVERY_BLOCKS == 0 or (block_idx + 1) == n_blocks:
            _log(
                f"[progress] hvgTiles {block_idx + 1}/{n_blocks} "
                f"read={read_sec:.1f}s src={source} "
                f"bytes={_fmt_bytes(bytes_read)} "
                f"wall={time.perf_counter() - t0:.1f}s"
            )
        del raw

    return {
        "nCells": int(len(cellIdx)),
        "nFeatures": int(len(featIdx)),
        "rowChunk": cell_chunk,
        "colChunk": feat_chunk,
        "nBlocks": chunks_read,
        "bytesRead": bytes_read,
        "arraySource": "countsT" if use_counts_t else "counts",
    }


def _stream_marker_batches(
    assay: Any,
    cellIdx: np.ndarray,
    plan: FeatureStreamPlan,
) -> dict[str, Any]:
    """All cells × gene batches, same path as marker search."""
    array_source = (
        "countsT" if getattr(assay, "rawDataT", None) is not None else "counts"
    )
    n_blocks = len(plan.blocks)
    _log(
        f"[plan] markerBatches cells={len(cellIdx)} "
        f"feats={sum(len(block.indices) for block in plan.blocks)} "
        f"batches={n_blocks} source={array_source}"
    )
    batches = [block.indices for block in plan.blocks]
    zarr_arr = (
        assay.rawDataT
        if getattr(assay, "rawDataT", None) is not None
        else assay.rawData._backing
    )

    def read_block(block_idx: int) -> np.ndarray:
        columns = batches[block_idx]
        if array_source == "countsT":
            return _read_block(zarr_arr, columns, cellIdx).T
        return _read_block(zarr_arr, cellIdx, columns)

    bytes_read = 0
    done = 0
    t0 = time.perf_counter()
    for block_idx, raw, read_sec, source in iter_column_blocks(
        n_blocks,
        read_block,
        workers=plan.readWorkers,
        io_concurrency=plan.ioConcurrency,
    ):
        feat_cols = batches[block_idx]
        bytes_read += int(raw.nbytes)
        done += 1
        if done % _LOG_EVERY_BLOCKS == 0 or done == n_blocks:
            _log(
                f"[progress] markerBatches {done}/{n_blocks} "
                f"width={len(feat_cols)} read={read_sec:.1f}s src={source} "
                f"bytes={_fmt_bytes(bytes_read)} "
                f"wall={time.perf_counter() - t0:.1f}s"
            )
        del raw
        del feat_cols
    return {
        "nCells": int(len(cellIdx)),
        "nFeatures": int(sum(len(batch) for batch in batches)),
        "geneBatchSize": int(max((len(batch) for batch in batches), default=0)),
        "nBlocks": done,
        "bytesRead": bytes_read,
        "arraySource": array_source,
    }


def _stream_graph_raw_cell_bands(
    assay: Any,
    cellIdx: np.ndarray,
    hvgIdx: np.ndarray,
) -> dict[str, Any]:
    """Selected cells by HVG columns in normalization row bands."""
    zarr_arr = assay.rawData._backing
    chunks = getattr(zarr_arr, "chunks", None)
    row_chunk = int(chunks[0]) if chunks and len(chunks) > 0 else len(cellIdx)
    row_chunk = max(1, row_chunk)
    n_blocks = int(np.ceil(len(cellIdx) / row_chunk))
    _log(
        f"[plan] graphRawCellBands cells={len(cellIdx)} hvgs={len(hvgIdx)} "
        f"rowChunk={row_chunk} bands={n_blocks}"
    )
    bytes_read = 0
    n_done = 0
    t0 = time.perf_counter()
    for start in range(0, len(cellIdx), row_chunk):
        end = min(start + row_chunk, len(cellIdx))
        raw = _read_block(zarr_arr, cellIdx[start:end], hvgIdx)
        bytes_read += int(raw.nbytes)
        n_done += 1
        _progress("graphRawCellBands", n_done - 1, n_blocks, bytes_read, t0)
        del raw
    return {
        "nCells": int(len(cellIdx)),
        "nFeatures": int(len(hvgIdx)),
        "rowChunk": row_chunk,
        "nBlocks": n_done,
        "bytesRead": bytes_read,
    }


def _stream_graph_normed_cell_bands(store: DataStore, assayName: str) -> dict[str, Any]:
    """Row chunks of the dense normalized-HVG matrix."""
    assay = store.get_assay(assayName)
    cell_key = assay.attrs.get("latest_cell_key", "I")
    feat_key = assay.attrs["latest_feat_key"]
    loc = f"{store.get_normalized_group_path(assayName, cell_key, feat_key)}/data"
    arr = as_zarr_array(store.zw[loc], name="data")
    n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
    chunks = getattr(arr, "chunks", None)
    row_chunk = int(chunks[0]) if chunks and len(chunks) > 0 else n_rows
    row_chunk = max(1, row_chunk)
    n_blocks = int(np.ceil(n_rows / row_chunk))
    _log(
        f"[plan] graphNormedCellBands path={loc} shape=({n_rows},{n_cols}) "
        f"rowChunk={row_chunk} bands={n_blocks} dtype={arr.dtype}"
    )
    bytes_read = 0
    n_done = 0
    t0 = time.perf_counter()
    for start in range(0, n_rows, row_chunk):
        end = min(start + row_chunk, n_rows)
        block = np.asarray(arr[start:end, :])
        bytes_read += int(block.nbytes)
        n_done += 1
        _progress("graphNormedCellBands", n_done - 1, n_blocks, bytes_read, t0)
        del block
    return {
        "arrayPath": loc,
        "nCells": n_rows,
        "nFeatures": n_cols,
        "rowChunk": row_chunk,
        "nBlocks": n_done,
        "bytesRead": bytes_read,
        "dtype": str(arr.dtype),
    }


def run_io_baseline_body(
    config: ProfilingConfig,
    *,
    nRows: int = 1_000_000,
    resultLabel: str | None = None,
    columnOnly: bool = False,
) -> dict[str, Any]:
    resources = config.resourcesFor("markHvgs")
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    store_uri = config.storeUri(nRows)
    _log(
        f"[job] runTag={config.runTag} store={store_uri} "
        f"machine={resources.modalCpuRequest}c/{resources.modalMemoryLimitMb}MiB "
        f"region={config.modalRegion}"
    )

    store_holder: dict[str, DataStore] = {}

    def _open() -> dict[str, Any]:
        store_holder["store"] = _open_store(store_uri, resources)
        return {"storeUri": store_uri}

    open_info = _measure("openStore", _open)
    if open_info["status"] != "ok":
        return {
            "runTag": config.runTag,
            "nRows": nRows,
            "storeUri": store_uri,
            "status": "error",
            "patterns": [open_info],
        }
    store = store_holder["store"]
    assay = store.get_assay(config.workflow.assayName)
    cell_idx = np.asarray(
        store.cells.active_index(config.workflow.cellKey), dtype=np.intp
    )
    feat_idx = np.asarray(
        assay.feats.active_index(config.workflow.markerFeatureKey), dtype=np.intp
    )
    # mark_hvgs stores the column as ``{cell_key}__{hvg_key}``.
    hvg_col = f"{config.workflow.cellKey}__{config.workflow.hvgKey}"
    hvg_idx = np.asarray(assay.feats.active_index(hvg_col), dtype=np.intp)
    if getattr(assay, "rawDataT", None) is not None:
        marker_source = assay.rawDataT
        feature_axis, cell_axis = 0, 1
    else:
        marker_source = assay.rawData._backing
        feature_axis, cell_axis = 1, 0
    marker_resources = ResourceBudget(
        memoryBytes=resources.scarfMemoryBudget,
        workers=resources.workers,
    )
    marker_plan = plan_feature_stream(
        marker_source,
        featureAxis=feature_axis,
        cellAxis=cell_axis,
        featureIndices=feat_idx,
        cellIndices=cell_idx,
        resources=marker_resources,
        blockBytes=lambda width: max(
            1,
            len(cell_idx) * width * max(1, int(np.dtype(marker_source.dtype).itemsize)),
        ),
        requestedBatchSize=config.workflow.markerGeneBatchSize,
    )
    marker_batch = max(
        (len(block.indices) for block in marker_plan.blocks),
        default=0,
    )
    raw_chunks = list(getattr(assay.rawData._backing, "chunks", ()) or ())
    _log(
        f"[setup] activeCells={len(cell_idx)} activeFeats={len(feat_idx)} "
        f"hvgs={len(hvg_idx)} markerBatch={marker_batch} rawChunks={raw_chunks}"
    )

    results = [open_info]
    results.append(
        _measure(
            "hvgTiles",
            lambda: _stream_hvg_tiles(assay, cell_idx, feat_idx),
        )
    )
    results.append(
        _measure(
            "markerBatches",
            lambda: _stream_marker_batches(
                assay,
                cell_idx,
                marker_plan,
            ),
        )
    )
    if not columnOnly:
        results.append(
            _measure(
                "graphRawCellBands",
                lambda: _stream_graph_raw_cell_bands(assay, cell_idx, hvg_idx),
            )
        )
        results.append(
            _measure(
                "graphNormedCellBands",
                lambda: _stream_graph_normed_cell_bands(
                    store, config.workflow.assayName
                ),
            )
        )

    total_seconds = sum(r["seconds"] for r in results if r["status"] == "ok")
    summary = {
        "runTag": config.runTag,
        "nRows": nRows,
        "storeUri": store_uri,
        "modalCpu": resources.modalCpuRequest,
        "modalMemoryMb": resources.modalMemoryLimitMb,
        "modalRegion": config.modalRegion,
        "columnOnly": columnOnly,
        "markerGeneBatchSize": marker_batch,
        "nActiveCells": int(len(cell_idx)),
        "nActiveFeatures": int(len(feat_idx)),
        "nHvgs": int(len(hvg_idx)),
        "rawChunks": raw_chunks,
        "totalSeconds": round(total_seconds, 3),
        "patterns": results,
    }
    _log(f"[job] finished totalSeconds={total_seconds:.1f}")
    suffix = f"-{resultLabel}" if resultLabel else ""
    result_uri = (
        f"{config.resultsUri.rstrip('/')}/io-baseline/{config.runTag}{suffix}.json"
    )
    try:
        put_json(result_uri, summary)
        _log(f"[job] wrote {result_uri}")
        summary["resultUri"] = result_uri
    except Exception as exc:
        _log(f"[job] failed to write result JSON: {type(exc).__name__}: {exc}")
    return summary
