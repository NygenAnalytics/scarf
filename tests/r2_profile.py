#!/usr/bin/env python3
"""Profile Scarf RNA workflow on Zarr (local or R2).

Requires tests/.env with R2 credentials when using s3:// URIs.

Usage (from repo root):
    uv run python -m tests.r2_profile --uri s3://bucket/prefix/data.zarr
    uv run python -m tests.r2_profile --from-h5ad data.h5ad --uri s3://bucket/prefix/data.zarr
    uv run python -m tests.r2_profile --from-h5ad-url https://datasets.cellxgene.cziscience.com/<id>.h5ad \\
        --auto-r2-uri cellxgene_<id>.zarr
    uv run python -m tests.r2_profile --from-mtx cellranger_out/ --uri s3://bucket/prefix/data.zarr
"""

import argparse
import gc
import json
import os
import resource
import sys
import time
import tracemalloc
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import h5py

from scarf.datastore.datastore import DataStore
from scarf.readers import CrDirReader, H5adReader
from scarf.storage.budget import resolve_budget, set_resource_budget
from scarf.storage.zarr_store import compute_zarr_layout, set_zarr_layout
from scarf.writers import CrToZarr, H5adToZarr

_ENV_PATH = Path(__file__).resolve().parent / ".env"
_DEFAULT_DOWNLOAD_DIR = Path(__file__).resolve().parent / "datasets" / "cellxgene"
_FEATURE_NAME_CANDIDATES = (
    "feature_name",
    "gene_symbols",
    "gene_short_name",
    "name",
    "_index",
    "index",
)
STEPS = ("create", "open", "filter", "hvg", "graph", "leiden", "umap")
WORKFLOW_STEPS = STEPS[1:]


def load_env() -> None:
    if not _ENV_PATH.is_file():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_r2_uri(zarr_name: str) -> str:
    bucket = os.environ.get("SCARF_R2_BUCKET", "")
    prefix = os.environ.get("SCARF_R2_PREFIX", "").strip("/")
    if not bucket:
        raise RuntimeError("SCARF_R2_BUCKET is not set")
    zarr_name = zarr_name.lstrip("/")
    if prefix:
        return f"s3://{bucket}/{prefix}/{zarr_name}"
    return f"s3://{bucket}/{zarr_name}"


def h5ad_url_basename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name.endswith(".h5ad"):
        raise ValueError(f"expected .h5ad URL, got: {url}")
    return name


def download_h5ad(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        print(f"using cached h5ad: {dest}", flush=True)
        return dest
    print(f"downloading {url}", flush=True)
    print(f"  -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def inspect_h5ad(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        var_keys = sorted(h5["var"].keys()) if "var" in h5 else []
        obs_keys = sorted(h5["obs"].keys()) if "obs" in h5 else []
        reader = H5adReader(
            str(path),
            feature_name_key=resolve_feature_name_key(path, "auto"),
        )
        info: dict[str, Any] = {
            "path": str(path),
            "nCells": reader.nCells,
            "nFeatures": reader.nFeatures,
            "matrixDtype": str(reader.matrixDtype),
            "obsKeys": obs_keys[:20],
            "varKeys": var_keys[:20],
            "featureNameKey": reader.featNamesKey,
            "featureIdsKey": reader.featIdsKey,
            "cellIdsKey": reader.cellIdsKey,
        }
        reader.h5.close()
        return info


def resolve_feature_name_key(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    with h5py.File(path, "r") as h5:
        if "var" not in h5:
            return "gene_short_name"
        keys = set(h5["var"].keys())
        for candidate in _FEATURE_NAME_CANDIDATES:
            alt = candidate.lstrip("_")
            if candidate in keys:
                return candidate
            if alt in keys:
                return alt
    return "_index"


def parse_local_cache(value: str) -> bool | str:
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return value


def storage_options(uri: str) -> dict[str, str] | None:
    if not uri.startswith("s3://"):
        return None
    load_env()
    missing = [
        k
        for k in (
            "SCARF_R2_ENDPOINT",
            "SCARF_R2_ACCESS_KEY_ID",
            "SCARF_R2_SECRET_ACCESS_KEY",
        )
        if not os.environ.get(k)
    ]
    if missing:
        raise RuntimeError(f"Missing in tests/.env: {', '.join(missing)}")
    return {
        "access_key_id": os.environ["SCARF_R2_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["SCARF_R2_SECRET_ACCESS_KEY"],
        "endpoint": os.environ["SCARF_R2_ENDPOINT"].rstrip("/"),
    }


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@dataclass
class StepResult:
    name: str
    seconds: float = 0.0
    ok: bool = True
    error: str | None = None
    memoryMb: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seconds": self.seconds,
            "ok": self.ok,
            "error": self.error,
            "memoryMb": self.memoryMb,
        }


@contextmanager
def timed_step(name: str):
    gc.collect()
    tracemalloc.start()
    rss_before = rss_mb()
    peak = rss_before
    t0 = time.perf_counter()
    result = StepResult(name=name)
    try:
        yield result
    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        peak = max(peak, rss_mb())
        _, traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = rss_mb()
        peak = max(peak, rss_after)
        result.seconds = round(time.perf_counter() - t0, 3)
        result.memoryMb = {
            "rssBeforeMb": round(rss_before, 1),
            "rssAfterMb": round(rss_after, 1),
            "rssPeakMb": round(peak, 1),
            "tracedPeakMb": round(traced / (1024 * 1024), 1),
        }
        mem = result.memoryMb
        status = "ok" if result.ok else "FAIL"
        print(
            f"{name:8} {result.seconds:7.1f}s  "
            f"rss {mem['rssBeforeMb']:.0f}->{mem['rssAfterMb']:.0f}MiB  "
            f"peak {mem['rssPeakMb']:.0f}MiB  [{status}]",
            flush=True,
        )


def graph_params(
    nCells: int, nFeatures: int, *, localCache: bool | str
) -> dict[str, int | str]:
    topN = min(2000, max(500, nFeatures // 25))
    return {
        "top_n": topN,
        "k": 11,
        "dims": min(50, max(11, topN)),
        "n_centroids": max(100, nCells // 100),
        "local_cache": localCache,
    }


def write_zarr_from_input(
    source: Path,
    kind: Literal["h5ad", "mtx"],
    uri: str,
    nthreads: int,
    mem_budget: str | None,
    *,
    h5adCellIdsKey: str,
    h5adFeatureIdsKey: str,
    h5adFeatureNameKey: str,
    dumpBatchSize: int,
) -> None:
    opts = storage_options(uri)
    remote = uri.startswith("s3://")
    set_resource_budget(resolve_budget(memory=mem_budget, workers=nthreads))

    if kind == "h5ad":
        feature_name_key = resolve_feature_name_key(source, h5adFeatureNameKey)
        reader = H5adReader(
            str(source),
            cell_ids_key=h5adCellIdsKey,
            feature_ids_key=h5adFeatureIdsKey,
            feature_name_key=feature_name_key,
        )
        n_cells, n_features = reader.nCells, reader.nFeatures
        layout = compute_zarr_layout(n_cells, n_features, remote=remote)
        set_zarr_layout(layout)
        writer = H5adToZarr(
            reader,
            uri,
            chunk_size=layout.countChunks,
            storage_options=opts,
        )
        try:
            writer.dump(batch_size=dumpBatchSize)
        finally:
            reader.h5.close()
        return

    reader = CrDirReader(str(source))
    n_cells, n_features = reader.nCells, reader.nFeatures
    layout = compute_zarr_layout(n_cells, n_features, remote=remote)
    set_zarr_layout(layout)
    writer = CrToZarr(
        reader,
        uri,
        chunk_size=layout.countChunks,
        storage_options=opts,
    )
    writer.dump()


def run_workflow(
    uri: str,
    steps: list[str],
    nthreads: int,
    mem_budget: str | None,
    *,
    inputSource: Path | None,
    inputKind: Literal["h5ad", "mtx"] | None,
    h5adCellIdsKey: str,
    h5adFeatureIdsKey: str,
    h5adFeatureNameKey: str,
    dumpBatchSize: int,
    localCache: bool | str,
) -> tuple[list[dict[str, Any]], DataStore | None]:
    opts = storage_options(uri)
    ds: DataStore | None = None
    results: list[dict[str, Any]] = []
    gparams: dict[str, int | str] = {}

    for step in steps:
        if step == "create":
            if inputSource is None or inputKind is None:
                raise RuntimeError("create step requires --from-h5ad or --from-mtx")

            def do_create() -> None:
                write_zarr_from_input(
                    inputSource,
                    inputKind,
                    uri,
                    nthreads,
                    mem_budget,
                    h5adCellIdsKey=h5adCellIdsKey,
                    h5adFeatureIdsKey=h5adFeatureIdsKey,
                    h5adFeatureNameKey=h5adFeatureNameKey,
                    dumpBatchSize=dumpBatchSize,
                )

            with timed_step("create") as r:
                do_create()
            results.append(r.to_dict())
            continue

        if step == "open":

            def do_open() -> None:
                nonlocal ds, gparams
                ds = DataStore(
                    uri,
                    default_assay="RNA",
                    assay_types={"RNA": "RNA"},
                    nthreads=nthreads,
                    storage_options=opts,
                    zarrProfile="cloud" if uri.startswith("s3://") else None,
                    mem_budget=mem_budget,
                )
                layout = compute_zarr_layout(
                    ds.cells.N, ds.RNA.feats.N, remote=uri.startswith("s3://")
                )
                set_zarr_layout(layout)
                gparams = graph_params(
                    ds.cells.N, ds.RNA.feats.N, localCache=localCache
                )

            with timed_step("open") as r:
                do_open()
            results.append(r.to_dict())
            continue

        if ds is None:
            raise RuntimeError("open must run before other steps")

        if step == "filter":

            def do_filter() -> None:
                assert ds is not None
                ds.auto_filter_cells(show_qc_plots=False)

            with timed_step("filter") as r:
                do_filter()
        elif step == "hvg":

            def do_hvg() -> None:
                assert ds is not None
                ds.mark_hvgs(top_n=int(gparams["top_n"]), min_cells=20, show_plot=False)

            with timed_step("hvg") as r:
                do_hvg()
        elif step == "graph":

            def do_graph() -> None:
                assert ds is not None
                ds.make_graph(
                    feat_key="hvgs",
                    k=int(gparams["k"]),
                    dims=int(gparams["dims"]),
                    n_centroids=int(gparams["n_centroids"]),
                    local_cache=localCache,
                )

            with timed_step("graph") as r:
                do_graph()
        elif step == "leiden":

            def do_leiden() -> None:
                assert ds is not None
                ds.run_leiden_clustering(resolution=1.0)

            with timed_step("leiden") as r:
                do_leiden()
        elif step == "umap":

            def do_umap() -> None:
                assert ds is not None
                ds.run_umap(n_epochs=200, parallel=True)

            with timed_step("umap") as r:
                do_umap()
        else:
            raise ValueError(f"unknown step: {step}")

        results.append(r.to_dict())

    return results, ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uri",
        default=None,
        help="Zarr store path (local dir or s3://bucket/prefix/data.zarr)",
    )
    parser.add_argument(
        "--auto-r2-uri",
        metavar="ZARR_NAME",
        default=None,
        help="Build s3:// URI from SCARF_R2_BUCKET and SCARF_R2_PREFIX",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--from-h5ad", type=Path, metavar="PATH")
    src.add_argument(
        "--from-h5ad-url",
        metavar="URL",
        help="Download an h5ad file before profiling",
    )
    src.add_argument("--from-mtx", type=Path, metavar="DIR")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=_DEFAULT_DOWNLOAD_DIR,
        help="Cache directory for --from-h5ad-url downloads",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if the h5ad file is cached",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=list(WORKFLOW_STEPS),
    )
    parser.add_argument(
        "--h5ad-cell-ids-key", default="_index", help="obs column for cell barcodes"
    )
    parser.add_argument(
        "--h5ad-feature-ids-key", default="_index", help="var column for gene ids"
    )
    parser.add_argument(
        "--h5ad-feature-name-key",
        default="auto",
        help="var column for gene names (auto detects cellxgene-style keys)",
    )
    parser.add_argument(
        "--dump-batch-size",
        type=int,
        default=1000,
        help="Cells per batch when writing h5ad/mtx to Zarr",
    )
    parser.add_argument(
        "--local-cache",
        type=parse_local_cache,
        default="auto",
        help="local_cache for make_graph: auto, true, false, or a directory path",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=int(os.environ.get("SCARF_BENCHMARK_NTHREADS", "4")),
    )
    parser.add_argument("--mem-budget", default=None)
    parser.add_argument("--json", type=Path, default=None, help="Write results JSON")
    args = parser.parse_args()

    if args.auto_r2_uri:
        if args.uri:
            parser.error("use either --uri or --auto-r2-uri, not both")
        args.uri = resolve_r2_uri(args.auto_r2_uri)
    if not args.uri:
        parser.error("--uri or --auto-r2-uri is required")

    if args.from_h5ad_url:
        dest = args.download_dir / h5ad_url_basename(args.from_h5ad_url)
        args.from_h5ad = download_h5ad(
            args.from_h5ad_url, dest, force=args.force_download
        )

    if args.from_h5ad and not args.from_h5ad.is_file():
        parser.error(f"--from-h5ad not found: {args.from_h5ad}")
    if args.from_mtx and not args.from_mtx.is_dir():
        parser.error(f"--from-mtx not found: {args.from_mtx}")

    if args.from_h5ad or args.from_mtx:
        if "create" not in args.steps:
            args.steps = ["create", *args.steps]
    elif "create" in args.steps:
        parser.error("create step requires --from-h5ad or --from-mtx")

    return args


def main() -> int:
    args = parse_args()
    input_source = args.from_h5ad or args.from_mtx
    input_kind: Literal["h5ad", "mtx"] | None = None
    if args.from_h5ad:
        input_kind = "h5ad"
    elif args.from_mtx:
        input_kind = "mtx"

    dataset_info: dict[str, Any] | None = None
    if input_kind == "h5ad" and input_source is not None:
        dataset_info = inspect_h5ad(input_source)
        print(
            f"dataset: {dataset_info['nCells']} cells x {dataset_info['nFeatures']} features  "
            f"feature_name_key={dataset_info['featureNameKey']}",
            flush=True,
        )

    print(
        f"uri={args.uri}  nthreads={args.nthreads}  "
        f"dump_batch_size={args.dump_batch_size}  local_cache={args.local_cache!r}",
        flush=True,
    )
    if input_source:
        print(f"input={input_source} ({input_kind})", flush=True)

    started = datetime.now(UTC).isoformat()
    try:
        results, _ = run_workflow(
            args.uri,
            args.steps,
            args.nthreads,
            args.mem_budget,
            inputSource=input_source,
            inputKind=input_kind,
            h5adCellIdsKey=args.h5ad_cell_ids_key,
            h5adFeatureIdsKey=args.h5ad_feature_ids_key,
            h5adFeatureNameKey=args.h5ad_feature_name_key,
            dumpBatchSize=args.dump_batch_size,
            localCache=args.local_cache,
        )
    except Exception as exc:
        print(f"aborted: {exc}", file=sys.stderr, flush=True)
        return 1

    payload = {
        "uri": args.uri,
        "input": str(input_source) if input_source else None,
        "inputUrl": args.from_h5ad_url,
        "dataset": dataset_info,
        "tuning": {
            "nthreads": args.nthreads,
            "memBudget": args.mem_budget,
            "dumpBatchSize": args.dump_batch_size,
            "localCache": args.local_cache,
        },
        "started": started,
        "finished": datetime.now(UTC).isoformat(),
        "steps": results,
    }
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}", flush=True)

    failed = sum(1 for s in results if not s["ok"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
