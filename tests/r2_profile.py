#!/usr/bin/env python3
"""Profile Scarf RNA workflow on Zarr (local or R2).

Requires tests/.env with R2 credentials when using s3:// URIs.

Usage (from repo root):
    uv run python -m tests.r2_profile --uri s3://bucket/prefix/data.zarr
    uv run python -m tests.r2_profile --from-h5ad data.h5ad --uri s3://bucket/prefix/data.zarr
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
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from scarf.datastore.datastore import DataStore
from scarf.readers import CrDirReader, H5adReader
from scarf.storage.budget import resolve_budget, set_resource_budget
from scarf.writers import CrToZarr, H5adToZarr

_ENV_PATH = Path(__file__).resolve().parent / ".env"
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


def graph_params(nCells: int, nFeatures: int) -> dict[str, int | str]:
    topN = min(2000, max(500, nFeatures // 25))
    return {
        "top_n": topN,
        "k": 11,
        "dims": min(50, max(11, topN)),
        "n_centroids": max(100, nCells // 100),
        "local_cache": "auto",
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
) -> None:
    opts = storage_options(uri)
    set_resource_budget(resolve_budget(memory=mem_budget, workers=nthreads))

    if kind == "h5ad":
        reader = H5adReader(
            str(source),
            cell_ids_key=h5adCellIdsKey,
            feature_ids_key=h5adFeatureIdsKey,
            feature_name_key=h5adFeatureNameKey,
        )
        writer = H5adToZarr(reader, uri, storage_options=opts)
        try:
            writer.dump()
        finally:
            reader.h5.close()
        return

    reader = CrDirReader(str(source))
    writer = CrToZarr(reader, uri, storage_options=opts)
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
                gparams = graph_params(ds.cells.N, ds.RNA.feats.N)

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
                    local_cache=str(gparams["local_cache"]),
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
        required=True,
        help="Zarr store path (local dir or s3://bucket/prefix/data.zarr)",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--from-h5ad", type=Path, metavar="PATH")
    src.add_argument("--from-mtx", type=Path, metavar="DIR")
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
        default="gene_short_name",
        help="var column for gene names",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=int(os.environ.get("SCARF_BENCHMARK_NTHREADS", "4")),
    )
    parser.add_argument("--mem-budget", default=None)
    parser.add_argument("--json", type=Path, default=None, help="Write results JSON")
    args = parser.parse_args()

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

    print(f"uri={args.uri}  nthreads={args.nthreads}", flush=True)
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
        )
    except Exception as exc:
        print(f"aborted: {exc}", file=sys.stderr, flush=True)
        return 1

    payload = {
        "uri": args.uri,
        "input": str(input_source) if input_source else None,
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
