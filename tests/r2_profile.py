#!/usr/bin/env python3
"""Profile Scarf RNA workflow on Zarr (local or R2).

Requires tests/.env with R2 credentials when using s3:// URIs.

Usage (from repo root):
    uv run python -m tests.r2_profile --uri s3://bucket/prefix/data.zarr
    uv run python -m tests.r2_profile --from-h5ad data.h5ad --uri s3://bucket/prefix/data.zarr
    uv run python -m tests.r2_profile --from-mtx cellranger_out/ --uri s3://bucket/prefix/data.zarr
    uv run python -m tests.r2_profile --from-cr-h5 filtered_feature_bc_matrix.h5 --uri s3://bucket/prefix/data.zarr
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from scarf.datastore.datastore import DataStore
from scarf.readers import CrDirReader, CrH5Reader, H5adReader
from scarf.utils import process_rss_mb, rss_peak_tracker, set_verbosity
from scarf.writers import CrToZarr, H5adToZarr

_ENV_PATH = Path(__file__).resolve().parent / ".env"
STEPS = ("create", "open", "filter", "hvg", "graph", "leiden", "umap", "markers")
WORKFLOW_STEPS = STEPS[1:]


@dataclass(frozen=True)
class MachineBudget:
    """Memory and worker overrides simulating one machine's resource budget.

    The write side drives on-disk chunk and shard geometry; the read side drives
    open-time streaming and concurrency. Keeping them separate lets a run write
    as if on one machine and read back as if on another.
    """

    memory: str | None = None
    workers: int = 4
    workingCopies: int | None = None


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


# (preferred, legacy) env var name pairs; the legacy names match
# tests/.env.example and tests/zarr_cloud_exp/r2.py.
_CREDENTIAL_KEYS = (
    ("SCARF_R2_ENDPOINT", "R2_ENDPOINT"),
    ("SCARF_R2_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID"),
    ("SCARF_R2_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY"),
)


def _env_value(preferred: str, legacy: str) -> str | None:
    return os.environ.get(preferred) or os.environ.get(legacy)


def storage_options(uri: str) -> dict[str, str] | None:
    if not uri.startswith("s3://"):
        return None
    load_env()
    missing = [
        preferred
        for preferred, legacy in _CREDENTIAL_KEYS
        if not _env_value(preferred, legacy)
    ]
    if missing:
        raise RuntimeError(
            f"Missing in tests/.env: {', '.join(missing)} "
            f"(or the corresponding R2_* legacy name)"
        )
    endpoint = _env_value("SCARF_R2_ENDPOINT", "R2_ENDPOINT")
    access_key = _env_value("SCARF_R2_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID")
    secret_key = _env_value("SCARF_R2_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY")
    assert endpoint and access_key and secret_key
    return {
        "access_key_id": access_key,
        "secret_access_key": secret_key,
        "endpoint": endpoint.rstrip("/"),
    }


def rss_mb() -> float:
    return process_rss_mb()


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
    t0 = time.perf_counter()
    result = StepResult(name=name)
    with rss_peak_tracker() as peak_rss:
        try:
            yield result
        except Exception as exc:
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _, traced = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rss_after = rss_mb()
            peak = max(rss_before, peak_rss(), rss_after)
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
        "marker_group_key": "RNA_leiden_cluster",
        "marker_feat_key": "I",
    }


def write_zarr_from_input(
    source: Path,
    kind: Literal["h5ad", "mtx", "cr_h5"],
    uri: str,
    budget: MachineBudget,
    *,
    h5adCellIdsKey: str,
    h5adFeatureIdsKey: str,
    h5adFeatureNameKey: str,
) -> None:
    opts = storage_options(uri)

    if kind == "cr_h5":
        reader = CrH5Reader(str(source))
        writer = CrToZarr(
            reader,
            uri,
            storage_options=opts,
            mem_budget=budget.memory,
            nthreads=budget.workers,
            working_copies=budget.workingCopies,
        )
        writer.dump()
        return

    if kind == "h5ad":
        reader = H5adReader(
            str(source),
            cell_ids_key=h5adCellIdsKey,
            feature_ids_key=h5adFeatureIdsKey,
            feature_name_key=h5adFeatureNameKey,
        )
        writer = H5adToZarr(
            reader,
            uri,
            storage_options=opts,
            mem_budget=budget.memory,
            nthreads=budget.workers,
            working_copies=budget.workingCopies,
        )
        try:
            writer.dump()
        finally:
            reader.h5.close()
        return

    reader = CrDirReader(str(source))
    writer = CrToZarr(
        reader,
        uri,
        storage_options=opts,
        mem_budget=budget.memory,
        nthreads=budget.workers,
        working_copies=budget.workingCopies,
    )
    writer.dump()


def run_workflow(
    uri: str,
    steps: list[str],
    writeBudget: MachineBudget,
    readBudget: MachineBudget,
    *,
    inputSource: Path | None,
    inputKind: Literal["h5ad", "mtx", "cr_h5"] | None,
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
                raise RuntimeError(
                    "create step requires --from-h5ad, --from-mtx, or --from-cr-h5"
                )

            def do_create() -> None:
                write_zarr_from_input(
                    inputSource,
                    inputKind,
                    uri,
                    writeBudget,
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
                    nthreads=readBudget.workers,
                    storage_options=opts,
                    zarrProfile="cloud" if uri.startswith("s3://") else None,
                    mem_budget=readBudget.memory,
                    working_copies=readBudget.workingCopies,
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
        elif step == "markers":

            def do_markers() -> None:
                assert ds is not None
                ds.run_marker_search(
                    group_key=str(gparams["marker_group_key"]),
                    feat_key=str(gparams["marker_feat_key"]),
                )

            with timed_step("markers") as r:
                do_markers()
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
    src.add_argument("--from-cr-h5", type=Path, metavar="PATH")
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
        help="Shared worker count; fallback for --write-workers/--read-workers",
    )
    parser.add_argument(
        "--mem-budget",
        default=None,
        help="Shared memory budget; fallback for --write-mem/--read-mem",
    )
    write = parser.add_argument_group(
        "write machine", "resource budget driving on-disk chunk/shard geometry"
    )
    write.add_argument("--write-mem", default=None, help="overrides --mem-budget")
    write.add_argument(
        "--write-workers", type=int, default=None, help="overrides --nthreads"
    )
    write.add_argument("--write-working-copies", type=int, default=None)
    read = parser.add_argument_group(
        "read machine", "resource budget driving open-time streaming/concurrency"
    )
    read.add_argument("--read-mem", default=None, help="overrides --mem-budget")
    read.add_argument(
        "--read-workers", type=int, default=None, help="overrides --nthreads"
    )
    read.add_argument("--read-working-copies", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None, help="Write results JSON")
    args = parser.parse_args()

    if args.from_h5ad and not args.from_h5ad.is_file():
        parser.error(f"--from-h5ad not found: {args.from_h5ad}")
    if args.from_mtx and not args.from_mtx.is_dir():
        parser.error(f"--from-mtx not found: {args.from_mtx}")
    if args.from_cr_h5 and not args.from_cr_h5.is_file():
        parser.error(f"--from-cr-h5 not found: {args.from_cr_h5}")

    if args.from_h5ad or args.from_mtx or args.from_cr_h5:
        if "create" not in args.steps:
            args.steps = ["create", *args.steps]
    elif "create" in args.steps:
        parser.error("create step requires --from-h5ad, --from-mtx, or --from-cr-h5")

    return args


def _budget(
    memory: str | None,
    workers: int | None,
    working_copies: int | None,
    *,
    shared_memory: str | None,
    shared_workers: int,
) -> MachineBudget:
    return MachineBudget(
        memory=memory if memory is not None else shared_memory,
        workers=workers if workers is not None else shared_workers,
        workingCopies=working_copies,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    set_verbosity("INFO")
    args = parse_args()
    input_source = args.from_h5ad or args.from_mtx or args.from_cr_h5
    input_kind: Literal["h5ad", "mtx", "cr_h5"] | None = None
    if args.from_h5ad:
        input_kind = "h5ad"
    elif args.from_mtx:
        input_kind = "mtx"
    elif args.from_cr_h5:
        input_kind = "cr_h5"
        input_source = args.from_cr_h5

    write_budget = _budget(
        args.write_mem,
        args.write_workers,
        args.write_working_copies,
        shared_memory=args.mem_budget,
        shared_workers=args.nthreads,
    )
    read_budget = _budget(
        args.read_mem,
        args.read_workers,
        args.read_working_copies,
        shared_memory=args.mem_budget,
        shared_workers=args.nthreads,
    )

    print(
        f"uri={args.uri}  write={write_budget}  read={read_budget}",
        flush=True,
    )
    if input_source:
        print(f"input={input_source} ({input_kind})", flush=True)

    started = datetime.now(UTC).isoformat()
    try:
        results, _ = run_workflow(
            args.uri,
            args.steps,
            write_budget,
            read_budget,
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
        "writeBudget": {
            "memory": write_budget.memory,
            "workers": write_budget.workers,
            "workingCopies": write_budget.workingCopies,
        },
        "readBudget": {
            "memory": read_budget.memory,
            "workers": read_budget.workers,
            "workingCopies": read_budget.workingCopies,
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
