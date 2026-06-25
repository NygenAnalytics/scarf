"""Execute all myst-nb vignettes in parallel and merge into docs/.jupyter_cache."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent
PARALLEL_CACHE = DOCS_ROOT / ".jupyter_cache" / "_parallel"


def _worker(name: str) -> Path:
    repo_root = DOCS_ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from docs.execute_vignette import execute_vignette

    cache_path = PARALLEL_CACHE / name
    if cache_path.exists():
        shutil.rmtree(cache_path)
    execute_vignette(
        name,
        cache_path=cache_path,
        execution_in_temp=True,
    )
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of parallel workers (default: min(4, cpu_count))",
    )
    parser.add_argument(
        "vignettes",
        nargs="*",
        help="Vignette names without .md extension (default: all)",
    )
    args = parser.parse_args()

    repo_root = DOCS_ROOT.parent
    sys.path.insert(0, str(repo_root))
    from docs.execute_vignette import DEFAULT_CACHE, list_vignettes, merge_caches

    names = args.vignettes or list_vignettes()
    if not names:
        raise SystemExit("No vignettes found")

    print(f"Executing {len(names)} vignettes with {args.jobs} workers", flush=True)
    PARALLEL_CACHE.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    completed_caches: list[Path] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_worker, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                completed_caches.append(future.result())
            except Exception as exc:
                failures.append((name, str(exc)))
                print(f"[{name}] FAILED: {exc}", flush=True)

    if completed_caches:
        print(
            f"Merging {len(completed_caches)} caches into docs/.jupyter_cache",
            flush=True,
        )
        merge_caches(completed_caches, DEFAULT_CACHE)

    if PARALLEL_CACHE.exists():
        shutil.rmtree(PARALLEL_CACHE, ignore_errors=True)

    if failures:
        print("\nFailed vignettes:", flush=True)
        for name, msg in failures:
            print(f"  - {name}: {msg}", flush=True)
        raise SystemExit(1)

    print(
        f"All {len(names)} vignettes executed and merged into docs/.jupyter_cache",
        flush=True,
    )


if __name__ == "__main__":
    main()
