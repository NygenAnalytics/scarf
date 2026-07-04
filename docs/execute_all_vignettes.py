"""Execute all myst-nb vignettes in parallel and merge into docs/.jupyter_cache."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from docs.execute_vignette import configure_doc_execution_env

DOCS_ROOT = Path(__file__).resolve().parent
PARALLEL_CACHE = DOCS_ROOT / ".jupyter_cache" / "_parallel"
EXECUTE_SCRIPT = DOCS_ROOT / "execute_vignette.py"


def _run_vignette_subprocess(name: str) -> Path:
    """Run one vignette in a fresh Python process so memory is fully released."""
    cache_path = PARALLEL_CACHE / name
    if cache_path.exists():
        shutil.rmtree(cache_path)

    env = os.environ.copy()
    configure_doc_execution_env()
    env.update(
        {
            "SCARF_MEM_BUDGET": os.environ["SCARF_MEM_BUDGET"],
            "SCARF_WORKING_COPIES": os.environ["SCARF_WORKING_COPIES"],
            "SCARF_WORKERS": os.environ["SCARF_WORKERS"],
            "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
            "NUMEXPR_NUM_THREADS": os.environ["NUMEXPR_NUM_THREADS"],
        }
    )
    cmd = [
        sys.executable,
        str(EXECUTE_SCRIPT),
        name,
        "--cache-path",
        str(cache_path),
        "--force",
    ]
    result = subprocess.run(
        cmd,
        cwd=DOCS_ROOT.parent,
        env=env,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of vignettes to run concurrently (default: 1). "
            "Each vignette runs in its own Python process."
        ),
    )
    parser.add_argument(
        "vignettes",
        nargs="*",
        help="Vignette names without .md extension (default: all)",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.jobs > 1:
        print(
            "WARNING: Each vignette can use several GB. Prefer -j 1 on WSL.",
            flush=True,
        )

    repo_root = DOCS_ROOT.parent
    sys.path.insert(0, str(repo_root))
    from docs.execute_vignette import (
        DEFAULT_CACHE,
        list_executable_docs,
        merge_caches,
        prune_stale_cache,
    )

    names = args.vignettes or list_executable_docs()
    if not names:
        raise SystemExit("No vignettes found")

    print(f"Executing {len(names)} vignettes with {args.jobs} workers", flush=True)
    PARALLEL_CACHE.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    completed_caches: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_vignette_subprocess, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                completed_caches.append(future.result())
            except subprocess.CalledProcessError as exc:
                failures.append((name, f"exit code {exc.returncode}"))
                print(f"[{name}] FAILED: exit code {exc.returncode}", flush=True)
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

    pruned = prune_stale_cache(DEFAULT_CACHE)
    if pruned:
        print(f"Pruned {len(pruned)} stale cache entries", flush=True)

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
