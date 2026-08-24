"""Execute, validate, prune, and publish the documentation notebook cache."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jupyter_cache import get_cache  # noqa: E402

from docs.execute_vignette import (  # noqa: E402
    CacheBuildError,
    CacheToolError,
    DEFAULT_CACHE,
    ParsedSource,
    ValidationReport,
    _publish_candidate_locked,
    _remove_path,
    build_candidate,
    close_cache,
    configure_doc_execution_env,
    discover_sources,
    execution_fingerprint,
    recover_interrupted_swap,
    resolve_doc_source,
    serialization_lock,
    transfer_source_bundle,
    validate_cache,
)

type PageRunner = Callable[[ParsedSource, Path], Path]
type PageRunnerFactory = Callable[[list[ParsedSource]], PageRunner]

EXECUTE_SCRIPT = DOCS_ROOT / "execute_vignette.py"
MANIFEST_VERSION = 2
LOCAL_RUNNER_IDENTITY = "local"


class ExecutionBatchError(CacheToolError):
    pass


def _load_manifest(resume_dir: Path) -> dict[str, object] | None:
    path = resume_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
        return None
    return data


def _save_manifest(resume_dir: Path, manifest: dict[str, object]) -> None:
    resume_dir.mkdir(parents=True, exist_ok=True)
    path = resume_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_resume(
    resume_dir: Path,
    fingerprint: str,
    requested_uris: set[str],
    *,
    use_resume: bool,
    runner_identity: str,
) -> dict[str, object]:
    existing = _load_manifest(resume_dir) if use_resume else None
    if (
        existing is None
        or existing.get("executionFingerprint") != fingerprint
        or existing.get("runnerIdentity") != runner_identity
    ):
        _remove_path(resume_dir)
        manifest: dict[str, object] = {
            "version": MANIFEST_VERSION,
            "executionFingerprint": fingerprint,
            "runnerIdentity": runner_identity,
            "entries": {},
        }
    else:
        manifest = existing
        if not isinstance(manifest.get("entries"), dict):
            manifest["entries"] = {}
    manifest["requestedUris"] = sorted(requested_uris)
    _save_manifest(resume_dir, manifest)
    return manifest


def _valid_resume_uris(
    sources: list[ParsedSource],
    requested_uris: set[str],
    resume_dir: Path,
    manifest: dict[str, object],
) -> set[str]:
    entries = manifest.get("entries")
    result_cache_path = resume_dir / "cache"
    if not isinstance(entries, dict) or not result_cache_path.is_dir():
        return set()

    result_cache = get_cache(result_cache_path)
    valid: set[str] = set()
    try:
        for source in sources:
            if source.uri not in requested_uris:
                continue
            if entries.get(source.uri) != source.hashkey:
                continue
            try:
                result_cache.match_cache_notebook(source.notebook)
            except KeyError:
                continue
            valid.add(source.uri)
    finally:
        close_cache(result_cache)
    return valid


def _record_result(
    source: ParsedSource,
    page_cache_path: Path,
    resume_dir: Path,
    manifest: dict[str, object],
) -> None:
    page_cache = get_cache(page_cache_path)
    result_cache = get_cache(resume_dir / "cache")
    try:
        transfer_source_bundle(page_cache, result_cache, source)
    finally:
        close_cache(page_cache)
        close_cache(result_cache)

    entries = manifest.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise CacheBuildError("Resume manifest entries are invalid")
    entries[source.uri] = source.hashkey
    _save_manifest(resume_dir, manifest)


def _page_cache_path(work_root: Path, source: ParsedSource) -> Path:
    digest = hashlib.sha256(source.uri.encode()).hexdigest()[:16]
    return work_root / f"page-{digest}"


def _run_page_subprocess(
    source: ParsedSource,
    cache_path: Path,
    *,
    docs_root: Path = DOCS_ROOT,
) -> Path:
    env = configure_doc_execution_env(os.environ.copy())
    script = docs_root / "execute_vignette.py"
    command = [
        sys.executable,
        str(script),
        source.uri,
        "--cache-path",
        str(cache_path),
    ]
    result = subprocess.run(command, cwd=docs_root.parent, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return cache_path


def _snapshot_cache(target_path: Path, snapshot_path: Path) -> Path | None:
    if not target_path.exists():
        return None
    shutil.copytree(target_path, snapshot_path)
    return snapshot_path


def _same_sources(
    before: list[ParsedSource],
    after: list[ParsedSource],
) -> bool:
    return {source.uri: source.hashkey for source in before} == {
        source.uri: source.hashkey for source in after
    }


def _resolve_requested(
    sources: list[ParsedSource],
    pages: list[str],
    *,
    source_dir: Path,
) -> list[ParsedSource]:
    resolved: list[ParsedSource] = []
    seen: set[str] = set()
    for page in pages:
        source = resolve_doc_source(page, sources, source_dir)
        if source.uri not in seen:
            resolved.append(source)
            seen.add(source.uri)
    return resolved


def _resume_scope(resume_dir: Path) -> list[str]:
    manifest = _load_manifest(resume_dir)
    if manifest is None:
        return []
    requested = manifest.get("requestedUris")
    if not isinstance(requested, list) or not all(
        isinstance(uri, str) for uri in requested
    ):
        return []
    return requested


def execute_and_publish(
    pages: list[str] | None = None,
    *,
    jobs: int = 1,
    full: bool = False,
    resume: bool = False,
    docs_root: Path = DOCS_ROOT,
    page_runner: PageRunner | None = None,
    page_runner_factory: PageRunnerFactory | None = None,
    warn_parallel_memory: bool = True,
    runner_identity: str = LOCAL_RUNNER_IDENTITY,
) -> ValidationReport:
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if not runner_identity.strip():
        raise ValueError("runner_identity must not be empty")
    if page_runner is not None and page_runner_factory is not None:
        raise ValueError("page_runner and page_runner_factory are mutually exclusive")
    docs_root = docs_root.resolve()
    source_dir = docs_root / "source"
    target_path = docs_root / ".jupyter_cache"
    resume_dir = docs_root / ".jupyter_cache.resume"
    pages = list(pages or [])

    with serialization_lock(target_path):
        recover_interrupted_swap(target_path)
        sources = discover_sources(source_dir, docs_root)
        if not sources:
            raise CacheBuildError("No executable documentation pages found")
        if full and pages:
            raise ValueError("--full cannot be combined with page names")
        if resume and not full and not pages:
            pages = _resume_scope(resume_dir)
            if not pages:
                raise CacheBuildError("No resumable documentation run was found")
        full_run = full or not pages
        requested = (
            list(sources)
            if full_run
            else _resolve_requested(sources, pages, source_dir=source_dir)
        )
        requested_uris = {source.uri for source in requested}

        run_id = uuid.uuid4().hex
        work_root = docs_root / f".jupyter_cache.work-{run_id}"
        candidate_path = docs_root / f".jupyter_cache.candidate-{run_id}"
        snapshot_path = work_root / "snapshot"
        work_root.mkdir()
        try:
            snapshot = None if full_run else _snapshot_cache(target_path, snapshot_path)
            fingerprint = execution_fingerprint(docs_root.parent, docs_root)
            manifest = _prepare_resume(
                resume_dir,
                fingerprint,
                requested_uris,
                use_resume=resume,
                runner_identity=runner_identity,
            )
            resumed = _valid_resume_uris(
                sources,
                requested_uris,
                resume_dir,
                manifest,
            )
            pending = [source for source in requested if source.uri not in resumed]
            print(
                f"Executing {len(pending)} page(s), reusing {len(resumed)} staged result(s)",
                flush=True,
            )
            if jobs > 1 and warn_parallel_memory:
                print(
                    "WARNING: Each page can use several GB. Prefer one worker on WSL.",
                    flush=True,
                )

            failures: list[tuple[str, str]] = []

            def default_runner(source: ParsedSource, path: Path) -> Path:
                return _run_page_subprocess(
                    source,
                    path,
                    docs_root=docs_root,
                )

            runner = page_runner or default_runner
            if page_runner_factory is not None:
                runner = page_runner_factory(pending)
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {
                    pool.submit(
                        runner, source, _page_cache_path(work_root, source)
                    ): source
                    for source in pending
                }
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        page_cache_path = Path(future.result())
                        _record_result(
                            source,
                            page_cache_path,
                            resume_dir,
                            manifest,
                        )
                    except Exception as exc:
                        failures.append((source.uri, str(exc)))
                        print(f"[{source.uri}] FAILED: {exc}", flush=True)

            if failures:
                details = "\n".join(f"  {uri}: {message}" for uri, message in failures)
                raise ExecutionBatchError(f"Page execution failed:\n{details}")

            current_sources = discover_sources(source_dir, docs_root)
            if not _same_sources(sources, current_sources):
                raise CacheBuildError("Documentation sources changed during execution")
            if execution_fingerprint(docs_root.parent, docs_root) != fingerprint:
                raise CacheBuildError("Execution inputs changed during execution")

            build_candidate(
                current_sources,
                requested_uris,
                resume_dir / "cache",
                snapshot,
                candidate_path,
            )
            report = validate_cache(
                candidate_path,
                source_dir=source_dir,
                docs_root=docs_root,
            )
            _publish_candidate_locked(candidate_path, target_path)
            _remove_path(resume_dir)
            print(
                f"Published {report.record_count} cache record(s) for "
                f"{report.source_count} source page(s)",
                flush=True,
            )
            return report
        finally:
            _remove_path(candidate_path)
            _remove_path(work_root)


def prune_and_publish(docs_root: Path = DOCS_ROOT) -> ValidationReport:
    docs_root = docs_root.resolve()
    source_dir = docs_root / "source"
    target_path = docs_root / ".jupyter_cache"
    with serialization_lock(target_path):
        recover_interrupted_swap(target_path)
        sources = discover_sources(source_dir, docs_root)
        run_id = uuid.uuid4().hex
        work_root = docs_root / f".jupyter_cache.work-{run_id}"
        candidate_path = docs_root / f".jupyter_cache.candidate-{run_id}"
        work_root.mkdir()
        try:
            snapshot = _snapshot_cache(target_path, work_root / "snapshot")
            build_candidate(
                sources,
                set(),
                work_root / "unused-results",
                snapshot,
                candidate_path,
            )
            report = validate_cache(
                candidate_path,
                source_dir=source_dir,
                docs_root=docs_root,
            )
            _publish_candidate_locked(candidate_path, target_path)
            return report
        finally:
            _remove_path(candidate_path)
            _remove_path(work_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pages",
        nargs="*",
        help="Pages for a partial run; no pages means a full run",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of isolated page processes",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force every executable page",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse staged results with matching source and execution fingerprints",
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--validate",
        action="store_true",
        help="Validate the committed cache without changing it",
    )
    operation.add_argument(
        "--prune",
        action="store_true",
        help="Publish a cache containing only matching current sources",
    )
    args = parser.parse_args()

    if (args.validate or args.prune) and (
        args.pages or args.full or args.resume or args.jobs != 1
    ):
        parser.error("validation and pruning do not accept execution options")

    try:
        if args.validate:
            with serialization_lock(DEFAULT_CACHE):
                recover_interrupted_swap(DEFAULT_CACHE)
                report = validate_cache()
            print(
                f"Validated {report.record_count} cache record(s) for "
                f"{report.source_count} source page(s)",
                flush=True,
            )
        elif args.prune:
            report = prune_and_publish()
            print(
                f"Published pruned cache with {report.record_count} record(s)",
                flush=True,
            )
        else:
            execute_and_publish(
                args.pages,
                jobs=args.jobs,
                full=args.full,
                resume=args.resume,
            )
    except (CacheToolError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
