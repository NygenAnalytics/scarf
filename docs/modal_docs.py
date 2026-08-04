"""Execute documentation pages concurrently on Modal."""

import os
from pathlib import Path
from typing import Any

import modal

from docs.execute_all_vignettes import execute_and_publish
from docs.execute_vignette import ParsedSource, discover_sources, execute_page
from docs.modal_cache import PageCachePayload, SpawnedPageRunner, pack_page_cache

DOCS_ROOT = Path(__file__).resolve().parent
REMOTE_CACHE = Path("/tmp/scarf-doc-page-cache")
MODAL_ENVIRONMENT_NAME = "scarf_profiling"
MODAL_PYTHON_VERSION = "3.14"
MODAL_RUNNER_IDENTITY = f"modal-python-{MODAL_PYTHON_VERSION}"
MODAL_TIMEOUT_SECONDS = 7_200
MODAL_CALL_GRACE_SECONDS = 300
MODAL_EPHEMERAL_DISK_MB = 524_288


def _ignore_docs_source(path: Path) -> bool:
    return bool({"scarf_datasets", "_build", "__pycache__"} & set(path.parts))


app = modal.App("scarf-docs")

image = (
    modal.Image.debian_slim(python_version=MODAL_PYTHON_VERSION)
    .apt_install(
        "build-essential",
        "git",
        "libfftw3-dev",
        "libmetis-dev",
    )
    .uv_sync(
        groups=["docs-modal"],
        extras=["docs", "extra"],
        frozen=True,
        extra_options="--no-default-groups",
        env={"HNSWLIB_NO_NATIVE": "1"},
    )
    .env({"HNSWLIB_NO_NATIVE": "1"})
    .add_local_python_source("scarf", copy=True)
    .add_local_dir(
        str(DOCS_ROOT / "source"),
        "/root/docs/source",
        copy=True,
        ignore=_ignore_docs_source,
    )
    .add_local_file(
        str(DOCS_ROOT / "execute_vignette.py"),
        "/root/docs/execute_vignette.py",
        copy=True,
    )
    .add_local_file(
        str(DOCS_ROOT / "execute_all_vignettes.py"),
        "/root/docs/execute_all_vignettes.py",
        copy=True,
    )
    .add_local_file(
        str(DOCS_ROOT / "modal_cache.py"),
        "/root/docs/modal_cache.py",
        copy=True,
    )
    .add_local_file(
        str(DOCS_ROOT / "modal_docs.py"),
        "/root/docs/modal_docs.py",
        copy=True,
    )
)


@app.function(
    image=image,
    cpu=2.0,
    memory=8_192,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MB,
    timeout=MODAL_TIMEOUT_SECONDS,
    retries=0,
    single_use_containers=True,
    include_source=False,
)
def execute_doc_page(uri: str) -> PageCachePayload:
    result = execute_page(uri, cache_path=REMOTE_CACHE)
    return (
        str(result["uri"]),
        str(result["hashkey"]),
        pack_page_cache(REMOTE_CACHE),
    )


def _spawn_page(source: ParsedSource) -> Any:
    return execute_doc_page.spawn(source.uri)


def _validate_modal_environment() -> None:
    actual = os.environ.get("MODAL_ENVIRONMENT")
    if actual != MODAL_ENVIRONMENT_NAME:
        current = "unset" if actual is None else repr(actual)
        raise RuntimeError(
            f"MODAL_ENVIRONMENT must be {MODAL_ENVIRONMENT_NAME!r}; got {current}"
        )


@app.local_entrypoint()
def main(pages: str = "", resume: bool = False) -> None:
    _validate_modal_environment()
    requested_pages = pages.split()
    sources = discover_sources()
    launcher = SpawnedPageRunner(
        _spawn_page,
        deadline_seconds=MODAL_TIMEOUT_SECONDS + MODAL_CALL_GRACE_SECONDS,
    )
    try:
        execute_and_publish(
            requested_pages,
            full=not requested_pages and not resume,
            resume=resume,
            jobs=max(1, len(sources)),
            page_runner_factory=launcher.prepare,
            warn_parallel_memory=False,
            runner_identity=MODAL_RUNNER_IDENTITY,
        )
    finally:
        launcher.cancel_unclaimed()
