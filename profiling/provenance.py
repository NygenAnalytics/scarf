"""Capture run provenance for profiling result JSON."""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PACKAGES = ("scarf", "profiling")
_SOURCE_FILES = ("pyproject.toml", "uv.lock")
# Code and config identity. The submitting client captures it once and every
# stage and funnel result carries it under these flat keys.
_IDENTITY_KEYS = (
    "gitSha",
    "gitDescribe",
    "gitDirty",
    "gitDiffSha256",
    "sourceTreeSha256",
    "lockfileSha256",
    "configSha256",
    "packageVersions",
)


def _run_git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


@lru_cache(maxsize=1)
def _lockfile_digest() -> str | None:
    lock_path = _REPO_ROOT / "uv.lock"
    if not lock_path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(lock_path.read_bytes())
    return digest.hexdigest()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _hash_bytes(path.read_bytes())


def source_tree_files(root: Path) -> list[str]:
    """Return the code files under ``root`` that identify a profiling run.

    The Modal image ships exactly these files at the same relative paths: the Python
    sources of ``scarf`` and ``profiling`` plus the dependency definition. The client
    and the container therefore hash the same list, and uncommitted edits count.
    """
    names = [
        path.relative_to(root).as_posix()
        for package in _SOURCE_PACKAGES
        for path in (root / package).rglob("*.py")
        if path.is_file()
    ]
    names.extend(name for name in _SOURCE_FILES if (root / name).is_file())
    return sorted(names)


def source_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for name in source_tree_files(root):
        digest.update(name.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _executed_source_tree_digest() -> str:
    return source_tree_digest(_REPO_ROOT)


def _git_diff_digest() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "diff", "HEAD", "--", "scarf", "profiling", "pyproject.toml"],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _hash_bytes(completed.stdout)


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
    except ImportError:
        return None
    try:
        return version(name)
    except Exception:
        return None


def config_digest(config_payload: dict[str, Any] | Path | str | None) -> str | None:
    """Stable digest of a profiling config payload or file."""
    if config_payload is None:
        return None
    if isinstance(config_payload, Path | str):
        return _hash_file(Path(config_payload))
    cleaned = {
        key: value for key, value in config_payload.items() if key != "clientProvenance"
    }
    encoded = json.dumps(cleaned, sort_keys=True, default=str).encode()
    return _hash_bytes(encoded)


def collect_client_code_identity(
    *,
    configPayload: dict[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Capture code/config identity, normally on the submitting client before Modal.

    ``collect_run_provenance`` calls it only when no client identity was sent.
    Results that lack these fields should be treated as diagnostic only.
    """
    dirty = _run_git("status", "--porcelain")
    return {
        "gitSha": _run_git("rev-parse", "HEAD"),
        "gitDescribe": _run_git("describe", "--always", "--dirty", "--tags"),
        "gitDirty": bool(dirty) if dirty is not None else None,
        "gitDiffSha256": _git_diff_digest(),
        "sourceTreeSha256": source_tree_digest(_REPO_ROOT),
        "lockfileSha256": _lockfile_digest(),
        "configSha256": config_digest(configPayload),
        "packageVersions": {
            name: _package_version(name)
            for name in ("zarr", "numba", "numpy", "obstore", "scarf")
        },
        "capturedOn": "client",
    }


def attach_client_provenance(
    config_dict: dict[str, Any],
    *,
    configPath: Path | str | None = None,
) -> dict[str, Any]:
    """Return a config dict that carries client code identity for Modal jobs."""
    payload = dict(config_dict)
    payload["clientProvenance"] = collect_client_code_identity(
        configPayload=configPath if configPath is not None else config_dict,
    )
    return payload


def collect_run_provenance(
    *,
    nonpreemptible: bool | None = None,
    clientProvenance: dict[str, Any] | None = None,
    configDigestValue: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable provenance payload for stage/funnel results.

    Code identity comes from ``clientProvenance`` when the submitting client
    captured it. Otherwise this process collects it. The digest of the code this
    process runs is recorded beside it, so a stale deployment shows as a mismatch.
    """
    identity = clientProvenance or collect_client_code_identity()
    provenance = {key: identity.get(key) for key in _IDENTITY_KEYS}
    executed_digest = _executed_source_tree_digest()
    client_digest = provenance["sourceTreeSha256"] if clientProvenance else None
    if provenance["configSha256"] is None:
        provenance["configSha256"] = configDigestValue
    zarr_pipeline = None
    zarr_async_concurrency = None
    try:
        import zarr

        zarr_pipeline = zarr.config.get("codec_pipeline.path")
        zarr_async_concurrency = zarr.config.get("async.concurrency")
    except Exception:
        pass
    modal_input_id = None
    modal_function_call_id = None
    try:
        import modal

        modal_input_id = modal.current_input_id()
        modal_function_call_id = modal.current_function_call_id()
    except Exception:
        pass

    provenance.update(
        {
            "pythonVersion": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "modalInputId": modal_input_id,
            "modalFunctionCallId": modal_function_call_id,
            "cpuModel": _cpu_model(),
            "cpuCountLogical": os.cpu_count(),
            "zarrCodecPipeline": zarr_pipeline,
            "zarrAsyncConcurrency": zarr_async_concurrency,
            "scarfZarrProfile": os.environ.get("SCARF_ZARR_PROFILE"),
            "nonpreemptible": nonpreemptible,
            "hasClientCodeIdentity": client_digest is not None,
            "executedSourceTreeSha256": executed_digest,
            "sourceTreeMatchesClient": (
                None if client_digest is None else client_digest == executed_digest
            ),
        }
    )
    if clientProvenance:
        provenance["clientCapturedOn"] = clientProvenance.get("capturedOn")
    return provenance


def provenance_from_config(
    config: Any,
    *,
    nonpreemptible: bool | None = None,
) -> dict[str, Any]:
    """Collect provenance, preferring client identity carried on the config."""
    client = getattr(config, "clientProvenance", None)
    config_hash = None
    try:
        if hasattr(config, "model_dump"):
            config_hash = config_digest(config.model_dump(mode="python"))
    except Exception:
        config_hash = None
    return collect_run_provenance(
        nonpreemptible=nonpreemptible,
        clientProvenance=client if isinstance(client, dict) else None,
        configDigestValue=config_hash,
    )


def _cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return platform.processor() or None
    for line in path.read_text().splitlines():
        if line.startswith("model name"):
            _, _, value = line.partition(":")
            return value.strip() or None
    return platform.processor() or None
