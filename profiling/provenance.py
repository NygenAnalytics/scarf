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


def _source_tree_digest() -> str | None:
    """Digest tracked source files that affect package/profiling behavior."""
    try:
        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "scarf",
                "profiling",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    digest = hashlib.sha256()
    if completed.returncode == 0:
        names = [
            name.decode("utf-8", errors="surrogateescape")
            for name in completed.stdout.split(b"\0")
            if name
        ]
    else:
        names = []
        for root_name in ("scarf", "profiling"):
            root = _REPO_ROOT / root_name
            names.extend(
                str(path.relative_to(_REPO_ROOT))
                for pattern in ("*.py", "*.pyi")
                for path in root.rglob(pattern)
                if "__pycache__" not in path.parts
            )
        names.extend(("pyproject.toml", "uv.lock"))
    for name in sorted(set(names)):
        encoded_name = name.encode("utf-8", errors="surrogateescape")
        path = _REPO_ROOT / name
        if not path.is_file():
            continue
        digest.update(encoded_name)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
    """Capture code/config identity on the submitting client before Modal.

    Results that lack these fields should be treated as diagnostic only.
    """
    dirty = _run_git("status", "--porcelain")
    return {
        "gitSha": _run_git("rev-parse", "HEAD"),
        "gitDescribe": _run_git("describe", "--always", "--dirty", "--tags"),
        "gitDirty": bool(dirty) if dirty is not None else None,
        "gitDiffSha256": _git_diff_digest(),
        "sourceTreeSha256": _source_tree_digest(),
        "lockfileSha256": _lockfile_digest(),
        "configSha256": config_digest(configPayload),
        "packageVersions": {
            "zarr": _package_version("zarr"),
            "numba": _package_version("numba"),
            "numpy": _package_version("numpy"),
            "obstore": _package_version("obstore"),
            "scarf": _package_version("scarf"),
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
    """Return a JSON-serializable provenance payload for stage/funnel results."""
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

    dirty = _run_git("status", "--porcelain")
    provenance: dict[str, Any] = {
        "gitSha": _run_git("rev-parse", "HEAD"),
        "gitDescribe": _run_git("describe", "--always", "--dirty", "--tags"),
        "gitDirty": bool(dirty) if dirty is not None else None,
        "gitDiffSha256": _git_diff_digest(),
        "sourceTreeSha256": _source_tree_digest(),
        "lockfileSha256": _lockfile_digest(),
        "configSha256": configDigestValue,
        "pythonVersion": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "modalInputId": modal_input_id,
        "modalFunctionCallId": modal_function_call_id,
        "cpuModel": _cpu_model(),
        "cpuCountLogical": os.cpu_count(),
        "packageVersions": {
            "zarr": _package_version("zarr"),
            "numba": _package_version("numba"),
            "numpy": _package_version("numpy"),
            "obstore": _package_version("obstore"),
            "scarf": _package_version("scarf"),
        },
        "zarrCodecPipeline": zarr_pipeline,
        "zarrAsyncConcurrency": zarr_async_concurrency,
        "scarfZarrProfile": os.environ.get("SCARF_ZARR_PROFILE"),
        "nonpreemptible": nonpreemptible,
        "hasClientCodeIdentity": False,
    }
    if clientProvenance:
        for key in (
            "gitSha",
            "gitDescribe",
            "gitDirty",
            "gitDiffSha256",
            "sourceTreeSha256",
            "lockfileSha256",
            "configSha256",
            "packageVersions",
        ):
            value = clientProvenance.get(key)
            if value is not None:
                provenance[key] = value
        provenance["hasClientCodeIdentity"] = bool(
            clientProvenance.get("sourceTreeSha256")
        )
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
