"""Download one local H5AD with aria2c and retain its source identity and hash."""

import hashlib
import os
import re
import shutil
import subprocess
from collections import deque
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from .._storage import retry

_PROGRESS = re.compile(
    r"\[#[0-9a-f]+ (?P<completed>\d+)B/(?P<total>\d+)B"
    r"(?:\(\d+%\))? CN:(?P<connections>\d+)(?: DL:(?P<speed>\d+)B)?"
)
_BUFFER_BYTES = 1024 * 1024


def download_connections() -> int:
    """Return the maximum number of aria2 connections to the source server."""
    try:
        connections = int(os.environ.get("CYTEBASE_DOWNLOAD_CONNECTIONS", "2"))
    except ValueError as error:
        raise ValueError("CYTEBASE_DOWNLOAD_CONNECTIONS must be 1 to 4") from error
    if not 1 <= connections <= 4:
        raise ValueError("CYTEBASE_DOWNLOAD_CONNECTIONS must be 1 to 4")
    return connections


def _probe(client: httpx.Client, url: str) -> tuple[int, str]:
    """Read one byte to confirm range support, size and a strong source ETag."""
    with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as response:
        response.raise_for_status()
        match = re.fullmatch(
            r"bytes 0-0/([1-9]\d*)", response.headers.get("Content-Range", "")
        )
        if response.status_code != 206 or match is None:
            raise ValueError("CELLxGENE did not return the requested byte range")
        etag = response.headers.get("ETag", "")
        if not re.fullmatch(r'"[^"\r\n]+"', etag):
            raise ValueError("CELLxGENE must supply a strong ETag")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("Source download requires an uncompressed HTTP response")
        size = 0
        for block in response.iter_raw(chunk_size=_BUFFER_BYTES):
            size += len(block)
            if size > 1:
                raise ValueError("CELLxGENE returned more bytes than requested")
        if size != 1:
            raise httpx.RemoteProtocolError("Incomplete CELLxGENE source probe")
        return int(match.group(1)), etag


def _run_aria2(
    executable: str,
    url: str,
    partial: Path,
    total: int,
    etag: str,
    connections: int,
    report: Callable[..., None],
) -> None:
    command = [
        executable,
        "--no-conf=true",
        "--no-netrc=true",
        "--continue=true",
        "--always-resume=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--file-allocation=none",
        f"--split={connections}",
        f"--max-connection-per-server={connections}",
        "--max-concurrent-downloads=1",
        "--max-tries=4",
        "--retry-wait=5",
        "--connect-timeout=30",
        "--timeout=60",
        "--auto-save-interval=15",
        "--summary-interval=15",
        "--human-readable=false",
        "--show-console-readout=false",
        "--truncate-console-readout=false",
        "--enable-color=false",
        "--download-result=hide",
        "--http-accept-gzip=false",
        "--header=Accept-Encoding: identity",
        f"--header=If-Match: {etag}",
        f"--stop-with-process={os.getpid()}",
        f"--dir={partial.parent.resolve()}",
        f"--out={partial.name}",
        "--",
        url,
    ]
    recent: deque[str] = deque(maxlen=12)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "LC_ALL": "C"},
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            match = _PROGRESS.search(line)
            if match is None:
                if line.strip():
                    recent.append(line.strip()[:500])
                continue
            completed, reported_total = int(match["completed"]), int(match["total"])
            if reported_total == 0 and completed == 0:
                continue  # aria2 has not received source headers yet.
            if reported_total != total or completed > total:
                raise ValueError("aria2 reported an unexpected source size")
            report(
                "downloading",
                completed=completed,
                total=total,
                downloadedBytes=completed,
                activeConnections=int(match["connections"]),
                downloadSpeedBytesPerSecond=int(match["speed"] or 0),
                unit="bytes",
                phase="transfer",
                message="aria2 downloading the local H5AD",
            )
        code = process.wait()
        if code:
            raise RuntimeError(
                f"aria2c failed with exit code {code}: " + " | ".join(recent)
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()


def download_h5ad(
    url: str,
    destination: Path,
    expected_bytes: int | None = None,
    *,
    progress: Callable[..., None] | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[int, str]:
    """Let aria2 manage parallel connections and local resume, then return size/hash.

    This invocation owns its partial file and aria2 control file. Both are removed
    on exit. Losing the worker loses the download; no remote checkpoint is kept.
    """
    executable = shutil.which("aria2c")
    if executable is None:
        raise RuntimeError(
            "aria2c is required for downloads. Install the aria2 system package "
            "or deploy the updated Cytebase Modal image."
        )
    connections = download_connections()
    destination = Path(destination)
    partial = destination.with_name(f"{destination.name}.part")
    control = partial.with_name(f"{partial.name}.aria2")
    if any(os.path.lexists(path) for path in (destination, partial, control)):
        raise FileExistsError(
            "Destination, partial and aria2 control paths must be absent"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    timings = timings if timings is not None else {}

    def report(stage: str, **values: Any) -> None:
        if progress is not None:
            progress(stage, **values)

    transfer_started = monotonic()
    report(
        "downloading",
        completed=0,
        total=expected_bytes,
        unit="bytes",
        phase="probe",
        message="Checking source identity and byte-range support",
    )
    with httpx.Client(
        timeout=httpx.Timeout(60, connect=30),
        follow_redirects=True,
        headers={"Accept-Encoding": "identity"},
    ) as client:
        total, etag = retry(lambda: _probe(client, url), progress=report)
        if expected_bytes is not None and total != expected_bytes:
            raise ValueError(
                f"CELLxGENE source is {total} bytes, expected {expected_bytes}"
            )
        # Exclusive creation makes this invocation the owner before aria2 opens it.
        partial.touch(exist_ok=False)
        try:
            report(
                "downloading",
                completed=0,
                total=total,
                downloadedBytes=0,
                activeConnections=0,
                unit="bytes",
                phase="transfer",
                message=f"Starting aria2 with up to {connections} source connections",
            )
            try:
                _run_aria2(executable, url, partial, total, etag, connections, report)
                if control.exists() or partial.stat().st_size != total:
                    raise ValueError("aria2 did not finish the expected H5AD")
                report(
                    "downloading",
                    completed=total,
                    total=total,
                    downloadedBytes=total,
                    activeConnections=0,
                    unit="bytes",
                    phase="source_check",
                    message="Confirming source identity after aria2 completed",
                )
                if retry(lambda: _probe(client, url), progress=report) != (total, etag):
                    raise ValueError("CELLxGENE source changed during download")
            finally:
                timings["downloadTransferSeconds"] = monotonic() - transfer_started

            report(
                "hashing_source",
                completed=0,
                total=total,
                downloadedBytes=total,
                activeConnections=0,
                unit="bytes",
                phase="checksum",
                message="Calculating local H5AD SHA-256",
            )
            digest = hashlib.sha256()
            hashed = 0
            started = last_report = monotonic()
            try:
                with partial.open("rb") as source:
                    while block := source.read(_BUFFER_BYTES):
                        digest.update(block)
                        hashed += len(block)
                        now = monotonic()
                        if now - last_report >= 1 or hashed == total:
                            report(
                                "hashing_source",
                                completed=hashed,
                                total=total,
                                downloadedBytes=total,
                                activeConnections=0,
                                unit="bytes",
                                phase="checksum",
                                message="Calculating local H5AD SHA-256",
                            )
                            last_report = now
            finally:
                timings["downloadHashSeconds"] = monotonic() - started
            if hashed != total:
                raise ValueError("Local H5AD size changed while hashing")
            # Finalize without copying or replacing an existing destination.
            destination.hardlink_to(partial)
        finally:
            partial.unlink(missing_ok=True)
            control.unlink(missing_ok=True)

    report(
        "downloaded",
        completed=total,
        total=total,
        downloadedBytes=total,
        activeConnections=0,
        unit="bytes",
        phase="complete",
        message="Local H5AD downloaded, size checked and SHA-256 recorded",
    )
    return total, digest.hexdigest()
