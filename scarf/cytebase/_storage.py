"""Shared authenticated bucket transfers for the SDK and Cytebase pipeline."""

import json
import os
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import httpx
from huggingface_hub import (
    BucketFile,
    batch_bucket_files,
    download_bucket_files,
    get_token,
    list_bucket_tree,
    sync_bucket,
)
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from huggingface_hub.utils import disable_progress_bars, parse_ratelimit_headers


def dataset_prefix(cytebase_id: str) -> str:
    if not re.fullmatch(r"[a-z0-9_]{1,80}", cytebase_id):
        raise ValueError(
            "Cytebase IDs use 1 to 80 lowercase letters, digits, or underscores"
        )
    return f"datasets/{cytebase_id}"


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode()


def error_message(error: Exception) -> str:
    """Return an error suitable for persisted diagnostics without HF credentials."""
    message = f"{type(error).__name__}: {error}"
    for token in (os.environ.get("HF_TOKEN"), get_token()):
        if token:
            message = message.replace(token, "[redacted]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", message)


def _retry_delay(error: httpx.HTTPError, attempt: int) -> float:
    delay = float(2 ** (attempt + 1))
    response = getattr(error, "response", None)
    if response is None:
        return delay
    headers = response.headers
    raw = headers.get("Retry-After")
    if raw:
        try:
            delay = max(delay, float(raw))
        except ValueError:
            try:
                moment = parsedate_to_datetime(raw)
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=UTC)
                delay = max(delay, (moment - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    raw_reset = headers.get("RateLimit-Reset") or headers.get("X-RateLimit-Reset")
    if raw_reset:
        try:
            reset = float(raw_reset)
            delay = max(delay, reset - time.time() if reset > 1_000_000_000 else reset)
        except ValueError:
            pass
    rate_limit = parse_ratelimit_headers(headers)
    if rate_limit is not None and rate_limit.remaining == 0:
        delay = max(delay, float(rate_limit.reset_in_seconds))
    return delay


def retry[T](
    operation: Callable[[], T],
    progress: Callable | None = None,
    *,
    stop_event: Event | None = None,
) -> T:
    """Retry transient HTTP failures three times, respecting server backoff.

    Server waits above five minutes are left for an explicit job retry instead
    of sleeping indefinitely or retrying before the server permits it.
    A supplied stop event interrupts backoff and prevents another attempt.
    """
    for attempt in range(4):
        if stop_event is not None and stop_event.is_set():
            raise CancelledError("Transfer cancelled")
        try:
            return operation()
        except (httpx.TransportError, httpx.HTTPStatusError, HfHubHTTPError) as error:
            response = getattr(error, "response", None)
            transient = (
                isinstance(error, httpx.TransportError)
                or response is not None
                and (response.status_code in {408, 429} or response.status_code >= 500)
            )
            if not transient or attempt == 3:
                raise
            delay = _retry_delay(error, attempt)
            if delay > 300:
                raise
            if progress is not None:
                progress(
                    "waiting_for_rate_limit"
                    if response is not None and response.status_code == 429
                    else "retrying_transfer",
                    message=f"Retry {attempt + 1}/3 after {delay:g} seconds",
                )
            if stop_event is None:
                time.sleep(delay)
            elif stop_event.wait(delay):
                raise CancelledError("Transfer cancelled") from error
    raise AssertionError("Unreachable retry state")


def _path(path: str, *, prefix: bool = False) -> str:
    checked = path.rstrip("/") if prefix else path
    if prefix and not checked:
        return ""
    if (
        not checked
        or any(part in {"", ".", ".."} for part in checked.split("/"))
        or any(character in checked for character in "*?[]\\")
    ):
        raise ValueError("Expected an exact, relative bucket object path")
    return path


class Bucket:
    """Access one explicit bucket without a public or production fallback."""

    def __init__(
        self, bucket: str | None = None, token: str | bool | None = None
    ) -> None:
        raw = bucket if bucket is not None else os.environ.get("CYTEBASE_BUCKET", "")
        bucket_id = raw.strip().removeprefix("hf://buckets/").rstrip("/")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", bucket_id
        ):
            raise ValueError("Supply bucket='namespace/name' or set CYTEBASE_BUCKET")
        self.bucket_id = bucket_id
        self.root = f"hf://buckets/{bucket_id}"
        self.token = (get_token() or False) if token is None else token
        self.progress: Callable | None = None

    def upload(self, files: Iterable[tuple[str | Path | bytes, str]]) -> None:
        additions = [(source, _path(path)) for source, path in files]
        if additions:
            retry(
                lambda: batch_bucket_files(
                    self.bucket_id, add=additions, token=self.token
                ),
                self.progress,
            )

    def write_json(self, path: str, value: dict) -> None:
        self.upload([(json_bytes(value), path)])

    def download(self, remote: str, destination: Path) -> None:
        remote = _path(remote)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with disable_progress_bars("huggingface_hub.download_bucket_files"):
            retry(
                lambda: download_bucket_files(
                    self.bucket_id,
                    files=[(remote, destination)],
                    raise_on_missing_files=True,
                    token=self.token,
                ),
                self.progress,
            )

    def read_bytes(self, path: str) -> bytes | None:
        with TemporaryDirectory(prefix="cytebase-read-") as directory:
            destination = Path(directory) / "object"
            try:
                self.download(path, destination)
            except EntryNotFoundError:
                return None
            return destination.read_bytes()

    def read_json(self, path: str) -> dict | None:
        raw = self.read_bytes(path)
        if raw is None:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}")
        return value

    def list_files(self, prefix: str) -> list[str]:
        prefix = _path(prefix, prefix=True)

        def listing() -> list[str]:
            paths = set()
            try:
                for item in list_bucket_tree(
                    self.bucket_id,
                    prefix=prefix or None,
                    recursive=True,
                    token=self.token,
                ):
                    if isinstance(item, BucketFile) and (
                        not prefix
                        or item.path == prefix.rstrip("/")
                        or item.path.startswith(prefix.rstrip("/") + "/")
                    ):
                        paths.add(_path(item.path))
            except EntryNotFoundError:
                if paths:
                    raise
            return sorted(paths)

        return retry(listing, self.progress)

    def sync_store(self, directory: Path, cytebase_id: str) -> None:
        target = f"{self.root}/{dataset_prefix(cytebase_id)}/data.zarr"
        retry(
            lambda: sync_bucket(str(directory), target, delete=False, token=self.token),
            self.progress,
        )

    def delete_exact(self, paths: list[str]) -> None:
        exact = sorted({_path(path) for path in paths})
        if exact:
            retry(
                lambda: batch_bucket_files(
                    self.bucket_id, delete=exact, token=self.token
                ),
                self.progress,
            )
