"""Readable run records and exclusive local writers, independent of Scarf storage."""

import errno
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RecordError(RuntimeError):
    """A run record is missing, inconsistent, or cannot be safely replaced."""


class RunLockedError(RecordError):
    """Another process holds the requested writer lock."""


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RecordError(f"Record is not finite, UTF-8 JSON: {exc}") from exc


def digest(value: Any) -> str:
    """Hash a JSON value using the exact canonical representation saved to disk."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> Any:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_pairs,
        )
        _json_bytes(value)
        return value
    except (OSError, ValueError, UnicodeError) as exc:
        raise RecordError(f"Cannot read record {path.name}: {exc}") from exc


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any, *, immutable: bool) -> None:
    payload = _json_bytes(value)
    missing = []
    parent = path.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _sync_directory(directory.parent)
    if immutable and path.exists():
        if path.read_bytes() == payload:
            return
        raise RecordError(f"Immutable record already exists: {path.name}")
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                # A no-replace publication is necessary even when callers hold locks.
                os.link(temporary_path, path)
            except FileExistsError as exc:
                if path.read_bytes() != payload:
                    raise RecordError(
                        f"Concurrent record creation: {path.name}"
                    ) from exc
        else:
            os.replace(temporary_path, path)
        _sync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


class RunRecords:
    """An immutable manifest and ordered, checked events in a local directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir).expanduser().resolve()
        self._manifest = self.read_json("run.json")
        if not isinstance(self._manifest, dict):
            raise RecordError("run.json must contain a JSON object")
        self.events()

    @classmethod
    def create(cls, run_dir: str | Path, manifest: dict[str, Any]) -> "RunRecords":
        """Reserve a new directory and persist a manifest before any execution."""
        _json_bytes(manifest)
        if not isinstance(manifest, dict):
            raise RecordError("The run manifest must be a JSON object")
        directory = Path(run_dir).expanduser().resolve()
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError as exc:
            raise RecordError(f"Run directory already exists: {directory}") from exc
        _sync_directory(directory.parent)
        _atomic_json(directory / "run.json", manifest, immutable=True)
        (directory / "events").mkdir()
        _sync_directory(directory)
        return cls(directory)

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a detached copy; the saved request cannot be changed in place."""
        value: dict[str, Any] = json.loads(_json_bytes(self._manifest))
        return value

    def _path(self, relative: str | Path) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise RecordError(
                "Record paths must be nonempty relative paths without '..'"
            )
        path = self.path
        for part in candidate.parts:
            path = path / part
            if path.is_symlink():
                raise RecordError(f"Record paths cannot contain symlinks: {relative}")
        if not path.resolve().is_relative_to(self.path):
            raise RecordError("Record path escapes the run directory")
        return path

    def read_json(self, relative: str | Path) -> Any:
        return _read(self._path(relative))

    def write_json(
        self, relative: str | Path, value: Any, immutable: bool = True
    ) -> str:
        """Save evidence or call data; existing immutable content must match exactly."""
        path = self._path(relative)
        normalized = path.relative_to(self.path)
        if normalized == Path("run.json") or normalized.parts[0] == "events":
            raise RecordError("Use append for events; the manifest cannot be replaced")
        _atomic_json(path, value, immutable=immutable)
        return normalized.as_posix()

    def write_evidence(self, name: str, value: Any) -> str:
        """Persist evidence and return the relative reference used in events."""
        return self.write_json(f"evidence/{name}.json", value)

    def events(self) -> list[dict[str, Any]]:
        """Read the complete ordered log, rejecting gaps and edited event records."""
        directory = self._path("events")
        if not directory.is_dir():
            raise RecordError("The run is missing its events directory")
        paths = sorted(
            path for path in directory.iterdir() if not path.name.startswith(".tmp-")
        )
        events: list[dict[str, Any]] = []
        previous: str | None = None
        for sequence, path in enumerate(paths, start=1):
            if (
                path.name != f"{sequence:06d}.json"
                or path.is_symlink()
                or not path.is_file()
            ):
                raise RecordError(f"Invalid event sequence at {path.name}")
            event = _read(path)
            if not isinstance(event, dict):
                raise RecordError(f"Event must be an object: {path.name}")
            record_hash = event.get("recordHash")
            unsigned = {
                key: value for key, value in event.items() if key != "recordHash"
            }
            if (
                type(event.get("sequence")) is not int
                or event["sequence"] != sequence
                or not isinstance(event.get("kind"), str)
                or not event["kind"].strip()
                or event.get("previousHash") != previous
                or record_hash != digest(unsigned)
            ):
                raise RecordError(f"Invalid event integrity: {path.name}")
            try:
                timestamp = datetime.fromisoformat(event["timestamp"])
                if timestamp.tzinfo is None:
                    raise ValueError("timestamp has no timezone")
            except (KeyError, TypeError, ValueError) as exc:
                raise RecordError(f"Invalid event timestamp: {path.name}") from exc
            events.append(event)
            previous = record_hash
        return events

    def append(self, kind: str, **payload: Any) -> dict[str, Any]:
        if not isinstance(kind, str) or not kind.strip():
            raise RecordError("Event kind must be a nonempty string")
        reserved = {"sequence", "timestamp", "kind", "recordHash", "previousHash"}
        if reserved.intersection(payload):
            raise RecordError("Event payload must not override envelope fields")
        events = self.events()
        sequence = len(events) + 1
        event = {
            "sequence": sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "previousHash": events[-1]["recordHash"] if events else None,
            **payload,
        }
        event["recordHash"] = digest(event)
        _atomic_json(self._path(f"events/{sequence:06d}.json"), event, immutable=True)
        return event

    def latest(self, kind: str) -> dict[str, Any] | None:
        return next(
            (event for event in reversed(self.events()) if event["kind"] == kind), None
        )


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RecordError(f"Writer lock must not be a symlink: {path}")
    with path.open("a+b") as stream:
        try:
            if sys.platform == "win32":
                import msvcrt

                if path.stat().st_size == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise RunLockedError(f"Another agent writer holds {path}") from exc
            raise
        try:
            yield
        finally:
            if sys.platform == "win32":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    # Keep the lock inode: deleting it races with another process acquiring it.


def run_lock(run_dir: str | Path) -> AbstractContextManager[None]:
    """Exclusively lock a run, including its initial creation, without creating it."""
    path = Path(run_dir).expanduser().resolve()
    return _lock(path.parent / f".{path.name}.scarf-agent-run.lock")


def source_lock(source: str | Path) -> AbstractContextManager[None]:
    """Lock a local mount via an adjacent file, outside its internal store layout."""
    path = Path(source).expanduser().resolve()
    return _lock(path.parent / f".{path.name}.scarf-agent.lock")


def _linux_process(pid: int) -> tuple[str, str] | None:
    try:
        # comm is parenthesized and can itself contain spaces and parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def process_identity() -> dict[str, Any]:
    """Record PID reuse-resistant Linux identity; other platforms remain conservative."""
    result: dict[str, Any] = {"pid": os.getpid(), "platform": sys.platform}
    if sys.platform.startswith("linux"):
        try:
            identity = _linux_process(os.getpid())
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if identity is not None:
                result["startTicks"] = identity[1]
                result["bootId"] = boot_id
        except (OSError, IndexError):
            pass
    return result


def process_alive(identity: Mapping[str, Any]) -> bool:
    """Return false only when the recorded process is confirmed to have stopped."""
    pid = identity.get("pid")
    if type(pid) is not int or pid <= 0 or identity.get("platform") != sys.platform:
        return True
    if sys.platform.startswith("linux") and identity.get("startTicks"):
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if identity.get("bootId") != boot_id:
                return False
            current = _linux_process(pid)
            return (
                current is not None
                and current[0] != "Z"
                and current[1] == identity["startTicks"]
            )
        except (OSError, IndexError):
            return True
    if os.name == "nt":
        # os.kill(pid, 0) can terminate processes on Windows. Never use it there.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def procedure_identity() -> str:
    """Hash shipped Scarf Python sources, excluding tests and generated bytecode."""
    root = Path(__file__).parent.parent
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        hasher.update(relative.as_posix().encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()
