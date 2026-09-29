"""Transport isolated documentation page caches."""

import io
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

from docs.execute_vignette import PageRunner, ParsedSource
from profiling.modal_support import await_function_call

type PageCachePayload = tuple[str, str, bytes]
type PageCallSpawner = Callable[[ParsedSource], Any]

_REQUIRED_CACHE_FILES = frozenset({"__version__.txt", "global.db"})


class CacheTransportError(RuntimeError):
    pass


def pack_page_cache(cache_path: Path) -> bytes:
    if not cache_path.is_dir():
        raise CacheTransportError(f"Page cache does not exist: {cache_path}")

    entries = sorted(cache_path.rglob("*"))
    files = {
        path.relative_to(cache_path).as_posix() for path in entries if path.is_file()
    }
    missing = _REQUIRED_CACHE_FILES - files
    if missing:
        names = ", ".join(sorted(missing))
        raise CacheTransportError(f"Page cache is missing required files: {names}")

    stream = io.BytesIO()
    with tarfile.open(
        fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT
    ) as archive:
        for path in entries:
            relative = path.relative_to(cache_path).as_posix()
            if path.is_symlink():
                raise CacheTransportError(f"Page cache contains a symlink: {relative}")
            if not path.is_file() and not path.is_dir():
                raise CacheTransportError(
                    f"Page cache contains an unsupported entry: {relative}"
                )
            archive.add(path, arcname=relative, recursive=False)
    return stream.getvalue()


def _validated_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    names: set[str] = set()
    file_names: set[str] = set()
    for member in members:
        pure = PurePosixPath(member.name)
        canonical = pure.as_posix()
        if (
            not member.name
            or pure.is_absolute()
            or "\\" in member.name
            or ".." in pure.parts
            or canonical != member.name.rstrip("/")
        ):
            raise CacheTransportError(
                f"Page cache archive contains an unsafe path: {member.name!r}"
            )
        if canonical in names:
            raise CacheTransportError(
                f"Page cache archive contains a duplicate path: {canonical}"
            )
        if member.issym() or member.islnk():
            raise CacheTransportError(
                f"Page cache archive contains a link: {canonical}"
            )
        if not member.isfile() and not member.isdir():
            raise CacheTransportError(
                f"Page cache archive contains an unsupported entry: {canonical}"
            )
        names.add(canonical)
        if member.isfile():
            file_names.add(canonical)

    missing = _REQUIRED_CACHE_FILES - file_names
    if missing:
        raise CacheTransportError(
            "Page cache archive is missing required files: "
            + ", ".join(sorted(missing))
        )
    return members


def restore_page_cache(
    payload: PageCachePayload,
    destination: Path,
    *,
    expected_uri: str,
    expected_hashkey: str,
) -> Path:
    if (
        not isinstance(payload, tuple)
        or len(payload) != 3
        or not isinstance(payload[0], str)
        or not isinstance(payload[1], str)
        or not isinstance(payload[2], bytes)
    ):
        raise CacheTransportError("Modal page result has an invalid payload")

    uri, hashkey, archive_bytes = payload
    if uri != expected_uri:
        raise CacheTransportError(
            f"Modal page result URI {uri!r} does not match {expected_uri!r}"
        )
    if hashkey != expected_hashkey:
        raise CacheTransportError(
            f"Modal page result hash {hashkey!r} does not match {expected_hashkey!r}"
        )
    if destination.exists():
        raise CacheTransportError(
            f"Page cache destination already exists: {destination}"
        )

    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = _validated_members(archive)
            destination.mkdir()
            try:
                archive.extractall(destination, members=members, filter="data")
            except BaseException:
                shutil.rmtree(destination, ignore_errors=True)
                raise
    except (tarfile.TarError, OSError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise CacheTransportError(f"Cannot extract Modal page cache: {exc}") from exc
    return destination


class SpawnedPageRunner:
    def __init__(
        self,
        spawn: PageCallSpawner,
        *,
        poll_seconds: float = 20.0,
        deadline_seconds: float = 10_800.0,
    ) -> None:
        self._spawn = spawn
        self._poll_seconds = poll_seconds
        self._deadline_seconds = deadline_seconds
        self._calls: dict[str, Any] = {}
        self._claimed: set[str] = set()
        self._prepared = False
        self._lock = Lock()

    def prepare(self, sources: list[ParsedSource]) -> PageRunner:
        if self._prepared:
            raise RuntimeError("Modal page runner has already been prepared")
        self._prepared = True
        try:
            for source in sources:
                self._calls[source.uri] = self._spawn(source)
        except BaseException:
            self.cancel_unclaimed()
            raise
        print(f"Submitted {len(self._calls)} page(s) to Modal", flush=True)
        return self.run

    def run(self, source: ParsedSource, destination: Path) -> Path:
        try:
            call = self._calls[source.uri]
        except KeyError as exc:
            raise CacheTransportError(
                f"No Modal call was submitted for {source.uri}"
            ) from exc
        with self._lock:
            self._claimed.add(source.uri)
        payload = await_function_call(
            call,
            pollSeconds=self._poll_seconds,
            deadlineSeconds=self._deadline_seconds,
        )
        return restore_page_cache(
            payload,
            destination,
            expected_uri=source.uri,
            expected_hashkey=source.hashkey,
        )

    def cancel_unclaimed(self) -> None:
        with self._lock:
            calls = [
                call for uri, call in self._calls.items() if uri not in self._claimed
            ]
        for call in calls:
            try:
                call.cancel()
            except Exception:
                pass
