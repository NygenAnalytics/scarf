import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from obstore.exceptions import AlreadyExistsError
from obstore.store import from_url

_ENV_PATH = Path(__file__).resolve().parent / ".env"
_DEFAULT_TRANSFER_CHUNK_BYTES = 16 * 1024 * 1024
_CREDENTIAL_KEYS = (
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
)
# Large objects (Cellxgene source ~46 GiB) exceed obstore's default 30s request timeout.
_CLIENT_OPTIONS = {
    "timeout": "12h",
    "connect_timeout": "120s",
    "read_timeout": "30m",
}
_RETRY_CONFIG = {
    "max_retries": 20,
    "retry_timeout": timedelta(minutes=30),
}


@dataclass(frozen=True, slots=True)
class ObjectDownload:
    fileBytes: int
    eTag: str | None


@dataclass(frozen=True, slots=True)
class ObjectUpload:
    fileBytes: int
    eTag: str | None


def _load_local_env(path: Path = _ENV_PATH) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def storage_options(uri: str) -> dict[str, str] | None:
    if not uri.startswith("s3://"):
        return None
    _load_local_env()
    missing = [name for name in _CREDENTIAL_KEYS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing R2 environment settings: {', '.join(missing)}")
    endpoint = os.environ["R2_ENDPOINT"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    assert endpoint and access_key and secret_key
    return {
        "access_key_id": access_key,
        "secret_access_key": secret_key,
        "endpoint": endpoint.rstrip("/"),
    }


def open_r2_object(uri: str) -> tuple[Any, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// object URI, got: {uri}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError("R2 object URI must include an object key")
    options = storage_options(uri)
    assert options is not None
    store = from_url(
        f"s3://{parsed.netloc}",
        client_options=_CLIENT_OPTIONS,
        retry_config=_RETRY_CONFIG,
        **options,
    )
    return store, key


def join_uri(prefix: str, *parts: str) -> str:
    base = prefix.rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part)
    return f"{base}/{suffix}" if suffix else base


def object_exists(uri: str) -> bool:
    store, key = open_r2_object(uri)
    try:
        store.head(key)
    except FileNotFoundError:
        return False
    return True


def object_size(uri: str) -> int | None:
    store, key = open_r2_object(uri)
    try:
        meta = store.head(key)
    except FileNotFoundError:
        return None
    return int(meta["size"])


def object_metadata(uri: str) -> dict[str, Any] | None:
    store, key = open_r2_object(uri)
    try:
        meta = store.head(key)
    except FileNotFoundError:
        return None
    e_tag = meta.get("e_tag")
    return {
        "size": int(meta["size"]),
        "eTag": str(e_tag) if e_tag else None,
    }


def list_objects(
    prefixUri: str,
    *,
    maxKeys: int = 256,
) -> list[dict[str, Any]]:
    if maxKeys < 1:
        raise ValueError("maxKeys must be positive")
    parsed = urlsplit(prefixUri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// object URI, got: {prefixUri}")
    prefix = parsed.path.lstrip("/")
    store, _key = open_r2_object(prefixUri if prefix else f"{prefixUri.rstrip('/')}/.")
    listed: list[dict[str, Any]] = []
    for batch in store.list(prefix=prefix or None, chunk_size=min(50, maxKeys)):
        for item in batch:
            path = str(item["path"])
            e_tag = item.get("e_tag")
            listed.append(
                {
                    "uri": f"s3://{parsed.netloc}/{path}",
                    "path": path,
                    "size": int(item["size"]),
                    "eTag": str(e_tag) if e_tag else None,
                }
            )
            if len(listed) >= maxKeys:
                return listed
    return listed


def list_common_prefixes(prefixUri: str) -> list[str]:
    parsed = urlsplit(prefixUri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// object URI, got: {prefixUri}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    store, _key = open_r2_object(
        prefixUri if parsed.path.lstrip("/") else f"{prefixUri.rstrip('/')}/."
    )
    result = store.list_with_delimiter(prefix or None)
    prefixes = result.get("common_prefixes") or []
    uris: list[str] = []
    for item in prefixes:
        path = str(item).strip("/")
        uris.append(f"s3://{parsed.netloc}/{path}")
    return uris


def get_json(uri: str) -> dict[str, Any]:
    body = get_bytes(uri)
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {uri}")
    return payload


def get_bytes(uri: str) -> bytes:
    store, key = open_r2_object(uri)
    return bytes(store.get(key).bytes())


def get_text(uri: str) -> str:
    return get_bytes(uri).decode("utf-8")


def _encode_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def put_json(uri: str, value: dict[str, Any]) -> None:
    store, key = open_r2_object(uri)
    store.put(key, _encode_json(value))


def put_json_if_absent(uri: str, value: dict[str, Any]) -> bool:
    return put_bytes_if_absent(uri, _encode_json(value))


def put_bytes_if_absent(uri: str, value: bytes) -> bool:
    store, key = open_r2_object(uri)
    try:
        store.put(
            key,
            value,
            mode="create",
            use_multipart=False,
        )
    except AlreadyExistsError:
        return False
    return True


def put_text_if_absent(uri: str, value: str) -> bool:
    return put_bytes_if_absent(uri, value.encode("utf-8"))


def download_file(
    uri: str,
    destination: str | Path,
    *,
    chunkBytes: int = _DEFAULT_TRANSFER_CHUNK_BYTES,
    maxAttempts: int = 8,
    maxWorkers: int | None = None,
) -> ObjectDownload:
    """Download with concurrent ranged GETs into a preallocated file."""
    if chunkBytes <= 0:
        raise ValueError("chunkBytes must be positive")
    if maxAttempts <= 0:
        raise ValueError("maxAttempts must be positive")

    store, key = open_r2_object(uri)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    meta = store.head(key)
    total = int(meta["size"])
    part_path = destination_path.with_name(f".{destination_path.name}.part")
    if part_path.is_file() and part_path.stat().st_size != total:
        part_path.unlink()
    if total == 0:
        part_path.write_bytes(b"")
        os.replace(part_path, destination_path)
        e_tag = meta.get("e_tag")
        return ObjectDownload(fileBytes=0, eTag=str(e_tag) if e_tag else None)

    ranges = [
        (start, min(start + chunkBytes, total)) for start in range(0, total, chunkBytes)
    ]
    workers = max(1, int(maxWorkers) if maxWorkers is not None else min(8, len(ranges)))
    with part_path.open("wb") as handle:
        handle.truncate(total)

    def fetch_range(start: int, end: int) -> None:
        attempts = 0
        while True:
            try:
                local_store, local_key = open_r2_object(uri)
                chunk = bytes(local_store.get_range(local_key, start=start, end=end))
            except Exception:
                attempts += 1
                if attempts >= maxAttempts:
                    raise
                time.sleep(min(60.0, 2.0**attempts))
                continue
            if not chunk:
                raise RuntimeError(f"Empty range response for {uri} at offset {start}")
            if start + len(chunk) > end:
                raise RuntimeError(
                    f"Range response for {uri} at offset {start} exceeded "
                    f"{end - start} bytes"
                )
            with part_path.open("r+b") as handle:
                handle.seek(start)
                handle.write(chunk)
            return

    if workers == 1 or len(ranges) == 1:
        for start, end in ranges:
            fetch_range(start, end)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_range, start, end) for start, end in ranges]
            for future in as_completed(futures):
                future.result()

    if part_path.stat().st_size != total:
        raise RuntimeError(
            f"Downloaded {part_path.stat().st_size} bytes from {uri}, expected {total}"
        )
    os.replace(part_path, destination_path)
    e_tag = meta.get("e_tag")
    return ObjectDownload(fileBytes=total, eTag=str(e_tag) if e_tag else None)


def upload_file(source: str | Path, uri: str) -> ObjectUpload:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    store, key = open_r2_object(uri)
    file_bytes = source_path.stat().st_size
    store.put(key, source_path, use_multipart=True)
    meta = store.head(key)
    e_tag = meta.get("e_tag")
    return ObjectUpload(fileBytes=file_bytes, eTag=str(e_tag) if e_tag else None)
