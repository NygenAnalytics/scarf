import os
from pathlib import Path
from typing import Any

import zarr
from obstore.store import from_url
from zarr.storage import ObjectStore

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_REQUIRED_ENV = (
    "R2_BUCKET",
    "R2_PREFIX",
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
)


def load_env(path: Path = _ENV_PATH) -> None:
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


def get_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing R2 setting: {name}")
    return value


def r2_uri(filename: str) -> str:
    bucket = get_env("R2_BUCKET")
    prefix = get_env("R2_PREFIX").strip("/")
    key = filename.lstrip("/")
    if prefix:
        key = f"{prefix}/{key}"
    return f"s3://{bucket}/{key}"


def open_r2_group(
    filename: str,
    *,
    mode: str = "r",
    read_only: bool | None = None,
    env_path: Path = _ENV_PATH,
    **open_group_kwargs: Any,
) -> zarr.Group:
    load_env(env_path)
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing R2 settings: {', '.join(missing)}")

    store = from_url(
        r2_uri(filename),
        endpoint=get_env("R2_ENDPOINT").rstrip("/"),
        access_key_id=get_env("R2_ACCESS_KEY_ID"),
        secret_access_key=get_env("R2_SECRET_ACCESS_KEY"),
    )
    object_store = ObjectStore(
        store=store,
        read_only=mode == "r" if read_only is None else read_only,
    )
    return zarr.open_group(store=object_store, mode=mode, **open_group_kwargs)
