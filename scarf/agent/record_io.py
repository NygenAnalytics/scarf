"""Policy-free byte and key operations for agent JSON records."""

import json
from typing import Any

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import collect_aiterator, sync


def join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def display_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def read_key(group: zarr.Group, key: str) -> bytes | None:
    value = sync(group.store.get(key, prototype=default_buffer_prototype()))
    return None if value is None else value.to_bytes()


def list_keys(group: zarr.Group, prefix: str) -> list[str]:
    resolved_prefix = f"{prefix.rstrip('/')}/"
    return sorted(collect_aiterator(group.store.list_prefix(resolved_prefix)))
