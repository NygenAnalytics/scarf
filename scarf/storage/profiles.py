import os
from typing import Literal

from zarr.abc.store import Store

type StorageProfile = Literal["fast_local", "cloud"]
type ZarrLocation = str | Store


def is_remote_zarr_location(location: str) -> bool:
    """Return whether a location is a non-local URI."""
    if "://" not in location:
        return False
    return not location.startswith("file://")


def is_local_zarr_path(location: ZarrLocation) -> bool:
    """Return whether a location is a plain local filesystem path."""
    return isinstance(location, str) and not is_remote_zarr_location(location)


def resolve_storage_profile(
    location: ZarrLocation,
    requested: StorageProfile | None = None,
) -> StorageProfile:
    """Resolve a storage profile without changing process state."""
    if requested is not None:
        return requested
    env_profile = os.environ.get("SCARF_ZARR_PROFILE")
    if env_profile == "cloud":
        return "cloud"
    if env_profile == "fast_local":
        return "fast_local"
    if isinstance(location, str):
        return "cloud" if is_remote_zarr_location(location) else "fast_local"
    if type(location).__name__ in {"LocalStore", "MemoryStore"}:
        return "fast_local"
    return "cloud"
