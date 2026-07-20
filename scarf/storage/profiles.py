import os
from typing import Literal

import zarr
from zarr.abc.store import Store

from .budget import get_resource_budget

type StorageProfile = Literal["fast_local", "cloud"]
type ZarrLocation = str | Store

_activeProfile: StorageProfile | None = None


def set_storage_profile(profile: StorageProfile | None) -> None:
    """Override the active Zarr storage profile for this process."""
    global _activeProfile
    _activeProfile = profile


def _get_storage_profile_override() -> StorageProfile | None:
    return _activeProfile


def get_storage_profile() -> StorageProfile:
    """Return active profile from override or ``SCARF_ZARR_PROFILE`` env var."""
    if _activeProfile is not None:
        return _activeProfile
    envProfile = os.environ.get("SCARF_ZARR_PROFILE", "fast_local")
    if envProfile == "cloud":
        return "cloud"
    return "fast_local"


def configure_zarr_io_for_profile() -> None:
    """Set Zarr async IO parallelism to the budget's worker count."""
    budget = get_resource_budget()
    zarr.config.set({"async.concurrency": max(1, budget.workers)})


def is_remote_zarr_location(location: str) -> bool:
    """Return whether a location is a non-local URI."""
    if "://" not in location:
        return False
    return not location.startswith("file://")


def is_local_zarr_path(location: ZarrLocation) -> bool:
    """Return whether a location is a plain local filesystem path."""
    return isinstance(location, str) and not is_remote_zarr_location(location)


def _maybe_auto_cloud_profile(location: ZarrLocation) -> None:
    if _activeProfile is not None:
        return
    if isinstance(location, str) and not is_remote_zarr_location(location):
        return
    if isinstance(location, Store):
        return
    set_storage_profile("cloud")
