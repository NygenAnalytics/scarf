"""Open verified Cytebase stores and keep writable analysis on local mounts."""

import asyncio
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import FsspecStore

from ._storage import Bucket, dataset_prefix, json_bytes

if TYPE_CHECKING:
    from scarf import DataStore


class _HfReadStore(FsspecStore):
    """Read one published store with unique listings and metadata cached in RAM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._metadata: dict[str, asyncio.Task[bytes | None]] = {}

    async def _read_metadata(
        self, key: str, prototype: BufferPrototype
    ) -> bytes | None:
        value = await super().get(key, prototype)
        return None if value is None else value.to_bytes()

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if (
            not self.read_only
            or byte_range is not None
            or key.rsplit("/", 1)[-1]
            not in {"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"}
        ):
            return await super().get(key, prototype, byte_range)
        task = self._metadata.get(key)
        if task is None:

            def discard_failed(completed: asyncio.Task[bytes | None]) -> None:
                # Consume errors even if every waiting reader was cancelled.
                if completed.cancelled() or completed.exception() is not None:
                    if self._metadata.get(key) is completed:
                        del self._metadata[key]

            task = asyncio.create_task(self._read_metadata(key, prototype))
            self._metadata[key] = task
            task.add_done_callback(discard_failed)
        # Cancelling one reader must not cancel a fetch shared by other readers.
        if not task.done():
            await asyncio.wait((task,))
        raw = task.result()
        return None if raw is None else prototype.buffer.from_bytes(raw)

    def close(self) -> None:
        self._metadata.clear()
        super().close()

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # Concurrent HF listings can append the same paths to its directory cache.
        seen: set[str] = set()
        async for name in super().list_dir(prefix):
            if name not in seen:
                seen.add(name)
                yield name


def _identity(storage: Bucket, cytebase_id: str) -> dict:
    """Resolve one committed ready record without trusting a lagging catalog row."""
    import numpy as np

    prefix = dataset_prefix(cytebase_id)
    record = storage.read_json(f"{prefix}/dataset.json")
    if record is None:
        raise KeyError(f"No dataset is registered as {cytebase_id!r}")
    if record.get("cytebaseId") != cytebase_id:
        raise ValueError("Dataset record identity does not match its path")
    version = record.get("processedVersionId")
    if (
        record.get("status") != "ready"
        or not version
        or version != record.get("latestVersionId")
    ):
        raise RuntimeError(
            f"Dataset {cytebase_id!r} is not ready for its registered version "
            f"(status={record.get('status')!r})"
        )
    dataset_id = str(UUID(record["datasetId"]))
    version = str(UUID(version))
    expected_uri = f"{storage.root}/{prefix}/data.zarr"
    inspection = record.get("inspection")
    receipt = record.get("buildReceipt")
    if not isinstance(inspection, dict) or not isinstance(receipt, dict):
        raise ValueError(
            "Ready dataset has no verified inspection and build receipt; rebuild it"
        )
    checksum = inspection.get("sourceSha256")
    verification = receipt.get("verification")
    if (
        not isinstance(checksum, str)
        or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        or inspection.get("datasetId") != dataset_id
        or inspection.get("collectionId") != record.get("collectionId")
        or inspection.get("datasetVersionId") != version
        or receipt.get("datasetVersionId") != version
        or receipt.get("sourceSha256") != checksum
        or receipt.get("zarrUri") != expected_uri
        or record.get("zarrUri") != expected_uri
        or not receipt.get("verifiedAt")
        or not isinstance(verification, dict)
        or verification.get("countsTMatches") is not True
        or verification.get("sourceSampleMatches", True) is not True
        or verification.get("nObs") != inspection.get("nObs")
        or verification.get("nVars") != inspection.get("nVars")
        or verification.get("countsTShape")
        != [inspection.get("nVars"), inspection.get("nObs")]
        or not isinstance(verification.get("countsDtype"), str)
    ):
        raise ValueError(
            "Ready dataset provenance does not match its verified Scarf store; rebuild it"
        )
    dtype = np.dtype(verification["countsDtype"])
    if dtype.kind not in "iuf":
        raise ValueError("Verified Scarf counts must have a real numeric dtype")
    return {
        "cytebaseId": cytebase_id,
        "datasetId": dataset_id,
        "processedVersionId": version,
        "sourceSha256": checksum,
        "zarrUri": expected_uri,
        "verifiedAt": receipt["verifiedAt"],
        "nObs": inspection["nObs"],
        "nVars": inspection["nVars"],
        "countsDtype": str(dtype),
    }


def _assert_unchanged(storage: Bucket, cytebase_id: str, expected: dict) -> None:
    if _identity(storage, cytebase_id) != expected:
        raise RuntimeError(
            "The published dataset changed while opening it; close this analysis and retry"
        )


def _verify_opened(datastore: "DataStore", expected: dict) -> None:
    """Match actual source array metadata to the committed conversion receipt."""
    import numpy as np

    assay = datastore.get_assay("RNA")
    shape = (expected["nObs"], expected["nVars"])
    counts, counts_t = assay.rawData, assay.rawDataT
    if (
        (datastore.cells.N, assay.feats.N) != shape
        or tuple(counts.shape) != shape
        or counts_t is None
        or tuple(counts_t.shape) != shape[::-1]
        or np.dtype(counts.dtype) != np.dtype(expected["countsDtype"])
        or np.dtype(counts_t.dtype) != np.dtype(expected["countsDtype"])
    ):
        raise ValueError(
            "Opened Scarf array metadata differs from the committed build receipt"
        )


def _close(datastore: "DataStore") -> None:
    stores = [datastore.z.store]
    matrix_root = getattr(datastore, "_matrix_z", None)
    if matrix_root is not None and matrix_root.store is not datastore.z.store:
        stores.append(matrix_root.store)
    for store in stores:
        store.close()


def _options(options: dict, *, mode: str) -> dict:
    result = dict(options)
    for reserved in ("zarr_loc", "storage_options"):
        if reserved in result:
            raise TypeError(
                f"{reserved} is resolved by Catalog and cannot be overridden"
            )
    if result.pop("zarr_mode", mode) != mode:
        raise ValueError(f"This Cytebase access requires zarr_mode={mode!r}")
    if mode == "r" and result.get("min_features_per_cell", -1) != -1:
        raise ValueError(
            "Read-only access requires min_features_per_cell=-1; use a local mount for filtering"
        )
    result.setdefault("min_features_per_cell", -1)
    result.setdefault("default_assay", "RNA")
    return result


def open_dataset(
    storage: Bucket, cytebase_id: str, **datastore_options: Any
) -> "DataStore":
    """Open current published counts read-only, failing if publication changes."""
    from scarf import DataStore

    options = _options(datastore_options, mode="r")
    identity = _identity(storage, cytebase_id)
    storage_options = {"token": storage.token, "skip_instance_cache": True}
    store = _HfReadStore.from_url(
        identity["zarrUri"],
        read_only=True,
        storage_options=storage_options,
    )
    try:
        datastore = DataStore(
            store,
            zarr_mode="r",
            storage_options=storage_options,
            zarrProfile="cloud",
            **options,
        )
    except Exception:
        store.close()
        raise
    try:
        _verify_opened(datastore, identity)
        _assert_unchanged(storage, cytebase_id, identity)
    except Exception:
        _close(datastore)
        raise
    return datastore


def mount_dataset(
    storage: Bucket, cytebase_id: str, at: str | Path, **datastore_options: Any
) -> "DataStore":
    """Create or reopen a local mount pinned to the current verified source build.

    The adjacent ``.cytebase.json`` sidecar contains source identity only. HF
    credentials are supplied afresh for every open and never written to disk.
    A replaced source requires a new local analysis; existing mounts are never
    silently retargeted to the replacement.
    """
    from scarf import DataStore, mount_datastore

    if "://" in str(at):
        raise ValueError("Cytebase analysis mounts require a local destination")
    target = Path(at).expanduser().absolute()
    sidecar = target.with_name(target.name + ".cytebase.json")
    identity = _identity(storage, cytebase_id)
    options = _options(datastore_options, mode="r+")
    storage_options = {"token": storage.token, "skip_instance_cache": True}
    reopening = target.exists()
    if reopening:
        if not sidecar.is_file():
            raise FileExistsError(
                "Destination exists without a Cytebase mount receipt; choose a new local path"
            )
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if saved != identity:
            raise ValueError(
                "This mount refers to a different dataset build; create a new local analysis"
            )
        import zarr

        root = zarr.open_group(str(target), mode="r")
        try:
            source = root.attrs.get("matrixSource")
            if (
                not isinstance(source, dict)
                or source.get("location") != identity["zarrUri"]
            ):
                raise ValueError(
                    "Mounted matrix source does not match the Cytebase receipt"
                )
        finally:
            root.store.close()
        datastore = DataStore(
            str(target), zarr_mode="r+", storage_options=storage_options, **options
        )
    else:
        if sidecar.exists():
            raise FileExistsError(
                "A Cytebase mount receipt already exists at this destination; choose a new local path"
            )
        datastore = mount_datastore(
            identity["zarrUri"],
            at=str(target),
            storage_options=storage_options,
            **options,
        )
    try:
        _verify_opened(datastore, identity)
        _assert_unchanged(storage, cytebase_id, identity)
        if not reopening:
            with sidecar.open("xb") as handle:
                handle.write(json_bytes(identity))
    except Exception:
        _close(datastore)
        raise
    return datastore
