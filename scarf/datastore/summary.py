from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import zarr

from ..assay import Assay
from ..graph.state import AssayState
from ..metadata import MetaData
from ..storage.artifacts import ArtifactStatus
from ..storage.budget import ResourceBudget
from ..storage.profiles import StorageProfile
from ..storage.refs import ArtifactRef, ArtifactScope
from ..storage.types import ZarrMode


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    memory_bytes: int
    workers: int
    storage_profile: StorageProfile

    def to_dict(self) -> dict[str, int | str]:
        return {
            "memory_bytes": self.memory_bytes,
            "workers": self.workers,
            "storage_profile": self.storage_profile,
        }


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    ref: ArtifactRef
    operation: str | None
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "operation": self.operation,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class AssaySummary:
    name: str
    assay_type: str
    total_features: int
    active_features: int
    feature_columns: tuple[str, ...]
    dataset_fingerprint: str | None
    state: AssayState | None
    artifacts: tuple[ArtifactSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "assay_type": self.assay_type,
            "total_features": self.total_features,
            "active_features": self.active_features,
            "feature_columns": list(self.feature_columns),
            "dataset_fingerprint": self.dataset_fingerprint,
            "state": self.state.to_dict() if self.state is not None else None,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class DataStoreSummary:
    zarr_mode: ZarrMode
    workspace: str | None
    default_assay: str
    scarf_version: str
    resources: ResourceSummary
    total_cells: int
    active_cells: int
    cell_columns: tuple[str, ...]
    assays: tuple[AssaySummary, ...]
    artifacts: tuple[ArtifactSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic representation containing JSON-safe values."""
        return {
            "zarr_mode": self.zarr_mode,
            "workspace": self.workspace,
            "default_assay": self.default_assay,
            "scarf_version": self.scarf_version,
            "resources": self.resources.to_dict(),
            "total_cells": self.total_cells,
            "active_cells": self.active_cells,
            "cell_columns": list(self.cell_columns),
            "assays": [assay.to_dict() for assay in self.assays],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


class _SummaryStore(Protocol):
    zarr_mode: ZarrMode
    workspace: str | None
    _defaultAssay: str
    resources: ResourceBudget
    storageProfile: StorageProfile
    cells: MetaData

    @property
    def assay_names(self) -> list[str]: ...

    @property
    def zw(self) -> zarr.Group: ...

    def _get_assay(self, from_assay: str | None) -> Assay: ...

    def get_assay_state(self, from_assay: str | None = None) -> AssayState | None: ...

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        from_assay: str | None = None,
        scope: ArtifactScope = "assay",
        complete_only: bool = False,
    ) -> list[ArtifactRef]: ...

    def inspect_artifact(self, ref: ArtifactRef) -> ArtifactStatus: ...


def _count_active(metadata: MetaData) -> int:
    if metadata.N == 0:
        return 0
    return sum(
        int(block.active_global_indices.size)
        for block in metadata.iter_row_blocks(cell_key="I")
    )


def _summarize_artifacts(
    store: _SummaryStore,
    refs: list[ArtifactRef],
) -> tuple[ArtifactSummary, ...]:
    summaries = []
    for ref in sorted(
        refs,
        key=lambda item: (item.kind, item.assay or "", item.artifact_id),
    ):
        status = store.inspect_artifact(ref)
        summaries.append(
            ArtifactSummary(
                ref=ref,
                operation=status.operation,
                complete=status.complete,
            )
        )
    return tuple(summaries)


def _assay_type_mapping(root: zarr.Group) -> dict[str, str]:
    value = root.attrs.get("assayTypes", {})
    if not isinstance(value, Mapping):
        return {}
    return {str(name): str(assay_type) for name, assay_type in value.items()}


def build_datastore_summary(
    store: _SummaryStore,
    *,
    scarf_version: str,
) -> DataStoreSummary:
    assay_types = _assay_type_mapping(store.zw)
    assays = []
    for assay_name in sorted(store.assay_names):
        assay = store._get_assay(assay_name)
        fingerprint = assay.attrs.get("dataset_fingerprint")
        assays.append(
            AssaySummary(
                name=assay_name,
                assay_type=assay_types.get(assay_name, "Assay"),
                total_features=int(assay.feats.N),
                active_features=_count_active(assay.feats),
                feature_columns=tuple(sorted(assay.feats.columns)),
                dataset_fingerprint=(
                    str(fingerprint) if fingerprint is not None else None
                ),
                state=store.get_assay_state(assay_name),
                artifacts=_summarize_artifacts(
                    store,
                    store.list_artifacts(
                        from_assay=assay_name,
                        scope="assay",
                    ),
                ),
            )
        )

    return DataStoreSummary(
        zarr_mode=store.zarr_mode,
        workspace=store.workspace,
        default_assay=store._defaultAssay,
        scarf_version=scarf_version,
        resources=ResourceSummary(
            memory_bytes=store.resources.memoryBytes,
            workers=store.resources.workers,
            storage_profile=store.storageProfile,
        ),
        total_cells=int(store.cells.N),
        active_cells=_count_active(store.cells),
        cell_columns=tuple(sorted(store.cells.columns)),
        assays=tuple(assays),
        artifacts=_summarize_artifacts(
            store,
            store.list_artifacts(scope="datastore"),
        ),
    )
