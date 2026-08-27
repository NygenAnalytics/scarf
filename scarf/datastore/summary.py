from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import zarr

from ..assay import Assay
from ..metadata import MetaData
from ..storage.artifacts import (
    ArtifactStatus,
    inspect_artifact,
    list_artifacts as list_artifact_refs,
)
from ..storage.budget import ResourceBudget, resolve_budget
from ..storage.profiles import StorageProfile, resolve_storage_profile
from ..storage.pipeline_runs import list_pipeline_run_records
from ..storage.refs import ArtifactRef, ArtifactScope
from ..storage.schema import validate_assay_name
from ..storage.stores import load_zarr
from ..storage.types import ZarrMode, as_zarr_group


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
    artifacts: tuple[ArtifactSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "assay_type": self.assay_type,
            "total_features": self.total_features,
            "active_features": self.active_features,
            "feature_columns": list(self.feature_columns),
            "dataset_fingerprint": self.dataset_fingerprint,
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
    pipeline_run_counts: dict[str, int]
    labeled_pipeline_runs: tuple[tuple[str, str], ...]

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
            "pipeline_run_counts": dict(self.pipeline_run_counts),
            "labeled_pipeline_runs": [
                {"label": label, "run_id": run_id}
                for label, run_id in self.labeled_pipeline_runs
            ],
        }


class _AssaySummaryView(Protocol):
    feats: MetaData
    attrs: Mapping[str, Any]


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

    def _get_assay(self, from_assay: str | None) -> Assay | _AssaySummaryView: ...

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        from_assay: str | None = None,
        scope: ArtifactScope = "assay",
        complete_only: bool = False,
    ) -> list[ArtifactRef]: ...

    def inspect_artifact(self, ref: ArtifactRef) -> ArtifactStatus: ...


@dataclass(slots=True)
class _ReadOnlyAssayView:
    name: str
    feats: MetaData
    attrs: Mapping[str, Any]


class _ReadOnlySummaryStore:
    """Read-only adapter that summarizes a Scarf Zarr without DataStore init."""

    zarr_mode: ZarrMode = "r"

    def __init__(
        self,
        zarr_path: str,
        *,
        default_assay: str | None = None,
        workspace: str | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self.resources = resolve_budget()
        self.storageProfile = resolve_storage_profile(zarr_path)
        self._root = load_zarr(zarr_path, mode="r", storage_options=storage_options)
        self.cells = MetaData(as_zarr_group(self.zw["cellData"], name="cellData"))
        names = self.assay_names
        if not names:
            raise ValueError(f"No assays found in Zarr store at {zarr_path}")
        self._defaultAssay = self._resolve_default_assay(default_assay, names)

    @property
    def zw(self) -> zarr.Group:
        if self.workspace is None:
            return self._root
        return as_zarr_group(self._root[self.workspace], name=self.workspace)

    @property
    def assay_names(self) -> list[str]:
        names: list[str] = []
        for name in sorted(dict.fromkeys(self.zw.group_keys())):
            node = self.zw[name]
            if isinstance(node, zarr.Group) and "is_assay" in node.attrs:
                validate_assay_name(name)
                names.append(name)
        return names

    def _resolve_default_assay(
        self,
        requested: str | None,
        assay_names: list[str],
    ) -> str:
        if requested is not None:
            if requested not in assay_names:
                raise ValueError(
                    f"Default assay {requested!r} was not found. "
                    f"Choose one from: {' '.join(assay_names)}"
                )
            return requested
        stored = self.zw.attrs.get("defaultAssay")
        if isinstance(stored, str) and stored in assay_names:
            return stored
        if "RNA" in assay_names:
            return "RNA"
        return assay_names[0]

    def _get_assay(self, from_assay: str | None) -> Assay | _AssaySummaryView:
        assay = from_assay or self._defaultAssay
        if assay not in self.assay_names:
            raise ValueError(f"Assay {assay!r} not found in the Zarr file")
        feature_path = f"{assay}/featureData"
        display_path = (
            feature_path
            if self.workspace is None
            else f"{self.workspace}/{feature_path}"
        )
        assay_group = as_zarr_group(
            self.zw[assay],
            name=assay if self.workspace is None else f"{self.workspace}/{assay}",
        )
        return _ReadOnlyAssayView(
            name=assay,
            feats=MetaData(as_zarr_group(self.zw[feature_path], name=display_path)),
            attrs=assay_group.attrs,
        )

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        from_assay: str | None = None,
        scope: ArtifactScope = "assay",
        complete_only: bool = False,
    ) -> list[ArtifactRef]:
        if scope == "assay" and from_assay is None:
            from_assay = self._defaultAssay
        return list_artifact_refs(
            self.zw,
            scope=scope,
            assay=from_assay,
            kind=kind,
            complete_only=complete_only,
        )

    def inspect_artifact(self, ref: ArtifactRef) -> ArtifactStatus:
        return inspect_artifact(self.zw, ref)


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
                artifacts=_summarize_artifacts(
                    store,
                    store.list_artifacts(
                        from_assay=assay_name,
                        scope="assay",
                    ),
                ),
            )
        )

    runs = list_pipeline_run_records(store.zw, limit=2**31 - 1)
    pipeline_run_counts = {
        "total": len(runs),
        "running": sum(run.status == "running" for run in runs),
        "completed": sum(run.status == "completed" for run in runs),
        "failed": sum(run.status == "failed" for run in runs),
        "interrupted": sum(run.status == "interrupted" for run in runs),
        "incomplete": sum(not run.complete for run in runs),
    }
    labeled_pipeline_runs = tuple(
        sorted(
            (
                (run.label, run.run_id)
                for run in runs
                if run.successfully_completed and run.label is not None
            ),
            key=lambda item: item[0],
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
        pipeline_run_counts=pipeline_run_counts,
        labeled_pipeline_runs=labeled_pipeline_runs,
    )


def summarize_zarr_readonly(
    zarr_path: str,
    *,
    default_assay: str | None = None,
    workspace: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> DataStoreSummary:
    """Summarize an existing Scarf Zarr without mutating it."""
    from .. import __version__

    store = _ReadOnlySummaryStore(
        zarr_path,
        default_assay=default_assay,
        workspace=workspace,
        storage_options=storage_options,
    )
    return build_datastore_summary(store, scarf_version=__version__)
