from dataclasses import dataclass, field
from typing import Any, Literal

from ..storage.profiles import StorageProfile

type CountsTPolicy = Literal["rna", "all", "none"]
type MissingAssayPolicy = Literal["zero_fill", "error"]
type ComponentAction = Literal["write", "resume", "skip", "blocked"]


@dataclass(frozen=True, slots=True)
class AssayMergePlan:
    """Resolved plan for one assay in a DataStoreMerge.

    ``featureOverlapFraction`` is the share of union feature IDs present in at
    least two sources with that assay. A modality present in only one source
    reports ``1.0`` because zero-fill is intentional for the missing sources.
    """

    assayName: str
    sourcePresent: tuple[bool, ...]
    missingSources: tuple[str, ...]
    nFeatures: int
    featureOverlapFraction: float
    dtype: str
    chunks: tuple[int, int]
    shards: tuple[int, int] | None
    writeCountsT: bool
    estimatedWriteTasks: int
    countsAction: ComponentAction
    countsTAction: ComponentAction


@dataclass(frozen=True, slots=True)
class MergePlan:
    """Side-effect-free preview of a DataStoreMerge."""

    zarrPath: str
    outWorkspace: str | None
    sourceNames: tuple[str, ...]
    nCells: int
    assays: tuple[AssayMergePlan, ...]
    profile: StorageProfile
    seed: int | None
    countsT: CountsTPolicy
    missingAssayPolicy: MissingAssayPolicy
    willResume: bool
    canDump: bool
    blockedReason: str | None
    cellDataAction: ComponentAction
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComponentResult:
    """Outcome for one written or skipped merge component."""

    name: str
    action: ComponentAction


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result returned by DataStoreMerge.dump()."""

    zarrPath: str
    nCells: int
    assayNames: tuple[str, ...]
    components: tuple[ComponentResult, ...]
    resumed: bool
