import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from profiling.datasets import DEFAULT_TARGET_SIZES

StageName = Literal[
    "createStore",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    "makeGraph",
    "runUmap",
    "runLeiden",
    "findMarkers",
    "getImputed",
    "runClustering",
    "runPseudotime",
    "prepareMappingQuery",
    "runMapping",
    "makeGraphHarmony",
    "subsetZarr",
    "toH5ad",
]

CORE_STAGE_ORDER: tuple[StageName, ...] = (
    "createStore",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    "makeGraph",
    "runUmap",
    "runLeiden",
    "findMarkers",
)

OPTIONAL_STAGE_ORDER: tuple[StageName, ...] = (
    "getImputed",
    "runClustering",
    "runPseudotime",
    "prepareMappingQuery",
    "runMapping",
    "makeGraphHarmony",
    "subsetZarr",
    "toH5ad",
)

# Default when config.stages is unset (backward compatible).
STAGE_ORDER: tuple[StageName, ...] = CORE_STAGE_ORDER

FULL_STAGE_ORDER: tuple[StageName, ...] = CORE_STAGE_ORDER + OPTIONAL_STAGE_ORDER

MAX_TIMEOUT_SECONDS = 86_400


class WorkflowParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assayName: str = "RNA"
    cellKey: str = "I"
    filterAttrs: tuple[str, ...] = (
        "RNA_nCounts",
        "RNA_nFeatures",
        "RNA_percentMito",
        "RNA_percentRibo",
    )
    filterMinQuantile: float = 0.01
    filterMaxQuantile: float = 0.99
    minFeaturesPerCell: int = 10
    minCellsPerFeature: int = 20
    h5adBatchSize: int = 1000
    topN: int = 2000
    hvgMinCells: int = 20
    hvgKey: str = "hvgs"
    k: int = 11
    dims: int = 50
    nCentroids: int = 1000
    graphSeed: int = 4466
    annParallel: bool = False
    umapEpochs: int = 300
    umapSeed: int = 4444
    umapParallel: bool = False
    umapLabel: str = "UMAP"
    leidenResolution: float = 1.0
    leidenSeed: int = 4444
    leidenLabel: str = "leiden_cluster"
    markerFeatureKey: str = "I"
    markerGeneBatchSize: int | None = None
    graphLocalCache: bool | str = "auto"
    # Optional extras (countsT funnel + non-core stages).
    harmonyBatchColumn: str = "synth_batch"
    harmonyNBatches: int = 4
    harmonyBatchSeed: int = 1234
    imputeGeneCount: int = 25
    imputeDiffusionT: int = 2
    parisNClusters: int = 20
    parisLabel: str = "paris_cluster"
    # When True, cut the Paris dendrogram with BalancedCut. min/max default to
    # the observed Leiden cluster sizes on the same cell key (see paris worker).
    parisBalancedCut: bool = False
    parisMinSize: int | None = None
    parisMaxSize: int | None = None
    pseudotimeLabel: str = "pseudotime"
    mappingQueryRows: int = 25_000
    mappingTargetName: str = "query25k"
    mappingTargetFeatKey: str = "hvgs_query25k"
    mappingSaveK: int = 3
    mappingBatchSize: int = 1000

    @property
    def resolvedHvgKey(self) -> str:
        return f"{self.cellKey}__{self.hvgKey}"

    @property
    def resolvedMarkerGroupKey(self) -> str:
        if self.cellKey == "I":
            return f"{self.assayName}_{self.leidenLabel}"
        return f"{self.assayName}_{self.cellKey}_{self.leidenLabel}"

    @model_validator(mode="after")
    def _check_extras(self) -> Self:
        if self.harmonyNBatches < 2:
            raise ValueError("harmonyNBatches must be >= 2")
        if self.imputeGeneCount <= 0:
            raise ValueError("imputeGeneCount must be positive")
        if self.imputeDiffusionT <= 0:
            raise ValueError("imputeDiffusionT must be positive")
        if self.parisNClusters <= 1:
            raise ValueError("parisNClusters must be > 1")
        if self.parisMinSize is not None and self.parisMinSize < 1:
            raise ValueError("parisMinSize must be >= 1")
        if self.parisMaxSize is not None and self.parisMaxSize < 1:
            raise ValueError("parisMaxSize must be >= 1")
        if (
            self.parisMinSize is not None
            and self.parisMaxSize is not None
            and self.parisMinSize > self.parisMaxSize
        ):
            raise ValueError("parisMinSize must be <= parisMaxSize")
        if self.mappingQueryRows <= 0:
            raise ValueError("mappingQueryRows must be positive")
        if self.mappingSaveK <= 0 or self.mappingBatchSize <= 0:
            raise ValueError("mappingSaveK and mappingBatchSize must be positive")
        return self


class StageResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    modalMemoryRequestMb: int
    modalMemoryLimitMb: int
    modalCpuRequest: float
    modalCpuLimit: float
    scarfMemoryBudget: int
    workers: int
    workingCopies: int
    timeoutSeconds: int
    ephemeralDiskMb: int

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.modalMemoryRequestMb <= 0 or self.modalMemoryLimitMb <= 0:
            raise ValueError("Modal memory values must be positive")
        if self.modalMemoryLimitMb < self.modalMemoryRequestMb:
            raise ValueError("modalMemoryLimitMb must be >= modalMemoryRequestMb")
        if self.modalCpuRequest <= 0 or self.modalCpuLimit <= 0:
            raise ValueError("Modal CPU values must be positive")
        if self.modalCpuLimit < self.modalCpuRequest:
            raise ValueError("modalCpuLimit must be >= modalCpuRequest")
        if self.scarfMemoryBudget <= 0:
            raise ValueError("scarfMemoryBudget must be positive")
        if self.workers <= 0 or self.workingCopies <= 0:
            raise ValueError("workers and workingCopies must be positive")
        if not (0 < self.timeoutSeconds <= MAX_TIMEOUT_SECONDS):
            raise ValueError(f"timeoutSeconds must be in 1..{MAX_TIMEOUT_SECONDS}")
        if self.ephemeralDiskMb <= 0:
            raise ValueError("ephemeralDiskMb must be positive")
        return self


class PrepareResources(BaseModel):
    """Modal sizing for dataset prepare (full CSR resident in memory)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    modalMemoryRequestMb: int = 196_608
    modalMemoryLimitMb: int = 212_992
    modalCpuRequest: float = 8.0
    modalCpuLimit: float = 16.0
    timeoutSeconds: int = 86_400
    ephemeralDiskMb: int = 524_288

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.modalMemoryRequestMb <= 0 or self.modalMemoryLimitMb <= 0:
            raise ValueError("Modal memory values must be positive")
        if self.modalMemoryLimitMb < self.modalMemoryRequestMb:
            raise ValueError("modalMemoryLimitMb must be >= modalMemoryRequestMb")
        if self.modalCpuRequest <= 0 or self.modalCpuLimit <= 0:
            raise ValueError("Modal CPU values must be positive")
        if self.modalCpuLimit < self.modalCpuRequest:
            raise ValueError("modalCpuLimit must be >= modalCpuRequest")
        if not (0 < self.timeoutSeconds <= MAX_TIMEOUT_SECONDS):
            raise ValueError(f"timeoutSeconds must be in 1..{MAX_TIMEOUT_SECONDS}")
        if self.ephemeralDiskMb <= 0:
            raise ValueError("ephemeralDiskMb must be positive")
        return self


class StorageLayout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    targetChunkBytes: int | None = None
    minFeatureChunk: int = 500
    maxFeatureChunk: int = 10_000

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.targetChunkBytes is not None and self.targetChunkBytes <= 0:
            raise ValueError("targetChunkBytes must be positive when set")
        if self.minFeatureChunk <= 0 or self.maxFeatureChunk <= 0:
            raise ValueError("feature chunk bounds must be positive")
        if self.minFeatureChunk > self.maxFeatureChunk:
            raise ValueError("minFeatureChunk must be <= maxFeatureChunk")
        return self


class ProfilingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    modalEnvironmentName: str = "scarf_profiling"
    modalAppName: str = "scarf-profiling"
    modalSecretName: str
    modalRegion: str
    r2EndpointUrl: str
    datasetPrefixUri: str
    resultsUri: str
    runTag: str = ""
    storageLayout: StorageLayout = Field(default_factory=StorageLayout)
    targetSizes: tuple[int, ...] = Field(default_factory=lambda: DEFAULT_TARGET_SIZES)
    samplingSeed: int = 0
    workflow: WorkflowParameters = Field(default_factory=WorkflowParameters)
    prepareResources: PrepareResources = Field(default_factory=PrepareResources)
    stageResources: dict[StageName, StageResources]
    # If unset, only CORE_STAGE_ORDER runs. Extras layouts set the full list.
    stages: tuple[StageName, ...] | None = None

    @property
    def effectiveStages(self) -> tuple[StageName, ...]:
        return self.stages if self.stages is not None else CORE_STAGE_ORDER

    @model_validator(mode="after")
    def _check_config(self) -> Self:
        if self.modalEnvironmentName != "scarf_profiling":
            raise ValueError("modalEnvironmentName must be scarf_profiling")
        if not self.datasetPrefixUri.startswith("s3://"):
            raise ValueError("datasetPrefixUri must be an s3:// URI")
        if not self.resultsUri.startswith("s3://"):
            raise ValueError("resultsUri must be an s3:// URI")
        if "/" in self.runTag or "\\" in self.runTag or self.runTag in {".", ".."}:
            raise ValueError("runTag must be a single path segment")
        if not self.targetSizes:
            raise ValueError("targetSizes must not be empty")
        if any(size <= 0 for size in self.targetSizes):
            raise ValueError("targetSizes must be positive")
        if tuple(sorted(self.targetSizes)) != self.targetSizes:
            raise ValueError("targetSizes must be strictly increasing")
        if len(set(self.targetSizes)) != len(self.targetSizes):
            raise ValueError("targetSizes must be unique")
        selected = self.effectiveStages
        if not selected:
            raise ValueError("stages must not be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("stages must be unique")
        missing = [stage for stage in selected if stage not in self.stageResources]
        if missing:
            raise ValueError(f"Missing stageResources for: {', '.join(missing)}")
        if "prepareMappingQuery" in selected or "runMapping" in selected:
            query_rows = self.workflow.mappingQueryRows
            if query_rows >= min(self.targetSizes):
                raise ValueError(
                    "mappingQueryRows must be smaller than every targetSizes entry"
                )
        return self

    def datasetUri(self, nRows: int) -> str:
        return f"{self.datasetPrefixUri.rstrip('/')}/{nRows}.h5ad"

    def sourceUri(self) -> str:
        return f"{self.datasetPrefixUri.rstrip('/')}/source.h5ad"

    def _tagged_prefix(self, kind: str) -> str:
        base = f"{self.resultsUri.rstrip('/')}/{kind}"
        if self.runTag:
            return f"{base}/{self.runTag}"
        return base

    def storeUri(self, nRows: int) -> str:
        return f"{self._tagged_prefix('stores')}/{nRows}.zarr"

    def queryStoreUri(self, nRows: int) -> str:
        q = self.workflow.mappingQueryRows
        return f"{self._tagged_prefix('stores')}/{nRows}_query_{q}.zarr"

    def resultUri(self, nRows: int, stage: StageName) -> str:
        return f"{self._tagged_prefix('results')}/{nRows}/{stage}.json"

    def resourcesFor(self, stage: StageName) -> StageResources:
        return self.stageResources[stage]


def load_profiling_config(path: str | Path) -> ProfilingConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text())
    return ProfilingConfig.model_validate(_normalize_raw_config(raw))


def _normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    fixed = payload.pop("fixedResources", None)
    if fixed is not None and "stageResources" not in payload:
        stages = payload.get("stages")
        selected = tuple(stages) if stages is not None else CORE_STAGE_ORDER
        payload["stageResources"] = {stage: fixed for stage in selected}
    payload.pop("sourceProvenance", None)
    payload.pop("capacityCases", None)
    payload.pop("blockSeeds", None)
    return payload
