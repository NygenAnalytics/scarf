import math
import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from profiling.datasets import DEFAULT_TARGET_SIZES

StageName = Literal[
    "createStore",
    "writeCountsT",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "runUmap",
    "runLeiden",
    "runClustering",
    "findMarkers",
    "importClusters",
    "validateExperiment",
]

GRAPH_CONSTRUCTION_STAGE_ORDER: tuple[StageName, ...] = (
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
)

CORE_STAGE_ORDER: tuple[StageName, ...] = (
    "createStore",
    "writeCountsT",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    *GRAPH_CONSTRUCTION_STAGE_ORDER,
    "runUmap",
    "runLeiden",
    "runClustering",
    "findMarkers",
)

SELECTED_STAGE_ORDER: tuple[StageName, ...] = (
    "createStore",
    "writeCountsT",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "importClusters",
    "findMarkers",
    "validateExperiment",
)

SELECTED_STAGE_DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    "createStore": (),
    "writeCountsT": ("createStore",),
    "initializeStore": ("writeCountsT",),
    "reopenStore": ("initializeStore",),
    "filterCells": ("reopenStore",),
    "markHvgs": ("filterCells",),
    "runNormalization": ("markHvgs",),
    "runPca": ("runNormalization",),
    "importClusters": ("filterCells",),
    "findMarkers": ("importClusters",),
    "validateExperiment": ("runPca", "findMarkers"),
}

ALL_STAGE_CHOICES: tuple[StageName, ...] = tuple(
    dict.fromkeys((*CORE_STAGE_ORDER, *SELECTED_STAGE_ORDER))
)


def validate_requested_stages(stages: tuple[StageName, ...]) -> None:
    if not stages:
        raise ValueError("stages must not be empty")
    if len(set(stages)) != len(stages):
        raise ValueError("stages must be unique")
    selected = set(stages)
    positions = {stage: index for index, stage in enumerate(stages)}
    core_set = set(CORE_STAGE_ORDER)
    if selected <= core_set:
        core_index = {stage: index for index, stage in enumerate(CORE_STAGE_ORDER)}
        ordered = tuple(sorted(stages, key=lambda stage: core_index[stage]))
        if stages != ordered:
            raise ValueError("CORE stages must appear in CORE_STAGE_ORDER")
        return
    for stage in stages:
        deps = SELECTED_STAGE_DEPENDENCIES.get(stage)
        if deps is None:
            raise ValueError(f"{stage} is not in the selected-stage graph")
        for dep in deps:
            if dep not in selected:
                raise ValueError(f"{stage} requires {dep}")
            if positions[dep] >= positions[stage]:
                raise ValueError(f"{dep} must precede {stage}")


MAX_TIMEOUT_SECONDS = 86_400
CLUSTER_SOURCES_PATH = Path(__file__).resolve().parent / "cluster_sources.toml"


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
    k: int = 17
    dims: int = 50
    nCentroids: int = 1000
    graphSeed: int = 4466
    kmeansSampling: float = 0.1
    kmeansBatchSize: int = 10_000
    annParallel: bool = False
    umapEpochs: int = 300
    umapSeed: int = 4444
    umapParallel: bool = False
    umapLabel: str = "UMAP"
    leidenResolution: float = 1.0
    leidenSeed: int = 4444
    leidenLabel: str = "leiden_cluster"
    markerFeatureKey: str = "I"
    graphLocalCache: bool | str = "auto"
    parisNClusters: int | Literal["auto"] = "auto"
    parisLabel: str = "paris_cluster"
    parisMinClusterSize: int | None = None
    clusterSourceUri: str | None = None
    clusterLabelColumn: str = "RNA_leiden_cluster"

    @property
    def resolvedHvgKey(self) -> str:
        return f"{self.cellKey}__{self.hvgKey}"

    @property
    def resolvedMarkerGroupKey(self) -> str:
        if self.cellKey == "I":
            return f"{self.assayName}_{self.leidenLabel}"
        return f"{self.assayName}_{self.cellKey}_{self.leidenLabel}"

    @model_validator(mode="after")
    def _check_workflow(self) -> Self:
        if not math.isfinite(self.kmeansSampling) or not 0 < self.kmeansSampling <= 1:
            raise ValueError("kmeansSampling must be greater than 0 and at most 1")
        if self.kmeansBatchSize <= 0:
            raise ValueError("kmeansBatchSize must be positive")
        if self.parisNClusters != "auto" and self.parisNClusters <= 1:
            raise ValueError("parisNClusters must be > 1")
        if self.parisMinClusterSize is not None:
            if self.parisMinClusterSize < 2:
                raise ValueError("parisMinClusterSize must be >= 2")
            if self.parisNClusters != "auto":
                raise ValueError("parisMinClusterSize requires parisNClusters='auto'")
        if self.clusterSourceUri is not None:
            uri = self.clusterSourceUri.strip()
            if not uri:
                raise ValueError("clusterSourceUri must be non-empty when set")
            if not (
                uri.startswith("s3://")
                or uri.startswith("/")
                or uri.startswith("file://")
            ):
                raise ValueError("clusterSourceUri must be an s3:// URI or local path")
        if not self.clusterLabelColumn:
            raise ValueError("clusterLabelColumn must be non-empty")
        return self


class StageResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    modalMemoryRequestMb: int
    modalMemoryLimitMb: int
    modalCpuRequest: float
    modalCpuLimit: float
    scarfMemoryBudget: int
    workers: int
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
        if self.workers <= 0:
            raise ValueError("workers must be positive")
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


class CountMatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unitBytes: int = 1_000_000_000
    chunkBytes: int = 100_000_000

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if min(self.unitBytes, self.chunkBytes) < 1:
            raise ValueError("countMatrix values must be positive")
        if self.unitBytes < self.chunkBytes:
            raise ValueError("unitBytes must be at least chunkBytes")
        return self


class StorageIoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    readWorkers: int | None = None
    computeWorkers: int | None = None
    writeWorkers: int | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        for name in ("readWorkers", "computeWorkers", "writeWorkers"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive when set")
        return self


class ClusterSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    nRows: int
    storeUri: str
    labelColumn: str = "RNA_leiden_cluster"

    @model_validator(mode="after")
    def _check_ref(self) -> Self:
        if self.nRows <= 0:
            raise ValueError("cluster source nRows must be positive")
        if not self.storeUri.startswith("s3://"):
            raise ValueError("cluster source storeUri must be an s3:// URI")
        if not self.labelColumn:
            raise ValueError("cluster source labelColumn must be non-empty")
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
    # When set, stage jobs read/write this store instead of stores/{runTag}/...
    # Useful for consume A/B runs against an existing store with a fresh result tag.
    storeUriOverride: str | None = None
    countMatrix: CountMatrixConfig | None = None
    storageIo: StorageIoConfig | None = None
    clusterSources: tuple[ClusterSourceRef, ...] = ()
    targetSizes: tuple[int, ...] = Field(default_factory=lambda: DEFAULT_TARGET_SIZES)
    samplingSeed: int = 0
    workflow: WorkflowParameters = Field(default_factory=WorkflowParameters)
    prepareResources: PrepareResources = Field(default_factory=PrepareResources)
    stageResources: dict[StageName, StageResources]
    # If unset, the complete current funnel runs.
    stages: tuple[StageName, ...] | None = None
    # Filled by the submitting client before Modal spawn; not a TOML setting.
    clientProvenance: dict[str, Any] | None = None

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
        if self.storeUriOverride is not None:
            override = self.storeUriOverride.strip()
            if not override:
                raise ValueError("storeUriOverride must be non-empty when set")
            if not (
                override.startswith("s3://")
                or override.startswith("/")
                or override.startswith("file://")
            ):
                raise ValueError(
                    "storeUriOverride must be an s3:// URI or a local filesystem path"
                )
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
        validate_requested_stages(selected)
        cluster_sizes = [item.nRows for item in self.clusterSources]
        if len(set(cluster_sizes)) != len(cluster_sizes):
            raise ValueError("clusterSources nRows must be unique")
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
        if self.storeUriOverride is not None:
            return self.storeUriOverride.rstrip("/")
        return f"{self._tagged_prefix('stores')}/{nRows}.zarr"

    def resultUri(self, nRows: int, stage: StageName) -> str:
        return f"{self._tagged_prefix('results')}/{nRows}/{stage}.json"

    def funnelResultUri(self, nRows: int) -> str:
        return f"{self._tagged_prefix('results')}/{nRows}/funnel.json"

    def e2eClaimUri(self) -> str:
        return f"{self._tagged_prefix('results')}/e2e-claim.json"

    def resourcesFor(self, stage: StageName) -> StageResources:
        return self.stageResources[stage]

    def clusterSourceFor(self, nRows: int) -> ClusterSourceRef | None:
        for item in self.clusterSources:
            if item.nRows == nRows:
                return item
        return None


def load_profiling_config(path: str | Path) -> ProfilingConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text())
    config = ProfilingConfig.model_validate(_normalize_raw_config(raw))
    if config.clusterSources or not CLUSTER_SOURCES_PATH.is_file():
        return config
    extra = tomllib.loads(CLUSTER_SOURCES_PATH.read_text())
    refs = tuple(
        ClusterSourceRef.model_validate(item)
        for item in extra.get("clusterSources", [])
    )
    if not refs:
        return config
    return config.model_copy(update={"clusterSources": refs})


def bind_cluster_source(config: ProfilingConfig, nRows: int) -> WorkflowParameters:
    if config.workflow.clusterSourceUri:
        return config.workflow
    source = config.clusterSourceFor(nRows)
    if source is None:
        return config.workflow
    return config.workflow.model_copy(
        update={
            "clusterSourceUri": source.storeUri,
            "clusterLabelColumn": source.labelColumn,
        }
    )


def _normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    fixed = payload.pop("fixedResources", None)
    if fixed is not None and "stageResources" not in payload:
        stages = payload.get("stages")
        selected = tuple(stages) if stages is not None else CORE_STAGE_ORDER
        payload["stageResources"] = {stage: fixed for stage in selected}
    return payload
