"""Config for the Scanpy out-of-core e2e profiling sibling."""

import math
import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from profiling.datasets import DEFAULT_TARGET_SIZES

ScanpyStageName = Literal[
    "loadLazy",
    "calculateQc",
    "filterCells",
    "normalizeTotal",
    "log1p",
    "markHvgs",
    "runPca",
    "runNeighbors",
    "runUmap",
    "runLeiden",
    "rankGenesGroups",
]

SCANPY_STAGE_ORDER: tuple[ScanpyStageName, ...] = (
    "loadLazy",
    "calculateQc",
    "filterCells",
    "normalizeTotal",
    "log1p",
    "markHvgs",
    "runPca",
    "runNeighbors",
    "runUmap",
    "runLeiden",
    "rankGenesGroups",
)

MAX_TIMEOUT_SECONDS = 86_400


class ScanpyWorkflowParameters(BaseModel):
    """Analysis knobs aligned with Scarf e2e where possible.

    Normalization uses Scanpy's published ``target_sum=1e4`` (not Scarf's sf=1000).
    Cell filtering approximates Scarf's 1st/99th quantile auto-filter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    nTopGenes: int = 2000
    nComps: int = 50
    nNeighbors: int = 11
    umapEpochs: int = 300
    umapSeed: int = 4444
    leidenResolution: float = 1.0
    leidenSeed: int = 4444
    leidenNIterations: int = 2
    leidenFlavor: Literal["igraph", "leidenalg"] = "igraph"
    leidenKeyAdded: str = "leiden"
    targetSum: float = 1e4
    filterMinQuantile: float = 0.01
    filterMaxQuantile: float = 0.99
    minGenesPerCell: int = 10
    minCellsPerGene: int = 20
    mitoPattern: str = r"(?i)^mt-"
    riboPattern: str = r"(?i)^rp[sl]"
    hvgFlavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = (
        "seurat"
    )
    neighborsTransformer: Literal["annoy", "default"] = "annoy"
    rankMethod: Literal["wilcoxon", "t-test", "t-test_overestim_var", "logreg"] = (
        "wilcoxon"
    )
    rankNGenes: int | None = None
    featureNameKey: str = "feature_name"

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.nTopGenes <= 0:
            raise ValueError("nTopGenes must be positive")
        if self.nComps <= 0:
            raise ValueError("nComps must be positive")
        if self.nNeighbors <= 1:
            raise ValueError("nNeighbors must be > 1")
        if self.umapEpochs <= 0:
            raise ValueError("umapEpochs must be positive")
        if not math.isfinite(self.leidenResolution) or self.leidenResolution <= 0:
            raise ValueError("leidenResolution must be positive")
        if self.leidenNIterations < -1:
            raise ValueError("leidenNIterations must be >= -1")
        if not math.isfinite(self.targetSum) or self.targetSum <= 0:
            raise ValueError("targetSum must be positive")
        if not 0 <= self.filterMinQuantile < self.filterMaxQuantile <= 1:
            raise ValueError("filter quantiles must satisfy 0 <= min < max <= 1")
        if self.minGenesPerCell < 0 or self.minCellsPerGene < 0:
            raise ValueError("minGenesPerCell and minCellsPerGene must be >= 0")
        if self.rankNGenes is not None and self.rankNGenes <= 0:
            raise ValueError("rankNGenes must be positive when set")
        return self


class ScanpyDaskParameters(BaseModel):
    """Dask LocalCluster knobs. Tunable for the 8 CPU / 32 GiB box."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nWorkers: int = 1
    threadsPerWorker: int = 1
    memoryPerWorker: str = "50GB"
    sparseChunkSize: int = 20_000
    processes: bool = True
    deathTimeoutSeconds: float = 120.0
    dashboardAddress: str | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.nWorkers <= 0:
            raise ValueError("nWorkers must be positive")
        if self.threadsPerWorker <= 0:
            raise ValueError("threadsPerWorker must be positive")
        if self.sparseChunkSize <= 0:
            raise ValueError("sparseChunkSize must be positive")
        if self.deathTimeoutSeconds <= 0:
            raise ValueError("deathTimeoutSeconds must be positive")
        return self


class ScanpyModalResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    modalMemoryRequestMb: int = 65_536
    modalMemoryLimitMb: int = 65_536
    modalCpuRequest: float = 8.0
    modalCpuLimit: float = 8.0
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


class ScanpyProfilingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    modalEnvironmentName: str = "scarf_profiling"
    modalAppName: str = "scarf-profiling-scanpy"
    modalSecretName: str
    modalRegion: str
    r2EndpointUrl: str
    datasetPrefixUri: str
    resultsUri: str
    runTag: str = ""
    targetSizes: tuple[int, ...] = Field(default_factory=lambda: DEFAULT_TARGET_SIZES)
    workflow: ScanpyWorkflowParameters = Field(
        default_factory=ScanpyWorkflowParameters
    )
    dask: ScanpyDaskParameters = Field(default_factory=ScanpyDaskParameters)
    resources: ScanpyModalResources = Field(default_factory=ScanpyModalResources)

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
        return self

    def datasetUri(self, nRows: int) -> str:
        return f"{self.datasetPrefixUri.rstrip('/')}/{nRows}.h5ad"

    def _tagged_prefix(self, kind: str) -> str:
        base = f"{self.resultsUri.rstrip('/')}/{kind}"
        if self.runTag:
            return f"{base}/{self.runTag}"
        return base

    def resultUri(self, nRows: int, stage: ScanpyStageName) -> str:
        return f"{self._tagged_prefix('results')}/{nRows}/{stage}.json"

    def funnelResultUri(self, nRows: int) -> str:
        return f"{self._tagged_prefix('results')}/{nRows}/funnel.json"

    def e2eClaimUri(self) -> str:
        return f"{self._tagged_prefix('results')}/e2e-claim.json"


def load_scanpy_profiling_config(path: str | Path) -> ScanpyProfilingConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text())
    return ScanpyProfilingConfig.model_validate(raw)
