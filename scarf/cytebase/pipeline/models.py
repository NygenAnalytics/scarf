"""CamelCase requests, source inspections and flat dataset records."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Manifest(BaseModel):
    collectionId: UUID
    datasetId: UUID
    datasetVersionId: UUID
    title: str | None = None
    citation: str | None = None
    doi: str | None = None
    sourceUrl: str
    sourceBytes: int
    sourceSha256: str
    schemaVersion: str | None = None
    organism: str | None = None
    nObs: int
    nVars: int
    countsLocation: Literal["X", "raw/X", "layers/counts", "layers/raw_counts", "none"]
    apiRawDataLocation: str | None = None
    countsSelectionSource: Literal["curation_api", "inspection"] | None = None
    countsDtype: str | None = None
    countsIntegerLike: bool | None = None
    countsMax: float | None = None
    countsSampleMax: float | None = None
    countsValidationMode: Literal["sampled_rows"] | None = None
    isPrimaryDataCounts: dict[str, int]
    featureIdKey: str
    featureNameKey: str
    # Missing counts or feature metadata can produce a needsInput inspection.
    featureAttrsKey: str | None = None
    selectionDiagnostics: dict = Field(default_factory=dict)
    selectionNeedsInput: dict | None = None
    embeddings: list[str]
    metadataSource: Literal["curation_api", "notes_only"]
    ingestedAt: datetime
    pipelineVersion: str


DatasetState = Literal[
    "registered",
    "processing",
    "downloaded",
    "ready",
    "failed",
    "update_available",
    "needsInput",
]


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collectionIds: list[UUID] = Field(min_length=1)


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cytebaseIds: list[str] | None = Field(default=None, min_length=1)
    collectionId: UUID | None = None
    collectionIds: list[UUID] | None = Field(default=None, min_length=1)
    force: bool = False
    approvedDeletionPaths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_one_selector(self) -> Self:
        if (
            sum(
                value is not None
                for value in (self.cytebaseIds, self.collectionId, self.collectionIds)
            )
            != 1
        ):
            raise ValueError(
                "Supply exactly one of cytebaseIds, collectionId, or collectionIds"
            )
        return self


class FacetTerm(BaseModel):
    termId: str | None = None
    label: str


class DatasetVersion(BaseModel):
    datasetVersionId: UUID
    seenAt: datetime
    processedAt: datetime | None = None
    sourceSha256: str | None = None


class DatasetRecord(BaseModel):
    """Latest registered metadata and the identity of the verified stored files."""

    cytebaseId: str = Field(pattern=r"^[a-z0-9_]{1,80}$")
    datasetId: UUID
    collectionId: UUID
    latestVersionId: UUID
    processedVersionId: UUID | None = None
    versions: list[DatasetVersion] = Field(default_factory=list)
    title: str | None = None
    citation: str | None = None
    doi: str | None = None
    firstAuthor: str | None = None
    year: int | None = None
    facets: dict[str, list[FacetTerm]] = Field(default_factory=dict)
    cellCount: int | None = None
    primaryCellCount: int | None = None
    nGenes: int | None = None
    schemaVersion: str | None = None
    status: DatasetState = "registered"
    sourceUrl: str
    sourceBytes: int | None = None
    zarrUri: str | None = None
    h5adUri: str | None = None
    cellxgeneUrl: str
    explorerUrl: str | None = None
    registeredAt: datetime
    updatedAt: datetime
    processedAt: datetime | None = None
    pipelineVersion: str
    inspection: Manifest | None = None
    # Populated only after a build passes remote verification.
    buildReceipt: dict | None = None

    stage: str | None = None
    stageOutcome: str | None = None
    runId: str | None = None
    callId: str | None = None
    attempt: int = Field(default=0, ge=0)
    startedAt: datetime | None = None
    error: str | None = None
    needsInput: dict | None = None
    timings: dict[str, float] = Field(default_factory=dict)
