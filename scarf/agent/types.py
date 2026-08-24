"""Shared result types for scarf.agent stages and decide()."""

from typing import Literal

from ._deps import AGENT_INSTALL_HINT

try:
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


type StageStatus = Literal["done", "needsInput", "failed"]


class EvidenceItem(BaseModel):
    id: str
    label: str
    summary: str


class Decision(BaseModel):
    selectedId: str = Field(
        description="Exact evidence id string from the provided list, nothing else"
    )
    rationale: str = Field(description="Short reason for the choice")
    evidenceIds: list[str] = Field(
        default_factory=list,
        description="Evidence ids used; must include selectedId and only provided ids",
    )


class NeedsInput(BaseModel):
    question: str
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)


class StageResult(BaseModel):
    status: StageStatus
    decision: Decision | None = None
    needsInput: NeedsInput | None = None
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
