"""Small helpers shared by Scarf domain-agent tools."""

from collections.abc import Iterable
from typing import Any

from ..types import ArtifactReferenceModel

__all__ = [
    "artifact_reference",
    "bounded_list",
    "core_artifact_reference",
]


def bounded_list(values: Iterable[Any], *, limit: int) -> list[Any]:
    """Return at most ``limit`` JSON-facing values."""
    if isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")
    output: list[Any] = []
    for value in values:
        output.append(value)
        if len(output) == limit:
            break
    return output


def artifact_reference(ref: Any) -> ArtifactReferenceModel:
    """Convert a core artifact reference into an agent Pydantic model."""
    if isinstance(ref, ArtifactReferenceModel):
        return ArtifactReferenceModel.model_validate(ref.model_dump())
    return ArtifactReferenceModel.from_artifact_ref(ref)


def core_artifact_reference(ref: Any) -> Any:
    """Convert an agent artifact model back to Scarf's exact core reference."""
    if not isinstance(ref, ArtifactReferenceModel):
        return ref
    from ...storage.refs import ArtifactRef

    return ArtifactRef(
        scope=ref.scope,
        kind=ref.kind,
        artifact_id=ref.artifactId,
        assay=ref.assay,
    )
