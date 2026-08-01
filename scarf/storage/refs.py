import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

type ArtifactScope = Literal["assay", "datastore"]

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ARTIFACT_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ARTIFACT_KINDS = frozenset(
    {
        "ann_index",
        "batch_correction",
        "cell_cycle",
        "cell_selection",
        "cluster_cut",
        "cluster_hierarchy",
        "cluster_labels",
        "coalesced_tree",
        "connectivity_map",
        "dendrogram",
        "diffusion_operator",
        "doublet_score",
        "embedding",
        "embedding_initialization",
        "enrichment_scores",
        "fate_map",
        "feature_scaling",
        "feature_selection",
        "feature_summary",
        "hto_identity",
        "integrated_graph",
        "intersection_ann_index",
        "imported_coordinates",
        "mapping_reference",
        "marker_table",
        "membership_strength",
        "metadata_snapshot",
        "neighbors",
        "normalized",
        "projection",
        "pseudotime",
        "pseudotime_aggregation",
        "pseudotime_markers",
        "quality_metric",
        "reduction",
        "sampling",
        "smart_label",
        "wnn_coordinates",
    }
)


def _validate_name(value: str, label: str) -> None:
    if _NAME_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a snake_case identifier, got {value!r}")


def _validate_artifact_kind(kind: str) -> None:
    _validate_name(kind, "kind")
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"Unknown artifact kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    scope: ArtifactScope
    kind: str
    artifact_id: str
    assay: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"assay", "datastore"}:
            raise ValueError(f"Invalid artifact scope: {self.scope!r}")
        _validate_artifact_kind(self.kind)
        if _ARTIFACT_ID_PATTERN.fullmatch(self.artifact_id) is None:
            raise ValueError("artifact_id must be a 64-character lowercase hex token")
        if self.scope == "assay":
            if self.assay is None or not self.assay or "/" in self.assay:
                raise ValueError("assay-scoped artifact references require an assay")
        elif self.assay is not None:
            raise ValueError("datastore-scoped artifact references cannot set assay")
        if self.kind == "imported_coordinates" and self.scope != "assay":
            raise ValueError("imported_coordinates artifacts must be assay-scoped")

    def __repr__(self) -> str:
        location = f"assay={self.assay!r}" if self.assay is not None else "datastore"
        return (
            f"ArtifactRef({location}, kind={self.kind!r}, "
            f"artifact_id='{self.artifact_id[:12]}...')"
        )

    def to_dict(self) -> dict[str, str]:
        value = {
            "type": "artifact",
            "scope": self.scope,
            "kind": self.kind,
            "artifact_id": self.artifact_id,
        }
        if self.assay is not None:
            value["assay"] = self.assay
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        if value.get("type") != "artifact":
            raise ValueError("Artifact reference type must be 'artifact'")
        scope = value.get("scope")
        if scope not in {"assay", "datastore"}:
            raise ValueError(f"Invalid artifact scope: {scope!r}")
        kind = value.get("kind")
        artifact_id = value.get("artifact_id")
        assay = value.get("assay")
        if not isinstance(kind, str) or not isinstance(artifact_id, str):
            raise TypeError("Artifact reference kind and artifact_id must be strings")
        if assay is not None and not isinstance(assay, str):
            raise TypeError("Artifact reference assay must be a string or null")
        return cls(
            scope=scope,
            assay=assay,
            kind=kind,
            artifact_id=artifact_id,
        )


def artifact_path(ref: ArtifactRef) -> str:
    if ref.scope == "assay":
        return f"{ref.assay}/artifacts/{ref.kind}/{ref.artifact_id}"
    return f"artifacts/{ref.kind}/{ref.artifact_id}"


def parse_artifact_path(path: str) -> ArtifactRef:
    parts = path.strip("/").split("/")
    if len(parts) == 3 and parts[0] == "artifacts":
        return ArtifactRef(
            scope="datastore",
            kind=parts[1],
            artifact_id=parts[2],
        )
    if len(parts) == 4 and parts[1] == "artifacts":
        return ArtifactRef(
            scope="assay",
            assay=parts[0],
            kind=parts[2],
            artifact_id=parts[3],
        )
    raise ValueError(f"Not an artifact path: {path!r}")
