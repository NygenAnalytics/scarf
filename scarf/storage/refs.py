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
        "cluster_selection",
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
        "imported_coordinates",
        "label_transfer",
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
        "reference_labels",
        "sampling",
        "smart_label",
        "statistical_tests",
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
        if not isinstance(value, Mapping):
            raise TypeError("Artifact reference must be a mapping")
        if value.get("type") != "artifact":
            raise ValueError("Artifact reference type must be 'artifact'")
        scope = value.get("scope")
        if scope not in {"assay", "datastore"}:
            raise ValueError(f"Invalid artifact scope: {scope!r}")
        expected_keys = {"type", "scope", "kind", "artifact_id"}
        if scope == "assay":
            expected_keys.add("assay")
        if set(value) != expected_keys:
            raise ValueError(
                "Artifact reference fields do not match its declared scope"
            )
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


@dataclass(frozen=True, slots=True)
class ExternalArtifactRef:
    """One exact artifact held by another datastore.

    ``dataset_fingerprint`` names the dataset whose axes the artifact is
    aligned to: the prepared dataset fingerprint of one of its assays, the
    anchor assay. It names a dataset, not a datastore, so a mount and its
    source share it, and the caller supplies the datastore that holds the
    artifact. The anchor is the artifact's own assay unless ``anchor_assay``
    names another one, which a datastore-scoped artifact, or an artifact of a
    different assay, must do.
    """

    dataset_fingerprint: str
    ref: ArtifactRef
    anchor_assay: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_fingerprint, str):
            raise TypeError("dataset_fingerprint must be a string")
        if not self.dataset_fingerprint:
            raise ValueError("dataset_fingerprint must be non-empty")
        if not isinstance(self.ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if self.anchor_assay is None:
            if self.ref.scope != "assay" or self.ref.assay is None:
                raise ValueError(
                    "External artifact references require an assay-scoped "
                    "ArtifactRef or an anchor_assay"
                )
            return
        if not isinstance(self.anchor_assay, str):
            raise TypeError("anchor_assay must be a string or None")
        if not self.anchor_assay or "/" in self.anchor_assay:
            raise ValueError("anchor_assay must be an assay name")
        if self.anchor_assay == self.ref.assay:
            raise ValueError(
                "anchor_assay is set only when it differs from the artifact's assay"
            )

    @property
    def fingerprint_assay(self) -> str:
        """The assay of the other datastore that ``dataset_fingerprint`` describes."""
        assay = self.ref.assay if self.anchor_assay is None else self.anchor_assay
        assert assay is not None
        return assay

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": "external_artifact",
            "dataset_fingerprint": self.dataset_fingerprint,
            "ref": self.ref.to_dict(),
        }
        if self.anchor_assay is not None:
            value["anchor_assay"] = self.anchor_assay
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalArtifactRef":
        if not isinstance(value, Mapping):
            raise TypeError("External artifact reference must be a mapping")
        required_keys = {"type", "dataset_fingerprint", "ref"}
        if set(value) not in (required_keys, required_keys | {"anchor_assay"}):
            raise ValueError(
                "External artifact reference must contain exactly "
                "'type', 'dataset_fingerprint', and 'ref', and may add "
                "'anchor_assay'"
            )
        if value["type"] != "external_artifact":
            raise ValueError(
                "External artifact reference type must be 'external_artifact'"
            )
        dataset_fingerprint = value["dataset_fingerprint"]
        if not isinstance(dataset_fingerprint, str):
            raise TypeError("dataset_fingerprint must be a string")
        raw_ref = value["ref"]
        if not isinstance(raw_ref, Mapping):
            raise TypeError("External artifact ref must be a mapping")
        anchor_assay = value.get("anchor_assay")
        if "anchor_assay" in value and not isinstance(anchor_assay, str):
            raise TypeError("anchor_assay must be a string")
        return cls(
            dataset_fingerprint=dataset_fingerprint,
            ref=ArtifactRef.from_dict(raw_ref),
            anchor_assay=anchor_assay,
        )


type ArtifactLocator = ArtifactRef | ExternalArtifactRef


def artifact_path(ref: ArtifactRef) -> str:
    if ref.scope == "assay":
        return f"{ref.assay}/artifacts/{ref.kind}/{ref.artifact_id}"
    return f"artifacts/{ref.kind}/{ref.artifact_id}"
