"""Standalone interpretation rejects stale artifacts before any model call."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from scarf.agent.biological_interpretation.contracts import (
    BiologicalInterpretationReport,
)
from scarf.agent.biological_interpretation.validation import (
    _prepare_biological_interpretation_dependencies,
    validate_biological_interpretation_report,
)
from scarf.storage.refs import ArtifactRef
from tests.test_agent_biological_interpretation import (
    FakeStore,
    artifact_model,
    context,
)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"cluster": None}, "exact cluster artifact"),
        ({"graph_assay": "Other"}, "different assay"),
        ({"max_clusters": 0}, "max_clusters"),
        ({"max_markers": 0}, "max_markers"),
        ({"marker_min_score": 0}, "marker_min_score"),
        ({"marker_min_fraction": 2}, "marker_min_fraction"),
        ({"allow_marker_search": True, "marker": None}, "marker_features is required"),
        (
            {
                "marker": ArtifactRef(
                    scope="assay",
                    assay="RNA",
                    kind="feature_selection",
                    artifact_id="b" * 64,
                )
            },
            "marker_table",
        ),
        (
            {
                "marker_features": ArtifactRef(
                    scope="assay",
                    assay="RNA",
                    kind="marker_table",
                    artifact_id="b" * 64,
                )
            },
            "feature_selection",
        ),
        (
            {
                "cluster": ArtifactRef(
                    scope="assay",
                    assay="RNA",
                    kind="marker_table",
                    artifact_id="b" * 64,
                )
            },
            "cluster_labels or cluster_cut",
        ),
        (
            {
                "marker_features": ArtifactRef(
                    scope="assay",
                    assay="Other",
                    kind="feature_selection",
                    artifact_id="b" * 64,
                )
            },
            "different assay",
        ),
    ],
)
def test_interpretation_input_contract_rejects_unsupported_work(
    change: dict[str, Any],
    reason: str,
) -> None:
    store = FakeStore()
    arguments = dict(
        cluster=store.cluster,
        from_assay=None,
        graph_assay=None,
        marker_assay_type=None,
        sample_column=None,
        condition_column=None,
        tuning_handoff=None,
        experimental_handoff=None,
        marker=store.marker,
        marker_features=None,
        allow_marker_search=False,
        max_clusters=8,
        max_markers=10,
        marker_min_score=0.1,
        marker_min_fraction=0.1,
    )
    with pytest.raises((TypeError, ValueError), match=reason):
        _prepare_biological_interpretation_dependencies(store, **(arguments | change))
    assert store.marker_calls == 0


@pytest.mark.parametrize(
    ("artifact", "fields", "reason"),
    [
        ("cluster", {"exists": False}, "cluster artifact does not exist"),
        ("cluster", {"complete": False}, "cluster artifact is incomplete"),
        ("cluster", {"inputs": {}}, "no cell-selection input"),
        ("marker", {"exists": False}, "marker artifact does not exist"),
        ("marker", {"complete": False}, "marker artifact is incomplete"),
        ("marker", {"inputs": {}}, "exact cluster artifact"),
    ],
)
def test_interpretation_artifact_status_is_validated_before_model_execution(
    monkeypatch: Any,
    artifact: str,
    fields: dict[str, Any],
    reason: str,
) -> None:
    store = FakeStore()
    inspect = store.inspect_artifact
    damaged_ref = getattr(store, artifact)

    def damaged(ref: ArtifactRef) -> Any:
        status = inspect(ref)
        if ref != damaged_ref:
            return status
        return SimpleNamespace(
            **(
                {
                    "exists": status.exists,
                    "complete": status.complete,
                    "inputs": status.inputs,
                }
                | fields
            )
        )

    monkeypatch.setattr(store, "inspect_artifact", damaged)
    with pytest.raises(ValueError, match=reason):
        _prepare_biological_interpretation_dependencies(
            store,
            cluster=store.cluster,
            from_assay=None,
            graph_assay=None,
            marker_assay_type=None,
            sample_column=None,
            condition_column=None,
            tuning_handoff=None,
            experimental_handoff=None,
            marker=store.marker,
            marker_features=None,
            allow_marker_search=False,
            max_clusters=8,
            max_markers=10,
            marker_min_score=0.1,
            marker_min_fraction=0.1,
        )
    assert store.marker_calls == 0


@pytest.mark.parametrize("field", ["clusterArtifact", "markerArtifact"])
def test_interpretation_model_cannot_replace_exact_artifact_bindings(
    field: str,
) -> None:
    store = FakeStore()
    deps = context(store, marker=store.marker).deps
    deps.clusterValues = ["0"]
    reference = store.cluster if field == "clusterArtifact" else store.marker
    changed = artifact_model(reference).model_copy(update={"artifactId": "f" * 64})
    report = BiologicalInterpretationReport.model_validate(
        {"status": "done", field: changed}
    )
    with pytest.raises(ModelRetry, match=f"{field} does not match"):
        validate_biological_interpretation_report(report, deps)
