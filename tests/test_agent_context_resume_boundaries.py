"""Committed context decisions and bounded details retain exact scientific inputs."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelRetry, UnexpectedModelBehavior

from scarf.agent.experimental_context import agent as context_agent
from scarf.agent.experimental_context import tools
from scarf.agent.experimental_context.contracts import (
    CellQcProfileEvidence,
    CovariateCharacterization,
    CovariateEvidence,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    QcMetricSourceEvidence,
    NamedArtifactSource,
)
from scarf.storage.refs import ArtifactRef
from scarf.agent.types import AgentRunInfo, ArtifactReferenceModel, ToolCallInfo
from tests.test_agent_design_comparisons import _design, _proposal
from tests.test_agent_experimental_context import _Store


def test_committed_context_result_replays_without_a_model_and_rejects_other_selection(
    monkeypatch,
):
    store = _Store()
    selected = ArtifactReferenceModel.from_artifact_ref(store.cell_selection)
    report = ExperimentalContextResult.get_blank().model_copy(
        update={"cellSelection": selected}
    )
    monkeypatch.setattr(
        context_agent, "_derive_missing_percentage_artifacts", lambda *a, **k: []
    )
    monkeypatch.setattr(
        context_agent,
        "run_agent_sync",
        lambda **k: pytest.fail("Committed result must not invoke a model"),
    )
    agent = context_agent.ExperimentalContextAgent(object())
    result = agent.run(
        store,
        cell_selection=store.cell_selection,
        checkpoint_read=lambda key: (
            {"report": report.model_dump(mode="json")} if key == "result" else None
        ),
    )
    assert result == report
    report.cellSelection = selected.model_copy(update={"artifactId": "f" * 64})
    with pytest.raises(ValueError, match="different cell selection"):
        agent.run(
            store,
            cell_selection=store.cell_selection,
            checkpoint_read=lambda key: (
                {"report": report.model_dump(mode="json")} if key == "result" else None
            ),
        )


@pytest.mark.parametrize("rounds", [1, 2])
@pytest.mark.parametrize("previous_status", ["done", "needsInput"])
@pytest.mark.parametrize(
    "objective",
    ["", "Discover populations while preserving condition."],
    ids=["without-objective", "with-objective"],
)
def test_explicit_context_revision_preserves_rounds_and_complete_study_text(
    monkeypatch, rounds, previous_status, objective
):
    store = _Store()
    previous = ExperimentalContextResult.get_blank().model_copy(
        update={
            "status": previous_status,
            "cellSelection": ArtifactReferenceModel.from_artifact_ref(
                store.cell_selection
            ),
            "characterization": CovariateCharacterization(
                status="done", columns=[{"name": "condition"}, {"name": "batch"}]
            ),
            "runInfo": AgentRunInfo(
                agentName="context",
                modelName="test",
                toolCalls=[
                    ToolCallInfo(toolName="analyze_experimental_design")
                    for _ in range(rounds)
                ],
            ),
        }
    )
    blocker = "Confirm whether the supplied batch identifies physical capture."
    if previous_status == "needsInput":
        previous.decision.needsInput = [blocker]
    saved_previous = previous.model_dump(mode="json")
    text = "Study provenance. " * 150 + "Assess condition and batch jointly."
    monkeypatch.setattr(
        context_agent, "_derive_missing_percentage_artifacts", lambda *a, **k: []
    )
    monkeypatch.setattr(
        context_agent,
        "characterize_covariates",
        lambda *a, **k: pytest.fail("A revision must reuse measured characterization"),
    )
    seen = []

    def inspect(**kwargs):
        deps = kwargs["deps"]
        seen.append(deps.designRounds)
        assert deps.characterization == previous.characterization
        assert text in kwargs["user_prompt"]
        assert "Committed evidence already measured" in kwargs["user_prompt"]
        assert "Explicit requested questions" in kwargs["user_prompt"]
        assert (
            f"Study objective: {objective or 'not provided'}" in kwargs["user_prompt"]
        )
        if previous_status == "needsInput":
            prompt = kwargs["user_prompt"]
            assert blocker in prompt
            assert "Keep genuinely essential unresolved questions" in prompt
            evidence_json = prompt.split(
                "Current objective requirements and their measured support: ", 1
            )[1].split("\nThe previous interpretation stopped", 1)[0]
            evidence = json.loads(evidence_json)
            if objective:
                joint = next(
                    item
                    for item in evidence["evidenceRequirements"]
                    if item["question"] == "Assess condition and batch jointly"
                )
                assert joint["columns"] == ["batch", "condition"]
                assert joint["essential"] is True
                measured = {
                    item["requirementId"]: item for item in evidence["evidenceCoverage"]
                }
                assert measured[joint["requirementId"]]["status"] == "unsupported"
                assert "marginal comparisons do not answer" in " ".join(
                    measured[joint["requirementId"]]["reasons"]
                )
                assert measured["studyDesign"]["status"] == "computed"
            else:
                assert evidence == {"evidenceRequirements": [], "evidenceCoverage": []}
        raise UnexpectedModelBehavior("No additional provider attempt is available")

    monkeypatch.setattr(context_agent, "run_agent_sync", inspect)
    result = context_agent.ExperimentalContextAgent(object()).run(
        store,
        cell_selection=store.cell_selection,
        study_context=text,
        study_objective=objective,
        previous_context=previous,
    )
    assert result.status == "failed"
    assert seen == [rounds]
    assert previous.model_dump(mode="json") == saved_previous


def test_saved_details_return_complete_inventory_and_policy_without_computation():
    cells, characterization = _design()
    from scarf.agent.experimental_context.comparisons import compare_covariates

    measured = compare_covariates(
        cells, characterization, _proposal(), selection_identity={}
    )
    characterization.comparisons = [measured]
    characterization.confounding = [
        {"coefficient": "response", "reason": "exact joint alias"}
    ]
    characterization.columns[0]["levelCounts"] = [
        {"value": str(i), "count": 2} for i in range(12)
    ]
    profile = CellQcProfileEvidence(
        profileId="capture",
        action="sampleMad",
        sampleColumn="sample",
        attributes=["counts"],
        parameters={
            "resolvedBounds": [{"upper": 8}],
            "captureComparisons": [{"unsafe": True}],
        },
        resolvedBounds=[{"upper": 8}],
    )
    deps = ExperimentalContextDependencies(
        characterization=characterization, qcProfiles={"capture": profile}
    )
    ctx = SimpleNamespace(deps=deps)
    compact = tools.compact_context_evidence(
        CovariateEvidence(characterization=characterization, qcProfiles=[profile])
    )
    assert compact["characterization"]["columns"][0]["levelCountsOmitted"] == 4
    detail = asyncio.run(tools.inspect_context_evidence(ctx, "column", "sample"))
    assert len(detail["levelCounts"]) == 12
    assert asyncio.run(
        tools.inspect_context_evidence(ctx, "comparison", measured.evidenceId)
    ) == measured.model_dump(mode="json")
    assert (
        asyncio.run(tools.inspect_context_evidence(ctx, "confounding", "0"))
        == characterization.confounding[0]
    )
    policy = asyncio.run(tools.inspect_context_evidence(ctx, "qcProfile", "capture"))
    assert "captureDetailsRequired" in policy
    assert "resolvedBounds" not in policy
    assert profile.resolvedBounds == [{"upper": 8}]
    for section, key, capture in [
        ("qcProfile", "missing", None),
        ("column", "missing", None),
        ("confounding", "99", None),
        ("column", "sample", "capture"),
    ]:
        with pytest.raises(ModelRetry):
            asyncio.run(tools.inspect_context_evidence(ctx, section, key, capture))
    with pytest.raises(ModelRetry, match="Inspect covariates"):
        asyncio.run(
            tools.inspect_context_evidence(
                SimpleNamespace(deps=ExperimentalContextDependencies()),
                "column",
                "sample",
            )
        )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"metadataColumn": None}, "only metadataColumn"),
        ({"origin": "derivedArtifact"}, "ingestionMetadata"),
        ({"sourceType": "artifact", "metadataColumn": None}, "only artifact"),
        ({"activeCells": 2, "missingCells": 3}, "counts are inconsistent"),
        (
            {"activeCells": 3, "missingCells": 1, "usableForFiltering": True},
            "cannot drive filtering",
        ),
    ],
)
def test_qc_metric_source_cannot_claim_valid_filtering_with_inconsistent_provenance(
    change, reason
):
    with pytest.raises(ValidationError, match=reason):
        QcMetricSourceEvidence.model_validate(
            {
                "sourceId": "source",
                "metricName": "counts",
                "metadataColumn": "counts",
                **change,
            }
        )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"response": ""}, "non-empty"),
        (
            {"explanatoryColumns": ["treatment"], "protectCombination": True},
            "two explanatory",
        ),
    ],
)
def test_proposal_cannot_omit_its_question_or_invent_combination_protection(
    change, reason
):
    with pytest.raises(ValidationError, match=reason):
        _proposal(**change)


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("runAndSelection", "mutually exclusive"),
        ("foreignRun", "opened from this datastore"),
        ("notSelection", "datastore cell_selection"),
        ("wrongNeighbors", "neighbors ArtifactRef"),
        ("foreignNeighbors", "same cell selection"),
        ("wrongConnectivity", "connectivity graph"),
        ("foreignConnectivity", "same cell selection"),
        ("cellKey", "cellQc.cellKey"),
        ("duplicateMetric", "source names must be unique"),
    ],
)
def test_context_rejects_foreign_artifacts_and_ambiguous_sources_before_model_execution(
    monkeypatch, damage, reason
):
    store = _Store()
    wrong = ArtifactRef("datastore", "cell_selection", "f" * 64)
    metric = NamedArtifactSource(
        name="mitochondrial",
        artifact=ArtifactReferenceModel(
            scope="assay", assay="RNA", kind="quality_metric", artifactId="c" * 64
        ),
    )
    monkeypatch.setattr(
        context_agent,
        "_derive_missing_percentage_artifacts",
        lambda *a, **k: [metric, metric] if damage == "duplicateMetric" else [],
    )
    monkeypatch.setattr(
        context_agent, "resolve_cell_aligned_artifact", lambda *a, **k: None
    )
    monkeypatch.setattr(context_agent, "graph_cell_selection", lambda *a: wrong)
    monkeypatch.setattr(
        context_agent,
        "run_agent_sync",
        lambda **k: pytest.fail("Invalid artifacts cannot reach the model"),
    )
    kwargs = {"cell_selection": store.cell_selection}
    if damage == "runAndSelection":
        kwargs["run"] = SimpleNamespace(_owner=store)
    elif damage == "foreignRun":
        kwargs = {"run": SimpleNamespace(_owner=object())}
    elif damage == "notSelection":
        kwargs["cell_selection"] = ArtifactRef("assay", "neighbors", "a" * 64, "RNA")
    elif damage == "wrongNeighbors":
        kwargs["neighbors"] = store.cell_selection
    elif damage == "foreignNeighbors":
        kwargs["neighbors"] = ArtifactRef("assay", "neighbors", "a" * 64, "RNA")
    elif damage == "wrongConnectivity":
        kwargs["connectivity_map"] = store.cell_selection
    elif damage == "foreignConnectivity":
        kwargs["connectivity_map"] = ArtifactRef(
            "assay", "connectivity_map", "a" * 64, "RNA"
        )
    elif damage == "cellKey":
        kwargs["directions"] = {"cellQc": {"cellKey": "I"}}
    with pytest.raises((ValueError, TypeError), match=reason):
        context_agent.ExperimentalContextAgent(object()).run(store, **kwargs)
