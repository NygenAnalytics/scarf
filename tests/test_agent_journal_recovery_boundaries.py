"""Recovery requires the saved scientific identity, not merely a usable store."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import zarr

from scarf.agent import record_io
from scarf.agent.decisions.kernel import DecisionSelection, DecisionSpec
from scarf.agent.orchestrator import AgentOrchestrator, journal
from scarf.agent.orchestrator import main as orchestrator_main
from scarf.agent.orchestrator.models import (
    AutomatedWorkflowRequest,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
)
from scarf.agent.types import AgentDataModel
from tests.agent_journal_store import memory_journal
from tests.test_agent_decision_kernel import _decision_spec


class ScientificEvidence(AgentDataModel):
    conclusion: str = "Supported on the measured cohort"


class OtherEvidence(AgentDataModel):
    conclusion: str = "A different scientific checkpoint"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (
            {"inputs": {}, "outputs": {}, "oldHistoryVersion": 1},
            "Unsupported RNA checkpoint contract",
        ),
        ({"inputs": [], "outputs": {}}, "inputs and outputs must be mappings"),
        ({"inputs": {}, "outputs": []}, "inputs and outputs must be mappings"),
    ],
)
def test_read_checkpoint_rejects_unsupported_history_without_rewriting_it(
    payload: dict[str, Any], reason: str
) -> None:
    store, prefix, record = memory_journal()
    data = dict(payload)
    if set(data) == {"inputs", "outputs"}:
        data["contentSha256"] = hashlib.sha256(
            record_io.canonical_json_bytes(data)
        ).hexdigest()
    key = journal._checkpoint_key(prefix, record.workflowRunId, "qc/evidence")
    raw = record_io.canonical_json_bytes(data)
    journal._write_key_once(store.zw, key, raw)
    with pytest.raises(ValueError, match=reason):
        journal.read_checkpoint(store, prefix, record.workflowRunId, "qc/evidence")
    assert record_io.read_key(store.zw, key) == raw


def test_stage_recovery_rejects_wrong_report_type_checksum_and_ambiguous_ownership() -> (
    None
):
    store, prefix, record = memory_journal()
    started = journal._start_attempt(
        store.zw, prefix, record.workflowRunId, "ingest", record, []
    )
    saved, reference = journal._save_stage_report(
        store, started, ScientificEvidence(), expected_type=ScientificEvidence
    )
    assert journal._recover_persisted_stage_report(
        store, started, expected_type=ScientificEvidence
    ) == (saved, reference)
    with pytest.raises(ValueError, match="different scientific result type"):
        journal._recover_persisted_stage_report(
            store, started, expected_type=OtherEvidence
        )
    with pytest.raises(ValueError, match="exact reference"):
        journal.read_stage_evidence(
            store, reference.model_copy(update={"contentSha256": "f" * 64})
        )
    duplicate = started.model_copy(update={"reportReferences": [reference, reference]})
    with pytest.raises(ValueError, match="exactly one evidence report"):
        journal.load_stage_report(store, duplicate, ScientificEvidence)
    assert journal.read_stage_evidence(store, reference) == saved.model_dump(
        mode="json"
    )


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("native", "must contain decisionId"),
        (
            {
                "decisionId": "correction",
                "optionId": "native",
                "rationale": "Confounded",
                "useHarmony": False,
            },
            "contain exactly",
        ),
        (
            {
                "decisionId": "some_other_decision",
                "optionId": "native",
                "rationale": "Confounded",
            },
            "does not match decision",
        ),
        (
            {"decisionId": "correction", "optionId": "native", "rationale": " "},
            "non-empty rationale",
        ),
        (
            {
                "decisionId": "correction",
                "optionId": "invented",
                "rationale": "Better mixing",
            },
            "persisted option",
        ),
    ],
)
def test_resume_answer_is_bound_to_the_exact_saved_question_and_options(
    answer: Any, reason: str
) -> None:
    paused = WorkflowStageAttempt(
        status="needsInput",
        stage="experimental_context",
        needsInput=WorkflowNeedsInput(
            questions=[
                WorkflowQuestion(
                    questionId="correctionChoice",
                    decisionId="correction",
                    question="The design confounds batch with protected biology. Which offered action is justified?",
                    options=["native", "clarify"],
                )
            ]
        ),
    )
    assert (
        journal._resume_answer_errors(
            paused,
            {
                "correctionChoice": {
                    "decisionId": "correction",
                    "optionId": "native",
                    "rationale": "The design cannot separate batch from condition.",
                }
            },
        )
        == []
    )
    errors = journal._resume_answer_errors(paused, {"correctionChoice": answer})
    assert any(reason in error for error in errors)


@pytest.mark.parametrize(
    "field", ["workflowRunId", "requestSha256", "configSha256", "contentSha256"]
)
def test_request_recovery_does_not_trust_a_corrupted_identity(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    store, prefix, record = memory_journal()
    altered = record.model_copy(
        update={field: "wrong-workflow" if field == "workflowRunId" else "f" * 64}
    )
    monkeypatch.setattr(journal, "_read_model", lambda *_args: altered)
    with pytest.raises(ValueError, match="identity or checksum"):
        journal.read_request(store.zw, prefix, record.workflowRunId)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"zarrPath": None}, "no store path"),
        ({"zarrPath": "other-study.zarr"}, "does not match the saved request"),
        ({"workspace": "other_workspace"}, "does not match the saved request"),
        ({"primaryAssay": ""}, "no selected RNA assay"),
    ],
)
def test_result_store_is_validated_before_opening_datastore(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, Any], reason: str
) -> None:
    store, _prefix, record = memory_journal()
    altered = record.model_copy(
        update={"request": record.request.model_copy(update=change)}
    )
    calls: list[str] = []
    monkeypatch.setattr(journal.zarr, "open_group", lambda *_args, **_kwargs: store.zw)
    monkeypatch.setattr(journal, "read_request", lambda *_args: altered)
    monkeypatch.setattr(
        journal, "DataStore", lambda *_args, **_kwargs: calls.append("open")
    )
    with pytest.raises(ValueError, match=reason):
        journal.open_analysis_store("analysis.zarr", record.workflowRunId)
    assert calls == []


@pytest.mark.parametrize("entry", ["artifact_array", "group"])
def test_result_and_beginner_cannot_treat_an_array_as_a_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry: str
) -> None:
    destination = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(destination), mode="w")
    root.create_array("artifact_array", shape=(2,), dtype="i4")
    root.create_group("group")
    # The array is valid Zarr data, but cannot own the selected analysis history.
    if entry == "artifact_array":
        with pytest.raises(ValueError, match="workspace is not a group"):
            journal.open_analysis_store(destination, "workflow", workspace=entry)
        result = AgentOrchestrator("test-model").run(
            AutomatedWorkflowRequest(
                sourcePath=str(destination),
                workspace=entry,
                studyContext="RNA cells",
                studyObjective="Characterize populations",
            )
        )
        assert result.status == "failed"
        assert "workspace is not a group" in " ".join(result.notes)
    else:
        # A real group proceeds as far as exact history lookup, not numerical work.
        monkeypatch.setattr(
            journal,
            "read_request",
            lambda *_args: (_ for _ in ()).throw(ValueError("Exact history absent")),
        )
        with pytest.raises(ValueError, match="Exact history absent"):
            journal.open_analysis_store(destination, "workflow", workspace=entry)


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"baselineOptionId": "invented"}, "baselineOptionId must reference"),
        (
            {"metricPreferredOptionId": "invented"},
            "metricPreferredOptionId must reference",
        ),
        ({"metricPreferredOptionId": None}, "requires metricPreferredOptionId"),
        ({"allowedSources": []}, "must not be empty"),
        ({"allowedSources": ["agent", "agent"]}, "must not contain duplicates"),
    ],
)
def test_closed_decision_spec_rejects_unregistered_authority(
    update: dict[str, Any], reason: str
) -> None:
    payload = _decision_spec().model_dump(mode="json") | update
    with pytest.raises(ValueError, match=reason):
        DecisionSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        (
            {"overrideEvidenceIds": ["uncited"], "overrideOfOptionId": "baseline"},
            "must be included in evidenceIds",
        ),
        ({"overrideEvidenceIds": ["markers"]}, "require overrideOfOptionId"),
        ({"overrideOfOptionId": "chosen"}, "must differ from selectedOptionId"),
        ({"selectedOptionId": "invented option with spaces"}, "stable identifier"),
    ],
)
def test_model_override_cannot_invent_a_comparator_or_uncited_support(
    update: dict[str, Any], reason: str
) -> None:
    payload = {
        "selectedOptionId": "chosen",
        "evidenceIds": ["markers"],
        "rationale": "Marker programs support the selected populations.",
    } | update
    with pytest.raises(ValueError, match=reason):
        DecisionSelection.model_validate(payload)


@pytest.mark.parametrize(
    ("batch_columns", "reason"),
    [
        ("library", "list of exact observation-column names"),
        ([" "], "list of exact observation-column names"),
        (["donor"], "must include the CELLxGENE"),
    ],
)
def test_manifest_declared_batch_cannot_be_overridden_before_context_enrichment(
    monkeypatch: pytest.MonkeyPatch, batch_columns: Any, reason: str
) -> None:
    runner = AgentOrchestrator("test-model")
    monkeypatch.setattr(runner, "_reuse_or_resume", lambda _request: None)
    monkeypatch.setattr(
        orchestrator_main,
        "inspect_h5ad_manifest",
        lambda *_args, **_kwargs: SimpleNamespace(declaredBatchColumns=["library"]),
    )
    result = runner.run(
        AutomatedWorkflowRequest(
            sourcePath="study.h5ad",
            studyContext="Two library preparations",
            studyObjective="Compare populations",
            experimentalDirections={"batchColumns": batch_columns},
        )
    )
    assert result.status == "failed"
    assert result.currentStage == "ingest"
    assert reason in " ".join(result.notes)
    assert result.workflowRunId is None


@pytest.mark.parametrize(
    ("manifest_status", "input_policy", "expected"),
    [
        ("needsInput", "unattended", "abstained"),
        ("needsInput", "pause", "needsInput"),
        ("abstained", "unattended", "abstained"),
    ],
)
def test_ambiguous_counts_never_reach_expensive_analysis(
    monkeypatch: pytest.MonkeyPatch,
    manifest_status: str,
    input_policy: str,
    expected: str,
) -> None:
    from scarf.agent.orchestrator import AutomatedWorkflowConfig

    runner = AgentOrchestrator(
        "test-model", config=AutomatedWorkflowConfig(inputPolicy=input_policy)
    )
    monkeypatch.setattr(runner, "_reuse_or_resume", lambda _request: None)
    manifest = SimpleNamespace(
        declaredBatchColumns=["library"],
        decision=SimpleNamespace(
            status=manifest_status,
            summary="The count matrix is ambiguous",
            options=["X", "raw/X"],
            evidenceIds=["counts:X", "counts:raw"],
        ),
        priorFiltering=SimpleNamespace(
            limitations=["Only published cells are available"]
        ),
    )
    seen: list[Any] = []

    def inspect(*args: Any, **kwargs: Any) -> Any:
        seen.append((args, kwargs))
        return manifest

    monkeypatch.setattr(orchestrator_main, "inspect_h5ad_manifest", inspect)
    result = runner.run(
        AutomatedWorkflowRequest(
            sourcePath="study.h5ad",
            studyContext="RNA libraries",
            studyObjective="Assess populations",
        )
    )
    assert result.status == expected
    assert result.currentStage == "ingest"
    assert result.limitations == ["Only published cells are available"]
    assert len(seen) == 1
    assert result.workflowRunId is None
    if expected == "needsInput":
        assert result.needsInput.questions[0].options == ["X", "raw/X"]
    else:
        assert "count" in " ".join(result.notes)
