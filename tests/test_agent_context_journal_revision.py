"""Context evidence revisions append history and reject changed measured metadata."""

from types import SimpleNamespace

import pytest

from scarf.agent.experimental_context import agent as context_agent
from scarf.agent.experimental_context.contracts import (
    ExperimentalContextDecision,
    ExperimentalContextResult,
)
from scarf.agent.experimental_context.characterization import characterize_covariates
from scarf.agent.experimental_context.study import build_study_contract
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
)
from scarf.agent.orchestrator import context, journal
from scarf.agent.orchestrator.models import (
    OrchestrationRequestRecord,
    StageEvidenceReference,
    WorkflowIdentity,
)
from scarf.agent.types import ArtifactReferenceModel
from scarf.datastore.datastore import DataStore
from tests.agent_orchestrator_store import create_store


@pytest.mark.parametrize("damage", [None, "currentMetadata", "historicalMetadata"])
def test_missing_joint_question_appends_revision_without_replacing_completed_context(
    tmp_path, monkeypatch, damage
):
    path = create_store(tmp_path / "context-revision.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    selected = store.snapshot_cell_selection("I")
    selection = ArtifactReferenceModel.from_artifact_ref(selected)
    known = characterize_covariates(
        store, cellSelection=selected, model=None, directions={}
    )
    report = ExperimentalContextResult(
        status="done",
        decision=ExperimentalContextDecision(),
        characterization=known,
        cellSelection=selection,
    )
    contract = build_study_contract(
        study_context="Known observed covariates.",
        study_objective="Describe populations.",
        experimental_result=report,
    )
    request = OrchestrationRequestRecord(
        workflowRunId="context-revision",
        modelIdentity="test",
        inputIdentity={},
        config=AutomatedWorkflowConfig(inputPolicy="unattended"),
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="Known observed covariates. Assess individual and combined covariates.",
            studyObjective="Describe populations.",
        ),
    )
    workflow = WorkflowIdentity(request.workflowRunId)
    prefix = journal._ensure_orchestration_store(store)
    old_inputs = (
        {"metadataFingerprints": {"ids": "changed"}}
        if damage == "currentMetadata"
        else {}
    )
    started = journal._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "experimental_context",
        request,
        [],
        inputs=old_inputs,
    )
    _, reference = journal._save_stage_report(
        store, started, report, expected_type=ExperimentalContextResult
    )
    old = journal._complete_attempt(
        started,
        status="done",
        report_references=[reference],
        artifacts={"cellSelection": selection},
        outputs={"studyContract": contract.model_dump(mode="json")},
    )
    journal._save_outcome(store.zw, prefix, old)
    original = journal.read_stage_evidence(store, reference)
    if damage == "historicalMetadata":
        journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "parameter_tuning",
            request,
            [],
            inputs={"metadataFingerprints": {"ids": "changed"}},
        )
    observed = []

    class Review:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            observed.append(kwargs["previous_context"])
            return report.model_copy(
                update={
                    "status": "needsInput",
                    "notes": ["Joint evidence remains unresolved"],
                }
            )

    monkeypatch.setattr(context, "ExperimentalContextAgent", Review)

    def execute():
        return AgentOrchestrator(object()).experimental_context_stage(
            store,
            workflow,
            request,
            [],
            selection,
            StageEvidenceReference(
                workflowRunId=workflow.workflowRunId,
                stage="data_enrichment",
                key="unused",
                contentSha256="a" * 64,
            ),
            [],
            [],
            {},
        )

    if damage:
        with pytest.raises(ValueError, match="metadata"):
            execute()
        assert not observed
    else:
        revised, result = execute()
        assert revised.status == "failed"
        assert result.status == "needsInput"
        assert revised.attemptId != old.attemptId
        assert revised.inputs["reassessContextReport"] == reference.model_dump(
            mode="json"
        )
        assert (
            revised.inputs["requiredDesignQuestions"][0]["question"]
            == "Assess individual and combined covariates"
        )
        assert observed == [report]
    assert journal.read_stage_evidence(store, reference) == original


@pytest.mark.parametrize(
    "directions,error",
    [
        ({"excludeColumns": "cell_type"}, "excludeColumns must be a list"),
        (
            {"coefficientsOfInterest": ["cell_type"], "excludeColumns": ["cell_type"]},
            "forbids runtime use",
        ),
        ({"nested": {"protected": ["cell_type"]}}, "forbids runtime use"),
    ],
)
def test_held_out_author_annotations_cannot_reenter_context_through_nested_directions(
    tmp_path, monkeypatch, directions, error
):
    import numpy as np

    path = create_store(tmp_path / "holdout.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    store.cells.insert("cell_type", np.asarray(["A", "A", "B", "B"]))
    request = OrchestrationRequestRecord(
        workflowRunId="holdout",
        modelIdentity="test",
        inputIdentity={},
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="Observed samples.",
            studyObjective="Describe populations.",
            experimentalDirections=directions,
        ),
    )
    monkeypatch.setattr(
        context_agent,
        "run_agent_sync",
        lambda **k: pytest.fail(
            "Held-out annotations must fail before model execution"
        ),
    )
    with pytest.raises(ValueError, match=error):
        AgentOrchestrator(object()).experimental_context_stage(
            store,
            WorkflowIdentity("holdout"),
            request,
            [],
            ArtifactReferenceModel.from_artifact_ref(
                store.snapshot_cell_selection("I")
            ),
            SimpleNamespace(),
            [],
            [],
            {},
        )
