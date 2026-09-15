"""Finalization distinguishes disabled scoring from missing required evidence."""

from types import SimpleNamespace

from scarf.agent.orchestrator import finalization
from scarf.agent.orchestrator.models import (
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    OrchestrationRequestRecord,
    PreprocessedAssayHandoff,
    StageEvidenceReference,
    WorkflowIdentity,
    WorkflowStageAttempt,
    artifact_model_to_ref,
)
from scarf.agent.parameter_tuning.contracts import (
    ArtifactRecord,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterTuningReport,
)

from tests.agent_examples import example


def _fixture(monkeypatch, *, scored=False, warning=None):
    sequence = iter(range(1, 30))

    def reference(kind, *, assay="RNA"):
        return ArtifactRecord(
            kind=kind,
            assay=assay,
            scope="datastore" if assay is None else "assay",
            artifactId=f"{next(sequence):064x}",
        )

    cells = reference("cell_selection", assay=None)
    normalized = reference("normalized")
    marker_features = reference("feature_selection")
    artifacts = {
        "pca": reference("reduction"),
        "connectivityMap": reference("connectivity_map"),
        "clusters": reference("cluster_labels"),
        "markerTable": reference("marker_table"),
    }
    selection = reference("cell_selection", assay=None)
    score = reference("doublet_score")
    if scored:
        artifacts.update({"doubletScore:0": score, "doubletCellSelection:0": selection})
    selected = ParameterCandidateEvaluation(
        candidateId="selected",
        parameters=ParameterCandidate(candidateId="selected"),
        status="done",
        eligible=True,
        cellSelection=cells,
        artifacts=artifacts,
        metrics=ParameterMetrics(nClusters=3, seedStability=0.95, markerCoherence=1),
        warnings=[] if warning is None else [warning],
    )
    report = ParameterTuningReport(
        status="done",
        fromAssay="RNA",
        cellSelection=cells,
        evaluations=[selected],
        recommendedCandidateId=selected.candidateId,
        finalClusterArtifact=artifacts["clusters"],
    )
    statuses = {}

    def bind(ref, **inputs):
        statuses[artifact_model_to_ref(ref)] = SimpleNamespace(
            exists=True,
            complete=True,
            inputs={
                name: artifact_model_to_ref(value).to_dict()
                for name, value in inputs.items()
            },
        )

    bind(normalized, cell_selection=cells)
    bind(
        artifacts["clusters"], cell_selection=cells, graph=artifacts["connectivityMap"]
    )
    bind(artifacts["markerTable"], cell_selection=cells, clusters=artifacts["clusters"])
    bind(score, cell_selection=selection)
    layouts = []

    def initialize(ref, **kwargs):
        layouts.append(("initialize", ref))
        return artifact_model_to_ref(reference("embedding_initialization"))

    def umap(ref, initialization, **kwargs):
        layouts.append(("umap", ref))
        return artifact_model_to_ref(reference("embedding"))

    store = SimpleNamespace(
        zw=None,
        summary=lambda: SimpleNamespace(
            assays=[SimpleNamespace(name="RNA", assay_type="RNA")]
        ),
        inspect_artifact=statuses.__getitem__,
        load_artifact=lambda ref: {},
        build_embedding_initialization=initialize,
        run_umap=umap,
    )
    monkeypatch.setattr(
        finalization,
        "graph_cell_selection",
        lambda root, graph: artifact_model_to_ref(cells),
    )
    monkeypatch.setattr(
        finalization.journal, "_ensure_orchestration_store", lambda store: "agents"
    )
    monkeypatch.setattr(
        finalization.journal, "_validated_done_outcome", lambda *args: None
    )
    monkeypatch.setattr(
        finalization.journal,
        "_start_attempt",
        lambda *args, **kwargs: WorkflowStageAttempt(stage="analysis_finalization"),
    )
    monkeypatch.setattr(finalization.journal, "_save_outcome", lambda *args: None)
    monkeypatch.setattr(
        finalization.journal,
        "read_stage_evidence",
        lambda *args: report.model_dump(mode="json"),
    )
    record = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test",
        request=example(AutomatedWorkflowRequest).model_copy(
            update={
                "primaryAssay": "RNA",
                "markerAssay": "RNA",
                "analysisAssays": ["RNA"],
            }
        ),
        config=AutomatedWorkflowConfig(),
    )
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        cellSelection=cells,
        assays=[
            AssayPreprocessingPlan(assay="RNA", assayType="RNA", graphEligible=True)
        ],
    )
    handoff = PreprocessedAssayHandoff(
        assay="RNA",
        assayType="RNA",
        cellSelection=cells,
        normalized=normalized,
        markerFeatures=marker_features,
        nCells=30,
    )

    def run():
        return finalization.FinalizationStagesMixin().analysis_finalization_stage(
            store,
            WorkflowIdentity(workflowRunId="test-workflow"),
            record,
            [],
            plan,
            [handoff],
            report,
            StageEvidenceReference(
                workflowRunId="test-workflow",
                stage="parameter_tuning",
                key="report",
                contentSha256="0" * 64,
            ),
        )

    return SimpleNamespace(
        run=run,
        selected=selected,
        score=score,
        selection=selection,
        layouts=layouts,
        artifacts=artifacts,
    )


def test_explicitly_disabled_doublets_finalize_with_visible_limitation(monkeypatch):
    warning = (
        "Advisory doublet scoring was not run for assay 'RNA' because "
        "score_doublets=False. Doublet contamination was not assessed."
    )
    fixture = _fixture(monkeypatch, warning=warning)
    outcome, final = fixture.run()
    assert outcome.status == "done", outcome.error
    assert final.doubletScores == final.doubletScoreSelections == []
    assert warning in final.limitations
    assert warning in outcome.notes
    assert [name for name, ref in fixture.layouts] == ["initialize", "umap"]
    assert final.markers.artifactId == fixture.artifacts["markerTable"].artifactId
    assert fixture.selected.metrics.seedStability == 0.95


def test_missing_doublets_without_a_recorded_limitation_prevent_completion(monkeypatch):
    fixture = _fixture(monkeypatch)
    outcome, final = fixture.run()
    assert outcome.status == "failed"
    assert "lacks advisory doublet scores" in outcome.error
    assert final.umap is None


def test_existing_scores_retain_exact_capture_selection(monkeypatch):
    fixture = _fixture(monkeypatch, scored=True)
    outcome, final = fixture.run()
    assert outcome.status == "done", outcome.error
    assert final.doubletScores[0].artifactId == fixture.score.artifactId
    assert final.doubletScoreSelections[0].artifactId == fixture.selection.artifactId
    assert final.doubletScoreSelections[0] != final.cellSelection
