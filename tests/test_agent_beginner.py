"""Beginner RNA entry point and exact completed-result access."""

from tests.agent_examples import example

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from scarf.agent import AnalysisError, AutomatedWorkflowResult, analyze_rna
from scarf.agent.orchestrator import api, journal
from scarf.agent.orchestrator.main import AgentOrchestrator
from scarf.agent.orchestrator.models import (
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    FinalAnalysisHandoff,
    OrchestrationRequestRecord,
    artifact_model_to_ref,
)
from scarf.agent.types import ArtifactReferenceModel
from scarf.agent.record_io import canonical_json_bytes


def _completed_result(root: Path) -> AutomatedWorkflowResult:
    return AutomatedWorkflowResult(
        status="completed",
        currentStage="analysis_finalization",
        zarrPath=str(root),
        workspace="analysis",
        workflowRunId="workflow-1",
    )


def _final_analysis() -> FinalAnalysisHandoff:
    final = example(FinalAnalysisHandoff)
    final.umap = ArtifactReferenceModel(
        assay="RNA", kind="embedding", artifactId="6" * 64
    )
    final.markers = ArtifactReferenceModel(
        assay="RNA", kind="marker_table", artifactId="7" * 64
    )
    return final


def test_analyze_rna_passes_one_request_and_bounded_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}
    outcome = _completed_result(Path("study.zarr"))

    class Orchestrator:
        def __init__(self, model: Any, *, config: AutomatedWorkflowConfig) -> None:
            called.update(model=model, config=config)

        def run(self, request: AutomatedWorkflowRequest) -> AutomatedWorkflowResult:
            called["request"] = request
            return outcome

    monkeypatch.setattr(api, "AgentOrchestrator", Orchestrator)
    model = object()
    result = analyze_rna(
        Path("study.h5ad"),
        model=model,
        study_context="Human blood, one donor.",
        study_objective="Identify stable populations.",
        assay="counts",
        zarr_path=Path("study.zarr"),
    )
    assert result is outcome
    assert called["model"] is model
    config = called["config"]
    assert config.inputPolicy == "unattended"
    assert config.scoreDoublets is False
    assert config.model_dump(mode="json")["scoreDoublets"] is False
    assert config.screeningCells is None
    assert config.maxScreeningCells == 100_000
    assert config.maxScreeningEvaluations == 24
    assert config.maxTotalScreeningEvaluations == 48
    assert config.maxFullGraphs == 4
    assert config.maxFullPartitions == 8
    assert config.maxFullRepairs == 1
    request = called["request"]
    assert request.sourcePath == "study.h5ad"
    assert request.zarrPath == "study.zarr"
    assert request.primaryAssay == request.markerAssay == "counts"
    assert request.analysisAssays == ["counts"]
    assert request.ingestDirections == {}


@pytest.mark.parametrize("enabled", [True, False])
def test_beginner_can_control_advisory_doublet_scoring(monkeypatch, enabled):
    observed = []
    outcome = _completed_result(Path("study.zarr"))

    def orchestrator(model, *, config):
        observed.append(config)
        return SimpleNamespace(run=lambda request: outcome)

    monkeypatch.setattr(api, "AgentOrchestrator", orchestrator)
    assert (
        analyze_rna(
            "study.h5ad",
            model=object(),
            study_context="Human blood from independent physical captures.",
            study_objective="Discover stable populations.",
            score_doublets=enabled,
        )
        is outcome
    )
    assert observed[0].scoreDoublets is enabled
    if not enabled:
        assert observed[0].model_dump(mode="json")["scoreDoublets"] is False
    assert "scoreDoublets" in observed[0].model_fields_set


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_doublet_scoring_flag_requires_a_boolean_before_pipeline_work(
    monkeypatch, value
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid scoring policy must not start an orchestrator")

    monkeypatch.setattr(api, "AgentOrchestrator", forbidden)
    with pytest.raises(ValueError, match="scoreDoublets"):
        analyze_rna(
            "study.h5ad",
            model=object(),
            study_context="Human blood.",
            study_objective="Discover stable populations.",
            score_doublets=value,
        )
    with pytest.raises(ValueError, match="scoreDoublets"):
        AutomatedWorkflowConfig(scoreDoublets=value)


def test_legacy_doublet_policy_preserves_saved_request_bytes_and_checksums():
    legacy_config = AutomatedWorkflowConfig(screeningCells=50_000).model_dump(
        mode="json", exclude={"scoreDoublets"}
    )
    restored = AutomatedWorkflowConfig.model_validate(legacy_config)
    assert restored.scoreDoublets is True
    assert restored.model_dump(mode="json") == legacy_config
    assert "scoreDoublets" not in AutomatedWorkflowConfig().model_dump(mode="json")
    assert "scoreDoublets" not in AutomatedWorkflowConfig(
        scoreDoublets=True
    ).model_dump(mode="json")
    request = example(AutomatedWorkflowRequest).model_dump(mode="json")
    saved = {
        "recordType": "automatedWorkflowRequest",
        "inputIdentity": {"source": "study.h5ad"},
        "modelIdentity": "test-model",
        "workflowRunId": "legacy-workflow",
        "createdAtNs": 1,
        "request": request,
        "config": legacy_config,
        "requestSha256": hashlib.sha256(canonical_json_bytes(request)).hexdigest(),
        "configSha256": hashlib.sha256(canonical_json_bytes(legacy_config)).hexdigest(),
    }
    saved["contentSha256"] = hashlib.sha256(canonical_json_bytes(saved)).hexdigest()
    original = canonical_json_bytes(saved)
    loaded = OrchestrationRequestRecord.model_validate_json(original)
    assert loaded.config.scoreDoublets is True
    assert canonical_json_bytes(loaded.model_dump(mode="json")) == original
    assert journal._record_checksum(loaded) == saved["contentSha256"]
    assert (
        hashlib.sha256(
            canonical_json_bytes(loaded.config.model_dump(mode="json"))
        ).hexdigest()
        == saved["configSha256"]
    )


def test_doublet_policy_cannot_be_changed_while_resuming_saved_work():
    legacy = AutomatedWorkflowConfig.model_validate({"screeningCells": 50_000})
    AgentOrchestrator(object())._validate_resume_config(legacy)
    AgentOrchestrator(
        object(), config=AutomatedWorkflowConfig(scoreDoublets=True)
    )._validate_resume_config(legacy)
    with pytest.raises(ValueError, match="execution settings differ"):
        AgentOrchestrator(
            object(), config=AutomatedWorkflowConfig(scoreDoublets=False)
        )._validate_resume_config(legacy)

    disabled = AutomatedWorkflowConfig(scoreDoublets=False)
    AgentOrchestrator(object())._validate_resume_config(disabled)
    AgentOrchestrator(
        object(), config=AutomatedWorkflowConfig(scoreDoublets=False)
    )._validate_resume_config(disabled)
    with pytest.raises(ValueError, match="execution settings differ"):
        AgentOrchestrator(
            object(), config=AutomatedWorkflowConfig(scoreDoublets=True)
        )._validate_resume_config(disabled)


@pytest.mark.parametrize("status", ["failed", "abstained", "needsInput"])
def test_beginner_failure_raises_with_resumable_result(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    outcome = AutomatedWorkflowResult.model_validate(
        {
            "status": status,
            "currentStage": "experimental_context",
            "zarrPath": "study.zarr",
            "workflowRunId": "workflow-1",
            "notes": ["Capture identity is unresolved"],
        }
    )
    monkeypatch.setattr(
        api,
        "AgentOrchestrator",
        lambda *_args, **_kwargs: SimpleNamespace(run=lambda _request: outcome),
    )
    with pytest.raises(AnalysisError, match="Capture identity is unresolved") as error:
        analyze_rna(
            "study.zarr",
            model=object(),
            study_context="Human blood.",
            study_objective="Identify stable populations.",
        )
    assert error.value.result is outcome
    assert "Resume workflow 'workflow-1' in 'study.zarr'" in str(error.value)


def test_beginner_rejects_removed_candidate_control() -> None:
    with pytest.raises(TypeError, match="max_candidates"):
        analyze_rna(
            "study.zarr",
            model=object(),
            study_context="Human blood.",
            study_objective="Identify stable populations.",
            **{"max_candidates": 1},
        )


@pytest.mark.parametrize(
    "obsolete",
    [
        "maxRefinedCandidatesPerAssay",
        "maxHarmonyCandidatesPerAssay",
        "runConfoundedHarmonyDiagnostic",
        "maxCandidateEvaluations",
        "maxIdentityFeatures",
        "hvgCandidateCounts",
        "pcaCandidateDimensions",
        "graphNeighborCandidates",
        "leidenResolutionCandidates",
        "maxRevisions",
        "maxCandidateBranches",
        "primaryInitialCandidates",
        "secondaryInitialCandidates",
        "integrationResolutionCandidates",
        "maxGraphAssays",
        "leidenSeeds",
        "clusterSubsamples",
        "clusterSubsampleFraction",
    ],
)
def test_legacy_config_and_saved_requests_fail_explicitly(obsolete: str) -> None:
    old_config = {obsolete: 1}
    with pytest.raises(ValueError, match="Create a new single-RNA workflow"):
        AutomatedWorkflowConfig.model_validate(old_config)
    saved = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test",
        request=example(AutomatedWorkflowRequest),
    ).model_dump(mode="json")
    saved["config"].update(old_config)
    with pytest.raises(ValueError, match="cannot be resumed or regenerated"):
        OrchestrationRequestRecord.model_validate(saved)


@pytest.mark.parametrize(
    "limits",
    [
        {"screeningCells": 19},
        {"screeningCells": 100, "maxScreeningCells": 99},
        {"maxScreeningEvaluations": 3},
        {"maxScreeningEvaluations": 12, "maxTotalScreeningEvaluations": 11},
        {"maxFullGraphs": 0},
        {"maxFullPartitions": 0},
        {"maxFullRepairs": 2},
    ],
)
def test_impossible_work_limits_fail_before_execution(limits: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        AutomatedWorkflowConfig.model_validate(limits)


@pytest.mark.parametrize(
    "routing",
    [
        {"analysisAssays": ["RNA", "ADT"]},
        {"pairedAssays": ["RNA", "ADT"]},
        {"primaryAssay": "RNA", "analysisAssays": ["counts"]},
        {"primaryAssay": "RNA", "markerAssay": "ADT"},
        {"experimentalDirections": {"hypothesisTesting": {}}},
    ],
)
def test_request_rejects_unsupported_routing(routing: dict[str, Any]) -> None:
    values = example(AutomatedWorkflowRequest).model_dump(mode="json")
    with pytest.raises(ValueError):
        AutomatedWorkflowRequest.model_validate({**values, **routing})


def test_result_helpers_resolve_exact_journal_refs_and_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scarf.agent import _plots

    result = _completed_result(tmp_path)
    final = _final_analysis()
    original = result.model_dump(mode="json")
    opened: list[tuple[str, str, str | None]] = []
    snapshots: list[tuple[Any, str]] = []
    plotted: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    plot_result, marker_table = object(), object()
    store = SimpleNamespace(
        get_markers=lambda **options: markers.append(options) or marker_table
    )

    def open_store(path: str, run_id: str, *, workspace: str | None) -> Any:
        opened.append((path, run_id, workspace))
        return store

    def snapshot(target: Any, run_id: str) -> dict[str, Any]:
        snapshots.append((target, run_id))
        return {"status": "completed", "finalAnalysis": final.model_dump(mode="json")}

    def plot(target: Any, **options: Any) -> Any:
        assert target is store
        plotted.append(options)
        return plot_result

    monkeypatch.setattr(journal, "open_analysis_store", open_store)
    monkeypatch.setattr(journal, "analysis_snapshot", snapshot)
    monkeypatch.setattr(_plots, "plot_final_umap", plot)
    assert result.plot_embedding(figsize=(8, 5)) is plot_result
    assert result.get_markers(group_id="2", min_score=0.5) is marker_table
    assert opened == [(str(tmp_path), "workflow-1", "analysis")] * 2
    assert snapshots == [(store, "workflow-1")] * 2
    assert final.umap is not None and final.clusters is not None
    assert final.cellSelection is not None and final.graph is not None
    assert final.markers is not None
    assert plotted == [
        {
            "umap": artifact_model_to_ref(final.umap),
            "clusters": artifact_model_to_ref(final.clusters),
            "cell_selection": artifact_model_to_ref(final.cellSelection),
            "graph": artifact_model_to_ref(final.graph),
            "figsize": (8, 5),
        }
    ]
    assert markers == [
        {
            "marker": artifact_model_to_ref(final.markers),
            "group_id": "2",
            "min_score": 0.5,
            "min_frac_exp": 0.2,
        }
    ]
    assert result.model_dump(mode="json") == original
    assert "finalAnalysis" not in original
    assert "workflowRun" not in original
    with pytest.raises(ValueError, match="exact completed cluster map"):
        result.plot_embedding(layout=artifact_model_to_ref(final.umap))
    with pytest.raises(ValueError, match="exact completed cluster map"):
        result.plot_embedding(color_by="condition")


@pytest.mark.parametrize("method", ["plot_embedding", "get_markers", "report"])
def test_result_helpers_explain_noncompleted_outcome(method: str) -> None:
    result = AutomatedWorkflowResult(notes=["Input file is missing"])
    with pytest.raises(
        AnalysisError, match="failed during ingest.*Input file is missing"
    ):
        getattr(result, method)()


def test_result_report_regenerates_from_exact_saved_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.report.generator as generator

    result = _completed_result(tmp_path)
    final = _final_analysis()
    store = object()
    expected = tmp_path / "analysis/agents/orchestrations/workflow-1/report/index.html"
    generated: list[tuple[Any, str]] = []

    def open_store(path: str, run_id: str, *, workspace: str | None) -> Any:
        assert (path, run_id, workspace) == (str(tmp_path), "workflow-1", "analysis")
        return store

    def generate(target: Any, run_id: str) -> Path:
        generated.append((target, run_id))
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_text("<html>Analysis</html>", encoding="utf-8")
        return expected

    monkeypatch.setattr(journal, "open_analysis_store", open_store)
    monkeypatch.setattr(
        journal,
        "analysis_snapshot",
        lambda *_args: {
            "status": "completed",
            "finalAnalysis": final.model_dump(mode="json"),
        },
    )
    monkeypatch.setattr(generator, "generate_agent_report", generate)
    assert result.report() == expected
    assert result.report() == expected
    assert generated == [(store, "workflow-1")] * 2


@pytest.mark.parametrize("workspace", [None, "analysis"])
def test_legacy_saved_config_blocks_resume_and_report_without_changing_artifacts(
    tmp_path: Path, workspace: str | None
) -> None:
    import hashlib

    import numpy as np

    from scarf.agent import record_io
    from scarf.agent.orchestrator import AgentOrchestrator
    from scarf.agent.orchestrator.models import AutomatedWorkflowResumeRequest
    from scarf.agent.report import generate_agent_report
    from scarf.datastore.datastore import DataStore
    from tests.agent_orchestrator_store import create_store

    path = create_store(tmp_path / "legacy.zarr", workspace=workspace)
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        nthreads=2,
        workspace=workspace,
    )
    cells = store.snapshot_cell_selection()
    features = store.select_all_features(from_assay="RNA")
    normalized = store.run_normalization(cells, features)
    original_values = np.asarray(store.load_artifact(normalized)["data"][:])
    prefix = journal._ensure_orchestration_store(store)
    request = AutomatedWorkflowRequest(
        sourcePath=str(path),
        zarrPath=str(path),
        studyContext="One human RNA sample.",
        studyObjective="Inspect stable cell populations.",
        workspace=workspace,
        primaryAssay="RNA",
    )
    old_config = AutomatedWorkflowConfig().model_dump(mode="json")
    old_config["maxCandidateBranches"] = 24
    payload = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test",
        workflowRunId="legacy-config",
        request=request,
        requestSha256=journal._sha256_model(request),
    ).model_dump(mode="json")
    payload["config"] = old_config
    payload["configSha256"] = hashlib.sha256(
        record_io.canonical_json_bytes(old_config)
    ).hexdigest()
    payload["contentSha256"] = hashlib.sha256(
        record_io.canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "contentSha256"}
        )
    ).hexdigest()
    request_key = journal._request_key(prefix, "legacy-config")
    original_request = record_io.display_json_bytes(payload)
    journal._write_key_once(store.zw, request_key, original_request)

    message = "Unsupported saved agent workflow.*cannot be resumed or regenerated"
    resume_request = AutomatedWorkflowResumeRequest(
        zarrPath=str(path), workflowRunId="legacy-config", workspace=workspace
    )
    orchestrator = AgentOrchestrator(object())
    with pytest.raises(ValueError, match=message):
        orchestrator.load_request_for_resume(resume_request)
    failed = orchestrator.resume(resume_request)
    assert failed.status == "failed"
    assert failed.workflowRunId == "legacy-config"
    assert any("Unsupported saved agent workflow" in note for note in failed.notes)

    with pytest.raises(ValueError, match=message):
        generate_agent_report(path, "legacy-config", workspace=workspace)

    reopened = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        nthreads=2,
        zarr_mode="r",
        workspace=workspace,
    )
    assert record_io.read_key(reopened.zw, request_key) == original_request
    np.testing.assert_array_equal(
        reopened.load_artifact(normalized)["data"][:], original_values
    )
