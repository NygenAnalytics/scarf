"""Supported local HTML report contracts for completed agent workflows."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import zarr
from pydantic_ai import ModelRetry, UnexpectedModelBehavior

import scarf.agent as agent_api
import scarf.agent.biological_interpretation as biological_module
import scarf.agent.config.agent_exec as agent_exec_module
import scarf.agent.data_enrichment as enrichment_module
import scarf.agent.experimental_context as experimental_module
import scarf.agent.report as report_module
import scarf.agent.orchestrator.journal as journal_module
import scarf.agent.orchestrator.main as orchestrator_main
from scarf.agent import (
    AgentWorkflowRun,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    FinalAnalysisHandoff,
    create_agent_workflow,
    generate_agent_report,
    list_agent_workflows,
    load_agent_workflow,
)
from scarf.agent.biological_interpretation import (
    BiologicalInterpretationDependencies,
    BiologicalInterpretationNeedsInput,
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterMarkerEvidence,
)
from scarf.agent.characterize_covariates import CovariateCharacterization
from scarf.agent.data_enrichment import (
    AssayFeatureInspection,
    DataEnrichmentAgent,
    DataEnrichmentDependencies,
    DataEnrichmentToolCall,
)
from scarf.agent.experimental_context import (
    CellQcProfileEvidence,
    ExperimentalContextDependencies,
)
from scarf.agent.orchestrator.models import (
    NativeAnalysisHandoff,
    OrchestrationRequestRecord,
    WorkflowStageAttempt,
)
from scarf.agent.types import ArtifactReferenceModel


def _workflow(
    *,
    workspace: str | None = None,
    status: Literal["completed", "running"] = "completed",
) -> AgentWorkflowRun:
    return AgentWorkflowRun(
        workflowRunId="report-workflow",
        workspace=workspace,
        createdAtNs=1,
        finalizedAtNs=2 if status != "running" else 0,
        status=status,
        finalizationMessage="analysis completed" if status != "running" else "",
        analysisStore="data.zarr",
        datasetFingerprints={"RNA": "dataset-rna"},
    )


def _reports(study_context: str) -> dict[str, list[dict[str, Any]]]:
    run_info = {
        "agentName": "data_enrichment",
        "runId": "provider-run",
        "modelName": "test-model",
        "durationSeconds": 1.5,
        "usage": {
            "requests": 2,
            "toolCalls": 1,
            "inputTokens": 20,
            "outputTokens": 5,
            "totalTokens": 25,
        },
    }
    candidate = {
        "candidateId": "refined",
        "phase": "refined",
        "status": "done",
        "eligible": True,
        "parameters": {
            "reductionMethod": "pca",
            "dimensions": 21,
            "neighborsK": 11,
            "leidenResolution": 0.75,
            "useHarmony": False,
        },
        "metrics": {
            "nClusters": 7,
            "minClusterCells": 42,
            "graphSilhouetteMedian": 0.343,
        },
    }
    return {
        "data_enrichment": [
            {
                "status": "done",
                "studyContextSummary": {
                    "studyContext": study_context,
                    "organismReferences": ["human"],
                    "tissueReferences": ["blood"],
                },
                "policies": [{"assay": "RNA", "policyId": "rna-default"}],
                "runInfo": run_info,
            }
        ],
        "experimental_context": [
            {
                "status": "done",
                "decision": {"batchCorrection": {"action": "skip"}},
                "cellQc": {"action": "globalGaussian", "driverAssay": "RNA"},
            }
        ],
        "parameter_tuning": [
            {
                "status": "done",
                "fromAssay": "RNA",
                "totalCandidates": 2,
                "recommendedByAssay": {"RNA": "refined"},
                "rationale": "The refined candidate balanced cluster viability.",
                "stopReason": "The bounded refinement completed.",
                "assayReports": {
                    "RNA": {
                        "recommendedCandidateId": "refined",
                        "confidence": "medium",
                        "evaluations": [candidate],
                        "comparisons": [
                            {
                                "candidateId": "baseline",
                                "summary": (
                                    "The refined candidate retained larger "
                                    "minimum clusters."
                                ),
                                "evidenceIds": ["candidate:refined:clusters"],
                            }
                        ],
                        "searchPlan": {
                            "status": "refine",
                            "objectives": ["Test an intermediate resolution."],
                        },
                    }
                },
            }
        ],
        "biological_interpretation": [
            {
                "status": "done",
                "clusterInterpretations": [
                    {
                        "clusterId": "0",
                        "proposedIdentity": "T cell",
                        "identityIsHypothesis": True,
                    }
                ],
                "treatmentObservations": [],
                "followUps": ["Validate the proposed identities."],
            }
        ],
    }


def _patch_completed_workflow(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    workspace: str | None = None,
    study_context: str = "A human blood study.",
    plots: bool = True,
) -> Path:
    group = zarr.open_group(str(root), mode="w", zarr_format=3)
    if workspace is not None:
        group.create_group(workspace)
    workflow = _workflow(workspace=workspace)
    final = FinalAnalysisHandoff.get_example().model_copy(
        update={"workflowRunId": workflow.workflowRunId}
    )
    result = AutomatedWorkflowResult(
        status="completed",
        currentStage="biological_interpretation",
        zarrPath=str(root),
        workflowRun=workflow,
        finalAnalysis=final,
    )
    request = AutomatedWorkflowRequest(
        sourcePath="input.h5ad",
        zarrPath=str(root),
        studyContext=study_context,
        workspace=workspace,
    )
    request_record = SimpleNamespace(
        request=request,
        config=AutomatedWorkflowConfig(),
    )

    monkeypatch.setattr(
        report_module,
        "load_agent_workflow",
        lambda *_a, **_k: workflow,
    )
    monkeypatch.setattr(report_module, "_open_datastore", lambda *_a, **_k: object())
    monkeypatch.setattr(
        report_module,
        "_load_completed_result",
        lambda *_a, **_k: ("agents/orchestrations", result, request_record),
    )
    monkeypatch.setattr(
        report_module,
        "_collect_reports",
        lambda *_a, **_k: _reports(study_context),
    )
    monkeypatch.setattr(
        report_module,
        "_collect_history",
        lambda *_a, **_k: (
            [
                {
                    "stage": "parameter_tuning",
                    "status": "done",
                    "durationSeconds": 3.5,
                    "actions": ["evaluate_refined_candidate"],
                    "reportCount": 1,
                    "artifactCount": 4,
                    "artifacts": {
                        "selectedGraph": {
                            "scope": "assay",
                            "assay": "RNA",
                            "kind": "connectivity_map",
                            "artifactId": "a" * 64,
                        }
                    },
                    "parentAttempts": ["preprocessing:attempt-1"],
                    "questionIds": [],
                    "noteCount": 0,
                    "errorType": None,
                }
            ],
            [],
        ),
    )

    def collect_artifacts(
        _store: object,
        _result: AutomatedWorkflowResult,
        plot_dir: Path,
    ) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, str], list[str]]:
        if not plots:
            return (
                {"0": 3, "1": 2},
                [],
                {},
                ["umapClusters: ImportError: plotting dependencies are unavailable"],
            )
        plot_dir.mkdir(parents=True, exist_ok=True)
        (plot_dir / "final_umap.png").write_bytes(b"png")
        (plot_dir / "final_umap.png.json").write_text(
            '{"artifact":"umap"}\n', encoding="utf-8"
        )
        return (
            {"0": 3, "1": 2},
            [{"group_id": "0", "feature_name": "CD3D", "score": 8.5}],
            {"umapClusters": "plots/final_umap.png"},
            [],
        )

    monkeypatch.setattr(report_module, "_collect_final_artifacts", collect_artifacts)
    return root


def test_public_report_generates_branded_readable_html_and_relative_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        study_context='Human blood <script>alert("unsafe")</script> & treatment.',
    )
    immutable_record = root / "agents/runs/report-workflow/workflow.json"
    immutable_record.parent.mkdir(parents=True)
    immutable_record.write_bytes(b'{"immutable":true}\n')

    report_path = generate_agent_report(root, "report-workflow")
    markup = report_path.read_text(encoding="utf-8")

    assert agent_api.generate_agent_report is generate_agent_report
    assert report_path == root / "agents/runs/report-workflow/report/index.html"
    assert immutable_record.read_bytes() == b'{"immutable":true}\n'
    assert 'href="https://www.nygen.io/"' in markup
    assert ">Nygen Analytics</a>" in markup
    assert 'href="https://www.nygen.io/products/scarfweb"' in markup
    assert (
        "Distributed, secure infrastructure for intuitive secondary analysis, "
        "browser-native."
    ) in markup
    assert "Human blood &lt;script&gt;alert" in markup
    assert '<script>alert("unsafe")</script>' not in markup
    assert "Parameter tuning and graph selection" in markup
    assert "refined" in markup
    assert "0.343" in markup
    assert "The refined candidate retained larger minimum clusters." in markup
    assert "Stage artifact inventory" in markup
    assert "connectivity_map" in markup
    assert "Recorded totals" in markup
    assert "evaluate_refined_candidate" in markup
    assert 'src="plots/final_umap.png"' in markup
    assert 'href="plots/final_umap.png.json"' in markup
    assert (report_path.parent / "plots/final_umap.png").read_bytes() == b"png"


def test_report_uses_workspace_path_and_can_be_regenerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        workspace="analysis",
        study_context="First context",
    )

    first = generate_agent_report(root, "report-workflow", workspace="analysis")
    assert first == (root / "analysis/agents/runs/report-workflow/report/index.html")
    first_markup = first.read_text(encoding="utf-8")
    assert "First context" in first_markup

    monkeypatch.setattr(
        report_module,
        "_collect_reports",
        lambda *_a, **_k: _reports("Regenerated context"),
    )
    second = generate_agent_report(root, "report-workflow", workspace="analysis")

    assert second == first
    second_markup = second.read_text(encoding="utf-8")
    assert "Regenerated context" in second_markup
    assert second_markup != first_markup


def test_report_remains_available_when_optional_plots_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        plots=False,
    )

    report_path = generate_agent_report(root, "report-workflow")
    markup = report_path.read_text(encoding="utf-8")

    assert report_path.is_file()
    assert "No plots could be rendered" in markup
    assert "plotting dependencies are unavailable" in markup
    assert "Final cluster sizes" in markup


def test_report_rejects_remote_and_non_completed_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="local filesystem"):
        generate_agent_report("s3://bucket/data.zarr", "report-workflow")

    root = tmp_path / "data.zarr"
    zarr.open_group(str(root), mode="w", zarr_format=3)
    running = _workflow(status="running")
    monkeypatch.setattr(
        report_module,
        "load_agent_workflow",
        lambda *_a, **_k: running,
    )
    monkeypatch.setattr(report_module, "_open_datastore", lambda *_a, **_k: object())

    with pytest.raises(RuntimeError, match="completed workflows"):
        generate_agent_report(root, running.workflowRunId)


def test_orchestrator_generates_only_completed_local_reports_non_fatally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generated: list[tuple[object, str]] = []
    local_store = SimpleNamespace(z=object())
    completed = _workflow()

    monkeypatch.setattr(
        orchestrator_main,
        "zarr_root_path",
        lambda _store: tmp_path / "data.zarr",
    )
    monkeypatch.setattr(
        report_module,
        "generate_agent_report",
        lambda target, workflow_run_id: (
            generated.append((target, workflow_run_id))
            or tmp_path / "data.zarr/agents/runs/report-workflow/report/index.html"
        ),
    )

    orchestrator_main._generate_completed_report(local_store, completed)

    assert generated == [(local_store, completed.workflowRunId)]
    assert "Agent workflow report:" in capsys.readouterr().out

    orchestrator_main._generate_completed_report(
        local_store,
        _workflow(status="running"),
    )
    monkeypatch.setattr(orchestrator_main, "zarr_root_path", lambda _store: None)
    orchestrator_main._generate_completed_report(local_store, completed)
    assert len(generated) == 1

    monkeypatch.setattr(
        orchestrator_main,
        "zarr_root_path",
        lambda _store: tmp_path / "data.zarr",
    )
    monkeypatch.setattr(
        report_module,
        "generate_agent_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )
    orchestrator_main._generate_completed_report(local_store, completed)


def test_derived_report_files_do_not_change_workflow_record_discovery(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "data.zarr"), mode="w", zarr_format=3)
    root.create_group("cellData")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.attrs["dataset_fingerprint"] = "dataset-rna"
    workflow = create_agent_workflow(root, workflow_run_id="report-workflow")
    report_dir = (
        tmp_path / "data.zarr" / "agents" / "runs" / workflow.workflowRunId / "report"
    )
    plot_dir = report_dir / "plots"
    plot_dir.mkdir(parents=True)
    (report_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (plot_dir / "final_umap.png").write_bytes(b"png")
    (plot_dir / "final_umap.png.json").write_text("{}\n", encoding="utf-8")

    assert load_agent_workflow(root, workflow.workflowRunId) == workflow
    assert list_agent_workflows(root, include_incomplete=True) == [workflow]


def test_report_store_request_and_result_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "data.zarr"
    local_root.mkdir()

    class LocalDataStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs
            self.workspace = kwargs.get("workspace")
            self.z = object()

    monkeypatch.setattr(report_module, "DataStore", LocalDataStore)
    monkeypatch.setattr(report_module, "zarr_root_path", lambda _store: None)
    with pytest.raises(ValueError, match="local filesystem"):
        report_module._local_root(LocalDataStore())

    monkeypatch.setattr(
        report_module,
        "zarr_root_path",
        lambda _store: local_root,
    )
    assert report_module._local_root(f"file://{local_root}") == local_root.resolve()
    assert report_module._local_root(str(local_root)) == local_root.resolve()
    with pytest.raises(TypeError, match="local filesystem"):
        report_module._local_root(object())
    with pytest.raises(FileNotFoundError):
        report_module._local_root(tmp_path / "missing.zarr")

    workflow = _workflow()
    with pytest.raises(ValueError, match="workspace"):
        report_module._open_datastore(
            LocalDataStore(workspace="other"),
            local_root,
            workflow,
        )
    opened = report_module._open_datastore(local_root, local_root, workflow)
    assert opened.args == (str(local_root),)
    assert opened.kwargs["default_assay"] == "RNA"
    assert opened.kwargs["zarr_mode"] == "r"

    request = AutomatedWorkflowRequest.get_example()
    config = AutomatedWorkflowConfig.get_example()
    valid_record = OrchestrationRequestRecord(
        workflowRunId="workflow-1",
        request=request,
        config=config,
        requestSha256=journal_module._sha256_model(request),
        configSha256=journal_module._sha256_model(config),
    )
    valid_record.contentSha256 = journal_module._record_checksum(valid_record)
    current_record = valid_record
    monkeypatch.setattr(
        journal_module,
        "_read_model",
        lambda *_args, **_kwargs: current_record,
    )
    store = SimpleNamespace(z=object(), zw=object())
    assert report_module._load_request(store, "agents", "workflow-1") == valid_record

    invalid_records = (
        (
            valid_record.model_copy(update={"workflowRunId": "another-workflow"}),
            "another workflow",
        ),
        (
            valid_record.model_copy(update={"requestSha256": "f" * 64}),
            "request checksum",
        ),
        (
            valid_record.model_copy(update={"configSha256": "f" * 64}),
            "configuration checksum",
        ),
        (
            valid_record.model_copy(update={"contentSha256": "f" * 64}),
            "request envelope",
        ),
    )
    for current_record, message in invalid_records:
        with pytest.raises(ValueError, match=message):
            report_module._load_request(store, "agents", "workflow-1")

    monkeypatch.setattr(
        journal_module,
        "_ensure_orchestration_store",
        lambda _store: "agents/orchestrations",
    )
    terminal_result: object | None = None
    monkeypatch.setattr(
        journal_module,
        "_load_terminal_result",
        lambda *_args, **_kwargs: terminal_result,
    )
    with pytest.raises(FileNotFoundError, match="no terminal result"):
        report_module._load_completed_result(store, workflow)

    terminal_result = SimpleNamespace(status="failed", finalAnalysis=object())
    with pytest.raises(ValueError, match="final analysis"):
        report_module._load_completed_result(store, workflow)

    terminal_result = SimpleNamespace(status="completed", finalAnalysis=object())
    monkeypatch.setattr(
        report_module,
        "_load_request",
        lambda *_args, **_kwargs: SimpleNamespace(
            request=SimpleNamespace(workspace="other")
        ),
    )
    with pytest.raises(ValueError, match="request workspace"):
        report_module._load_completed_result(store, workflow)

    invalid_attempt = WorkflowStageAttempt(
        status="failed",
        startedAtNs=1,
        completedAtNs=2,
        error="not a valid error type!?: details",
    )
    assert report_module._stage_summary(invalid_attempt)["errorType"] == (
        "WorkflowStageError"
    )

    data_store = LocalDataStore(workspace="analysis")
    with pytest.raises(ValueError, match="workspace does not match"):
        generate_agent_report(
            data_store,
            "report-workflow",
            workspace="other",
        )


def test_report_renderer_edge_branches() -> None:
    assert report_module._safe_assay_name("RNA / strange assay", "fallback") == (
        "rna_strange_assay"
    )
    assert report_module._safe_assay_name("***", "fallback") == "fallback"
    assert report_module._scalar(None) == "Not provided"
    assert "Nothing" in report_module._chips(None, empty="Nothing")
    assert "value" in report_module._chips("value")
    assert report_module._latest({"agent": {"status": "done"}}, "agent") == {
        "status": "done"
    }
    assert report_module._latest({}, "agent") == {}

    native_plot = report_module._render_plots(
        {"nativeUmapRna": "plots/native.png"},
        [],
    )
    assert "Rna native UMAP" in native_plot
    assert "finalized native Rna" in native_plot
    assert "No final cluster counts" in report_module._render_clusters({})

    legacy_parameter = {
        "fromAssay": "RNA",
        "evaluations": [{"candidateId": "native"}],
    }
    assert report_module._parameter_rows(legacy_parameter)[0]["assay"] == "RNA"
    assert "No Parameter Tuning report" in report_module._render_parameter_tuning({})
    rendered_parameter = report_module._render_parameter_tuning(
        {
            "fromAssay": "RNA",
            "searchPlan": {"status": "refine"},
            "comparisons": [{"candidateId": "native"}],
            "finalSelection": {"comparisons": [{"candidateId": "integrated"}]},
        }
    )
    assert "RNA" in rendered_parameter
    assert "final graph" in rendered_parameter
    assert "No provider execution metadata" in report_module._render_executions({})


def test_report_collects_bounded_artifact_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def artifact(
        kind: str,
        digit: str,
        *,
        scope: Literal["assay", "datastore"] = "assay",
    ) -> ArtifactReferenceModel:
        return ArtifactReferenceModel(
            scope=scope,
            assay=None if scope == "datastore" else "RNA",
            kind=kind,
            artifactId=digit * 64,
        )

    final_clusters = artifact("cluster_labels", "3")
    final_umap = artifact("embedding", "4")
    final = FinalAnalysisHandoff(
        workflowRunId="report-workflow",
        primaryAssay="RNA",
        markerAssay="RNA",
        cellSelection=artifact("cell_selection", "c", scope="datastore"),
        graph=artifact("connectivity_map", "2"),
        clusters=final_clusters,
        umap=final_umap,
        markers=artifact("marker_table", "5"),
        nativeAnalyses=[
            NativeAnalysisHandoff.get_blank(),
            NativeAnalysisHandoff(
                assay="RNA / strange assay",
                reductionMethod="pca",
                clusters=artifact("cluster_labels", "6"),
                umap=artifact("embedding", "7"),
            ),
            NativeAnalysisHandoff(
                assay="RNA / strange assay",
                reductionMethod="pca",
                clusters=artifact("cluster_labels", "8"),
                umap=artifact("embedding", "9"),
            ),
        ],
    )
    result = AutomatedWorkflowResult(
        status="completed",
        currentStage="biological_interpretation",
        zarrPath=str(tmp_path / "data.zarr"),
        workflowRun=_workflow(),
        finalAnalysis=final,
    )

    class PlotMethods:
        @staticmethod
        def marker_heatmap(**_kwargs: object) -> object:
            raise RuntimeError("heatmap unavailable")

    class ArtifactStore:
        plots = PlotMethods()

        @staticmethod
        def load_artifact(reference: object) -> dict[str, np.ndarray]:
            if getattr(reference, "artifact_id", None) == "3" * 64:
                return {"values": np.asarray(["0", "0", "1"])}
            return {}

        @staticmethod
        def inspect_artifact(_reference: object) -> SimpleNamespace:
            return SimpleNamespace(
                parameters={
                    "normalization": {
                        "log_transform": True,
                        "renormalize_subset": True,
                    }
                }
            )

        @staticmethod
        def get_markers(
            _marker: object,
            *,
            group_id: str,
            min_score: float,
            min_frac_exp: float,
        ) -> pd.DataFrame:
            assert min_score == -1
            assert min_frac_exp == -1
            if group_id == "1":
                raise RuntimeError("marker table unavailable")
            return pd.DataFrame(
                {
                    "group_id": ["unknown", "0", "0"],
                    "feature_name": ["ignored", "CD3D", "unresolved"],
                    "feature_id": ["ignored-id", "ENSG00000167286", None],
                    "feature_index": [None, None, None],
                    "score": [3.0, 2.0, 1.0],
                }
            )

    monkeypatch.setattr(report_module, "MAX_EMBEDDING_PLOT_CELLS", 0)
    monkeypatch.setattr(report_module, "MAX_COMPOSITION_PLOT_CELLS", 0)
    monkeypatch.setattr(report_module, "MAX_DOTPLOT_CELLS", 0)
    monkeypatch.setattr(report_module, "MAX_CONNECTIVITY_PLOT_CELLS", 0)

    store = ArtifactStore()
    counts, markers, plots, notes = report_module._collect_final_artifacts(
        store,
        result,
        tmp_path / "plots",
    )
    assert counts == {"0": 2, "1": 1}
    assert len(markers) == 3
    assert plots == {}
    assert any("nativeUmapRnaStrangeAssay2" in note for note in notes)
    assert any("marker export for cluster 1" in note for note in notes)
    assert any("markerDotplot: skipped" in note for note in notes)
    assert any("clusterConnectivity: skipped" in note for note in notes)

    monkeypatch.setattr(report_module, "MAX_MARKER_DOTPLOT_FEATURES", 1)
    _counts, _markers, _plots, one_marker_notes = (
        report_module._collect_final_artifacts(store, result, tmp_path / "plots-one")
    )
    assert any("markerDotplot: skipped" in note for note in one_marker_notes)

    monkeypatch.setattr(report_module, "MAX_MARKER_DOTPLOT_FEATURES", object())
    _counts, _markers, _plots, invalid_limit_notes = (
        report_module._collect_final_artifacts(
            store, result, tmp_path / "plots-invalid"
        )
    )
    assert any("markerDotplot: TypeError" in note for note in invalid_limit_notes)

    incomplete = result.model_copy(
        update={"finalAnalysis": FinalAnalysisHandoff.get_blank()}
    )
    with pytest.raises(ValueError, match="lacks its selection"):
        report_module._collect_final_artifacts(
            store,
            incomplete,
            tmp_path / "plots-incomplete",
        )


def test_data_enrichment_cache_rollback_and_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = AssayFeatureInspection.get_example()
    completed = DataEnrichmentDependencies(
        store=object(),
        assays=["RNA"],
        inspections={"RNA": inspection},
        toolCalls=[
            DataEnrichmentToolCall(
                name="inspect_assay_features_batch",
                assay="all",
            )
        ],
    )
    completed_context = SimpleNamespace(deps=completed)

    assert (
        asyncio.run(
            enrichment_module.inspect_assay_features(
                completed_context,
                assay_name="RNA",
            )
        )
        == inspection
    )
    cached_batch = asyncio.run(
        enrichment_module.inspect_assay_features_batch(completed_context)
    )
    assert cached_batch.inspections == [inspection]
    assert cached_batch.evidenceIds == inspection.evidenceIds

    incomplete = DataEnrichmentDependencies(
        assays=["RNA"],
        toolCalls=[DataEnrichmentToolCall(name="sentinel", assay="RNA")],
    )
    with pytest.raises(ModelRetry, match="datastore"):
        asyncio.run(
            enrichment_module.inspect_assay_features_batch(
                SimpleNamespace(deps=incomplete)
            )
        )
    assert [call.name for call in incomplete.toolCalls] == ["sentinel"]

    provider_error = UnexpectedModelBehavior("provider output failed")
    with pytest.raises(UnexpectedModelBehavior, match="provider output failed"):
        enrichment_module.fallback_data_enrichment_report(
            DataEnrichmentDependencies(assays=["RNA"]),
            error=provider_error,
            model_name="test-model",
        )
    fallback = enrichment_module.fallback_data_enrichment_report(
        DataEnrichmentDependencies(
            assays=["RNA"],
            inspections={"RNA": inspection},
            evidenceIds=set(inspection.evidenceIds),
        ),
        error=provider_error,
        model_name="test-model",
    )
    assert fallback.policies[0].species == "homo_sapiens"
    assert fallback.policies[0].speciesConfidence == "high"

    def fail_before_inspection(**_kwargs: object) -> object:
        raise UnexpectedModelBehavior("no inspection completed")

    monkeypatch.setattr(enrichment_module, "run_agent_sync", fail_before_inspection)
    store = SimpleNamespace(assay_names=["RNA"])
    with pytest.raises(UnexpectedModelBehavior, match="no inspection completed"):
        DataEnrichmentAgent(object()).run(store)


def test_biological_interpretation_cache_and_fallback_branches() -> None:
    composition = ClusterCompositionEvidence.get_example()
    composition_deps = BiologicalInterpretationDependencies(
        compositionEvidence=composition
    )
    assert (
        asyncio.run(
            biological_module.inspect_cluster_composition(
                SimpleNamespace(deps=composition_deps)
            )
        )
        == composition
    )

    marker = ClusterMarkerEvidence.get_example()
    marker_deps = BiologicalInterpretationDependencies(
        clusterValues={marker.clusterId: 0},
        markerEvidence={marker.clusterId: marker},
    )
    assert (
        asyncio.run(
            biological_module.inspect_cluster_markers(
                SimpleNamespace(deps=marker_deps),
                cluster_id=marker.clusterId,
            )
        )
        == marker
    )

    invalid_report = BiologicalInterpretationReport(
        status="done",
        needsInput=BiologicalInterpretationNeedsInput(question="More context?"),
    )
    with pytest.raises(ModelRetry, match="Only a needsInput"):
        biological_module.validate_biological_interpretation_report(
            invalid_report,
            BiologicalInterpretationDependencies(clusterValues={"0": 0}),
        )

    provider_error = UnexpectedModelBehavior("structured output failed")
    with pytest.raises(UnexpectedModelBehavior, match="structured output failed"):
        biological_module.fallback_biological_interpretation_report(
            BiologicalInterpretationDependencies(),
            error=provider_error,
            model_name="test-model",
        )
    needs_markers = biological_module.fallback_biological_interpretation_report(
        BiologicalInterpretationDependencies(
            clusterValues={"0": 0},
            evidenceIds={"composition:clusters"},
        ),
        error=provider_error,
        model_name="test-model",
    )
    assert needs_markers.status == "needsInput"
    assert needs_markers.needsInput is not None
    assert needs_markers.evidenceIds == ["composition:clusters"]


def test_experimental_context_rejects_invalid_batches_and_builds_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_batches = (
        (
            [{"name": "batch", "domain": "technical", "kind": "categorical"}],
            "missing",
            "Unknown batch column",
        ),
        (
            [{"name": "condition", "domain": "biological", "kind": "categorical"}],
            "condition",
            "must be classified as technical",
        ),
        (
            [{"name": "depth", "domain": "technical", "kind": "continuous"}],
            "depth",
            "must be categorical",
        ),
    )
    for columns, batch_column, message in invalid_batches:
        deps = ExperimentalContextDependencies(
            characterization=CovariateCharacterization(
                status="done",
                columns=columns,
            )
        )
        with pytest.raises(ModelRetry, match=message):
            asyncio.run(
                experimental_module.analyze_experimental_design(
                    SimpleNamespace(deps=deps),
                    column_domains={},
                    coefficients_of_interest=[],
                    units_of_inference={},
                    batch_columns=[batch_column],
                )
            )

    characterization = CovariateCharacterization(
        status="done",
        columns=[{"name": "condition", "domain": "biological", "kind": "categorical"}],
    )
    monkeypatch.setattr(
        experimental_module,
        "characterize_covariates",
        lambda *_args, **_kwargs: characterization,
    )

    def offer_profile(
        deps: ExperimentalContextDependencies,
        _characterization: CovariateCharacterization,
    ) -> list[CellQcProfileEvidence]:
        profile = CellQcProfileEvidence.get_example()
        deps.qcProfiles[profile.profileId] = profile
        return [profile]

    monkeypatch.setattr(experimental_module, "_offered_qc_profiles", offer_profile)
    fallback_deps = ExperimentalContextDependencies(
        cellSelection=ArtifactReferenceModel(
            scope="datastore",
            kind="cell_selection",
            artifactId="c" * 64,
        ),
        htoIdentityColumns=["hto_identity"],
    )
    fallback = experimental_module.fallback_experimental_context_result(
        fallback_deps,
        error=UnexpectedModelBehavior("design output failed"),
        model_name="test-model",
    )
    assert fallback.status == "done"
    assert fallback_deps.characterization is characterization
    assert fallback.cellQc.profileId == CellQcProfileEvidence.get_example().profileId


def test_agent_execution_logs_nested_failures_for_sync_and_async_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAgent:
        async def __aenter__(self) -> "FailingAgent":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def run(self, *_args: object, **_kwargs: object) -> object:
            try:
                raise ValueError("inner failure")
            except ValueError as cause:
                raise RuntimeError("outer failure") from cause

    monkeypatch.setattr(
        agent_exec_module,
        "_build_agent",
        lambda **_kwargs: FailingAgent(),
    )
    messages: list[str] = []
    monkeypatch.setattr(agent_exec_module.logger, "error", messages.append)
    with pytest.raises(RuntimeError, match="outer failure"):
        agent_exec_module.run_agent_sync(
            model=object(),
            output_type=dict,
            system_prompt="system",
            user_prompt="user",
            name="sync-failure",
        )
    with pytest.raises(RuntimeError, match="outer failure"):
        asyncio.run(
            agent_exec_module.run_agent(
                model=object(),
                output_type=dict,
                system_prompt="system",
                user_prompt="user",
                name="async-failure",
            )
        )
    assert all("caused by ValueError: inner failure" in message for message in messages)
    assert "sync-failure" in messages[0]
    assert "async-failure" in messages[1]


def test_journal_retryable_error_handles_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    assert journal_module.is_retryable_model_error(RuntimeError("unavailable")) is False
