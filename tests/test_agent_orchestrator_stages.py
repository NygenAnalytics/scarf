"""Context, preprocessing, tuning, integration, and finalization contracts."""

import uuid
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scarf.agent.orchestrator.context as context_module
import scarf.agent.orchestrator.journal as journal_module
import scarf.agent.orchestrator.tuning as tuning_module
import scarf.agent.parameter_tuning as parameter_tuning_module
from scarf.agent.config import AgentRunConfig
from scarf.agent.data_enrichment import (
    AssayFeatureInspection,
    DataEnrichmentReport,
    FeatureFamilyEvidence,
    FeatureReference,
    FeatureSelectionPolicy,
)
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    CellQcPlan,
    CellQcProfileEvidence,
    ExperimentalContextResult,
    NamedArtifactSource,
)
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    PreprocessedAssayHandoff,
    WorkflowStageAttempt,
    WorkflowStageLink,
)
from scarf.agent.orchestrator.models import OrchestrationRequestRecord
from scarf.agent.persistence import (
    AgentInvocation,
    AgentReportReference,
    create_agent_workflow,
    list_agent_reports,
    load_agent_record,
    load_agent_workflow,
    save_agent_report,
)
from scarf.agent.parameter_tuning import (
    ArtifactRecord,
    FinalGraphNeedsInput,
    FinalGraphSelection,
    IntegrationCandidateEvaluation,
    IntegrationMetrics,
    ParameterCandidateEvaluation,
    ParameterTuningReport,
    final_graph_options,
    finalize_parameter_tuning_selection,
    select_final_parameter_graph,
)
from scarf.agent.types import (
    AgentRunInfo,
    ArtifactReferenceModel,
    BatchSafetyEvidence,
    ExperimentalTuningHandoff,
)
from scarf.datastore.datastore import DataStore
from scarf.storage.refs import ArtifactRef
from tests.agent_orchestrator_store import create_store


_PLAN_CHECKSUM = "a" * 64


def _cell_selection_model() -> ArtifactReferenceModel:
    return ArtifactReferenceModel(
        scope="datastore",
        kind="cell_selection",
        artifactId="c" * 64,
    )


class _FeatureTable:
    def __init__(self, ids: list[str], names: list[str]) -> None:
        self._values = {
            "ids": np.asarray(ids),
            "names": np.asarray(names),
        }

    def fetch_all(self, column: str) -> np.ndarray:
        return self._values[column]


class _PlanningStore:
    """Narrow datastore surface consumed by preprocessing-plan construction."""

    def __init__(
        self,
        assays: Mapping[str, tuple[str, list[str], list[str]]],
        *,
        active_cells: int = 100,
    ) -> None:
        self.assay_names = list(assays)
        self._assays = {
            name: SimpleNamespace(feats=_FeatureTable(ids, names))
            for name, (_assay_type, ids, names) in assays.items()
        }
        self._summary = SimpleNamespace(
            active_cells=active_cells,
            assays=[
                SimpleNamespace(
                    name=name,
                    assay_type=assay_type,
                    total_features=len(ids),
                )
                for name, (assay_type, ids, _names) in assays.items()
            ],
        )

    def summary(self) -> Any:
        return self._summary

    def get_assay(self, name: str) -> Any:
        return self._assays[name]


def _modality_policy(
    assay: str,
    assay_type: str,
    *,
    controls: list[FeatureReference] | None = None,
    exclude_features: list[str] | None = None,
    artificial_features: list[str] | None = None,
    peak_status: str = "notApplicable",
) -> FeatureSelectionPolicy:
    supported = assay_type in {"RNA", "ATAC", "ADT"}
    modality = (
        assay_type if assay_type in {"RNA", "ATAC", "ADT", "HTO"} else "unsupported"
    )
    return FeatureSelectionPolicy(
        assay=assay,
        assayType=assay_type,
        assayModality=modality,
        graphEligible=supported,
        markerEligible=supported,
        demultiplexEligible=assay_type == "HTO",
        exactControlFeatures=controls or [],
        excludeFeatures=exclude_features or [],
        artificialFeatures=artificial_features or [],
        peakCoordinateStatus=peak_status,
        evidenceIds=[f"assay:{assay}:modality"],
    )


def _planning_inputs(
    assays: Mapping[str, tuple[str, list[str], list[str]]],
    *,
    controls: Mapping[str, list[FeatureReference]] | None = None,
    exclude_features: Mapping[str, list[str]] | None = None,
    artificial_features: Mapping[str, list[str]] | None = None,
    peak_statuses: Mapping[str, str] | None = None,
    primary_assay: str | None = None,
    marker_assay: str | None = None,
    analysis_assays: list[str] | None = None,
    config: AutomatedWorkflowConfig | None = None,
) -> tuple[
    _PlanningStore,
    OrchestrationRequestRecord,
    DataEnrichmentReport,
    ExperimentalContextResult,
    WorkflowStageAttempt,
]:
    store = _PlanningStore(assays)
    policies = [
        _modality_policy(
            name,
            assay_type,
            controls=(controls or {}).get(name),
            exclude_features=(exclude_features or {}).get(name),
            artificial_features=(artificial_features or {}).get(name),
            peak_status=(peak_statuses or {}).get(name, "notApplicable"),
        )
        for name, (assay_type, _ids, _names) in assays.items()
    ]
    enrichment = DataEnrichmentReport(status="done", policies=policies)
    request = AutomatedWorkflowRequest(
        sourcePath="dataset.zarr",
        zarrPath="dataset.zarr",
        studyContext="A bounded plan-construction test.",
        allowAssumptions=True,
        primaryAssay=primary_assay,
        markerAssay=marker_assay,
        analysisAssays=analysis_assays or list(assays),
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId="planning-test",
        request=request,
        config=config or AutomatedWorkflowConfig(),
    )
    experimental = ExperimentalContextResult.get_example()
    ingest_outcome = WorkflowStageAttempt(
        workflowRunId="planning-test",
        stage="ingest",
        attemptId="ingest-test",
        status="done",
        startedAtNs=1,
        completedAtNs=2,
        outputs={"format": "zarr"},
    )
    return store, request_record, enrichment, experimental, ingest_outcome


def _build_plan(
    assays: Mapping[str, tuple[str, list[str], list[str]]],
    **kwargs: Any,
) -> AutomatedPreprocessingPlan:
    inputs = _planning_inputs(assays, **kwargs)
    return AgentOrchestrator(object()).build_preprocessing_plan(*inputs)


def _native_assay_report(assay: str, token: int) -> ParameterTuningReport:
    evaluation = ParameterCandidateEvaluation.get_example().model_copy(
        update={
            "artifacts": {
                "neighbors": ArtifactRecord(
                    assay=assay,
                    kind="neighbors",
                    artifactId=f"{token + 2:064x}",
                ),
                "connectivityMap": ArtifactRecord(
                    assay=assay,
                    kind="connectivity_map",
                    artifactId=f"{token:064x}",
                ),
                "clusters": ArtifactRecord(
                    assay=assay,
                    kind="cluster_labels",
                    artifactId=f"{token + 1:064x}",
                ),
            },
            "cellSelection": _cell_selection_model(),
            "clusterColumn": f"{assay}_agent_clusters",
            "evidenceIds": ["candidate:baseline:clusters"],
        }
    )
    return ParameterTuningReport(
        status="done",
        fromAssay=assay,
        cellSelection=_cell_selection_model(),
        evaluations=[evaluation],
        recommendedCandidateId=evaluation.candidateId,
        selectedArtifacts=dict(evaluation.artifacts),
        evidenceIds=list(evaluation.evidenceIds),
        stopReason="The bounded screen completed.",
    )


def _native_batch_report(*assays: str) -> ParameterTuningReport:
    reports = {
        assay: _native_assay_report(assay, index * 10 + 1)
        for index, assay in enumerate(assays)
    }
    primary = reports[assays[0]]
    return ParameterTuningReport(
        status="done",
        fromAssay=assays[0],
        cellSelection=_cell_selection_model(),
        evaluations=list(primary.evaluations),
        recommendedCandidateId=primary.recommendedCandidateId,
        selectedArtifacts=dict(primary.selectedArtifacts),
        assayReports=reports,
        recommendedByAssay={
            assay: report.recommendedCandidateId or ""
            for assay, report in reports.items()
        },
        totalCandidates=sum(len(report.evaluations) for report in reports.values()),
        graphAssay=assays[0],
        markerAssay=assays[0],
        runInfo=AgentRunInfo(
            agentName="parameter_tuning",
            runId=uuid.uuid4().hex,
        ),
    )


def _eligible_integration() -> IntegrationCandidateEvaluation:
    return IntegrationCandidateEvaluation(
        integrationId="wnn_resolution_1",
        method="wnn",
        assays=["RNA", "ADT"],
        status="done",
        eligible=True,
        cellSelection=_cell_selection_model(),
        resolution=1.0,
        graphArtifact=ArtifactRecord(
            scope="datastore",
            kind="integrated_graph",
            artifactId="8" * 64,
        ),
        clusterArtifact=ArtifactRecord(
            scope="datastore",
            kind="cluster_labels",
            artifactId="9" * 64,
        ),
        clusterColumn="agent_wnn_cluster",
        metrics=IntegrationMetrics(
            nClusters=2,
            minClusterCells=20,
            modalityWeightsValid=True,
        ),
        evidenceIds=["integration:wnn_resolution_1:clusters"],
    )


def test_unsafe_experimental_context_pauses_and_explicit_skip_reuses_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = create_store(tmp_path / "unsafe-context.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = create_agent_workflow(store, workflow_run_id="unsafe-context")
    cell_selection = ArtifactReferenceModel.from_artifact_ref(
        store.snapshot_cell_selection("I")
    )
    enrichment = DataEnrichmentReport.get_example().model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="data_enrichment",
                runId=uuid.uuid4().hex,
            )
        }
    )
    enrichment_reference = save_agent_report(
        store,
        workflow.workflowRunId,
        enrichment,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"studyContext": "A deliberately confounded study."},
        ),
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="Treatment is confounded with batch.",
            allowAssumptions=True,
        ),
    )
    example = ExperimentalContextResult.get_example()
    evidence_id = "batchEstimability:treatment:batch"
    unsafe_plan = example.decision.batchCorrection.model_copy(
        update={
            "action": "unsafe",
            "evidenceIds": [evidence_id],
        }
    )
    unsafe_report = example.model_copy(
        update={
            "cellSelection": cell_selection,
            "qualityMetricArtifacts": [],
            "htoIdentityArtifacts": [],
            "decision": example.decision.model_copy(
                update={"batchCorrection": unsafe_plan}
            ),
            "batchSafety": [
                BatchSafetyEvidence.get_example().model_copy(
                    update={
                        "status": "unsafe",
                        "estimability": {
                            "status": "ok",
                            "coefficientEstimable": False,
                            "rankDeficient": True,
                        },
                    }
                )
            ],
            "runInfo": AgentRunInfo(
                agentName="experimental_context",
                runId=uuid.uuid4().hex,
            ),
        }
    )

    class UnsafeAgent:
        calls = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.config = AgentRunConfig()

        def run(self, *_args: Any, **_kwargs: Any) -> ExperimentalContextResult:
            type(self).calls += 1
            return unsafe_report

    monkeypatch.setattr(context_module, "ExperimentalContextAgent", UnsafeAgent)
    orchestrator = AgentOrchestrator(object())
    paused_outcome, paused_report = orchestrator.experimental_context_stage(
        store,
        workflow,
        request_record,
        [],
        cell_selection,
        enrichment_reference,
        [],
        [],
        {},
    )

    assert paused_report.status == "done"
    assert paused_outcome.status == "needsInput"
    assert paused_outcome.outputs["unsafeBatchCorrection"] is True
    assert paused_outcome.needsInput is not None
    assert paused_outcome.needsInput.questions[0].options == [
        "skipHarmony",
        "provideClarification",
    ]
    assert journal_module._resume_answer_errors(
        paused_outcome,
        {"experimentalDirections": "unsafe"},
    )
    assert not journal_module._resume_answer_errors(
        paused_outcome,
        {"experimentalDirections": "skipHarmony"},
    )
    assert not journal_module._resume_answer_errors(
        paused_outcome,
        {
            "experimentalDirections": {
                "selection": "provideClarification",
                "clarification": "Batch denotes sequencing lane within each donor.",
            }
        },
    )

    done_outcome, resolved_report = orchestrator.experimental_context_stage(
        store,
        workflow,
        request_record,
        [],
        cell_selection,
        enrichment_reference,
        [],
        [],
        {"experimentalDirections": "skipHarmony"},
    )

    assert done_outcome.status == "done"
    assert done_outcome.actions == ["resolve_unsafe_batch_correction:skip"]
    assert resolved_report.decision.batchCorrection.action == "skip"
    assert resolved_report.decision.batchCorrection.batchColumns == []
    assert resolved_report.decision.batchCorrection.preserveColumns == (
        unsafe_plan.preserveColumns
    )
    assert UnsafeAgent.calls == 1


def test_preprocessing_plan_routes_supported_modalities_and_skips_others() -> None:
    assays = {
        "peaks": (
            "ATAC",
            ["chr1:1-10", "chr1:20-30", "chr2:1-20"],
            ["peak-1", "peak-2", "peak-3"],
        ),
        "tags": ("HTO", ["tag-1", "tag-2"], ["sample-1", "sample-2"]),
        "proteins": (
            "ADT",
            ["adt-1", "adt-2", "adt-3"],
            ["CD3", "CD19", "CD45"],
        ),
        "custom": ("CRISPR", ["guide-1"], ["guide-1"]),
        "transcriptome": (
            "RNA",
            ["gene-1", "gene-2", "gene-3", "gene-4"],
            ["A", "B", "C", "D"],
        ),
    }

    plan = _build_plan(assays)
    routes = {value.assay: value for value in plan.assays}

    assert (plan.primaryAssay, plan.markerAssay) == (
        "transcriptome",
        "transcriptome",
    )
    assert plan.cellSelection == _cell_selection_model()
    assert (
        routes["transcriptome"].role,
        routes["transcriptome"].featureMethod,
        routes["transcriptome"].reductionMethod,
    ) == ("graph", "hvg", "pca")
    assert (
        routes["peaks"].role,
        routes["peaks"].featureMethod,
        routes["peaks"].reductionMethod,
        routes["peaks"].reductionParameters["skipFirst"],
    ) == ("graph", "prevalentPeaks", "lsi", True)
    assert (
        routes["proteins"].role,
        routes["proteins"].featureMethod,
        routes["proteins"].reductionMethod,
    ) == ("graph", "panel", "identity")
    assert routes["tags"].role == "hto"
    assert not routes["tags"].graphEligible
    assert not routes["tags"].markerEligible
    assert routes["tags"].reductionMethod == "none"
    assert routes["tags"].normalizationParameters == {}
    assert routes["custom"].role == "unsupported"
    assert not routes["custom"].graphEligible
    assert any("Unsupported assay 'custom'" in value for value in plan.limitations)


def test_converted_input_preserves_exact_selection_and_typed_qc() -> None:
    inputs = list(
        _planning_inputs(
            {
                "RNA": (
                    "RNA",
                    ["gene-1", "gene-2", "gene-3"],
                    ["A", "B", "C"],
                )
            }
        )
    )
    request_record = inputs[1]
    request = request_record.request.model_copy(update={"sourcePath": "dataset.h5ad"})
    inputs[1] = request_record.model_copy(update={"request": request})
    inputs[4] = inputs[4].model_copy(update={"outputs": {"format": "h5ad"}})

    plan = AgentOrchestrator(object()).build_preprocessing_plan(*inputs)

    assert plan.cellSelection == inputs[3].cellSelection
    assert isinstance(plan.cellQc, CellQcPlan)


@pytest.mark.parametrize(
    ("pairing_provenance", "explicit_pairing", "expected"),
    [
        (None, [], []),
        ("singleSourceSharedCellAxis", [], ["RNA", "ADT"]),
        (None, ["RNA", "ADT"], ["RNA", "ADT"]),
    ],
)
def test_multimodal_pairing_requires_persisted_or_explicit_provenance(
    pairing_provenance: str | None,
    explicit_pairing: list[str],
    expected: list[str],
) -> None:
    inputs = list(
        _planning_inputs(
            {
                "RNA": (
                    "RNA",
                    ["gene-1", "gene-2", "gene-3"],
                    ["A", "B", "C"],
                ),
                "ADT": (
                    "ADT",
                    ["adt-1", "adt-2", "adt-3"],
                    ["CD3", "CD19", "CD45"],
                ),
            },
            primary_assay="RNA",
        )
    )
    request_record = inputs[1]
    inputs[1] = request_record.model_copy(
        update={
            "request": request_record.request.model_copy(
                update={"pairedAssays": explicit_pairing}
            )
        }
    )
    inputs[4] = inputs[4].model_copy(
        update={
            "outputs": {
                "format": "h5ad" if pairing_provenance else "zarr",
                "pairingProvenance": pairing_provenance,
            }
        }
    )

    plan = AgentOrchestrator(object()).build_preprocessing_plan(*inputs)

    assert plan.pairedAssays == expected


def test_percent_features_follow_deterministic_inspection_not_policy_lists(
    tmp_path: Path,
) -> None:
    path = create_store(tmp_path / "inspected-families.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    cell_selection = ArtifactReferenceModel.from_artifact_ref(
        store.snapshot_cell_selection("I")
    )
    assert "RNA_percentMito" not in store.cells.columns
    workflow = create_agent_workflow(store, workflow_run_id="inspected-families")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic feature-family test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    policy = _modality_policy("RNA", "RNA")
    assert policy.excludeFamilies == []
    assert policy.protectFamilies == []
    enrichment = DataEnrichmentReport(
        status="done",
        policies=[policy],
        inspections=[
            AssayFeatureInspection(
                assay="RNA",
                families=[
                    FeatureFamilyEvidence(
                        family="mitochondrial",
                        count=1,
                        examples=["MT-CO1"],
                        evidenceId="assay:RNA:family:mitochondrial",
                    )
                ],
            )
        ],
    )

    orchestrator = AgentOrchestrator(object())
    outcome = orchestrator._hto_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment,
        cell_selection,
    )

    assert outcome.status == "done"
    assert "RNA_percentMito" not in store.cells.columns
    assert "compute_percent_mito:RNA" in outcome.actions
    metric_source = NamedArtifactSource.model_validate(
        outcome.outputs["qualityMetricArtifacts"][0]
    )
    assert metric_source.name == "RNA_percentMito"
    assert metric_source.artifact.kind == "quality_metric"
    assert outcome.artifacts[metric_source.name] == metric_source.artifact
    assert orchestrator._named_stage_artifacts(
        outcome,
        "qualityMetricArtifacts",
        "quality_metric",
    ) == [metric_source]
    operation = outcome.outputs["operations"][0]
    assert operation["operation"] == "run_feature_percentage"
    assert operation["cellSelection"] == cell_selection.model_dump(mode="json")
    assert operation["features"]["kind"] == "feature_selection"
    assert operation["artifact"] == metric_source.artifact.model_dump(mode="json")

    profile = CellQcProfileEvidence(
        profileId="cellQc:RNA:RNA:globalGaussian:0.01:0.99",
        action="globalGaussian",
        driverAssay="RNA",
        driverAssayType="RNA",
        attributes=["RNA_nCounts"],
        artifactMetrics=[metric_source],
        parameters={"minP": 0.01, "maxP": 0.99},
        activeCells=4,
        retainedCells=2,
        retainedFraction=0.5,
        evidenceId="qcProfile:artifact-backed-global",
    )
    plan = CellQcPlan(
        action=profile.action,
        profileId=profile.profileId,
        driverAssay=profile.driverAssay,
        driverAssayType=profile.driverAssayType,
        attributes=profile.attributes,
        artifactMetrics=profile.artifactMetrics,
        evidenceIds=[profile.evidenceId],
    )
    experimental = ExperimentalContextResult.get_blank().model_copy(
        update={
            "cellSelection": cell_selection,
            "cellQc": plan,
            "qcProfiles": [profile],
            "qualityMetricArtifacts": [metric_source],
        }
    )
    qc_actions: list[str] = []
    qc_operations: list[dict[str, Any]] = []
    filtered = AgentOrchestrator(object()).apply_cell_qc(
        store,
        experimental,
        ArtifactRef(
            scope="datastore",
            kind="cell_selection",
            artifact_id=cell_selection.artifactId,
        ),
        qc_actions,
        qc_operations,
    )

    filtered_status = store.inspect_artifact(filtered)
    assert filtered.kind == "cell_selection"
    assert filtered_status.inputs["artifact_metrics"] == {
        metric_source.name: ArtifactRef(
            scope=metric_source.artifact.scope,
            assay=metric_source.artifact.assay,
            kind=metric_source.artifact.kind,
            artifact_id=metric_source.artifact.artifactId,
        ).to_dict()
    }
    assert qc_operations[0]["artifactMetrics"] == [
        metric_source.model_dump(mode="json")
    ]
    assert "RNA_percentMito" not in store.cells.columns


def test_hto_demultiplexing_is_checkpointed_once_and_never_graph_bearing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = create_store(tmp_path / "hto-once.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    cell_selection = ArtifactReferenceModel.from_artifact_ref(
        store.snapshot_cell_selection("I")
    )
    workflow = create_agent_workflow(store, workflow_run_id="hto-once")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic HTO checkpoint test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    calls = 0

    identity_ref = ArtifactRef(
        scope="assay",
        assay="HTO",
        kind="hto_identity",
        artifact_id="d" * 64,
    )

    def run_hto(
        selection: ArtifactRef,
        **_kwargs: Any,
    ) -> ArtifactRef:
        nonlocal calls
        assert selection == ArtifactRef(
            scope="datastore",
            kind="cell_selection",
            artifact_id=cell_selection.artifactId,
        )
        calls += 1
        return identity_ref

    load_artifact = store.load_artifact
    monkeypatch.setattr(store, "run_hto_demultiplexing", run_hto)
    monkeypatch.setattr(
        store,
        "load_artifact",
        lambda reference: (
            {"values": np.asarray(["negative", "singlet", "doublet", "singlet"])}
            if reference == identity_ref
            else load_artifact(reference)
        ),
    )
    enrichment = DataEnrichmentReport(
        status="done",
        policies=[
            FeatureSelectionPolicy(
                assay="HTO",
                assayType="HTO",
                assayModality="HTO",
                demultiplexEligible=True,
                exactTagFeatures=[
                    FeatureReference(featureId="tag-1", featureName="sample-1"),
                    FeatureReference(featureId="tag-2", featureName="sample-2"),
                ],
                evidenceIds=["assay:HTO:modality"],
            )
        ],
    )
    orchestrator = AgentOrchestrator(object())

    first = orchestrator._hto_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment,
        cell_selection,
    )
    second = orchestrator._hto_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment,
        cell_selection,
    )

    assert first == second
    assert calls == 1
    assert first.outputs["operations"][0]["operation"] == "run_hto_demultiplexing"
    assert first.artifacts["HTO_htoIdentity"].artifactId == identity_ref.artifact_id
    assert "htoIdentityColumns" not in first.outputs
    identity_source = NamedArtifactSource.model_validate(
        first.outputs["htoIdentityArtifacts"][0]
    )
    assert identity_source.name == "HTO_htoIdentity"
    assert identity_source.artifact == first.artifacts[identity_source.name]
    assert orchestrator._named_stage_artifacts(
        first,
        "htoIdentityArtifacts",
        "hto_identity",
    ) == [identity_source]
    missing_source = first.model_copy(
        update={
            "outputs": {
                **first.outputs,
                "htoIdentityArtifacts": [],
            }
        }
    )
    with pytest.raises(ValueError, match="must name every persisted"):
        orchestrator._named_stage_artifacts(
            missing_source,
            "htoIdentityArtifacts",
            "hto_identity",
        )
    assert all("graph" not in action for action in first.actions)

    enrichment_reference = save_agent_report(
        store,
        workflow.workflowRunId,
        enrichment,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"assays": ["HTO"]},
        ),
    )
    captured_sources: list[NamedArtifactSource] = []
    context_example = ExperimentalContextResult.get_example()
    context_report = context_example.model_copy(
        update={
            "cellSelection": cell_selection,
            "cellQc": CellQcPlan(),
            "qcProfiles": [],
            "qualityMetricArtifacts": [],
            "htoIdentityColumns": [],
            "htoIdentityArtifacts": [identity_source],
            "decision": context_example.decision.model_copy(
                update={"cellQc": CellQcPlan()}
            ),
        }
    )

    class CapturingContextAgent:
        calls = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.config = AgentRunConfig()

        def run(
            self,
            *_args: Any,
            quality_metric_artifacts: list[NamedArtifactSource],
            hto_identity_artifacts: list[NamedArtifactSource],
            **_kwargs: Any,
        ) -> ExperimentalContextResult:
            type(self).calls += 1
            assert quality_metric_artifacts == []
            captured_sources.extend(hto_identity_artifacts)
            return context_report

    monkeypatch.setattr(
        context_module,
        "ExperimentalContextAgent",
        CapturingContextAgent,
    )
    context_outcome, _ = orchestrator.experimental_context_stage(
        store,
        workflow,
        request_record,
        [journal_module._parent_link(first)],
        cell_selection,
        enrichment_reference,
        [],
        [identity_source],
        {},
    )

    assert context_outcome.status == "done"
    assert captured_sources == [identity_source]
    assert context_outcome.artifacts[identity_source.name] == identity_source.artifact
    context_record = load_agent_record(
        store,
        context_outcome.reportReferences[0],
    )
    assert (
        context_record.invocation.artifacts[identity_source.name]
        == identity_source.artifact
    )
    reused_context, reused_report = orchestrator.experimental_context_stage(
        store,
        workflow,
        request_record,
        [journal_module._parent_link(first)],
        cell_selection,
        enrichment_reference,
        [],
        [identity_source],
        {},
    )
    assert reused_context == context_outcome
    assert reused_report == context_report
    assert CapturingContextAgent.calls == 1


def test_selected_sample_mad_qc_passes_exact_artifact_sources() -> None:
    cell_selection = ArtifactReferenceModel(
        scope="datastore",
        kind="cell_selection",
        artifactId="c" * 64,
    )
    metric = NamedArtifactSource(
        name="RNA_percentMito",
        artifact=ArtifactReferenceModel(
            scope="assay",
            assay="RNA",
            kind="quality_metric",
            artifactId="d" * 64,
        ),
    )
    sample = NamedArtifactSource(
        name="HTO_htoIdentity",
        artifact=ArtifactReferenceModel(
            scope="assay",
            assay="HTO",
            kind="hto_identity",
            artifactId="e" * 64,
        ),
    )
    profile = CellQcProfileEvidence(
        profileId="cellQc:RNA:RNA:sampleMad:artifact:HTO_htoIdentity",
        action="sampleMad",
        driverAssay="RNA",
        driverAssayType="RNA",
        sampleArtifact=sample,
        attributes=["RNA_nCounts"],
        artifactMetrics=[metric],
        parameters={"nMads": 4.0, "minCellsPerSample": 12},
        activeCells=100,
        retainedCells=95,
        retainedFraction=0.95,
        evidenceId="qcProfile:artifact-sample",
    )
    plan = CellQcPlan(
        action=profile.action,
        profileId=profile.profileId,
        driverAssay=profile.driverAssay,
        driverAssayType=profile.driverAssayType,
        sampleArtifact=profile.sampleArtifact,
        attributes=profile.attributes,
        artifactMetrics=profile.artifactMetrics,
        evidenceIds=[profile.evidenceId],
    )
    experimental = ExperimentalContextResult.get_blank().model_copy(
        update={
            "cellSelection": cell_selection,
            "cellQc": plan,
            "qcProfiles": [profile],
            "qualityMetricArtifacts": [metric],
            "htoIdentityArtifacts": [sample],
        }
    )
    output_selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="f" * 64,
    )
    captured: dict[str, Any] = {}

    class Store:
        def auto_filter_cells(self, **kwargs: Any) -> ArtifactRef:
            captured.update(kwargs)
            return output_selection

    actions: list[str] = []
    operations: list[dict[str, Any]] = []
    result = AgentOrchestrator(object()).apply_cell_qc(
        Store(),
        experimental,
        ArtifactRef(
            scope="datastore",
            kind="cell_selection",
            artifact_id=cell_selection.artifactId,
        ),
        actions,
        operations,
    )

    assert result == output_selection
    assert captured["attrs"] == ["RNA_nCounts"]
    assert captured["artifact_metrics"][0].name == metric.name
    assert captured["artifact_metrics"][0].artifact.artifact_id == (
        metric.artifact.artifactId
    )
    assert captured["sample_column"] is None
    assert captured["sample_artifact"].name == sample.name
    assert captured["sample_artifact"].artifact.artifact_id == (
        sample.artifact.artifactId
    )
    assert captured["n_mads"] == 4.0
    assert captured["min_cells_per_sample"] == 12
    assert operations[0]["sampleArtifact"] == sample.model_dump(mode="json")
    assert operations[0]["artifactMetrics"] == [metric.model_dump(mode="json")]


@pytest.mark.parametrize(
    ("assay_types", "explicit", "expected"),
    [
        (["ATAC", "ADT", "RNA"], None, "RNA"),
        (["ATAC", "ADT"], None, "ADT"),
        (["ATAC"], None, "ATAC"),
        (["RNA", "ADT", "ATAC"], "ATAC", "ATAC"),
    ],
)
def test_marker_assay_precedence(
    assay_types: list[str],
    explicit: str | None,
    expected: str,
) -> None:
    assays = {
        assay_type: (
            assay_type,
            [f"{assay_type}-1", f"{assay_type}-2", f"{assay_type}-3"],
            [f"{assay_type}-1", f"{assay_type}-2", f"{assay_type}-3"],
        )
        for assay_type in assay_types
    }

    plan = _build_plan(
        assays,
        primary_assay=assay_types[0],
        marker_assay=explicit,
    )

    assert plan.primaryAssay == assay_types[0]
    assert plan.markerAssay == expected


def test_adt_identity_limit_and_exact_observed_control_exclusion() -> None:
    assays = {
        "ADT": (
            "ADT",
            ["adt-1", "control-id", "adt-2", "adt-3"],
            [
                "CD3",
                "Mouse IgG1 isotype control",
                "control response protein",
                "CD19",
            ],
        )
    }
    controls = {
        "ADT": [
            FeatureReference(
                featureId="control-id",
                featureName="Mouse IgG1 isotype control",
            )
        ]
    }

    identity = _build_plan(
        assays,
        controls=controls,
        exclude_features={"ADT": ["adt-1"]},
        artificial_features={"ADT": ["adt-2"]},
        config=AutomatedWorkflowConfig(maxIdentityFeatures=3),
    ).assays[0]
    pca = _build_plan(
        assays,
        controls=controls,
        exclude_features={"ADT": ["adt-1"]},
        artificial_features={"ADT": ["adt-2"]},
        config=AutomatedWorkflowConfig(maxIdentityFeatures=2),
    ).assays[0]

    assert identity.exactExcludedFeatures == [
        "control-id",
        "Mouse IgG1 isotype control",
    ]
    assert identity.reductionMethod == "identity"
    assert identity.reductionParameters["dimensions"] == 3
    assert pca.reductionMethod == "pca"
    assert pca.reductionParameters["dimensions"] == 2


def test_atac_invalid_coordinates_are_limited_without_changing_lsi_route() -> None:
    assays = {
        "ATAC": (
            "ATAC",
            ["chr1:1-20", "not-a-coordinate", "chr2:10-30"],
            ["peak-1", "peak-2", "peak-3"],
        )
    }

    plan = _build_plan(assays, peak_statuses={"ATAC": "invalid"})
    route = plan.assays[0]

    assert route.graphEligible
    assert route.featureMethod == "prevalentPeaks"
    assert route.reductionMethod == "lsi"
    assert route.reductionParameters == {"dimensions": 50, "skipFirst": True}
    assert route.limitations == [
        "ATAC feature coordinates are not uniformly valid chrom:start-end "
        "intervals; the genome build remains unknown"
    ]


@pytest.mark.parametrize(
    ("method", "n_cells", "n_features", "expected_dimensions"),
    [
        ("pca", 5, 3, {2}),
        ("lsi", 6, 4, {3}),
        ("identity", 4, 2, {2}),
    ],
)
def test_initial_candidates_are_rank_valid(
    method: str,
    n_cells: int,
    n_features: int,
    expected_dimensions: set[int],
) -> None:
    orchestrator = AgentOrchestrator(object())
    handoff = PreprocessedAssayHandoff(
        assay="assay",
        assayType="ADT" if method == "identity" else method.upper(),
        reductionMethod=method,
        normalized=ArtifactReferenceModel(
            assay="assay",
            kind="normalized",
            artifactId="1" * 64,
        ),
        nCells=n_cells,
        nFeatures=n_features,
    )

    candidates = orchestrator.initial_parameter_candidates(
        "rank-test",
        handoff,
        count=5,
        neighbors_k=n_cells - 1,
    )

    assert len(candidates) == 5
    assert {value.dimensions for value in candidates} == expected_dimensions
    assert all(2 <= value.dimensions for value in candidates)
    if method != "identity":
        assert all(value.dimensions < min(n_cells, n_features) for value in candidates)
    assert all(2 <= value.neighborsK < n_cells for value in candidates)


def test_initial_candidates_reject_fully_invalid_rank_or_neighbor_count() -> None:
    orchestrator = AgentOrchestrator(object())
    rank_invalid = PreprocessedAssayHandoff(
        assay="RNA",
        assayType="RNA",
        reductionMethod="pca",
        normalized=ArtifactReferenceModel.get_example(),
        nCells=4,
        nFeatures=2,
    )
    identity_invalid = rank_invalid.model_copy(
        update={"assayType": "ADT", "reductionMethod": "identity", "nFeatures": 1}
    )

    with pytest.raises(ValueError, match="no rank-valid graph candidate"):
        orchestrator.initial_parameter_candidates(
            "rank-test",
            rank_invalid,
            count=3,
            neighbors_k=3,
        )
    with pytest.raises(ValueError, match="no rank-valid graph candidate"):
        orchestrator.initial_parameter_candidates(
            "rank-test",
            identity_invalid,
            count=3,
            neighbors_k=3,
        )
    with pytest.raises(ValueError, match="no rank-valid graph candidate"):
        orchestrator.initial_parameter_candidates(
            "rank-test",
            rank_invalid.model_copy(update={"nFeatures": 3}),
            count=3,
            neighbors_k=4,
        )


def test_parameter_tuning_rejects_plan_above_global_branch_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = create_store(tmp_path / "branch-cap.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = create_agent_workflow(store, workflow_run_id="branch-cap")
    config = AutomatedWorkflowConfig(
        primaryInitialCandidates=5,
        maxRefinedCandidatesPerAssay=1,
        maxHarmonyCandidatesPerAssay=0,
        maxCandidateBranches=5,
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A branch-cap test.",
            allowAssumptions=True,
        ),
        config=config,
    )
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        assays=[AssayPreprocessingPlan.get_example()],
    )
    handoff = PreprocessedAssayHandoff.get_example()
    experimental = ExperimentalContextResult.get_example()
    decision = experimental.decision.model_copy(
        update={"batchCorrection": BatchCorrectionPlan(action="skip")}
    )
    experimental = experimental.model_copy(
        update={"decision": decision, "batchSafety": []}
    )

    class UnusedAgent:
        def run_batch(self, *_args: Any, **_kwargs: Any) -> None:
            pytest.fail("The tuning agent must not run above the global branch cap")

    monkeypatch.setattr(
        tuning_module,
        "ParameterTuningAgent",
        lambda *_args, **_kwargs: UnusedAgent(),
    )
    outcome, report = AgentOrchestrator(object()).parameter_tuning_stage(
        store,
        workflow,
        request_record,
        [],
        plan,
        [handoff],
        experimental,
        AgentReportReference.get_example(),
        AgentReportReference.get_example(),
        {},
    )

    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "exceeds the global branch limit 5" in outcome.error
    assert report.status == "failed"
    assert load_agent_workflow(store, workflow.workflowRunId).status == "failed"


def test_final_selection_pause_exposes_exact_options_and_resumes_without_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = create_store(tmp_path / "selection-resume.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = create_agent_workflow(store, workflow_run_id="selection-resume")
    enrichment = DataEnrichmentReport.get_example().model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="data_enrichment",
                runId=uuid.uuid4().hex,
            )
        }
    )
    experimental = ExperimentalContextResult.get_example().model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="experimental_context",
                runId=uuid.uuid4().hex,
            )
        }
    )
    enrichment_reference = save_agent_report(
        store,
        workflow.workflowRunId,
        enrichment,
        invocation=AgentInvocation(
            agentName="data_enrichment",
            inputs={"studyContext": "A final-selection resume test."},
        ),
    )
    experimental_reference = save_agent_report(
        store,
        workflow.workflowRunId,
        experimental,
        invocation=AgentInvocation(
            agentName="experimental_context",
            inputs={"cellSelection": _cell_selection_model().model_dump(mode="json")},
            artifacts={"cellSelection": _cell_selection_model()},
        ),
    )
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A final-selection resume test.",
            allowAssumptions=True,
        ),
    )
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        assays=[
            AssayPreprocessingPlan.get_example(),
            AssayPreprocessingPlan(
                assay="ADT",
                assayType="ADT",
                role="graph",
                graphEligible=True,
                markerEligible=True,
                featureMethod="panel",
                reductionMethod="identity",
            ),
        ],
        pairedAssays=["RNA", "ADT"],
    )
    preprocessed = [
        PreprocessedAssayHandoff.get_example(),
        PreprocessedAssayHandoff(
            assay="ADT",
            assayType="ADT",
            cellSelection=_cell_selection_model(),
            reductionMethod="identity",
            normalized=ArtifactReferenceModel(
                assay="ADT",
                kind="normalized",
                artifactId="2" * 64,
            ),
            nCells=100,
            nFeatures=5,
        ),
    ]
    integration = _eligible_integration()
    calls = {"screen": 0, "promote": 0, "integrate": 0, "selection": 0}

    def selection_execution(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            output=FinalGraphSelection(
                status="needsInput",
                rationale="The supplied evidence does not resolve the final graph.",
                needsInput=FinalGraphNeedsInput(
                    question="An unconstrained provider question.",
                    options=["invented-option"],
                ),
            ),
            runInfo=AgentRunInfo(
                agentName="parameter_tuning_final_graph",
                runId=uuid.uuid4().hex,
            ),
        )

    monkeypatch.setattr(parameter_tuning_module, "run_agent_sync", selection_execution)

    class CountingAgent:
        config = AgentRunConfig()

        def run_batch(self, *_args: Any, **_kwargs: Any) -> ParameterTuningReport:
            calls["screen"] += 1
            return _native_batch_report("RNA", "ADT")

        def promote(self, *_args: Any, **_kwargs: Any) -> None:
            calls["promote"] += 1

        def select_final(
            self,
            *,
            report: ParameterTuningReport,
            integration_evaluations: list[IntegrationCandidateEvaluation],
            marker_assay: str,
        ) -> ParameterTuningReport:
            calls["selection"] += 1
            return select_final_parameter_graph(
                model=object(),
                report=report,
                integration_evaluations=integration_evaluations,
                marker_assay=marker_assay,
                config=self.config,
            )

    monkeypatch.setattr(
        tuning_module,
        "ParameterTuningAgent",
        lambda *_args, **_kwargs: CountingAgent(),
    )
    orchestrator = AgentOrchestrator(object())

    def evaluate_integrations(*_args: Any, **_kwargs: Any) -> list[Any]:
        calls["integrate"] += 1
        return [integration]

    monkeypatch.setattr(orchestrator, "evaluate_integrations", evaluate_integrations)
    real_validated_outcome = journal_module._validated_done_outcome
    paused_attempts: list[WorkflowStageAttempt] = []

    def validated_outcome(
        target_store: Any,
        prefix: str,
        workflow_run_id: str,
        stage: str,
        record: OrchestrationRequestRecord,
        parents: list[WorkflowStageLink],
        *,
        required_status: str = "done",
    ) -> WorkflowStageAttempt | None:
        if stage == "parameter_tuning":
            if required_status == "needsInput" and paused_attempts:
                return paused_attempts[-1]
            return None
        return real_validated_outcome(
            target_store,
            prefix,
            workflow_run_id,
            stage,
            record,
            parents,
            required_status=required_status,
        )

    monkeypatch.setattr(
        journal_module,
        "_validated_done_outcome",
        validated_outcome,
    )
    paused, paused_report = orchestrator.parameter_tuning_stage(
        store,
        workflow,
        request_record,
        [],
        plan,
        preprocessed,
        experimental,
        enrichment_reference,
        experimental_reference,
        {},
    )
    paused_attempts.append(paused)
    expected_options = sorted(final_graph_options(paused_report, [integration]))

    assert paused.status == "needsInput"
    assert paused.needsInput is not None
    assert paused.needsInput.questions[0].questionId == "finalGraphOptionId"
    assert paused.needsInput.questions[0].options == expected_options
    assert paused_report.needsInput is not None
    assert paused_report.needsInput.options == expected_options
    assert paused_report.finalSelection is not None
    assert paused_report.finalSelection.needsInput is not None
    assert paused_report.finalSelection.needsInput.options == expected_options
    assert paused_report.totalCandidates == 3

    completed, completed_report = orchestrator.parameter_tuning_stage(
        store,
        workflow,
        request_record,
        [],
        plan,
        preprocessed,
        experimental,
        enrichment_reference,
        experimental_reference,
        {"finalGraphOptionId": "native:RNA:baseline"},
    )

    assert completed.status == "done"
    assert completed_report.status == "done"
    assert completed_report.finalSelection is not None
    assert completed_report.finalSelection.selectedOptionId == "native:RNA:baseline"
    assert completed_report.totalCandidates == 3
    assert calls == {"screen": 1, "promote": 2, "integrate": 1, "selection": 1}


def test_integration_evaluations_contribute_to_final_candidate_count() -> None:
    report = _native_batch_report("RNA", "ADT")
    integration = _eligible_integration()

    finalized = finalize_parameter_tuning_selection(
        report,
        marker_assay="RNA",
        integration_evaluations=[integration],
        native_assay="RNA",
    )

    assert report.totalCandidates == 2
    assert finalized.totalCandidates == 3


def test_long_assay_names_produce_bounded_unique_candidate_ids() -> None:
    orchestrator = AgentOrchestrator(object())
    prefix = "Very long assay name with punctuation / and spaces " + "x" * 120
    first = PreprocessedAssayHandoff(
        assay=f"{prefix} one",
        assayType="RNA",
        reductionMethod="pca",
        normalized=ArtifactReferenceModel.get_example(),
        nCells=100,
        nFeatures=50,
    )
    second = first.model_copy(update={"assay": f"{prefix} two"})

    first_ids = {
        candidate.candidateId
        for candidate in orchestrator.initial_parameter_candidates(
            "long-name-test",
            first,
            count=5,
            neighbors_k=11,
        )
    }
    second_ids = {
        candidate.candidateId
        for candidate in orchestrator.initial_parameter_candidates(
            "long-name-test",
            second,
            count=5,
            neighbors_k=11,
        )
    }

    assert first_ids.isdisjoint(second_ids)
    assert all(len(candidate_id) <= 64 for candidate_id in first_ids | second_ids)
    assert all(
        all(
            character.isdigit() or character.islower() or character in {"_", "-"}
            for character in candidate_id
        )
        for candidate_id in first_ids | second_ids
    )
    assert all(
        len(f"{candidate_id}_harmony") <= 64 for candidate_id in first_ids | second_ids
    )


def test_single_integration_resolution_is_centered_and_workflow_unique() -> None:
    report = _native_batch_report("RNA", "ADT")
    primary_report = report.assayReports["RNA"]
    primary_evaluation = primary_report.evaluations[0]
    centered_evaluation = primary_evaluation.model_copy(
        update={
            "parameters": primary_evaluation.parameters.model_copy(
                update={"leidenResolution": 1.25}
            )
        }
    )
    primary_report = primary_report.model_copy(
        update={"evaluations": [centered_evaluation]}
    )
    report = report.model_copy(
        update={
            "evaluations": [centered_evaluation],
            "assayReports": {
                **report.assayReports,
                "RNA": primary_report,
            },
        }
    )

    class IntegrationStore:
        def __init__(self) -> None:
            self.integration_calls: list[dict[str, Any]] = []
            self.cluster_calls: list[dict[str, Any]] = []

        def load_artifact(self, reference: ArtifactRef) -> dict[str, np.ndarray]:
            if reference.kind == "integrated_graph":
                return {"modality_weights": np.full((4, 2), 0.5)}
            return {"values": np.asarray([0, 0, 1, 1])}

        def integrate_assays(
            self,
            sources: list[ArtifactRef],
            **kwargs: Any,
        ) -> ArtifactRef:
            self.integration_calls.append({"sources": sources, **kwargs})
            token = len(self.integration_calls)
            return ArtifactRef(
                scope="datastore",
                kind="integrated_graph",
                artifact_id=f"{100 + token:064x}",
            )

        def run_leiden_clustering(
            self,
            graph: ArtifactRef,
            **kwargs: Any,
        ) -> ArtifactRef:
            self.cluster_calls.append({"graph": graph, **kwargs})
            token = len(self.cluster_calls)
            return ArtifactRef(
                scope="datastore",
                kind="cluster_labels",
                artifact_id=f"{200 + token:064x}",
            )

    store = IntegrationStore()
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        pairedAssays=["RNA", "ADT"],
    )
    evaluations = AgentOrchestrator(object()).evaluate_integrations(
        store,
        "integration-center",
        plan,
        report,
        ExperimentalTuningHandoff(
            cellSelection=_cell_selection_model(),
            batchAction="skip",
        ),
        AutomatedWorkflowConfig(
            integrationResolutionCandidates=1,
            minClusterCells=1,
        ),
    )

    assert len(evaluations) == 2
    assert {value.method for value in evaluations} == {"snn", "wnn"}
    assert {value.resolution for value in evaluations} == {1.25}
    assert len(store.integration_calls) == 2
    assert all(call["invalidate_cache"] is True for call in store.integration_calls)
    assert {call["method"] for call in store.integration_calls} == {"snn", "wnn"}
    for call in store.integration_calls:
        expected_kind = "connectivity_map" if call["method"] == "snn" else "neighbors"
        assert {source.kind for source in call["sources"]} == {expected_kind}
        assert "label" not in call
    assert len(store.cluster_calls) == 2
    assert {call["resolution"] for call in store.cluster_calls} == {1.25}


def test_integration_requires_trusted_label_connectivity() -> None:
    class IntegrationStore:
        def load_artifact(self, reference: ArtifactRef) -> dict[str, np.ndarray]:
            if reference.kind == "integrated_graph":
                return {"modality_weights": np.full((4, 2), 0.5)}
            return {"values": np.asarray([0, 0, 1, 1])}

        def integrate_assays(
            self,
            _sources: list[ArtifactRef],
            **_kwargs: Any,
        ) -> ArtifactRef:
            return ArtifactRef(
                scope="datastore",
                kind="integrated_graph",
                artifact_id="7" * 64,
            )

        def run_leiden_clustering(
            self,
            _graph: ArtifactRef,
            **_kwargs: Any,
        ) -> ArtifactRef:
            return ArtifactRef(
                scope="datastore",
                kind="cluster_labels",
                artifact_id="8" * 64,
            )

        def metric_graph_connectivity(self, *_args: Any, **_kwargs: Any) -> float:
            raise ValueError("trusted label is unavailable")

    evaluations = AgentOrchestrator(object()).evaluate_integrations(
        IntegrationStore(),
        "integration-connectivity",
        AutomatedPreprocessingPlan(
            primaryAssay="RNA",
            markerAssay="RNA",
            pairedAssays=["RNA", "ADT"],
        ),
        _native_batch_report("RNA", "ADT"),
        ExperimentalTuningHandoff(
            cellSelection=_cell_selection_model(),
            batchAction="skip",
            preservationColumns=["trusted_cell_type"],
        ),
        AutomatedWorkflowConfig(
            integrationResolutionCandidates=1,
            minClusterCells=1,
        ),
    )

    assert evaluations
    assert all(not value.eligible for value in evaluations)
    assert all(
        any(
            "trusted-label connectivity" in reason
            for reason in value.eligibilityReasons
        )
        for value in evaluations
    )


def test_integration_checkpoints_prevent_retry_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = create_store(tmp_path / "integration-checkpoint.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    workflow = create_agent_workflow(store, workflow_run_id="integration-checkpoint")
    request_record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic integration retry test.",
            allowAssumptions=True,
        ),
        config=AutomatedWorkflowConfig(),
    )
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw,
        prefix,
        workflow.workflowRunId,
        "parameter_tuning",
        request_record,
        [],
        inputs={"semanticInput": "stable"},
    )
    integration_calls: list[dict[str, Any]] = []
    cluster_calls: list[dict[str, Any]] = []

    def load_artifact(reference: ArtifactRef) -> dict[str, np.ndarray]:
        if reference.kind == "integrated_graph":
            return {"modality_weights": np.full((4, 2), 0.5)}
        return {"values": np.asarray([0, 0, 1, 1])}

    def integrate_assays(
        sources: list[ArtifactRef],
        **kwargs: Any,
    ) -> ArtifactRef:
        integration_calls.append({"sources": sources, **kwargs})
        return ArtifactRef(
            scope="datastore",
            kind="integrated_graph",
            artifact_id=f"{100 + len(integration_calls):064x}",
        )

    def cluster(graph: ArtifactRef, **kwargs: Any) -> ArtifactRef:
        cluster_calls.append({"graph": graph, **kwargs})
        return ArtifactRef(
            scope="datastore",
            kind="cluster_labels",
            artifact_id=f"{200 + len(cluster_calls):064x}",
        )

    monkeypatch.setattr(store, "load_artifact", load_artifact)
    monkeypatch.setattr(store, "integrate_assays", integrate_assays)
    monkeypatch.setattr(store, "run_leiden_clustering", cluster)
    plan = AutomatedPreprocessingPlan(
        primaryAssay="RNA",
        markerAssay="RNA",
        pairedAssays=["RNA", "ADT"],
    )
    report = _native_batch_report("RNA", "ADT")
    config = AutomatedWorkflowConfig(
        integrationResolutionCandidates=1,
        minClusterCells=1,
    )
    first_actions: list[str] = []
    orchestrator = AgentOrchestrator(object())

    first = orchestrator.evaluate_integrations(
        store,
        workflow.workflowRunId,
        plan,
        report,
        ExperimentalTuningHandoff(
            cellSelection=_cell_selection_model(),
            batchAction="skip",
        ),
        config,
        started=started,
        actions=first_actions,
    )
    retried = started.model_copy(
        update={"attemptId": "retry-attempt", "startedAtNs": started.startedAtNs + 1}
    )
    retry_actions: list[str] = []
    second = orchestrator.evaluate_integrations(
        store,
        workflow.workflowRunId,
        plan,
        report,
        ExperimentalTuningHandoff(
            cellSelection=_cell_selection_model(),
            batchAction="skip",
        ),
        config,
        started=retried,
        actions=retry_actions,
    )

    assert second == first
    assert len(integration_calls) == 2
    assert len(cluster_calls) == 2
    assert first_actions == [
        "checkpoint_integration:snn",
        "checkpoint_integration:wnn",
    ]
    assert retry_actions == [
        "recover_integration_checkpoint:snn",
        "recover_integration_checkpoint:wnn",
    ]
    checkpoint_reports = list_agent_reports(
        store,
        workflow.workflowRunId,
        agent_name="parameter_tuning",
    )
    assert len(checkpoint_reports) == 2
    assert all("_integration_" in value.agentRunId for value in checkpoint_reports)


def test_preprocess_atac_and_adt_feature_routes() -> None:
    class Features:
        def __init__(self, ids: list[str], names: list[str]) -> None:
            self.N = len(ids)
            self.values = {"ids": np.asarray(ids), "names": np.asarray(names)}

        def fetch_all(self, column: str) -> np.ndarray:
            return self.values[column]

    class Store:
        def __init__(self) -> None:
            self.assays = {
                "ATAC": SimpleNamespace(
                    feats=Features(
                        ["chr1:1-10", "chr1:20-30", "chr2:1-10", "chr2:20-30"],
                        ["peak1", "peak2", "peak3", "peak4"],
                    )
                ),
                "ADT": SimpleNamespace(
                    feats=Features(
                        ["adt1", "adt2", "adt3", "adt4"],
                        ["CD3", "control", "CD19", "CD45"],
                    )
                ),
            }
            self.masks: list[np.ndarray] = []

        def get_assay(self, assay: str) -> Any:
            return self.assays[assay]

        @staticmethod
        def select_prevalent_peaks(*_args: Any, **_kwargs: Any) -> ArtifactRef:
            return ArtifactRef(
                scope="assay",
                assay="ATAC",
                kind="feature_selection",
                artifact_id="1" * 64,
            )

        def set_feature_selection(
            self, *, from_assay: str, mask: np.ndarray, **_kwargs: Any
        ) -> ArtifactRef:
            self.masks.append(mask.copy())
            return ArtifactRef(
                scope="assay",
                assay=from_assay,
                kind="feature_selection",
                artifact_id=("2" if from_assay == "ADT" else "3") * 64,
            )

        @staticmethod
        def run_normalization(
            _selection: ArtifactRef, *, features: ArtifactRef, **_kwargs: Any
        ) -> ArtifactRef:
            return ArtifactRef(
                scope="assay",
                assay=features.assay,
                kind="normalized",
                artifact_id=("4" if features.assay == "ATAC" else "5") * 64,
            )

        def load_artifact(self, ref: ArtifactRef) -> dict[str, np.ndarray]:
            selected = 3 if ref.assay == "ATAC" else int(self.masks[-1].sum())
            return {"values": np.asarray([True] * selected + [False] * (4 - selected))}

    store = Store()
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="c" * 64,
    )
    selection_model = ArtifactReferenceModel.from_artifact_ref(selection)
    orchestrator = AgentOrchestrator(object())
    actions: list[str] = []
    operations: list[dict[str, Any]] = []
    artifacts: dict[str, ArtifactReferenceModel] = {}
    atac = AssayPreprocessingPlan(
        assay="ATAC",
        assayType="ATAC",
        role="graph",
        graphEligible=True,
        markerEligible=True,
        featureMethod="prevalentPeaks",
        reductionMethod="lsi",
        featureParameters={"topN": 10, "minCells": 1},
        normalizationParameters={"logTransform": False, "renormalizeSubset": False},
    )
    atac_handoff = orchestrator.preprocess_assay(
        store,
        atac,
        cell_selection=selection,
        cell_selection_model=selection_model,
        active_cells=12,
        actions=actions,
        operations=operations,
        artifacts=artifacts,
    )
    assert atac_handoff.nFeatures == 3
    assert operations[0]["topN"] == 3

    adt = AssayPreprocessingPlan(
        assay="ADT",
        assayType="ADT",
        role="graph",
        graphEligible=True,
        markerEligible=True,
        featureMethod="panel",
        reductionMethod="identity",
        exactExcludedFeatures=["control", "adt4"],
        normalizationParameters={"logTransform": True, "renormalizeSubset": True},
    )
    adt_handoff = orchestrator.preprocess_assay(
        store,
        adt,
        cell_selection=selection,
        cell_selection_model=selection_model,
        active_cells=12,
        actions=actions,
        operations=operations,
        artifacts=artifacts,
    )
    assert adt_handoff.nFeatures == 2
    np.testing.assert_array_equal(store.masks[-1], [True, False, True, False])

    with pytest.raises(ValueError, match="fewer than two non-control"):
        orchestrator.preprocess_assay(
            store,
            adt.model_copy(update={"exactExcludedFeatures": ["adt2", "adt3", "adt4"]}),
            cell_selection=selection,
            cell_selection_model=selection_model,
            active_cells=12,
            actions=[],
            operations=[],
            artifacts={},
        )
    with pytest.raises(ValueError, match="Unsupported feature route"):
        orchestrator.preprocess_assay(
            store,
            adt.model_copy(update={"featureMethod": "none"}),
            cell_selection=selection,
            cell_selection_model=selection_model,
            active_cells=12,
            actions=[],
            operations=[],
            artifacts={},
        )


def test_exact_feature_exclusion_covers_all_supported_families() -> None:
    class Features:
        N = 4

        @staticmethod
        def fetch_all(column: str) -> np.ndarray:
            values = {
                "ids": np.asarray(["MT-CO1", "RPS3", "HIST1H1", "gene4"]),
                "names": np.asarray(["mito", "ribo", "histone", "GENE4"]),
            }
            return values[column]

    class Store:
        def __init__(self) -> None:
            self.mask: np.ndarray | None = None

        @staticmethod
        def get_assay(_assay: str) -> Any:
            return SimpleNamespace(feats=Features())

        @staticmethod
        def load_artifact(_source: ArtifactRef) -> dict[str, np.ndarray]:
            return {"values": np.ones(4, dtype=bool)}

        def set_feature_selection(
            self, *, from_assay: str, mask: np.ndarray, **_kwargs: Any
        ) -> ArtifactRef:
            self.mask = mask.copy()
            return ArtifactRef(
                scope="assay",
                assay=from_assay,
                kind="feature_selection",
                artifact_id="6" * 64,
            )

    plan = AssayPreprocessingPlan(
        assay="RNA",
        assayType="RNA",
        role="graph",
        graphEligible=True,
        markerEligible=True,
        featureMethod="hvg",
        reductionMethod="pca",
        featureParameters={
            "excludeFamilies": ["mitochondrial", "ribosomal", "histone"]
        },
    )
    orchestrator = AgentOrchestrator(object())
    blacklist = orchestrator.rna_blacklist(plan)
    assert all(token in blacklist for token in ("MT-", "RPS", "HIST"))
    source = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        artifact_id="7" * 64,
    )
    store = Store()
    result = orchestrator.exclude_exact_features(store, plan, source)
    assert result.kind == "feature_selection"
    assert store.mask is not None
    np.testing.assert_array_equal(store.mask, [False, False, False, True])
    with pytest.raises(ValueError, match="removed every feature"):
        orchestrator.exclude_exact_features(
            store,
            plan.model_copy(update={"exactExcludedFeatures": ["gene4"]}),
            source,
        )


def test_cell_qc_artifact_and_execution_validation_edges() -> None:
    metric = NamedArtifactSource(
        name="metric",
        artifact=ArtifactReferenceModel(
            scope="assay",
            assay="RNA",
            kind="quality_metric",
            artifactId="8" * 64,
        ),
    )
    sample = NamedArtifactSource(
        name="sample",
        artifact=ArtifactReferenceModel(
            scope="assay",
            assay="HTO",
            kind="hto_identity",
            artifactId="9" * 64,
        ),
    )
    global_plan = CellQcPlan(
        action="globalGaussian",
        profileId="global",
        driverAssay="RNA",
        driverAssayType="RNA",
        artifactMetrics=[metric],
        evidenceIds=["qcProfile:global"],
    )
    orchestrator = AgentOrchestrator(object())
    with pytest.raises(ValueError, match="metric names must be unique"):
        orchestrator._cell_qc_stage_artifacts(
            global_plan.model_copy(update={"artifactMetrics": [metric, metric]})
        )
    with pytest.raises(ValueError, match="collides with a metric"):
        orchestrator._cell_qc_stage_artifacts(
            global_plan.model_copy(
                update={
                    "sampleArtifact": sample.model_copy(update={"name": metric.name})
                }
            )
        )
    artifacts = orchestrator._cell_qc_stage_artifacts(
        global_plan.model_copy(update={"sampleArtifact": sample})
    )
    assert set(artifacts) == {"cellQcMetric:metric", "cellQcSample:sample"}

    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="a" * 64,
    )
    skip_profile = CellQcProfileEvidence(
        profileId="skip",
        action="skip",
        activeCells=4,
        retainedCells=4,
        retainedFraction=1.0,
        evidenceId="qcProfile:skip",
    )
    skip_plan = CellQcPlan(
        action="skip",
        profileId="skip",
        evidenceIds=[skip_profile.evidenceId],
    )

    def report(
        plan: CellQcPlan,
        profile: CellQcProfileEvidence,
        *,
        quality: list[NamedArtifactSource] | None = None,
        identities: list[NamedArtifactSource] | None = None,
    ) -> ExperimentalContextResult:
        return ExperimentalContextResult.get_blank().model_copy(
            update={
                "cellQc": plan,
                "qcProfiles": [profile],
                "qualityMetricArtifacts": list(quality or []),
                "htoIdentityArtifacts": list(identities or []),
            }
        )

    with pytest.raises(ValueError, match="unknown QC profile"):
        orchestrator.apply_cell_qc(
            object(),
            report(skip_plan, skip_profile).model_copy(update={"qcProfiles": []}),
            selection,
            [],
            [],
        )
    with pytest.raises(ValueError, match="does not match"):
        orchestrator.apply_cell_qc(
            object(),
            report(skip_plan.model_copy(update={"driverAssay": "RNA"}), skip_profile),
            selection,
            [],
            [],
        )

    global_profile = CellQcProfileEvidence(
        profileId="global",
        action="globalGaussian",
        driverAssay="RNA",
        driverAssayType="RNA",
        artifactMetrics=[metric],
        activeCells=4,
        retainedCells=3,
        retainedFraction=0.75,
        evidenceId="qcProfile:global",
    )
    with pytest.raises(ValueError, match="absent from Experimental Context"):
        orchestrator.apply_cell_qc(
            object(),
            report(global_plan, global_profile),
            selection,
            [],
            [],
        )

    sample_profile = CellQcProfileEvidence(
        profileId="sample",
        action="sampleMad",
        driverAssay="RNA",
        driverAssayType="RNA",
        sampleArtifact=sample,
        attributes=["RNA_nCounts"],
        activeCells=4,
        retainedCells=3,
        retainedFraction=0.75,
        evidenceId="qcProfile:sample",
    )
    sample_plan = CellQcPlan(
        action="sampleMad",
        profileId="sample",
        driverAssay="RNA",
        driverAssayType="RNA",
        sampleArtifact=sample,
        attributes=["RNA_nCounts"],
        evidenceIds=[sample_profile.evidenceId],
    )
    with pytest.raises(ValueError, match="sample artifact is absent"):
        orchestrator.apply_cell_qc(
            object(),
            report(sample_plan, sample_profile),
            selection,
            [],
            [],
        )

    actions: list[str] = []
    operations: list[dict[str, Any]] = []
    assert (
        orchestrator.apply_cell_qc(
            object(),
            report(skip_plan, skip_profile),
            selection,
            actions,
            operations,
        )
        == selection
    )
    assert actions == ["skip_cell_qc"]

    invalid_global_plan = global_plan.model_copy(update={"sampleColumn": "sample"})
    invalid_global_profile = global_profile.model_copy(
        update={"sampleColumn": "sample"}
    )
    with pytest.raises(ValueError, match="cannot include a sample"):
        orchestrator.apply_cell_qc(
            object(),
            report(invalid_global_plan, invalid_global_profile, quality=[metric]),
            selection,
            [],
            [],
        )
    invalid_sample_plan = sample_plan.model_copy(update={"sampleArtifact": None})
    invalid_sample_profile = sample_profile.model_copy(update={"sampleArtifact": None})
    with pytest.raises(ValueError, match="requires exactly one"):
        orchestrator.apply_cell_qc(
            object(),
            report(invalid_sample_plan, invalid_sample_profile),
            selection,
            [],
            [],
        )
    unsupported_plan = skip_plan.model_copy(update={"action": "unsupported"})
    unsupported_profile = skip_profile.model_copy(update={"action": "unsupported"})
    with pytest.raises(ValueError, match="Unsupported cell QC action"):
        orchestrator.apply_cell_qc(
            object(),
            report(unsupported_plan, unsupported_profile),
            selection,
            [],
            [],
        )


def test_preprocessing_plan_rejects_invalid_assay_routing() -> None:
    orchestrator = AgentOrchestrator(object())
    unsupported = _planning_inputs(
        {"custom": ("CRISPR", ["guide"], ["guide"])},
    )
    with pytest.raises(ValueError, match="No supported graph-bearing"):
        orchestrator.build_preprocessing_plan(*unsupported)

    too_many = list(
        _planning_inputs(
            {
                "RNA": ("RNA", ["g1", "g2", "g3"], ["G1", "G2", "G3"]),
                "ADT": ("ADT", ["a1", "a2", "a3"], ["A1", "A2", "A3"]),
            },
            config=AutomatedWorkflowConfig(maxGraphAssays=1),
        )
    )
    with pytest.raises(ValueError, match="Too many graph-bearing"):
        orchestrator.build_preprocessing_plan(*too_many)

    duplicate = list(
        _planning_inputs(
            {
                "RNA1": ("RNA", ["g1", "g2", "g3"], ["G1", "G2", "G3"]),
                "RNA2": ("RNA", ["g4", "g5", "g6"], ["G4", "G5", "G6"]),
            }
        )
    )
    request_record = duplicate[1]
    duplicate[1] = request_record.model_copy(
        update={
            "request": request_record.request.model_copy(update={"analysisAssays": []})
        }
    )
    with pytest.raises(ValueError, match="same-kind biological assays"):
        orchestrator.build_preprocessing_plan(*duplicate)

    base = list(
        _planning_inputs({"RNA": ("RNA", ["g1", "g2", "g3"], ["G1", "G2", "G3"])})
    )
    request_record = base[1]
    for updates, message in (
        ({"primaryAssay": "missing"}, "primaryAssay"),
        ({"markerAssay": "missing"}, "markerAssay"),
        ({"pairedAssays": ["RNA", "missing"]}, "non-graph assays"),
    ):
        values = list(base)
        values[1] = request_record.model_copy(
            update={"request": request_record.request.model_copy(update=updates)}
        )
        with pytest.raises(ValueError, match=message):
            orchestrator.build_preprocessing_plan(*values)

    paired = list(
        _planning_inputs(
            {
                "RNA": ("RNA", ["g1", "g2", "g3"], ["G1", "G2", "G3"]),
                "ADT": ("ADT", ["a1", "a2", "a3"], ["A1", "A2", "A3"]),
            },
            primary_assay="RNA",
        )
    )
    request_record = paired[1]
    paired[1] = request_record.model_copy(
        update={
            "request": request_record.request.model_copy(
                update={"pairedAssays": ["ADT"]}
            )
        }
    )
    with pytest.raises(ValueError, match="must include the primary"):
        orchestrator.build_preprocessing_plan(*paired)
