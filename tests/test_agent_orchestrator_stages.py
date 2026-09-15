"""Context, preprocessing, tuning, integration, and finalization contracts."""

from tests.agent_examples import example

import uuid
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scarf.agent.orchestrator.context as context_module
import scarf.agent.orchestrator.journal as journal_module
from scarf.agent.parameter_tuning.selection import harmony_acceptance_gate
from scarf.agent.orchestrator.preprocessing import PreprocessingStagesMixin
from scarf.agent.config import AgentRunConfig
from scarf.agent.data_enrichment import (
    AssayFeatureInspection,
    DataEnrichmentReport,
    FeatureFamilyEvidence,
    FeatureReference,
    FeatureSelectionPolicy,
)
from scarf.agent.experimental_context import (
    CellQcPlan,
    CellQcProfileEvidence,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    NamedArtifactSource,
)
from scarf.agent.experimental_context.tools import (
    persist_context_evidence,
    restore_context_evidence,
)
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
)
from scarf.agent.orchestrator.models import (
    AutomatedPreprocessingPlan,
    WorkflowStageAttempt,
)
from scarf.agent.orchestrator.models import OrchestrationRequestRecord, WorkflowIdentity
from scarf.agent.parameter_tuning import (
    ArtifactRecord,
    ParameterCandidateEvaluation,
)
from scarf.agent.cell_quality.profiles import RegisteredCellQcProfile
from scarf.agent.types import (
    AgentRunInfo,
    ArtifactReferenceModel,
    BatchSafetyEvidence,
)
from scarf.agent.parameter_tuning.diagnostics import (
    _select_capture_cells,
    resolve_native_doublet_inputs,
)
from scarf.datastore.datastore import DataStore
from scarf.storage.refs import ArtifactRef
from tests.agent_orchestrator_store import create_store


_PLAN_CHECKSUM = "a" * 64


def _save_input_evidence(store, workflow, request, enrichment):
    prefix = journal_module._ensure_orchestration_store(store)
    started = journal_module._start_attempt(
        store.zw, prefix, workflow.workflowRunId, "data_enrichment", request, []
    )
    return journal_module._save_stage_report(
        store, started, enrichment, expected_type=DataEnrichmentReport
    )[1]


def _measured_context_characterization(store, cell_selection):
    from scarf.agent.experimental_context.characterization import (
        characterize_covariates,
    )
    from scarf.agent.orchestrator.models import artifact_model_to_ref

    for name, values in {
        "sample": ["s1", "s1", "s2", "s2"],
        "donor": ["d1", "d1", "d2", "d2"],
        "treatment": ["control", "control", "treated", "treated"],
        "batch": ["b1", "b1", "b2", "b2"],
    }.items():
        store.cells.insert(name, np.asarray(values))
    return characterize_covariates(
        store,
        cellSelection=artifact_model_to_ref(cell_selection),
        model=None,
        directions={
            "columnDomains": {
                "sample": "design",
                "donor": "design",
                "treatment": "biological",
                "batch": "technical",
            },
            "coefficientsOfInterest": ["treatment"],
            "unitsOfInference": {
                "treatment": {
                    "observationUnit": "sample",
                    "independentUnit": "donor",
                }
            },
        },
    )


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
    CellQcPlan,
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
        studyObjective="Discover stable RNA populations.",
        primaryAssay=primary_assay,
        markerAssay=marker_assay,
        analysisAssays=analysis_assays or [],
    )
    request_record = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test-model",
        workflowRunId="planning-test",
        request=request,
        config=config or AutomatedWorkflowConfig(),
    )
    experimental = example(ExperimentalContextResult)
    ingest_outcome = WorkflowStageAttempt(
        workflowRunId="planning-test",
        stage="ingest",
        attemptId="ingest-test",
        status="done",
        startedAtNs=1,
        completedAtNs=2,
        outputs={"format": "zarr"},
    )
    return (
        store,
        request_record,
        enrichment,
        experimental,
        ingest_outcome,
        example(CellQcPlan),
    )


def _build_plan(
    assays: Mapping[str, tuple[str, list[str], list[str]]],
    **kwargs: Any,
) -> AutomatedPreprocessingPlan:
    inputs = _planning_inputs(assays, **kwargs)
    return AgentOrchestrator(object()).build_preprocessing_plan(*inputs)


@pytest.mark.parametrize("missing_replication", [False, True])
def test_unsafe_experimental_context_pauses_and_explicit_skip_reuses_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_replication: bool,
) -> None:
    path = create_store(tmp_path / "unsafe-context.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = WorkflowIdentity("unsafe-context")
    cell_selection = ArtifactReferenceModel.from_artifact_ref(
        store.snapshot_cell_selection("I")
    )
    enrichment = example(DataEnrichmentReport).model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="data_enrichment",
                runId=uuid.uuid4().hex,
            )
        }
    )
    request_record = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test-model",
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="Treatment is confounded with batch.",
            studyObjective="Preserve treatment while discovering populations.",
        ),
    )
    enrichment_reference = _save_input_evidence(
        store, workflow, request_record, enrichment
    )
    sample_context = example(ExperimentalContextResult)
    evidence_id = "batchEstimability:treatment:batch"
    unsafe_plan = sample_context.decision.batchCorrection.model_copy(
        update={
            "action": "unsafe",
            "evidenceIds": [evidence_id],
        }
    )
    unsafe_report = sample_context.model_copy(
        update={
            "characterization": _measured_context_characterization(
                store, cell_selection
            ),
            "cellSelection": cell_selection,
            "qualityMetricArtifacts": [],
            "htoIdentityArtifacts": [],
            "decision": sample_context.decision.model_copy(
                update={"batchCorrection": unsafe_plan}
            ),
            "batchSafety": [
                example(BatchSafetyEvidence).model_copy(
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
    if missing_replication:
        unsafe_report.characterization.coefficients[0]["replication"] = {}

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

    if missing_replication:
        assert done_outcome.status == "needsInput"
        assert done_outcome.needsInput is not None
        assert (
            "replication evidence is unavailable"
            in done_outcome.needsInput.questions[0].question
        )
    else:
        assert done_outcome.status == "done"
    assert done_outcome.actions == ["resolve_unsafe_batch_correction:skip"]
    assert resolved_report.decision.batchCorrection.action == "skip"
    assert resolved_report.decision.batchCorrection.batchColumns == []
    assert resolved_report.decision.batchCorrection.preserveColumns == (
        unsafe_plan.preserveColumns
    )
    assert UnsafeAgent.calls == 1


@pytest.mark.parametrize("after_commit", ["none", "interrupt", "exception"])
@pytest.mark.parametrize("report_status", ["failed", "needsInput"])
def test_failed_context_retries_without_overwriting_or_replaying_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_commit: str,
    report_status: str,
) -> None:
    path = create_store(tmp_path / "retry-context.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = WorkflowIdentity("retry-context")
    selection = ArtifactReferenceModel.from_artifact_ref(
        store.snapshot_cell_selection("I")
    )
    request = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test-model",
        workflowRunId=workflow.workflowRunId,
        config=AutomatedWorkflowConfig(inputPolicy="unattended"),
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="Treatment is confounded with batch.",
            studyObjective="Preserve treatment while discovering populations.",
        ),
    )
    enrichment_ref = _save_input_evidence(
        store, workflow, request, example(DataEnrichmentReport)
    )
    report = example(ExperimentalContextResult).model_copy(
        update={
            "characterization": _measured_context_characterization(store, selection),
            "cellSelection": selection,
            "cellQc": CellQcPlan(),
            "qcProfiles": [],
            "qualityMetricArtifacts": [],
            "htoIdentityColumns": [],
            "htoIdentityArtifacts": [],
        }
    )
    report.decision.batchCorrection.action = "unsafe"
    unresolved_question = "Assess the joint tissue and treatment contrast by donor."
    failed_report = report.model_copy(
        update={"status": report_status, "notes": ["Invalid design proposal"]},
        deep=True,
    )
    failed_report.decision.batchCorrection.action = "needsInput"
    failed_report.decision.needsInput = [unresolved_question]
    evidence_readers = []

    class RecoveringAgent:
        calls = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self, *_args: Any, **kwargs: Any) -> ExperimentalContextResult:
            type(self).calls += 1
            read = kwargs["checkpoint_read"]
            evidence_readers.append(read)
            # Every new interpretation has its own result, while the complete
            # measured design and its consumed allowance remain shared.
            assert read("result") is None
            deps = ExperimentalContextDependencies(
                checkpointRead=read,
                checkpointWrite=kwargs["checkpoint_write"],
            )
            if self.calls == 1:
                assert not restore_context_evidence(deps)
                deps.characterization = report.characterization
                deps.toolCalls = ["inspect_cell_covariates"]
                persist_context_evidence(deps, "inspection")
                deps.designRounds = 1
                deps.toolCalls.append("analyze_experimental_design")
                deps.evidenceIds.add("saved-design-evidence")
                persist_context_evidence(deps, "design1")
            else:
                assert restore_context_evidence(deps)
                assert deps.characterization == report.characterization
                assert deps.designRounds == 1
                assert deps.evidenceIds == {"saved-design-evidence"}
                assert deps.toolCalls.count("analyze_experimental_design") == 1
            current = failed_report if self.calls <= 2 else report
            kwargs["checkpoint_write"](
                "result", {"report": current.model_dump(mode="json")}
            )
            assert read("result")["report"]["status"] == current.status
            return current

    monkeypatch.setattr(context_module, "ExperimentalContextAgent", RecoveringAgent)
    orchestrator = AgentOrchestrator(object())

    def execute():
        return orchestrator.experimental_context_stage(
            store, workflow, request, [], selection, enrichment_ref, [], [], {}
        )

    first, _ = execute()
    second, _ = execute()
    assert first.status == second.status == "failed"
    if report_status == "needsInput":
        assert unresolved_question in first.error
        assert unresolved_question in second.error
    assert RecoveringAgent.calls == 2
    assert first.reportReferences != second.reportReferences
    assert second.inputs["retryAfterFailedReport"] == first.reportReferences[
        0
    ].model_dump(mode="json")
    original_failure = journal_module.read_stage_evidence(
        store, first.reportReferences[0]
    )

    if after_commit != "none":
        save_outcome = journal_module._save_outcome

        def interrupt(_group, _prefix, outcome):
            if outcome.status == "done":
                if after_commit == "exception":
                    raise RuntimeError(
                        "persistence failed after committing the decision"
                    )
                raise KeyboardInterrupt("interrupted after committing the decision")
            save_outcome(_group, _prefix, outcome)

        monkeypatch.setattr(journal_module, "_save_outcome", interrupt)
        if after_commit == "interrupt":
            with pytest.raises(KeyboardInterrupt, match="committing"):
                execute()
        else:
            exception_outcome, _ = execute()
            assert exception_outcome.status == "failed"
            assert exception_outcome.reportReferences == []
        monkeypatch.setattr(journal_module, "_save_outcome", save_outcome)
    else:
        done, _ = execute()
        assert done.status == "done"

    resumed, resumed_report = execute()
    assert resumed.status == "done"
    assert resumed_report.status == "done"
    assert resumed_report.decision.batchCorrection.action == "unsafe"
    assert RecoveringAgent.calls == 3
    assert [read("result")["report"]["status"] for read in evidence_readers] == [
        report_status,
        report_status,
        "done",
    ]
    for key in ("inspection", "design1"):
        assert all(
            read(key) == evidence_readers[0](key) for read in evidence_readers[1:]
        )
    if after_commit != "none":
        assert "recover_persisted_experimental_context_report" in resumed.actions
    assert (
        journal_module.read_stage_evidence(store, first.reportReferences[0])
        == original_failure
    )
    assert (
        journal_module.read_stage_evidence(store, second.reportReferences[0])["status"]
        == report_status
    )


def test_explicit_no_inference_skip_resolves_context_without_provider_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = create_store(tmp_path / "no-inference-context.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    workflow = WorkflowIdentity("no-inference-context")
    cell_selection = ArtifactReferenceModel.from_artifact_ref(
        store.snapshot_cell_selection("I")
    )
    enrichment = example(DataEnrichmentReport).model_copy(
        update={
            "runInfo": AgentRunInfo(
                agentName="data_enrichment",
                runId=uuid.uuid4().hex,
            )
        }
    )
    request_record = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test-model",
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A study with unresolved replication.",
            studyObjective="Discover stable RNA populations.",
        ),
    )
    enrichment_reference = _save_input_evidence(
        store, workflow, request_record, enrichment
    )
    sample_context = example(ExperimentalContextResult)
    needs_input_plan = sample_context.decision.batchCorrection.model_copy(
        update={
            "action": "needsInput",
            "batchColumns": [],
            "preserveColumns": [],
            "metricsRequired": [],
        }
    )
    needs_input_report = sample_context.model_copy(
        update={
            "characterization": _measured_context_characterization(
                store, cell_selection
            ),
            "status": "needsInput",
            "cellSelection": cell_selection,
            "cellQc": CellQcPlan(),
            "qcProfiles": [],
            "qualityMetricArtifacts": [],
            "htoIdentityColumns": [],
            "htoIdentityArtifacts": [],
            "decision": sample_context.decision.model_copy(
                update={
                    "batchCorrection": needs_input_plan,
                    "cellQc": CellQcPlan(),
                    "needsInput": [
                        "Provide replicated observation units or skip inference."
                    ],
                }
            ),
            "runInfo": AgentRunInfo(
                agentName="experimental_context",
                runId=uuid.uuid4().hex,
            ),
        }
    )

    class NeedsInputAgent:
        calls = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.config = AgentRunConfig()

        def run(self, *_args: Any, **_kwargs: Any) -> ExperimentalContextResult:
            type(self).calls += 1
            return needs_input_report

    monkeypatch.setattr(context_module, "ExperimentalContextAgent", NeedsInputAgent)
    orchestrator = AgentOrchestrator(object())
    paused_outcome, _ = orchestrator.experimental_context_stage(
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

    assert paused_outcome.status == "needsInput"
    resolved_outcome, resolved_report = orchestrator.experimental_context_stage(
        store,
        workflow,
        request_record,
        [],
        cell_selection,
        enrichment_reference,
        [],
        [],
        {
            "experimentalDirections": {
                "coefficientsOfInterest": [],
                "unitsOfInference": {},
                "batchCorrection": {"action": "skip"},
            }
        },
    )

    assert resolved_outcome.status == "done"
    assert resolved_outcome.artifacts == paused_outcome.artifacts
    assert resolved_outcome.actions == [
        "resolve_experimental_context:no_inference_skip_harmony"
    ]
    assert resolved_report.status == "done"
    assert resolved_report.decision.coefficientsOfInterest == []
    assert resolved_report.decision.unitsOfInference == {}
    assert resolved_report.decision.batchCorrection.action == "skip"
    assert resolved_report.decision.batchCorrection.batchColumns == []
    assert resolved_report.decision.needsInput == []
    assert resolved_report.runInfo.agentName == "experimental_context_resolution"
    assert resolved_report.runInfo.runId == ""
    assert resolved_report.runInfo.usage.requests == 0
    assert NeedsInputAgent.calls == 1
    assert journal_module.read_stage_evidence(
        store, resolved_outcome.reportReferences[0]
    ) == resolved_report.model_dump(mode="json")
    assert paused_outcome.reportReferences[0] != resolved_outcome.reportReferences[0]


def test_qc_profile_safety_rejects_self_normalizing_failed_captures() -> None:
    def profile(
        registered_profile: RegisteredCellQcProfile,
        *,
        failed: list[str],
        references: list[str] | None = None,
    ) -> CellQcProfileEvidence:
        return CellQcProfileEvidence(
            profileId=f"cellQc:RNA:{registered_profile}",
            action="registeredMad",
            registeredProfile=registered_profile,
            driverAssay="RNA",
            driverAssayType="RNA",
            sampleColumn="capture",
            attributes=["RNA_nCounts"],
            parameters={"pooledReferenceCaptures": references or []},
            activeCells=100,
            retainedCells=90,
            retainedFraction=0.9,
            failedCaptureCandidates=failed,
            evidenceId=f"qcProfile:{registered_profile}",
        )

    assert not PreprocessingStagesMixin._profile_is_safe(
        profile("captureMad5", failed=["capture-b"])
    )
    assert PreprocessingStagesMixin._profile_is_safe(
        profile(
            "pooledReferenceMad5",
            failed=["capture-b"],
            references=["capture-a"],
        )
    )
    assert not PreprocessingStagesMixin._profile_is_safe(
        profile(
            "pooledReferenceMad5",
            failed=["capture-a"],
            references=["capture-a"],
        )
    )


def test_preprocessing_plan_selects_rna_and_ignores_other_modalities() -> None:
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
    assert set(routes) == {"transcriptome"}
    assert plan.pairedAssays == []


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


def test_percent_features_use_exact_symbols_even_when_enrichment_omits_a_family(
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
    workflow = WorkflowIdentity("inspected-families")
    request_record = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test-model",
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic feature-family test.",
            studyObjective="Discover stable RNA populations.",
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
    outcome = orchestrator._rna_quality_metrics_stage(
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
    metric_sources = orchestrator._named_stage_artifacts(
        outcome,
        "qualityMetricArtifacts",
        "quality_metric",
    )
    assert metric_sources[0] == metric_source
    assert [item.name for item in metric_sources] == [
        "RNA_percentMito",
        "RNA_percentRibo",
    ]
    assert outcome.outputs["percentageDefinitions"] == [
        {"family": "mitochondrial", "pattern": r"(?i)^MT-", "matchedGenes": 1},
        {
            "family": "ribosomal",
            "pattern": r"(?i)^(RPS|RPL|MRPS|MRPL)",
            "matchedGenes": 1,
        },
    ]
    operation = outcome.outputs["operations"][0]
    assert operation["operation"] == "run_feature_percentage"
    assert operation["cellSelection"] == cell_selection.model_dump(mode="json")
    assert operation["features"]["kind"] == "feature_selection"
    assert operation["artifact"] == metric_source.artifact.model_dump(mode="json")

    from scarf.agent.cell_quality.profiles import project_auto_filter_profile
    from scarf.metadata.selection import resolve_cell_aligned_artifact

    mito_values = np.asarray(
        resolve_cell_aligned_artifact(
            store.zw,
            ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="quality_metric",
                artifact_id=metric_source.artifact.artifactId,
            ),
            cell_selection=ArtifactRef(
                scope="datastore",
                kind="cell_selection",
                artifact_id=cell_selection.artifactId,
            ),
            expected_kind="quality_metric",
        ).values
    )
    projection = project_auto_filter_profile(
        "globalGaussian",
        values_by_metric={
            "RNA_nCounts": store.cells.fetch("RNA_nCounts"),
            metric_source.name: mito_values,
        },
        active=np.ones(len(mito_values), dtype=bool),
    )
    profile = CellQcProfileEvidence(
        profileId="cellQc:RNA:RNA:globalGaussian:0.01:0.99",
        action="globalGaussian",
        driverAssay="RNA",
        driverAssayType="RNA",
        attributes=["RNA_nCounts"],
        artifactMetrics=[metric_source],
        parameters=projection.parameters,
        resolvedBounds=projection.parameters["resolvedBounds"],
        flaggedCells=projection.flagCounts,
        activeCells=len(mito_values),
        retainedCells=projection.retainedCells,
        retainedFraction=projection.retainedCells / len(mito_values),
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


def test_hto_processing_is_not_executed_by_rna_workflow(
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
    workflow = WorkflowIdentity("hto-once")
    request_record = OrchestrationRequestRecord(
        inputIdentity={},
        modelIdentity="test-model",
        workflowRunId=workflow.workflowRunId,
        request=AutomatedWorkflowRequest(
            sourcePath=str(path),
            zarrPath=str(path),
            studyContext="A deterministic HTO checkpoint test.",
            studyObjective="Discover stable RNA populations.",
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

    with pytest.raises(ValueError, match="only the selected RNA assay"):
        orchestrator._rna_quality_metrics_stage(
            store, workflow, request_record, [], enrichment, cell_selection
        )
    enrichment = DataEnrichmentReport(
        status="done", policies=[_modality_policy("RNA", "RNA")]
    )
    first = orchestrator._rna_quality_metrics_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment,
        cell_selection,
    )
    second = orchestrator._rna_quality_metrics_stage(
        store,
        workflow,
        request_record,
        [],
        enrichment,
        cell_selection,
    )

    assert first == second
    assert first.status == "done"
    assert calls == 0
    assert first.outputs["htoIdentityArtifacts"] == []
    assert all(ref.kind != "hto_identity" for ref in first.artifacts.values())


def test_selected_sample_mad_qc_passes_exact_artifact_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.orchestrator import preprocessing as preprocessing_module

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
        parameters={"nMads": 3.0, "minCellsPerSample": 20},
        resolvedBounds={"capture-a": {"RNA_nCounts": {"low": 2.0, "high": 200.0}}},
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

    def execute(_store: Any, action: str, **kwargs: Any):
        assert action == "sampleMad"
        captured.update(kwargs)
        return output_selection, None

    monkeypatch.setattr(preprocessing_module, "execute_auto_cell_qc", execute)

    actions: list[str] = []
    operations: list[dict[str, Any]] = []
    result = AgentOrchestrator(object()).apply_cell_qc(
        object(),
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
    assert captured["profile_parameters"] == profile.parameters
    assert captured["expected_resolved_bounds"] == profile.resolvedBounds
    assert captured["expected_active_cells"] == 100
    assert captured["expected_retained_cells"] == 95
    assert operations[0]["sampleArtifact"] == sample.model_dump(mode="json")
    assert operations[0]["artifactMetrics"] == [metric.model_dump(mode="json")]


def test_capture_cell_selections_are_exact_and_idempotent(tmp_path: Path) -> None:
    path = create_store(tmp_path / "capture-selections.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        zarr_mode="r+",
    )
    capture_values = np.asarray(["a", "a", "b", "b"])
    store.cells.insert("capture", capture_values, overwrite=True)
    parent = store.snapshot_cell_selection("I")
    active_indices = np.arange(store.cells.N, dtype=np.int64)

    first_a, count_a = _select_capture_cells(
        store,
        parent,
        column="capture",
        value="a",
        active_indices=active_indices,
        active_values=capture_values,
    )
    second_a, repeated_count = _select_capture_cells(
        store,
        parent,
        column="capture",
        value="a",
        active_indices=active_indices,
        active_values=capture_values,
    )
    selected_a = np.asarray(store.load_artifact(first_a)["values"][:], dtype=bool)

    assert first_a == second_a
    assert count_a == repeated_count == 2
    assert selected_a.tolist() == [True, True, False, False]


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
    with pytest.raises(ValueError, match="cannot use a core sample source"):
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


def test_harmony_doublet_graph_matches_selected_native_parameters() -> None:
    class Store:
        def __init__(self) -> None:
            self.neighbors_k: int | None = None
            self.resolution: float | None = None

        def load_artifact(self, reference: ArtifactRef) -> dict[str, Any]:
            assert reference.kind == "reduction"
            return {}

        def build_ann_index(
            self, coordinates: ArtifactRef, **_kwargs: Any
        ) -> ArtifactRef:
            assert coordinates.kind == "reduction"
            return ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="ann_index",
                artifact_id="2" * 64,
            )

        def query_neighbors(
            self,
            _ann: ArtifactRef,
            *,
            k: int,
            **_kwargs: Any,
        ) -> ArtifactRef:
            self.neighbors_k = k
            return ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="neighbors",
                artifact_id="3" * 64,
            )

        def build_connectivity_map(
            self,
            _neighbors: ArtifactRef,
            **_kwargs: Any,
        ) -> ArtifactRef:
            return ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="connectivity_map",
                artifact_id="4" * 64,
            )

        def run_leiden_clustering(
            self,
            _graph: ArtifactRef,
            *,
            resolution: float,
            **_kwargs: Any,
        ) -> ArtifactRef:
            self.resolution = resolution
            return ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="cluster_labels",
                artifact_id="5" * 64,
            )

    base = example(ParameterCandidateEvaluation)
    native = base.model_copy(
        update={
            "candidateId": "native",
            "parameters": base.parameters.model_copy(
                update={
                    "candidateId": "native",
                    "dimensions": 20,
                    "neighborsK": 11,
                    "leidenResolution": 0.5,
                    "useHarmony": False,
                }
            ),
        }
    )
    selected = base.model_copy(
        update={
            "candidateId": "harmony",
            "parameters": base.parameters.model_copy(
                update={
                    "candidateId": "harmony",
                    "dimensions": 20,
                    "neighborsK": 41,
                    "leidenResolution": 1.5,
                    "useHarmony": True,
                }
            ),
            "artifacts": {
                **base.artifacts,
                "pca": ArtifactRecord(
                    assay="RNA",
                    kind="reduction",
                    artifactId="1" * 64,
                ),
            },
        }
    )
    store = Store()
    clusters, graph = resolve_native_doublet_inputs(
        store,
        selected,
        [native, selected],
    )

    assert store.neighbors_k == 41
    assert store.resolution == 1.5
    assert graph.artifact_id == "4" * 64
    assert clusters.artifact_id == "5" * 64


@pytest.mark.parametrize(
    ("batch_mixing", "marker_coherence", "expected"),
    [
        (0.56, 0.80, True),
        (0.54, 0.80, False),
        (0.56, 0.74, False),
        (None, 0.80, False),
    ],
)
def test_harmony_acceptance_requires_improvement_without_biological_loss(
    batch_mixing: float | None,
    marker_coherence: float,
    expected: bool,
) -> None:
    base = example(ParameterCandidateEvaluation)
    native_parameters = base.parameters.model_copy(
        update={"candidateId": "native", "useHarmony": False}
    )
    harmony_parameters = base.parameters.model_copy(
        update={"candidateId": "harmony", "useHarmony": True}
    )
    native = base.model_copy(
        update={
            "candidateId": "native",
            "parameters": native_parameters,
            "metrics": base.metrics.model_copy(
                update={
                    "batchMixing": {"batch": 0.50},
                    "biologicalPreservation": {
                        "condition": {
                            "clisi": 0.80,
                            "graphConnectivity": 0.80,
                        }
                    },
                    "crossUnitSupport": 0.80,
                    "markerCoherence": 0.80,
                }
            ),
        }
    )
    harmony = base.model_copy(
        update={
            "candidateId": "harmony",
            "parameters": harmony_parameters,
            "metrics": base.metrics.model_copy(
                update={
                    "batchMixing": (
                        {} if batch_mixing is None else {"batch": batch_mixing}
                    ),
                    "biologicalPreservation": {
                        "condition": {
                            "clisi": 0.80,
                            "graphConnectivity": 0.80,
                        }
                    },
                    "crossUnitSupport": 0.80,
                    "markerCoherence": marker_coherence,
                }
            ),
        }
    )

    accepted, reasons = harmony_acceptance_gate(
        native,
        harmony,
        batch_columns=["batch"],
        protected_columns=["condition"],
        independent_unit_columns=["donor"],
    )

    assert accepted is expected
    assert bool(reasons) is not expected


@pytest.mark.parametrize(
    "assays,updates,message",
    [
        ({"custom": ("CRISPR", ["g"], ["G"])}, {}, "requires one RNA"),
        (
            {
                "RNA1": ("RNA", ["a", "b", "c"], ["A", "B", "C"]),
                "RNA2": ("RNA", ["d", "e", "f"], ["D", "E", "F"]),
            },
            {},
            "found 2",
        ),
        (
            {"RNA": ("RNA", ["a", "b", "c"], ["A", "B", "C"])},
            {"primaryAssay": "missing"},
            "Unknown requested RNA",
        ),
        (
            {"RNA": ("RNA", ["a", "b", "c"], ["A", "B", "C"])},
            {"markerAssay": "missing"},
            "markerAssay",
        ),
        (
            {"RNA": ("RNA", ["a", "b", "c"], ["A", "B", "C"])},
            {"pairedAssays": ["RNA", "other"]},
            "pairedAssays",
        ),
    ],
)
def test_preprocessing_plan_rejects_invalid_assay_routing(
    assays: dict[str, Any],
    updates: dict[str, Any],
    message: str,
) -> None:
    inputs = list(_planning_inputs(assays))
    record = inputs[1]
    inputs[1] = record.model_copy(
        update={"request": record.request.model_copy(update=updates)}
    )
    with pytest.raises(ValueError, match=message):
        AgentOrchestrator(object()).build_preprocessing_plan(*inputs)
