"""Tests for the tool-driven Experimental Context Agent."""

from tests.agent_examples import example

import asyncio
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
import zarr
from pydantic import ValidationError
from pydantic_ai import ModelRetry, RunContext, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from zarr.storage import MemoryStore

import scarf.agent.experimental_context.agent as experimental_context_agent
import scarf.agent.experimental_context.contracts as experimental_context_contracts
import scarf.agent.experimental_context.qc_evidence as experimental_context_qc
import scarf.agent.experimental_context.tools as experimental_context_tools
import scarf.agent.experimental_context.validation as experimental_context_validation
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    BatchSafetyEvidence,
    CaptureFailureEvidence,
    CellQcPlan,
    CellQcProfileEvidence,
    ContrastPlan,
    CovariateEvidence,
    ExperimentalContextAgent,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    InferenceUnit,
    NamedArtifactSource,
    RepresentationEvaluation,
    analyze_experimental_design,
    inspect_cell_covariates,
    score_current_representation,
    validate_experimental_context,
)
from scarf.agent.experimental_context.characterization import (
    CovariateCharacterization,
    _SelectionBoundCells,
)
from scarf.agent.experimental_context.study import StudyContract, build_study_contract
from scarf.agent.types import (
    ArtifactReferenceModel,
    ExperimentalBiologyHandoff,
    ExperimentalTuningHandoff,
)
from scarf.datastore.pipeline_run import PipelineRun
from scarf.metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.quality_control.filtering import gaussian_quantile_bounds
from scarf.storage.pipeline_runs import PipelineOutputRecord, PipelineRunRecord
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import resolve_selection_artifact

type TestAction = Literal["skip", "evaluateHarmony", "unsafe", "needsInput"]


class _Root(dict[str, Any]):
    def __init__(self, assay_types: dict[str, str]) -> None:
        super().__init__()
        self.attrs = {"assayTypes": assay_types}


class _Cells:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self._values = values
        self.N = len(next(iter(values.values())))

    @property
    def columns(self) -> list[str]:
        return list(self._values)

    def fetch_all(self, column: str) -> np.ndarray:
        return self._values[column]

    def _get_array(self, column: str) -> np.ndarray:
        return self._values[column]

    @staticmethod
    def default_block_rows(_column: str = "I") -> int:
        return 100

    def fetch(self, column: str, key: str = "I") -> np.ndarray:
        return np.asarray(
            self._values[column][np.asarray(self._values[key], dtype=bool)]
        )

    def iter_row_blocks(
        self,
        *,
        cell_key: str = "I",
        columns: list[str] | None = None,
        block_rows: int | None = None,
    ) -> Any:
        del block_rows
        selected = np.asarray(self._values[cell_key], dtype=bool)
        requested = self.columns if columns is None else columns
        yield SimpleNamespace(
            values={name: self._values[name][selected] for name in requested}
        )


class _Store:
    assay_names = ["RNA"]

    def __init__(self) -> None:
        n_cells = 12
        self.assay_state_lookups: list[str | None] = []
        self.artifact_inputs: dict[ArtifactRef, dict[str, Any]] = {}
        self.inspected_artifacts: list[ArtifactRef] = []
        self.cells = _Cells(
            {
                "I": np.ones(n_cells, dtype=bool),
                "ids": np.array([f"cell-{index}" for index in range(n_cells)]),
                "names": np.array([f"cell-{index}" for index in range(n_cells)]),
                "donor": np.array(["d1"] * 6 + ["d2"] * 6),
                "sample": np.array(["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3 + ["s4"] * 3),
                "batch": np.array(["b1"] * 6 + ["b2"] * 6),
                "disease": np.array(["case"] * 6 + ["control"] * 6),
                "cell_type": np.array(
                    ["alpha", "beta", "alpha", "beta", "alpha", "beta"] * 2
                ),
                "sequencing_depth": np.array(
                    [1000.0] * 3 + [2000.0] * 3 + [3500.0] * 3 + [4800.0] * 3
                ),
            }
        )
        self.zw = zarr.open_group(store=MemoryStore(), mode="w")
        cell_data = self.zw.create_group("cellData")
        row_ids = np.asarray(self.cells._values["ids"], dtype="U16")
        cell_data.create_array("ids", data=row_ids)
        cell_data.create_array("I", data=self.cells._values["I"])
        self.cell_selection = resolve_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=self.cells._values["I"],
            row_ids=row_ids,
            operation="test_experimental_context_selection",
            parameters={},
            inputs={},
            source_column="I",
        )
        self.zw.attrs["_test_cell_selection"] = self.cell_selection.to_dict()

    def refresh_cell_selection(self) -> None:
        row_ids = np.asarray(self.cells._values["ids"], dtype="U16")
        self.cell_selection = resolve_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=self.cells._values["I"],
            row_ids=row_ids,
            operation="test_experimental_context_selection",
            parameters={},
            inputs={},
            source_column="I",
        )
        self.zw.attrs["_test_cell_selection"] = self.cell_selection.to_dict()

    def get_assay_state(self, from_assay: str | None = None) -> None:
        self.assay_state_lookups.append(from_assay)
        return None

    @staticmethod
    def list_artifacts(from_assay: str | None = None) -> list[Any]:
        del from_assay
        return []

    def inspect_artifact(self, ref: ArtifactRef) -> SimpleNamespace:
        self.inspected_artifacts.append(ref)
        return SimpleNamespace(
            exists=True,
            complete=True,
            inputs=self.artifact_inputs[ref],
        )


def _replace_store_cells(store: _Store, values: dict[str, np.ndarray]) -> None:
    store.cells = _Cells(values)
    store.zw = zarr.open_group(store=MemoryStore(), mode="w")
    cell_data = store.zw.create_group("cellData")
    cell_data.create_array("ids", data=np.asarray(values["ids"]).astype("U16"))
    cell_data.create_array("I", data=np.asarray(values["I"], dtype=bool))
    store.refresh_cell_selection()


def _write_cell_artifact(
    store: _Store,
    *,
    name: str,
    kind: Literal["quality_metric", "hto_identity"],
    values: np.ndarray,
    assay: str,
    operation: str | None = None,
    inputs: dict[str, Any] | None = None,
) -> NamedArtifactSource:
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay=assay,
        kind=kind,
        operation=operation or f"test_{kind}_source",
        parameters={"name": name},
        inputs=dict(inputs or {}),
        execution_options={},
        cell_selection=store.cell_selection,
        arrays={"values": ((len(values),), None)},
    )
    write_cell_data_artifact(
        store.zw,
        planned,
        {"values": values},
    )
    return NamedArtifactSource(
        name=name,
        artifact=ArtifactReferenceModel.from_artifact_ref(planned.ref),
    )


class _MetricStore(_Store):
    def __init__(self) -> None:
        super().__init__()
        self.metric_calls: list[tuple[str, str, ArtifactRef]] = []

    def metric_ilisi(self, column: str, neighbors: ArtifactRef) -> float:
        self.metric_calls.append(("metric_ilisi", column, neighbors))
        return 0.7

    def metric_proportional_batch_mixing(
        self,
        column: str,
        neighbors: ArtifactRef,
    ) -> float:
        self.metric_calls.append(
            ("metric_proportional_batch_mixing", column, neighbors)
        )
        return 0.8

    def metric_clisi(self, column: str, neighbors: ArtifactRef) -> float:
        self.metric_calls.append(("metric_clisi", column, neighbors))
        return 0.9

    def metric_graph_connectivity(self, column: str, graph: ArtifactRef) -> float:
        self.metric_calls.append(("metric_graph_connectivity", column, graph))
        return 0.95


@pytest.fixture(autouse=True)
def _resolve_fake_graph_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from scarf.agent.experimental_context import agent as module

    def resolve(root: zarr.Group, _graph: ArtifactRef) -> ArtifactRef:
        return ArtifactRef.from_dict(root.attrs["_test_cell_selection"])

    monkeypatch.setattr(module, "graph_cell_selection", resolve)


def _graph_refs() -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    cell_selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="c" * 64,
    )
    neighbors = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="a" * 64,
    )
    connectivity_map = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="b" * 64,
    )
    return cell_selection, neighbors, connectivity_map


def _configure_graph_lineage(
    store: _Store,
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    _, neighbors, connectivity_map = _graph_refs()
    cell_selection = store.cell_selection
    store.artifact_inputs = {
        neighbors: {"cell_selection": cell_selection.to_dict()},
        connectivity_map: {"cell_selection": cell_selection.to_dict()},
    }
    return cell_selection, neighbors, connectivity_map


def _completed_graph_run(
    store: _Store,
    neighbors: ArtifactRef,
    connectivity_map: ArtifactRef,
) -> PipelineRun:
    record = PipelineRunRecord(
        run_id="d" * 64,
        recipe="basic_rna_analysis",
        requested_label="agent-input",
        label="agent-input",
        assay="RNA",
        started_at_ns=1,
        finished_at_ns=2,
        status="completed",
        complete=True,
        scarf_version="1.0.0",
        config={},
        stage_order=("graph",),
        outputs=(
            PipelineOutputRecord(
                "analysis_cell_selection",
                store.cell_selection,
            ),
            PipelineOutputRecord("neighbors", neighbors),
            PipelineOutputRecord("connectivity_map", connectivity_map),
        ),
        fields=(),
        error=None,
        interruption=None,
    )
    return PipelineRun(store, record)


def _context(
    store: _Store,
    *,
    directions: dict[str, object] | None = None,
    neighbors: ArtifactRef | None = None,
    connectivity_map: ArtifactRef | None = None,
    cell_selection: ArtifactRef | None = None,
    quality_metric_artifacts: list[NamedArtifactSource] | None = None,
    hto_identity_artifacts: list[NamedArtifactSource] | None = None,
) -> RunContext[ExperimentalContextDependencies]:
    selection = cell_selection or store.cell_selection
    hto_sources = list(hto_identity_artifacts or [])
    return RunContext(
        deps=ExperimentalContextDependencies(
            store=store,
            neighbors=neighbors,
            connectivityMap=connectivity_map,
            cellSelection=selection,
            cells=_SelectionBoundCells(
                store.zw,
                store.cells,
                selection,
                artifacts={
                    source.name: ArtifactRef(
                        scope=source.artifact.scope,
                        assay=source.artifact.assay,
                        kind=source.artifact.kind,
                        artifact_id=source.artifact.artifactId,
                    )
                    for source in hto_sources
                },
            ),
            studyContext="Case-control study with samples nested in donors.",
            directions=dict(directions or {}),
            qualityMetricArtifacts=list(quality_metric_artifacts or []),
            htoIdentityArtifacts=hto_sources,
        ),
        model=TestModel(),
        usage=RunUsage(),
    )


def _design_decision(action: TestAction = "unsafe") -> ExperimentalContextDecision:
    return ExperimentalContextDecision(
        columnDomains={
            "donor": "design",
            "sample": "design",
            "batch": "technical",
            "disease": "biological",
            "cell_type": "biological",
            "sequencing_depth": "technical",
        },
        coefficientsOfInterest=["disease"],
        unitsOfInference={
            "disease": InferenceUnit(
                observationUnit="sample",
                independentUnit="donor",
            )
        },
        batchCorrection=BatchCorrectionPlan(
            action=action,
            batchColumns=["batch"],
            preserveColumns=["disease"],
            metricsRequired=["iLISI", "cLISI"],
            rationale="Batch is perfectly aligned with disease.",
            evidenceIds=[
                "column:batch",
                "confounding:disease:batch",
                "estimability:disease",
                "batchEstimability:disease:batch",
            ],
        ),
        rationale="Disease is the primary between-sample coefficient.",
        evidenceIds=["column:disease", "column:sample", "column:donor"],
    )


def test_agent_models_have_blank_and_example_constructors() -> None:
    models = (
        InferenceUnit,
        BatchCorrectionPlan,
        BatchSafetyEvidence,
        NamedArtifactSource,
        CellQcPlan,
        CellQcProfileEvidence,
        CovariateEvidence,
        ExperimentalContextDecision,
        RepresentationEvaluation,
        ExperimentalContextResult,
        ExperimentalContextDependencies,
        ExperimentalTuningHandoff,
        ExperimentalBiologyHandoff,
    )
    for model in models:
        assert isinstance(model.get_blank(), model)
        assert isinstance(example(model), model)
        assert all("_" not in field_name for field_name in model.model_fields)
    assert set(RepresentationEvaluation.model_fields) == {
        "available",
        "assay",
        "cellSelection",
        "neighbors",
        "connectivityMap",
        "metrics",
        "notes",
        "evidenceIds",
    }


def test_system_prompt_does_not_embed_fictional_output_values() -> None:
    prompt = ExperimentalContextAgent(object()).system_prompt

    assert "Output contract example" not in prompt
    assert "column:batch" not in prompt
    assert "estimability:treatment" not in prompt


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"studyContext": ""}, "studyContext must be non-empty"),
        ({"studyObjective": ""}, "studyObjective must be non-empty"),
        (
            {"independentUnitColumns": ["donor", "donor"]},
            "must not contain duplicates",
        ),
        (
            {
                "physicalCaptureColumn": "sample",
                "conditionColumns": ["sample"],
            },
            "cannot be the physical capture",
        ),
        (
            {
                "technicalBatchColumns": ["batch"],
                "conditionColumns": ["batch"],
            },
            "cannot also be condition columns",
        ),
        (
            {
                "correctionLicense": "safe",
                "technicalBatchColumns": [],
            },
            "requires batch columns",
        ),
    ],
)
def test_study_contract_rejects_inconsistent_design_authority(
    changes: dict[str, object],
    message: str,
) -> None:
    values = StudyContract.get_blank().model_dump()
    values.update(changes)

    with pytest.raises(ValidationError, match=message):
        StudyContract.model_validate(values)


def test_study_contract_builder_records_correction_and_label_policies() -> None:
    with pytest.raises(ValueError, match="must be done"):
        build_study_contract(
            study_context="Study context",
            study_objective="Study objective",
            experimental_result=SimpleNamespace(status="needsInput"),
        )

    def result(action: TestAction) -> SimpleNamespace:
        return SimpleNamespace(
            status="done",
            decision=_design_decision(action=action),
            characterization=CovariateCharacterization(status="done"),
            batchSafety=[],
            notes=[],
        )

    unsafe = build_study_contract(
        study_context="Study context",
        study_objective="Study objective",
        experimental_result=result("unsafe"),
    )
    unresolved = build_study_contract(
        study_context="Study context",
        study_objective="Study objective",
        experimental_result=result("needsInput"),
    )
    preservation = build_study_contract(
        study_context="Study context",
        study_objective="Study objective",
        experimental_result=result("skip"),
        author_label_policy="preservation",
    )

    assert unsafe.correctionLicense == "unsafeConfounded"
    assert unresolved.correctionLicense == "indeterminate"
    assert preservation.authorLabelPolicy == "preservation"
    assert any("ineligible" in item for item in preservation.limitations)


def test_validator_rejects_serialized_fields_inside_narrative() -> None:
    decision = ExperimentalContextDecision(
        rationale='Study design is unresolved.", "evidenceIds": ["column:batch"]',
    )

    with pytest.raises(ModelRetry, match="plain prose"):
        validate_experimental_context(decision, _context(_Store()).deps)


def test_agent_runs_only_read_only_tools_and_returns_a_grounded_report() -> None:
    store = _Store()
    tool_names: set[str] = set()
    state = {"request": 0}

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        tool_names.update(tool.name for tool in info.function_tools)
        request = state["request"]
        state["request"] += 1
        if request == 0:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="inspect_cell_covariates", args={})]
            )
        if request == 1:
            decision = _design_decision()
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="analyze_experimental_design",
                        args={
                            "column_domains": decision.columnDomains,
                            "coefficients_of_interest": (
                                decision.coefficientsOfInterest
                            ),
                            "units_of_inference": {
                                name: unit.model_dump()
                                for name, unit in decision.unitsOfInference.items()
                            },
                            "batch_columns": decision.batchCorrection.batchColumns[0],
                        },
                    )
                ]
            )
        if request == 2:
            decision = _design_decision()
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="analyze_experimental_design",
                        args={
                            "column_domains": decision.columnDomains,
                            "coefficients_of_interest": (
                                decision.coefficientsOfInterest
                            ),
                            "units_of_inference": {
                                name: unit.model_dump()
                                for name, unit in decision.unitsOfInference.items()
                            },
                            "batch_columns": decision.batchCorrection.batchColumns,
                        },
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=_design_decision().model_dump(),
                )
            ]
        )

    result = ExperimentalContextAgent(FunctionModel(reply)).run(
        store,
        study_context="Case-control study with samples nested in donors.",
        cell_selection=store.cell_selection,
    )

    assert result.status == "done"
    assert result.decision.batchCorrection.action == "unsafe"
    assert result.batchSafety[0].status == "unsafe"
    tuning_handoff = result.to_parameter_tuning_handoff()
    assert tuning_handoff.batchAction == "unsafe"
    assert tuning_handoff.batchSafety[0].evidenceId in tuning_handoff.evidenceIds
    biology_handoff = result.to_biological_handoff()
    assert biology_handoff.conditionColumn == "disease"
    assert biology_handoff.observationUnit == "sample"
    assert result.runInfo.agentName == "experimental_context"
    assert result.cellQc == result.decision.cellQc
    assert result.cellQc.action == "skip"
    assert result.qcProfiles[0].activeCells == 12
    assert [call.toolName for call in result.runInfo.toolCalls] == [
        "inspect_cell_covariates",
        "analyze_experimental_design",
        "analyze_experimental_design",
    ]
    assert tool_names == {
        "inspect_cell_covariates",
        "inspect_context_evidence",
        "analyze_experimental_design",
        "score_current_representation",
    }
    assert not any(
        token in tool_name
        for tool_name in tool_names
        for token in ("write", "run_harmony", "python", "shell", "zarr")
    )
    assert store.assay_state_lookups == []
    assert sorted(store.zw.group_keys()) == ["artifacts", "cellData"]


def test_agent_fails_after_design_tool_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    analyze_retries: list[int | None] = []

    def unavailable_design(**kwargs: Any) -> None:
        deps = kwargs["deps"]
        analyze_tool = next(
            tool
            for tool in kwargs["tools"]
            if tool.name == "analyze_experimental_design"
        )
        analyze_retries.append(analyze_tool.max_retries)
        asyncio.run(
            inspect_cell_covariates(
                RunContext(
                    deps=deps,
                    model=TestModel(),
                    usage=RunUsage(),
                )
            )
        )
        raise UnexpectedModelBehavior(
            "Tool 'analyze_experimental_design' exceeded max retries count of 1"
        )

    monkeypatch.setattr(
        experimental_context_agent,
        "run_agent_sync",
        unavailable_design,
    )
    result = ExperimentalContextAgent(object()).run(
        store,
        study_context="Case-control study with samples nested in donors.",
        cell_selection=store.cell_selection,
    )

    assert analyze_retries == [3]
    assert result.status == "failed"
    assert result.decision.batchCorrection.action == "needsInput"
    assert result.decision.batchCorrection.batchColumns == []
    assert result.cellSelection is not None
    assert result.cellSelection.artifactId == store.cell_selection.artifact_id
    assert result.cellQc.profileId == ""
    assert result.qcProfiles
    assert result.runInfo.agentName == "experimental_context_failed"
    with pytest.raises(ValueError, match="must be done"):
        result.to_parameter_tuning_handoff()
    assert any("did not produce" in note for note in result.notes)


def test_agent_rejects_malformed_batch_tool_call_without_default_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()

    def unavailable_design(**kwargs: Any) -> None:
        deps = kwargs["deps"]
        asyncio.run(
            inspect_cell_covariates(
                RunContext(
                    deps=deps,
                    model=TestModel(),
                    usage=RunUsage(),
                )
            )
        )
        raise UnexpectedModelBehavior(
            "Tool 'analyze_experimental_design' exceeded max retries count of 3; "
            "caused by ModelRetry: Unknown batch column 'batchassay'"
        )

    monkeypatch.setattr(
        experimental_context_agent,
        "run_agent_sync",
        unavailable_design,
    )
    result = ExperimentalContextAgent(object()).run(
        store,
        study_context="Population discovery across sequencing batches.",
        study_objective="Discover stable cell populations.",
        cell_selection=store.cell_selection,
        directions={
            "columnDomains": {
                "batch": "technical",
                "donor": "design",
                "sample": "design",
            },
            "coefficientsOfInterest": [],
            "unitsOfInference": {},
            "batchColumns": ["batch"],
        },
    )

    assert result.status == "failed"
    assert result.decision.batchCorrection.batchColumns == []
    with pytest.raises(ValueError, match="must be done"):
        result.to_parameter_tuning_handoff()
    assert result.runInfo.agentName == "experimental_context_failed"
    assert any("batchassay" in note for note in result.notes)


def test_agent_preserves_validated_design_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()

    def unresolved_design(**kwargs: Any) -> SimpleNamespace:
        deps = kwargs["deps"]
        context = RunContext(deps=deps, model=TestModel(), usage=RunUsage())
        asyncio.run(inspect_cell_covariates(context))
        planned = _design_decision("needsInput")
        asyncio.run(
            analyze_experimental_design(
                context,
                column_domains=planned.columnDomains,
                coefficients_of_interest=planned.coefficientsOfInterest,
                units_of_inference=planned.unitsOfInference,
                batch_columns=planned.batchCorrection.batchColumns,
            )
        )
        planned.needsInput = ["Physical capture provenance remains unresolved."]
        decision = kwargs["output_validator"](planned)
        return SimpleNamespace(
            output=decision,
            runInfo=experimental_context_contracts.AgentRunInfo(
                agentName="experimental_context", modelName="test"
            ),
        )

    monkeypatch.setattr(experimental_context_agent, "run_agent_sync", unresolved_design)
    result = ExperimentalContextAgent(object()).run(
        store,
        study_context="Case-control study with samples nested in donors.",
        cell_selection=store.cell_selection,
    )
    assert result.status == "needsInput"
    assert result.decision.needsInput == [
        "Physical capture provenance remains unresolved."
    ]
    assert result.runInfo.agentName == "experimental_context"
    with pytest.raises(ValueError, match="must be done"):
        result.to_parameter_tuning_handoff()


def test_handoff_builders_reject_incomplete_or_ambiguous_results() -> None:
    incomplete = ExperimentalContextResult.get_blank()
    with pytest.raises(ValueError, match="must be done"):
        incomplete.to_parameter_tuning_handoff()

    ambiguous = example(ExperimentalContextResult)
    ambiguous.decision.coefficientsOfInterest.append("second_coefficient")
    with pytest.raises(ValueError, match="Select one coefficient explicitly"):
        ambiguous.to_biological_handoff()


def test_tools_build_a_grounded_design_report_without_mutation() -> None:
    store = _Store()
    context = _context(store)
    columns_before = set(store.cells.columns)
    values_before = {
        column: values.copy() for column, values in store.cells._values.items()
    }
    artifacts_before = store.list_artifacts(from_assay="RNA")

    inspected = asyncio.run(inspect_cell_covariates(context))
    analyzed = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=_design_decision().columnDomains,
            coefficients_of_interest=["disease"],
            units_of_inference={
                "disease": InferenceUnit(
                    observationUnit="sample",
                    independentUnit="donor",
                )
            },
            batch_columns=["batch"],
        )
    )

    assert inspected.characterization.status == "done"
    assert "column:batch" in analyzed.evidenceIds
    assert "confounding:disease:batch" in analyzed.evidenceIds
    assert analyzed.batchSafety[0].status == "unsafe"
    assert analyzed.batchSafety[0].batchColumns == ["batch"]
    assert (
        analyzed.characterization.confounding[0]["estimability"]["coefficientEstimable"]
        is False
    )
    assert set(store.cells.columns) == columns_before
    for column, values in values_before.items():
        np.testing.assert_array_equal(store.cells._values[column], values)
    assert store.list_artifacts(from_assay="RNA") == artifacts_before
    assert store.assay_state_lookups == []
    assert sorted(store.zw.group_keys()) == ["artifacts", "cellData"]


def test_exact_batch_direction_overrides_model_tool_arguments() -> None:
    store = _Store()
    context = _context(store, directions={"batchColumns": ["batch"]})
    decision = _design_decision()

    asyncio.run(inspect_cell_covariates(context))
    analyzed = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=[],
        )
    )

    assert analyzed.batchSafety[0].batchColumns == ["batch"]
    skip = decision.model_copy(
        update={
            "batchCorrection": BatchCorrectionPlan(
                action="skip",
                rationale="Skip the declared batch condition.",
                evidenceIds=["column:batch"],
            )
        }
    )
    with pytest.raises(ModelRetry, match="exact directed batch"):
        validate_experimental_context(skip, context.deps)


def test_qc_profiles_use_persisted_modality_and_shared_cell_selection() -> None:
    store = _Store()
    store.assay_names = ["protein", "peaks", "transcript"]
    store.zw.attrs["assayTypes"] = {
        "protein": "ADT",
        "peaks": "ATAC",
        "transcript": "RNA",
    }
    store.cells._values["I"][:2] = False
    store.refresh_cell_selection()
    store.cells._values["transcript_nCounts"] = np.asarray(
        [1, 2, 5, 9, 10, 11, 12, 13, 14, 15, 50, 100],
        dtype=float,
    )
    store.cells._values["transcript_nFeatures"] = np.asarray(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
        dtype=float,
    )
    store.cells._values["peaks_nCounts"] = np.arange(12, dtype=float) + 1
    context = _context(store)

    inspected = asyncio.run(inspect_cell_covariates(context))

    assert {profile.action for profile in inspected.qcProfiles} == {
        "skip",
        "globalGaussian",
        "registeredMad",
    }
    for profile in inspected.qcProfiles:
        assert profile.driverAssay == "transcript"
        assert profile.driverAssayType == "RNA"
        assert profile.activeCells == 10
        assert profile.retainedCells <= 10
        assert "cellKey" not in CellQcProfileEvidence.model_fields
    global_profile = next(
        profile
        for profile in inspected.qcProfiles
        if profile.action == "globalGaussian"
    )
    assert global_profile.attributes == [
        "transcript_nCounts",
        "transcript_nFeatures",
    ]
    active_counts = store.cells._values["transcript_nCounts"][2:]
    active_features = store.cells._values["transcript_nFeatures"][2:]
    expected_count_bounds = gaussian_quantile_bounds(active_counts, 0.01, 0.99)
    expected_feature_bounds = gaussian_quantile_bounds(active_features, 0.01, 0.99)
    assert global_profile.parameters["resolvedBounds"] == {
        "transcript_nCounts": {
            "low": expected_count_bounds[0],
            "high": expected_count_bounds[1],
        },
        "transcript_nFeatures": {
            "low": expected_feature_bounds[0],
            "high": expected_feature_bounds[1],
        },
    }
    full_count_bounds = gaussian_quantile_bounds(
        store.cells._values["transcript_nCounts"],
        0.01,
        0.99,
    )
    assert expected_count_bounds != pytest.approx(full_count_bounds)


def test_design_tool_offers_only_grounded_sample_mad_profiles() -> None:
    store = _Store()
    store.cells._values["RNA_nCounts"] = np.arange(12, dtype=float) + 1
    store.cells._values["RNA_nFeatures"] = np.arange(12, dtype=float) + 5
    context = _context(store)
    decision = _design_decision()

    asyncio.run(inspect_cell_covariates(context))
    analyzed = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=decision.batchCorrection.batchColumns,
        )
    )

    sample_profile = next(
        profile for profile in analyzed.qcProfiles if profile.action == "sampleMad"
    )
    assert sample_profile.sampleColumn == "sample"
    assert sample_profile.parameters == {
        "nMads": 3.0,
        "minCellsPerSample": 20,
        "nSamples": 4,
        "nSkippedSamples": 4,
    }
    assert sample_profile.retainedCells == 12
    assert sample_profile.evidenceId in analyzed.evidenceIds

    validated = validate_experimental_context(decision, context.deps)
    assert validated.cellQc == CellQcPlan.get_blank()
    assert sample_profile in context.deps.qcProfiles.values()


def test_caller_qc_direction_shapes_evidence_without_preselection() -> None:
    store = _Store()
    store.cells._values["RNA_nCounts"] = np.arange(12, dtype=float) + 1
    store.cells._values["RNA_nFeatures"] = np.arange(12, dtype=float) + 5
    context = _context(
        store,
        directions={"cellQc": {"action": "sampleMad", "sampleColumn": "sample"}},
    )
    decision = _design_decision()

    asyncio.run(inspect_cell_covariates(context))
    asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=decision.batchCorrection.batchColumns,
        )
    )
    validated = validate_experimental_context(decision, context.deps)

    assert validated.cellQc == CellQcPlan.get_blank()
    assert any(
        profile.action == "sampleMad" and profile.sampleColumn == "sample"
        for profile in context.deps.qcProfiles.values()
    )


def test_adt_and_hto_do_not_drive_qc_and_hto_identity_remains_metadata() -> None:
    store = _Store()
    store.assay_names = ["protein", "hashtags"]
    store.zw.attrs["assayTypes"] = {"protein": "ADT", "hashtags": "HTO"}
    store.cells._values["sample_id"] = np.asarray(
        ["sample-a"] * 5 + ["sample-b"] * 5 + ["Negative", "Doublet"]
    )
    context = _context(
        store,
        directions={"htoIdentityColumn": "sample_id"},
    )

    inspected = asyncio.run(inspect_cell_covariates(context))

    assert inspected.htoIdentityColumns == ["sample_id"]
    assert "qcProfile:cellQc:none:none:skip" in inspected.evidenceIds
    assert len(inspected.qcProfiles) == 1
    profile = inspected.qcProfiles[0]
    assert profile.action == "skip"
    assert profile.driverAssay is None
    assert profile.driverAssayType is None
    assert profile.retainedCells == 12


def test_artifact_metrics_and_hto_grouping_are_exact_context_evidence() -> None:
    store = _Store()
    store.assay_names = ["RNA", "HTO"]
    store.zw.attrs["assayTypes"] = {"RNA": "RNA", "HTO": "HTO"}
    store.cells._values["RNA_nCounts"] = np.arange(12, dtype=float) + 10
    store.cells._values["RNA_nFeatures"] = np.arange(12, dtype=float) + 5
    metric = _write_cell_artifact(
        store,
        name="RNA_percentMito",
        kind="quality_metric",
        values=np.asarray(
            [1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 1.4, 1.1, 0.8, 1.2, 1.0, 40.0]
        ),
        assay="RNA",
    )
    ribo_metric = _write_cell_artifact(
        store,
        name="RNA_percentRibo",
        kind="quality_metric",
        values=np.asarray(
            [5.0, 5.2, 4.9, 5.1, 5.3, 5.0, 5.4, 5.1, 4.8, 5.2, 5.0, 35.0]
        ),
        assay="RNA",
    )
    identity = _write_cell_artifact(
        store,
        name="HTO_htoIdentity",
        kind="hto_identity",
        values=np.asarray(["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3 + ["s4"] * 3),
        assay="HTO",
    )
    columns_before = set(store.cells.columns)
    context = _context(
        store,
        quality_metric_artifacts=[metric, ribo_metric],
        hto_identity_artifacts=[identity],
    )

    inspected = asyncio.run(inspect_cell_covariates(context))

    global_profile = next(
        profile
        for profile in inspected.qcProfiles
        if profile.action == "globalGaussian"
    )
    assert global_profile.attributes == ["RNA_nCounts", "RNA_nFeatures"]
    assert global_profile.artifactMetrics == [metric, ribo_metric]
    hto_profile = next(
        profile
        for profile in inspected.qcProfiles
        if profile.action == "sampleMad" and profile.sampleArtifact is not None
    )
    assert hto_profile.sampleColumn is None
    assert hto_profile.sampleArtifact == identity
    assert hto_profile.artifactMetrics == [metric, ribo_metric]
    hto_record = next(
        record
        for record in inspected.characterization.columns
        if record["name"] == identity.name
    )
    assert hto_record["domain"] == "design"
    assert hto_record["sourceType"] == "artifact"
    assert (
        hto_record["artifact"]
        == ArtifactRef(
            scope="assay",
            assay="HTO",
            kind="hto_identity",
            artifact_id=identity.artifact.artifactId,
        ).to_dict()
    )
    assert inspected.htoIdentityArtifacts == [identity]
    assert (
        f"htoIdentityArtifact:{identity.name}:{identity.artifact.artifactId}"
        in inspected.evidenceIds
    )
    assert set(store.cells.columns) == columns_before
    assert "RNA_percentMito" not in store.cells.columns
    assert "RNA_percentRibo" not in store.cells.columns
    assert "HTO_htoIdentity" not in store.cells.columns


def test_qc_metric_sources_report_metadata_artifact_concordance() -> None:
    store = _Store()
    counts = np.linspace(10.0, 100.0, store.cells.N)
    features = np.linspace(5.0, 50.0, store.cells.N)
    store.cells._values["RNA_nCounts"] = counts
    store.cells._values["RNA_nFeatures"] = features
    count_artifact = _write_cell_artifact(
        store,
        name="RNA_nCounts",
        kind="quality_metric",
        values=counts.copy(),
        assay="RNA",
    )
    feature_artifact = _write_cell_artifact(
        store,
        name="RNA_nFeatures",
        kind="quality_metric",
        values=features + 1e-9,
        assay="RNA",
    )
    context = _context(
        store,
        quality_metric_artifacts=[count_artifact, feature_artifact],
    )

    inspected = asyncio.run(inspect_cell_covariates(context))

    concordance = {item.metricRole: item for item in inspected.qcSourceConcordance}
    assert concordance["count"].exactlyEqual is True
    assert concordance["count"].numericallyClose is True
    assert concordance["count"].meanAbsoluteDifference == 0.0
    assert concordance["feature"].exactlyEqual is False
    assert concordance["feature"].numericallyClose is True
    assert concordance["feature"].pearsonCorrelation == pytest.approx(1.0)
    assert {item.evidenceId for item in inspected.qcSourceConcordance}.issubset(
        inspected.evidenceIds
    )


def test_capture_failure_evidence_preserves_paired_design_after_exclusion() -> None:
    store = _Store()
    captures = np.repeat(["capture-a", "capture-b", "failed"], 20)
    donors = np.repeat(["donor-a", "donor-b", "donor-c"], 20)
    disease = np.tile(np.repeat(["case", "control"], 10), 3)
    samples = np.asarray(
        [
            f"{donor}-{condition}"
            for donor, condition in zip(donors, disease, strict=True)
        ]
    )
    counts = np.concatenate(
        [
            np.linspace(90.0, 110.0, 20),
            np.linspace(95.0, 115.0, 20),
            np.linspace(3.0, 7.0, 20),
        ]
    )
    features = np.concatenate(
        [
            np.linspace(45.0, 55.0, 20),
            np.linspace(48.0, 58.0, 20),
            np.linspace(2.0, 6.0, 20),
        ]
    )
    mito = np.concatenate(
        [
            np.linspace(1.0, 3.0, 20),
            np.linspace(1.0, 4.0, 20),
            np.linspace(25.0, 35.0, 20),
        ]
    )
    n_cells = len(captures)
    _replace_store_cells(
        store,
        {
            "I": np.ones(n_cells, dtype=bool),
            "ids": np.asarray([f"cell-{index}" for index in range(n_cells)]),
            "names": np.asarray([f"cell-{index}" for index in range(n_cells)]),
            "capture": captures,
            "donor": donors,
            "sample": samples,
            "disease": disease,
            "RNA_nCounts": counts,
            "RNA_nFeatures": features,
            "RNA_percentMito": mito,
        },
    )
    context = _context(
        store,
        directions={"physicalCaptureColumn": "capture"},
    )
    domains = {
        "capture": "design",
        "donor": "design",
        "sample": "design",
        "disease": "biological",
    }
    units = {
        "disease": InferenceUnit(
            observationUnit="sample",
            independentUnit="donor",
        )
    }

    asyncio.run(inspect_cell_covariates(context))
    analyzed = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=domains,
            coefficients_of_interest=["disease"],
            units_of_inference=units,
            batch_columns=[],
        )
    )

    profile = next(
        item for item in analyzed.qcProfiles if item.registeredProfile == "captureMad5"
    )
    failure = next(
        item for item in profile.captureFailureEvidence if item.capture == "failed"
    )
    assert failure.wholeCaptureFailure is True
    assert failure.independentAdverseAxes >= 2
    assert failure.preservesConditionCoverage is True
    assert failure.preservesIndependentUnitCoverage is True
    assert failure.exclusionEligible is True
    assert failure.conditionAndUnitSafety[0]["completePairsAfterExclusion"] == 2
    assert failure.conditionAndUnitSafety[0]["incompletePairsAfterExclusion"] == 0
    assert "failed" in profile.failedCaptureCandidates
    assert "failed" in profile.excludableCaptureCandidates
    for source in profile.metricSources:
        assert source.missingCellsByCapture == {
            "capture-a": 0,
            "capture-b": 0,
            "failed": 0,
        }


def test_unusable_qc_sources_preserve_provenance_and_degradation_evidence() -> None:
    store = _Store()
    store.cells._values["RNA_nCounts"] = np.asarray(["bad"] * store.cells.N)
    store.cells._values["RNA_nFeatures"] = np.asarray(
        [*np.linspace(10.0, 20.0, store.cells.N - 1), np.nan]
    )
    artifact = _write_cell_artifact(
        store,
        name="RNA_percentMito",
        kind="quality_metric",
        values=np.asarray([*np.linspace(1.0, 2.0, store.cells.N - 1), np.inf]),
        assay="RNA",
        operation="run_feature_percentage",
        inputs={
            "lineage": [
                {
                    "scope": "invalid",
                    "kind": "invalid",
                    "artifact_id": "invalid",
                },
                store.cell_selection.to_dict(),
                store.cell_selection.to_dict(),
            ]
        },
    )
    context = _context(store, quality_metric_artifacts=[artifact])

    inspected = asyncio.run(inspect_cell_covariates(context))

    sources = {
        (source.sourceType, source.metricName): source
        for source in inspected.qcMetricSources
    }
    nonnumeric = sources[("metadataColumn", "RNA_nCounts")]
    assert nonnumeric.usableForFiltering is False
    assert nonnumeric.missingCells == store.cells.N
    assert nonnumeric.notes == ["Metric is not numeric and cannot drive filtering"]
    nonfinite = sources[("metadataColumn", "RNA_nFeatures")]
    assert nonfinite.usableForFiltering is False
    assert nonfinite.missingCells == 1
    derived = sources[("artifact", "RNA_percentMito")]
    assert derived.origin == "derivedArtifact"
    assert derived.usableForFiltering is False
    assert derived.missingCells == 1
    assert derived.inputArtifacts == [
        ArtifactReferenceModel.from_artifact_ref(store.cell_selection)
    ]
    skip = next(profile for profile in inspected.qcProfiles if profile.action == "skip")
    assert any("not numeric" in note for note in skip.notes)
    assert any("non-finite" in note for note in skip.notes)


def test_directed_capture_artifact_enables_pooled_reference_profile() -> None:
    store = _Store()
    labels = np.repeat(["capture-a", "capture-b", "capture-c"], 20)
    counts = np.concatenate(
        [
            np.linspace(90.0, 110.0, 20),
            np.linspace(95.0, 115.0, 20),
            np.linspace(100.0, 120.0, 20),
        ]
    )
    n_cells = len(labels)
    _replace_store_cells(
        store,
        {
            "I": np.ones(n_cells, dtype=bool),
            "ids": np.asarray([f"cell-{index}" for index in range(n_cells)]),
            "names": np.asarray([f"cell-{index}" for index in range(n_cells)]),
            "RNA_nCounts": counts,
        },
    )
    capture = _write_cell_artifact(
        store,
        name="capture",
        kind="hto_identity",
        values=labels,
        assay="HTO",
    )
    context = _context(
        store,
        directions={
            "physicalCaptureColumn": "capture",
            "cellQc": {
                "pooledReferenceCaptures": ["capture-a", "capture-b"],
            },
        },
        hto_identity_artifacts=[capture],
    )

    inspected = asyncio.run(inspect_cell_covariates(context))

    profile = next(
        item
        for item in inspected.qcProfiles
        if item.registeredProfile == "pooledReferenceMad5"
    )
    assert profile.captureColumn is None
    assert profile.captureArtifact == capture
    assert profile.sampleArtifact == capture
    assert profile.parameters["pooledReferenceCaptures"] == [
        "capture-a",
        "capture-b",
    ]
    assert profile.activeCellsByCapture == {
        "capture-a": 20,
        "capture-b": 20,
        "capture-c": 20,
    }


def test_registered_only_without_qc_driver_returns_explicit_skip() -> None:
    store = _Store()
    store.assay_names = []
    context = _context(store, directions={"registeredQcOnly": True})

    inspected = asyncio.run(inspect_cell_covariates(context))

    assert len(inspected.qcProfiles) == 1
    profile = inspected.qcProfiles[0]
    assert profile.action == "skip"
    assert profile.driverAssay is None
    assert profile.driverAssayType is None
    assert profile.notes == [
        "No RNA or ATAC assay is eligible to drive automatic cell QC"
    ]


def test_missing_rna_percentage_metrics_are_derived_through_public_artifacts() -> None:
    store = _Store()
    feature_values = {
        "ids": np.asarray(["MT-CO1", "RPS3", "GAPDH"]),
        "names": np.asarray(["MT-CO1", "RPS3", "GAPDH"]),
    }
    assay = SimpleNamespace(
        feats=SimpleNamespace(
            N=3,
            fetch_all=lambda column: feature_values[column],
        )
    )
    selection_calls: list[dict[str, object]] = []
    percentage_calls: list[dict[str, object]] = []

    def set_feature_selection(**kwargs: object) -> ArtifactRef:
        selection_calls.append(kwargs)
        return ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="feature_selection",
            artifact_id=str(len(selection_calls)) * 64,
        )

    def run_feature_percentage(
        cell_selection: ArtifactRef,
        feature_selection: ArtifactRef,
        *,
        invalidate_cache: bool,
    ) -> ArtifactRef:
        percentage_calls.append(
            {
                "cellSelection": cell_selection,
                "featureSelection": feature_selection,
                "invalidateCache": invalidate_cache,
            }
        )
        return ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="quality_metric",
            artifact_id=str(len(percentage_calls) + 2) * 64,
        )

    store.get_assay = lambda assay_name: assay
    store.set_feature_selection = set_feature_selection
    store.run_feature_percentage = run_feature_percentage

    sources = experimental_context_qc._derive_missing_percentage_artifacts(
        store,
        cell_selection=store.cell_selection,
        driver=("RNA", "RNA"),
        quality_sources=[],
    )

    assert [source.name for source in sources] == [
        "RNA_percentMito",
        "RNA_percentRibo",
    ]
    np.testing.assert_array_equal(
        selection_calls[0]["mask"],
        [True, False, False],
    )
    np.testing.assert_array_equal(
        selection_calls[1]["mask"],
        [False, True, False],
    )
    assert all(call["from_assay"] == "RNA" for call in selection_calls)
    assert all(call["invalidate_cache"] is False for call in selection_calls)
    assert [call["cellSelection"] for call in percentage_calls] == [
        store.cell_selection,
        store.cell_selection,
    ]
    assert all(call["invalidateCache"] is False for call in percentage_calls)


def test_validator_rejects_harmony_when_batch_confounds_biology() -> None:
    store = _Store()
    context = _context(store)
    decision = _design_decision(action="evaluateHarmony")

    asyncio.run(inspect_cell_covariates(context))
    asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=decision.batchCorrection.batchColumns,
        )
    )
    with pytest.raises(ModelRetry, match="correction is unsafe"):
        validate_experimental_context(decision, context.deps)

    accepted = validate_experimental_context(
        _design_decision(action="unsafe"),
        context.deps,
    )
    assert accepted.batchCorrection.action == "unsafe"


def test_harmony_safety_uses_only_exact_proposed_batch_columns() -> None:
    store = _Store()
    store.cells._values["disease"] = np.repeat(
        np.array(["case", "control", "case", "control"]),
        3,
    )
    store.cells._values["batch"] = np.repeat(
        np.array(["b1", "b1", "b2", "b2"]),
        3,
    )
    store.cells._values["sequencing_depth"] = np.repeat(
        np.array([1000.0, 2000.0, 1000.0, 2000.0]),
        3,
    )
    context = _context(store)
    decision = _design_decision(action="evaluateHarmony")

    asyncio.run(inspect_cell_covariates(context))
    analyzed = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=decision.batchCorrection.batchColumns,
        )
    )

    assert (
        analyzed.characterization.confounding[0]["estimability"]["coefficientEstimable"]
        is False
    )
    assert analyzed.batchSafety[0].status == "safe"
    validated = validate_experimental_context(decision, context.deps)
    assert validated.batchCorrection.action == "evaluateHarmony"


def test_batch_safety_does_not_depend_on_pairwise_selected_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.experimental_context import tools as module

    store = _Store()
    decision = _design_decision(action="evaluateHarmony")
    directions = {
        "columnDomains": decision.columnDomains,
        "coefficientsOfInterest": decision.coefficientsOfInterest,
        "unitsOfInference": {
            name: unit.model_dump(exclude_none=True)
            for name, unit in decision.unitsOfInference.items()
        },
    }
    characterization = module.characterize_covariates(
        store,
        studyContext="Case-control study with samples nested in donors.",
        model=None,
        cellSelection=store.cell_selection,
        directions=directions,
    )
    for report in characterization.confounding:
        for pair in report["pairs"]:
            pair["selected"] = False
    monkeypatch.setattr(
        module,
        "characterize_covariates",
        lambda *_args, **_kwargs: characterization,
    )
    context = _context(store)

    asyncio.run(inspect_cell_covariates(context))
    analyzed = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=decision.batchCorrection.batchColumns,
        )
    )

    assert all(
        pair["selected"] is False
        for pair in analyzed.characterization.confounding[0]["pairs"]
    )
    assert analyzed.batchSafety[0].status == "unsafe"


def test_batch_safety_checks_multiple_columns_jointly() -> None:
    store = _Store()
    store.cells._values["plate"] = np.repeat(
        np.array(["p1", "p2", "p1", "p2"]),
        3,
    )
    store.cells._values["disease"] = np.repeat(
        np.array([0.0, 1.0, 1.0, 2.0]),
        3,
    )
    decision = _design_decision(action="evaluateHarmony")
    decision.columnDomains["plate"] = "technical"
    context = _context(
        store,
        directions={"columnKinds": {"disease": "continuous"}},
    )

    asyncio.run(inspect_cell_covariates(context))
    batch_only = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=["batch"],
        )
    )
    plate_only = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=["plate"],
        )
    )
    context = _context(store, directions={"columnKinds": {"disease": "continuous"}})
    asyncio.run(inspect_cell_covariates(context))
    joint = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=["batch", "plate"],
        )
    )

    assert batch_only.batchSafety[0].status == "safe"
    assert plate_only.batchSafety[0].status == "safe"
    assert joint.batchSafety[0].status == "unsafe"


def test_harmony_requires_safety_for_exact_proposed_batch_set() -> None:
    store = _Store()
    context = _context(store)
    decision = _design_decision(action="evaluateHarmony")

    asyncio.run(inspect_cell_covariates(context))
    asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=[],
        )
    )

    with pytest.raises(ModelRetry, match="exact proposed batch columns"):
        validate_experimental_context(decision, context.deps)


def test_score_tool_reports_missing_graph_without_writing() -> None:
    store = _Store()
    context = _context(store)
    artifacts_before = store.list_artifacts(from_assay="RNA")

    evaluation = asyncio.run(
        score_current_representation(
            context,
            batch_column="batch",
            biological_column="cell_type",
        )
    )

    assert evaluation.available is False
    assert evaluation.metrics == {}
    assert evaluation.notes == ["No exact neighbors artifact was supplied"]
    assert store.list_artifacts(from_assay="RNA") == artifacts_before
    assert store.assay_state_lookups == []
    assert sorted(store.zw.group_keys()) == ["artifacts", "cellData"]


def test_score_tool_rejects_non_categorical_or_nontechnical_batch_column() -> None:
    context = _context(_Store())
    context.deps.characterization = CovariateCharacterization(
        status="done",
        columns=[
            {
                "name": "sequencing_depth",
                "domain": "ignore",
                "kind": "continuous",
            }
        ],
    )

    with pytest.raises(ModelRetry, match="categorical technical batch column"):
        asyncio.run(
            score_current_representation(
                context,
                batch_column="sequencing_depth",
            )
        )


def test_metric_evidence_namespaces_exact_representation_artifacts() -> None:
    store = _MetricStore()
    cell_selection, neighbors, connectivity_map = _configure_graph_lineage(store)
    context = _context(
        store,
        neighbors=neighbors,
        connectivity_map=connectivity_map,
        cell_selection=cell_selection,
    )

    evaluation = asyncio.run(
        score_current_representation(
            context,
            batch_column="batch",
            biological_column="cell_type",
        )
    )

    assert evaluation.available is True
    assert evaluation.assay == "RNA"
    assert evaluation.cellSelection is not None
    assert evaluation.cellSelection.artifactId == cell_selection.artifact_id
    assert evaluation.neighbors is not None
    assert evaluation.neighbors.artifactId == neighbors.artifact_id
    assert evaluation.connectivityMap is not None
    assert evaluation.connectivityMap.artifactId == connectivity_map.artifact_id
    assert evaluation.metrics == {
        "iLISI:batch": 0.7,
        "proportionalBatchMixing:batch": 0.8,
        "cLISI:cell_type": 0.9,
        "graphConnectivity:cell_type": 0.95,
    }
    assert all("assay:RNA" in value for value in evaluation.evidenceIds)
    assert any(
        f"neighbors:{neighbors.artifact_id}" in value
        for value in evaluation.evidenceIds
    )
    assert any(
        f"connectivity:{connectivity_map.artifact_id}" in value
        for value in evaluation.evidenceIds
    )
    assert store.metric_calls == [
        ("metric_ilisi", "batch", neighbors),
        ("metric_proportional_batch_mixing", "batch", neighbors),
        ("metric_clisi", "cell_type", neighbors),
        ("metric_graph_connectivity", "cell_type", connectivity_map),
    ]
    assert store.assay_state_lookups == []
    assert sorted(store.zw.group_keys()) == ["artifacts", "cellData"]


@pytest.mark.parametrize("graph_source", ["explicit", "pipelineRun"])
def test_agent_uses_exact_graph_lineage_without_current_state_lookup(
    graph_source: str,
) -> None:
    store = _MetricStore()
    cell_selection, neighbors, connectivity_map = _configure_graph_lineage(store)
    values_before = {
        column: values.copy() for column, values in store.cells._values.items()
    }
    state = {"request": 0}

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        request = state["request"]
        state["request"] += 1
        if request == 0:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="inspect_cell_covariates", args={})]
            )
        if request == 1:
            decision = _design_decision()
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="analyze_experimental_design",
                        args={
                            "column_domains": decision.columnDomains,
                            "coefficients_of_interest": (
                                decision.coefficientsOfInterest
                            ),
                            "units_of_inference": {
                                name: unit.model_dump()
                                for name, unit in decision.unitsOfInference.items()
                            },
                            "batch_columns": decision.batchCorrection.batchColumns,
                        },
                    )
                ]
            )
        if request == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="score_current_representation",
                        args={
                            "batch_column": "batch",
                            "biological_column": "cell_type",
                        },
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=_design_decision().model_dump(),
                )
            ]
        )

    agent = ExperimentalContextAgent(FunctionModel(reply))
    if graph_source == "pipelineRun":
        result = agent.run(
            store,
            study_context="Case-control study with samples nested in donors.",
            run=_completed_graph_run(store, neighbors, connectivity_map),
        )
    else:
        result = agent.run(
            store,
            study_context="Case-control study with samples nested in donors.",
            cell_selection=cell_selection,
            neighbors=neighbors,
            connectivity_map=connectivity_map,
        )

    assert result.status == "done"
    assert result.cellSelection is not None
    assert result.cellSelection.artifactId == cell_selection.artifact_id
    assert result.currentRepresentation.cellSelection == result.cellSelection
    assert result.currentRepresentation.neighbors is not None
    assert result.currentRepresentation.neighbors.artifactId == neighbors.artifact_id
    assert result.currentRepresentation.connectivityMap is not None
    assert (
        result.currentRepresentation.connectivityMap.artifactId
        == connectivity_map.artifact_id
    )
    tuning_handoff = result.to_parameter_tuning_handoff()
    biology_handoff = result.to_biological_handoff()
    assert tuning_handoff.cellSelection == result.cellSelection
    assert biology_handoff.cellSelection == result.cellSelection
    assert "cellKey" not in tuning_handoff.model_dump()
    assert "cellKey" not in biology_handoff.model_dump()
    assert store.inspected_artifacts == []
    assert store.metric_calls == [
        ("metric_ilisi", "batch", neighbors),
        ("metric_proportional_batch_mixing", "batch", neighbors),
        ("metric_clisi", "cell_type", neighbors),
        ("metric_graph_connectivity", "cell_type", connectivity_map),
    ]
    assert store.assay_state_lookups == []
    assert sorted(store.zw.group_keys()) == ["artifacts", "cellData"]
    for column, values in values_before.items():
        np.testing.assert_array_equal(store.cells._values[column], values)


def test_harmony_requires_resolved_units_and_estimability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.experimental_context import validation as module

    characterization = CovariateCharacterization(
        status="needsInput",
        columns=[
            {"name": "batch", "domain": "technical", "kind": "categorical"},
            {"name": "disease", "domain": "biological", "kind": "categorical"},
            {"name": "sample", "domain": "design", "kind": "categorical"},
        ],
        coefficients=[
            {
                "name": "disease",
                "scope": "unresolvedUnit",
                "observationUnit": None,
                "independentUnit": None,
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "characterize_context",
        lambda *_args, **_kwargs: characterization,
    )
    store = _Store()
    deps = ExperimentalContextDependencies(
        store=store,
        cellSelection=store.cell_selection,
        toolCalls=["inspect_cell_covariates", "analyze_experimental_design"],
    )
    decision = ExperimentalContextDecision(
        columnDomains={
            "batch": "technical",
            "disease": "biological",
            "sample": "design",
        },
        coefficientsOfInterest=["disease"],
        unitsOfInference={"disease": InferenceUnit(observationUnit="sample")},
        batchCorrection=BatchCorrectionPlan(
            action="evaluateHarmony",
            batchColumns=["batch"],
            preserveColumns=["disease"],
            metricsRequired=["iLISI", "cLISI"],
            evidenceIds=["column:batch", "coefficient:disease"],
        ),
    )

    with pytest.raises(ModelRetry, match="between-unit coefficient"):
        validate_experimental_context(decision, deps)


def test_harmony_rejects_nonbiological_preservation_column() -> None:
    store = _Store()
    context = _context(store)
    decision = _design_decision(action="evaluateHarmony")
    decision.batchCorrection.preserveColumns.append("batch")

    asyncio.run(inspect_cell_covariates(context))
    asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=decision.batchCorrection.batchColumns,
        )
    )

    with pytest.raises(ModelRetry, match="must be biological"):
        validate_experimental_context(decision, context.deps)


def test_returned_decision_canonicalizes_caller_directions() -> None:
    store = _Store()
    context = _context(
        store,
        directions={"columnDomains": {"batch": "technical"}},
    )
    decision = _design_decision(action="unsafe")
    decision.columnDomains["batch"] = "biological"

    asyncio.run(inspect_cell_covariates(context))
    asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=decision.batchCorrection.batchColumns,
        )
    )
    validated = validate_experimental_context(decision, context.deps)

    assert validated.columnDomains["batch"] == "technical"
    assert validated.coefficientsOfInterest == ["disease"]
    assert validated.unitsOfInference["disease"] == InferenceUnit(
        observationUnit="sample",
        independentUnit="donor",
    )


def test_named_artifact_and_qc_source_validation_edges() -> None:
    metric = example(NamedArtifactSource)
    identity = NamedArtifactSource(
        name="HTO_identity",
        artifact=ArtifactReferenceModel(
            assay="HTO",
            kind="hto_identity",
            artifactId="2" * 64,
        ),
    )

    with pytest.raises(ValidationError, match="surrounding whitespace"):
        NamedArtifactSource(
            name=" metric ",
            artifact=metric.artifact,
        )
    with pytest.raises(ValidationError, match="requires both name and artifact"):
        NamedArtifactSource(name="metric")
    with pytest.raises(ValidationError, match="requires both name and artifact"):
        NamedArtifactSource(artifact=metric.artifact)

    validate = experimental_context_contracts._validate_qc_sources
    with pytest.raises(ValueError, match="metadata attributes must be unique"):
        validate(
            action="globalGaussian",
            attributes=["a", "a"],
            artifact_metrics=[],
            sample_column=None,
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="cannot be blank"):
        validate(
            action="globalGaussian",
            attributes=[" a "],
            artifact_metrics=[],
            sample_column=None,
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="artifact metric names must be unique"):
        validate(
            action="globalGaussian",
            attributes=[],
            artifact_metrics=[metric, metric],
            sample_column=None,
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="quality_metric"):
        validate(
            action="globalGaussian",
            attributes=[],
            artifact_metrics=[identity],
            sample_column=None,
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="names collide"):
        validate(
            action="globalGaussian",
            attributes=[metric.name],
            artifact_metrics=[metric],
            sample_column=None,
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate(
            action="sampleMad",
            attributes=["metric"],
            artifact_metrics=[],
            sample_column="sample",
            sample_artifact=identity,
        )
    with pytest.raises(ValueError, match="sampleColumn cannot be blank"):
        validate(
            action="sampleMad",
            attributes=["metric"],
            artifact_metrics=[],
            sample_column=" sample ",
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="must reference an hto_identity"):
        validate(
            action="sampleMad",
            attributes=["metric"],
            artifact_metrics=[],
            sample_column=None,
            sample_artifact=metric,
        )
    colliding_identity = identity.model_copy(update={"name": metric.name})
    with pytest.raises(ValueError, match="sample and metric artifact names"):
        validate(
            action="sampleMad",
            attributes=[],
            artifact_metrics=[metric],
            sample_column=None,
            sample_artifact=colliding_identity,
        )
    with pytest.raises(ValueError, match="skip cannot include"):
        validate(
            action="skip",
            attributes=["metric"],
            artifact_metrics=[],
            sample_column=None,
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="requires at least one metric"):
        validate(
            action="globalGaussian",
            attributes=[],
            artifact_metrics=[],
            sample_column=None,
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="requires exactly one"):
        validate(
            action="sampleMad",
            attributes=["metric"],
            artifact_metrics=[],
            sample_column=None,
            sample_artifact=None,
        )
    with pytest.raises(ValueError, match="Only sampleMad"):
        validate(
            action="globalGaussian",
            attributes=["metric"],
            artifact_metrics=[],
            sample_column="sample",
            sample_artifact=None,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"coefficient": " disease "}, "coefficient cannot contain"),
        ({"sampleBy": " "}, "sampleBy must be"),
        ({"pairBy": " donor "}, "pairBy must be"),
        ({"groupOrder": ["case", "case"]}, "groupOrder must contain unique"),
        ({"groupOrder": ["case", np.inf]}, "cannot contain non-finite"),
        ({"expressionCutoff": np.nan}, "expressionCutoff must be finite"),
        (
            {"expressionCutoff": 0.1},
            "expressionCutoff is only used with fraction",
        ),
        (
            {"groupOrder": ["a", "b", "c"], "test": "mann_whitney"},
            "mann_whitney requires exactly two",
        ),
        (
            {"groupOrder": ["a", "b"], "test": "kruskal_wallis"},
            "kruskal_wallis requires at least three",
        ),
        ({"test": "wilcoxon"}, "wilcoxon requires exactly two groups"),
        (
            {"test": "mann_whitney", "pairBy": "donor"},
            "paired contrast must use the wilcoxon",
        ),
        ({"replicationPassed": False}, "licensed contrast requires resolved"),
        (
            {"status": "blocked", "blockedReasons": []},
            "non-licensed contrast requires blockedReasons",
        ),
    ],
)
def test_contrast_plan_rejects_inconsistent_licenses(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "coefficient": "disease",
        "groupOrder": ["case", "control"],
        "sampleBy": "sample",
        "test": "mann_whitney",
        "status": "licensed",
        "betweenUnitDesign": True,
        "replicationPassed": True,
        "estimabilityPassed": True,
    }
    values.update(changes)
    with pytest.raises(ValidationError, match=message):
        ContrastPlan.model_validate(values)


def test_contrast_plan_accepts_complete_paired_license() -> None:
    plan = ContrastPlan(
        coefficient="treatment",
        groupOrder=["treated", "control"],
        sampleBy="sample",
        pairBy="donor",
        test="wilcoxon",
        status="licensed",
        betweenUnitDesign=True,
        replicationPassed=True,
        estimabilityPassed=True,
        pairedCoveragePassed=True,
    )

    assert plan.status == "licensed"
    assert plan.test == "wilcoxon"


@pytest.mark.parametrize(
    ("registered_profile", "action", "sample_column", "sample_artifact", "message"),
    [
        (
            "retainWithFlags",
            "registeredMad",
            None,
            None,
            "non-filtering skip",
        ),
        (
            "retainWithFlags",
            "skip",
            "capture",
            None,
            "cannot include a capture source",
        ),
        ("globalMad5", "globalGaussian", None, None, "registeredMad action"),
        (
            "captureMad5",
            "registeredMad",
            None,
            None,
            "requires exactly one proven capture source",
        ),
        (
            "globalMad5",
            "registeredMad",
            "capture",
            None,
            "cannot include a capture source",
        ),
    ],
)
def test_registered_qc_source_contract_rejects_inconsistent_modes(
    registered_profile: str,
    action: str,
    sample_column: str | None,
    sample_artifact: NamedArtifactSource | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        experimental_context_contracts._validate_qc_sources(
            action=action,
            attributes=["RNA_nCounts"],
            artifact_metrics=[],
            sample_column=sample_column,
            sample_artifact=sample_artifact,
            registered_profile=registered_profile,
        )


def test_registered_qc_source_contract_accepts_global_capture_and_retain_modes() -> (
    None
):
    capture = NamedArtifactSource(
        name="capture",
        artifact=ArtifactReferenceModel(
            assay="HTO",
            kind="hto_identity",
            artifactId="4" * 64,
        ),
    )
    validate = experimental_context_contracts._validate_qc_sources

    validate(
        action="skip",
        attributes=[],
        artifact_metrics=[],
        sample_column=None,
        sample_artifact=None,
        registered_profile="retainWithFlags",
    )
    validate(
        action="registeredMad",
        attributes=["RNA_nCounts"],
        artifact_metrics=[],
        sample_column=None,
        sample_artifact=None,
        registered_profile="globalMad5",
    )
    validate(
        action="registeredMad",
        attributes=["RNA_nCounts"],
        artifact_metrics=[],
        sample_column=None,
        sample_artifact=capture,
        registered_profile="captureMad5",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"independentAdverseAxes": 1},
            "axis count must match",
        ),
        (
            {"wholeCaptureFailure": False},
            "requires at least two independent",
        ),
        (
            {"preservesConditionCoverage": False},
            "exclusion requires failure and preserved",
        ),
    ],
)
def test_capture_failure_evidence_rejects_inconsistent_state(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "capture": "failed",
        "activeCells": 20,
        "retainedCells": 2,
        "retainedFraction": 0.1,
        "adverseAxes": ["count", "feature"],
        "independentAdverseAxes": 2,
        "wholeCaptureFailure": True,
        "preservesConditionCoverage": True,
        "preservesIndependentUnitCoverage": True,
        "exclusionEligible": True,
    }
    values.update(changes)
    with pytest.raises(ValidationError, match=message):
        CaptureFailureEvidence.model_validate(values)


def test_cell_qc_profile_requires_exact_capture_failure_inventory() -> None:
    capture = NamedArtifactSource(
        name="capture",
        artifact=ArtifactReferenceModel(
            assay="HTO",
            kind="hto_identity",
            artifactId="5" * 64,
        ),
    )
    failure = CaptureFailureEvidence(
        capture="failed",
        activeCells=20,
        retainedCells=2,
        retainedFraction=0.1,
        adverseAxes=["count", "feature"],
        independentAdverseAxes=2,
        wholeCaptureFailure=True,
        preservesConditionCoverage=True,
        preservesIndependentUnitCoverage=True,
        exclusionEligible=True,
    )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        CellQcProfileEvidence(
            action="skip",
            captureColumn="capture",
            captureArtifact=capture,
        )
    with pytest.raises(ValidationError, match="failure evidence must be unique"):
        CellQcProfileEvidence(
            action="skip",
            captureFailureEvidence=[failure, failure],
        )
    with pytest.raises(ValidationError, match="failed captures must match"):
        CellQcProfileEvidence(
            action="skip",
            captureFailureEvidence=[failure],
        )
    with pytest.raises(ValidationError, match="excludable captures must match"):
        CellQcProfileEvidence(
            action="skip",
            captureFailureEvidence=[failure],
            failedCaptureCandidates=["failed"],
        )


def test_experimental_handoff_validation_edges() -> None:
    result = example(ExperimentalContextResult)
    without_selection = result.model_copy(update={"cellSelection": None})
    with pytest.raises(ValueError, match="lacks a cell selection"):
        without_selection.to_parameter_tuning_handoff()
    with pytest.raises(ValueError, match="lacks a cell selection"):
        without_selection.to_biological_handoff()

    with pytest.raises(ValueError, match="lacks exact batch safety"):
        result.model_copy(update={"batchSafety": []}).to_parameter_tuning_handoff()

    plan = result.decision.batchCorrection
    uncited_plan = plan.model_copy(
        update={
            "evidenceIds": [
                value
                for value in plan.evidenceIds
                if not value.startswith("batchEstimability:")
            ]
        }
    )
    with pytest.raises(ValueError, match="does not cite"):
        result.model_copy(
            update={
                "decision": result.decision.model_copy(
                    update={"batchCorrection": uncited_plan}
                )
            }
        ).to_parameter_tuning_handoff()

    unsafe_safety = result.batchSafety[0].model_copy(update={"status": "unsafe"})
    with pytest.raises(ValueError, match="non-safe"):
        result.model_copy(
            update={"batchSafety": [unsafe_safety]}
        ).to_parameter_tuning_handoff()

    unsafe_plan = plan.model_copy(update={"action": "unsafe"})
    with pytest.raises(ValueError, match="lacks exact unsafe"):
        result.model_copy(
            update={
                "decision": result.decision.model_copy(
                    update={"batchCorrection": unsafe_plan}
                )
            }
        ).to_parameter_tuning_handoff()

    with pytest.raises(ValueError, match="must be done"):
        result.model_copy(update={"status": "failed"}).to_biological_handoff()
    with pytest.raises(ValueError, match="Unknown coefficient"):
        result.to_biological_handoff("unknown")
    with pytest.raises(ValueError, match="Missing characterization"):
        result.model_copy(
            update={
                "characterization": result.characterization.model_copy(
                    update={"coefficients": []}
                )
            }
        ).to_biological_handoff("treatment")


def test_experimental_context_private_input_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    deps = _context(
        store,
        directions={
            "htoIdentityColumns": ["sample", "missing"],
            "htoIdentityColumn": "donor",
        },
    ).deps
    assert experimental_context_qc._hto_identity_columns(deps) == [
        "sample",
        "donor",
    ]

    unknown_tool = SimpleNamespace(name="future_tool")
    assert (
        experimental_context_tools._prepare_experimental_context_tool(
            SimpleNamespace(deps=deps),
            unknown_tool,
        )
        is unknown_tool
    )
    characterization = CovariateCharacterization(
        status="done",
        confounding=[{"coefficient": 3, "pairs": []}],
    )
    assert (
        experimental_context_contracts.characterization_evidence(characterization)
        == set()
    )

    deps.cellSelection = None
    with pytest.raises(ValueError, match="exact artifact"):
        experimental_context_qc._cell_selection_ref(deps)
    deps.cellSelection = _artifact_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cell_selection",
        artifact_id="4" * 64,
    )
    with pytest.raises(ValueError, match="datastore cell selection"):
        experimental_context_qc._cell_selection_ref(deps)
    del _artifact_ref

    with pytest.raises(TypeError, match="NamedArtifactSource"):
        experimental_context_qc._source_ref(
            object(),
            expected_kind="quality_metric",
        )
    blank_source = NamedArtifactSource.model_construct(
        name="",
        artifact=ArtifactReferenceModel(),
    )
    with pytest.raises(ValueError, match="non-empty semantic name"):
        experimental_context_qc._source_ref(
            blank_source,
            expected_kind="quality_metric",
        )
    with pytest.raises(ValueError, match="quality_metric"):
        experimental_context_qc._source_ref(
            NamedArtifactSource(
                name="identity",
                artifact=ArtifactReferenceModel(
                    assay="HTO",
                    kind="hto_identity",
                    artifactId="5" * 64,
                ),
            ),
            expected_kind="quality_metric",
        )

    duplicate = NamedArtifactSource(
        name="identity",
        artifact=ArtifactReferenceModel(
            assay="HTO",
            kind="hto_identity",
            artifactId="6" * 64,
        ),
    )
    deps.htoIdentityArtifacts = [duplicate, duplicate]
    with pytest.raises(ValueError, match="names must be unique"):
        experimental_context_qc._hto_artifact_map(deps)

    deps.cellSelection = store.cell_selection
    monkeypatch.setattr(
        experimental_context_qc,
        "read_stored_selection_mask",
        lambda *_args, **_kwargs: np.ones(store.cells.N + 1, dtype=bool),
    )
    with pytest.raises(ValueError, match="aligned boolean selection"):
        experimental_context_qc._active_cell_count(deps)


def test_core_qc_reports_unavailable_default_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    context = _context(store)
    deps = context.deps
    active = np.ones(store.cells.N, dtype=bool)
    notes: list[str] = []
    profile = experimental_context_qc._global_qc_profile(
        deps,
        ("RNA", "RNA"),
        active,
        store.cells.N,
        {"constant": np.ones(store.cells.N)},
        ["constant"],
        [],
        notes,
    )
    assert profile is None
    assert "constant metric 'constant'" in notes[0]

    monkeypatch.setattr(
        experimental_context_qc,
        "gaussian_quantile_bounds",
        lambda *_args, **_kwargs: (float("nan"), float("nan")),
    )
    notes = []
    profile = experimental_context_qc._global_qc_profile(
        deps,
        ("RNA", "RNA"),
        active,
        store.cells.N,
        {"metric": np.arange(store.cells.N, dtype=float)},
        ["metric"],
        [],
        notes,
    )
    assert profile is None
    assert "non-finite Gaussian bounds" in notes[0]


def test_design_analysis_rejects_invalid_batch_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    decision = _design_decision()

    for batch_columns, message in (
        (["batch", "batch"], "must be unique"),
        (["unknown"], "Unknown batch column"),
        (["disease"], "classified as technical"),
        (["sequencing_depth"], "categorical for Harmony"),
    ):
        context = _context(store)
        with pytest.raises(ModelRetry, match=message):
            asyncio.run(
                analyze_experimental_design(
                    context,
                    column_domains=decision.columnDomains,
                    coefficients_of_interest=decision.coefficientsOfInterest,
                    units_of_inference=decision.unitsOfInference,
                    batch_columns=batch_columns,
                )
            )

    failed = CovariateCharacterization(
        status="failed",
        notes=["design failed"],
    )
    monkeypatch.setattr(
        experimental_context_tools,
        "characterize_covariates",
        lambda *_args, **_kwargs: failed,
    )
    with pytest.raises(ModelRetry, match="design failed"):
        asyncio.run(
            analyze_experimental_design(
                _context(store),
                column_domains={},
                coefficients_of_interest=[],
                units_of_inference={},
                batch_columns=[],
            )
        )


def test_design_analysis_records_not_computed_estimability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    decision = _design_decision()
    context = _context(store)
    unresolved = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference={},
            batch_columns=["batch"],
        )
    )
    assert unresolved.batchSafety[0].status == "notComputed"
    assert unresolved.batchSafety[0].estimability["reason"] == (
        "unresolvedCoefficientDesign"
    )

    monkeypatch.setattr(
        experimental_context_tools,
        "reduce_observation_units",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad design")),
    )
    context = _context(store)
    failed = asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=["batch"],
        )
    )
    assert failed.batchSafety[0].status == "notComputed"
    assert failed.batchSafety[0].estimability["reason"] == "ValueError"


def test_representation_scoring_input_and_metric_failure_edges() -> None:
    context = _context(_Store())
    with pytest.raises(ModelRetry, match="Unknown batch column"):
        asyncio.run(score_current_representation(context, batch_column="unknown"))
    with pytest.raises(ModelRetry, match="Unknown biological column"):
        asyncio.run(
            score_current_representation(
                context,
                batch_column="batch",
                biological_column="unknown",
            )
        )

    context.deps.neighbors = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="7" * 64,
    )
    with pytest.raises(ModelRetry, match="exact neighbors"):
        asyncio.run(score_current_representation(context, batch_column="batch"))

    context.deps.neighbors = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="8" * 64,
    )
    context.deps.connectivityMap = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="reduction",
        artifact_id="9" * 64,
    )
    with pytest.raises(ModelRetry, match="exact connectivity graph"):
        asyncio.run(score_current_representation(context, batch_column="batch"))

    class FailingMetricStore(_MetricStore):
        def metric_ilisi(self, *_args: Any, **_kwargs: Any) -> float:
            raise ValueError("ilisi unavailable")

        def metric_proportional_batch_mixing(
            self, *_args: Any, **_kwargs: Any
        ) -> float:
            raise RuntimeError("mixing unavailable")

        def metric_clisi(self, *_args: Any, **_kwargs: Any) -> float:
            raise KeyError("clisi unavailable")

        def metric_graph_connectivity(self, *_args: Any, **_kwargs: Any) -> float:
            raise TypeError("connectivity unavailable")

    store = FailingMetricStore()
    cell_selection, neighbors, connectivity = _configure_graph_lineage(store)
    context = _context(
        store,
        cell_selection=cell_selection,
        neighbors=neighbors,
        connectivity_map=connectivity,
    )
    evaluation = asyncio.run(
        score_current_representation(
            context,
            batch_column="batch",
            biological_column="cell_type",
        )
    )
    assert not evaluation.available
    assert len(evaluation.notes) == 4


def test_batch_correction_plan_validation_edges() -> None:
    store = _Store()
    context = _context(store)
    decision = _design_decision(action="unsafe")
    asyncio.run(inspect_cell_covariates(context))
    asyncio.run(
        analyze_experimental_design(
            context,
            column_domains=decision.columnDomains,
            coefficients_of_interest=decision.coefficientsOfInterest,
            units_of_inference=decision.unitsOfInference,
            batch_columns=decision.batchCorrection.batchColumns,
        )
    )
    characterization = context.deps.characterization
    assert characterization is not None
    records = {
        record["name"]: dict(record)
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    coefficient_records = {
        record["name"]: dict(record)
        for record in characterization.coefficients
        if isinstance(record.get("name"), str)
    }
    units = {
        name: value.model_dump(exclude_none=True)
        for name, value in decision.unitsOfInference.items()
    }

    def validate(
        candidate: ExperimentalContextDecision,
        *,
        deps: ExperimentalContextDependencies | None = None,
        candidate_records: dict[str, dict[str, Any]] | None = None,
        candidate_coefficients: dict[str, dict[str, Any]] | None = None,
        requested: set[str] | None = None,
    ) -> None:
        experimental_context_validation._validate_batch_correction_plan(
            candidate,
            deps or context.deps,
            characterization,
            requested if requested is not None else {"disease"},
            units,
            candidate_records or records,
            candidate_coefficients or coefficient_records,
        )

    validate(decision)

    def changed_plan(**updates: Any) -> ExperimentalContextDecision:
        return decision.model_copy(
            update={
                "batchCorrection": decision.batchCorrection.model_copy(update=updates)
            }
        )

    cases: list[tuple[ExperimentalContextDecision, str]] = [
        (
            decision.model_copy(
                update={
                    "columnDomains": {**decision.columnDomains, "ghost": "technical"}
                }
            ),
            "Unknown column domain",
        ),
        (changed_plan(action="evaluateHarmony", batchColumns=[]), "at least one"),
        (changed_plan(action="unsafe", batchColumns=[]), "exact batch columns"),
        (changed_plan(action="skip", batchColumns=["batch"]), "must not include"),
        (
            changed_plan(action="needsInput", batchColumns=[]).model_copy(
                update={"needsInput": []}
            ),
            "concrete question",
        ),
        (changed_plan(batchColumns=["batch", "batch"]), "must be unique"),
        (changed_plan(batchColumns=["ghost"]), "Unknown batch column"),
        (changed_plan(batchColumns=["disease"]), "classified as technical"),
        (
            changed_plan(batchColumns=["sequencing_depth"]),
            "must be categorical",
        ),
        (
            changed_plan(
                action="evaluateHarmony",
                metricsRequired=["cLISI"],
            ),
            "requires iLISI",
        ),
        (
            changed_plan(
                action="evaluateHarmony",
                metricsRequired=["iLISI"],
            ),
            "requires cLISI",
        ),
        (
            changed_plan(
                action="evaluateHarmony",
                preserveColumns=[],
                metricsRequired=["iLISI", "cLISI"],
            ),
            "preserveColumns must include",
        ),
        (
            changed_plan(
                action="evaluateHarmony",
                preserveColumns=["disease", "ghost"],
                metricsRequired=["iLISI", "cLISI"],
            ),
            "Unknown preservation column",
        ),
    ]
    for candidate, message in cases:
        with pytest.raises(ModelRetry, match=message):
            validate(candidate)

    batch_is_coefficient_records = {**records, "batch": dict(records["batch"])}
    with pytest.raises(ModelRetry, match="cannot be a coefficient"):
        validate(
            decision,
            candidate_records=batch_is_coefficient_records,
            requested={"batch"},
        )

    unresolved = {name: dict(value) for name, value in coefficient_records.items()}
    unresolved["disease"]["scope"] = "unresolvedUnit"
    with pytest.raises(ModelRetry, match="between-unit coefficient"):
        validate(
            changed_plan(action="evaluateHarmony"),
            candidate_coefficients=unresolved,
        )

    continuous_records = {
        **records,
        "continuous_biology": {
            "name": "continuous_biology",
            "domain": "biological",
            "kind": "continuous",
        },
    }
    with pytest.raises(ModelRetry, match="Batch correction is unsafe"):
        validate(
            changed_plan(
                action="evaluateHarmony",
                preserveColumns=["disease", "continuous_biology"],
            ),
            candidate_records=continuous_records,
        )

    safety = next(iter(context.deps.batchSafety.values()))
    missing_evidence = changed_plan(
        action="evaluateHarmony",
        evidenceIds=[
            value
            for value in decision.batchCorrection.evidenceIds
            if value != safety.evidenceId
        ],
    )
    with pytest.raises(ModelRetry, match="must cite exact batch"):
        validate(missing_evidence)

    not_computed_deps = context.deps.model_copy(
        update={
            "batchSafety": {
                safety.evidenceId: safety.model_copy(update={"status": "notComputed"})
            }
        }
    )
    with pytest.raises(ModelRetry, match="could not be computed"):
        validate(changed_plan(action="evaluateHarmony"), deps=not_computed_deps)

    safe_deps = context.deps.model_copy(
        update={
            "batchSafety": {
                safety.evidenceId: safety.model_copy(update={"status": "safe"})
            }
        }
    )
    with pytest.raises(ModelRetry, match="use action='evaluateHarmony'"):
        validate(decision, deps=safe_deps)

    unknown_evidence = changed_plan(
        action="skip",
        batchColumns=[],
        evidenceIds=["unknown:evidence"],
    )
    with pytest.raises(ModelRetry, match="Unknown evidence IDs"):
        validate(unknown_evidence)

    no_evidence = changed_plan(action="skip", batchColumns=[], evidenceIds=[])
    with pytest.raises(ModelRetry, match="require evidence IDs"):
        validate(no_evidence)

    stale_id = "metric:iLISI:batch:stale"
    stale_deps = context.deps.model_copy(
        update={"evidenceIds": {*context.deps.evidenceIds, stale_id}}
    )
    stale = changed_plan(
        action="skip",
        batchColumns=[],
        evidenceIds=[stale_id],
    )
    with pytest.raises(ModelRetry, match="returned exact representation"):
        validate(stale, deps=stale_deps)
