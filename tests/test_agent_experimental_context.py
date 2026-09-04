"""Tests for the tool-driven Experimental Context Agent."""

import asyncio
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
import zarr
from pydantic import ValidationError
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from zarr.storage import MemoryStore

import scarf.agent.experimental_context as experimental_context_module
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    BatchSafetyEvidence,
    CellQcPlan,
    CellQcProfileEvidence,
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
from scarf.agent.characterize_covariates import (
    CovariateCharacterization,
    _SelectionBoundCells,
)
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


def _write_cell_artifact(
    store: _Store,
    *,
    name: str,
    kind: Literal["quality_metric", "hto_identity"],
    values: np.ndarray,
    assay: str,
) -> NamedArtifactSource:
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay=assay,
        kind=kind,
        operation=f"test_{kind}_source",
        parameters={"name": name},
        inputs={},
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
    from scarf.agent import experimental_context as module

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
        assert isinstance(model.get_example(), model)
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
    ]
    assert tool_names == {
        "inspect_cell_covariates",
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


def test_handoff_builders_reject_incomplete_or_ambiguous_results() -> None:
    incomplete = ExperimentalContextResult.get_blank()
    with pytest.raises(ValueError, match="must be done"):
        incomplete.to_parameter_tuning_handoff()

    ambiguous = ExperimentalContextResult.get_example()
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

    decision.cellQc = CellQcPlan(
        action=sample_profile.action,
        profileId=sample_profile.profileId,
        driverAssay=sample_profile.driverAssay,
        driverAssayType=sample_profile.driverAssayType,
        sampleColumn=sample_profile.sampleColumn,
        sampleArtifact=sample_profile.sampleArtifact,
        attributes=sample_profile.attributes,
        artifactMetrics=sample_profile.artifactMetrics,
        rationale="Use sample-aware retention evidence.",
        evidenceIds=[sample_profile.evidenceId],
    )
    validated = validate_experimental_context(decision, context.deps)
    assert validated.cellQc == decision.cellQc


def test_caller_qc_direction_overrides_model_profile_selection() -> None:
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
    decision.cellQc = CellQcPlan(
        action="skip",
        profileId="model-authored-profile",
        evidenceIds=["model-authored-evidence"],
    )

    validated = validate_experimental_context(decision, context.deps)

    assert validated.cellQc.action == "sampleMad"
    assert validated.cellQc.sampleColumn == "sample"
    assert validated.cellQc.evidenceIds == [
        next(
            profile.evidenceId
            for profile in context.deps.qcProfiles.values()
            if profile.action == "sampleMad"
        )
    ]


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
    from scarf.agent import experimental_context as module

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
    from scarf.agent import experimental_context as module

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
        "characterize_covariates",
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
    metric = NamedArtifactSource.get_example()
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

    validate = experimental_context_module._validate_qc_sources
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


def test_experimental_handoff_validation_edges() -> None:
    result = ExperimentalContextResult.get_example()
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
    assert experimental_context_module._hto_identity_columns(deps) == [
        "sample",
        "donor",
    ]

    unknown_tool = SimpleNamespace(name="future_tool")
    assert (
        experimental_context_module._prepare_experimental_context_tool(
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
        experimental_context_module.characterization_evidence(characterization) == set()
    )

    deps.cellSelection = None
    with pytest.raises(ValueError, match="exact artifact"):
        experimental_context_module._cell_selection_ref(deps)
    deps.cellSelection = _artifact_ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cell_selection",
        artifact_id="4" * 64,
    )
    with pytest.raises(ValueError, match="datastore cell selection"):
        experimental_context_module._cell_selection_ref(deps)
    del _artifact_ref

    with pytest.raises(TypeError, match="NamedArtifactSource"):
        experimental_context_module._source_ref(
            object(),
            expected_kind="quality_metric",
        )
    blank_source = NamedArtifactSource.model_construct(
        name="",
        artifact=ArtifactReferenceModel(),
    )
    with pytest.raises(ValueError, match="non-empty semantic name"):
        experimental_context_module._source_ref(
            blank_source,
            expected_kind="quality_metric",
        )
    with pytest.raises(ValueError, match="quality_metric"):
        experimental_context_module._source_ref(
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
        experimental_context_module._hto_artifact_map(deps)

    deps.cellSelection = store.cell_selection
    monkeypatch.setattr(
        experimental_context_module,
        "read_stored_selection_mask",
        lambda *_args, **_kwargs: np.ones(store.cells.N + 1, dtype=bool),
    )
    with pytest.raises(ValueError, match="aligned boolean selection"):
        experimental_context_module._active_cell_count(deps)


def test_qc_profile_degradation_and_selection_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    context = _context(store)
    deps = context.deps
    active = np.ones(store.cells.N, dtype=bool)
    notes: list[str] = []
    profile = experimental_context_module._global_qc_profile(
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
    assert notes == ["Ignored constant QC metric 'constant'"]

    monkeypatch.setattr(
        experimental_context_module,
        "gaussian_quantile_bounds",
        lambda *_args, **_kwargs: (float("nan"), float("nan")),
    )
    notes = []
    profile = experimental_context_module._global_qc_profile(
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

    skip = CellQcProfileEvidence(
        profileId="skip",
        action="skip",
        activeCells=store.cells.N,
        retainedCells=store.cells.N,
        retainedFraction=1.0,
        evidenceId="qcProfile:skip",
    )
    global_profile = CellQcProfileEvidence(
        profileId="global",
        action="globalGaussian",
        driverAssay="RNA",
        driverAssayType="RNA",
        attributes=["RNA_nCounts"],
        activeCells=store.cells.N,
        retainedCells=store.cells.N - 1,
        retainedFraction=(store.cells.N - 1) / store.cells.N,
        evidenceId="qcProfile:global",
    )
    deps.qcProfiles = {"skip": skip, "global": global_profile}
    characterization = CovariateCharacterization(status="done")

    deps.directions = {"cellQc": {"profileId": 3}}
    with pytest.raises(ModelRetry, match="profileId direction must be a string"):
        experimental_context_module._canonical_cell_qc_plan(
            CellQcPlan(), deps, characterization
        )
    deps.directions = {
        "cellQc": {"sampleColumn": "sample", "sampleArtifactName": "identity"}
    }
    with pytest.raises(ModelRetry, match="cannot select both"):
        experimental_context_module._canonical_cell_qc_plan(
            CellQcPlan(), deps, characterization
        )
    deps.directions = {"cellQc": {"sampleArtifactName": 3}}
    with pytest.raises(ModelRetry, match="sampleArtifactName must be a string"):
        experimental_context_module._canonical_cell_qc_plan(
            CellQcPlan(), deps, characterization
        )
    deps.directions = {"cellQc": {"action": "unknown"}}
    with pytest.raises(ModelRetry, match="Unsupported cellQc.action"):
        experimental_context_module._canonical_cell_qc_plan(
            CellQcPlan(), deps, characterization
        )
    deps.directions = {"cellQc": {"action": "sampleMad"}}
    with pytest.raises(ModelRetry, match="exactly one offered profile"):
        experimental_context_module._canonical_cell_qc_plan(
            CellQcPlan(), deps, characterization
        )
    deps.directions = {"cellQc": {"profileId": "unknown"}}
    with pytest.raises(ModelRetry, match="was not offered"):
        experimental_context_module._canonical_cell_qc_plan(
            CellQcPlan(), deps, characterization
        )

    deps.directions = {}
    selected = experimental_context_module._canonical_cell_qc_plan(
        CellQcPlan(), deps, characterization
    )
    assert selected.profileId == "global"

    mismatched = CellQcPlan(
        action="globalGaussian",
        profileId="global",
        driverAssay="RNA",
        driverAssayType="RNA",
        attributes=["different_metric"],
        evidenceIds=["qcProfile:global"],
    )
    with pytest.raises(ModelRetry, match="copy the selected offered profile"):
        experimental_context_module._canonical_cell_qc_plan(
            mismatched, deps, characterization
        )
    missing_evidence = mismatched.model_copy(
        update={"attributes": ["RNA_nCounts"], "evidenceIds": []}
    )
    with pytest.raises(ModelRetry, match="cite its exact profile"):
        experimental_context_module._canonical_cell_qc_plan(
            missing_evidence, deps, characterization
        )


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
        experimental_context_module,
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
        experimental_context_module,
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
        experimental_context_module._validate_batch_correction_plan(
            candidate,
            deps or context.deps,
            characterization,
            requested if requested is not None else {"disease"},
            units,
            candidate.cellQc,
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
    with pytest.raises(ModelRetry, match="must be categorical"):
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
