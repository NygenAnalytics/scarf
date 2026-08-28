"""Tests for the tool-driven Experimental Context Agent."""

import asyncio
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    BatchSafetyEvidence,
    CovariateEvidence,
    ExperimentalContextAgent,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    InferenceUnit,
    RepresentationEvaluation,
    analyze_experimental_design,
    inspect_cell_covariates,
    score_current_representation,
    validate_experimental_context,
)
from scarf.agent.characterize_covariates import CovariateCharacterization
from scarf.agent.types import ExperimentalBiologyHandoff, ExperimentalTuningHandoff

type TestAction = Literal["skip", "evaluateHarmony", "unsafe", "needsInput"]


class _Cells:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self._values = values
        self.N = len(next(iter(values.values())))

    @property
    def columns(self) -> list[str]:
        return list(self._values)

    def fetch_all(self, column: str) -> np.ndarray:
        return self._values[column]

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
    zw: dict[str, Any] = {}

    def __init__(self) -> None:
        n_cells = 12
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

    @staticmethod
    def get_assay_state(from_assay: str | None = None) -> None:
        del from_assay
        return None

    @staticmethod
    def list_artifacts(from_assay: str | None = None) -> list[Any]:
        del from_assay
        return []


def _context(
    store: _Store,
    *,
    directions: dict[str, object] | None = None,
) -> RunContext[ExperimentalContextDependencies]:
    return RunContext(
        deps=ExperimentalContextDependencies(
            store=store,
            studyContext="Case-control study with samples nested in donors.",
            cellKey="I",
            directions=dict(directions or {}),
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
    artifacts_before = store.list_artifacts(from_assay="RNA")
    state_before = store.get_assay_state("RNA")

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
    assert store.list_artifacts(from_assay="RNA") == artifacts_before
    assert store.get_assay_state("RNA") == state_before


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
        cellKey="I",
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
            from_assay="RNA",
        )
    )

    assert evaluation.available is False
    assert evaluation.metrics == {}
    assert evaluation.notes == ["No current neighbors artifact is available"]
    assert store.list_artifacts(from_assay="RNA") == artifacts_before


def test_metric_evidence_namespaces_exact_representation_artifacts() -> None:
    class MetricStore(_Store):
        def get_assay_state(self, from_assay: str | None = None) -> SimpleNamespace:
            assert from_assay == "RNA"
            return SimpleNamespace(
                assay="RNA",
                cell_key="I",
                neighbors=SimpleNamespace(artifact_id="neighbors-1"),
                connectivity_map=SimpleNamespace(artifact_id="connectivity-1"),
            )

        @staticmethod
        def metric_ilisi(*_args: object, **_kwargs: object) -> float:
            return 0.7

        @staticmethod
        def metric_proportional_batch_mixing(
            *_args: object,
            **_kwargs: object,
        ) -> float:
            return 0.8

        @staticmethod
        def metric_clisi(*_args: object, **_kwargs: object) -> float:
            return 0.9

        @staticmethod
        def metric_graph_connectivity(*_args: object, **_kwargs: object) -> float:
            return 0.95

    context = _context(MetricStore())

    evaluation = asyncio.run(
        score_current_representation(
            context,
            batch_column="batch",
            biological_column="cell_type",
            from_assay="RNA",
        )
    )

    assert evaluation.available is True
    assert all("assay:RNA" in value for value in evaluation.evidenceIds)
    assert any("neighbors:neighbors-1" in value for value in evaluation.evidenceIds)
    assert any(
        "connectivity:connectivity-1" in value for value in evaluation.evidenceIds
    )


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
    deps = ExperimentalContextDependencies(
        store=_Store(),
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
