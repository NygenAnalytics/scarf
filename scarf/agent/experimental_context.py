"""Tool-driven experimental-design and batch-correction assessment."""

import json
import math
from collections.abc import Mapping
from textwrap import dedent
from typing import Any, Literal

from .config._deps import AGENT_INSTALL_HINT
from .config.agent_exec import run_agent_sync
from .characterize_covariates import (
    CovariateCharacterization,
    characterize_covariates,
)
from .config import AgentRunConfig
from ..metadata.queries import reduce_observation_units
from ..metrics.association import coefficient_estimability
from .types import (
    AgentDataModel,
    AgentRunInfo,
    BatchCorrectionAction,
    BatchSafetyEvidence,
    BatchSafetyStatus,
    ExperimentalBiologyHandoff,
    ExperimentalTuningHandoff,
    StageStatus,
)

try:
    from pydantic import ConfigDict, Field
    from pydantic_ai import ModelRetry, RunContext
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "BatchCorrectionPlan",
    "BatchSafetyEvidence",
    "CovariateEvidence",
    "ExperimentalContextAgent",
    "ExperimentalContextDecision",
    "ExperimentalContextDependencies",
    "ExperimentalContextResult",
    "InferenceUnit",
    "RepresentationEvaluation",
    "analyze_experimental_design",
    "inspect_cell_covariates",
    "score_current_representation",
    "validate_experimental_context",
]

type ColumnDomain = Literal["biological", "technical", "design", "ignore", "unknown"]
type IntegrationMetric = Literal[
    "iLISI",
    "cLISI",
    "graphConnectivity",
    "proportionalBatchMixing",
]

_CONTEXT_LIMIT = 1200


class InferenceUnit(AgentDataModel):
    """Observation and independent units for one biological coefficient."""

    observationUnit: str | None = None
    independentUnit: str | None = None

    @classmethod
    def get_blank(cls) -> "InferenceUnit":
        return cls()

    @classmethod
    def get_example(cls) -> "InferenceUnit":
        return cls(observationUnit="sample", independentUnit="donor")


class BatchCorrectionPlan(AgentDataModel):
    """A grounded recommendation about whether Harmony should be evaluated."""

    action: BatchCorrectionAction
    batchColumns: list[str] = Field(default_factory=list)
    preserveColumns: list[str] = Field(default_factory=list)
    metricsRequired: list[IntegrationMetric] = Field(default_factory=list)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "BatchCorrectionPlan":
        return cls(action="needsInput")

    @classmethod
    def get_example(cls) -> "BatchCorrectionPlan":
        return cls(
            action="evaluateHarmony",
            batchColumns=["batch"],
            preserveColumns=["cell_type", "treatment"],
            metricsRequired=[
                "iLISI",
                "cLISI",
                "graphConnectivity",
            ],
            rationale=(
                "Batch is technical and crossed with treatment, so compare an exact "
                "Harmony candidate while protecting biological labels."
            ),
            evidenceIds=[
                "column:batch",
                "estimability:treatment",
                "batchEstimability:treatment:batch",
            ],
        )


class ExperimentalContextDecision(AgentDataModel):
    """Model-authored choices that are revalidated against the datastore."""

    columnDomains: dict[str, ColumnDomain] = Field(default_factory=dict)
    coefficientsOfInterest: list[str] = Field(default_factory=list)
    unitsOfInference: dict[str, InferenceUnit] = Field(default_factory=dict)
    batchCorrection: BatchCorrectionPlan = Field(
        default_factory=BatchCorrectionPlan.get_blank
    )
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    needsInput: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextDecision":
        return cls()

    @classmethod
    def get_example(cls) -> "ExperimentalContextDecision":
        return cls(
            columnDomains={
                "batch": "technical",
                "sample": "design",
                "donor": "design",
                "treatment": "biological",
            },
            coefficientsOfInterest=["treatment"],
            unitsOfInference={"treatment": InferenceUnit.get_example()},
            batchCorrection=BatchCorrectionPlan.get_example(),
            rationale="Treatment is the primary between-sample contrast.",
            evidenceIds=[
                "column:batch",
                "column:donor",
                "column:sample",
                "column:treatment",
            ],
        )


class RepresentationEvaluation(AgentDataModel):
    """Bounded integration metrics for the datastore's current graph state."""

    available: bool = False
    assay: str | None = None
    cellKey: str | None = None
    neighborsArtifactId: str | None = None
    connectivityArtifactId: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "RepresentationEvaluation":
        return cls()

    @classmethod
    def get_example(cls) -> "RepresentationEvaluation":
        return cls(
            available=True,
            assay="RNA",
            cellKey="I",
            neighborsArtifactId="example-neighbors",
            connectivityArtifactId="example-connectivity",
            metrics={"iLISI:batch": 0.71, "cLISI:cell_type": 0.94},
            evidenceIds=[
                "metric:iLISI:batch:assay:RNA:neighbors:example-neighbors",
                "metric:cLISI:cell_type:assay:RNA:neighbors:example-neighbors",
            ],
        )


class CovariateEvidence(AgentDataModel):
    """One deterministic covariate characterization returned by a tool."""

    characterization: CovariateCharacterization = Field(
        default_factory=lambda: CovariateCharacterization(status="needsInput")
    )
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "CovariateEvidence":
        return cls(
            characterization=CovariateCharacterization(
                status="done",
                notes=["Example deterministic covariate characterization"],
            ),
            evidenceIds=["column:batch"],
        )


class ExperimentalContextResult(AgentDataModel):
    """Canonical experimental-context report returned to the caller."""

    status: StageStatus
    decision: ExperimentalContextDecision
    characterization: CovariateCharacterization
    cellKey: str = "I"
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    currentRepresentation: RepresentationEvaluation = Field(
        default_factory=RepresentationEvaluation.get_blank
    )
    notes: list[str] = Field(default_factory=list)
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextResult":
        return cls(
            status="needsInput",
            decision=ExperimentalContextDecision.get_blank(),
            characterization=CovariateCharacterization(status="needsInput"),
        )

    @classmethod
    def get_example(cls) -> "ExperimentalContextResult":
        return cls(
            status="done",
            decision=ExperimentalContextDecision.get_example(),
            characterization=CovariateCharacterization(
                status="done",
                notes=["Example deterministic design characterization"],
            ),
            batchSafety=[BatchSafetyEvidence.get_example()],
            currentRepresentation=RepresentationEvaluation.get_example(),
            runInfo=AgentRunInfo.get_example(),
        )

    def to_parameter_tuning_handoff(self) -> ExperimentalTuningHandoff:
        """Return validated integration inputs for Parameter Tuning."""
        if self.status != "done":
            raise ValueError(
                "Experimental Context must be done before creating a tuning handoff"
            )
        plan = self.decision.batchCorrection
        batch_columns = sorted(plan.batchColumns)
        safety = sorted(
            (
                item
                for item in self.batchSafety
                if item.batchColumns == batch_columns
                and item.coefficient in self.decision.coefficientsOfInterest
            ),
            key=lambda item: item.coefficient,
        )
        if plan.action in {"evaluateHarmony", "unsafe"}:
            expected = set(self.decision.coefficientsOfInterest)
            if {item.coefficient for item in safety} != expected:
                raise ValueError(
                    "Experimental Context result lacks exact batch safety evidence"
                )
            if any(item.evidenceId not in plan.evidenceIds for item in safety):
                raise ValueError(
                    "Batch-correction plan does not cite its exact safety evidence"
                )
            if plan.action == "evaluateHarmony" and any(
                item.status != "safe" for item in safety
            ):
                raise ValueError("Harmony plan contains non-safe batch evidence")
            if plan.action == "unsafe" and (
                any(item.status == "notComputed" for item in safety)
                or not any(item.status == "unsafe" for item in safety)
            ):
                raise ValueError("Unsafe plan lacks exact unsafe batch evidence")
        return ExperimentalTuningHandoff(
            cellKey=self.cellKey,
            batchAction=plan.action,
            batchColumns=batch_columns,
            preservationColumns=list(plan.preserveColumns),
            coefficientsOfInterest=list(self.decision.coefficientsOfInterest),
            batchSafety=safety,
            evidenceIds=sorted({*self.decision.evidenceIds, *plan.evidenceIds}),
        )

    def to_biological_handoff(
        self,
        coefficient: str | None = None,
    ) -> ExperimentalBiologyHandoff:
        """Return one explicitly resolved biological coefficient."""
        if self.status != "done":
            raise ValueError(
                "Experimental Context must be done before creating a biology handoff"
            )
        coefficients = list(self.decision.coefficientsOfInterest)
        if coefficient is None:
            if len(coefficients) != 1:
                raise ValueError(
                    "Select one coefficient explicitly for biological interpretation"
                )
            coefficient = coefficients[0]
        if coefficient not in coefficients:
            raise ValueError(f"Unknown coefficient of interest {coefficient!r}")
        records = {
            record.get("name"): record
            for record in self.characterization.coefficients
            if isinstance(record.get("name"), str)
        }
        record = records.get(coefficient)
        if record is None:
            raise ValueError(f"Missing characterization for {coefficient!r}")
        reports = {
            report.get("coefficient"): report
            for report in self.characterization.confounding
            if isinstance(report.get("coefficient"), str)
        }
        report = reports.get(coefficient)
        known_evidence = characterization_evidence(self.characterization)
        relevant_evidence = {
            f"column:{coefficient}",
            f"coefficient:{coefficient}",
            f"estimability:{coefficient}",
            *(
                evidence_id
                for evidence_id in known_evidence
                if evidence_id.startswith(f"confounding:{coefficient}:")
            ),
        }
        for unit_name in (
            record.get("observationUnit"),
            record.get("independentUnit"),
        ):
            if isinstance(unit_name, str):
                relevant_evidence.add(f"column:{unit_name}")
        return ExperimentalBiologyHandoff(
            cellKey=self.cellKey,
            conditionColumn=coefficient,
            observationUnit=record.get("observationUnit"),
            independentUnit=record.get("independentUnit"),
            coefficientScope=str(record.get("scope", "")),
            estimability=dict(report.get("estimability") or {}) if report else {},
            evidenceIds=sorted(relevant_evidence.intersection(known_evidence)),
        )


class ExperimentalContextDependencies(AgentDataModel):
    """Runtime-only state shared by the agent's read-only tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    store: Any = Field(default=None, exclude=True)
    studyContext: str = ""
    cellKey: str = "I"
    directions: dict[str, Any] = Field(default_factory=dict)
    evidenceIds: set[str] = Field(default_factory=set)
    characterization: CovariateCharacterization | None = None
    batchSafety: dict[str, BatchSafetyEvidence] = Field(default_factory=dict)
    currentRepresentation: RepresentationEvaluation = Field(
        default_factory=RepresentationEvaluation.get_blank
    )
    toolCalls: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextDependencies":
        return cls()

    @classmethod
    def get_example(cls) -> "ExperimentalContextDependencies":
        return cls(
            studyContext="Case-control study with samples nested in donors.",
            cellKey="I",
            directions={"columnDomains": {"batch": "technical"}},
        )


def characterization_evidence(
    characterization: CovariateCharacterization,
) -> set[str]:
    """Build stable evidence IDs from one deterministic characterization."""
    evidence_ids = {
        f"column:{record['name']}"
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    for record in characterization.coefficients:
        coefficient = record.get("name")
        if isinstance(coefficient, str):
            evidence_ids.add(f"coefficient:{coefficient}")
    for report in characterization.confounding:
        coefficient = report.get("coefficient")
        if not isinstance(coefficient, str):
            continue
        evidence_ids.add(f"estimability:{coefficient}")
        for pair in report.get("pairs", []):
            technical = pair.get("technical")
            if isinstance(technical, str):
                evidence_ids.add(f"confounding:{coefficient}:{technical}")
    return evidence_ids


async def inspect_cell_covariates(
    ctx: RunContext[ExperimentalContextDependencies],
) -> CovariateEvidence:
    """Inspect cell metadata without making model-driven choices or writing data."""
    characterization = characterize_covariates(
        ctx.deps.store,
        studyContext=ctx.deps.studyContext,
        model=None,
        cellKey=ctx.deps.cellKey,
        directions=ctx.deps.directions,
    )
    ctx.deps.characterization = characterization
    evidence_ids = characterization_evidence(characterization)
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("inspect_cell_covariates")
    return CovariateEvidence(
        characterization=characterization,
        evidenceIds=sorted(evidence_ids),
    )


async def analyze_experimental_design(
    ctx: RunContext[ExperimentalContextDependencies],
    column_domains: dict[str, ColumnDomain],
    coefficients_of_interest: list[str],
    units_of_inference: dict[str, InferenceUnit],
    batch_columns: list[str] | str | None = None,
) -> CovariateEvidence:
    """Validate proposed domains and inference units and compute confounding.

    Args:
        ctx: Pydantic AI run context containing the existing datastore.
        column_domains: Domain assignment for each metadata column under review.
        coefficients_of_interest: Biological columns representing study contrasts.
        units_of_inference: Observation and independent units for each coefficient.
        batch_columns: Exact technical columns proposed for Harmony evaluation.
            A single column may be supplied as either a string or a one-item list.
    """
    directions = dict(ctx.deps.directions)
    directed_domains = dict(column_domains)
    directed_domains.update(dict(directions.get("columnDomains") or {}))
    directions["columnDomains"] = directed_domains
    directed_coefficients = list(
        dict.fromkeys(
            [
                *coefficients_of_interest,
                *(directions.get("coefficientsOfInterest") or []),
            ]
        )
    )
    directions["coefficientsOfInterest"] = directed_coefficients
    directed_units = {
        name: unit.model_dump(exclude_none=True)
        for name, unit in units_of_inference.items()
    }
    directed_units.update(dict(directions.get("unitsOfInference") or {}))
    directions["unitsOfInference"] = directed_units

    characterization = characterize_covariates(
        ctx.deps.store,
        studyContext=ctx.deps.studyContext,
        model=None,
        cellKey=ctx.deps.cellKey,
        directions=directions,
    )
    if characterization.status == "failed":
        raise ModelRetry("; ".join(characterization.notes))

    proposed_batch_columns = (
        [batch_columns] if isinstance(batch_columns, str) else list(batch_columns or [])
    )
    canonical_batch_columns = sorted(set(proposed_batch_columns))
    if len(canonical_batch_columns) != len(proposed_batch_columns):
        raise ModelRetry("Proposed batch columns must be unique")
    column_records = {
        record.get("name"): record
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    for batch_column in canonical_batch_columns:
        record = column_records.get(batch_column)
        if record is None:
            raise ModelRetry(f"Unknown batch column {batch_column!r}")
        if record.get("domain") != "technical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be classified as technical"
            )
        if record.get("kind") != "categorical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be categorical for Harmony"
            )

    coefficient_records = {
        record.get("name"): record
        for record in characterization.coefficients
        if isinstance(record.get("name"), str)
    }
    confounding_reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    batch_safety: list[BatchSafetyEvidence] = []
    for coefficient in directed_coefficients:
        if not canonical_batch_columns:
            break
        coefficient_record = coefficient_records.get(coefficient)
        report = confounding_reports.get(coefficient)
        coefficient_kind = (
            coefficient_record.get("kind") if coefficient_record is not None else None
        )
        if coefficient_kind not in {"categorical", "continuous"}:
            coefficient_kind = None
        observation_unit = (
            report.get("observationUnit")
            if report is not None
            else (
                coefficient_record.get("observationUnit")
                if coefficient_record is not None
                else None
            )
        )
        unit_constant = {
            pair.get("technical")
            for pair in (report.get("pairs", []) if report is not None else [])
            if isinstance(pair.get("technical"), str)
        }
        effective_batch_columns = [
            name for name in canonical_batch_columns if name in unit_constant
        ]
        estimability: dict[str, Any]
        if (
            coefficient_record is None
            or coefficient_record.get("scope") != "betweenUnit"
            or report is None
            or not isinstance(observation_unit, str)
            or coefficient_kind is None
        ):
            estimability = {
                "status": "notComputed",
                "reason": "unresolvedCoefficientDesign",
            }
        else:
            try:
                design = reduce_observation_units(
                    ctx.deps.store.cells,
                    observation_unit,
                    [coefficient, *effective_batch_columns],
                    cell_key=ctx.deps.cellKey,
                )
                estimability = coefficient_estimability(
                    design[coefficient].to_numpy(),
                    coefficientKind=coefficient_kind,
                    technicals={
                        name: design[name].to_numpy()
                        for name in effective_batch_columns
                    },
                    technicalKinds={
                        name: column_records[name]["kind"]
                        for name in effective_batch_columns
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                estimability = {
                    "status": "notComputed",
                    "reason": type(exc).__name__,
                }
        if estimability.get("status") != "ok":
            safety_status: BatchSafetyStatus = "notComputed"
        elif estimability.get("coefficientEstimable") is True and not bool(
            estimability.get("rankDeficient")
        ):
            safety_status = "safe"
        else:
            safety_status = "unsafe"
        batch_token = ",".join(canonical_batch_columns)
        safety = BatchSafetyEvidence(
            coefficient=coefficient,
            coefficientKind=coefficient_kind,
            observationUnit=(
                observation_unit if isinstance(observation_unit, str) else None
            ),
            batchColumns=canonical_batch_columns,
            unitConstantBatchColumns=effective_batch_columns,
            status=safety_status,
            estimability=estimability,
            evidenceId=f"batchEstimability:{coefficient}:{batch_token}",
        )
        batch_safety.append(safety)
        ctx.deps.batchSafety[safety.evidenceId] = safety

    ctx.deps.characterization = characterization
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(item.evidenceId for item in batch_safety)
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("analyze_experimental_design")
    return CovariateEvidence(
        characterization=characterization,
        batchSafety=batch_safety,
        evidenceIds=sorted(evidence_ids),
    )


async def score_current_representation(
    ctx: RunContext[ExperimentalContextDependencies],
    batch_column: str,
    biological_column: str | None = None,
    from_assay: str | None = None,
) -> RepresentationEvaluation:
    """Score the current graph without changing analysis state.

    Args:
        ctx: Pydantic AI run context containing the existing datastore.
        batch_column: Categorical technical column used to assess batch mixing.
        biological_column: Optional biological label used to assess preservation.
        from_assay: Assay whose current graph state should be scored.
    """
    store = ctx.deps.store
    available_columns = set(store.cells.columns)
    if batch_column not in available_columns:
        raise ModelRetry(f"Unknown batch column {batch_column!r}")
    if biological_column is not None and biological_column not in available_columns:
        raise ModelRetry(f"Unknown biological column {biological_column!r}")

    state = store.get_assay_state(from_assay)
    if state is None or state.neighbors is None:
        evaluation = RepresentationEvaluation(
            assay=from_assay,
            cellKey=ctx.deps.cellKey,
            notes=["No current neighbors artifact is available"],
        )
        ctx.deps.currentRepresentation = evaluation
        ctx.deps.toolCalls.append("score_current_representation")
        return evaluation
    if state.cell_key != ctx.deps.cellKey:
        evaluation = RepresentationEvaluation(
            assay=state.assay,
            cellKey=state.cell_key,
            neighborsArtifactId=state.neighbors.artifact_id,
            notes=[
                "Current graph uses a different cell selection: "
                f"{state.cell_key!r} instead of {ctx.deps.cellKey!r}"
            ],
        )
        ctx.deps.currentRepresentation = evaluation
        ctx.deps.toolCalls.append("score_current_representation")
        return evaluation

    metrics: dict[str, float] = {}
    notes: list[str] = []
    evidence_ids: list[str] = []
    neighbor_route = f"assay:{state.assay}:neighbors:{state.neighbors.artifact_id}"
    try:
        value = float(
            store.metric_ilisi(
                batch_column,
                neighbors=state.neighbors,
                from_assay=state.assay,
                cell_key=state.cell_key,
            )
        )
        if math.isfinite(value):
            metrics[f"iLISI:{batch_column}"] = value
            evidence_ids.append(f"metric:iLISI:{batch_column}:{neighbor_route}")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        notes.append(f"iLISI could not be scored: {exc}")
    try:
        value = float(
            store.metric_proportional_batch_mixing(
                batch_column,
                neighbors=state.neighbors,
                from_assay=state.assay,
                cell_key=state.cell_key,
            )
        )
        if math.isfinite(value):
            metrics[f"proportionalBatchMixing:{batch_column}"] = value
            evidence_ids.append(
                f"metric:proportionalBatchMixing:{batch_column}:{neighbor_route}"
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        notes.append(f"Proportional batch mixing could not be scored: {exc}")
    if biological_column is not None:
        try:
            value = float(
                store.metric_clisi(
                    biological_column,
                    neighbors=state.neighbors,
                    from_assay=state.assay,
                    cell_key=state.cell_key,
                )
            )
            if math.isfinite(value):
                metrics[f"cLISI:{biological_column}"] = value
                evidence_ids.append(
                    f"metric:cLISI:{biological_column}:{neighbor_route}"
                )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            notes.append(f"cLISI could not be scored: {exc}")
        if state.connectivity_map is not None:
            try:
                value = float(
                    store.metric_graph_connectivity(
                        biological_column,
                        graph=state.connectivity_map,
                        from_assay=state.assay,
                        cell_key=state.cell_key,
                    )
                )
                if math.isfinite(value):
                    metrics[f"graphConnectivity:{biological_column}"] = value
                    evidence_ids.append(
                        "metric:graphConnectivity:"
                        f"{biological_column}:assay:{state.assay}:connectivity:"
                        f"{state.connectivity_map.artifact_id}"
                    )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                notes.append(f"Graph connectivity could not be scored: {exc}")

    evaluation = RepresentationEvaluation(
        available=bool(metrics),
        assay=state.assay,
        cellKey=state.cell_key,
        neighborsArtifactId=state.neighbors.artifact_id,
        connectivityArtifactId=(
            state.connectivity_map.artifact_id
            if state.connectivity_map is not None
            else None
        ),
        metrics=metrics,
        notes=notes,
        evidenceIds=evidence_ids,
    )
    ctx.deps.currentRepresentation = evaluation
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("score_current_representation")
    return evaluation


def validate_experimental_context(
    decision: ExperimentalContextDecision,
    deps: ExperimentalContextDependencies,
) -> ExperimentalContextDecision:
    """Recompute and validate every model-authored design choice."""
    directions = dict(deps.directions)
    column_domains = dict(decision.columnDomains)
    column_domains.update(dict(directions.get("columnDomains") or {}))
    directions["columnDomains"] = column_domains
    directions["coefficientsOfInterest"] = list(
        dict.fromkeys(
            [
                *decision.coefficientsOfInterest,
                *(directions.get("coefficientsOfInterest") or []),
            ]
        )
    )
    units_of_inference = {
        name: unit.model_dump(exclude_none=True)
        for name, unit in decision.unitsOfInference.items()
    }
    units_of_inference.update(dict(directions.get("unitsOfInference") or {}))
    directions["unitsOfInference"] = units_of_inference

    characterization = characterize_covariates(
        deps.store,
        studyContext=deps.studyContext,
        model=None,
        cellKey=deps.cellKey,
        directions=directions,
    )
    if characterization.status == "failed":
        raise ModelRetry("; ".join(characterization.notes))
    deps.characterization = characterization
    deps.evidenceIds.update(characterization_evidence(characterization))

    if "inspect_cell_covariates" not in deps.toolCalls:
        raise ModelRetry("Call inspect_cell_covariates before returning a decision")
    if "analyze_experimental_design" not in deps.toolCalls:
        raise ModelRetry("Call analyze_experimental_design before returning a decision")

    requested_coefficients = set(directions["coefficientsOfInterest"])
    characterized_coefficients = {
        record.get("name") for record in characterization.coefficients
    }
    missing_coefficients = sorted(
        name
        for name in requested_coefficients
        if name not in characterized_coefficients
    )
    if missing_coefficients:
        raise ModelRetry(
            "Coefficients of interest must be classified as biological: "
            f"{missing_coefficients}"
        )

    coefficient_records = {
        record.get("name"): record
        for record in characterization.coefficients
        if isinstance(record.get("name"), str)
    }
    confounding_reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }

    plan = decision.batchCorrection
    records = {
        record["name"]: record
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    unknown_columns = sorted(set(decision.columnDomains) - set(records))
    if unknown_columns:
        raise ModelRetry(f"Unknown column domain assignments: {unknown_columns}")
    unit_columns = {
        unit_name
        for unit in units_of_inference.values()
        for unit_name in (
            unit.get("observationUnit"),
            unit.get("independentUnit"),
        )
        if isinstance(unit_name, str)
    }
    if plan.action == "evaluateHarmony" and not plan.batchColumns:
        raise ModelRetry("evaluateHarmony requires at least one batch column")
    if plan.action == "unsafe" and not plan.batchColumns:
        raise ModelRetry("unsafe requires the exact batch columns that were assessed")
    if plan.action == "skip" and plan.batchColumns:
        raise ModelRetry("skip must not include batch columns")
    if plan.action == "needsInput" and not decision.needsInput:
        raise ModelRetry("needsInput action requires at least one concrete question")
    if len(set(plan.batchColumns)) != len(plan.batchColumns):
        raise ModelRetry("Batch columns must be unique")

    for batch_column in plan.batchColumns:
        record = records.get(batch_column)
        if record is None:
            raise ModelRetry(f"Unknown batch column {batch_column!r}")
        if record.get("domain") != "technical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be classified as technical"
            )
        if record.get("kind") != "categorical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be categorical for Harmony"
            )
        if batch_column in requested_coefficients or batch_column in unit_columns:
            raise ModelRetry(
                f"Batch column {batch_column!r} cannot be a coefficient or unit of inference"
            )

    if plan.action == "evaluateHarmony":
        mixing_metrics = {"iLISI", "proportionalBatchMixing"}
        preservation_metrics = {"cLISI", "graphConnectivity"}
        if not mixing_metrics.intersection(plan.metricsRequired):
            raise ModelRetry(
                "evaluateHarmony requires iLISI or proportionalBatchMixing"
            )
        if plan.preserveColumns and not preservation_metrics.intersection(
            plan.metricsRequired
        ):
            raise ModelRetry(
                "evaluateHarmony requires cLISI or graphConnectivity for preservation"
            )
        missing_preserve = sorted(requested_coefficients - set(plan.preserveColumns))
        if missing_preserve:
            raise ModelRetry(
                "preserveColumns must include every coefficient of interest: "
                f"{missing_preserve}"
            )
        unresolved_coefficients = sorted(
            coefficient
            for coefficient in requested_coefficients
            if coefficient_records[coefficient].get("scope") != "betweenUnit"
            or coefficient not in confounding_reports
        )
        if unresolved_coefficients:
            raise ModelRetry(
                "evaluateHarmony requires a between-unit coefficient with a "
                "matching estimability report; use needsInput or unsafe for: "
                f"{unresolved_coefficients}"
            )
        for preserve_column in plan.preserveColumns:
            record = records.get(preserve_column)
            if record is None:
                raise ModelRetry(f"Unknown preservation column {preserve_column!r}")
            if record.get("domain") != "biological":
                raise ModelRetry(
                    f"Preservation column {preserve_column!r} must be biological"
                )
            if record.get("kind") != "categorical":
                raise ModelRetry(
                    f"Preservation column {preserve_column!r} must be categorical"
                )

    matched_safety: list[BatchSafetyEvidence] = []
    if plan.action in {"evaluateHarmony", "unsafe"}:
        canonical_batch_columns = sorted(plan.batchColumns)
        for coefficient in sorted(requested_coefficients):
            coefficient_record = coefficient_records[coefficient]
            report = confounding_reports.get(coefficient)
            observation_unit = (
                report.get("observationUnit")
                if report is not None
                else coefficient_record.get("observationUnit")
            )
            unit_constant = {
                pair.get("technical")
                for pair in (report.get("pairs", []) if report is not None else [])
                if isinstance(pair.get("technical"), str)
            }
            expected_effective = [
                name for name in canonical_batch_columns if name in unit_constant
            ]
            candidates = [
                item
                for item in deps.batchSafety.values()
                if item.coefficient == coefficient
                and item.coefficientKind == coefficient_record.get("kind")
                and item.observationUnit == observation_unit
                and item.batchColumns == canonical_batch_columns
                and item.unitConstantBatchColumns == expected_effective
            ]
            if len(candidates) != 1:
                raise ModelRetry(
                    "Call analyze_experimental_design with the exact proposed batch "
                    f"columns before returning a recommendation for {coefficient!r}"
                )
            matched_safety.append(candidates[0])
        missing_safety_evidence = sorted(
            item.evidenceId
            for item in matched_safety
            if item.evidenceId not in plan.evidenceIds
        )
        if missing_safety_evidence:
            raise ModelRetry(
                "Batch-correction recommendations must cite exact batch "
                f"estimability evidence: {missing_safety_evidence}"
            )
        not_computed = [
            item.coefficient for item in matched_safety if item.status == "notComputed"
        ]
        if not_computed:
            raise ModelRetry(
                "Batch estimability could not be computed; use action='needsInput' "
                f"for: {sorted(not_computed)}"
            )
        unsafe_coefficients = [
            item.coefficient for item in matched_safety if item.status == "unsafe"
        ]
        if plan.action == "evaluateHarmony" and unsafe_coefficients:
            raise ModelRetry(
                "Batch correction is unsafe because the biological coefficient is "
                "not estimable after the exact proposed batch columns; use "
                f"action='unsafe' for: {sorted(unsafe_coefficients)}"
            )
        if plan.action == "unsafe" and not unsafe_coefficients:
            raise ModelRetry(
                "The exact proposed batch columns were estimable for every "
                "coefficient; use action='evaluateHarmony' or 'skip'"
            )

    cited_ids = [*decision.evidenceIds, *plan.evidenceIds]
    unknown_evidence = sorted(set(cited_ids) - deps.evidenceIds)
    if unknown_evidence:
        raise ModelRetry(f"Unknown evidence IDs: {unknown_evidence}")
    if plan.action in {"evaluateHarmony", "skip", "unsafe"} and not plan.evidenceIds:
        raise ModelRetry("Batch-correction recommendations require evidence IDs")
    current_metric_evidence = set(deps.currentRepresentation.evidenceIds)
    stale_metric_evidence = sorted(
        evidence_id
        for evidence_id in cited_ids
        if evidence_id.startswith("metric:")
        and evidence_id not in current_metric_evidence
    )
    if stale_metric_evidence:
        raise ModelRetry(
            "Metric evidence must come from the returned exact representation: "
            f"{stale_metric_evidence}"
        )

    canonical_domains = {
        name: records[name]["domain"]
        for name in column_domains
        if name in records
        and records[name].get("domain")
        in {
            "biological",
            "technical",
            "design",
            "ignore",
            "unknown",
        }
    }
    canonical_units = {
        coefficient: InferenceUnit(
            observationUnit=coefficient_records[coefficient].get("observationUnit"),
            independentUnit=coefficient_records[coefficient].get("independentUnit"),
        )
        for coefficient in directions["coefficientsOfInterest"]
        if coefficient in coefficient_records
    }
    return decision.model_copy(
        update={
            "columnDomains": canonical_domains,
            "coefficientsOfInterest": list(directions["coefficientsOfInterest"]),
            "unitsOfInference": canonical_units,
        }
    )


class ExperimentalContextAgent:
    """A narrow agent for study design and batch-correction planning."""

    def __init__(
        self,
        model: Any,
        *,
        config: AgentRunConfig | None = None,
    ) -> None:
        self.model = model
        self.config = (config or AgentRunConfig()).with_limits(
            request_limit=6,
            tool_call_limit=3,
            output_token_limit=32768,
            timeout_seconds=600.0,
        )
        self.system_prompt = dedent(
            """
            You are Scarf's Experimental Context Agent. Work only through the
            provided read-only tools and return the structured decision schema.

            Call inspect_cell_covariates exactly once. Then call
            analyze_experimental_design exactly once with all explicit domains,
            all biological coefficients, every unit of inference, and the complete
            exact batch-column set being considered. You may call
            score_current_representation at most once when a current graph can add
            evidence. Do not split metadata, coefficients, or batch columns across
            calls, and do not repeat a tool call. Pass batch_columns as a JSON array,
            including when the array contains exactly one column.

            A batch column must be categorical and technical. Never use donor,
            sample, observation-unit, independent-unit, biological, cluster, or
            embedding columns as Harmony batch columns. A biological coefficient
            that is not estimable with the exact proposed batch columns makes
            correction unsafe.
            LISI evaluates a representation; it does not identify which metadata
            column is a batch. Recommend evaluateHarmony, not application, because
            Parameter Tuning must compare exact uncorrected and corrected artifacts.

            Cite only evidenceIds returned by tools. Ask for input when study
            design cannot be resolved. Never propose Python, shell commands,
            direct Zarr access, or any datastore mutation. Return only fields
            defined by the structured output schema.
            """
        ).strip()

    def run(
        self,
        store: Any,
        *,
        study_context: str | None = None,
        cell_key: str = "I",
        directions: Mapping[str, Any] | None = None,
    ) -> ExperimentalContextResult:
        """Inspect one datastore and return a validated experimental-context report."""
        study_context = (study_context or "").strip()
        if len(study_context) > _CONTEXT_LIMIT:
            study_context = study_context[: _CONTEXT_LIMIT - 3] + "..."
        direction_map = dict(directions or {})
        deps = ExperimentalContextDependencies(
            store=store,
            studyContext=study_context,
            cellKey=cell_key,
            directions=direction_map,
        )
        user_prompt = (
            dedent(
                """
                Characterize this experiment's metadata and decide whether Harmony
                should be evaluated.

                Study context: {study_context}
                Active cell selection: {cell_key}
                Caller directions: {directions}
                """
            )
            .strip()
            .format(
                study_context=study_context or "not provided",
                cell_key=cell_key,
                directions=json.dumps(direction_map, sort_keys=True, default=str),
            )
        )
        execution = run_agent_sync(
            model=self.model,
            output_type=ExperimentalContextDecision,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            tools=(
                inspect_cell_covariates,
                analyze_experimental_design,
                score_current_representation,
            ),
            deps_type=ExperimentalContextDependencies,
            deps=deps,
            config=self.config,
            name="experimental_context",
            output_validator=lambda decision: validate_experimental_context(
                decision,
                deps,
            ),
        )
        decision = ExperimentalContextDecision.model_validate(execution.output)
        decision = validate_experimental_context(decision, deps)
        characterization = deps.characterization
        if characterization is None:
            characterization = characterize_covariates(
                store,
                studyContext=study_context,
                model=None,
                cellKey=cell_key,
                directions=direction_map,
            )
        if characterization.status == "failed":
            status: StageStatus = "failed"
        elif decision.needsInput or decision.batchCorrection.action == "needsInput":
            status = "needsInput"
        else:
            status = "done"
        return ExperimentalContextResult(
            status=status,
            decision=decision,
            characterization=characterization,
            cellKey=cell_key,
            batchSafety=list(deps.batchSafety.values()),
            currentRepresentation=deps.currentRepresentation,
            notes=[*characterization.notes, *decision.needsInput],
            runInfo=execution.runInfo,
        )
