"""Objective-led RNA experiments on frozen screening and full-cohort cells."""

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import Field, create_model, model_validator
from pydantic.json_schema import SkipJsonSchema

from ...metadata.rows import read_metadata_rows_chunkwise
from ...storage.refs import ArtifactRef
from ...storage.selections import (
    read_stored_selection_indices,
    resolve_generated_selection_artifact,
)
from ...utils.logging import logger
from .. import record_io
from ..config.agent_exec import (
    ImageEvidence,
    ImageInputUnsupportedError,
    build_visual_evidence_prompt,
    run_agent_sync,
)
from ..experimental_context.contracts import CovariateComparison
from ..experimental_context.study import (
    StudyContract,
    unsupported_comparison_limitations,
)
from ..parameter_tuning.agent import prepare_parameter_tuning_dependencies
from ..parameter_tuning.comparisons import (
    CombinedSettings,
    ComparisonConclusion,
    ComparisonTradeoff,
    PopulationConcern,
    bind_comparison_measurements,
    comparison_advantages,
    partition_comparison_evidence,
    setting_changes,
    validate_comparison_review,
)
from ..parameter_tuning.contracts import (
    ArtifactRecord,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
)
from ..parameter_tuning.diagnostics import (
    SCARF_DEFAULT_DIAGNOSTIC_FAMILIES,
    _family_mask,
    _neighbor_overlap,
    augment_cluster_evaluations,
    augment_pca_evaluations,
    population_support_evidence,
    restore_advisory_doublets,
    score_advisory_doublets,
)
from ..parameter_tuning.execution import (
    diagnostic_call,
    diagnostic_reuse,
    execute_parameter_candidate,
    refresh_candidate_design_evidence,
)
from ..parameter_tuning.hvg import (
    HvgGroupVariability,
    aggregate_hvg_rankings,
    rank_core_hvgs,
)
from ..parameter_tuning.selection import (
    finalize_parameter_tuning_selection,
    harmony_acceptance_gate,
)
from ..types import AgentDataModel, ArtifactReferenceModel
from . import journal
from .budget import CandidateBudget, CandidateBudgetExceeded, candidate_identity
from .models import (
    AutomatedPreprocessingPlan,
    OrchestrationRequestRecord,
    PreprocessedAssayHandoff,
    artifact_model_to_ref,
)

_STRUCTURED_VISUAL_LIMITATION = (
    "The model assessed structured marker, PCA loading and diagnostic evidence; "
    "visual inspection was unavailable because the configured model does not accept images."
)


def _configured_image_input(model: Any) -> bool | None:
    """Honor an explicit input capability without inferring it from image output."""
    declared = getattr(model, "supports_image_input", None)
    if isinstance(declared, bool):
        return declared
    profile = getattr(model, "profile", None)
    if isinstance(profile, Mapping):
        declared = profile.get("supports_image_input")
        if isinstance(declared, bool):
            return declared
    return None


def screening_sizes(n_cells: int, config: Any) -> tuple[int, ...]:
    """Resolve the saved fixed policy or the bounded ten-percent default."""
    if n_cells < 1:
        raise ValueError("Screening requires a non-empty retained cohort")
    if config.screeningCells is None:
        maximum = min(n_cells, config.maxScreeningCells, 100_000)
        initial = min(maximum, max(10_000, (n_cells + 9) // 10))
    else:
        initial = min(n_cells, config.screeningCells)
        maximum = min(n_cells, config.maxScreeningCells)
    return tuple(dict.fromkeys((initial, maximum)))


class TuningAction(AgentDataModel):
    """An assessment of observed evidence and at most one registered experiment."""

    action: Literal["accept", "combine", "experiment", "enlarge", "defer"] = Field(
        description=(
            "Accept supported observed evidence, request one next experiment, "
            "enlarge a screening sample, or defer an unresolved essential question."
        )
    )
    selectedCandidateId: str = Field(
        description=(
            "Copy an observed candidate ID. An experiment must keep "
            "currentCandidateId as its fixed baseline."
        )
    )
    experimentId: str | None = Field(
        default=None,
        description=(
            "For action=experiment, copy one exact key from experiments. "
            "This requests the next operation; it has not yet run. Otherwise null."
        ),
    )
    correctionNeed: Literal["needed", "notNeeded", "uncertain", "notApplicable"]
    evidenceIds: list[str] = Field(
        min_length=1,
        description=(
            "Copy exact IDs from availableEvidenceIds, including the selected "
            "candidate's anchor or one of its supplied diagnostic evidence IDs."
        ),
    )
    quantitativeFindings: list[str] = Field(
        min_length=1,
        description="Describe supplied observed measurements, not predicted results.",
    )
    qualitativeFindings: list[str] = Field(min_length=1)
    comparisonConclusions: list[ComparisonConclusion]
    populationConcerns: list[PopulationConcern] = Field(default_factory=list)
    plainLanguageSummary: str = Field(min_length=1)
    combinedSettings: CombinedSettings | None = None
    concern: str = ""
    expectedImprovement: str = Field(
        default="",
        description="Predict what the requested experiment should improve and why.",
    )
    objectivePreservation: str = Field(min_length=1)
    rationale: str = Field(
        min_length=1,
        description=(
            "Explain why this action follows from observed evidence. "
            "Do not describe an unexecuted experiment as a completed result."
        ),
    )

    @model_validator(mode="after")
    def validate_action(self) -> "TuningAction":
        if (self.action == "combine") != (self.combinedSettings is not None):
            raise ValueError("Only a combine action proposes combined settings")
        if (self.action == "experiment") != (self.experimentId is not None):
            raise ValueError("Only an experiment action names an experiment")
        if self.action == "experiment" and (
            not self.concern.strip() or not self.expectedImprovement.strip()
        ):
            raise ValueError(
                "An experiment needs an observed concern and expected improvement"
            )
        return self


def _assessment_output_type(
    candidate_ids: Sequence[str],
    experiment_ids: Sequence[str],
    *,
    scope: str,
    phase: str = "combined",
) -> type[TuningAction]:
    """Constrain new model choices without changing the saved action contract."""
    if not candidate_ids:
        raise ValueError("RNA assessment requires observed candidates")
    tradeoff_type = create_model(
        "ObservedTradeoffInterpretation",
        __base__=ComparisonTradeoff,
        preferredValue=(Annotated[float | None, SkipJsonSchema()], None),
        alternativeValue=(Annotated[float | None, SkipJsonSchema()], None),
    )
    conclusion_type = create_model(
        "ObservedComparisonInterpretation",
        __base__=ComparisonConclusion,
        tradeoffs=(
            cast(Any, list)[tradeoff_type],
            Field(
                description=(
                    "Required array: for each comparisonAdvantages row matching this "
                    "axis and preferredCandidateId, supply alternativeCandidateId, "
                    "metric and your interpretation. Explanations in quantitativeReason "
                    "or biologicalReason do not replace these entries. For combine or "
                    "accept, use [] only when this preference has no measured "
                    "alternative advantages. Include only matching inventory rows: "
                    "ties and measurements favoring your preference belong in "
                    "quantitativeReason, not tradeoffs. Scarf attaches the measured values."
                )
            ),
        ),
    )
    actions = tuple(
        action
        for action in ("accept", "combine", "experiment", "enlarge", "defer")
        if (action != "enlarge" or scope != "full")
        and (action != "experiment" or experiment_ids)
        and (action != "combine" or phase == "sensitivity")
        and (action != "accept" or phase != "sensitivity")
    )
    return create_model(
        "ObservedRnaAssessment",
        __base__=TuningAction,
        comparisonConclusions=(cast(Any, list)[conclusion_type], ...),
        action=(
            Literal[actions],
            Field(description=TuningAction.model_fields["action"].description),
        ),
        selectedCandidateId=(
            Literal[tuple(dict.fromkeys(candidate_ids))],
            Field(
                description=TuningAction.model_fields["selectedCandidateId"].description
            ),
        ),
        experimentId=(
            Literal[tuple(dict.fromkeys(experiment_ids))] | None
            if experiment_ids
            else type(None),
            Field(
                default=None,
                description=TuningAction.model_fields["experimentId"].description,
            ),
        ),
    )


class RnaSetting(AgentDataModel):
    parameters: ParameterCandidate
    features: ArtifactReferenceModel
    eligibleFeatures: ArtifactReferenceModel
    hvgCount: int = 1000
    ranking: Literal["global", "batchAware"] = "global"
    rankingColumn: str | None = None


def validate_completed_comparison_evidence(review: Mapping[str, Any]) -> None:
    """Check exact completed review coverage before finalization or report reuse."""
    if not {
        "comparisonCoverage",
        "comparisonConclusions",
        "plainLanguageSummary",
    }.issubset(review):
        raise ValueError(
            "Saved analysis lacks mandatory comparison evidence; start a new workflow"
        )
    action = TuningAction.model_validate(
        {
            key: value
            for key, value in review.items()
            if key in TuningAction.model_fields
        }
    )
    coverage = review.get("comparisonCoverage")
    if not isinstance(coverage, Mapping):
        raise ValueError(
            "Saved analysis lacks mandatory comparison evidence; start a new workflow"
        )
    validate_comparison_review(coverage, action.model_dump(mode="json"))
    for candidate in review.get("candidates", []):
        setting = coverage["candidateSettings"].get(candidate["candidateId"])
        if (
            setting is None
            or any(
                setting[key] != candidate[key]
                for key in ("parameters", "cellSelection")
            )
            or setting["features"]
            != candidate.get("artifacts", {}).get("graphFeatures")
        ):
            raise ValueError(
                "Comparison evidence differs from the exact reviewed candidate"
            )
        if any(
            candidate["metrics"].get(key) != value
            for key, value in setting.get("metrics", {}).items()
        ):
            raise ValueError(
                "Comparison measurements differ from the reviewed candidate"
            )


def uniform_screening_selection(
    store: Any, parent: ArtifactRef, *, size: int, seed: int
) -> ArtifactRef:
    """Select nested uniform prefixes without weighting study groups differently."""
    indices = read_stored_selection_indices(
        store.zw,
        parent,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    if size >= len(indices):
        return parent
    if size < 3:
        raise ValueError("Screening requires at least three cells")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(indices))
    mask = np.zeros(store.cells.N, dtype=bool)
    mask[indices[order[:size]]] = True
    selection, _ = resolve_generated_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=mask,
        row_ids=np.asarray(store.cells.fetch_all("ids")),
        operation="agent_uniform_rna_screen",
        parameters={"size": size, "seed": seed, "generator": "PCG64"},
        inputs={"parent_selection": parent},
        source_column="agent_screening",
    )
    return selection


def screening_coverage(
    store: Any,
    parent: ArtifactRef,
    sample: ArtifactRef,
    columns: Sequence[str],
    combinations: Sequence[Sequence[str]] = (),
) -> tuple[dict[str, Any], list[str]]:
    """Report population and sample proportions without oversampling rare groups."""

    def indices(selection: ArtifactRef) -> np.ndarray:
        return read_stored_selection_indices(
            store.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)

    full_indices, sample_indices = indices(parent), indices(sample)
    evidence: dict[str, Any] = {
        "populationCells": len(full_indices),
        "screeningCells": len(sample_indices),
        "sampling": "Uniform without replacement; no group oversampling or weights.",
        "groups": {},
    }
    concerns: list[str] = []
    grouped = {}
    for column in dict.fromkeys(columns):
        full = np.asarray(
            read_metadata_rows_chunkwise(store.cells, column, full_indices)
        ).astype(str)
        sampled = np.asarray(
            read_metadata_rows_chunkwise(store.cells, column, sample_indices)
        ).astype(str)
        grouped[column] = (full, sampled)
    if combinations:
        from ..experimental_context.characterization import _SelectionBoundCells
        from ..experimental_context.comparisons import combination_labels

        full_cells = _SelectionBoundCells(store.zw, store.cells, parent)
        sample_cells = _SelectionBoundCells(store.zw, store.cells, sample)
        for group_columns in combinations:
            key = "joint:" + json.dumps(list(group_columns), separators=(",", ":"))
            grouped[key] = (
                combination_labels(full_cells, group_columns),
                combination_labels(sample_cells, group_columns),
            )
    for column, (full, sampled) in grouped.items():
        values, counts = np.unique(full, return_counts=True)
        sample_values, sample_counts = np.unique(sampled, return_counts=True)
        lookup = dict(zip(sample_values, sample_counts, strict=True))
        rows = []
        for value, count in zip(values, counts, strict=True):
            observed = int(lookup.get(value, 0))
            rows.append(
                {
                    "value": str(value),
                    "populationCells": int(count),
                    "screeningCells": observed,
                    "populationFraction": float(count / len(full)),
                    "screeningFraction": observed / len(sampled),
                }
            )
            if observed < min(int(count), 20):
                concerns.append(
                    f"{column}={value}: {observed} sampled of {int(count)} cells"
                )
        evidence["groups"][column] = rows
    return evidence, concerns


class RnaTuningRun:
    """Execute evidence-requested comparisons with one authoritative history."""

    def __init__(
        self,
        owner: Any,
        store: Any,
        workflow: Any,
        request: OrchestrationRequestRecord,
        plan: AutomatedPreprocessingPlan,
        handoff: PreprocessedAssayHandoff,
        study: StudyContract,
        answers: Mapping[str, Any],
        provenance: dict[str, Any],
        *,
        design_comparisons: Sequence[CovariateComparison] = (),
        previous_provenances: Sequence[dict[str, Any]] = (),
    ) -> None:
        if (
            handoff.cellSelection is None
            or handoff.graphFeatures is None
            or handoff.markerFeatures is None
        ):
            raise ValueError(
                "RNA tuning requires exact cells, graph genes and marker genes"
            )
        self.owner, self.store, self.workflow = owner, store, workflow
        self.request, self.plan, self.handoff, self.study = (
            request,
            plan,
            handoff,
            study,
        )
        self.answers, self.provenance = answers, provenance
        self.previous_provenances = tuple(previous_provenances)
        self.evidence_revision = (
            hashlib.sha256(record_io.canonical_json_bytes(provenance)).hexdigest()
            if previous_provenances
            else None
        )
        self.design_comparisons = tuple(design_comparisons)
        self.prefix = journal._ensure_orchestration_store(store)
        self.cells = artifact_model_to_ref(handoff.cellSelection)
        self.marker_features = artifact_model_to_ref(handoff.markerFeatures)
        self.budget = CandidateBudget(
            store,
            self.prefix,
            workflow.workflowRunId,
            request.config,
            provenance,
            previous_provenances=self.previous_provenances,
        )
        self.settings: dict[str, RnaSetting] = {}
        self.history: list[dict[str, Any]] = []
        self.evaluations: dict[str, list[ParameterCandidateEvaluation]] = {
            "sample0": [],
            "sample1": [],
            "full": [],
        }
        self.last_action: TuningAction | None = None
        self.full_repairs = len(self.budget.repairs)
        self.answer_consumed = False
        self.scope_sizes: dict[str, int] = {}
        self.feature_evidence_cache: dict[str, dict[str, Any]] = {}
        self.neighbor_comparisons: dict[tuple[str, str], float] = {}
        self.diagnostic_counts: dict[str, dict[str, int]] = {}
        self.comparison_rows: dict[str, list[dict[str, Any]]] = {}
        self.combined_candidates: dict[str, str] = {}
        self.resolution_candidates: dict[str, list[str]] = {}
        self.validation_sources: dict[str, dict[str, Any]] = {}
        self.discovery_scope: str | None = None
        self.recovery_scope: str | None = None
        self.full_repair: dict[str, Any] | None = None
        self.batch_columns = list(study.technicalBatchColumns)
        self.coverage_columns = [
            value
            for value in dict.fromkeys(
                [
                    study.physicalCaptureColumn,
                    *study.independentUnitColumns,
                    *study.conditionColumns,
                    *study.technicalBatchColumns,
                    *study.protectedColumns,
                ]
            )
            if value is not None and study.columnKinds.get(value) != "continuous"
        ]
        self.family_patterns = {
            str(row["family"]): str(row["pattern"])
            for row in (
                plan.assays[0].featureParameters.get("defaultFeatureInventory") or {}
            ).get("families", [])
            if isinstance(row, Mapping) and "family" in row and "pattern" in row
        }

    def baseline(self, resolution: float = 1.0) -> RnaSetting:
        assert self.handoff.graphFeatures is not None
        return RnaSetting(
            parameters=ParameterCandidate(
                candidateId="baseline",
                dimensions=min(21, self.handoff.nFeatures - 1, self.handoff.nCells - 1),
                neighborsK=min(11, self.handoff.nCells - 1),
                leidenResolution=resolution,
                useHarmony=False,
            ),
            features=self.handoff.graphFeatures,
            eligibleFeatures=self.handoff.graphFeatureCandidates["eligibleDefault"],
            hvgCount=self.handoff.nFeatures,
        )

    def checkpoint_scope(self, scope: str) -> str:
        """Separate revised interpretations while retaining numerical admissions."""
        if self.evidence_revision is not None:
            return (
                f"parameter_tuning/evidence_revisions/{self.evidence_revision}/{scope}"
            )
        return f"parameter_tuning/{scope}"

    def execution_inputs(
        self, cells: ArtifactRef, setting: RnaSetting
    ) -> dict[str, Any]:
        inputs: dict[str, Any] = {
            "cells": cells.to_dict(),
            "features": setting.features.model_dump(mode="json"),
            "parameters": setting.parameters.model_dump(mode="json"),
        }
        if setting.parameters.useHarmony and any(
            previous.get("studyContract", {}).get("technicalBatchColumns")
            != self.batch_columns
            for previous in self.previous_provenances
        ):
            inputs["harmonyBatchColumns"] = self.batch_columns
        return inputs

    def execute(
        self, scope: str, cells: ArtifactRef, setting: RnaSetting
    ) -> ParameterCandidateEvaluation:
        inputs = self.execution_inputs(cells, setting)
        identity = candidate_identity(inputs)
        parameters = setting.parameters.model_copy(
            update={"candidateId": f"rna_{identity[:24]}"}
        )
        setting = setting.model_copy(update={"parameters": parameters})
        self.settings[parameters.candidateId] = setting
        saved: dict[str, Any] | None
        source = self.budget.completed_source(inputs) if scope == "full" else None
        if source is not None:
            admission, saved = source
            if admission["scope"] != "full":
                self.validation_sources[parameters.candidateId] = {
                    "scope": admission["scope"],
                    "slot": admission["slot"],
                    "identity": admission["identity"],
                }
        else:
            admission = self.budget.admit(scope, inputs)
            saved = self.budget.completed(admission)
        if saved is not None:
            evaluation = ParameterCandidateEvaluation.model_validate(
                saved["evaluation"]
            )
            for artifact in evaluation.artifacts.values():
                status = self.store.inspect_artifact(artifact_model_to_ref(artifact))
                if not status.exists or not status.complete:
                    raise ValueError(
                        "Saved candidate evidence is unavailable or incomplete"
                    )
            diagnostic_reuse("diagnostic.primaryCandidateEvidence", "restored")
            evaluation = self._reassess_saved_evaluation(
                scope, setting, admission, evaluation
            )
        else:
            features = artifact_model_to_ref(setting.features)
            normalized = diagnostic_call(
                "core.primaryNormalization",
                self.store.run_normalization,
                cells,
                features=features,
                invalidate_cache=False,
            )
            deps, ids = prepare_parameter_tuning_dependencies(
                self.store,
                normalized=normalized,
                candidates=[parameters],
                batch_columns=self.batch_columns,
                preservation_columns=self.study.protectedColumns,
                pair_harmony_candidates=False,
                max_candidates=1,
                min_cluster_cells=1,
            )
            deps.protectedCombinations = tuple(
                tuple(columns) for columns in self.study.protectedCombinations
            )
            deps.columnKinds = self.study.columnKinds
            evaluation = execute_parameter_candidate(deps, ids[0])
            if evaluation.status != "done":
                raise RuntimeError(
                    evaluation.error
                    or "Candidate execution failed; its admitted work can be retried on resume"
                )
            evaluation.artifacts.update(
                {
                    "normalized": ArtifactRecord.from_ref(normalized),
                    "graphFeatures": ArtifactRecord.from_ref(features),
                }
            )
            evaluation = self._augment_evaluation(scope, setting, evaluation)
            evaluation = ParameterCandidateEvaluation.model_validate_json(
                record_io.canonical_json_bytes(evaluation.model_dump(mode="json"))
            )
            self.budget.complete(
                admission, {"evaluation": evaluation.model_dump(mode="json")}
            )
        if evaluation.candidateId not in {
            item.candidateId for item in self.evaluations[scope]
        }:
            self.evaluations[scope].append(evaluation)
        return evaluation

    def _augment_evaluation(
        self,
        scope: str,
        setting: RnaSetting,
        evaluation: ParameterCandidateEvaluation,
        *,
        preserve_doublets: bool = False,
    ) -> ParameterCandidateEvaluation:
        """Attach required loading, population, stability and doublet evidence."""
        features = artifact_model_to_ref(setting.features)
        parameters = setting.parameters
        evaluation = augment_pca_evaluations(
            self.store,
            [evaluation],
            feature_selection=features,
            nominated_families=SCARF_DEFAULT_DIAGNOSTIC_FAMILIES,
            protected_families=self.plan.assays[0].featureParameters.get(
                "protectFamilies", []
            ),
            technical_columns=self.batch_columns,
            batch_columns=self.batch_columns,
            protected_columns=self.study.protectedColumns,
            qc_columns=[
                *self.plan.cellQc.attributes,
                *(source.name for source in self.plan.cellQc.artifactMetrics),
            ],
            qc_artifacts={
                source.name: artifact_model_to_ref(source.artifact)
                for source in self.plan.cellQc.artifactMetrics
            },
            column_kinds={
                **self.study.columnKinds,
                **{
                    source.name: "continuous"
                    for source in self.plan.cellQc.artifactMetrics
                },
            },
        )[0]
        native = next(
            (
                item
                for item in self.evaluations[scope]
                if not item.parameters.useHarmony
                and self.settings[item.candidateId].features == setting.features
                and item.parameters.model_dump(exclude={"candidateId", "useHarmony"})
                == parameters.model_dump(exclude={"candidateId", "useHarmony"})
            ),
            None,
        )
        doublets = None
        if self.request.config.scoreDoublets or (
            self.study.correctionLicense == "safe" and self.batch_columns
        ):
            doublets = (
                restore_advisory_doublets(
                    evaluation, capture_column=self.study.physicalCaptureColumn
                )
                if preserve_doublets and "doubletNativeGraph" in evaluation.artifacts
                else score_advisory_doublets(
                    self.store,
                    native or evaluation,
                    [
                        item
                        for item in [*self.evaluations[scope], evaluation]
                        if item.artifacts.get("graphFeatures")
                        == evaluation.artifacts["graphFeatures"]
                        and item.cellSelection == evaluation.cellSelection
                    ],
                    assay=self.handoff.assay,
                    feature_selection=features,
                    capture_column=self.study.physicalCaptureColumn,
                )
            )
        else:
            limitation = (
                f"Advisory doublet scoring was not run for assay {self.handoff.assay!r} "
                "because score_doublets=False. Doublet contamination was not assessed."
            )
            evaluation = evaluation.model_copy(
                update={
                    "warnings": list(dict.fromkeys([*evaluation.warnings, limitation]))
                }
            )
            logger.info(limitation)
        evaluation = augment_cluster_evaluations(
            self.store,
            [evaluation],
            marker_assay=self.handoff.assay,
            marker_features=self.marker_features,
            independent_unit_columns=self.study.independentUnitColumns,
            technical_columns=self.batch_columns,
            nominated_families=SCARF_DEFAULT_DIAGNOSTIC_FAMILIES,
            protected_families=self.plan.assays[0].featureParameters.get(
                "protectFamilies", []
            ),
            doublet_evidence=doublets,
        )[0]
        return evaluation

    def _reassess_saved_evaluation(
        self,
        scope: str,
        setting: RnaSetting,
        admission: dict[str, Any],
        evaluation: ParameterCandidateEvaluation,
    ) -> ParameterCandidateEvaluation:
        original = self.budget.admission_provenance(admission)
        if original == self.provenance:
            return evaluation
        key = f"{self.checkpoint_scope(scope)}/evaluation{admission['slot']}/augmented"
        inputs = {"admission": admission, "provenance": self.provenance}
        saved = journal.load_checkpoint(
            self.store, self.prefix, self.workflow.workflowRunId, key, inputs
        )
        if saved is not None:
            restored = ParameterCandidateEvaluation.model_validate(saved["evaluation"])
            for artifact in restored.artifacts.values():
                status = self.store.inspect_artifact(artifact_model_to_ref(artifact))
                if not status.exists or not status.complete:
                    raise ValueError(
                        "Revised candidate evidence is unavailable or incomplete"
                    )
            return restored
        old = StudyContract.model_validate(original["studyContract"])
        fields = (
            "technicalBatchColumns",
            "protectedColumns",
            "protectedCombinations",
            "columnKinds",
        )
        if any(getattr(old, field) != getattr(self.study, field) for field in fields):
            normalized = artifact_model_to_ref(evaluation.artifacts["normalized"])
            deps, _ = prepare_parameter_tuning_dependencies(
                self.store,
                normalized=normalized,
                candidates=[evaluation.parameters],
                batch_columns=self.batch_columns,
                preservation_columns=self.study.protectedColumns,
                pair_harmony_candidates=False,
                max_candidates=1,
                min_cluster_cells=1,
            )
            deps.protectedCombinations = tuple(
                tuple(columns) for columns in self.study.protectedCombinations
            )
            deps.columnKinds = self.study.columnKinds
            evaluation = refresh_candidate_design_evidence(deps, evaluation)
        if any(
            getattr(old, field) != getattr(self.study, field)
            for field in (
                "technicalBatchColumns",
                "protectedColumns",
                "columnKinds",
                "independentUnitColumns",
                "physicalCaptureColumn",
            )
        ):
            evaluation = self._augment_evaluation(
                scope,
                setting,
                evaluation,
                preserve_doublets=old.physicalCaptureColumn
                == self.study.physicalCaptureColumn,
            )
        evaluation = ParameterCandidateEvaluation.model_validate_json(
            record_io.canonical_json_bytes(evaluation.model_dump(mode="json"))
        )
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            key,
            inputs,
            {
                "evaluation": evaluation.model_dump(mode="json"),
                "primaryArtifactsReused": True,
                "priorProvenance": original,
            },
        )
        return evaluation

    def execute_matched(
        self, scope: str, cells: ArtifactRef, setting: RnaSetting
    ) -> ParameterCandidateEvaluation:
        if setting.parameters.useHarmony:
            if self.study.correctionLicense != "safe" or not self.batch_columns:
                raise ValueError(
                    "Harmony requires a safe design license and approved batch columns"
                )
            native = setting.model_copy(
                update={
                    "parameters": setting.parameters.model_copy(
                        update={"useHarmony": False}
                    )
                }
            )
            self.budget.admit_many(
                scope,
                [
                    value
                    for value in (
                        self.execution_inputs(cells, native),
                        self.execution_inputs(cells, setting),
                    )
                    if scope != "full" or self.budget.completed_source(value) is None
                ],
            )
            self.execute(scope, cells, native)
        return self.execute(scope, cells, setting)

    def harmony_gate(
        self, scope: str, selected: ParameterCandidateEvaluation
    ) -> tuple[bool, list[str]]:
        if not selected.parameters.useHarmony:
            return True, []
        setting = self.settings[selected.candidateId]
        native = next(
            (
                item
                for item in self.evaluations[scope]
                if not item.parameters.useHarmony
                and self.settings[item.candidateId].features == setting.features
                and item.parameters.model_dump(exclude={"candidateId", "useHarmony"})
                == selected.parameters.model_dump(exclude={"candidateId", "useHarmony"})
            ),
            None,
        )
        return harmony_acceptance_gate(
            native,
            selected,
            batch_columns=self.batch_columns,
            protected_columns=[
                *self.study.protectedColumns,
                *(
                    "joint:" + json.dumps(columns, separators=(",", ":"))
                    for columns in self.study.protectedCombinations
                ),
            ],
            independent_unit_columns=self.study.independentUnitColumns,
            require_doublet_evidence=True,
        )

    def experiments(
        self, selected: ParameterCandidateEvaluation
    ) -> dict[str, dict[str, Any]]:
        """Offer individual interventions, without executing their numerical work."""
        setting = self.settings[selected.candidateId]
        n_cells = next(
            (
                self.scope_sizes[name]
                for name, rows in self.evaluations.items()
                if selected in rows and name in self.scope_sizes
            ),
            self.handoff.nCells,
        )
        options: dict[str, dict[str, Any]] = {}
        for field, values in (
            ("dimensions", (10, 21, 30, 50)),
            ("neighborsK", (11, 21, 41)),
            ("leidenResolution", (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)),
        ):
            for value in values:
                if value != getattr(setting.parameters, field):
                    if field == "dimensions" and value >= min(
                        setting.hvgCount, n_cells
                    ):
                        continue
                    if field == "neighborsK" and value >= n_cells:
                        continue
                    options[f"{field}:{value}"] = {"parameter": field, "value": value}
        for count in (1000, 2000, 4000):
            if count != setting.hvgCount:
                options[f"hvgCount:{count}"] = {"parameter": "hvgCount", "value": count}
        for column in self.batch_columns:
            if self.study.columnKinds.get(column) == "continuous":
                continue
            if setting.ranking != "batchAware" or setting.rankingColumn != column:
                options[f"hvgRanking:batchAware:{column}"] = {
                    "parameter": "hvgRanking",
                    "value": "batchAware",
                    "column": column,
                }
        if setting.ranking == "batchAware":
            options["hvgRanking:global"] = {
                "parameter": "hvgRanking",
                "value": "global",
            }
        options.update(self._feature_experiments(setting))
        if self.study.correctionLicense == "safe" and self.batch_columns:
            options["useHarmony:true"] = {"parameter": "useHarmony", "value": True}
        if setting.parameters.useHarmony:
            options["useHarmony:false"] = {"parameter": "useHarmony", "value": False}
        return options

    def _feature_experiments(self, setting: RnaSetting) -> dict[str, dict[str, Any]]:
        """Offer policy changes only when exact eligible genes can change safely."""
        policy = self.plan.assays[0].featureParameters
        assay = self.store.get_assay(self.handoff.assay)
        names = np.asarray(assay.feats.fetch_all("names")).astype(str)
        ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
        eligible = np.asarray(
            self.store.load_artifact(artifact_model_to_ref(setting.eligibleFeatures))[
                "values"
            ][:],
            dtype=bool,
        )
        allowed = np.asarray(
            self.store.load_artifact(
                artifact_model_to_ref(
                    self.handoff.graphFeatureCandidates["eligibleAll"]
                )
            )["values"][:],
            dtype=bool,
        )
        protected = np.isin(names, policy.get("protectFeatures", [])) | np.isin(
            ids, policy.get("protectFeatures", [])
        )
        for family in policy.get("protectFamilies", []):
            mask = _family_mask(names, family)
            if mask is not None:
                protected |= mask
        masks = {
            ("Family", family): np.asarray(
                [
                    re.search(pattern, name, flags=re.IGNORECASE) is not None
                    for name in names
                ]
            )
            for family, pattern in self.family_patterns.items()
        }
        masks.update(
            {
                ("Feature", feature): (names == feature) | (ids == feature)
                for feature in dict.fromkeys(
                    [
                        *policy.get("proposedExcludeFeatures", []),
                        *policy.get("protectFeatures", []),
                    ]
                )
            }
        )
        options = {}
        for (kind, value), mask in masks.items():
            for operation in ("include", "exclude"):
                if operation == "exclude" and (mask & protected).any():
                    continue
                changed = (
                    mask & allowed & ~eligible
                    if operation == "include"
                    else mask & eligible
                )
                if changed.any():
                    field = operation + kind
                    options[f"{field}:{value}"] = {
                        "parameter": field,
                        "value": value,
                        "affectedEligibleGenes": int(changed.sum()),
                    }
        return options

    def batch_ranking(
        self, eligible: ArtifactRef, count: int, column: str
    ) -> np.ndarray:
        """Use core per-group variability on the same globally eligible genes."""
        if column not in self.batch_columns:
            raise ValueError("Batch-aware ranking needs an approved technical column")
        from ..experimental_context.characterization import _SelectionBoundCells

        cells = _SelectionBoundCells(self.store.zw, self.store.cells, self.cells)
        values, counts = np.unique(cells.fetch(column), return_counts=True)
        groups = []
        for value, n_cells in zip(values, counts, strict=True):
            if n_cells < 20:
                continue
            selection = self.store.filter_cells(
                [column],
                [value],
                [value],
                cell_selection=self.cells,
                keep_bounds=True,
                invalidate_cache=False,
            )
            reference = self.store.select_hvgs(
                selection,
                from_assay=self.handoff.assay,
                top_n=self.store.get_assay(self.handoff.assay).feats.N,
                min_cells=1,
                max_cells=np.inf,
                blacklist="",
                show_plot=False,
                invalidate_cache=False,
            )
            status = self.store.inspect_artifact(reference)
            summary = ArtifactRef.from_dict(dict(status.inputs["feature_summary"]))
            group = self.store.load_artifact(reference)
            detected = (
                np.asarray(self.store.load_artifact(summary)["normed_n"][:]) >= 20
            )
            groups.append(
                HvgGroupVariability(
                    group_id=str(value),
                    cell_count=int(n_cells),
                    corrected_variance=np.asarray(
                        group["corrected_variance"][:], dtype=np.float64
                    ),
                    detected_features=detected,
                )
            )
        if len(groups) < 2:
            raise ValueError(
                "Batch-aware ranking lacks two groups with sufficient cells"
            )
        mask = np.asarray(self.store.load_artifact(eligible)["values"][:], dtype=bool)
        statistics = artifact_model_to_ref(
            self.handoff.graphFeatureCandidates["eligibleAll"]
        )
        variance = np.asarray(
            self.store.load_artifact(statistics)["corrected_variance"][:],
            dtype=np.float64,
        )
        ranking = aggregate_hvg_rankings(
            variance,
            mask,
            groups,
            valid_group_count=len(groups),
            candidate_targets=(count,),
        )
        return ranking.ranking

    def apply_experiment(
        self, selected: ParameterCandidateEvaluation, experiment: dict[str, Any]
    ) -> RnaSetting:
        setting = self.settings[selected.candidateId]
        field, value = experiment["parameter"], experiment["value"]
        if field in {"dimensions", "neighborsK", "leidenResolution", "useHarmony"}:
            return setting.model_copy(
                update={
                    "parameters": setting.parameters.model_copy(update={field: value})
                }
            )
        eligible = artifact_model_to_ref(setting.eligibleFeatures)
        count = int(value) if field == "hvgCount" else setting.hvgCount
        ranking_mode = value if field == "hvgRanking" else setting.ranking
        ranking_column = (
            experiment.get("column") if field == "hvgRanking" else setting.rankingColumn
        )
        if field in {
            "includeFamily",
            "excludeFamily",
            "includeFeature",
            "excludeFeature",
        }:
            mask = np.asarray(
                self.store.load_artifact(eligible)["values"][:], dtype=bool
            )
            names = np.asarray(
                self.store.get_assay(self.handoff.assay).feats.fetch_all("names")
            ).astype(str)
            feature_ids = np.asarray(
                self.store.get_assay(self.handoff.assay).feats.fetch_all("ids")
            ).astype(str)
            if field.endswith("Family"):
                pattern = re.compile(self.family_patterns[value], flags=re.IGNORECASE)
                family_mask = np.asarray(
                    [pattern.search(name) is not None for name in names]
                )
            else:
                family_mask = (names == value) | (feature_ids == value)
                if not family_mask.any():
                    raise ValueError(
                        "The nominated exact feature is absent from the assay"
                    )
            if field.startswith("include"):
                all_eligible = artifact_model_to_ref(
                    self.handoff.graphFeatureCandidates["eligibleAll"]
                )
                allowed = np.asarray(
                    self.store.load_artifact(all_eligible)["values"][:], dtype=bool
                )
                mask |= allowed & family_mask
            else:
                policy = self.plan.assays[0].featureParameters
                protected_mask = np.isin(
                    names, policy.get("protectFeatures", [])
                ) | np.isin(feature_ids, policy.get("protectFeatures", []))
                for family in policy.get("protectFamilies", []):
                    family_protection = _family_mask(names, family)
                    if family_protection is not None:
                        protected_mask |= family_protection
                    elif family in self.family_patterns:
                        pattern = re.compile(
                            self.family_patterns[family], flags=re.IGNORECASE
                        )
                        protected_mask |= np.asarray(
                            [pattern.search(name) is not None for name in names]
                        )
                if np.any(family_mask & protected_mask):
                    raise ValueError(
                        "An objective-protected feature or family cannot be excluded"
                    )
                mask &= ~family_mask
            eligible = self.store.set_feature_selection(
                from_assay=self.handoff.assay, mask=mask, invalidate_cache=False
            )
        if ranking_mode == "batchAware" and ranking_column is None:
            raise ValueError(
                "Batch-aware ranking requires an explicit technical column"
            )
        indices = (
            self.batch_ranking(eligible, count, ranking_column)
            if ranking_mode == "batchAware" and ranking_column is not None
            else None
        )
        features = rank_core_hvgs(
            self.store,
            eligible=eligible,
            statistics=artifact_model_to_ref(
                self.handoff.graphFeatureCandidates["eligibleAll"]
            ),
            top_n=count,
            ranking=indices,
        )
        n_features = int(
            np.asarray(
                self.store.load_artifact(features)["values"][:], dtype=bool
            ).sum()
        )
        if n_features <= setting.parameters.dimensions:
            raise ValueError(
                "This feature experiment cannot retain the fixed PCA dimension"
            )
        return setting.model_copy(
            update={
                "features": ArtifactReferenceModel.from_artifact_ref(features),
                "eligibleFeatures": ArtifactReferenceModel.from_artifact_ref(eligible),
                "hvgCount": n_features,
                "ranking": ranking_mode,
                "rankingColumn": ranking_column,
            }
        )

    def _setting_checkpoint(
        self,
        scope: str,
        key: str,
        baseline: RnaSetting,
        experiment: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        """Reuse exact feature work, with a new address for a revised intervention."""
        inputs: dict[str, Any] = {
            "baseline": baseline.model_dump(mode="json"),
            "experiment": experiment,
            "cells": self.cells.to_dict(),
        }
        checkpoint = f"parameter_tuning/{scope}/{key}/setting"
        previous = journal.read_checkpoint(
            self.store, self.prefix, self.workflow.workflowRunId, checkpoint
        )
        if self.evidence_revision is not None:
            ranking_changed = experiment.get("parameter") == "hvgRanking" and any(
                item["studyContract"]["technicalBatchColumns"] != self.batch_columns
                for item in self.previous_provenances
            )
            if previous is None or previous["inputs"] != inputs or ranking_changed:
                checkpoint = f"{self.checkpoint_scope(scope)}/{key}/setting"
                if ranking_changed:
                    inputs["rankingColumns"] = self.batch_columns
        return (
            checkpoint,
            inputs,
            journal.load_checkpoint(
                self.store,
                self.prefix,
                self.workflow.workflowRunId,
                checkpoint,
                inputs,
            ),
        )

    def _prepared_setting(
        self,
        scope: str,
        key: str,
        selected: ParameterCandidateEvaluation,
        experiment: dict[str, Any],
        cells: ArtifactRef,
    ) -> RnaSetting:
        """Commit feature work separately so interrupted reviews do not repeat it."""
        baseline = self.settings[selected.candidateId]
        checkpoint, inputs, saved = self._setting_checkpoint(
            scope, key, baseline, experiment
        )
        if saved is not None:
            return RnaSetting.model_validate(saved["setting"])
        proposed = self.execution_inputs(cells, baseline)
        proposed["features"] = {"requestedExperiment": inputs}
        self.budget.check_many(scope, [proposed])
        setting = self.apply_experiment(selected, experiment)
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            checkpoint,
            inputs,
            {"setting": setting.model_dump(mode="json")},
        )
        return setting

    def _feature_nomination(self, baseline: RnaSetting) -> dict[str, Any] | None:
        """Find one meaningful, previously justified policy intervention."""
        policy = self.plan.assays[0].featureParameters
        eligible = np.asarray(
            self.store.load_artifact(artifact_model_to_ref(baseline.eligibleFeatures))[
                "values"
            ][:],
            dtype=bool,
        )
        all_eligible = np.asarray(
            self.store.load_artifact(
                artifact_model_to_ref(
                    self.handoff.graphFeatureCandidates["eligibleAll"]
                )
            )["values"][:],
            dtype=bool,
        )
        names = np.asarray(
            self.store.get_assay(self.handoff.assay).feats.fetch_all("names")
        ).astype(str)
        ids = np.asarray(
            self.store.get_assay(self.handoff.assay).feats.fetch_all("ids")
        ).astype(str)
        nominations = [
            *(("includeFeature", value) for value in policy.get("protectFeatures", [])),
            *(("includeFamily", value) for value in policy.get("protectFamilies", [])),
            *(
                ("excludeFeature", value)
                for value in policy.get("proposedExcludeFeatures", [])
            ),
            *(
                ("excludeFamily", value)
                for value in policy.get("proposedExcludeFamilies", [])
            ),
        ]
        protected = np.isin(names, policy.get("protectFeatures", [])) | np.isin(
            ids, policy.get("protectFeatures", [])
        )
        for family in policy.get("protectFamilies", []):
            mask = _family_mask(names, family)
            if mask is not None:
                protected |= mask
        for operation, value in nominations:
            if operation.endswith("Family"):
                mask = _family_mask(names, value)
                if mask is None:
                    continue
                if value not in self.family_patterns:
                    value = next(
                        (
                            family
                            for family, pattern in self.family_patterns.items()
                            if np.array_equal(
                                mask,
                                np.asarray(
                                    [
                                        re.search(pattern, name, flags=re.IGNORECASE)
                                        is not None
                                        for name in names
                                    ]
                                ),
                            )
                        ),
                        None,
                    )
                    if value is None:
                        continue
            else:
                mask = (names == value) | (ids == value)
            if operation.startswith("include"):
                changed = mask & all_eligible & ~eligible
            else:
                if (mask & protected).any():
                    continue
                changed = mask & eligible
            if changed.any():
                return {"parameter": operation, "value": value}
        return None

    def _sensitivity_panel(
        self,
        scope: str,
        cells: ArtifactRef,
        baseline: ParameterCandidateEvaluation,
    ) -> None:
        """Execute a small single-axis floor, never a Cartesian product."""
        setting = self.settings[baseline.candidateId]
        n_cells = self.scope_sizes[scope]
        plans: list[tuple[str, str, dict[str, Any] | None, str]] = []
        for field, axis, values in (
            ("hvgCount", "hvgCount", (2000, 4000)),
            ("dimensions", "pca", (10, 30)),
            ("neighborsK", "neighbors", (21, 41)),
        ):
            for value in values:
                reason = ""
                if field == "dimensions" and value >= min(setting.hvgCount, n_cells):
                    reason = "The requested PCA dimension exceeds the observed cell or selected-feature rank."
                if field == "neighborsK" and value >= n_cells:
                    reason = (
                        "The requested neighbor count exceeds the observed cell count."
                    )
                plans.append(
                    (
                        f"{field}:{value}",
                        axis,
                        None if reason else {"parameter": field, "value": value},
                        reason,
                    )
                )
        columns = list(
            dict.fromkeys(
                [
                    *(
                        [self.study.physicalCaptureColumn]
                        if self.study.physicalCaptureColumn in self.batch_columns
                        else []
                    ),
                    *self.batch_columns,
                ]
            )
        )
        ranking = None
        ranking_groups = {}
        for column in columns:
            if self.study.columnKinds.get(column) == "continuous":
                continue
            indices = read_stored_selection_indices(
                self.store.zw,
                self.cells,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            )
            group_values = read_metadata_rows_chunkwise(
                self.store.cells, column, indices
            )
            _, counts = np.unique(group_values, return_counts=True)
            ranking_groups[column] = int((counts >= 20).sum())
            if int((counts >= 20).sum()) >= 2:
                ranking = {
                    "parameter": "hvgRanking",
                    "value": "batchAware",
                    "column": column,
                }
                break
        plans.append(
            (
                "hvgRanking",
                "hvgRanking",
                ranking,
                "No approved categorical technical grouping has two groups with at least 20 cells."
                if ranking is None
                else "",
            )
        )
        nomination = self._feature_nomination(setting)
        pending_feature = nomination is None and bool(
            self._feature_experiments(setting)
        )
        plans.append(
            (
                "featurePolicy",
                "featurePolicy",
                nomination,
                "Baseline loading, marker and objective evidence must nominate one of the offered policy interventions before combining."
                if pending_feature
                else "Every registered inclusion/exclusion is a no-op on the exact eligible genes or would remove objective-protected genes."
                if nomination is None
                else "",
            )
        )
        proposed: list[dict[str, Any]] = []
        for identifier, _, experiment, _ in plans:
            if experiment is None:
                continue
            proposal = self.execution_inputs(cells, setting)
            if experiment["parameter"] in {"dimensions", "neighborsK"}:
                proposal["parameters"] = {
                    **proposal["parameters"],
                    experiment["parameter"]: experiment["value"],
                }
            else:
                _, _, prepared = self._setting_checkpoint(
                    scope, f"sensitivity/{identifier}", setting, experiment
                )
                if prepared is not None:
                    known = RnaSetting.model_validate(prepared["setting"])
                    if (
                        known.features == setting.features
                        or np.array_equal(
                            self.store.load_artifact(
                                artifact_model_to_ref(known.features)
                            )["values"][:],
                            self.store.load_artifact(
                                artifact_model_to_ref(setting.features)
                            )["values"][:],
                        )
                        or (
                            identifier == "featurePolicy"
                            and known.hvgCount != setting.hvgCount
                        )
                    ):
                        continue
                    proposal = self.execution_inputs(cells, known)
                else:
                    proposal["features"] = {"requiredSensitivity": identifier}
            proposed.append(proposal)
        self.budget.check_many(scope, proposed)
        rows: list[dict[str, Any]] = [
            {
                "comparisonId": f"defaultResolution:{item.parameters.leidenResolution}",
                "axis": "partition",
                "status": "completed",
                "baselineCandidateId": baseline.candidateId,
                "alternativeCandidateId": item.candidateId,
                "reason": "",
            }
            for item in self.evaluations[scope]
            if item.candidateId != baseline.candidateId
            and item.parameters.leidenResolution in {0.5, 0.75, 1.25}
        ]
        for identifier, axis, experiment, reason in plans:
            alternative = None
            proof: dict[str, Any] = {
                "baselineFeatures": setting.features.model_dump(mode="json")
            }
            if axis in {"pca", "neighbors"}:
                proof["kind"] = "numericalBound"
            elif axis == "hvgRanking":
                proof.update(
                    kind="insufficientTechnicalGroups",
                    eligibleGroupsByColumn=ranking_groups,
                )
            elif axis == "featurePolicy":
                proof.update(
                    kind="noPermittedPolicy",
                    meaningfulPermittedInterventions=0,
                    registeredFamilies=list(self.family_patterns),
                )
            if experiment is not None:
                if experiment["parameter"] in {"dimensions", "neighborsK"}:
                    alternative_setting = self.apply_experiment(baseline, experiment)
                else:
                    alternative_setting = self._prepared_setting(
                        scope,
                        f"sensitivity/{identifier}",
                        baseline,
                        experiment,
                        cells,
                    )
                same_genes = alternative_setting.features == setting.features
                if not same_genes and axis in {
                    "hvgCount",
                    "hvgRanking",
                    "featurePolicy",
                }:
                    same_genes = np.array_equal(
                        self.store.load_artifact(
                            artifact_model_to_ref(alternative_setting.features)
                        )["values"][:],
                        self.store.load_artifact(
                            artifact_model_to_ref(setting.features)
                        )["values"][:],
                    )
                if (
                    axis == "featurePolicy"
                    and alternative_setting.hvgCount != setting.hvgCount
                ):
                    reason = "The nominated policy leaves too few eligible genes to hold the baseline HVG count fixed."
                    proof.update(
                        kind="insufficientEligibleGenes",
                        eligibleFeatureCount=alternative_setting.hvgCount,
                    )
                elif same_genes and axis in {"hvgCount", "hvgRanking", "featurePolicy"}:
                    reason = "The requested intervention produces exactly the same selected genes as the baseline; its numerical representation is already evaluated."
                    proof.update(
                        kind="identicalSelectedGenes",
                        verifiedEqualMasks=True,
                        alternativeSetting=alternative_setting.model_dump(mode="json"),
                    )
                elif (
                    axis in {"pca", "neighbors"}
                    and setting_changes(
                        setting.model_dump(mode="json"),
                        alternative_setting.model_dump(mode="json"),
                    )
                    == {}
                ):
                    reason = (
                        "The requested value is already the exact baseline setting."
                    )
                else:
                    alternative = self.execute(scope, cells, alternative_setting)
            rows.append(
                {
                    "comparisonId": identifier,
                    "axis": axis,
                    "status": "completed"
                    if alternative is not None
                    else "pending"
                    if identifier == "featurePolicy" and pending_feature
                    else "notApplicable",
                    "baselineCandidateId": baseline.candidateId,
                    "alternativeCandidateId": alternative.candidateId
                    if alternative is not None
                    else None,
                    "reason": reason,
                    "observedProof": proof if alternative is None else None,
                }
            )
        self.comparison_rows[scope] = rows

    def comparison_coverage(self, scope: str, cells: ArtifactRef) -> dict[str, Any]:
        source = (
            (self.discovery_scope or self.recovery_scope) if scope == "full" else scope
        )
        source = source or scope
        settings = {
            item.candidateId: {
                **self.settings[item.candidateId].model_dump(mode="json"),
                "scope": name,
                "status": item.status,
                "nCells": self.scope_sizes.get(name, self.handoff.nCells),
                "cellSelection": item.cellSelection.model_dump(mode="json")
                if item.cellSelection is not None
                else None,
                "metrics": item.metrics.model_dump(
                    mode="json",
                    include={
                        "nClusters",
                        "minClusterCells",
                        "seedStability",
                        "subsampleStability",
                        "markerCoherence",
                        "markerSpecificityMedian",
                        "macroF1",
                        "weightedF1",
                        "graphSilhouetteMedian",
                        "clusterConnectivity",
                        "crossUnitSupport",
                        "batchMixing",
                        "biologicalPreservation",
                        "topMarkerGenes",
                    },
                ),
            }
            for name, rows in self.evaluations.items()
            for item in rows
        }
        rows = list(self.comparison_rows.get(source, []))
        panel_source = (
            "full" if scope == "full" and "full" in self.combined_candidates else source
        )
        combined_id = self.combined_candidates.get(panel_source)
        resolution_ids = self.resolution_candidates.get(panel_source, [])
        if combined_id is not None:
            rows.extend(
                {
                    "comparisonId": f"combinedResolution:{settings[identifier]['parameters']['leidenResolution']}",
                    "axis": "partition",
                    "status": "completed",
                    "baselineCandidateId": combined_id,
                    "alternativeCandidateId": identifier,
                    "reason": "",
                }
                for identifier in resolution_ids
                if identifier != combined_id
            )
        return {
            "phase": "validation"
            if scope == "full"
            and (self.discovery_scope is not None or self.recovery_scope is not None)
            else "combined"
            if source in self.combined_candidates
            else "sensitivity",
            "population": "allCells" if cells == self.cells else "subset",
            "comparisons": rows,
            "candidateSettings": settings,
            "combinedCandidateId": combined_id,
            "resolutionCandidateIds": resolution_ids,
            "validationSources": dict(self.validation_sources),
            "fullRepair": self.full_repair,
        }

    def _combined_setting(
        self,
        scope: str,
        review_index: int,
        action: TuningAction,
        cells: ArtifactRef,
    ) -> RnaSetting:
        assert action.combinedSettings is not None
        choices = action.combinedSettings
        policy = self.settings[choices.featurePolicyCandidateId]
        count = self.settings[choices.hvgCountCandidateId].hvgCount
        ranking = self.settings[choices.hvgRankingCandidateId]
        parameters = policy.parameters.model_copy(
            update={
                "dimensions": self.settings[
                    choices.pcaCandidateId
                ].parameters.dimensions,
                "neighborsK": self.settings[
                    choices.neighborsCandidateId
                ].parameters.neighborsK,
                "leidenResolution": 1.0,
                "useHarmony": False,
            }
        )
        inputs = {
            "choices": choices.model_dump(mode="json"),
            "settings": {
                identifier: self.settings[identifier].model_dump(mode="json")
                for identifier in choices.model_dump().values()
            },
            "cells": self.cells.to_dict(),
        }
        key = f"{self.checkpoint_scope(scope)}/review{review_index}/combined_setting"
        saved = journal.load_checkpoint(
            self.store, self.prefix, self.workflow.workflowRunId, key, inputs
        )
        if saved is not None:
            return RnaSetting.model_validate(saved["setting"])
        eligible = artifact_model_to_ref(policy.eligibleFeatures)
        if (
            count == policy.hvgCount
            and ranking.ranking == policy.ranking
            and ranking.rankingColumn == policy.rankingColumn
        ):
            features = policy.features
        else:
            proposed = self.execution_inputs(cells, policy)
            proposed["features"] = {"combinedSettings": inputs}
            proposed["parameters"] = parameters.model_dump(mode="json")
            self.budget.check_many(scope, [proposed])
            order = (
                self.batch_ranking(eligible, count, ranking.rankingColumn)
                if ranking.ranking == "batchAware" and ranking.rankingColumn is not None
                else None
            )
            features = ArtifactReferenceModel.from_artifact_ref(
                rank_core_hvgs(
                    self.store,
                    eligible=eligible,
                    statistics=artifact_model_to_ref(
                        self.handoff.graphFeatureCandidates["eligibleAll"]
                    ),
                    top_n=count,
                    ranking=order,
                )
            )
        actual_count = int(
            np.asarray(
                self.store.load_artifact(artifact_model_to_ref(features))["values"][:],
                dtype=bool,
            ).sum()
        )
        if parameters.dimensions >= min(actual_count, self.scope_sizes[scope]):
            raise ValueError(
                "Combined settings cannot support the selected PCA dimension"
            )
        if actual_count != count:
            raise ValueError(
                "The proposed combined HVG count was not retained by the selected feature policy"
            )
        setting = RnaSetting(
            parameters=parameters,
            features=features,
            eligibleFeatures=policy.eligibleFeatures,
            hvgCount=actual_count,
            ranking=ranking.ranking,
            rankingColumn=ranking.rankingColumn,
        )
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            key,
            inputs,
            {"setting": setting.model_dump(mode="json")},
        )
        return setting

    def _resolution_panel(
        self,
        scope: str,
        cells: ArtifactRef,
        setting: RnaSetting,
    ) -> ParameterCandidateEvaluation:
        settings = [
            setting.model_copy(
                update={
                    "parameters": setting.parameters.model_copy(
                        update={"leidenResolution": resolution}
                    )
                }
            )
            for resolution in (0.5, 0.75, 1.0, 1.25)
        ]
        proposals = [self.execution_inputs(cells, item) for item in settings]
        if setting.parameters.useHarmony:
            proposals += [
                self.execution_inputs(
                    cells,
                    item.model_copy(
                        update={
                            "parameters": item.parameters.model_copy(
                                update={"useHarmony": False}
                            )
                        }
                    ),
                )
                for item in settings
            ]
        self.budget.check_many(
            scope,
            [
                row
                for row in proposals
                if scope != "full" or self.budget.completed_source(row) is None
            ],
        )
        if scope == "full" and self.recovery_scope is not None:
            recovery_inputs = {
                "sourceScope": self.recovery_scope,
                "sourceReview": self.last_action.model_dump(mode="json")
                if self.last_action is not None
                else None,
                "cells": cells.to_dict(),
                "setting": setting.model_dump(mode="json"),
            }
            journal.save_checkpoint(
                self.store,
                self.prefix,
                self.workflow.workflowRunId,
                "parameter_tuning/full/targeted_recovery/"
                + hashlib.sha256(
                    record_io.canonical_json_bytes(recovery_inputs)
                ).hexdigest(),
                inputs=recovery_inputs,
                outputs={
                    "purpose": "Resolve the recorded screening concern using full-cohort population, marker, stability and preservation evidence for the proposed settings.",
                    "screeningAccepted": False,
                    "plannedEvaluations": proposals,
                    "limits": self.budget.summary()["limits"],
                },
            )
            self.budget.admit_many(
                scope,
                [row for row in proposals if self.budget.completed_source(row) is None],
            )
        evaluations = [self.execute_matched(scope, cells, item) for item in settings]
        self.resolution_candidates[scope] = [item.candidateId for item in evaluations]
        return next(
            item for item in evaluations if item.parameters.leidenResolution == 1.0
        )

    def feature_evidence(
        self, selected: ParameterCandidateEvaluation
    ) -> dict[str, Any]:
        """Summarize frozen core statistics without rerunning feature variability."""
        setting = self.settings[selected.candidateId]
        key = setting.model_dump_json(exclude={"parameters"})
        if key in self.feature_evidence_cache:
            return self.feature_evidence_cache[key]
        mask = np.asarray(
            self.store.load_artifact(artifact_model_to_ref(setting.features))["values"][
                :
            ],
            dtype=bool,
        )
        eligible = np.asarray(
            self.store.load_artifact(artifact_model_to_ref(setting.eligibleFeatures))[
                "values"
            ][:],
            dtype=bool,
        )
        reference = artifact_model_to_ref(
            self.handoff.graphFeatureCandidates["eligibleAll"]
        )
        variance = np.asarray(
            self.store.load_artifact(reference)["corrected_variance"][:],
            dtype=np.float64,
        )
        names = np.asarray(
            self.store.get_assay(self.handoff.assay).feats.fetch_all("names")
        ).astype(str)
        indices = np.flatnonzero(eligible)
        ranked = indices[np.lexsort((indices, -variance[indices]))]
        families = {}
        for family, pattern in self.family_patterns.items():
            expression = re.compile(pattern, flags=re.IGNORECASE)
            membership = np.asarray(
                [expression.search(name) is not None for name in names]
            )
            families[family] = {
                "eligibleGenes": int((eligible & membership).sum()),
                "selectedGenes": int((mask & membership).sum()),
                "selectedExamples": names[
                    np.flatnonzero(mask & membership)[:8]
                ].tolist(),
                "excludedExamples": names[
                    np.flatnonzero(~eligible & membership)[:8]
                ].tolist(),
            }
        evidence: dict[str, Any] = {
            "statistics": reference.to_dict(),
            "statisticsCells": self.cells.to_dict(),
            "basis": "Core Scarf corrected variance on the full QC-retained cells; feature-axis summaries are descriptive, not a substitute for downstream comparisons.",
            "selectedGenes": int(mask.sum()),
            "eligibleGenes": int(eligible.sum()),
            "ranking": setting.ranking,
            "rankingColumn": setting.rankingColumn,
            "globalRankLandmarks": [
                {
                    "rank": rank,
                    "gene": str(names[ranked[rank - 1]]),
                    "correctedVariance": float(variance[ranked[rank - 1]]),
                }
                for rank in (1, 500, 1000, 2000, 4000)
                if rank <= len(ranked)
            ],
            "topSelectedGenes": names[
                [index for index in ranked if mask[index]][:20]
            ].tolist(),
            "families": families,
        }
        self.feature_evidence_cache[key] = evidence
        return evidence

    def review(
        self,
        scope: str,
        review_index: int,
        selected: ParameterCandidateEvaluation,
        coverage: dict[str, Any],
    ) -> TuningAction:
        from .tuning import _analysis_visual_content

        candidates = self.evaluations[scope]
        key = f"{self.checkpoint_scope(scope)}/review{review_index}"
        previous_review = journal.read_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            key,
        )
        candidate_evidence = [item.model_dump(mode="json") for item in candidates]
        setting_evidence = {
            item.candidateId: self.settings[item.candidateId].model_dump(mode="json")
            for item in candidates
        }
        committed_review = previous_review
        prepared_inputs = {
            "candidatesSha256": hashlib.sha256(
                record_io.canonical_json_bytes(candidate_evidence)
            ).hexdigest(),
            "settingsSha256": hashlib.sha256(
                record_io.canonical_json_bytes(setting_evidence)
            ).hexdigest(),
            "coverage": coverage,
            "provenance": self.provenance,
        }
        prepared = None
        if previous_review is None:
            for evidence_mode in ("structured", "visual"):
                prepared = journal.load_checkpoint(
                    self.store,
                    self.prefix,
                    self.workflow.workflowRunId,
                    f"{key}/evidence/{evidence_mode}",
                    inputs=prepared_inputs,
                )
                if prepared is not None:
                    previous_review = {"inputs": prepared["evidence"]}
                    break
        if previous_review is not None and (
            previous_review["inputs"].get("candidates") != candidate_evidence
            or previous_review["inputs"].get("settings") != setting_evidence
        ):
            raise ValueError(
                "Saved review has different candidate evidence or settings. "
                "If this review predates required Harmony comparisons for safe designs, "
                "start a new workflow in a different destination; the saved history and artifacts remain intact."
            )
        if (
            previous_review is not None
            and "comparisonCoverage" not in previous_review["inputs"]
        ):
            raise ValueError(
                "Saved analysis lacks mandatory sensitivity coverage; start a new workflow"
            )
        if previous_review is not None:
            diagnostic_reuse("diagnostic.reviewEvidence", "restored")
        experiments = (
            previous_review["inputs"]["experiments"]
            if previous_review is not None
            else self.experiments(selected)
        )
        completed_experiments: dict[str, str] = {}
        matched_comparisons = []
        current_setting = self.settings[selected.candidateId]
        for candidate in candidates:
            other = self.settings[candidate.candidateId]
            if (
                candidate.candidateId == selected.candidateId
                or candidate.cellSelection != selected.cellSelection
                or other.parameters.reductionMethod
                != current_setting.parameters.reductionMethod
            ):
                continue
            changes = setting_changes(
                current_setting.model_dump(mode="json"), other.model_dump(mode="json")
            )
            if len(changes) != 1:
                continue
            matched_comparisons.append(
                {
                    "currentCandidateId": selected.candidateId,
                    "alternativeCandidateId": candidate.candidateId,
                    "changedParameter": changes,
                    "partitionEvidence": diagnostic_call(
                        "diagnostic.partitionComparison",
                        partition_comparison_evidence,
                        self.store,
                        selected,
                        candidate,
                    )
                    if previous_review is None
                    else {},
                    "basis": "Same frozen cells and every unaffected logical setting; the named intervention alone changes. Feature comparisons hold ranking/count/eligible genes fixed except for their declared axis.",
                }
            )
            if previous_review is None and candidate.status == "done":
                field, values = next(iter(changes.items()))
                field = {
                    "pca": "dimensions",
                    "neighbors": "neighborsK",
                    "partition": "leidenResolution",
                    "correction": "useHarmony",
                }.get(field, field)
                for experiment_id, experiment in experiments.items():
                    requested_value = (
                        [experiment["value"], experiment.get("column")]
                        if field == "hvgRanking"
                        else experiment["value"]
                    )
                    if (
                        experiment["parameter"] == field
                        and requested_value == values["alternative"]
                    ):
                        completed_experiments[experiment_id] = candidate.candidateId
        if previous_review is None:
            experiments = {
                key: value
                for key, value in experiments.items()
                if key not in completed_experiments
            }
        comparisons = (
            previous_review["inputs"].get("neighborComparisons", [])
            if previous_review is not None
            else []
        )
        selected_neighbors = selected.artifacts.get("neighbors")
        for alternative in [] if previous_review is not None else candidates:
            other_neighbors = alternative.artifacts.get("neighbors")
            if (
                selected_neighbors is None
                or other_neighbors is None
                or selected_neighbors == other_neighbors
                or selected.parameters.neighborsK != alternative.parameters.neighborsK
            ):
                continue
            left_id, right_id = sorted(
                (selected_neighbors.artifactId, other_neighbors.artifactId)
            )
            pair = (left_id, right_id)
            if pair not in self.neighbor_comparisons:
                self.neighbor_comparisons[pair] = diagnostic_call(
                    "diagnostic.neighborOverlap",
                    _neighbor_overlap,
                    self.store,
                    artifact_model_to_ref(selected_neighbors),
                    artifact_model_to_ref(other_neighbors),
                )
            else:
                diagnostic_reuse("diagnostic.neighborOverlap", "cacheHits")
            comparisons.append(
                {
                    "leftCandidateId": selected.candidateId,
                    "rightCandidateId": alternative.candidateId,
                    "meanNeighborJaccard": self.neighbor_comparisons[pair],
                    "basis": "Same frozen cells and k; descriptive response to settings, not independent stability or evidence that correction is beneficial.",
                }
            )
        declared_image_input = _configured_image_input(self.owner.model)
        capability_key = "parameter_tuning/structured_evidence"
        capability_inputs = {
            **(
                self.previous_provenances[0]
                if self.previous_provenances
                else self.provenance
            ),
            "configuredImageInput": declared_image_input,
        }
        capability = journal.load_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            capability_key,
            inputs=capability_inputs,
        )
        if capability is None and declared_image_input is False:
            capability = journal.save_checkpoint(
                self.store,
                self.prefix,
                self.workflow.workflowRunId,
                capability_key,
                inputs=capability_inputs,
                outputs={
                    "evidenceMode": "structured",
                    "reason": "configuredImageInputUnsupported",
                },
            )
        if capability is not None and capability.get("evidenceMode") != "structured":
            raise ValueError("The recorded model capability is invalid")
        mode = (
            committed_review["inputs"].get("evidenceMode")
            if committed_review is not None
            else "structured"
            if capability is not None
            else previous_review["inputs"].get("evidenceMode")
            if previous_review is not None
            else "visual"
        )
        if mode not in {"visual", "structured"}:
            raise ValueError(
                "Saved review lacks an exact evidence mode; start a new workflow"
            )
        visual_inspection = "available" if mode == "visual" else "unavailable"
        images = []
        if (
            previous_review is not None
            and previous_review["inputs"].get("evidenceMode") == mode
        ):
            image_hashes = previous_review["inputs"].get("imageHashes", {})
            if prepared is not None:
                images = [
                    ImageEvidence(
                        identifier=item["identifier"],
                        data=base64.b64decode(item["dataBase64"], validate=True),
                        media_type=item["mediaType"],
                    )
                    for item in prepared["images"]
                ]
                if {
                    item.identifier: hashlib.sha256(item.data).hexdigest()
                    for item in images
                } != image_hashes:
                    raise ValueError(
                        "Saved review images do not match their exact evidence hashes"
                    )
        elif mode == "visual":
            images = _analysis_visual_content(
                self.store,
                selected,
                candidates,
                qc_columns=self.plan.cellQc.attributes,
                qc_artifact_metrics=[
                    (item.name, item.artifact)
                    for item in self.plan.cellQc.artifactMetrics
                ],
            )
            image_hashes = {
                image.identifier: hashlib.sha256(image.data).hexdigest()
                for image in images
            }
        else:
            image_hashes = {}
        if not isinstance(image_hashes, dict) or bool(image_hashes) != (
            mode == "visual"
        ):
            raise ValueError("Review images do not match its evidence mode")
        evidence_ids = (
            [
                identifier
                for identifier in previous_review["inputs"]["availableEvidenceIds"]
                if identifier not in previous_review["inputs"].get("imageHashes", {})
                or identifier in image_hashes
            ]
            if previous_review is not None
            else list(
                dict.fromkeys(
                    [
                        *(f"candidate:{item.candidateId}" for item in candidates),
                        *(key for item in candidates for key in item.evidenceIds),
                        *self.study.evidenceIds,
                        *self.plan.cellQc.evidenceIds,
                        *(item.evidenceId for item in self.design_comparisons),
                        *image_hashes,
                        "studyContract",
                        "qcPolicy",
                        "samplingCoverage",
                        "featureEvidence",
                        "neighborComparisons",
                        "assessmentContext",
                        "comparisonCoverage",
                    ]
                )
            )
        )
        if not isinstance(evidence_ids, list) or any(
            not isinstance(value, str) for value in evidence_ids
        ):
            raise ValueError("Review evidence IDs must be a list of strings")
        evidence: dict[str, Any] = {
            "studyContract": self.study.model_dump(mode="json"),
            "qcPolicy": self.plan.cellQc.model_dump(mode="json"),
            "scope": scope,
            "evidenceMode": mode,
            "visualInspection": visual_inspection,
            "configuredImageInput": declared_image_input,
            "coverage": coverage,
            "currentCandidateId": selected.candidateId,
            "candidates": candidate_evidence,
            "settings": setting_evidence,
            "featureEvidence": previous_review["inputs"]["featureEvidence"]
            if previous_review is not None
            else {item.candidateId: self.feature_evidence(item) for item in candidates},
            "neighborComparisons": comparisons,
            "harmonyGates": {
                item.candidateId: self.harmony_gate(scope, item)
                for item in candidates
                if item.parameters.useHarmony
            },
            "experiments": experiments,
            "availableEvidenceIds": evidence_ids,
            "imageHashes": image_hashes,
            "budget": {
                "visibleEvaluations": {
                    name: len(rows) for name, rows in self.evaluations.items()
                },
                "limits": self.budget.summary()["limits"],
            },
            "fullRepairsUsed": previous_review["inputs"]["fullRepairsUsed"]
            if previous_review is not None
            else self.full_repairs,
            "comparisonCoverage": self.comparison_coverage(
                scope, artifact_model_to_ref(selected.cellSelection)
            )
            if selected.cellSelection is not None
            else {},
            "pilotPopulationWarnings": {
                item.candidateId: "This sampled partition includes fewer than 20 cells in a population. Assess its relevance and support; it is not evidence of an invalid biological group. Accepting this partition requires a larger sample or full-cohort assessment."
                for item in candidates
                if scope != "full"
                and selected.cellSelection != self.handoff.cellSelection
                and item.metrics.minClusterCells is not None
                and item.metrics.minClusterCells < 20
            },
            "smallPopulationPolicy": "Small full-cohort groups require explicit marker, stability, graph and applicable independent-unit support assessment against the objective. Their size alone is neither proof of biology nor grounds for rejection. Defer when essential evidence is insufficient.",
        }
        if previous_review is None:
            support_columns = list(
                dict.fromkeys(
                    column
                    for column in (
                        self.study.physicalCaptureColumn,
                        *self.study.independentUnitColumns,
                    )
                    if column is not None
                )
            )
            evidence["assessmentContext"] = {
                "correctionPolicy": {
                    "license": self.study.correctionLicense,
                    "harmonyPermitted": self.study.correctionLicense == "safe"
                    and bool(self.batch_columns),
                    "nativeAcceptance": "A native descriptive analysis may be accepted when its required evidence supports the objective, while explicitly retaining confounding limitations. A Harmony gate is required only for accepting Harmony. Defer if the objective requires effects that the design cannot separate.",
                },
                "matchedComparisons": matched_comparisons,
                "alreadyEvaluatedExperiments": completed_experiments,
                "designComparisons": [
                    item.model_dump(mode="json") for item in self.design_comparisons
                ],
                "populationSupport": {
                    item.candidateId: diagnostic_call(
                        "diagnostic.populationSupport",
                        population_support_evidence,
                        self.store,
                        item,
                        support_columns,
                    )
                    for item in candidates
                    if scope == "full" or item.candidateId == selected.candidateId
                }
                if support_columns
                else {},
                "previousActions": [
                    {
                        "scope": row["scope"],
                        **{
                            name: row["review"][name]
                            for name in (
                                "action",
                                "selectedCandidateId",
                                "experimentId",
                                "correctionNeed",
                            )
                        },
                    }
                    for row in self.history
                    if "review" in row
                ],
            }
        elif "assessmentContext" in previous_review["inputs"]:
            evidence["assessmentContext"] = previous_review["inputs"][
                "assessmentContext"
            ]
        if committed_review is None:
            evidence["comparisonAdvantages"] = comparison_advantages(
                evidence["comparisonCoverage"]
            )
        elif "comparisonAdvantages" in committed_review["inputs"]:
            evidence["comparisonAdvantages"] = committed_review["inputs"][
                "comparisonAdvantages"
            ]

        def validate(action: TuningAction, *, replay: bool = False) -> TuningAction:
            if not replay:
                action = TuningAction.model_validate(
                    bind_comparison_measurements(
                        evidence["comparisonCoverage"], action.model_dump(mode="json")
                    )
                )
            by_id = {item.candidateId: item for item in candidates}
            if action.selectedCandidateId not in by_id:
                raise ValueError("Choose an observed candidate from the current cells")
            unknown_ids = sorted(set(action.evidenceIds).difference(evidence_ids))
            if unknown_ids:
                raise ValueError(
                    f"Assessment cited unknown evidence: {unknown_ids[:8]!r}. "
                    "Copy exact IDs from availableEvidenceIds; do not invent "
                    "suffixes or use artifact IDs as evidence IDs. The selected "
                    f"candidate anchor is 'candidate:{action.selectedCandidateId}'."
                )
            if mode == "visual" and not set(action.evidenceIds).intersection(
                image_hashes
            ):
                raise ValueError("Assessment must cite its actual visual evidence")
            chosen = by_id[action.selectedCandidateId]
            if not set(action.evidenceIds).intersection(
                {f"candidate:{chosen.candidateId}", *chosen.evidenceIds}
            ):
                raise ValueError(
                    "Assessment must cite the selected numerical evidence: "
                    f"'candidate:{chosen.candidateId}' or one of that candidate's "
                    "supplied evidenceIds in availableEvidenceIds."
                )
            if (
                action.experimentId is not None
                and action.experimentId not in experiments
            ):
                raise ValueError(
                    f"Unknown experiment ID {action.experimentId!r}. "
                    f"For current candidate {selected.candidateId!r}, copy one exact "
                    f"key from experiments: {list(experiments)!r}. "
                    "Only action='experiment' may name a next operation."
                )
            if (
                action.action == "experiment"
                and action.selectedCandidateId != selected.candidateId
            ):
                raise ValueError(
                    "An offered experiment must use the current candidate as its fixed baseline"
                )
            validate_comparison_review(
                evidence["comparisonCoverage"], action.model_dump(mode="json")
            )
            if action.combinedSettings is not None:
                choices = action.combinedSettings
                count = self.settings[choices.hvgCountCandidateId].hvgCount
                available = evidence["featureEvidence"][
                    choices.featurePolicyCandidateId
                ]["eligibleGenes"]
                if count > available:
                    raise ValueError(
                        "The combined feature policy cannot supply the requested HVG count; choose a compatible observed count or policy"
                    )
            if (
                self.study.correctionLicense == "unsafeConfounded"
                and action.correctionNeed in {"needed", "notNeeded"}
            ):
                message = (
                    "The design confounds the batch columns with protected biology; their association cannot establish a removable technical effect. "
                    "Harmony is not permitted, and a PCA or feature experiment is not a substitute for Harmony. "
                    "Use notApplicable for the prohibited correction with an explicit confounding limitation, or uncertain and defer if an essential question cannot be resolved."
                )
                if replay and action.action == "experiment":
                    self.history.append(
                        {
                            "scope": scope,
                            "reason": "A saved screening rationale claimed identifiable correction necessity despite the confounded design. Its numerical experiment remains in the audit history; that scientific claim must be reassessed from the supplied evidence.",
                        }
                    )
                else:
                    raise ValueError(message)
            if (
                self.study.correctionLicense == "safe"
                and action.correctionNeed == "notApplicable"
            ):
                raise ValueError(
                    "A safe correction design still requires an observed necessity assessment"
                )
            if action.action == "accept":
                if action.correctionNeed == "uncertain":
                    raise ValueError(
                        "Uncertain correction necessity requires further evidence or deferral before acceptance"
                    )
                if not chosen.eligible:
                    raise ValueError(
                        "The selected candidate failed required full-cell checks"
                    )
                accepted, reasons = self.harmony_gate(scope, chosen)
                if not accepted:
                    raise ValueError("Harmony acceptance failed: " + "; ".join(reasons))
                required = (
                    "seedStability",
                    "subsampleStability",
                    "markerCoherence",
                    "membershipStrengthMean",
                    "clusterConnectivity",
                )
                if any(getattr(chosen.metrics, field) is None for field in required):
                    raise ValueError(
                        "Required stability, marker or graph evidence is missing"
                    )
                if (
                    self.study.independentUnitColumns
                    and chosen.metrics.crossUnitSupport is None
                ):
                    raise ValueError(
                        "Required independent-unit support evidence is missing"
                    )
                if chosen.parameters.useHarmony and action.correctionNeed != "needed":
                    raise ValueError(
                        "Accepting Harmony requires observed correction necessity"
                    )
                if self.study.unsupportedProtection and action.correctionNeed in {
                    "needed",
                    "uncertain",
                }:
                    raise ValueError(
                        "Correction necessity remains unresolved because required matched biological protection is unsupported"
                    )
                if (
                    not chosen.parameters.useHarmony
                    and action.correctionNeed == "needed"
                ):
                    raise ValueError(
                        "Native acceptance leaves required correction unresolved; provide more evidence or defer"
                    )
                if self.study.correctionLicense == "safe":
                    chosen_setting = self.settings[chosen.candidateId]
                    has_comparison = any(
                        item.parameters.useHarmony
                        and item.status == "done"
                        and item.cellSelection == chosen.cellSelection
                        and item.harmonyBatchColumns == self.batch_columns
                        and self.settings[item.candidateId].features
                        == chosen_setting.features
                        and item.parameters.model_dump(
                            exclude={"candidateId", "useHarmony"}
                        )
                        == chosen.parameters.model_dump(
                            exclude={"candidateId", "useHarmony"}
                        )
                        for item in candidates
                    )
                    if not has_comparison:
                        raise ValueError(
                            "A safe correction design requires a completed, matched Harmony experiment "
                            "before acceptance, including a notNeeded conclusion. "
                            "Evaluate the offered useHarmony:true experiment or defer; "
                            "the comparison must use the same cells, features, settings, and approved batch columns."
                        )
                if self.study.correctionLicense == "indeterminate":
                    raise ValueError(
                        "Correction design authorization remains indeterminate"
                    )
            return action

        saved = journal.load_checkpoint(
            self.store, self.prefix, self.workflow.workflowRunId, key, inputs=evidence
        )
        pending_saved = saved is not None and saved["action"]["action"] == "defer"
        answer = self.answers.get(key)
        if (
            answer is None
            and not self.answer_consumed
            and (saved is None or pending_saved)
        ):
            answer = self.answers.get("parameter_tuning")
        if pending_saved:
            assert saved is not None
            answer_inputs = {**evidence, "deferredAction": saved["action"]}
            prior_answer = journal.load_checkpoint(
                self.store,
                self.prefix,
                self.workflow.workflowRunId,
                key + "/answer",
                inputs=answer_inputs,
            )
            if prior_answer is not None:
                if (
                    answer is not None
                    and TuningAction.model_validate(answer).model_dump(mode="json")
                    == prior_answer["action"]
                ):
                    self.answer_consumed = True
                key += "/answer"
                evidence = answer_inputs
                saved = prior_answer
                answer = None
            elif answer is not None:
                key += "/answer"
                evidence = answer_inputs
                saved = None
        if saved is not None:
            action = validate(TuningAction.model_validate(saved["action"]), replay=True)
        elif answer is not None:
            action = validate(TuningAction.model_validate(answer))
            self.answer_consumed = True
        else:
            journal.save_checkpoint(
                self.store,
                self.prefix,
                self.workflow.workflowRunId,
                f"{key}/evidence/{mode}",
                inputs=prepared_inputs,
                outputs={
                    "evidence": evidence,
                    "images": [
                        {
                            "identifier": item.identifier,
                            "mediaType": item.media_type,
                            "dataBase64": base64.b64encode(item.data).decode("ascii"),
                        }
                        for item in images
                    ],
                },
            )
            prompt = (
                "Assess this RNA analysis as a computational biologist against the exact study objective. "
                "Start from Scarf defaults; keep them when observed quantitative evidence and biological interpretation support them. "
                "Do not execute a search grid or favor a default solely because it is a default. "
                "Explain observed marker programs and relevant PCA loading genes, "
                "The supplied comparisonCoverage records which sensitivity comparisons actually ran. If featurePolicy is pending, nominate an offered family/feature experiment from observed baseline loading, marker and objective evidence before combining; missing nomination is not evidence of inapplicability. "
                "Conclude every comparisonCoverage axis using its exact completed baseline and alternatives, including candidates with better stability or marker coverage than your preference. "
                "In phase=sensitivity, use action=combine and choose each combinedSettings field from an observed candidate on that axis. A combination is a proposed hypothesis, not an observed result. "
                "Scarf will execute it and compare four resolutions on its exact graph before acceptance. Do not request already covered settings as experiments. "
                "comparisonConclusions must include quantitativeReason, biologicalReason and a short plainLanguageSummary for each axis. Explain how split or merged marker programs serve the stated objective, not just larger clusters or a single numerical maximum. "
                "comparisonAdvantages enumerates the exact measured advantages for each possible preference on every axis. For each row belonging to your preference, include a tradeoffs entry naming alternativeCandidateId, metric and interpretation. Scarf attaches the exact saved values; do not transcribe numbers into tradeoff fields. Address every listed advantage, including small differences; these require explanation, not automatic winner selection. "
                "tradeoffs contains only counterevidence: an alternative's measurement is strictly better than your preference. Copy only inventory rows matching the exact axis and preferredCandidateId; general comparisons favoring your preference and ties belong in quantitativeReason. "
                "If validation requests repair, fix every listed error in the same response using permittedTradeoffs for your current preferences. Preserve unaffected choices and explanations. Explicitly explain any changed preference, update its combinedSettings field when applicable, and use comparisonAdvantages for that new preference. "
                "Every numerical explanation must agree with the saved measurements, including direction and candidate identity. Unsupported biological interpretations must remain hypotheses or limitations; do not call protected sex-linked or cell-cycle programs artifacts merely because they appear in PCA loadings. "
                "For each selected cluster with empty topMarkerGenes, populationConcerns must name candidateId, clusterId, cited evidenceIds and explain whether it is a nonEssentialLimitation or unresolvedEssential. An unresolved essential population blocks acceptance; do not invent marker support. "
                "The action plainLanguageSummary should state the selected settings, what evidence changed the choice, and any unresolved population interpretations without workflow jargon. "
                "QC/capture retention, batch associations per PC and protected biological structure. "
                "Family dominance alone never proves nuisance; inclusion, exclusion and HVG bans need evidence and objective justification. "
                "If a specific concern warrants testing, choose exactly one offered experiment and state its expected improvement and "
                "what objective-relevant biology must be preserved. Family policy remains revisable when later evidence warrants it. "
                "PCA, HVG, k and resolution changes are separate comparisons. Do not equate a small sampled cluster with an artifact. "
                "For a safe design license, a matched Harmony experiment is required before acceptance, even when correction initially appears unnecessary. "
                "A notNeeded conclusion must follow the observed native/Harmony comparison, including batch/PC and biological evidence. "
                "Keep the native representation when the comparison supports it; evaluating Harmony does not require selecting it. Unsafe or unknown design is never authorization. "
                "Only accepting a Harmony representation requires a matched Harmony gate; a native representation does not require one. "
                "When correctionLicense is unsafeConfounded, do not infer correction necessity from batch mixing or PCA association. "
                "Use notApplicable for the prohibited correction, retain the confounding limitation, and assess whether native descriptive population discovery satisfies the objective. "
                "If essential effects remain inseparable, defer. Never request a different parameter change as a proxy for unavailable Harmony. "
                "Screening estimates do not prove full-cohort transfer. "
                "Findings describe observed results. The experimentId names the exact next operation and expectedImprovement predicts only that operation's effect. "
                "Already evaluated settings are existing alternatives, not new experiments; use the supplied matchedComparisons to avoid mixing PCA effects with resolution effects. "
                "For example, compare 10 versus 21 PCs at the same resolution rather than quoting the stability of another resolution. "
                "A high scaled cLISI is local purity of the supplied label, not proof that a clinical phenotype or all cell types are preserved. "
                "Graph connectivity is within-label connectivity. Low library mixing and high PCA association can reflect donor biology or cell composition; neither proves technical causality. "
                "A QC association is correlation. Check featureEvidence for actual selected genes: marker-family enrichment cannot show that an excluded family drives PCA. "
                "More retained cells, balanced group counts, or cross-unit support alone do not prove healthy cells or biological preservation. "
                "Check proposed cell identities against the tissue context. Unexpected marker programs require capture/donor and provenance investigation; do not declare them ordinary tissue populations or assert contamination without evidence. "
                "Detailed populationSupport is supplied for currentCandidateId during screening and every final validation candidate; do not claim to have compared unprovided distributions for alternatives. Inspect its capture/donor distribution and missing metadata. Broad support does not prove a biological identity; concentration alone does not prove contamination. "
                "Unsupported design comparisons establish neither association nor absence; keep their unresolved requirements visible. "
                "Previous actions are history, not scientific authority. Reassess their claims against the exact current evidence. "
                "Request enlarge when sample evidence is insufficient; on full cells there is one targeted repair, then defer. "
                "Copy evidence IDs exactly from availableEvidenceIds, including the selected candidate's anchor "
                "or one of its supplied diagnostic evidence IDs. Do not invent IDs, suffixes or substitute artifact IDs. "
                "Never accept merely because work limits are exhausted. "
            )
            prompt += (
                "The supplied images are available for visual assessment. Cite actual image evidence IDs and connect the observed plots to the numerical evidence."
                if mode == "visual"
                else "Visual inspection is unavailable: no images were supplied. Do not claim to have seen, inspected or compared plots or images, and do not cite image IDs. Use qualitativeFindings to interpret the reported marker identities, PCA loading genes, feature families and structured diagnostic tables. State limitations and defer if the supplied evidence cannot resolve an essential question."
            )
            serialized_evidence = json.dumps(
                evidence, sort_keys=True, separators=(",", ":")
            )
            logger.debug(
                f"Analysis assessment payload: {len(serialized_evidence.encode('utf-8')):,} "
                f"text bytes, {len(candidates)} candidates, {len(images)} images; "
                "all saved scientific evidence retained"
            )
            try:
                result = run_agent_sync(
                    model=self.owner.model,
                    output_type=_assessment_output_type(
                        [item.candidateId for item in candidates],
                        list(experiments),
                        scope=scope,
                        phase=evidence["comparisonCoverage"]["phase"],
                    ),
                    system_prompt=prompt,
                    user_prompt=build_visual_evidence_prompt(
                        serialized_evidence, images
                    )
                    if mode == "visual"
                    else serialized_evidence,
                    config=self.request.config.agentRunConfig.model_copy(
                        update={
                            "retries": min(
                                self.request.config.agentRunConfig.retries, 2
                            ),
                            "requestLimit": min(
                                self.request.config.agentRunConfig.requestLimit, 3
                            ),
                        }
                    ),
                    name=f"rna_{scope}_assessment",
                    output_validator=validate,
                    on_attempt=journal.model_attempt_callback(
                        self.store,
                        self.prefix,
                        self.workflow.workflowRunId,
                        key,
                        evidence,
                    ),
                )
            except ImageInputUnsupportedError:
                if mode != "visual":
                    raise
                journal.save_checkpoint(
                    self.store,
                    self.prefix,
                    self.workflow.workflowRunId,
                    capability_key,
                    inputs=capability_inputs,
                    outputs={
                        "evidenceMode": "structured",
                        "reason": "providerRejectedImageInput",
                    },
                )
                logger.info(
                    "Analysis assessment: model rejected images; continuing with structured marker, PCA and diagnostic evidence."
                )
                return self.review(scope, review_index, selected, coverage)
            action = validate(result.output)
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            key,
            inputs=evidence,
            outputs={"action": action.model_dump(mode="json")},
        )
        self.history.append(
            {
                "scope": scope,
                "evidenceMode": mode,
                "visualInspection": visual_inspection,
                "review": action.model_dump(mode="json"),
                "imageHashes": image_hashes,
                "checkpointKey": key,
                "checkpointSha256": hashlib.sha256(
                    record_io.canonical_json_bytes(
                        {
                            "inputs": evidence,
                            "outputs": {"action": action.model_dump(mode="json")},
                        }
                    )
                ).hexdigest(),
            }
        )
        self.last_action = action
        operation = (
            f"experiment {action.experimentId} from {action.selectedCandidateId}"
            if action.action == "experiment"
            else f"{action.action} {action.selectedCandidateId}"
        )
        logger.info(f"Analysis assessment: {action.plainLanguageSummary}")
        logger.debug(f"Analysis assessment ({operation}): {action.rationale}")
        return action

    def assess_scope(
        self,
        scope: str,
        cells: ArtifactRef,
        initial: RnaSetting | None,
    ) -> tuple[
        Literal["accept", "enlarge", "defer"], ParameterCandidateEvaluation | None
    ]:
        coverage_inputs = {
            "parentCells": self.cells.to_dict(),
            "cells": cells.to_dict(),
            "columns": self.coverage_columns,
            "protectedCombinations": self.study.protectedCombinations,
            "columnKinds": self.study.columnKinds,
            "provenance": self.provenance,
        }
        measured = journal.load_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            f"{self.checkpoint_scope(scope)}/coverage",
            inputs=coverage_inputs,
        )
        if measured is None:
            coverage, insufficient = screening_coverage(
                self.store,
                self.cells,
                cells,
                self.coverage_columns,
                self.study.protectedCombinations,
            )
            measured = journal.save_checkpoint(
                self.store,
                self.prefix,
                self.workflow.workflowRunId,
                f"{self.checkpoint_scope(scope)}/coverage",
                inputs=coverage_inputs,
                outputs={"coverage": coverage, "insufficient": insufficient},
            )
        coverage, insufficient = measured["coverage"], measured["insufficient"]
        logger.info(
            f"Analysis population: {coverage['screeningCells']:,} of {coverage.get('populationCells', self.handoff.nCells):,} retained cells; "
            + ("all retained cells." if cells == self.cells else "screening subset.")
        )
        self.scope_sizes[scope] = coverage["screeningCells"]
        self.history.append(
            {"scope": scope, "coverage": coverage, "coverageConcerns": insufficient}
        )
        if cells != self.cells and insufficient:
            return "enlarge", None
        if initial is None:
            if scope == "full":
                raise CandidateBudgetExceeded(
                    "No supported screening proposal is available for targeted full-cohort recovery. "
                    "The mandatory discovery comparisons cannot be replaced by a full-cohort baseline; "
                    "saved results and unmet coverage requirements are preserved."
                )
            baseline = self.baseline().model_copy(
                update={
                    "parameters": self.baseline().parameters.model_copy(
                        update={
                            "dimensions": min(
                                self.baseline().parameters.dimensions,
                                coverage["screeningCells"] - 1,
                            ),
                            "neighborsK": min(
                                self.baseline().parameters.neighborsK,
                                coverage["screeningCells"] - 1,
                            ),
                        }
                    )
                }
            )
            selected = self._resolution_panel(scope, cells, baseline)
            # The first panel describes defaults, not an accepted combined recipe.
            self.resolution_candidates.pop(scope, None)
            self._sensitivity_panel(scope, cells, selected)
        else:
            requested = initial
            if self.study.correctionLicense == "safe":
                initial = initial.model_copy(
                    update={
                        "parameters": initial.parameters.model_copy(
                            update={"useHarmony": True}
                        )
                    }
                )
            selected = (
                self._resolution_panel(scope, cells, initial)
                if scope == "full" and self.recovery_scope is not None
                else self.execute_matched(scope, cells, initial)
            )
            if scope == "full" and self.recovery_scope is not None:
                self.combined_candidates[scope] = selected.candidateId
            if not requested.parameters.useHarmony and initial.parameters.useHarmony:
                selected = self.execute(scope, cells, requested)
        limit = (
            self.request.config.maxFullPartitions
            if scope == "full"
            else self.request.config.maxScreeningEvaluations
        )
        for review_index in range(limit + 1):
            action = self.review(scope, review_index, selected, coverage)
            selected = next(
                item
                for item in self.evaluations[scope]
                if item.candidateId == action.selectedCandidateId
            )
            if action.action in {"accept", "enlarge", "defer"}:
                if action.action == "enlarge" and cells == self.cells:
                    self.history.append(
                        {
                            "scope": scope,
                            "reason": "All retained cells were already assessed; more sampling cannot supply the unresolved evidence.",
                        }
                    )
                    return "defer", selected
                if (
                    action.action == "accept"
                    and cells != self.cells
                    and selected.metrics.minClusterCells is not None
                    and selected.metrics.minClusterCells < 20
                ):
                    self.history.append(
                        {
                            "scope": scope,
                            "reason": "The selected sampled partition contains a small population requiring more cells for assessment; all its cells are retained.",
                        }
                    )
                    return "enlarge", selected
                if action.action == "accept" and scope != "full":
                    self.discovery_scope = scope
                return action.action, selected
            if action.action == "combine":
                setting = self._combined_setting(scope, review_index, action, cells)
                if self.study.correctionLicense == "safe":
                    setting = setting.model_copy(
                        update={
                            "parameters": setting.parameters.model_copy(
                                update={"useHarmony": True}
                            )
                        }
                    )
                else:
                    selected = self.execute(scope, cells, setting)
                    setting = self.settings[selected.candidateId]
                selected = self._resolution_panel(scope, cells, setting)
                self.combined_candidates[scope] = selected.candidateId
                continue
            assert action.experimentId is not None
            experiment = self.experiments(selected)[action.experimentId]
            if (
                scope == "full"
                and (
                    self.discovery_scope is not None
                    or scope in self.combined_candidates
                )
                and experiment["parameter"] != "useHarmony"
            ):
                self.budget.admit_repair(
                    ArtifactReferenceModel.from_artifact_ref(cells).model_dump(
                        mode="json"
                    ),
                    self.settings[selected.candidateId].model_dump(mode="json"),
                    experiment,
                )
                self.full_repairs = len(self.budget.repairs)
            if experiment["parameter"] in {
                "hvgRanking",
                "hvgCount",
                "includeFamily",
                "excludeFamily",
                "includeFeature",
                "excludeFeature",
            }:
                setting = self._prepared_setting(
                    scope,
                    f"review{review_index}/feature_experiment",
                    selected,
                    experiment,
                    cells,
                )
            else:
                setting = self.apply_experiment(selected, experiment)
            prior = selected
            pending_policy = next(
                (
                    row
                    for row in self.comparison_rows.get(scope, [])
                    if row["axis"] == "featurePolicy" and row["status"] == "pending"
                ),
                None,
            )
            policy_experiment = experiment["parameter"] in {
                "includeFamily",
                "excludeFamily",
                "includeFeature",
                "excludeFeature",
            }
            if pending_policy is not None and policy_experiment:
                if prior.candidateId != pending_policy["baselineCandidateId"]:
                    raise ValueError(
                        "The required policy comparison must retain the exact default baseline"
                    )
                prior_setting = self.settings[prior.candidateId]
                if setting.hvgCount != prior_setting.hvgCount or np.array_equal(
                    self.store.load_artifact(artifact_model_to_ref(setting.features))[
                        "values"
                    ][:],
                    self.store.load_artifact(
                        artifact_model_to_ref(prior_setting.features)
                    )["values"][:],
                ):
                    pending_policy.update(
                        status="notApplicable",
                        reason="The nominated policy cannot change selected genes while retaining the fixed baseline HVG count.",
                        observedProof={
                            "baselineFeatures": prior_setting.features.model_dump(
                                mode="json"
                            ),
                            **(
                                {
                                    "kind": "insufficientEligibleGenes",
                                    "eligibleFeatureCount": setting.hvgCount,
                                }
                                if setting.hvgCount != prior_setting.hvgCount
                                else {
                                    "kind": "identicalSelectedGenes",
                                    "verifiedEqualMasks": True,
                                    "alternativeSetting": setting.model_dump(
                                        mode="json"
                                    ),
                                }
                            ),
                        },
                    )
                    continue
            selected = (
                self._resolution_panel(scope, cells, setting)
                if scope == "full"
                and self.recovery_scope is not None
                and experiment["parameter"] != "leidenResolution"
                else self.execute_matched(scope, cells, setting)
            )
            if (
                scope == "full"
                and self.recovery_scope is not None
                and experiment["parameter"] != "leidenResolution"
            ):
                self.combined_candidates[scope] = selected.candidateId
            if pending_policy is not None and policy_experiment:
                pending_policy.update(
                    status="completed",
                    alternativeCandidateId=selected.candidateId,
                    reason="Baseline loading, marker and objective evidence nominated this exact policy comparison.",
                )
            if (
                scope == "full"
                and (
                    self.discovery_scope is not None
                    or scope in self.combined_candidates
                )
                and experiment["parameter"] != "useHarmony"
            ):
                self.full_repair = {
                    "baselineCandidateId": prior.candidateId,
                    "selectedCandidateId": selected.candidateId,
                    "experimentId": action.experimentId,
                }
            if scope != "full" and scope in self.combined_candidates:
                selected = self._resolution_panel(
                    scope, cells, self.settings[selected.candidateId]
                )
                self.combined_candidates[scope] = selected.candidateId
        return "defer", selected

    def run(self) -> tuple[ParameterTuningReport, dict[str, Any]]:
        with journal.diagnostic_attempt(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            {"provenance": self.provenance, "cells": self.cells.to_dict()},
        ) as counts:
            self.diagnostic_counts = counts
            selected: ParameterCandidateEvaluation | None = None
            final_status = "defer"
            reason = "Required scientific evidence remains unresolved."
            try:
                initial: RnaSetting | None = None
                recovery_candidate: ParameterCandidateEvaluation | None = None
                previous_sample = None
                sizes = screening_sizes(self.handoff.nCells, self.request.config)
                journal.save_checkpoint(
                    self.store,
                    self.prefix,
                    self.workflow.workflowRunId,
                    "parameter_tuning/sampling_policy",
                    inputs={
                        "cells": self.cells.to_dict(),
                        "config": self.request.config.model_dump(mode="json"),
                    },
                    outputs={
                        "policy": "fixed"
                        if self.request.config.screeningCells is not None
                        else "boundedTenPercent",
                        "retainedCells": self.handoff.nCells,
                        "populationSizes": list(sizes),
                        "seed": self.request.config.randomSeed,
                    },
                )
                for index, size in enumerate(sizes):
                    sample = uniform_screening_selection(
                        self.store,
                        self.cells,
                        size=size,
                        seed=self.request.config.randomSeed,
                    )
                    if sample == previous_sample:
                        break
                    previous_sample = sample
                    status, screened = self.assess_scope(f"sample{index}", sample, None)
                    if (
                        status == "enlarge"
                        and screened is not None
                        and f"sample{index}" in self.combined_candidates
                    ):
                        recovery_candidate = screened
                        self.recovery_scope = f"sample{index}"
                    if status == "accept" and screened is not None:
                        initial = self.settings[screened.candidateId]
                        self.discovery_scope = f"sample{index}"
                        self.recovery_scope = None
                        break
                    if status == "defer":
                        return self.report(
                            None,
                            self.last_action.rationale
                            if self.last_action is not None
                            else reason,
                        ), self.summary()
                    if sample == self.cells:
                        return self.report(
                            None,
                            "All retained cells were assessed, but the required scientific evidence remains unresolved; another sample cannot resolve this concern.",
                        ), self.summary()
                if initial is None:
                    if recovery_candidate is None:
                        return self.report(
                            None,
                            "Screening coverage remains inadequate and no measured combined proposal is available for targeted recovery within the full-cohort allowance. Saved evidence is preserved; essential comparisons remain unresolved.",
                        ), self.summary()
                    initial = self.settings[recovery_candidate.candidateId]
                    logger.info(
                        "Screening evidence requires targeted full-cohort validation of the agent's proposed settings; previous comparisons remain available."
                    )
                final_status, selected = self.assess_scope("full", self.cells, initial)
                if final_status != "accept" and self.last_action is not None:
                    reason = self.last_action.rationale
            except CandidateBudgetExceeded as exc:
                reason = f"Required comparison coverage is incomplete within the configured work limits: {exc}"
            if final_status != "accept":
                selected = None
            return self.report(selected, reason), self.summary()

    def summary(self) -> dict[str, Any]:
        budget = self.budget.summary()
        diagnostic_artifacts: dict[str, set[ArtifactRef]] = {
            "pcaDiagnostics": set(),
            "stabilityClusters": set(),
            "markerTable": set(),
            "doubletScore": set(),
        }
        subsample_evaluations = {}
        for scope, evaluations in self.evaluations.items():
            subsample_evaluations[scope] = len(
                {
                    evaluation.candidateId
                    for evaluation in evaluations
                    if evaluation.status == "done"
                    and evaluation.metrics.subsampleStability is not None
                }
            )
            for evaluation in evaluations:
                if evaluation.status != "done":
                    continue
                for name, artifact in evaluation.artifacts.items():
                    diagnostic = (
                        "pcaDiagnostics"
                        if name == "representationDiagnostic"
                        else name.split(":", 1)[0]
                    )
                    if diagnostic in diagnostic_artifacts:
                        diagnostic_artifacts[diagnostic].add(
                            artifact_model_to_ref(artifact)
                        )
        diagnostic_counts = {
            name: len(refs) for name, refs in diagnostic_artifacts.items()
        }
        for scope, counts in budget["scopes"].items():
            if counts["reserved"]["partitions"]:
                label = {
                    "sample0": "screening sample 1",
                    "sample1": "screening sample 2",
                    "full": "full cohort",
                }[scope]
                completed, reserved = counts["completed"], counts["reserved"]
                logger.info(
                    f"Tuning {label}: {completed['graphs']}/{reserved['graphs']} graphs "
                    f"and {completed['partitions']}/{reserved['partitions']} partitions "
                    "completed/reserved."
                )
        limits = budget["limits"]
        logger.info(
            f"Tuning limits: {limits['perScreen']} partitions per screen, "
            f"{limits['totalScreens']} across screens; full cohort "
            f"{limits['fullGraphs']} graphs and {limits['fullPartitions']} partitions."
        )
        logger.info(
            f"Diagnostic evidence: {diagnostic_counts['pcaDiagnostics']} PCA summaries, "
            f"{diagnostic_counts['stabilityClusters']} alternate-seed partitions, "
            f"{diagnostic_counts['markerTable']} marker tables, "
            f"{diagnostic_counts['doubletScore']} doublet scores; "
            f"{sum(subsample_evaluations.values())} subsample-stability evaluations. "
            "Saved evidence may be reused; these are not computation counts."
        )
        for operation, counts in sorted(self.diagnostic_counts.items()):
            logger.debug(
                f"Diagnostic work this invocation, {operation}: "
                f"{counts['completed']}/{counts['attempted']} calls completed/attempted, "
                f"{counts['failed']} failed; {counts['cacheHits']} metric cache hits, "
                f"{counts['restored']} saved evidence restores, "
                f"{counts['artifactReuses']} confirmed artifact reuses."
            )
        return {
            "history": self.history,
            "budget": budget,
            "diagnosticEvidence": {
                "uniqueArtifacts": diagnostic_counts,
                "subsampleStabilityEvaluations": subsample_evaluations,
                "interpretation": (
                    "Counts describe saved diagnostic evidence used by these evaluations "
                    "and may include reused artifacts or metrics, not new computations."
                ),
            },
            "fullRepairs": self.full_repairs,
            "diagnosticOperations": {
                "scope": "thisInvocation",
                "operations": self.diagnostic_counts,
                "interpretation": (
                    "Observed agent calls and explicit reuse events, not inferred core "
                    "numerical rebuilds or simulated-doublet writes. Earlier invocations "
                    "are recorded separately in the stage journal; absent historical "
                    "operation records mean unknown counts."
                ),
            },
        }

    def report(
        self, selected: ParameterCandidateEvaluation | None, reason: str
    ) -> ParameterTuningReport:
        evaluations = self.evaluations["full"]
        common: dict[str, Any] = {
            "fromAssay": self.handoff.assay,
            "cellSelection": self.handoff.cellSelection,
            "evaluations": evaluations,
            "totalCandidates": len(evaluations),
            "markerAssay": self.handoff.assay,
            "limitations": [
                "Screening comparisons describe their exact sampled cells; final artifacts and validation use the full QC-retained cohort.",
                *self.study.limitations,
                *unsupported_comparison_limitations(self.design_comparisons),
                *(
                    [_STRUCTURED_VISUAL_LIMITATION]
                    if any(
                        row.get("evidenceMode") == "structured" for row in self.history
                    )
                    else []
                ),
                *(
                    f"Unsupported matched protection: {column}."
                    for column in self.study.unsupportedProtection
                ),
                *(str(row["reason"]) for row in self.history if "reason" in row),
                *(
                    "Screening coverage concern: " + concern
                    for row in self.history
                    for concern in row.get("coverageConcerns", [])
                ),
            ],
        }
        common["limitations"] = list(dict.fromkeys(common["limitations"]))
        if selected is None:
            return ParameterTuningReport(
                **common,
                status="needsInput",
                rationale=reason,
                stopReason=reason,
                needsInput=ParameterTuningNeedsInput(question=reason),
            )
        assert self.last_action is not None
        common["limitations"].extend(
            f"Population {row.clusterId}: {row.explanation}"
            for row in self.last_action.populationConcerns
            if row.candidateId == selected.candidateId
        )
        report = ParameterTuningReport(
            **common,
            status="done",
            recommendedCandidateId=selected.candidateId,
            selectedArtifacts=selected.artifacts,
            confidence="medium",
            rationale=self.last_action.rationale,
            evidenceIds=self.last_action.evidenceIds,
            stopReason="The objective-driven assessment accepted the full-cohort evidence.",
            recommendedByAssay={self.handoff.assay: selected.candidateId},
        )
        return finalize_parameter_tuning_selection(
            report, marker_assay=self.handoff.assay, native_assay=self.handoff.assay
        )
