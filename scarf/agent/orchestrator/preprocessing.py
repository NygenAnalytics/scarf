"""Preprocessing planning and execution stages."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...assay import RNAassay
from ...datastore.datastore import DataStore
from ...datastore.summary import AssaySummary
from ...metadata.selection import NamedCellArtifact
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_mask
from ...storage.types import as_zarr_array
from ...utils.logging import logger
from .. import record_io
from ..cell_quality.execution import execute_auto_cell_qc, execute_registered_cell_qc
from ..cell_quality.profiles import CellQualityProfile, cell_qc_policy
from ..data_enrichment.contracts import (
    AssayFeatureInspection,
    DataEnrichmentReport,
    FeatureSelectionPolicy,
)
from ..decisions.kernel import DecisionEvidence, DecisionSelection, EvidenceBundle
from ..decisions.rna import (
    CellQualityExecutorPayload,
    QcGroupingExecutorPayload,
    build_cell_quality_decision,
    build_qc_grouping_decision,
    require_option_evidence,
)
from ..experimental_context.contracts import (
    CellQcPlan,
    CellQcProfileEvidence,
    ExperimentalContextResult,
)
from ..experimental_context.study import StudyContract
from ..parameter_tuning.execution import (
    candidate_metric_cache,
)
from ..types import ArtifactReferenceModel
from . import journal
from .decisions import DecisionStagesMixin
from .models import (
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    PreprocessedAssayHandoff,
    WorkflowIdentity,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    WorkflowStageName,
    artifact_model_to_ref,
)
from .rna import (
    selected_store_rna_assay,
    validate_rna_context,
    validate_rna_handoffs,
    validate_rna_plan,
)


class _DecisionNeedsInput(RuntimeError):
    def __init__(
        self,
        question: WorkflowQuestion,
        checkpoint_sha256: str,
    ) -> None:
        super().__init__("A registered RNA decision requires human input")
        self.question = question
        self.checkpointSha256 = checkpoint_sha256


class PreprocessingStagesMixin(DecisionStagesMixin):
    """Stages that plan and execute modality-specific preprocessing."""

    @staticmethod
    def _cell_qc_stage_artifacts(
        plan: CellQcPlan,
    ) -> dict[str, ArtifactReferenceModel]:
        artifacts = {
            f"cellQcMetric:{source.name}": source.artifact
            for source in plan.artifactMetrics
        }
        if len(artifacts) != len(plan.artifactMetrics):
            raise ValueError("Cell-QC artifact metric names must be unique")
        if plan.sampleArtifact is not None:
            if plan.sampleArtifact.name in {
                source.name for source in plan.artifactMetrics
            }:
                raise ValueError("Cell-QC sample artifact name collides with a metric")
            key = f"cellQcSample:{plan.sampleArtifact.name}"
            artifacts[key] = plan.sampleArtifact.artifact
        return artifacts

    @classmethod
    def _cell_qc_candidate_artifacts(
        cls,
        profiles: Sequence[CellQcProfileEvidence],
    ) -> dict[str, ArtifactReferenceModel]:
        artifacts: dict[str, ArtifactReferenceModel] = {}
        for profile in profiles:
            plan = CellQcPlan(
                action=profile.action,
                registeredProfile=profile.registeredProfile,
                profileId=profile.profileId,
                driverAssay=profile.driverAssay,
                driverAssayType=profile.driverAssayType,
                sampleColumn=profile.sampleColumn,
                sampleArtifact=profile.sampleArtifact,
                attributes=profile.attributes,
                artifactMetrics=profile.artifactMetrics,
                evidenceIds=[profile.evidenceId],
            )
            for key, artifact in cls._cell_qc_stage_artifacts(plan).items():
                existing = artifacts.get(key)
                if existing is not None and existing != artifact:
                    raise ValueError(
                        f"Cell-QC candidate artifact key {key!r} is ambiguous"
                    )
                artifacts[key] = artifact
        return artifacts

    @staticmethod
    def _decision_evidence_bundle(
        decision_id: str,
        evidence: list[DecisionEvidence],
    ) -> EvidenceBundle:
        digest = hashlib.sha256(
            record_io.canonical_json_bytes(
                [item.model_dump(mode="json") for item in evidence]
            )
        ).hexdigest()
        return EvidenceBundle(
            bundleId=f"bundle:{decision_id}:{digest[:24]}",
            decisionId=decision_id,
            evidence=evidence,
        ).with_content_sha256()

    @staticmethod
    def _profile_is_safe(profile: CellQcProfileEvidence) -> bool:
        if profile.action == "sampleMad" and (
            profile.sampleColumn != profile.captureColumn
            or profile.sampleArtifact != profile.captureArtifact
            or (profile.captureColumn is None and profile.captureArtifact is None)
        ):
            return False
        if profile.retainedCells == 0 or profile.unsafeRetentionGroups:
            return False
        if (
            profile.action == "sampleMad"
            or profile.registeredProfile
            in {
                "captureMad5",
                "captureMad3Sensitivity",
            }
        ) and profile.failedCaptureCandidates:
            return False
        if profile.registeredProfile == "pooledReferenceMad5" and set(
            profile.failedCaptureCandidates
        ).intersection(profile.parameters.get("pooledReferenceCaptures", [])):
            return False
        return True

    @staticmethod
    def _qc_decision_evidence(
        profiles: Sequence[CellQcProfileEvidence],
    ) -> dict[str, Any]:
        """Keep exact QC measurements while sharing repeated source/design evidence."""
        shared: dict[str, Any] = {}

        def reference(value: Any) -> str:
            digest = hashlib.sha256(record_io.canonical_json_bytes(value)).hexdigest()
            shared[digest] = value
            return digest

        policies = []
        for profile in profiles:
            policy = profile.model_dump(mode="json")
            for field in ("metricSources", "sourceConcordance"):
                policy[field + "Ref"] = reference(policy.pop(field))
            parameters = policy["parameters"]
            if "captureComparisons" in parameters:
                parameters["captureComparisonsRef"] = reference(
                    parameters.pop("captureComparisons")
                )
            for duplicate, canonical in (
                ("resolvedBounds", "resolvedBounds"),
                ("captureSizes", "activeCellsByCapture"),
            ):
                if (
                    duplicate in parameters
                    and parameters[duplicate] == policy[canonical]
                ):
                    parameters.pop(duplicate)
            for capture in policy["captureFailureEvidence"]:
                capture["conditionAndUnitSafetyRef"] = reference(
                    capture.pop("conditionAndUnitSafety")
                )
            policies.append(policy)
        return {
            "policies": policies,
            "sharedMeasurements": shared,
            "interpretation": (
                "Each Ref identifies the exact record in sharedMeasurements. "
                "Thresholds, distributions, flags and protected-group retention are "
                "measured policy evidence; short evidence summaries do not replace them."
            ),
        }

    @staticmethod
    def _profile_evidence(
        profile: CellQcProfileEvidence,
        retained_reference: CellQcProfileEvidence | None = None,
    ) -> DecisionEvidence:
        def retention_range(values: Sequence[int]) -> str:
            if not values:
                return "no groups"
            ordered = sorted(int(value) for value in values)
            return (
                f"{len(ordered)} groups, min/median/max="
                f"{ordered[0]}/{float(np.median(ordered)):g}/{ordered[-1]}"
            )

        capture_summary = retention_range(list(profile.sampleRetainedCells.values()))
        capture_fractions = [
            row.retainedFraction for row in profile.captureFailureEvidence
        ]
        capture_fraction_summary = (
            f"min/median/max={min(capture_fractions):.1%}/"
            f"{float(np.median(capture_fractions)):.1%}/{max(capture_fractions):.1%}"
            if capture_fractions
            else "not available"
        )
        active_by_column = (
            retained_reference.retainedCellsByColumn
            if retained_reference is not None
            and retained_reference.retainedCells
            == retained_reference.activeCells
            == profile.activeCells
            else {}
        )

        def design_retention(column: str, groups: Mapping[str, int]) -> str:
            active = active_by_column.get(column, {})
            if not groups or any(not active.get(group) for group in groups):
                return (
                    retention_range(list(groups.values())) + "; fractions unavailable"
                )
            if len(groups) <= 4:
                return ", ".join(
                    f"{group}={retained}/{active[group]} ({retained / active[group]:.1%})"
                    for group, retained in sorted(groups.items())
                )
            fractions = [retained / active[group] for group, retained in groups.items()]
            return (
                f"{retention_range(list(groups.values()))}, retained fractions "
                f"min/median/max={min(fractions):.1%}/{float(np.median(fractions)):.1%}/"
                f"{max(fractions):.1%}"
            )

        design_summary = "; ".join(
            f"{column}: {design_retention(column, groups)}"
            for column, groups in sorted(profile.retainedCellsByColumn.items())
        )
        summary = (
            f"{cell_qc_policy(profile.action, profile.registeredProfile) or profile.action} retains "
            f"{profile.retainedCells}/{profile.activeCells} active cells "
            f"({profile.retainedCells / profile.activeCells:.1%}); "
            if profile.activeCells
            else "No active cells; "
        ) + (
            f"metric flags={profile.metricFlaggedCells or 'not available'} "
            "(flags may overlap; high counts/features can be retained); "
            f"capture retention={capture_summary}; capture retained fractions="
            f"{capture_fraction_summary}; design retention="
            f"{design_summary or 'no groups'}; failed capture candidates="
            f"{profile.failedCaptureCandidates}; unsafe retention groups="
            f"{profile.unsafeRetentionGroups}; limitations={profile.notes[:4]}."
        )
        if len(summary) > 2_000:
            summary = f"{summary[:1_997].rstrip()}..."
        return DecisionEvidence(
            evidenceId=profile.evidenceId,
            evidenceClass="qualityControl",
            summary=summary,
            artifactReferences=[
                *[source.artifact for source in profile.artifactMetrics],
                *(
                    [profile.sampleArtifact.artifact]
                    if profile.sampleArtifact is not None
                    else []
                ),
            ],
        )

    def _resolve_qc_grouping_decision(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        experimental: ExperimentalContextResult,
        study_contract: StudyContract,
        answers: Mapping[str, Any],
    ) -> tuple[QcGroupingExecutorPayload, str]:
        profiles = [
            profile
            for profile in experimental.qcProfiles
            if cell_qc_policy(profile.action, profile.registeredProfile) is not None
        ]
        if not profiles:
            raise ValueError("RNA decision workflow requires executable QC evidence")
        safe_profiles = {
            cell_qc_policy(profile.action, profile.registeredProfile): profile
            for profile in profiles
            if self._profile_is_safe(profile)
        }
        capture_eligible = bool(
            study_contract.physicalCaptureColumn is not None
            and any(
                policy in safe_profiles for policy in ("coreSampleMad3", "captureMad5")
            )
        )
        pooled_eligible = bool(
            capture_eligible and "pooledReferenceMad5" in safe_profiles
        )
        design_id = "evidence:qcGrouping:studyContract"
        retained_reference = safe_profiles.get("retainWithFlags")
        evidence = [
            DecisionEvidence(
                evidenceId=design_id,
                evidenceClass="design",
                summary=(
                    "The validated physical capture is "
                    f"{study_contract.physicalCaptureColumn!r}; independent units="
                    f"{study_contract.independentUnitColumns}; conditions="
                    f"{study_contract.conditionColumns}. "
                    "Within-capture QC estimates technical quality boundaries; captures "
                    "need not be independent biological units or healthy references. "
                    "Healthy-reference provenance is required only for a pooled reference. "
                    "Compare global and capture five-MAD profiles when both are supplied; "
                    "different cutoff methods cannot isolate the effect of grouping."
                ),
            )
        ]
        mode_profile: dict[str, CellQcProfileEvidence] = {}
        matched_mad = "globalMad5" in safe_profiles and "captureMad5" in safe_profiles
        global_profile = (
            safe_profiles["globalMad5"]
            if matched_mad
            else safe_profiles.get("coreGlobalGaussian")
            or safe_profiles.get("globalMad5")
            or safe_profiles.get("retainWithFlags")
        )
        if global_profile is None:
            raise ValueError("No safe global or retain-only QC profile is available")
        mode_profile["qcGrouping:global"] = global_profile
        baseline = next(
            (profile for profile in profiles if profile.action == "globalGaussian"),
            None,
        )
        if baseline is not None and baseline != global_profile:
            evidence.append(self._profile_evidence(baseline, retained_reference))
        evidence.append(self._profile_evidence(global_profile, retained_reference))
        if capture_eligible:
            capture_profile = (
                safe_profiles["captureMad5"]
                if matched_mad
                else safe_profiles.get("coreSampleMad3") or safe_profiles["captureMad5"]
            )
            mode_profile["qcGrouping:physicalCapture"] = capture_profile
            evidence.append(self._profile_evidence(capture_profile, retained_reference))
        if pooled_eligible:
            pooled_profile = safe_profiles["pooledReferenceMad5"]
            mode_profile["qcGrouping:pooledReference"] = pooled_profile
            evidence.append(self._profile_evidence(pooled_profile, retained_reference))
        bundle = self._decision_evidence_bundle("qcGrouping", evidence)
        definition = build_qc_grouping_decision(
            evidence_bundle_id=bundle.bundleId,
            physical_capture_eligible=capture_eligible,
            pooled_reference_eligible=pooled_eligible,
        )
        definition = require_option_evidence(
            definition,
            {
                option_id: [design_id, profile.evidenceId]
                for option_id, profile in mode_profile.items()
            },
        )
        selectable_ids = [
            option.optionId
            for option in definition.spec.options
            if option.status != "defer"
        ]
        rule_selection = (
            DecisionSelection(
                selectedOptionId=selectable_ids[0],
                evidenceIds=list(
                    definition.spec.option_by_id()[
                        selectable_ids[0]
                    ].requiredEvidenceIds
                ),
                rationale=(
                    "Use the only grouping mode licensed by the validated design "
                    "and retention evidence."
                ),
            )
            if len(selectable_ids) == 1
            else None
        )
        resolution = self._resolve_rna_decision(
            store,
            request_record,
            definition,
            bundle,
            answers,
            rule_selection=rule_selection,
            qc_evidence=self._qc_decision_evidence(profiles),
        )
        if resolution.compiled is None:
            raise _DecisionNeedsInput(
                self._pending_decision_question(resolution, definition),
                resolution.checkpointSha256,
            )
        payload = resolution.compiled.executorPayload
        if not isinstance(payload, QcGroupingExecutorPayload):
            raise TypeError("QC-grouping decision compiled an unexpected payload")
        return payload, resolution.checkpointSha256

    def _resolve_cell_quality_decision(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        experimental: ExperimentalContextResult,
        grouping: QcGroupingExecutorPayload,
        answers: Mapping[str, Any],
    ) -> tuple[CellQualityExecutorPayload, CellQcPlan, str]:
        all_profiles = [
            profile
            for profile in experimental.qcProfiles
            if cell_qc_policy(profile.action, profile.registeredProfile) is not None
        ]
        all_profiles.sort(key=lambda profile: profile.action != "globalGaussian")
        allowed_by_grouping: dict[str, set[CellQualityProfile]] = {
            "global": {"retainWithFlags", "coreGlobalGaussian", "globalMad5"},
            "physicalCapture": {"retainWithFlags", "coreSampleMad3", "captureMad5"},
            "pooledReference": {"retainWithFlags", "pooledReferenceMad5"},
        }
        allowed = allowed_by_grouping[grouping.groupingMode]
        profiles = [
            profile
            for profile in all_profiles
            if cell_qc_policy(profile.action, profile.registeredProfile) in allowed
            and self._profile_is_safe(profile)
        ]
        if not profiles:
            raise ValueError("QC grouping has no safe executable profile")
        profiles.sort(key=lambda profile: profile.action != "globalGaussian")
        retained_reference = next(
            (p for p in all_profiles if p.registeredProfile == "retainWithFlags"), None
        )
        evidence = [
            self._profile_evidence(profile, retained_reference)
            for profile in all_profiles
        ]

        bundle = self._decision_evidence_bundle("cellQuality", evidence)
        available_profiles = [
            policy
            for profile in profiles
            if (policy := cell_qc_policy(profile.action, profile.registeredProfile))
            is not None
        ]
        definition = build_cell_quality_decision(
            evidence_bundle_id=bundle.bundleId,
            available_profiles=available_profiles,
        )
        definition = require_option_evidence(
            definition,
            {
                f"cellQuality:{cell_qc_policy(profile.action, profile.registeredProfile)}": [
                    profile.evidenceId
                ]
                for profile in profiles
            },
        )
        resolution = self._resolve_rna_decision(
            store,
            request_record,
            definition,
            bundle,
            answers,
            qc_evidence=self._qc_decision_evidence(all_profiles),
        )
        if resolution.compiled is None:
            raise _DecisionNeedsInput(
                self._pending_decision_question(resolution, definition),
                resolution.checkpointSha256,
            )
        payload = resolution.compiled.executorPayload
        if not isinstance(payload, CellQualityExecutorPayload):
            raise TypeError("Cell-quality decision compiled an unexpected payload")
        selected = next(
            (
                profile
                for profile in profiles
                if cell_qc_policy(profile.action, profile.registeredProfile)
                == payload.profile
            ),
            None,
        )
        if selected is None or resolution.record is None:
            raise ValueError("Audited cell-quality option lacks its exact profile")
        plan = CellQcPlan(
            action=selected.action,
            registeredProfile=selected.registeredProfile,
            profileId=selected.profileId,
            driverAssay=selected.driverAssay,
            driverAssayType=selected.driverAssayType,
            sampleColumn=selected.sampleColumn,
            sampleArtifact=selected.sampleArtifact,
            attributes=selected.attributes,
            artifactMetrics=selected.artifactMetrics,
            rationale=resolution.record.rationale,
            evidenceIds=list(resolution.record.evidenceIds),
        )
        return payload, plan, resolution.checkpointSha256

    @staticmethod
    def _plan_with_selected_hvg_counts(
        plan: AutomatedPreprocessingPlan,
        handoffs: Sequence[PreprocessedAssayHandoff],
    ) -> AutomatedPreprocessingPlan:
        selected_counts = {handoff.assay: handoff.nFeatures for handoff in handoffs}
        assays = [
            assay.model_copy(
                update={
                    "featureParameters": {
                        **assay.featureParameters,
                        "topN": selected_counts[assay.assay],
                    }
                }
            )
            if assay.featureMethod == "hvg" and assay.assay in selected_counts
            else assay
            for assay in plan.assays
        ]
        updated = plan.model_copy(update={"assays": assays, "planChecksum": ""})
        checksum = hashlib.sha256(
            record_io.canonical_json_bytes(
                updated.model_dump(mode="json", exclude={"planChecksum"})
            )
        ).hexdigest()
        return updated.model_copy(update={"planChecksum": checksum})

    def preprocessing_plan_stage(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        enrichment: DataEnrichmentReport,
        experimental: ExperimentalContextResult,
        ingest_outcome: WorkflowStageAttempt,
        study_contract: StudyContract,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, AutomatedPreprocessingPlan]:
        selected = selected_store_rna_assay(store, request_record.request)
        validate_rna_context(experimental, selected)
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "preprocessing_plan",
            request_record,
            parents,
        )
        if existing is not None:
            logger.debug(
                f"Workflow {workflow.workflowRunId}: reusing preprocessing plan"
            )
            cached_plan = AutomatedPreprocessingPlan.model_validate(
                existing.outputs["preprocessingPlan"]
            )
            validate_rna_plan(cached_plan, selected)
            return existing, cached_plan
        if experimental.cellSelection is None:
            raise ValueError("Experimental Context lacks an exact cell selection")
        cell_qc_artifacts = self._cell_qc_candidate_artifacts(experimental.qcProfiles)
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "preprocessing_plan",
            request_record,
            parents,
            inputs={
                "cellSelection": experimental.cellSelection.model_dump(mode="json"),
                "decisionPolicy": "evidenceBoundedAutomaticExecution",
                "studyContract": study_contract.model_dump(mode="json"),
            },
            resume_record=resume_record,
        )
        try:
            grouping_payload, grouping_decision_snapshot = (
                self._resolve_qc_grouping_decision(
                    store,
                    request_record,
                    experimental,
                    study_contract,
                    answers,
                )
            )
            cell_payload, cell_qc, cell_decision_snapshot = (
                self._resolve_cell_quality_decision(
                    store,
                    request_record,
                    experimental,
                    grouping_payload,
                    answers,
                )
            )
            plan = self.build_preprocessing_plan(
                store,
                request_record,
                enrichment,
                experimental,
                ingest_outcome,
                cell_qc,
            )
            validate_rna_plan(plan, selected)
            plan = plan.model_copy(update={"cellQualityPayload": cell_payload})
            # The exact core feature policy is a provisional baseline. Its
            # scientific assessment follows observed PCA and marker evidence.
            route_summary = ", ".join(
                f"{value.assay}:{value.featureMethod}/{value.reductionMethod}"
                for value in plan.assays
            )
            logger.debug(
                f"Workflow {workflow.workflowRunId}: preprocessing plan built "
                f"(primary={plan.primaryAssay!r}, marker={plan.markerAssay!r}, "
                f"routes=[{route_summary}])"
            )
        except _DecisionNeedsInput as pending:
            if request_record.config.inputPolicy == "unattended":
                outcome = journal._complete_attempt(
                    started,
                    status="failed",
                    artifacts={
                        "cellSelection": experimental.cellSelection,
                        **cell_qc_artifacts,
                    },
                    outputs={"decisionCheckpointSha256": pending.checkpointSha256},
                    error=(
                        "The unattended preprocessing plan returned an unresolved "
                        "registered decision"
                    ),
                )
            else:
                outcome = journal._complete_attempt(
                    started,
                    status="needsInput",
                    artifacts={
                        "cellSelection": experimental.cellSelection,
                        **cell_qc_artifacts,
                    },
                    outputs={"decisionCheckpointSha256": pending.checkpointSha256},
                    needs_input=WorkflowNeedsInput(questions=[pending.question]),
                    notes=["A registered filtering decision requires input."],
                )
            journal._save_outcome(store.zw, prefix, outcome)
            return outcome, AutomatedPreprocessingPlan.get_blank()
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts={
                    "cellSelection": experimental.cellSelection,
                    **cell_qc_artifacts,
                },
            )
            return outcome, AutomatedPreprocessingPlan.get_blank()
        logger.debug(
            f"Workflow {workflow.workflowRunId}: executing the evidence-bounded "
            "preprocessing plan"
        )
        outcome = journal._complete_attempt(
            started,
            status="done",
            artifacts={
                "cellSelection": experimental.cellSelection,
                **cell_qc_artifacts,
            },
            outputs={
                "preprocessingPlan": plan.model_dump(mode="json"),
                "qcGroupingDecisionCheckpoint": grouping_decision_snapshot,
                "cellQualityDecisionCheckpoint": cell_decision_snapshot,
            },
            actions=[
                "audit_qc_grouping_decision",
                "audit_cell_quality_decision",
                "accept_evidence_bounded_preprocessing_plan",
            ],
        )
        journal._save_outcome(store.zw, prefix, outcome)
        return outcome, plan

    def build_preprocessing_plan(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        enrichment: DataEnrichmentReport,
        experimental: ExperimentalContextResult,
        ingest_outcome: WorkflowStageAttempt,
        cell_qc: CellQcPlan,
    ) -> AutomatedPreprocessingPlan:
        del ingest_outcome
        request = request_record.request
        store_summary = store.summary()
        selected = selected_store_rna_assay(store, request)
        summary = next(
            value for value in store_summary.assays if value.name == selected
        )
        policy = next(
            (value for value in enrichment.policies if value.assay == selected), None
        )
        inspection = next(
            (value for value in enrichment.inspections if value.assay == selected), None
        )
        if policy is not None and policy.assayModality != "RNA":
            raise ValueError("Enrichment policy does not match the selected RNA assay")
        assay_plan = self.build_assay_preprocessing_plan(
            selected,
            summary,
            policy,
            inspection,
        )
        if not assay_plan.graphEligible:
            raise ValueError("RNA requires at least three features for PCA")
        final_plan = AutomatedPreprocessingPlan(
            primaryAssay=selected,
            markerAssay=selected,
            cellSelection=experimental.cellSelection,
            cellQc=cell_qc,
            assays=[assay_plan],
            limitations=list(dict.fromkeys(enrichment.limitations)),
        )
        checksum = hashlib.sha256(
            record_io.canonical_json_bytes(
                final_plan.model_dump(mode="json", exclude={"planChecksum"})
            )
        ).hexdigest()
        return final_plan.model_copy(update={"planChecksum": checksum})

    def build_assay_preprocessing_plan(
        self,
        assay_name: str,
        summary: AssaySummary,
        policy: FeatureSelectionPolicy | None,
        inspection: AssayFeatureInspection | None,
    ) -> AssayPreprocessingPlan:
        if summary.assay_type != "RNA":
            raise ValueError("Automated preprocessing supports RNA only")
        evidence_ids = list(policy.evidenceIds) if policy is not None else []
        graph_eligible = summary.total_features >= 3
        proposed_families = list(policy.excludeFamilies) if policy is not None else []
        return AssayPreprocessingPlan(
            assay=assay_name,
            assayType=summary.assay_type,
            role="graph" if graph_eligible else "unsupported",
            graphEligible=graph_eligible,
            markerEligible=graph_eligible,
            featureMethod="hvg" if graph_eligible else "none",
            reductionMethod="pca" if graph_eligible else "none",
            featureParameters={
                "excludeFamilies": [],
                "useScarfDefaultBlacklist": True,
                "proposedExcludeFamilies": proposed_families,
                "protectFamilies": (
                    list(policy.protectFamilies) if policy is not None else []
                ),
                "protectFeatures": list(policy.protectFeatures)
                if policy is not None
                else [],
                "proposedExcludeFeatures": list(
                    dict.fromkeys([*policy.excludeFeatures, *policy.artificialFeatures])
                )
                if policy is not None
                else [],
                "species": (
                    inspection.species if inspection is not None else "unknown"
                ),
                "defaultFeatureInventory": (
                    inspection.defaultFeatureInventory.model_dump(mode="json")
                    if inspection is not None
                    and inspection.defaultFeatureInventory is not None
                    else None
                ),
            },
            evidenceIds=evidence_ids,
            limitations=(
                []
                if graph_eligible
                else ["RNA requires at least three features for PCA"]
            ),
        )

    def preprocessing_stage(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        experimental: ExperimentalContextResult,
        study_contract: StudyContract,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
        stage_name: WorkflowStageName = "preprocessing",
    ) -> tuple[
        WorkflowStageAttempt,
        list[PreprocessedAssayHandoff],
        AutomatedPreprocessingPlan,
    ]:
        selected = selected_store_rna_assay(store, request_record.request)
        validate_rna_plan(plan, selected)
        validate_rna_context(experimental, selected)
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
        )
        if existing is not None:
            logger.debug(
                f"Workflow {workflow.workflowRunId}: reusing preprocessing artifacts"
            )
            cached_handoffs = [
                PreprocessedAssayHandoff.model_validate(value)
                for value in existing.outputs["assays"]
            ]
            cached_plan = AutomatedPreprocessingPlan.model_validate(
                existing.outputs["resolvedPreprocessingPlan"]
            )
            validate_rna_plan(cached_plan, selected)
            validate_rna_handoffs(cached_handoffs, selected)
            return existing, cached_handoffs, cached_plan
        if plan.cellSelection is None:
            raise ValueError("Preprocessing plan lacks an exact cell selection")
        if experimental.cellSelection != plan.cellSelection:
            raise ValueError(
                "Preprocessing plan and Experimental Context selections differ"
            )
        input_cell_selection = artifact_model_to_ref(plan.cellSelection)
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
            inputs={
                "preprocessingPlan": plan.model_dump(mode="json"),
                "cellSelection": plan.cellSelection.model_dump(mode="json"),
            },
            resume_record=resume_record,
        )
        actions: list[str] = []
        operations: list[dict[str, Any]] = []
        artifacts: dict[str, ArtifactReferenceModel] = {
            "inputCellSelection": plan.cellSelection,
            **self._cell_qc_stage_artifacts(plan.cellQc),
        }
        try:
            if plan.cellQualityPayload is None:
                raise ValueError(
                    "Decision-driven preprocessing requires an audited "
                    "cell-quality payload"
                )
            cell_selection = self.apply_cell_qc(
                store,
                experimental,
                input_cell_selection,
                actions,
                operations,
                selected_plan=plan.cellQc,
                decision_payload=plan.cellQualityPayload,
            )
            cell_selection_model = ArtifactReferenceModel.from_artifact_ref(
                cell_selection
            )
            artifacts["cellSelection"] = cell_selection_model
            if operations:
                diagnostic_flags = operations[-1].get("diagnosticFlags")
                if diagnostic_flags is not None:
                    artifacts["cellQcDiagnosticFlags"] = (
                        ArtifactReferenceModel.model_validate(diagnostic_flags)
                    )
            active_cells = int(
                read_stored_selection_mask(
                    store.zw,
                    cell_selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                ).sum()
            )
            selected_profile = next(
                (
                    profile
                    for profile in experimental.qcProfiles
                    if profile.profileId == plan.cellQc.profileId
                ),
                None,
            )
            if selected_profile is None:
                raise ValueError("Selected cell-QC profile is unavailable")
            if active_cells != selected_profile.retainedCells:
                raise ValueError(
                    "Executed cell-QC retention differs from the selected profile"
                )
            logger.info(
                f"QC: compared {len(experimental.qcProfiles)} policies; retained "
                f"{active_cells:,}/{selected_profile.activeCells:,} cells "
                f"({selected_profile.retainedFraction:.0%})."
            )
            if active_cells < 3:
                raise ValueError("Preprocessing requires at least three active cells")
            handoffs: list[PreprocessedAssayHandoff] = []
            with candidate_metric_cache():
                for assay_plan in plan.assays:
                    if not assay_plan.graphEligible:
                        continue
                    logger.debug(
                        f"Workflow {workflow.workflowRunId}: preprocessing assay "
                        f"{assay_plan.assay!r} via {assay_plan.featureMethod}/"
                        f"{assay_plan.reductionMethod}"
                    )
                    handoffs.append(
                        self.preprocess_assay(
                            store,
                            assay_plan,
                            cell_selection=cell_selection,
                            cell_selection_model=cell_selection_model,
                            active_cells=active_cells,
                            request_record=request_record,
                            study_contract=study_contract,
                            answers=answers,
                            actions=actions,
                            operations=operations,
                            artifacts=artifacts,
                        )
                    )
            resolved_plan = self._plan_with_selected_hvg_counts(
                plan,
                handoffs,
            )
            outcome = journal._complete_attempt(
                started,
                status="done",
                artifacts={
                    name: value
                    for name, value in artifacts.items()
                    if value is not None
                },
                outputs={
                    "assays": [value.model_dump(mode="json") for value in handoffs],
                    "cellSelection": cell_selection_model.model_dump(mode="json"),
                    "resolvedPreprocessingPlan": resolved_plan.model_dump(mode="json"),
                    "operations": operations,
                },
                actions=actions,
            )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.debug(
                f"Workflow {workflow.workflowRunId}: preprocessing produced "
                f"{len(handoffs)} graph-ready assay handoff(s)"
            )
            return outcome, handoffs, resolved_plan
        except _DecisionNeedsInput as pending:
            if request_record.config.inputPolicy == "unattended":
                outcome = journal._complete_attempt(
                    started,
                    status="failed",
                    artifacts={
                        name: value
                        for name, value in artifacts.items()
                        if value is not None
                    },
                    outputs={
                        "operations": operations,
                        "decisionCheckpointSha256": pending.checkpointSha256,
                    },
                    actions=actions,
                    error=(
                        "The unattended preprocessing stage returned an unresolved "
                        "registered decision"
                    ),
                )
            else:
                outcome = journal._complete_attempt(
                    started,
                    status="needsInput",
                    artifacts={
                        name: value
                        for name, value in artifacts.items()
                        if value is not None
                    },
                    outputs={
                        "operations": operations,
                        "decisionCheckpointSha256": pending.checkpointSha256,
                    },
                    needs_input=WorkflowNeedsInput(questions=[pending.question]),
                    actions=actions,
                    notes=["A registered RNA preprocessing decision requires input."],
                )
            journal._save_outcome(store.zw, prefix, outcome)
            return outcome, [], plan
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=artifacts,
                actions=actions,
                outputs={"operations": operations},
            )
            return outcome, [], plan

    def preprocess_assay(
        self,
        store: DataStore,
        assay_plan: AssayPreprocessingPlan,
        *,
        cell_selection: ArtifactRef,
        cell_selection_model: ArtifactReferenceModel,
        active_cells: int,
        request_record: OrchestrationRequestRecord | None = None,
        study_contract: StudyContract | None = None,
        answers: Mapping[str, Any] | None = None,
        actions: list[str],
        operations: list[dict[str, Any]],
        artifacts: dict[str, ArtifactReferenceModel],
    ) -> PreprocessedAssayHandoff:
        """Prepare exact core defaults and feature evidence without a graph search."""
        del request_record, study_contract, answers
        assay = store.get_assay(assay_plan.assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("RNA preprocessing requires an RNAassay")
        from ..parameter_tuning.hvg import core_hvg_evidence

        feature_refs = core_hvg_evidence(
            store, assay=assay_plan.assay, cells=cell_selection
        )
        baseline = feature_refs["scarfDefault"]
        selected_count = int(
            np.asarray(
                as_zarr_array(store.load_artifact(baseline)["values"], name="values")[
                    :
                ],
                dtype=bool,
            ).sum()
        )
        marker_features = store.select_all_features(from_assay=assay_plan.assay)
        feature_models = {
            key: ArtifactReferenceModel.from_artifact_ref(value)
            for key, value in feature_refs.items()
        }
        marker_model = ArtifactReferenceModel.from_artifact_ref(marker_features)
        artifacts.update(
            {
                f"{assay_plan.assay}_{key}": value
                for key, value in feature_models.items()
            }
        )
        artifacts[f"{assay_plan.assay}_marker_features"] = marker_model
        operations.append(
            {
                "operation": "prepare_core_hvg_baseline",
                "assay": assay_plan.assay,
                "cellSelection": cell_selection_model.model_dump(mode="json"),
                "requestedTopN": 1000,
                "selectedTopN": selected_count,
                "artifacts": {
                    key: value.model_dump(mode="json")
                    for key, value in feature_models.items()
                },
            }
        )
        actions.append(f"prepare_core_defaults:{assay_plan.assay}")
        logger.info(
            f"HVG baseline: core Scarf selected {selected_count:,} genes; "
            "representation experiments will use the screening cells."
        )
        return PreprocessedAssayHandoff(
            assay=assay_plan.assay,
            assayType="RNA",
            cellSelection=cell_selection_model,
            reductionMethod="pca",
            graphFeatures=feature_models["scarfDefault"],
            markerFeatures=marker_model,
            graphFeatureCandidates=feature_models,
            nCells=active_cells,
            nFeatures=selected_count,
        )

    def apply_cell_qc(
        self,
        store: DataStore,
        experimental: ExperimentalContextResult,
        cell_selection: ArtifactRef,
        actions: list[str],
        operations: list[dict[str, Any]],
        *,
        selected_plan: CellQcPlan | None = None,
        decision_payload: CellQualityExecutorPayload | None = None,
    ) -> ArtifactRef:
        plan = selected_plan or experimental.cellQc
        if decision_payload is not None:
            if (
                cell_qc_policy(plan.action, plan.registeredProfile)
                != decision_payload.profile
            ):
                raise ValueError(
                    "Cell-QC execution plan differs from its audited payload"
                )
            expected_capture = decision_payload.profile in {
                "coreSampleMad3",
                "captureMad5",
                "captureMad3Sensitivity",
                "pooledReferenceMad5",
            }
            if expected_capture != (
                plan.sampleColumn is not None or plan.sampleArtifact is not None
            ):
                raise ValueError(
                    "Cell-QC capture source differs from its audited payload"
                )
        input_model = ArtifactReferenceModel.from_artifact_ref(cell_selection)
        logger.info(
            f"Applying cell QC action={plan.action!r}, profile={plan.profileId!r}"
        )
        profile = next(
            (
                value
                for value in experimental.qcProfiles
                if value.profileId == plan.profileId
            ),
            None,
        )
        if profile is None:
            raise ValueError("Experimental Context selected an unknown QC profile")
        for field_name in (
            "action",
            "registeredProfile",
            "driverAssay",
            "driverAssayType",
            "sampleColumn",
            "sampleArtifact",
            "attributes",
            "artifactMetrics",
        ):
            if getattr(plan, field_name) != getattr(profile, field_name):
                raise ValueError(
                    "Selected cell-QC plan does not match its exact offered profile"
                )
        if any(
            source not in experimental.qualityMetricArtifacts
            for source in plan.artifactMetrics
        ):
            raise ValueError(
                "Selected cell-QC metrics are absent from Experimental Context"
            )
        if (
            plan.sampleArtifact is not None
            and plan.sampleArtifact not in experimental.htoIdentityArtifacts
        ):
            raise ValueError(
                "Selected cell-QC sample artifact is absent from Experimental Context"
            )
        artifact_metrics = [
            NamedCellArtifact(
                name=source.name,
                artifact=artifact_model_to_ref(source.artifact),
            )
            for source in plan.artifactMetrics
        ]
        sample_artifact = (
            None
            if plan.sampleArtifact is None
            else NamedCellArtifact(
                name=plan.sampleArtifact.name,
                artifact=artifact_model_to_ref(plan.sampleArtifact.artifact),
            )
        )
        if plan.registeredProfile is not None:
            result, diagnostic_flags = execute_registered_cell_qc(
                store,
                plan.registeredProfile,
                profile_parameters=profile.parameters,
                expected_active_cells=profile.activeCells,
                expected_retained_cells=profile.retainedCells,
                expected_flag_counts=profile.flaggedCells,
                attrs=plan.attributes,
                artifact_metrics=artifact_metrics,
                cell_selection=cell_selection,
                sample_column=plan.sampleColumn,
                sample_artifact=sample_artifact,
                invalidate_cache=False,
            )
            result_model = ArtifactReferenceModel.from_artifact_ref(result)
            flags_model = (
                None
                if diagnostic_flags is None
                else ArtifactReferenceModel.from_artifact_ref(diagnostic_flags)
            )
            actions.append(f"cell_qc_registered:{profile.registeredProfile}")
            operations.append(
                {
                    "operation": "run_registered_cell_qc",
                    "profileId": profile.profileId,
                    "registeredProfile": profile.registeredProfile,
                    "cellSelection": input_model.model_dump(mode="json"),
                    "attrs": list(plan.attributes),
                    "artifactMetrics": [
                        source.model_dump(mode="json")
                        for source in plan.artifactMetrics
                    ],
                    "sampleColumn": plan.sampleColumn,
                    "sampleArtifact": (
                        None
                        if plan.sampleArtifact is None
                        else plan.sampleArtifact.model_dump(mode="json")
                    ),
                    "profileParameters": profile.parameters,
                    "expectedActiveCells": profile.activeCells,
                    "expectedRetainedCells": profile.retainedCells,
                    "expectedFlagCounts": profile.flaggedCells,
                    "invalidateCache": False,
                    "diagnosticFlags": (
                        None
                        if flags_model is None
                        else flags_model.model_dump(mode="json")
                    ),
                    "artifact": result_model.model_dump(mode="json"),
                }
            )
            return result
        if plan.action == "skip":
            actions.append("skip_cell_qc")
            operations.append(
                {
                    "operation": "skip_cell_qc",
                    "profileId": plan.profileId,
                    "cellSelection": input_model.model_dump(mode="json"),
                    "artifact": input_model.model_dump(mode="json"),
                }
            )
            return cell_selection
        if plan.action in {"globalGaussian", "sampleMad"}:
            if not isinstance(profile.resolvedBounds, dict):
                raise ValueError("Core cell-QC evidence requires exact named bounds")
            capture_artifact = (
                NamedCellArtifact(
                    name=profile.captureArtifact.name,
                    artifact=artifact_model_to_ref(profile.captureArtifact.artifact),
                )
                if profile.captureArtifact is not None
                else None
            )
            result, diagnostic_flags = execute_auto_cell_qc(
                store,
                plan.action,
                profile_parameters=profile.parameters,
                expected_active_cells=profile.activeCells,
                expected_retained_cells=profile.retainedCells,
                expected_flag_counts=profile.flaggedCells,
                expected_resolved_bounds=profile.resolvedBounds,
                attrs=plan.attributes,
                artifact_metrics=artifact_metrics,
                cell_selection=cell_selection,
                sample_column=plan.sampleColumn,
                sample_artifact=sample_artifact,
                capture_column=profile.captureColumn,
                capture_artifact=capture_artifact,
                invalidate_cache=False,
            )
            result_model = ArtifactReferenceModel.from_artifact_ref(result)
            actions.append(f"cell_qc_{plan.action}:{profile.profileId}")
            operations.append(
                {
                    "operation": "auto_filter_cells",
                    "profileId": profile.profileId,
                    "cellSelection": input_model.model_dump(mode="json"),
                    "attrs": list(plan.attributes),
                    "artifactMetrics": [
                        source.model_dump(mode="json")
                        for source in plan.artifactMetrics
                    ],
                    "sampleColumn": plan.sampleColumn,
                    "sampleArtifact": plan.sampleArtifact.model_dump(mode="json")
                    if plan.sampleArtifact
                    else None,
                    "profileParameters": profile.parameters,
                    "resolvedBounds": profile.resolvedBounds,
                    "expectedRetainedCells": profile.retainedCells,
                    "diagnosticFlags": ArtifactReferenceModel.from_artifact_ref(
                        diagnostic_flags
                    ).model_dump(mode="json")
                    if diagnostic_flags
                    else None,
                    "invalidateCache": False,
                    "artifact": result_model.model_dump(mode="json"),
                }
            )
            return result
        raise ValueError(f"Unsupported cell QC action {plan.action!r}")
