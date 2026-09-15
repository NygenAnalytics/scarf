"""Controller for one resumable RNA analysis with checkpoint-owned state."""

import hashlib
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import zarr

from ...datastore.datastore import DataStore
from ...datastore.summary import summarize_zarr_readonly
from ...utils.logging import logger
from .. import record_io
from ..config.agent_exec import describe_agent_error
from ..experimental_context.study import StudyContract, validate_objective_evidence
from ..ingest import IngestResult, detect_format, ingest
from ..ingest.manifest import DatasetManifest, inspect_h5ad_manifest
from . import journal
from .context import ContextStagesMixin
from .finalization import FinalizationStagesMixin
from .models import (
    _STAGE_ORDER,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    AutomatedWorkflowResumeRequest,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    WorkflowIdentity,
    WorkflowNeedsInput,
    WorkflowQuestion,
)
from .preprocessing import PreprocessingStagesMixin
from .rna import (
    selected_rna_assay,
    selected_store_rna_assay,
    validate_rna_directions,
    validate_rna_request_fields,
    validate_saved_rna_history,
)
from .tuning import TuningStagesMixin


def _model_identity(model: Any) -> str:
    from ..config.agent_exec import _model_name

    settings = getattr(model, "settings", None) or {}
    provider = getattr(model, "provider", None)
    profile = getattr(model, "profile", None)
    image_input = getattr(model, "supports_image_input", None)
    if not isinstance(image_input, bool) and isinstance(profile, Mapping):
        image_input = profile.get("supports_image_input")
    identity = {
        "settings": settings,
        "system": getattr(model, "system", None),
        "provider": getattr(provider, "name", None),
        "baseUrl": str(getattr(provider, "base_url", "")),
        "supportsImageInput": image_input if isinstance(image_input, bool) else None,
    }
    digest = hashlib.sha256(record_io.canonical_json_bytes(identity)).hexdigest()
    return f"{type(model).__module__}.{type(model).__qualname__}:{_model_name(model)}:{digest}"


def _submitted_identity(request: AutomatedWorkflowRequest) -> str:
    value = request.model_dump(mode="json")
    value["sourcePath"] = str(Path(request.sourcePath).resolve())
    if request.zarrPath is not None:
        value["zarrPath"] = str(Path(request.zarrPath).resolve())
    return hashlib.sha256(record_io.canonical_json_bytes(value)).hexdigest()


def _source_identity(path: str) -> dict[str, Any]:
    source = Path(path).resolve()
    if source.is_file():
        stat = source.stat()
        return {
            "path": str(source),
            "bytes": stat.st_size,
            "modifiedNs": stat.st_mtime_ns,
        }
    return {"path": str(source)}


def _data_identity(
    store: DataStore,
    assay_name: str,
    *,
    columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Fingerprint the selected assay and original metadata in bounded blocks."""
    from ..parameter_tuning.execution import _metadata_column_fingerprint

    assay = store.get_assay(assay_name)
    digest = hashlib.sha256()
    digest.update(str(assay.rawData.shape).encode())
    digest.update(str(assay.rawData.dtype).encode())
    for block in assay.rawData.stream_blocks(nthreads=1, prefetch=1):
        digest.update(block.tobytes(order="C"))
    names = sorted(columns if columns is not None else store.cells.columns)
    feature_names = sorted(
        feature_columns if feature_columns is not None else assay.feats.columns
    )
    return {
        "assay": assay_name,
        "countsSha256": digest.hexdigest(),
        "featureMetadata": {
            name: _metadata_column_fingerprint(assay.feats, name)
            for name in feature_names
        },
        "metadata": {
            name: _metadata_column_fingerprint(store.cells, name) for name in names
        },
    }


class AgentOrchestrator(
    ContextStagesMixin,
    PreprocessingStagesMixin,
    TuningStagesMixin,
    FinalizationStagesMixin,
):
    """Run bounded RNA analysis; the journal owns all durable state."""

    def __init__(
        self, model: Any, *, config: AutomatedWorkflowConfig | None = None
    ) -> None:
        self.model = model
        self.config = config or AutomatedWorkflowConfig()
        self._explicit_config_fields = (
            set(config.model_fields_set) if config is not None else set()
        )

    def _validate_resume_config(self, saved: AutomatedWorkflowConfig) -> None:
        """Saved defaults remain authoritative unless a caller overrides them."""
        if any(
            getattr(saved, field) != getattr(self.config, field)
            for field in self._explicit_config_fields
        ):
            raise ValueError("Resume execution settings differ from the saved workflow")

    def run(self, request: AutomatedWorkflowRequest) -> AutomatedWorkflowResult:
        try:
            result = self._run(request)
        except Exception as exc:
            result = AutomatedWorkflowResult(notes=[describe_agent_error(exc)])
        if result.status != "completed":
            logger.error(
                f"RNA analysis {result.status} during {result.currentStage}: "
                + "; ".join(result.notes)
            )
        return result

    def _run(self, request: AutomatedWorkflowRequest) -> AutomatedWorkflowResult:
        submitted = request
        try:
            validate_rna_request_fields(request)
        except ValueError as exc:
            return AutomatedWorkflowResult(notes=[describe_agent_error(exc)])
        reused = self._reuse_or_resume(request)
        if reused is not None:
            return reused
        format_name = detect_format(request.sourcePath)
        dataset_manifest: DatasetManifest | None = None
        logger.info(
            f"Starting automated agent workflow from {format_name!r} input "
            f"(workspace={request.workspace is not None})"
        )
        if request.workspace is not None and format_name != "zarr":
            logger.warning(
                "Automated agent workflow rejected a workspace for a converted input"
            )
            return AutomatedWorkflowResult(
                status="failed",
                currentStage="ingest",
                notes=[
                    "workspace is supported for existing Zarr inputs; converted "
                    "inputs create their dataset at the root"
                ],
            )
        if format_name == "zarr" and request.zarrPath is not None:
            source_path = Path(request.sourcePath).resolve()
            requested_path = Path(request.zarrPath).resolve()
            if source_path != requested_path:
                logger.warning(
                    "Automated agent workflow rejected an implicit Zarr copy"
                )
                return AutomatedWorkflowResult(
                    status="failed",
                    currentStage="ingest",
                    notes=["An existing Zarr input cannot be copied implicitly"],
                )
        if format_name == "h5ad":
            matrix_key = request.ingestDirections.get("matrixKey")
            try:
                dataset_manifest = inspect_h5ad_manifest(
                    request.sourcePath,
                    source_uri=(
                        str(request.ingestDirections["sourceUri"])
                        if request.ingestDirections.get("sourceUri") is not None
                        else request.sourcePath
                    ),
                    author_label_policy=request.authorLabelPolicy,
                    matrix_key=str(matrix_key) if matrix_key is not None else None,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                return AutomatedWorkflowResult(
                    status="failed",
                    currentStage="ingest",
                    notes=[
                        f"CELLxGENE manifest inspection failed: {describe_agent_error(exc)}"
                    ],
                )
            if dataset_manifest.declaredBatchColumns:
                experimental_directions = dict(request.experimentalDirections)
                raw_batch_columns = experimental_directions.get("batchColumns")
                if raw_batch_columns is None:
                    experimental_directions["batchColumns"] = list(
                        dataset_manifest.declaredBatchColumns
                    )
                elif not isinstance(raw_batch_columns, list) or any(
                    not isinstance(value, str) or not value.strip()
                    for value in raw_batch_columns
                ):
                    return AutomatedWorkflowResult(
                        status="failed",
                        currentStage="ingest",
                        notes=[
                            "experimentalDirections.batchColumns must be a list "
                            "of exact observation-column names"
                        ],
                    )
                elif not set(dataset_manifest.declaredBatchColumns).issubset(
                    raw_batch_columns
                ):
                    return AutomatedWorkflowResult(
                        status="failed",
                        currentStage="ingest",
                        notes=[
                            "experimentalDirections.batchColumns must include the "
                            "CELLxGENE uns/batch_condition columns"
                        ],
                    )
                request = request.model_copy(
                    update={"experimentalDirections": experimental_directions}
                )
            manifest_decision = dataset_manifest.decision
            if manifest_decision.status == "needsInput":
                if self.config.inputPolicy == "unattended":
                    return AutomatedWorkflowResult(
                        status="abstained",
                        currentStage="ingest",
                        limitations=list(dataset_manifest.priorFiltering.limitations),
                        unresolvedClaims=[manifest_decision.summary],
                        notes=[
                            "The unattended workflow abstained because the count "
                            "matrix was ambiguous."
                        ],
                    )
                return AutomatedWorkflowResult(
                    status="needsInput",
                    currentStage="ingest",
                    needsInput=WorkflowNeedsInput(
                        questions=[
                            WorkflowQuestion(
                                questionId="datasetMatrixKey",
                                question=(
                                    manifest_decision.summary
                                    + ". Rerun with ingestDirections.matrixKey set "
                                    "to the selected option."
                                ),
                                options=list(manifest_decision.options),
                                evidenceIds=list(manifest_decision.evidenceIds),
                            )
                        ]
                    ),
                    limitations=list(dataset_manifest.priorFiltering.limitations),
                )
            if manifest_decision.status == "abstained":
                return AutomatedWorkflowResult(
                    status="abstained",
                    currentStage="ingest",
                    limitations=list(dataset_manifest.priorFiltering.limitations),
                    unresolvedClaims=[manifest_decision.summary],
                    notes=[
                        "The count-dependent RNA workflow did not run because its "
                        "input contract is not satisfied."
                    ],
                )
            selected_matrix = manifest_decision.selectedMatrixKey
            if selected_matrix is None:
                raise RuntimeError("Supported manifest lacks a selected matrix")
            ingest_directions = {
                **request.ingestDirections,
                "matrixKey": selected_matrix,
            }
            request = request.model_copy(update={"ingestDirections": ingest_directions})

        if format_name == "zarr":
            zarr_path = str(Path(request.sourcePath).resolve())
            effective_request = request.model_copy(update={"zarrPath": zarr_path})
            try:
                summary = summarize_zarr_readonly(
                    zarr_path,
                    workspace=request.workspace,
                )
            except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                return AutomatedWorkflowResult(
                    status="failed",
                    currentStage="ingest",
                    zarrPath=zarr_path,
                    notes=[
                        f"Opening the requested RNA store failed: {describe_agent_error(exc)}"
                    ],
                )
            ingest_result = IngestResult(
                status="done",
                format="zarr",
                zarrPath=zarr_path,
                assayNames=[assay.name for assay in summary.assays],
                summary=summary.to_dict(),
                actions=["summarize_zarr"],
            )
        else:
            ingest_result = ingest(
                path=request.sourcePath,
                zarrPath=request.zarrPath,
                model=self.model,
                directions=request.ingestDirections,
            )
            if ingest_result.zarrPath is not None:
                zarr_path = str(Path(ingest_result.zarrPath).resolve())
                ingest_result = ingest_result.model_copy(update={"zarrPath": zarr_path})
            effective_request = request.model_copy(
                update={"zarrPath": ingest_result.zarrPath}
            )
        logger.info(
            f"Automated workflow ingest returned status={ingest_result.status!r}, "
            f"format={ingest_result.format!r}, assays={len(ingest_result.assayNames)}"
        )
        if ingest_result.status != "done" or ingest_result.zarrPath is None:
            needs_input = None
            if ingest_result.needsInput is not None:
                needs_input = WorkflowNeedsInput(
                    questions=[
                        WorkflowQuestion(
                            questionId="ingest",
                            question=ingest_result.needsInput.question,
                            options=list(ingest_result.needsInput.options),
                            evidenceIds=list(ingest_result.needsInput.evidenceIds),
                        )
                    ]
                )
            if needs_input is not None and self.config.inputPolicy == "unattended":
                return AutomatedWorkflowResult(
                    status="abstained",
                    currentStage="ingest",
                    zarrPath=ingest_result.zarrPath,
                    unresolvedClaims=[
                        question.question for question in needs_input.questions
                    ],
                    notes=[
                        *ingest_result.notes,
                        "The unattended workflow abstained instead of waiting for "
                        "an ingest decision.",
                    ],
                )
            return AutomatedWorkflowResult(
                status=("needsInput" if needs_input is not None else "failed"),
                currentStage="ingest",
                zarrPath=ingest_result.zarrPath,
                needsInput=needs_input,
                notes=list(ingest_result.notes),
            )

        try:
            selected = selected_rna_assay(
                effective_request,
                {
                    value["name"]: value["assay_type"]
                    for value in (ingest_result.summary or {}).get("assays", [])
                },
            )
            effective_request = effective_request.model_copy(
                update={
                    "primaryAssay": selected,
                    "markerAssay": selected,
                    "analysisAssays": [selected],
                }
            )
            store = self.open_store(ingest_result.zarrPath, effective_request)
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return AutomatedWorkflowResult(
                zarrPath=ingest_result.zarrPath, notes=[describe_agent_error(exc)]
            )
        ignored = [name for name in store.assay_names if name != selected]
        logger.info(
            f"RNA analysis: selected assay {selected!r}"
            + (f"; ignored other assays {ignored}" if ignored else "")
        )
        workflow = WorkflowIdentity(uuid.uuid4().hex, effective_request.workspace)
        request_record = self.initialize_request(
            store, workflow, effective_request, submitted
        )
        prefix = journal._ensure_orchestration_store(store)
        self.record_ingest_stage(
            store,
            prefix,
            workflow,
            request_record,
            ingest_result,
            dataset_manifest,
        )
        return self._continue(
            store,
            workflow,
            request_record,
            answers={},
        )

    def _reuse_or_resume(
        self, request: AutomatedWorkflowRequest
    ) -> AutomatedWorkflowResult | None:
        source = Path(request.sourcePath)
        fmt = detect_format(request.sourcePath)
        if request.zarrPath is not None:
            destination = Path(request.zarrPath)
        elif fmt == "zarr":
            destination = source
        elif source.is_dir():
            destination = source.with_name(source.name + ".zarr")
        else:
            source_stem = (
                source.with_suffix("") if source.suffix.lower() == ".gz" else source
            )
            destination = source_stem.with_suffix(".zarr")
        if not destination.exists():
            return None
        root = zarr.open_group(str(destination), mode="r")
        active = root if request.workspace is None else root[request.workspace]
        if not isinstance(active, zarr.Group):
            raise ValueError("Requested workspace is not a group")
        prefix = record_io.join_key(active.path, "agents", "orchestrations")
        matches = []
        for key in journal._list_keys(active, prefix):
            if not key.endswith("/request.json"):
                continue
            identifier = key.rsplit("/", 2)[-2]
            try:
                saved = journal.read_request(active, prefix, identifier)
            except ValueError:
                continue
            if saved.inputIdentity.get("userRequestSha256") == _submitted_identity(
                request
            ):
                matches.append(saved)
        if len(matches) > 1:
            raise ValueError(
                "Several workflows match this request; use an exact advanced resume identifier"
            )
        if matches:
            saved = matches[0]
            self._validate_resume_config(saved.config)
            if saved.modelIdentity != _model_identity(self.model):
                raise ValueError(
                    "The destination contains this request with different model or execution settings; use a new destination or exact advanced workflow"
                )
            return self.resume(
                AutomatedWorkflowResumeRequest(
                    zarrPath=str(destination.resolve()),
                    workspace=request.workspace,
                    workflowRunId=saved.workflowRunId,
                )
            )
        if fmt != "zarr":
            raise FileExistsError(
                "The destination exists without an exactly matching RNA request; choose a different destination"
            )
        return None

    def initialize_request(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request: AutomatedWorkflowRequest,
        submitted: AutomatedWorkflowRequest | None = None,
    ) -> OrchestrationRequestRecord:
        prefix = journal._ensure_orchestration_store(store)
        if request.primaryAssay is None:
            raise ValueError("RNA selection must be resolved before saving the request")
        identity = {
            "userRequestSha256": _submitted_identity(submitted or request),
            "source": _source_identity(request.sourcePath),
            "data": _data_identity(store, request.primaryAssay),
        }
        record = OrchestrationRequestRecord(
            workflowRunId=workflow.workflowRunId,
            createdAtNs=time.time_ns(),
            request=request,
            config=self.config,
            requestSha256=journal._sha256_model(request),
            configSha256=journal._sha256_model(self.config),
            modelIdentity=_model_identity(self.model),
            inputIdentity=identity,
        )
        record = record.model_copy(
            update={"contentSha256": journal._record_checksum(record)}
        )
        journal._write_model_once(
            store.zw, journal._request_key(prefix, workflow.workflowRunId), record
        )
        return record

    def load_request_for_resume(
        self, request: AutomatedWorkflowResumeRequest
    ) -> tuple[OrchestrationRequestRecord, DataStore]:
        store = journal.open_analysis_store(
            request.zarrPath, request.workflowRunId, workspace=request.workspace
        )
        prefix = journal._orchestration_prefix(store)
        record = journal.read_request(store.zw, prefix, request.workflowRunId)
        if record.modelIdentity != _model_identity(self.model):
            raise ValueError("Resume model differs from the saved workflow")
        self._validate_resume_config(record.config)
        selected = selected_store_rna_assay(store, record.request)
        validate_saved_rna_history(store, prefix, request.workflowRunId, selected)
        expected = record.inputIdentity
        if expected["source"] != _source_identity(record.request.sourcePath):
            raise ValueError("Source input has changed since this workflow was started")
        observed = _data_identity(
            store,
            selected,
            columns=list(expected["data"]["metadata"]),
            feature_columns=list(expected["data"]["featureMetadata"]),
        )
        if observed != expected["data"]:
            raise ValueError(
                "Selected RNA data or relevant metadata changed; start a new analysis"
            )
        return record, self.open_store(request.zarrPath, record.request)

    def resume(
        self, request: AutomatedWorkflowResumeRequest
    ) -> AutomatedWorkflowResult:
        original_config = self.config
        try:
            result = self._resume(request)
        except Exception as exc:
            result = AutomatedWorkflowResult(
                zarrPath=request.zarrPath,
                workspace=request.workspace,
                workflowRunId=request.workflowRunId,
                notes=[describe_agent_error(exc)],
            )
        finally:
            self.config = original_config
        if result.status != "completed":
            logger.error(
                f"RNA analysis {result.status} during {result.currentStage}: "
                + "; ".join(result.notes)
            )
        return result

    def _resume(
        self, request: AutomatedWorkflowResumeRequest
    ) -> AutomatedWorkflowResult:
        record, store = self.load_request_for_resume(request)
        self.config = record.config
        workflow = WorkflowIdentity(record.workflowRunId, record.request.workspace)
        snapshot = journal.analysis_snapshot(store, workflow.workflowRunId)
        if snapshot["status"] == "completed":
            if request.answers:
                raise ValueError(
                    "A completed analysis cannot accept new decision answers"
                )
            result = AutomatedWorkflowResult(
                status="completed",
                currentStage="analysis_finalization",
                zarrPath=request.zarrPath,
                workspace=request.workspace,
                workflowRunId=request.workflowRunId,
            )
            final = snapshot["finalAnalysis"]
            result = result.model_copy(
                update={"limitations": list(final.get("limitations", []))}
            )
            try:
                result.report()
            except Exception as exc:
                return result.model_copy(
                    update={
                        "status": "failed",
                        "currentStage": "report",
                        "notes": [describe_agent_error(exc)],
                    }
                )
            return result
        stages = snapshot["stages"]
        latest = stages[-1] if stages else None
        prefix = journal._orchestration_prefix(store)
        starts = [
            value
            for stage in _STAGE_ORDER
            for value in journal._stage_starts(
                store.zw, prefix, workflow.workflowRunId, stage
            )
        ]
        latest_start = (
            max(starts, key=lambda value: value.startedAtNs) if starts else None
        )
        answers = dict(request.answers)
        resume_record = None
        if latest is not None and latest["status"] == "needsInput":
            from .models import WorkflowStageAttempt

            outcome = WorkflowStageAttempt.model_validate(
                {k: v for k, v in latest.items() if k not in {"report", "decisions"}}
            )
            answered = journal._parent_link(outcome)
            if (
                not answers
                and latest_start is not None
                and latest_start.startedAtNs > outcome.startedAtNs
            ):
                if latest_start.inputs.get("answeredAttempt") == answered.model_dump(
                    mode="json"
                ):
                    answers = dict(latest_start.inputs.get("resumeAnswers", {}))
            if not answers and outcome.stage != "parameter_tuning":
                return journal.paused_or_failed_result(store, workflow, record, outcome)
            if answers:
                errors = journal._resume_answer_errors(outcome, answers)
                if errors:
                    raise ValueError("; ".join(errors))
                assert outcome.needsInput is not None
                resume_record = OrchestrationResumeRecord(
                    workflowRunId=workflow.workflowRunId,
                    answeredAttempt=answered,
                    answers=answers,
                    questionIds=[q.questionId for q in outcome.needsInput.questions],
                )
            # Tuning replays committed actions and budgets. With no answer it
            # preserves a scientific defer, but retries an uncommitted assessment.
        elif answers:
            raise ValueError("Resume answers require an exact pending stage")
        elif latest_start is not None and latest_start.inputs.get("resumeAnswers"):
            from .models import WorkflowStageLink

            answers = dict(latest_start.inputs["resumeAnswers"])
            resume_record = OrchestrationResumeRecord(
                workflowRunId=workflow.workflowRunId,
                answeredAttempt=WorkflowStageLink.model_validate(
                    latest_start.inputs["answeredAttempt"]
                ),
                answers=answers,
                questionIds=list(answers),
            )
        directions = answers.get("experimentalDirections")
        if isinstance(directions, Mapping):
            validate_rna_directions(directions)
        return self._continue(
            store, workflow, record, answers=answers, resume_record=resume_record
        )

    def open_store(
        self,
        zarr_path: str,
        request: AutomatedWorkflowRequest,
    ) -> DataStore:
        default_assay = cast(
            str | None,
            request.ingestDirections.get("defaultAssay") or request.primaryAssay,
        )
        return DataStore(
            zarr_path,
            default_assay=default_assay,
            min_features_per_cell=-1,
            mito_pattern="",
            ribo_pattern="",
            zarr_mode="r+",
            workspace=request.workspace,
        )

    def _continue(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        *,
        answers: Mapping[str, Any],
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> AutomatedWorkflowResult:
        try:
            return self._execute_stages(
                store,
                workflow,
                request_record,
                answers=answers,
                resume_record=resume_record,
            )
        except Exception as exc:
            prefix = journal._orchestration_prefix(store)
            starts = [
                value
                for stage in _STAGE_ORDER
                for value in journal._stage_starts(
                    store.zw, prefix, workflow.workflowRunId, stage
                )
            ]
            latest = (
                max(starts, key=lambda value: value.startedAtNs) if starts else None
            )
            if latest is not None and not any(
                value.attemptId == latest.attemptId
                for value in journal._stage_outcomes(
                    store.zw, prefix, workflow.workflowRunId, latest.stage
                )
            ):
                try:
                    journal.finish_exception(store, prefix, workflow, latest, exc)
                except Exception as persistence_error:
                    exc.add_note(
                        "Saving the failed stage also failed: "
                        + describe_agent_error(persistence_error)
                    )
            return AutomatedWorkflowResult(
                currentStage=latest.stage if latest else "ingest",
                zarrPath=str(store.zarr_loc),
                workspace=workflow.workspace,
                workflowRunId=workflow.workflowRunId,
                notes=[describe_agent_error(exc)],
            )

    def _execute_stages(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        *,
        answers: Mapping[str, Any],
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> AutomatedWorkflowResult:
        """Continue the stage machine from the latest validated checkpoint."""
        logger.debug(f"Running stage sequence for workflow {workflow.workflowRunId}")
        prefix = journal._ensure_orchestration_store(store)
        ingest_outcome = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "ingest",
            request_record,
            [],
        )
        if ingest_outcome is None:
            raise RuntimeError("The persisted ingest stage is missing")
        cell_selection = ingest_outcome.artifacts.get("cellSelection")
        if cell_selection is None or cell_selection.kind != "cell_selection":
            raise RuntimeError(
                "The persisted ingest stage lacks an exact cell selection"
            )
        parents = [journal._parent_link(ingest_outcome)]

        enrichment_outcome, enrichment = self.data_enrichment_stage(
            store,
            workflow,
            request_record,
            parents,
            cell_selection,
            answers,
            resume_record=resume_record,
        )
        if enrichment_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                enrichment_outcome,
            )
        parents = [journal._parent_link(enrichment_outcome)]

        quality_outcome = self._rna_quality_metrics_stage(
            store,
            workflow,
            request_record,
            parents,
            enrichment,
            cell_selection,
            resume_record=resume_record,
        )
        if quality_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                quality_outcome,
            )
        quality_metric_artifacts = self._named_stage_artifacts(
            quality_outcome,
            "qualityMetricArtifacts",
            "quality_metric",
        )
        hto_identity_artifacts = self._named_stage_artifacts(
            quality_outcome,
            "htoIdentityArtifacts",
            "hto_identity",
        )
        parents = [journal._parent_link(quality_outcome)]

        context_outcome, experimental = self.experimental_context_stage(
            store,
            workflow,
            request_record,
            parents,
            cell_selection,
            enrichment_outcome.reportReferences[0],
            quality_metric_artifacts,
            hto_identity_artifacts,
            answers,
            resume_record=resume_record,
        )
        if context_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                context_outcome,
            )
        study_contract = StudyContract.model_validate(
            context_outcome.outputs["studyContract"]
        )
        validate_objective_evidence(study_contract, experimental)
        parents = [journal._parent_link(context_outcome)]

        plan_outcome, preprocessing_plan = self.preprocessing_plan_stage(
            store,
            workflow,
            request_record,
            parents,
            enrichment,
            experimental,
            ingest_outcome,
            study_contract,
            answers,
            resume_record=resume_record,
        )
        if plan_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                plan_outcome,
                study_contract=study_contract,
            )
        parents = [journal._parent_link(plan_outcome)]

        (
            preprocessing_outcome,
            preprocessed,
            preprocessing_plan,
        ) = self.preprocessing_stage(
            store,
            workflow,
            request_record,
            parents,
            preprocessing_plan,
            experimental,
            study_contract,
            answers,
            resume_record=resume_record,
        )
        if preprocessing_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                preprocessing_outcome,
                study_contract=study_contract,
            )
        parents = [journal._parent_link(preprocessing_outcome)]

        tuning_outcome, tuning_report = self.parameter_tuning_stage(
            store,
            workflow,
            request_record,
            parents,
            preprocessing_plan,
            preprocessed,
            experimental,
            enrichment_outcome.reportReferences[0],
            context_outcome.reportReferences[0],
            answers,
            study_contract=study_contract,
            resume_record=resume_record,
        )
        if tuning_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                tuning_outcome,
                study_contract=study_contract,
            )
        validate_objective_evidence(study_contract, experimental)
        parents = [journal._parent_link(tuning_outcome)]
        tuning_reference = tuning_outcome.reportReferences[0]
        selected = next(
            (
                value
                for value in tuning_report.evaluations
                if value.candidateId == tuning_report.recommendedCandidateId
            ),
            None,
        )
        if selected is None:
            raise ValueError("Full-cohort tuning did not select an evaluated candidate")
        preprocessed = [
            value.model_copy(
                update={
                    "normalized": selected.artifacts.get(
                        "normalized", value.normalized
                    ),
                    "graphFeatures": selected.artifacts.get(
                        "graphFeatures", value.graphFeatures
                    ),
                }
            )
            for value in preprocessed
        ]

        finalization_outcome, final_analysis = self.analysis_finalization_stage(
            store,
            workflow,
            request_record,
            parents,
            preprocessing_plan,
            preprocessed,
            tuning_report,
            tuning_reference,
            resume_record=resume_record,
        )
        if finalization_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                finalization_outcome,
                study_contract=study_contract,
            )

        completed = AutomatedWorkflowResult(
            status="completed",
            currentStage="analysis_finalization",
            zarrPath=str(store.zarr_loc),
            workspace=workflow.workspace,
            workflowRunId=workflow.workflowRunId,
            limitations=list(final_analysis.limitations),
            notes=["RNA analysis completed"],
        )
        from ..report.generator import generate_agent_report

        try:
            path = generate_agent_report(store, workflow.workflowRunId)
        except Exception as exc:
            logger.error(f"Analysis report failed: {describe_agent_error(exc)}")
            return completed.model_copy(
                update={
                    "status": "failed",
                    "currentStage": "report",
                    "notes": [
                        f"Report generation failed: {describe_agent_error(exc)}; the validated analysis is saved and can be resumed."
                    ],
                }
            )
        logger.info(f"Completed RNA analysis. Report saved to {path}")
        return completed
