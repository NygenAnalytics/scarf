"""A fixed RNA procedure. Decisions happen between complete pipeline calls."""

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .choices import (
    context_evidence_ids,
    native_probe_options,
    resolve_deferral,
    validate_annotations,
    validate_choice,
    validate_context,
)
from .evidence import (
    inspect_source,
    open_store,
    prepare_context,
    require_clean_analysis,
    verify_source,
)
from .execution import (
    compare_candidates,
    execute_pipeline,
    finalist_evidence,
    summarize_candidate,
    validate_harmony,
)
from .models import (
    AnalysisConfig,
    AnalysisInputError,
    AnnotationDecision,
    Candidate,
    Choice,
    ContextDecision,
    NeedsInput,
    RuntimeConfig,
    Study,
)
from .prompts import (
    ANNOTATE_INSTRUCTIONS,
    CONTEXT_INSTRUCTIONS,
    EXPLORE_INSTRUCTIONS,
    SELECT_INSTRUCTIONS,
)
from .provider import ProviderError, decide
from .records import (
    RecordError,
    RunRecords,
    digest,
    process_alive,
    process_identity,
    procedure_identity,
)

STAGES = (
    "inspect",
    "context",
    "preprocess",
    "explore",
    "finalists",
    "finalize",
    "annotate",
)


def stage_result(records: RunRecords, stage: str) -> Any | None:
    rows = [
        row
        for row in records.events()
        if row["kind"] == "stageCompleted" and row["stage"] == stage
    ]
    return records.read_json(rows[-1]["evidence"]) if rows else None


def _complete(records: RunRecords, stage: str, result: Any) -> None:
    done = {row["stage"] for row in records.events() if row["kind"] == "stageCompleted"}
    required = set(STAGES[: STAGES.index(stage)])
    if not required <= done:
        raise RuntimeError(f"Stage {stage} is missing its prerequisites")
    path = records.write_json(f"evidence/{stage}.json", result)
    records.append("stageCompleted", stage=stage, evidence=path)


def _start(records: RunRecords, stage: str) -> None:
    records.append("status", status="running", stage=stage)


def _answers(records: RunRecords) -> dict[str, Any]:
    result: dict[str, Any] = {}
    questions: dict[str, Any] = {}
    for row in records.events():
        for question in row.get("questions", []):
            questions[question["questionId"]] = question
        if row["kind"] == "answers":
            for key, answer in row["answers"].items():
                result[key] = {**questions[key], "answer": answer}
    return result


async def _decision(
    model: Any,
    records: RunRecords,
    runtime: RuntimeConfig,
    *,
    stage: str,
    key: str,
    evidence: dict[str, Any],
    output_type: Any,
    validate: Any,
    instructions: str,
) -> Any:
    answers = _answers(records)
    payload = {**evidence, "answers": answers} if answers else evidence
    # Evidence is scientific identity. A runtime budget change must not alter
    # saved decisions on resume; the provider separately enforces its limit.
    payload = _bounded_evidence(payload, 65_536)
    identifier = key + (":" + digest(answers)[:12] if answers else "")
    return await decide(
        model,
        decision_id=identifier,
        stage=stage,
        evidence=payload,
        output_type=output_type,
        validate=validate,
        records=records,
        runtime=runtime,
        instructions=instructions,
    )


def _bounded_evidence(evidence: dict[str, Any], limit: int) -> dict[str, Any]:
    """Trim optional detail deterministically, retaining options and gate inputs."""
    value: dict[str, Any] = json.loads(json.dumps(evidence, allow_nan=False))
    # Reserve room for the schema, instructions, provenance and repair feedback.
    budget = max(256, limit - 16_384)
    omissions: list[dict[str, Any]] = []
    priority_columns = set(value.get("diagnosticPriorityColumns", []))

    def size() -> int:
        payload = {**value, "promptOmissions": omissions} if omissions else value
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    optional = {
        "covariateAssociations",
        "loadingFamilies",
        "clusterCounts",
        "clusterSummaries",
        "byGroup",
        "matchedNames",
        "overlaps",
        "forwardOverlaps",
        "reverseOverlaps",
        "parentToCandidate",
        "candidateToParent",
        "groupComposition",
        "levels",
        "counts",
    }

    def compact(item: Any, path: str = "evidence") -> None:
        if isinstance(item, list):
            for index, child in enumerate(item):
                compact(child, f"{path}[{index}]")
        elif isinstance(item, dict):
            for key, child in item.items():
                if key == "covariateAssociations" and isinstance(child, list):
                    strongest: dict[str, dict[str, Any]] = {}
                    for row in child:
                        name = row["column"]
                        previous = strongest.get(name)
                        if previous is None or abs(row.get("association") or 0) > abs(
                            previous.get("association") or 0
                        ):
                            strongest[name] = row
                    rows = sorted(
                        strongest.values(),
                        key=lambda row: (
                            row["column"] not in priority_columns,
                            -abs(row.get("association") or 0),
                            row["column"],
                        ),
                    )[:24]
                    item[key] = rows
                    if len(rows) < len(child):
                        omissions.append(
                            {
                                "path": f"{path}.{key}",
                                "omitted": len(child) - len(rows),
                                "retained": len(rows),
                                "selection": "Strongest measured PC per column; declared roles and QC first, then association magnitude",
                            }
                        )
                elif key == "loadingFamilies" and isinstance(child, list):
                    item[key] = [
                        {**row, "topGenes": row.get("topGenes", [])[:5]}
                        for row in child[:3]
                    ]
                    omissions.append(
                        {
                            "path": f"{path}.{key}",
                            "selection": "First three PCs and five strongest loading genes each; complete family counts retained",
                        }
                    )
                elif key in {"parentToCandidate", "candidateToParent"} and isinstance(
                    child, list
                ):
                    # Preserve the strongest observed split/merge warnings in
                    # each direction; ARI and complete artifact bindings remain.
                    rows = sorted(
                        child,
                        key=lambda row: (
                            row["fraction"],
                            -row["sourceCells"],
                            str(row["clusterId"]),
                        ),
                    )[:4]
                    item[key] = rows
                    if len(rows) < len(child):
                        omissions.append(
                            {
                                "path": f"{path}.{key}",
                                "omitted": len(child) - len(rows),
                                "retained": len(rows),
                                "selection": "Four lowest directional overlap fractions, then largest source clusters",
                            }
                        )
                elif (
                    key == "levels"
                    and ".groupComposition." in path
                    and isinstance(child, list)
                ):
                    rows = sorted(
                        child, key=lambda row: (-row["count"], str(row["value"]))
                    )[:2]
                    item[key] = rows
                    if len(rows) < len(child):
                        item["omittedLevels"] = (
                            item.get("omittedLevels", 0) + len(child) - len(rows)
                        )
                        omissions.append(
                            {
                                "path": f"{path}.{key}",
                                "omitted": len(child) - len(rows),
                                "retained": len(rows),
                                "selection": "Two most frequent observed labels; missingness and full-cohort dominance retained",
                            }
                        )
                else:
                    compact(child, f"{path}.{key}")

    def trim(item: Any, path: str = "evidence") -> None:
        if not isinstance(item, dict):
            if isinstance(item, list):
                for index, child in enumerate(item):
                    trim(child, f"{path}[{index}]")
            return
        for key in sorted(item):
            child = item[key]
            if size() <= budget:
                return
            if key in optional and isinstance(child, (list, dict)) and child:
                if key == "covariateAssociations":
                    # Keep measured covariate evidence for each representation.
                    keep = min(6, len(child))
                    omissions.append(
                        {
                            "path": f"{path}.{key}",
                            "omitted": len(child) - keep,
                            "retained": keep,
                        }
                    )
                    item[key] = child[:keep]
                else:
                    omissions.append({"path": f"{path}.{key}", "omitted": len(child)})
                    item[key] = [] if isinstance(child, list) else {}
            else:
                trim(child, f"{path}.{key}")

    if size() > budget:
        compact(value)
    if size() > budget:
        trim(value)
    if omissions:
        value["promptOmissions"] = omissions
    return value


def _resolve_decision(
    records: RunRecords,
    value: Any,
    config: AnalysisConfig,
    *,
    key: str,
    stage: str,
    option_order: list[str],
    action: str = "choose",
) -> Any:
    """Record policy resolutions separately from the untouched model decision."""
    if not value.question and not (
        isinstance(value, Choice) and value.action == "defer"
    ):
        return value
    resolved = resolve_deferral(
        value,
        interaction_mode=config.interactionMode,
        option_order=option_order,
        unresolved_fact_ids=set(),
    )
    if resolved is None:
        if value.deferralReason == "unsupportedObjective":
            raise AnalysisInputError(
                "The requested objective is outside supported descriptive RNA analysis"
            )
        raise NeedsInput(value.question or "Supply the missing analysis input")
    answers = _answers(records)
    identifier = key + (":" + digest(answers)[:12] if answers else "")
    accepted = _find(records, "decisionAccepted", "decisionId", identifier)
    if accepted is None:
        raise RecordError("A policy resolution requires its saved model decision")
    payload = {
        "stage": stage,
        "acceptedSequence": accepted["sequence"],
        "acceptedDecisionId": identifier,
        "evidenceDigest": accepted["evidenceDigest"],
        "reason": value.deferralReason,
        "rule": "retainDeclaredRoles"
        if isinstance(value, ContextDecision)
        else "orderedAcceptableOption",
        "optionOrder": option_order,
        "resolved": resolved,
        "limitation": value.question,
    }
    previous = _find(records, "decisionResolved", "acceptedDecisionId", identifier)
    if previous:
        if any(previous.get(field) != item for field, item in payload.items()):
            raise RecordError(
                "Saved policy resolution no longer matches frozen evidence"
            )
    else:
        records.append("decisionResolved", **payload)
        records.append(
            "limitation", message=f"Automatic {stage} resolution: {value.question}"
        )
    if isinstance(value, ContextDecision):
        return value.model_copy(
            update={
                "columnRoles": {},
                "excludeFeatures": [],
                "question": None,
                "deferralReason": None,
            }
        )
    return value.model_copy(
        update={
            "action": action,
            "optionIds": resolved["optionIds"],
            "acceptableOptionIds": [],
            "question": None,
            "deferralReason": None,
        }
    )


def _numerical_failure(error: BaseException) -> dict[str, Any]:
    from numpy.linalg import LinAlgError
    from scipy.sparse.linalg import ArpackNoConvergence
    from scarf.datastore.pipeline_run import PipelineExecutionError

    if (
        isinstance(error, PipelineExecutionError)
        and error.stage in {"pca", "harmony"}
        and isinstance(error.__cause__, (LinAlgError, ArpackNoConvergence))
    ):
        return {
            "numericalFailure": type(error.__cause__).__name__,
            "numericalStage": error.stage,
        }
    return {}


def _find(
    records: RunRecords, kind: str, key: str, value: str
) -> dict[str, Any] | None:
    matches = [
        row for row in records.events() if row["kind"] == kind and row.get(key) == value
    ]
    return matches[-1] if matches else None


def _pipeline(
    records: RunRecords,
    source: Path,
    store: Any,
    prepared: dict[str, Any],
    study: Study,
    config: AnalysisConfig,
    runtime: RuntimeConfig,
    candidate: Candidate,
    *,
    operation: str,
    resolution: float | None = None,
    markers: bool = False,
    final: bool = False,
) -> Any:
    verify_source(source, prepared, study, config, runtime)
    completed = _find(records, "pipelineCompleted", "operation", operation)
    if completed:
        run = store.pipeline.open(run_id=completed["runId"])
        if run.status != "completed":
            raise AnalysisInputError("Recorded pipeline is no longer complete")
        return run
    planned = _find(records, "pipelinePlanned", "operation", operation)
    if planned:
        try:
            run = store.pipeline.open(label=planned["label"])
        except KeyError:
            run = None
        if run is not None:
            if run.status != "completed":
                raise AnalysisInputError("Recovered pipeline label is not complete")
            records.append(
                "pipelineCompleted",
                operation=operation,
                runId=run.run_id,
                label=planned["label"],
                recovered=True,
            )
            return run
        failed = _find(records, "pipelineFailed", "operation", operation)
        if (not failed or failed["sequence"] < planned["sequence"]) and process_alive(
            planned["process"]
        ):
            raise RuntimeError(
                "Previous pipeline process may still be alive; refuse concurrent retry"
            )
    attempt = sum(
        row["kind"] == "pipelinePlanned" and row.get("operation") == operation
        for row in records.events()
    )
    label = f"agent_{records.manifest['runId']}_{operation}_{attempt}"
    records.append(
        "pipelinePlanned",
        operation=operation,
        label=label,
        candidate=candidate.model_dump(mode="json"),
        resolution=resolution,
        markers=markers,
        final=final,
        process=process_identity(),
    )
    try:
        run = execute_pipeline(
            store,
            prepared,
            candidate,
            config,
            label=label,
            resolution=resolution,
            markers=markers,
            final=final,
        )
    except BaseException as error:
        records.append(
            "pipelineFailed",
            operation=operation,
            label=label,
            errorType=type(error).__name__,
            runId=getattr(error, "run_id", None),
            **_numerical_failure(error),
        )
        raise
    records.append(
        "pipelineCompleted",
        operation=operation,
        runId=run.run_id,
        label=label,
        recovered=False,
    )
    return run


def _native_match(
    candidate: Candidate, candidates: dict[str, Candidate]
) -> Candidate | None:
    return next(
        (
            row
            for row in candidates.values()
            if not row.useHarmony
            and (row.hvgCount, row.pcaDims, row.neighborsK)
            == (candidate.hvgCount, candidate.pcaDims, candidate.neighborsK)
        ),
        None,
    )


def _shortlist(
    ids: list[str],
    partitions: dict[str, dict[str, Any]],
    candidates: dict[str, Candidate],
    config: AnalysisConfig,
) -> list[str]:
    result = list(ids)
    for identifier in ids:
        partition = partitions[identifier]
        candidate = candidates[partition["candidateId"]]
        if not candidate.useHarmony:
            continue
        native = _native_match(candidate, candidates)
        match = next(
            (
                key
                for key, row in partitions.items()
                if native
                and row["candidateId"] == native.candidateId
                and row["resolution"] == partition["resolution"]
            ),
            None,
        )
        if match is None:
            raise AnalysisInputError(
                "A corrected finalist needs its matched native partition"
            )
        if match not in result:
            result.append(match)
    if len(result) > config.maxFinalists:
        raise AnalysisInputError(
            "Shortlist exceeds the finalist budget including required native control"
        )
    return result


def _coverage(records: RunRecords, slots: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "slots": slots,
        "nativeComplete": all(
            row["status"] == "measured" for row in slots if row["axis"] != "harmony"
        ),
    }
    previous = records.latest("explorationCoverage")
    if previous is None or previous["coverage"] != result:
        records.append("explorationCoverage", coverage=result)
    return result


def _skippable_failure(
    records: RunRecords, store: Any, candidate: Candidate, parent: dict[str, Any] | None
) -> dict[str, Any] | None:
    failure = _find(
        records, "pipelineFailed", "operation", f"screen_{candidate.candidateId}"
    )
    if (
        parent is None
        or failure is None
        or failure.get("numericalFailure") not in {"LinAlgError", "ArpackNoConvergence"}
    ):
        return None
    planned = _find(
        records, "pipelinePlanned", "operation", f"screen_{candidate.candidateId}"
    )
    if planned is None or failure["sequence"] < planned["sequence"]:
        return None
    failed = store.pipeline.open(run_id=failure["runId"])
    if failed.status != "failed":
        raise AnalysisInputError(
            "A skipped numerical attempt must have a durable failed outcome"
        )
    original = store.pipeline.open(run_id=parent["runId"])
    if original.status != "completed" or any(
        not store.inspect_artifact(ref).complete for ref in original.values()
    ):
        raise AnalysisInputError(
            "A numerical fallback requires intact completed parent artifacts"
        )
    return failure


def _measure_candidate(
    records: RunRecords,
    source: Path,
    store: Any,
    prepared: dict[str, Any],
    study: Study,
    config: AnalysisConfig,
    runtime: RuntimeConfig,
    candidate: Candidate,
    *,
    parent: dict[str, Any] | None = None,
    parent_candidate: Candidate | None = None,
    resolution: float | None = None,
) -> dict[str, Any] | None:
    verify_source(source, prepared, study, config, runtime)
    previous = _find(records, "candidateMeasured", "candidateId", candidate.candidateId)
    if previous:
        saved: dict[str, Any] = records.read_json(previous["evidence"])
        _validate_measurement(store, saved)
        return saved
    skipped = _find(records, "candidateFailed", "candidateId", candidate.candidateId)
    if skipped:
        if _skippable_failure(records, store, candidate, parent) is None:
            raise RecordError("A saved numerical skip lacks its failed attempt")
        return None
    failure = (
        _skippable_failure(records, store, candidate, parent)
        if config.interactionMode == "lenient"
        else None
    )
    if failure is None:
        if (
            _find(records, "candidateAdmitted", "candidateId", candidate.candidateId)
            is None
        ):
            records.append(
                "candidateAdmitted",
                candidateId=candidate.candidateId,
                candidate=candidate.model_dump(mode="json"),
            )
        try:
            run = _pipeline(
                records,
                source,
                store,
                prepared,
                study,
                config,
                runtime,
                candidate,
                operation=f"screen_{candidate.candidateId}",
                resolution=resolution,
            )
        except Exception:
            verify_source(source, prepared, study, config, runtime)
            failure = (
                _skippable_failure(records, store, candidate, parent)
                if config.interactionMode == "lenient"
                else None
            )
            if failure is None:
                raise
        else:
            summary = summarize_candidate(store, run, candidate, prepared, config)
            summary["artifactFingerprint"] = digest(
                {key: ref.to_dict() for key, ref in run.items()}
            )
            if parent is not None:
                if summary["selection"] != parent["selection"]:
                    raise AnalysisInputError(
                        "Candidate comparisons must share the exact frozen cohort"
                    )
                original = store.pipeline.open(run_id=parent["runId"])
                assert parent_candidate is not None
                summary["comparisons"] = compare_candidates(
                    store,
                    original,
                    run,
                    parent_candidate,
                    candidate,
                    config
                    if resolution is None
                    else config.model_copy(update={"resolutions": (resolution,)}),
                )
            path = records.write_json(
                f"evidence/candidate_{candidate.candidateId}.json", summary
            )
            records.append(
                "candidateMeasured", candidateId=candidate.candidateId, evidence=path
            )
            return summary
    records.append(
        "candidateFailed",
        candidateId=candidate.candidateId,
        failureSequence=failure["sequence"],
        reason=failure["numericalFailure"],
        stage=failure["numericalStage"],
    )
    records.append(
        "limitation",
        message=f"{candidate.candidateId}: numerical probe failed; sensitivity coverage is incomplete.",
    )
    return None


def _validate_measurement(store: Any, evidence: dict[str, Any]) -> None:
    run = store.pipeline.open(run_id=evidence["runId"])
    if run.status != "completed" or digest(
        {key: ref.to_dict() for key, ref in run.items()}
    ) != evidence.get("artifactFingerprint"):
        raise AnalysisInputError(
            "Saved comparison evidence no longer matches its complete numerical run"
        )
    if any(not store.inspect_artifact(ref).complete for ref in run.values()):
        raise AnalysisInputError(
            "Saved comparison evidence contains incomplete artifacts"
        )


def _partition_order(
    partitions: dict[str, Any], candidates: dict[str, Candidate], config: AnalysisConfig
) -> list[str]:
    return sorted(
        partitions,
        key=lambda name: (
            candidates[partitions[name]["candidateId"]].useHarmony,
            list(candidates).index(partitions[name]["candidateId"]),
            list(config.resolutions).index(partitions[name]["resolution"]),
            name,
        ),
    )


def _validate_exploration(
    value: Choice, evidence: dict[str, Any], config: AnalysisConfig
) -> None:
    pc = evidence.get("decisionKind") == "pcProbe"
    experiments = evidence["experiments"]
    partitions = {
        row["optionId"]: row
        for summary in evidence["candidates"]
        for row in summary["partitions"]
    }
    validate_choice(
        value,
        options=set(experiments)
        if pc or value.action == "experiment"
        else set(partitions),
        actions={"experiment", "defer"}
        if pc
        else {"shortlist", "defer"} | ({"experiment"} if experiments else set()),
        maximum=1 if pc or value.action == "experiment" else config.maxFinalists,
        evidence_ids=set(evidence["evidenceIds"]),
        unresolved_fact_ids=set(evidence.get("unresolvedFactIds", [])),
    )
    if value.action == "shortlist":
        candidates = {
            row["candidateId"]: Candidate.model_validate(row)
            for row in evidence["candidateRecipes"]
        }
        _shortlist(value.optionIds, partitions, candidates, config)


async def _explore(
    model: Any,
    records: RunRecords,
    source: Path,
    store: Any,
    prepared: dict[str, Any],
    study: Study,
    config: AnalysisConfig,
    runtime: RuntimeConfig,
) -> dict[str, Any]:
    baseline = Candidate(
        candidateId="c0",
        hvgCount=config.hvgCount,
        pcaDims=config.pcaDims,
        neighborsK=config.neighborsK,
    )
    measured = _measure_candidate(
        records, source, store, prepared, study, config, runtime, baseline
    )
    assert measured is not None
    if not measured["partitions"]:
        raise AnalysisInputError(
            "No valid baseline partitions were produced; inspect the numerical pipeline report"
        )
    candidates = {"c0": baseline}
    summaries = [measured]
    planned = records.latest("explorationPlanned")
    if planned:
        plan = records.read_json(planned["evidence"])
    else:
        options = native_probe_options(
            baseline,
            {
                **prepared,
                "actualHvgCount": measured.get("actualHvgCount", baseline.hvgCount),
            },
        )
        pc_options = options["pcaDims"]
        pc_order = sorted(
            pc_options, key=lambda name: (pc_options[name].pcaDims != 30, name)
        )
        pc_id = pc_order[0] if pc_order else None
        if len(pc_options) > 1:
            evidence: dict[str, Any] = {
                "decisionKind": "pcProbe",
                "allowedActions": ["experiment", "defer"],
                "objective": study.objective,
                "candidates": [measured],
                "candidateRecipes": [baseline.model_dump(mode="json")],
                "experiments": {
                    name: row.model_dump(mode="json")
                    for name, row in pc_options.items()
                },
                "evidenceIds": [
                    "c0",
                    *[row["optionId"] for row in measured["partitions"]],
                ],
                "optionOrder": pc_order,
                "unresolvedFactIds": [],
                "diagnosticPriorityColumns": list(
                    dict.fromkeys(
                        [
                            *(
                                row["column"]
                                for row in prepared.get("resolvedRoles", [])
                            ),
                            *prepared.get("qcFlags", {}),
                        ]
                    )
                ),
            }
            choice = await _decision(
                model,
                records,
                runtime,
                stage="explore",
                key="choose_pc_probe",
                evidence=evidence,
                output_type=Choice,
                validate=lambda value: _validate_exploration(value, evidence, config),
                instructions=EXPLORE_INSTRUCTIONS,
            )
            choice = _resolve_decision(
                records,
                choice,
                config,
                key="choose_pc_probe",
                stage="explore",
                option_order=pc_order,
                action="experiment",
            )
            pc_id = choice.optionIds[0]
        slots = [
            {
                "candidateId": "c0",
                "axis": "baseline",
                "parentId": None,
                "status": "measured",
                "reason": None,
                "parameters": baseline.model_dump(mode="json"),
            }
        ]
        for index, axis in enumerate(("hvgCount", "pcaDims", "neighborsK"), start=1):
            identifier = pc_id if axis == "pcaDims" else next(iter(options[axis]), None)
            candidate = (
                options[axis][identifier].model_copy(
                    update={"candidateId": f"c{index}"}
                )
                if identifier
                else None
            )
            slots.append(
                {
                    "candidateId": f"c{index}",
                    "axis": axis,
                    "parentId": "c0",
                    "status": "pending" if candidate else "infeasible",
                    "reason": None
                    if candidate
                    else "No feasible distinct registered setting",
                    "parameters": candidate.model_dump(mode="json")
                    if candidate
                    else None,
                }
            )
        plan = {
            "slots": slots,
            "resolutions": list(config.resolutions),
            "seed": config.randomSeed,
            "harmonyPermitted": config.maxCandidates == 5,
        }
        path = records.write_json("evidence/probe_plan.json", plan)
        records.append("explorationPlanned", evidence=path)
    slots = json.loads(json.dumps(plan["slots"]))
    _coverage(records, slots)
    for slot in slots[1:]:
        if slot["parameters"] is None:
            continue
        candidate = Candidate.model_validate(slot["parameters"])
        candidates[candidate.candidateId] = candidate
        result = _measure_candidate(
            records,
            source,
            store,
            prepared,
            study,
            config,
            runtime,
            candidate,
            parent=measured,
            parent_candidate=baseline,
        )
        slot["status"] = "measured" if result else "failed"
        if result:
            summaries.append(result)
            if (
                slot["axis"] == "hvgCount"
                and result.get("hvgSelectionDigest")
                and result["hvgSelectionDigest"] == measured.get("hvgSelectionDigest")
            ):
                slot.update(
                    status="infeasible",
                    reason="Measured HVG selection is identical to the baseline; no distinct sensitivity comparison",
                )
        else:
            slot["reason"] = "Recognized numerical failure; no automatic retry"
        _coverage(records, slots)
    partitions = {
        row["optionId"]: row for summary in summaries for row in summary["partitions"]
    }
    harmony = {}
    if (
        plan["harmonyPermitted"]
        and prepared["correctionEligible"]
        and config.scoreDoublets
        and config.maxFinalists == 2
    ):
        for name, partition in partitions.items():
            parent = candidates[partition["candidateId"]]
            harmony[f"{name}:harmony"] = {
                "nativeOptionId": name,
                "candidate": parent.model_copy(
                    update={
                        "candidateId": "c4",
                        "parentId": parent.candidateId,
                        "useHarmony": True,
                    }
                ).model_dump(mode="json"),
                "resolution": partition["resolution"],
            }
    order = _partition_order(partitions, candidates, config)
    evidence = {
        "decisionKind": "nativeShortlist",
        "allowedActions": ["shortlist", "experiment", "defer"]
        if harmony
        else ["shortlist", "defer"],
        "objective": study.objective,
        "candidates": summaries,
        "candidateRecipes": [
            row.model_dump(mode="json") for row in candidates.values()
        ],
        "experiments": harmony,
        "maxFinalists": config.maxFinalists,
        "explorationCoverage": _coverage(records, slots),
        "evidenceIds": [*candidates, *partitions],
        "optionOrder": order,
        "unresolvedFactIds": [],
        "diagnosticPriorityColumns": list(
            dict.fromkeys(
                [
                    *(row["column"] for row in prepared.get("resolvedRoles", [])),
                    *prepared.get("qcFlags", {}),
                ]
            )
        ),
    }
    choice = await _decision(
        model,
        records,
        runtime,
        stage="explore",
        key="shortlist_native",
        evidence=evidence,
        output_type=Choice,
        validate=lambda value: _validate_exploration(value, evidence, config),
        instructions=EXPLORE_INSTRUCTIONS,
    )
    choice = _resolve_decision(
        records,
        choice,
        config,
        key="shortlist_native",
        stage="explore",
        option_order=order,
        action="shortlist",
    )
    if choice.action == "experiment":
        selected = harmony[choice.optionIds[0]]
        candidate = Candidate.model_validate(selected["candidate"])
        candidates[candidate.candidateId] = candidate
        native = next(
            row for row in summaries if row["candidateId"] == candidate.parentId
        )
        slot = {
            "candidateId": "c4",
            "axis": "harmony",
            "parentId": candidate.parentId,
            "status": "pending",
            "reason": None,
            "parameters": candidate.model_dump(mode="json"),
        }
        slots.append(slot)
        _coverage(records, slots)
        result = _measure_candidate(
            records,
            source,
            store,
            prepared,
            study,
            config,
            runtime,
            candidate,
            parent=native,
            parent_candidate=candidates[str(candidate.parentId)],
            resolution=selected["resolution"],
        )
        ids = [selected["nativeOptionId"]]
        if result:
            summaries.append(result)
            partitions.update({row["optionId"]: row for row in result["partitions"]})
            corrected = f"c4:r{selected['resolution']:g}"
            if corrected not in partitions:
                raise AnalysisInputError("Corrected screen lacks its pinned resolution")
            ids.append(corrected)
            slot["status"] = "measured"
        else:
            slot.update(
                status="failed",
                reason="Correction failed numerically; retaining its native counterpart",
            )
        _coverage(records, slots)
    else:
        ids = _shortlist(choice.optionIds, partitions, candidates, config)
    return {
        "candidates": {
            key: row.model_dump(mode="json") for key, row in candidates.items()
        },
        "summaries": summaries,
        "partitions": partitions,
        "shortlist": ids,
        "explorationCoverage": _coverage(records, slots),
    }


async def run_workflow(
    records: RunRecords,
    source: Path,
    model: Any,
    study: Study,
    config: AnalysisConfig,
    runtime: RuntimeConfig,
) -> None:
    """Resume completed stages and execute the remaining fixed procedure."""
    stage = "inspect"
    records.append(
        "invocationStarted",
        invocationId=uuid4().hex,
        runtime=runtime.model_dump(mode="json", exclude_unset=True),
        process=process_identity(),
    )
    try:
        inspected = stage_result(records, "inspect")
        if inspected is None:
            _start(records, stage)
            inspected = inspect_source(source, study, config, runtime)
            _complete(records, stage, inspected)
        verify_source(source, inspected, study, config, runtime)

        stage = "context"
        context = stage_result(records, stage)
        if context is None:
            _start(records, stage)
            public = inspected["contextEvidence"]
            columns = set(public.get("columns", {}))
            decision = await _decision(
                model,
                records,
                runtime,
                stage=stage,
                key=stage,
                evidence={
                    "study": study.model_dump(
                        mode="json", exclude={"referenceFiles", "excludedColumns"}
                    ),
                    "observations": public,
                    "evidenceIds": sorted(context_evidence_ids(columns, study)),
                    "unresolvedFactIds": [],
                    "optionOrder": [],
                },
                output_type=ContextDecision,
                validate=lambda value: validate_context(
                    value, columns=columns, study=study
                ),
                instructions=CONTEXT_INSTRUCTIONS,
            )
            decision = _resolve_decision(
                records, decision, config, key=stage, stage=stage, option_order=[]
            )
            context = decision.model_dump(mode="json")
            _complete(records, stage, context)

        stage = "preprocess"
        prepared = stage_result(records, stage)
        if prepared is None:
            _start(records, stage)
            inspected = {**inspected, "source": str(source)}
            prepared = prepare_context(
                inspected,
                ContextDecision.model_validate(context),
                study,
                config,
                runtime,
            )
            _complete(records, stage, prepared)
        prepared = {**prepared, "source": str(source)}
        verify_source(source, prepared, study, config, runtime)
        if records.latest("pipelinePlanned") is None:
            # A paused context decision does not reserve a clean store forever.
            require_clean_analysis(
                open_store(source, config, runtime), prepared["assay"]
            )
        store = open_store(source, config, runtime, writable=True)

        stage = "explore"
        exploration = stage_result(records, stage)
        if exploration is None:
            _start(records, stage)
            exploration = await _explore(
                model, records, source, store, prepared, study, config, runtime
            )
            _complete(records, stage, exploration)
        candidates = {
            key: Candidate.model_validate(row)
            for key, row in exploration["candidates"].items()
        }
        partitions = exploration["partitions"]

        stage = "finalists"
        assessed = stage_result(records, stage)
        if assessed is None:
            _start(records, stage)
            finalists = {}
            for index, option in enumerate(exploration["shortlist"]):
                partition = partitions[option]
                candidate = candidates[partition["candidateId"]]
                existing = _find(records, "finalistMeasured", "optionId", option)
                if existing:
                    finalists[option] = records.read_json(existing["evidence"])
                    _validate_measurement(store, finalists[option])
                    continue
                run = _pipeline(
                    records,
                    source,
                    store,
                    prepared,
                    study,
                    config,
                    runtime,
                    candidate,
                    operation=f"finalist_{index}",
                    resolution=partition["resolution"],
                    markers=True,
                )
                evidence = finalist_evidence(store, run, candidate, prepared, config)
                evidence["artifactFingerprint"] = digest(
                    {key: ref.to_dict() for key, ref in run.items()}
                )
                evidence["resolution"] = partition["resolution"]
                path = records.write_json(f"evidence/finalist_{index}.json", evidence)
                records.append("finalistMeasured", optionId=option, evidence=path)
                finalists[option] = evidence
            rejected = {}
            eligible = set(finalists)
            for option, evidence in finalists.items():
                candidate = candidates[evidence["candidateId"]]
                if candidate.useHarmony:
                    native = _native_match(candidate, candidates)
                    control = next(
                        (
                            row
                            for row in finalists.values()
                            if native
                            and row["candidateId"] == native.candidateId
                            and row["resolution"] == evidence["resolution"]
                        ),
                        None,
                    )
                    reasons = (
                        validate_harmony(control, evidence)
                        if control
                        else ["Matched native evidence is missing"]
                    )
                    if reasons:
                        eligible.discard(option)
                        rejected[option] = reasons
            if not eligible:
                raise AnalysisInputError(
                    "No scientifically admissible finalist remains"
                )
            option_order = [
                option
                for option in _partition_order(partitions, candidates, config)
                if option in eligible
            ]
            choice = await _decision(
                model,
                records,
                runtime,
                stage=stage,
                key="choose_finalist",
                evidence={
                    "objective": study.objective,
                    "allowedActions": ["choose", "defer"],
                    "evidenceIds": sorted(finalists),
                    "finalists": {
                        key: _selection_evidence(value)
                        for key, value in finalists.items()
                    },
                    "eligibleOptions": sorted(eligible),
                    "rejectedOptions": rejected,
                    "optionOrder": option_order,
                    "unresolvedFactIds": [],
                },
                output_type=Choice,
                validate=lambda value: validate_choice(
                    value,
                    options=eligible,
                    actions={"choose", "defer"},
                    evidence_ids=set(finalists),
                ),
                instructions=SELECT_INSTRUCTIONS,
            )
            choice = _resolve_decision(
                records,
                choice,
                config,
                key="choose_finalist",
                stage=stage,
                option_order=option_order,
            )
            selected = choice.optionIds[0]
            assessed = {
                "selected": selected,
                "finalists": finalists,
                "rejected": rejected,
            }
            _complete(records, stage, assessed)

        stage = "finalize"
        final = stage_result(records, stage)
        if final is None:
            _start(records, stage)
            partition = partitions[assessed["selected"]]
            candidate = candidates[partition["candidateId"]]
            run = _pipeline(
                records,
                source,
                store,
                prepared,
                study,
                config,
                runtime,
                candidate,
                operation="final",
                resolution=partition["resolution"],
                markers=True,
                final=True,
            )
            finalist = store.pipeline.open(
                run_id=assessed["finalists"][assessed["selected"]]["runId"]
            )
            for key in ("analysis_cell_selection", "clusters", "markers"):
                if run[key] != finalist[key]:
                    raise AnalysisInputError(
                        f"Final artifact {key} does not match the accepted finalist"
                    )
            for reference in run.values():
                if not store.inspect_artifact(reference).complete:
                    raise AnalysisInputError("Final run contains incomplete artifacts")
            final = {
                "runId": run.run_id,
                "selected": assessed["selected"],
                "artifacts": {key: ref.to_dict() for key, ref in run.items()},
            }
            _complete(records, stage, final)

        validate_final(store, final)

        stage = "annotate"
        annotated = stage_result(records, stage)
        if annotated is None:
            _start(records, stage)
            clusters = assessed["finalists"][assessed["selected"]]["clusters"]
            rows: list[dict[str, Any]] = []
            for start in range(0, len(clusters), 8):
                batch = clusters[start : start + 8]
                decision = await _decision(
                    model,
                    records,
                    runtime,
                    stage=stage,
                    key=f"annotations_{start // 8}",
                    evidence={
                        "organism": study.organism,
                        "tissue": study.tissue,
                        "context": study.context,
                        "objective": study.objective,
                        "references": inspected["contextEvidence"].get(
                            "references", []
                        ),
                        "clusters": batch,
                        "doubletsAssessed": config.scoreDoublets,
                    },
                    output_type=AnnotationDecision,
                    validate=lambda value, current=batch: validate_annotations(
                        value, clusters=current
                    ),
                    instructions=ANNOTATE_INSTRUCTIONS,
                )
                rows.extend(row.model_dump(mode="json") for row in decision.annotations)
            annotated = {
                "runId": final["runId"],
                "clusters": final["artifacts"]["clusters"],
                "annotations": rows,
            }
            _complete(records, stage, annotated)
        records.append("status", status="completed", stage="annotate")
    except NeedsInput as error:
        question = {
            "questionId": digest(
                {"stage": stage, "question": error.question, "field": error.field}
            )[:16],
            "question": error.question,
            "field": error.field,
            "stage": stage,
        }
        records.append("status", status="needsInput", stage=stage, questions=[question])
    except (KeyboardInterrupt, asyncio.CancelledError):
        records.append("status", status="interrupted", stage=stage)
        raise
    except Exception as error:
        # Provider errors are already normalized and numerical details remain in
        # their pipeline report. Avoid persisting arbitrary exception bodies.
        records.append(
            "status",
            status="failed",
            stage=stage,
            errorType=type(error).__name__,
            message=str(error)[:2000]
            if isinstance(error, AnalysisInputError | ProviderError)
            else "Execution failed. Inspect the saved model attempts and pipeline reports before resuming.",
        )


def validate_final(store: Any, final: dict[str, Any]) -> None:
    """A saved stage never makes missing or changed numerical artifacts complete."""
    run = store.pipeline.open(run_id=final["runId"])
    if (
        run.status != "completed"
        or {key: ref.to_dict() for key, ref in run.items()} != final["artifacts"]
    ):
        raise AnalysisInputError(
            "Final pipeline no longer matches its saved artifact mapping"
        )
    if any(not store.inspect_artifact(ref).complete for ref in run.values()):
        raise AnalysisInputError("Final pipeline contains incomplete artifacts")


def _selection_evidence(value: dict[str, Any]) -> dict[str, Any]:
    """Bound selection prompts while retaining complete marker evidence on disk."""
    clusters = value["clusters"]
    ordered = sorted(clusters, key=lambda row: row["count"], reverse=True)
    weak = sorted(clusters, key=lambda row: row.get("qualifyingMarkerCount", 0))
    selected = {
        row["clusterId"]: row for row in [*ordered[:4], *ordered[-4:], *weak[:4]]
    }
    return {
        **{key: item for key, item in value.items() if key != "clusters"},
        "clusterCount": len(clusters),
        "omittedClusterCount": len(clusters) - len(selected),
        "clusterSelection": "four largest, four smallest, four fewest qualifying markers",
        "clusterSummaries": [
            {
                key: row[key]
                for key in (
                    "clusterId",
                    "count",
                    "qualifyingMarkerCount",
                    "sampleCount",
                    "qcSummary",
                )
                if key in row
            }
            for row in clusters[:64]
        ],
        "omittedClusterSummaryCount": max(0, len(clusters) - 64),
        "clusters": [
            {
                **row,
                "markers": row["markers"][:6],
                "weakMarkers": row.get("weakMarkers", [])[:3],
            }
            for row in selected.values()
        ],
    }


def replay(records: RunRecords) -> list[dict[str, Any]]:
    """Replay only saved decisions, applying the same pure semantic constraints."""
    from .provider import replay_decisions

    if records.manifest["procedureIdentity"] != procedure_identity():
        raise ValueError(
            "Scarf procedure changed; replay requires the original implementation and prompts"
        )
    supplied = records.latest("inputsResolved") or records.manifest
    study = Study.model_validate(supplied["study"])
    config = AnalysisConfig.model_validate(supplied["config"])

    def context(value: Any, evidence: dict[str, Any]) -> None:
        validate_context(
            value, columns=set(evidence["observations"]["columns"]), study=study
        )

    def explore(value: Any, evidence: dict[str, Any]) -> None:
        _validate_exploration(value, evidence, config)

    def select(value: Any, evidence: dict[str, Any]) -> None:
        validate_choice(
            value,
            options=set(evidence["eligibleOptions"]),
            actions={"choose", "defer"},
            evidence_ids=set(evidence["finalists"]),
        )

    def annotate(value: Any, evidence: dict[str, Any]) -> None:
        validate_annotations(value, clusters=evidence["clusters"])

    results = replay_decisions(
        records,
        schemas={
            "context": ContextDecision,
            "explore": Choice,
            "finalists": Choice,
            "annotate": AnnotationDecision,
        },
        validators={
            "context": context,
            "explore": explore,
            "finalists": select,
            "annotate": annotate,
        },
    )

    for row in results:
        row["source"] = "model"
    events = records.events()
    accepted = {
        row["sequence"]: row for row in events if row["kind"] == "decisionAccepted"
    }
    requests = {row["callId"]: row for row in events if row["kind"] == "modelRequest"}
    for row in events:
        if row["kind"] != "decisionResolved":
            continue
        original = accepted.get(row["acceptedSequence"])
        if (
            original is None
            or original["decisionId"] != row["acceptedDecisionId"]
            or original["evidenceDigest"] != row["evidenceDigest"]
        ):
            raise RecordError("Policy resolution does not match an accepted decision")
        schema = ContextDecision if original["stage"] == "context" else Choice
        value = schema.model_validate(original["output"])
        request = records.read_json(requests[original["callId"]]["requestPath"])
        evidence = json.loads(request["userPrompt"])["evidence"]
        if (
            row["optionOrder"] != evidence["optionOrder"]
            or row["reason"] != value.deferralReason
        ):
            raise RecordError(
                "Policy resolution differs from its frozen preference or reason"
            )
        resolved = resolve_deferral(
            value,
            interaction_mode=config.interactionMode,
            option_order=evidence["optionOrder"],
            unresolved_fact_ids=set(evidence.get("unresolvedFactIds", [])),
        )
        if resolved is None or row["resolved"] != resolved:
            raise RecordError("Policy resolution does not reproduce its frozen choice")
        results.append(
            {
                "decisionId": row["acceptedDecisionId"],
                "stage": row["stage"],
                "source": "policy",
                "valid": True,
            }
        )
    _replay_exploration(records, config)
    return results


def _replay_exploration(records: RunRecords, config: AnalysisConfig) -> None:
    planned = records.latest("explorationPlanned")
    if planned is None:
        return
    plan = records.read_json(planned["evidence"])
    slots = plan["slots"]
    if len(slots) != 4 or [row["axis"] for row in slots] != [
        "baseline",
        "hvgCount",
        "pcaDims",
        "neighborsK",
    ]:
        raise RecordError("Frozen exploration must contain all four native slots")
    if (
        plan["resolutions"] != list(config.resolutions)
        or plan["seed"] != config.randomSeed
        or plan["harmonyPermitted"] != (config.maxCandidates == 5)
    ):
        raise RecordError("Frozen exploration policy changed")
    baseline = Candidate(
        candidateId="c0",
        hvgCount=config.hvgCount,
        pcaDims=config.pcaDims,
        neighborsK=config.neighborsK,
    )
    if slots[0]["parameters"] != baseline.model_dump(mode="json"):
        raise RecordError("Frozen baseline differs from the scientific configuration")
    prepared = stage_result(records, "preprocess")
    if prepared is None:
        raise RecordError("Frozen exploration is missing its preprocessing evidence")
    baseline_event = _find(records, "candidateMeasured", "candidateId", "c0")
    if baseline_event is None:
        raise RecordError("Probe plan lacks its measured baseline")
    baseline_evidence = records.read_json(baseline_event["evidence"])
    offered = native_probe_options(
        baseline,
        {
            **prepared,
            "actualHvgCount": baseline_evidence.get(
                "actualHvgCount", baseline.hvgCount
            ),
        },
    )
    expected_pc = next(iter(offered["pcaDims"]), None)
    if len(offered["pcaDims"]) > 1:
        accepted = [
            row
            for row in records.events()
            if row["kind"] == "decisionAccepted"
            and row["decisionId"].split(":")[0] == "choose_pc_probe"
        ]
        if not accepted:
            raise RecordError("Probe plan lacks its PC direction decision")
        latest = accepted[-1]
        resolved = _find(
            records, "decisionResolved", "acceptedDecisionId", latest["decisionId"]
        )
        effective = resolved["resolved"] if resolved else latest["output"]
        expected_pc = effective["optionIds"][0]
    for index, slot in enumerate(slots[1:], start=1):
        if slot["candidateId"] != f"c{index}" or slot["parentId"] != "c0":
            raise RecordError("A native probe has a different parent or position")
        alternatives = offered[slot["axis"]]
        if slot["parameters"] is None:
            if alternatives:
                raise RecordError("A feasible probe was omitted from the frozen plan")
            continue
        candidate = Candidate.model_validate(slot["parameters"])
        selected_id = (
            expected_pc if slot["axis"] == "pcaDims" else next(iter(alternatives), None)
        )
        expected = (
            alternatives[selected_id].model_copy(
                update={"candidateId": slot["candidateId"]}
            )
            if selected_id in alternatives
            else None
        )
        if expected is None or candidate != expected:
            raise RecordError(
                "A frozen probe differs from its registered deterministic or model choice"
            )
    recipes = {
        row["candidateId"]: row["parameters"]
        for row in slots
        if row["parameters"] is not None
    }
    for event in records.events():
        if event["kind"] != "candidateAdmitted":
            continue
        candidate = Candidate.model_validate(event["candidate"])
        if candidate.candidateId == "c4":
            parents = recipes.get(str(candidate.parentId))
            if (
                not plan["harmonyPermitted"]
                or parents is None
                or candidate
                != Candidate.model_validate(parents).model_copy(
                    update={
                        "candidateId": "c4",
                        "parentId": candidate.parentId,
                        "useHarmony": True,
                    }
                )
            ):
                raise RecordError(
                    "Corrected candidate lacks its permitted native parent"
                )
            recipes["c4"] = candidate.model_dump(mode="json")
        elif candidate.model_dump(mode="json") != recipes.get(candidate.candidateId):
            raise RecordError("Admitted candidate differs from the frozen probe plan")
        event_measured = _find(
            records, "candidateMeasured", "candidateId", candidate.candidateId
        )
        if event_measured:
            measured = records.read_json(event_measured["evidence"])
            if measured.get("parameters") != candidate.model_dump(
                exclude={"candidateId", "parentId"}
            ):
                raise RecordError(
                    "Measured candidate parameters differ from its admission"
                )
    coverage = records.latest("explorationCoverage")
    if coverage:
        for slot in coverage["coverage"]["slots"]:
            expected = recipes.get(slot["candidateId"])
            if slot["parameters"] != expected:
                raise RecordError(
                    "Exploration coverage differs from the frozen candidate recipe"
                )
            if (
                slot["status"] == "measured"
                and _find(
                    records, "candidateMeasured", "candidateId", slot["candidateId"]
                )
                is None
            ):
                raise RecordError("Exploration coverage claims an unmeasured candidate")
            if (
                slot["status"] == "failed"
                and _find(
                    records, "candidateFailed", "candidateId", slot["candidateId"]
                )
                is None
            ):
                raise RecordError("Exploration coverage claims an unrecorded failure")
            if (records.latest("status") or {}).get("status") == "completed" and slot[
                "status"
            ] == "pending":
                raise RecordError("Completed analysis has an unfinished required probe")
