"""A fixed RNA procedure. Decisions happen between complete pipeline calls."""

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from .choices import (
    alternatives,
    validate_annotations,
    validate_choice,
    validate_context,
)
from .evidence import inspect_source, open_store, prepare_context, verify_source
from .execution import (
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
from .records import RunRecords, digest, process_alive, process_identity

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
    admissions = [row for row in records.events() if row["kind"] == "candidateAdmitted"]
    if not admissions:
        baseline = Candidate(
            candidateId="c0",
            hvgCount=config.hvgCount,
            pcaDims=config.pcaDims,
            neighborsK=config.neighborsK,
        )
        records.append("candidateAdmitted", candidate=baseline.model_dump(mode="json"))
    while True:
        candidates = {
            row["candidate"]["candidateId"]: Candidate.model_validate(row["candidate"])
            for row in records.events()
            if row["kind"] == "candidateAdmitted"
        }
        summaries = []
        for candidate in candidates.values():
            previous = _find(
                records, "candidateMeasured", "candidateId", candidate.candidateId
            )
            if previous:
                summaries.append(records.read_json(previous["evidence"]))
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
                operation=f"screen_{candidate.candidateId}",
            )
            summary = summarize_candidate(store, run, candidate, prepared, config)
            if summaries and summary["selection"] != summaries[0]["selection"]:
                raise AnalysisInputError(
                    "Candidate comparisons must share the exact frozen cohort"
                )
            path = records.write_json(
                f"evidence/candidate_{candidate.candidateId}.json", summary
            )
            records.append(
                "candidateMeasured", candidateId=candidate.candidateId, evidence=path
            )
            summaries.append(summary)
        partitions = {
            row["optionId"]: row
            for summary in summaries
            for row in summary["partitions"]
        }
        if not partitions:
            raise NeedsInput(
                "No valid partitions were produced; inspect the numerical pipeline reports"
            )
        options = (
            alternatives(list(candidates.values()), prepared, config)
            if len(candidates) < config.maxCandidates
            else {}
        )
        actions = {"shortlist", "defer"} | ({"experiment"} if options else set())

        def validate(value: Choice) -> None:
            validate_choice(
                value,
                options=set(options)
                if value.action == "experiment"
                else set(partitions),
                actions=actions,
                maximum=1 if value.action == "experiment" else config.maxFinalists,
                evidence_ids=set(partitions) | set(candidates),
            )
            if value.action == "shortlist":
                _shortlist(value.optionIds, partitions, candidates, config)

        decision = await _decision(
            model,
            records,
            runtime,
            stage="explore",
            key=f"explore_{len(candidates)}",
            evidence={
                "objective": study.objective,
                "candidates": summaries,
                "experiments": {
                    key: row.model_dump(mode="json") for key, row in options.items()
                },
                "maxFinalists": config.maxFinalists,
            },
            output_type=Choice,
            validate=validate,
            instructions=EXPLORE_INSTRUCTIONS,
        )
        if decision.action == "experiment":
            selected = options[decision.optionIds[0]].model_copy(
                update={"candidateId": f"c{len(candidates)}"}
            )
            records.append(
                "candidateAdmitted", candidate=selected.model_dump(mode="json")
            )
            continue
        if decision.action == "defer":
            raise NeedsInput(decision.question or "Clarify the requested experiment")
        ids = _shortlist(decision.optionIds, partitions, candidates, config)
        return {
            "candidates": {
                key: row.model_dump(mode="json") for key, row in candidates.items()
            },
            "summaries": summaries,
            "partitions": partitions,
            "shortlist": ids,
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
                },
                output_type=ContextDecision,
                validate=lambda value: validate_context(
                    value, columns=columns, study=study
                ),
                instructions=CONTEXT_INSTRUCTIONS,
            )
            if decision.question:
                raise NeedsInput(decision.question)
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
                raise NeedsInput("No scientifically admissible finalist remains")
            choice = await _decision(
                model,
                records,
                runtime,
                stage=stage,
                key="choose_finalist",
                evidence={
                    "objective": study.objective,
                    "finalists": {
                        key: _selection_evidence(value)
                        for key, value in finalists.items()
                    },
                    "eligibleOptions": sorted(eligible),
                    "rejectedOptions": rejected,
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
            if choice.action == "defer":
                raise NeedsInput(choice.question or "Clarify finalist selection")
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

    supplied = records.latest("inputsResolved") or records.manifest
    study = Study.model_validate(supplied["study"])
    config = AnalysisConfig.model_validate(supplied["config"])

    def context(value: Any, evidence: dict[str, Any]) -> None:
        validate_context(
            value, columns=set(evidence["observations"]["columns"]), study=study
        )

    def explore(value: Any, evidence: dict[str, Any]) -> None:
        candidates = {
            row["candidateId"]: Candidate(
                candidateId=row["candidateId"], **row["parameters"]
            )
            for row in evidence["candidates"]
        }
        partitions = {
            row["optionId"]: row
            for candidate in evidence["candidates"]
            for row in candidate["partitions"]
        }
        experiments = evidence["experiments"]
        validate_choice(
            value,
            options=set(experiments)
            if value.action == "experiment"
            else set(partitions),
            actions={"shortlist", "defer"} | ({"experiment"} if experiments else set()),
            maximum=1 if value.action == "experiment" else config.maxFinalists,
            evidence_ids=set(candidates) | set(partitions),
        )
        if value.action == "shortlist":
            _shortlist(value.optionIds, partitions, candidates, config)

    def select(value: Any, evidence: dict[str, Any]) -> None:
        validate_choice(
            value,
            options=set(evidence["eligibleOptions"]),
            actions={"choose", "defer"},
            evidence_ids=set(evidence["finalists"]),
        )

    def annotate(value: Any, evidence: dict[str, Any]) -> None:
        validate_annotations(value, clusters=evidence["clusters"])

    return replay_decisions(
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
