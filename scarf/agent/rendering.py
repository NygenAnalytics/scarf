"""Offline, escaped reports derived entirely from saved agent records."""

import csv
import html
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .records import RecordError, RunRecords

ANNOTATION_COLUMNS = (
    "clusterId",
    "identity",
    "confidence",
    "supportingMarkers",
    "contradictingMarkers",
    "rationale",
)


def stage_data(records: RunRecords, stage: str) -> Any | None:
    for event in reversed(records.events()):
        if event["kind"] == "stageCompleted" and event.get("stage") == stage:
            return records.read_json(event["evidence"])
    return None


def annotations(records: RunRecords) -> list[dict[str, Any]]:
    saved = stage_data(records, "annotate")
    if saved is None:
        return []
    final = stage_data(records, "finalize")
    if (
        final is None
        or saved["runId"] != final["runId"]
        or saved["clusters"] != final["artifacts"]["clusters"]
    ):
        raise RecordError("Annotations do not refer to the final frozen clustering")
    return list(saved["annotations"])


def summary(records: RunRecords) -> dict[str, Any]:
    """Collect inspectable provenance without contacting a numerical store."""
    events = records.events()
    current = records.latest("status") or {"status": "running", "stage": "inspect"}
    final = stage_data(records, "finalize")
    prepared = stage_data(records, "preprocess") or stage_data(records, "inspect") or {}
    supplied = records.latest("inputsResolved") or records.manifest
    limitations = list(prepared.get("limitations", []))
    limitations.extend(
        event["message"] for event in events if event["kind"] == "limitation"
    )
    diagnostics = []
    for event in events:
        if event["kind"] not in {"candidateMeasured", "finalistMeasured"}:
            continue
        evidence = records.read_json(event["evidence"])
        identifier = event.get("optionId", event.get("candidateId"))
        diagnostic = {
            "kind": event["kind"],
            "optionId": identifier,
            **{
                key: evidence[key]
                for key in (
                    "runId",
                    "parameters",
                    "silhouetteSampleCells",
                    "diagnosticScope",
                    "metrics",
                    "limitations",
                )
                if key in evidence
            },
        }
        diagnostics.append(diagnostic)
        limitations.extend(
            f"{identifier}: {message}" for message in evidence.get("limitations", [])
        )
    assessed = stage_data(records, "finalists")
    rejected = assessed.get("rejected", {}) if assessed else {}
    if assessed is None:
        # A model can pause or fail after a gate was measured. Its persisted
        # request already contains those gate outcomes, even without a stage end.
        for event in reversed(events):
            if event["kind"] == "modelRequest" and event.get("stage") == "finalists":
                request = records.read_json(event["requestPath"])
                rejected = json.loads(request["userPrompt"])["evidence"].get(
                    "rejectedOptions", {}
                )
                break
    limitations.extend(
        f"Correction rejected for {option}: {reason}"
        for option, reasons in rejected.items()
        for reason in reasons
    )
    errors = [
        {
            key: event[key]
            for key in ("kind", "stage", "decisionId", "errorType", "message", "errors")
            if key in event
        }
        for event in events
        if event["kind"]
        in {"pipelineFailed", "modelFailure", "decisionRejected", "reportError"}
        or event["kind"] == "status"
        and event.get("status") in {"failed", "interrupted"}
    ]
    final_plan = next(
        (
            event
            for event in reversed(events)
            if event["kind"] == "pipelinePlanned" and event.get("operation") == "final"
        ),
        None,
    )
    return {
        "runId": records.manifest.get("runId"),
        "status": current["status"],
        "stage": current.get("stage"),
        "study": supplied.get("study", {}),
        "config": supplied.get("config", {}),
        "procedureIdentity": records.manifest.get("procedureIdentity"),
        "inputCells": prepared.get("inputCells"),
        "retainedCells": prepared.get("retainedCells"),
        "qcFlags": prepared.get("qcFlags", {}),
        "diagnostics": diagnostics,
        "rejectedCorrections": rejected,
        "pendingQuestions": current.get("questions", [])
        if current["status"] == "needsInput"
        else [],
        "final": final,
        "selectedRecipe": {
            "candidate": final_plan.get("candidate"),
            "resolution": final_plan.get("resolution"),
        }
        if final_plan
        else None,
        "pipelineRuns": [
            {
                key: event[key]
                for key in ("operation", "runId", "label", "recovered")
                if key in event
            }
            for event in events
            if event["kind"] == "pipelineCompleted"
        ],
        "annotations": annotations(records),
        "limitations": list(dict.fromkeys(limitations)),
        "decisions": [
            {
                key: event[key]
                for key in ("decisionId", "stage", "output", "recovered")
                if key in event
            }
            for event in events
            if event["kind"] == "decisionAccepted"
        ],
        "errors": errors,
        "modelRequests": sum(event["kind"] == "modelRequest" for event in events),
        "usage": [
            event.get("usage") for event in events if event["kind"] == "modelResponse"
        ],
    }


def atomic_text(path: Path, text: str) -> None:
    """Replace a derived UTF-8 output only after its complete content is durable."""
    if path.is_symlink():
        raise RecordError("Derived output must not overwrite a symlink")
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        if os.name != "nt":
            descriptor = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def annotation_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ANNOTATION_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(row.get(key, []), ensure_ascii=False)
                if key in {"supportingMarkers", "contradictingMarkers"}
                else row.get(key, "")
                for key in ANNOTATION_COLUMNS
            }
        )
    return buffer.getvalue()


def _compact(value: Any, limit: int = 1800) -> str:
    text = (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        if not isinstance(value, str)
        else value
    )
    return text if len(text) <= limit else text[:limit] + " … [see saved records]"


def render_report(records: RunRecords) -> Path:
    """Render any saved outcome. No provider, core store, or plotting is opened."""
    result = summary(records)
    lines = [
        "# Scarf RNA analysis",
        "",
        f"Status: {result['status']}",
        f"Stage: {result['stage']}",
        f"Run: {result['runId']}",
        "",
        "## Study",
        "",
        _compact(result["study"].get("context", "No context saved.")),
        "",
        _compact(result["study"].get("objective", "No objective saved.")),
        "",
        "## Outcome",
        "",
        f"Input cells: {result['inputCells']}; retained cells: {result['retainedCells']}.",
        f"Completed pipeline invocations: {len(result['pipelineRuns'])}.",
        f"Observed model requests: {result['modelRequests']}. Unavailable usage remains unknown.",
        "",
    ]
    if result["final"]:
        lines.extend(
            [
                f"Final pipeline: {result['final']['runId']}",
                "",
                f"Selected recipe: {_compact(result['selectedRecipe'])}",
                "",
            ]
        )
    sections = [
        ("Pending questions", result["pendingQuestions"]),
        ("Limitations", result["limitations"]),
        (
            "QC outlier summaries",
            [
                {"column": column, **flags}
                for column, flags in result["qcFlags"].items()
            ],
        ),
        ("Measured diagnostics", result["diagnostics"]),
        ("Accepted decisions", result["decisions"]),
        ("Failures and rejected decisions", result["errors"]),
        ("Provisional annotations", result["annotations"]),
        ("Pipeline invocations", result["pipelineRuns"]),
        ("Recorded model usage", result["usage"]),
    ]
    for title, items in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {_compact(item)}" for item in items[:100])
        if not items:
            lines.append("None recorded.")
        if len(items) > 100:
            lines.append(f"Additional entries: {len(items) - 100}; see saved records.")
        lines.append("")
    lines.extend(
        [
            "## Saved records",
            "",
            "[Manifest](run.json), [events](events/), [evidence](evidence/), "
            "[model exchanges](calls/), [annotations](annotations.csv).",
            "",
            "Annotations are provisional; unassigned clusters remain explicit.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    links = " | ".join(
        f'<a href="{target}">{name}</a>'
        for name, target in (
            ("Manifest", "run.json"),
            ("Events", "events/"),
            ("Evidence", "evidence/"),
            ("Model exchanges", "calls/"),
            ("Annotations", "annotations.csv"),
            ("Markdown", "report.md"),
        )
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Scarf RNA analysis</title><body>"
        f'<nav>{links}</nav><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'
        f"{html.escape(markdown)}</pre></body></html>\n"
    )
    atomic_text(records.path / "annotations.csv", annotation_csv(result["annotations"]))
    atomic_text(records.path / "report.md", markdown)
    atomic_text(records.path / "report.html", page)
    return records.path / "report.html"
