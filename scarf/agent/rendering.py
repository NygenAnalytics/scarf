"""Offline, escaped reports derived entirely from saved agent records."""

import base64
import csv
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


def exploration_coverage(records: RunRecords) -> dict[str, Any] | None:
    """Read recorded exploration coverage without inferring historical checks."""
    event = records.latest("explorationCoverage")
    if event is not None:
        return dict(event["coverage"])
    explored = stage_data(records, "explore")
    return explored.get("explorationCoverage") if explored else None


def decision_resolutions(records: RunRecords) -> list[dict[str, Any]]:
    """Keep policy resolutions distinct from accepted provider responses."""
    return [event for event in records.events() if event["kind"] == "decisionResolved"]


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
        in {
            "pipelineFailed",
            "modelFailure",
            "decisionRejected",
            "reportError",
            "resultPublicationError",
        }
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
        "explorationCoverage": exploration_coverage(records),
        "decisionResolutions": decision_resolutions(records),
        "resolvedRoles": prepared.get("resolvedRoles"),
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


def _preview(path: Path) -> str | None:
    """Read an optional bounded PNG, without following a symbolic link."""
    if not path.is_symlink() and path.is_file():
        with path.open("rb") as stream:
            image = stream.read(8 * 1024 * 1024 + 1)
        if len(image) <= 8 * 1024 * 1024 and image.startswith(b"\x89PNG\r\n\x1a\n"):
            return "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    return None


def _report_evidence(records: RunRecords, result: dict[str, Any]) -> dict[str, Any]:
    """Join only the selected saved evidence; never reopen the numerical source."""
    prepared = stage_data(records, "preprocess") or stage_data(records, "inspect") or {}
    assessed = stage_data(records, "finalists") or {}
    selected = (result["final"] or {}).get("selected")
    finalist = assessed.get("finalists", {}).get(selected, {}) if selected else {}
    candidates = []
    events = records.events()
    for event in events:
        if event["kind"] == "candidateMeasured":
            evidence = records.read_json(event["evidence"])
            candidates.append({**evidence, "candidateId": event["candidateId"]})
    return {
        **result,
        "prepared": prepared,
        "clusterEvidence": finalist.get("clusters", []),
        "candidateEvidence": candidates,
        "selectedOption": selected or assessed.get("selected"),
        "completedStages": [
            event["stage"] for event in events if event["kind"] == "stageCompleted"
        ],
        "startedStages": [
            event["stage"]
            for event in events
            if event["kind"] == "status" and event.get("stage")
        ],
        "currentMessage": (records.latest("status") or {}).get("message"),
        "preview": _preview(records.path / "umap_clusters.png"),
        "markerPreview": _preview(records.path / "marker_dotplot.png"),
    }


def render_report(records: RunRecords) -> Path:
    """Render any saved outcome with local assets and no provider or store access."""
    from .report_charts import cluster_size_svg
    from .report_html import build_report

    result = summary(records)
    evidence = _report_evidence(records, result)
    chart = cluster_size_svg(evidence["clusterEvidence"])
    if chart:
        atomic_text(records.path / "cluster_sizes.svg", chart)
        evidence["clusterSizeChart"] = "data:image/svg+xml;base64," + base64.b64encode(
            chart.encode("utf-8")
        ).decode("ascii")
    page, markdown = build_report(evidence)
    atomic_text(records.path / "annotations.csv", annotation_csv(result["annotations"]))
    atomic_text(records.path / "report.md", markdown)
    atomic_text(records.path / "report.html", page)
    return records.path / "report.html"
