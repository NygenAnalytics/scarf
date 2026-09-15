"""Generate one local analysis summary from the authoritative stage history."""

import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import _local_root, report_directory, scientific_summary
from .plots import collect_analysis_artifacts
from .rendering import render_analysis_document

if TYPE_CHECKING:
    from ...datastore.datastore import DataStore


def render_analysis_report(
    store: "DataStore", snapshot: Mapping[str, Any], output_dir: Path
) -> Path:
    """Replace derived display files without changing the saved analysis."""
    if snapshot.get("status") != "completed" or not snapshot.get("finalAnalysis"):
        raise ValueError("Reports require a completed analysis with final artifacts")
    payload = scientific_summary(snapshot)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload.update(
        collect_analysis_artifacts(store, payload["finalAnalysis"], output_dir)
    )
    destination = output_dir / "index.html"
    temporary = output_dir / f".index.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(render_analysis_document(payload), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def generate_agent_report(
    target: "str | Path | DataStore",
    workflow_run_id: str,
    *,
    workspace: str | None = None,
) -> Path:
    """Return one concise local report for a completed analysis.

    Decisions, their evidence, and numerical artifacts come from the saved
    stage history. Regeneration makes no model or scientific computation calls.
    """
    from ...datastore.datastore import DataStore
    from ..orchestrator import journal

    root = _local_root(target)
    if isinstance(target, DataStore):
        if workspace is not None and workspace != target.workspace:
            raise ValueError("workspace does not match the DataStore workspace")
        store = target
        workspace = store.workspace
    else:
        store = journal.open_analysis_store(root, workflow_run_id, workspace=workspace)
    snapshot = journal.analysis_snapshot(store, workflow_run_id)
    return render_analysis_report(
        store, snapshot, report_directory(root, workflow_run_id, workspace)
    )
