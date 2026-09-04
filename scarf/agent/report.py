"""Local HTML reports for completed automated Scarf agent workflows.

Reports are derived presentation files. They are written beside the immutable
agent records, but they are not Zarr components and never participate in
workflow checksums or artifact lineage.
"""

import html
import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .. import __version__
from ..datastore.datastore import DataStore
from ..storage.stores import zarr_root_path
from ..utils.logging import logger
from . import record_io
from .orchestrator import journal
from .orchestrator.models import (
    _STAGE_ORDER,
    AutomatedWorkflowResult,
    OrchestrationRequestRecord,
    WorkflowStageAttempt,
    artifact_model_to_ref,
)
from .persistence import (
    AgentWorkflowRun,
    load_agent_report,
    load_agent_workflow,
)

MAX_MARKER_DOTPLOT_FEATURES = 24
CLUSTER_COUNT_BLOCK_SIZE = 100_000
MAX_EMBEDDING_PLOT_CELLS = 250_000
MAX_DOTPLOT_CELLS = 75_000
MAX_CONNECTIVITY_PLOT_CELLS = 100_000
MAX_COMPOSITION_PLOT_CELLS = 1_000_000


def _local_root(target: str | Path | DataStore) -> Path:
    """Resolve a local filesystem root without accepting remote stores."""
    if isinstance(target, DataStore):
        location = zarr_root_path(target.z)
        if location is None:
            raise ValueError("Agent HTML reports require a local filesystem store")
        path = Path(location)
    elif isinstance(target, Path):
        path = target
    elif isinstance(target, str) and target.startswith("file://"):
        path = Path(target.removeprefix("file://"))
    elif isinstance(target, str):
        if "://" in target:
            raise ValueError("Agent HTML reports require a local filesystem store")
        path = Path(target)
    else:
        raise TypeError("Agent HTML reports require a local filesystem store")
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _open_datastore(
    target: str | Path | DataStore,
    root: Path,
    workflow: AgentWorkflowRun,
) -> DataStore:
    if isinstance(target, DataStore):
        if target.workspace != workflow.workspace:
            raise ValueError("Workflow workspace does not match the DataStore")
        return target
    default_assay = next(iter(workflow.datasetFingerprints))
    return DataStore(
        str(root),
        default_assay=default_assay,
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r",
        workspace=workflow.workspace,
    )


def _load_request(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
) -> OrchestrationRequestRecord:
    record = cast(
        OrchestrationRequestRecord,
        journal._read_model(
            store.zw,
            journal._request_key(prefix, workflow_run_id),
            OrchestrationRequestRecord,
        ),
    )
    if record.workflowRunId != workflow_run_id:
        raise ValueError("Stored orchestration request belongs to another workflow")
    if record.requestSha256 != journal._sha256_model(record.request):
        raise ValueError("Stored orchestration request checksum is invalid")
    if record.configSha256 != journal._sha256_model(record.config):
        raise ValueError("Stored orchestration configuration checksum is invalid")
    if record.contentSha256 != journal._record_checksum(record):
        raise ValueError("Stored orchestration request envelope is invalid")
    return record


def _load_completed_result(
    store: DataStore,
    workflow: AgentWorkflowRun,
) -> tuple[str, AutomatedWorkflowResult, OrchestrationRequestRecord]:
    if workflow.status != "completed":
        raise RuntimeError(
            "Agent HTML reports can only be generated for completed workflows"
        )
    prefix = journal._ensure_orchestration_store(store)
    result = journal._load_terminal_result(store, prefix, workflow)
    if result is None:
        raise FileNotFoundError(
            f"Completed workflow {workflow.workflowRunId!r} has no terminal result"
        )
    if result.status != "completed" or result.finalAnalysis is None:
        raise ValueError("Completed workflow result lacks its final analysis handoff")
    request = _load_request(store, prefix, workflow.workflowRunId)
    if request.request.workspace != workflow.workspace:
        raise ValueError("Stored request workspace does not match the workflow")
    return prefix, result, request


def _collect_reports(
    store: DataStore,
    result: AutomatedWorkflowResult,
) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for reference in result.reportReferences:
        report = load_agent_report(store, reference)
        reports.setdefault(reference.agentName, []).append(
            report.model_dump(mode="json")
        )
    return reports


def _stage_summary(attempt: WorkflowStageAttempt) -> dict[str, Any]:
    duration = (
        (attempt.completedAtNs - attempt.startedAtNs) / 1_000_000_000
        if attempt.completedAtNs
        else None
    )
    error_type = None
    if attempt.error:
        candidate = attempt.error.partition(":")[0].strip()
        error_type = (
            candidate
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", candidate)
            else "WorkflowStageError"
        )
    return {
        "stage": attempt.stage,
        "attemptId": attempt.attemptId,
        "status": attempt.status,
        "durationSeconds": duration,
        "actions": list(attempt.actions),
        "reportCount": len(attempt.reportReferences),
        "artifactCount": len(attempt.artifacts),
        "artifacts": {
            name: artifact.model_dump(mode="json")
            for name, artifact in attempt.artifacts.items()
        },
        "parentAttempts": [
            f"{parent.stage}:{parent.attemptId}" for parent in attempt.parentAttempts
        ],
        "questionIds": (
            [question.questionId for question in attempt.needsInput.questions]
            if attempt.needsInput is not None
            else []
        ),
        "noteCount": len(attempt.notes),
        "notes": list(attempt.notes),
        "errorType": error_type,
    }


def _collect_history(
    store: DataStore,
    prefix: str,
    workflow: AgentWorkflowRun,
    request: OrchestrationRequestRecord,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts: dict[tuple[str, str], WorkflowStageAttempt] = {}
    summaries: list[tuple[int, dict[str, Any]]] = []
    for stage in _STAGE_ORDER:
        starts = {
            item.attemptId: item
            for item in journal._stage_starts(
                store.zw, prefix, workflow.workflowRunId, stage
            )
        }
        outcomes = {
            item.attemptId: item
            for item in journal._stage_outcomes(
                store.zw, prefix, workflow.workflowRunId, stage
            )
        }
        if not set(outcomes).issubset(starts):
            raise ValueError("Workflow history contains an outcome without a start")
        for attempt_id, started in starts.items():
            attempt = outcomes.get(attempt_id, started)
            identity = (stage, attempt_id)
            if identity in attempts:
                raise ValueError("Workflow history contains duplicate stage attempts")
            attempts[identity] = attempt
            summaries.append((attempt.startedAtNs, _stage_summary(attempt)))

    for attempt in attempts.values():
        for parent in attempt.parentAttempts:
            observed = attempts.get((parent.stage, parent.attemptId))
            if (
                observed is None
                or observed.status != "done"
                or observed.contentSha256 != parent.contentSha256
            ):
                raise ValueError("Workflow parent-stage lineage does not resolve")

    biological_reports = [
        reference
        for reference in workflow.reports
        if reference.agentName == "biological_interpretation"
    ]
    if not biological_reports:
        raise ValueError("Completed workflow lacks a Biological Interpretation report")
    terminal_report = biological_reports[-1]
    terminal_candidates = [
        attempt
        for attempt in attempts.values()
        if attempt.stage == "biological_interpretation"
        and attempt.status == "done"
        and terminal_report in attempt.reportReferences
    ]
    if len(terminal_candidates) != 1:
        raise ValueError("Completed workflow lacks one exact terminal stage attempt")
    current = terminal_candidates[0]
    terminal_chain: set[tuple[str, str]] = set()
    while True:
        identity = (current.stage, current.attemptId)
        if identity in terminal_chain:
            raise ValueError("Workflow stage lineage contains a cycle")
        terminal_chain.add(identity)
        if not journal._stage_outcome_resolves(
            store,
            prefix,
            workflow.workflowRunId,
            request,
            current,
        ):
            raise ValueError("Terminal workflow stage artifacts do not resolve")
        stage_index = _STAGE_ORDER.index(current.stage)
        if stage_index == 0:
            if current.parentAttempts:
                raise ValueError("The ingest stage cannot have a parent")
            break
        if len(current.parentAttempts) != 1:
            raise ValueError("Every terminal-chain stage must have one parent")
        parent = current.parentAttempts[0]
        if parent.stage != _STAGE_ORDER[stage_index - 1]:
            raise ValueError("Terminal workflow lineage skips a stage")
        current = attempts[(parent.stage, parent.attemptId)]

    resumes: list[dict[str, Any]] = []
    resume_prefix = record_io.join_key(prefix, workflow.workflowRunId, "resumes")
    for key in record_io.list_keys(store.zw, resume_prefix):
        if not key.endswith(".json"):
            continue
        resume_id = key.rsplit("/", 1)[-1].removesuffix(".json")
        resume = journal._validated_resume_record(
            store, prefix, workflow.workflowRunId, resume_id
        )
        resumes.append(
            {
                "resumeId": resume.resumeId,
                "createdAtNs": resume.createdAtNs,
                "answeredStage": (
                    resume.answeredAttempt.stage
                    if resume.answeredAttempt is not None
                    else None
                ),
                "answeredAttemptId": (
                    resume.answeredAttempt.attemptId
                    if resume.answeredAttempt is not None
                    else None
                ),
                "questionIds": list(resume.questionIds),
            }
        )
    resumes.sort(key=lambda value: (value["createdAtNs"], value["resumeId"]))
    ordered = sorted(
        summaries,
        key=lambda item: (
            item[0],
            str(item[1]["stage"]),
            str(item[1]["attemptId"]),
        ),
    )
    return [value for _, value in ordered], resumes


def _save_plot(plot: Any, path: Path) -> None:
    """Atomically save one plot and its provenance, always closing its figure."""
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.stem}.{token}{path.suffix}")
    sidecar = path.with_suffix(path.suffix + ".json")
    temporary_sidecar = sidecar.with_name(f".{sidecar.stem}.{token}{sidecar.suffix}")
    try:
        plot.save(temporary, dpi=150)
        plot.save_provenance(temporary_sidecar, figure_path=path, dpi=150)
        os.replace(temporary, path)
        os.replace(temporary_sidecar, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        plot.close()


def _safe_assay_name(value: str, fallback: str) -> str:
    label = "_".join(part.lower() for part in re.findall(r"[A-Za-z0-9]+", value))
    return label[:64].rstrip("_") or fallback


def _collect_final_artifacts(
    store: DataStore,
    result: AutomatedWorkflowResult,
    plot_dir: Path,
) -> tuple[
    dict[str, int],
    list[dict[str, Any]],
    dict[str, str],
    list[str],
]:
    """Validate the final handoff and derive bounded tables and plots."""
    import numpy as np

    final = result.finalAnalysis
    assert final is not None
    if final.cellSelection is None or final.clusters is None or final.umap is None:
        raise ValueError("Final handoff lacks its selection, clusters, or UMAP")

    artifact_models = [
        final.cellSelection,
        final.graph,
        final.clusters,
        final.embeddingInitialization,
        final.umap,
        final.markerFeatures,
        final.markers,
    ]
    for native in final.nativeAnalyses:
        artifact_models.extend(
            [
                native.featureSelection,
                native.markerFeatures,
                native.normalized,
                native.reduction,
                native.batchCorrection,
                native.annIndex,
                native.embeddingInitialization,
                native.neighbors,
                native.graph,
                native.clusters,
                native.umap,
            ]
        )
    for artifact in artifact_models:
        if artifact is not None:
            store.load_artifact(artifact_model_to_ref(artifact))

    cluster_ref = artifact_model_to_ref(final.clusters)
    umap_ref = artifact_model_to_ref(final.umap)
    cluster_artifact: Any = store.load_artifact(cluster_ref)
    values = cluster_artifact["values"]
    counts: Counter[str] = Counter()
    for start in range(0, int(values.shape[0]), CLUSTER_COUNT_BLOCK_SIZE):
        block = np.asarray(values[start : start + CLUSTER_COUNT_BLOCK_SIZE]).astype(str)
        block_labels, frequencies = np.unique(block, return_counts=True)
        counts.update(
            {
                str(label): int(frequency)
                for label, frequency in zip(block_labels, frequencies, strict=True)
            }
        )
    cluster_counts = dict(sorted(counts.items()))
    cluster_labels = list(cluster_counts)
    n_cells = sum(cluster_counts.values())
    plots: dict[str, str] = {}
    notes: list[str] = []
    plot_dir.mkdir(parents=True, exist_ok=True)

    def render_plot(name: str, filename: str, create: Any) -> None:
        try:
            path = plot_dir / filename
            _save_plot(create(), path)
            plots[name] = f"plots/{filename}"
        except Exception as exc:
            notes.append(f"{name}: {type(exc).__name__}: {exc}")

    if n_cells <= MAX_EMBEDDING_PLOT_CELLS:
        render_plot(
            "umapClusters",
            "final_umap.png",
            lambda: store.plots.embedding(
                layout=umap_ref,
                color_by=cluster_ref,
                show=False,
            ),
        )
    else:
        notes.append(
            "umapClusters: skipped because the final selection has "
            f"{n_cells:,} cells, above the memory-safe report limit of "
            f"{MAX_EMBEDDING_PLOT_CELLS:,}"
        )

    observed_native_names: set[str] = set()
    for index, native in enumerate(final.nativeAnalyses):
        if native.umap is None or native.clusters is None:
            continue
        native_umap = artifact_model_to_ref(native.umap)
        native_clusters = artifact_model_to_ref(native.clusters)
        if native_umap == umap_ref and native_clusters == cluster_ref:
            continue
        suffix = _safe_assay_name(native.assay, f"assay_{index + 1}")
        base_name = "nativeUmap" + "".join(
            part.capitalize() for part in suffix.split("_")
        )
        plot_name = base_name
        serial = 1
        while plot_name in observed_native_names:
            serial += 1
            plot_name = f"{base_name}{serial}"
        observed_native_names.add(plot_name)
        file_suffix = suffix if serial == 1 else f"{suffix}_{serial}"
        if n_cells <= MAX_EMBEDDING_PLOT_CELLS:
            render_plot(
                plot_name,
                f"native_umap_{file_suffix}.png",
                lambda layout=native_umap, color=native_clusters: store.plots.embedding(
                    layout=layout, color_by=color, show=False
                ),
            )
        else:
            notes.append(
                f"{plot_name}: skipped because {n_cells:,} cells exceed the "
                f"memory-safe report limit of {MAX_EMBEDDING_PLOT_CELLS:,}"
            )

    if n_cells <= MAX_COMPOSITION_PLOT_CELLS:
        render_plot(
            "clusterComposition",
            "cluster_composition.png",
            lambda: store.plots.composition(
                categories=cluster_ref,
                show_percent_labels=len(cluster_labels) <= 12,
                show=False,
            ),
        )
    else:
        notes.append(
            "clusterComposition: skipped because the final selection has "
            f"{n_cells:,} cells, above the memory-safe report limit of "
            f"{MAX_COMPOSITION_PLOT_CELLS:,}"
        )

    top_markers: list[dict[str, Any]] = []
    if final.markers is not None:
        marker_ref = artifact_model_to_ref(final.markers)
        marker_parameters = store.inspect_artifact(marker_ref).parameters or {}
        raw_normalization = marker_parameters.get("normalization", {})
        marker_normalization = (
            dict(raw_normalization) if isinstance(raw_normalization, Mapping) else {}
        )
        marker_log_transform = marker_normalization.get("log_transform", False) is True
        if marker_normalization.get("renormalize_subset", False) is True:
            notes.append(
                "marker visualizations: the persisted marker search renormalized "
                "its feature subset; current plotting APIs preserve its log "
                "transform but visualize assay-wide normalized values"
            )
        for label in cluster_labels:
            try:
                table = store.get_markers(
                    marker_ref,
                    group_id=label,
                    min_score=-1,
                    min_frac_exp=-1,
                )
                if not table.empty:
                    if "score" in table:
                        table = table.sort_values(
                            "score", ascending=False, kind="stable"
                        )
                    top_markers.extend(
                        json.loads(table.head(5).to_json(orient="records"))
                    )
            except Exception as exc:
                notes.append(
                    f"marker export for cluster {label}: {type(exc).__name__}: {exc}"
                )

        render_plot(
            "markerHeatmap",
            "marker_heatmap.png",
            lambda: store.plots.marker_heatmap(
                marker=marker_ref,
                log_transform=marker_log_transform,
                show=False,
            ),
        )

        try:
            from ..plotting import FeatureRef, NormalizationSpec

            by_cluster: dict[str, list[tuple[tuple[str, str], Any]]] = {
                label: [] for label in cluster_labels
            }
            for marker in top_markers:
                group_id = str(marker.get("group_id", ""))
                if group_id not in by_cluster:
                    continue
                feature_name = marker.get("feature_name")
                feature_id = marker.get("feature_id")
                feature_index = marker.get("feature_index")
                label = str(feature_name or feature_id or feature_index or "")
                if isinstance(feature_index, (int, float)):
                    identity = ("index", str(int(feature_index)))
                    feature = FeatureRef(
                        value=int(feature_index),
                        assay=final.markerAssay,
                        by="index",
                        label=label,
                    )
                elif isinstance(feature_id, str) and feature_id:
                    identity = ("id", feature_id)
                    feature = FeatureRef(
                        value=feature_id,
                        assay=final.markerAssay,
                        by="id",
                        label=label,
                    )
                else:
                    continue
                if all(observed != identity for observed, _ in by_cluster[group_id]):
                    by_cluster[group_id].append((identity, feature))

            marker_groups: dict[str, list[Any]] = {}
            selected: set[tuple[str, str]] = set()
            max_rank = max(map(len, by_cluster.values()), default=0)
            rank = 0
            while rank < max_rank and len(selected) < MAX_MARKER_DOTPLOT_FEATURES:
                for cluster in cluster_labels:
                    features = by_cluster[cluster]
                    if rank >= len(features):
                        continue
                    identity, feature = features[rank]
                    if identity in selected:
                        continue
                    marker_groups.setdefault(f"Cluster {cluster}", []).append(feature)
                    selected.add(identity)
                    if len(selected) == MAX_MARKER_DOTPLOT_FEATURES:
                        break
                rank += 1
            if marker_groups and n_cells <= MAX_DOTPLOT_CELLS:
                render_plot(
                    "markerDotplot",
                    "marker_dotplot.png",
                    lambda: store.plots.dotplot(
                        features=marker_groups,
                        groups=cluster_ref,
                        from_assay=final.markerAssay,
                        normalization=NormalizationSpec(
                            source="assay",
                            transform=("log1p" if marker_log_transform else "none"),
                        ),
                        standardize="feature",
                        show=False,
                    ),
                )
            elif marker_groups:
                notes.append(
                    "markerDotplot: skipped because the final selection has "
                    f"{n_cells:,} cells, above the memory-safe report limit of "
                    f"{MAX_DOTPLOT_CELLS:,}"
                )
        except Exception as exc:
            notes.append(f"markerDotplot: {type(exc).__name__}: {exc}")

    if final.graph is not None:
        graph_ref = artifact_model_to_ref(final.graph)
        if n_cells <= MAX_CONNECTIVITY_PLOT_CELLS:
            render_plot(
                "clusterConnectivity",
                "cluster_connectivity.png",
                lambda: store.plots.cluster_connectivity(
                    groups=cluster_ref,
                    layout=umap_ref,
                    graph=graph_ref,
                    show=False,
                ),
            )
        else:
            notes.append(
                "clusterConnectivity: skipped because the final selection has "
                f"{n_cells:,} cells, above the memory-safe report limit of "
                f"{MAX_CONNECTIVITY_PLOT_CELLS:,}"
            )
    return cluster_counts, top_markers, plots, notes


REPORT_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400&display=swap');

:root {
  --blue: #0077fc;
  --black: #000000;
  --gray: #b4b4b4;
  --white: #ffffff;
}

* { box-sizing: border-box; }
html { background: var(--white); color: var(--black); font-family: Inter, sans-serif; }
body {
  margin: 0;
  background: var(--white);
  color: var(--black);
  font-family: Inter, sans-serif;
  font-weight: 300;
  letter-spacing: -0.04em;
  line-height: 1.2;
}
a { color: var(--blue); }
header, main, footer {
  width: min(100%, 1240px);
  margin: 0 auto;
  padding-left: clamp(1.25rem, 5vw, 4.5rem);
  padding-right: clamp(1.25rem, 5vw, 4.5rem);
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--black);
  padding-top: 1.75rem;
  padding-bottom: 1.75rem;
}
.brand {
  color: var(--black);
  font-size: 1rem;
  font-weight: 400;
  text-decoration: none;
}
main { padding-top: clamp(3rem, 8vw, 7rem); padding-bottom: 6rem; }
footer {
  border-top: 1px solid var(--black);
  padding-top: 2rem;
  padding-bottom: 2rem;
}
h1, h2, h3, p { margin-top: 0; }
h1 {
  max-width: 15ch;
  margin-bottom: 1.5rem;
  font-size: clamp(2.75rem, 7vw, 5rem);
  font-weight: 400;
  letter-spacing: 0;
  line-height: 1.2;
}
h2 {
  margin-bottom: 1.5rem;
  font-size: clamp(1.65rem, 3vw, 2.25rem);
  font-weight: 400;
  letter-spacing: -0.04em;
  line-height: 1.2;
}
h3 {
  margin-bottom: .8rem;
  font-size: 1rem;
  font-weight: 300;
  letter-spacing: -0.04em;
  line-height: 1.2;
}
p, li, td, th, summary, code, pre, a {
  font-family: Inter, sans-serif;
  letter-spacing: -0.04em;
  line-height: 1.2;
}
strong { font-weight: 400; }
.eyebrow {
  margin-bottom: 1rem;
  color: var(--gray);
  font-size: .75rem;
  font-weight: 400;
  text-transform: uppercase;
}
.lead {
  max-width: 48ch;
  font-size: clamp(1.2rem, 2vw, 1.7rem);
  font-weight: 300;
}
.pill-row, .chip-row, .metric-grid {
  display: flex;
  flex-wrap: wrap;
  gap: .65rem;
}
.pill-row { margin-top: 1.75rem; }
.pill, .chip {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  font-size: .82rem;
  font-weight: 400;
  line-height: 1.2;
}
.pill {
  border: 1px solid var(--blue);
  padding: .68rem 1.1rem;
  background: var(--blue);
  color: var(--white);
  text-decoration: none;
}
.pill-outline {
  background: var(--white);
  box-shadow: inset 0 0 0 1px var(--blue);
  color: var(--blue);
}
.chip {
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: .42rem .75rem;
  color: var(--black);
}
.metric-grid { margin-top: 2rem; }
.metric {
  display: flex;
  min-width: 9rem;
  flex-direction: column;
  gap: .2rem;
  border-radius: 999px;
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: .8rem 1.2rem;
}
.metric-label {
  color: var(--gray);
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.metric-value { font-size: .95rem; font-weight: 400; overflow-wrap: anywhere; }
.section { margin-top: 4rem; border-top: 1px solid var(--black); padding-top: 1.5rem; }
.section:target { scroll-margin-top: 1rem; }
.section-heading {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: start;
}
.subsection { margin-top: 2rem; }
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 19rem), 1fr));
  gap: 1rem;
}
.card, .callout {
  border: 1px solid var(--black);
  padding: 1.25rem;
  background: var(--white);
}
.callout { border-color: var(--blue); }
.product-callout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: center;
  margin-top: 2.5rem;
  border-radius: 999px;
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: 1.4rem;
}
.product-callout p { margin-bottom: 0; max-width: 55rem; }
.empty { color: var(--gray); font-style: italic; }
.table-wrap { width: 100%; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .86rem; }
th, td {
  border-bottom: 1px solid var(--black);
  padding: .8rem .7rem;
  text-align: left;
  vertical-align: top;
}
th { color: var(--gray); font-weight: 400; text-transform: uppercase; }
td { font-weight: 300; overflow-wrap: anywhere; }
tr.selected { box-shadow: inset 4px 0 0 var(--blue); }
dl { margin: 0; }
.details > div {
  display: grid;
  grid-template-columns: minmax(8rem, 14rem) minmax(0, 1fr);
  gap: 1rem;
  border-bottom: 1px solid var(--gray);
  padding: .55rem 0;
}
dt { color: var(--gray); font-size: .78rem; font-weight: 400; text-transform: uppercase; }
dd { margin: 0; overflow-wrap: anywhere; }
.plot-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
  gap: 2rem;
}
figure { margin: 0; }
figure.primary { grid-column: 1 / -1; }
figure img { display: block; width: 100%; height: auto; border: 1px solid var(--black); }
figcaption { margin-top: .7rem; color: var(--black); font-size: .85rem; }
.cluster-row {
  display: grid;
  grid-template-columns: minmax(5rem, auto) minmax(8rem, 1fr) auto;
  gap: .7rem;
  align-items: center;
  margin: .5rem 0;
}
.cluster-track { height: .7rem; border-radius: 999px; background: var(--gray); overflow: hidden; }
.cluster-fill { height: 100%; border-radius: 999px; background: var(--blue); }
.text-list { padding-left: 1.2rem; }
.text-list li { margin: .45rem 0; }
details { margin-top: 1rem; border-top: 1px solid var(--gray); padding-top: .8rem; }
summary { cursor: pointer; font-weight: 400; }
pre {
  max-height: 36rem;
  overflow: auto;
  background: var(--white);
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: 1rem;
  font-size: .76rem;
  white-space: pre-wrap;
  word-break: break-word;
}
@media (max-width: 680px) {
  .section-heading, .product-callout { grid-template-columns: 1fr; }
  .details > div { grid-template-columns: 1fr; gap: .25rem; }
}
"""


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _label(value: Any) -> str:
    text = str(value).replace("_", " ").strip()
    words: list[str] = []
    for index, character in enumerate(text):
        if (
            index
            and character.isupper()
            and not text[index - 1].isupper()
            and text[index - 1] != " "
        ):
            words.append(" ")
        words.append(character)
    text = "".join(words)
    return text[:1].upper() + text[1:]


def _scalar(value: Any) -> str:
    if value is None or value == "":
        return "Not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 10_000:
            return f"{value:.3g}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _chips(value: Any, empty: str = "Not provided") -> str:
    if not _present(value):
        return f'<span class="empty">{html.escape(empty)}</span>'
    if isinstance(value, Mapping):
        items = [f"{_label(key)}: {_scalar(item)}" for key, item in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
    else:
        items = [value]
    return '<span class="chip-row">{}</span>'.format(
        "".join(
            f'<span class="chip">{html.escape(_scalar(item))}</span>' for item in items
        )
    )


def _value(value: Any) -> str:
    if not _present(value):
        return '<span class="empty">Not provided</span>'
    if isinstance(value, Mapping):
        rows = "".join(
            f"<div><dt>{html.escape(_label(key))}</dt><dd>{_value(item)}</dd></div>"
            for key, item in value.items()
            if _present(item)
        )
        return f'<dl class="details">{rows}</dl>'
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if all(not isinstance(item, Mapping) for item in value):
            return _chips(value)
        return '<div class="card-grid">{}</div>'.format(
            "".join(f'<div class="card">{_value(item)}</div>' for item in value)
        )
    return html.escape(_scalar(value))


def _table(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    empty: str = "No records available.",
) -> str:
    normalized = [dict(row) for row in rows]
    if not normalized:
        return f'<p class="empty">{html.escape(empty)}</p>'
    visible = list(columns or ()) or list(
        dict.fromkeys(
            key for row in normalized for key in row if not str(key).startswith("_")
        )
    )
    headings = "".join(f"<th>{html.escape(_label(key))}</th>" for key in visible)
    body = "".join(
        ('<tr class="selected">' if row.get("_selected") else "<tr>")
        + "".join(f"<td>{_value(row.get(key))}</td>" for key in visible)
        + "</tr>"
        for row in normalized
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{headings}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _latest(reports: Mapping[str, Any], agent_name: str) -> dict[str, Any]:
    values = reports.get(agent_name)
    if isinstance(values, Mapping):
        return dict(values)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for value in reversed(values):
            if isinstance(value, Mapping):
                return dict(value)
    return {}


def _render_plots(plots: Mapping[str, str], notes: Sequence[str]) -> str:
    titles = {
        "umapClusters": (
            "Final UMAP by cluster",
            "The selected final representation, colored by final cluster.",
        ),
        "markerHeatmap": (
            "Marker heatmap",
            "Marker-feature patterns across the final clusters.",
        ),
        "markerDotplot": (
            "Marker dot plot",
            "A bounded expression summary for exact exported marker features.",
        ),
        "clusterComposition": (
            "Cluster composition",
            "The relative size of each cluster in the final cell selection.",
        ),
        "clusterConnectivity": (
            "Cluster connectivity",
            "Connectivity between final clusters in the selected graph.",
        ),
    }
    order = [
        "umapClusters",
        *(name for name in plots if name.startswith("nativeUmap")),
        "markerHeatmap",
        "markerDotplot",
        "clusterComposition",
        "clusterConnectivity",
        *plots,
    ]
    figures: list[str] = []
    for name in dict.fromkeys(order):
        source = plots.get(name)
        if source is None:
            continue
        if name.startswith("nativeUmap"):
            assay = name.removeprefix("nativeUmap") or "assay"
            title = f"{assay} native UMAP"
            caption = f"The finalized native {assay} representation and clusters."
        else:
            title, caption = titles.get(
                name, (_label(name), "A finalized Scarf analysis plot.")
            )
        escaped_source = html.escape(source, quote=True)
        provenance = html.escape(source + ".json", quote=True)
        plot_class = ' class="primary"' if name == "umapClusters" else ""
        figures.append(
            f"<figure{plot_class}>"
            f'<a href="{escaped_source}"><img src="{escaped_source}" '
            f'alt="{html.escape(title, quote=True)}" loading="lazy"></a>'
            f"<figcaption><strong>{html.escape(title)}</strong><br>"
            f"{html.escape(caption)} "
            f'<a href="{provenance}">Plot provenance</a></figcaption></figure>'
        )
    if not figures:
        plot_markup = (
            '<div class="callout"><p>No plots could be rendered. The structured '
            "analysis remains available below. Install Scarf with the "
            "<code>extra</code> dependency group to enable plotting.</p></div>"
        )
    else:
        plot_markup = f'<div class="plot-grid">{"".join(figures)}</div>'
    note_markup = ""
    if notes:
        note_markup = (
            "<details><summary>Plot availability notes</summary>"
            '<ul class="text-list">'
            + "".join(f"<li>{html.escape(note)}</li>" for note in notes)
            + "</ul></details>"
        )
    return plot_markup + note_markup


def _render_clusters(cluster_counts: Mapping[str, int]) -> str:
    if not cluster_counts:
        return '<p class="empty">No final cluster counts were available.</p>'
    maximum = max(cluster_counts.values(), default=1) or 1
    return "".join(
        '<div class="cluster-row">'
        f"<span>Cluster {html.escape(str(label))}</span>"
        '<span class="cluster-track">'
        f'<span class="cluster-fill" style="width: {count / maximum * 100:.2f}%">'
        "</span></span>"
        f"<span>{count:,}</span></div>"
        for label, count in cluster_counts.items()
    )


def _parameter_rows(parameter: Mapping[str, Any]) -> list[dict[str, Any]]:
    assay_reports = _mapping(parameter.get("assayReports"))
    if not assay_reports and _present(parameter.get("evaluations")):
        assay_reports = {str(parameter.get("fromAssay") or "Primary"): dict(parameter)}
    recommended = _mapping(parameter.get("recommendedByAssay"))
    rows: list[dict[str, Any]] = []
    for assay, raw_report in assay_reports.items():
        report = _mapping(raw_report)
        selected = recommended.get(assay) or report.get("recommendedCandidateId")
        for evaluation in _mappings(report.get("evaluations")):
            parameters = _mapping(evaluation.get("parameters"))
            rows.append(
                {
                    "_selected": evaluation.get("candidateId") == selected,
                    "assay": assay,
                    "candidate": evaluation.get("candidateId"),
                    "phase": evaluation.get("phase"),
                    "status": evaluation.get("status"),
                    "eligible": evaluation.get("eligible"),
                    "selection confidence": report.get("confidence"),
                    "reduction": parameters.get("reductionMethod"),
                    "dimensions": parameters.get("dimensions"),
                    "neighbors K": parameters.get("neighborsK"),
                    "resolution": parameters.get("leidenResolution"),
                    "Harmony": parameters.get("useHarmony"),
                    "metrics": evaluation.get("metrics"),
                }
            )
    return rows


def _render_parameter_tuning(parameter: Mapping[str, Any]) -> str:
    if not parameter:
        return '<p class="empty">No Parameter Tuning report was persisted.</p>'
    candidate_rows = _parameter_rows(parameter)
    integration_rows = _mappings(parameter.get("integrationEvaluations"))
    plans: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    root_plan = _mapping(parameter.get("searchPlan"))
    if root_plan:
        plans.append({"assay": parameter.get("fromAssay"), **root_plan})
    for comparison in _mappings(parameter.get("comparisons")):
        comparisons.append(
            {"scope": parameter.get("fromAssay") or "primary assay", **comparison}
        )
    for assay, report in _mapping(parameter.get("assayReports")).items():
        assay_report = _mapping(report)
        plan = _mapping(assay_report.get("searchPlan"))
        if plan and plan not in plans:
            plans.append({"assay": assay, **plan})
        for comparison in _mappings(assay_report.get("comparisons")):
            comparisons.append({"scope": assay, **comparison})
    final_selection = _mapping(parameter.get("finalSelection"))
    for comparison in _mappings(final_selection.get("comparisons")):
        comparisons.append({"scope": "final graph", **comparison})
    narrative = {
        "status": parameter.get("status"),
        "totalCandidates": parameter.get("totalCandidates"),
        "recommendedByAssay": parameter.get("recommendedByAssay"),
        "recommendedIntegrationId": parameter.get("recommendedIntegrationId"),
        "confidence": parameter.get("confidence"),
        "rationale": parameter.get("rationale"),
        "tradeoffs": parameter.get("tradeoffs"),
        "stopReason": parameter.get("stopReason"),
        "finalSelection": final_selection,
    }
    return (
        '<div class="callout"><h3>Final graph selection</h3>'
        f"{_value(narrative)}</div>"
        '<div class="subsection"><h3>Native and Harmony candidates</h3>'
        f"{_table(candidate_rows, empty='No native candidates were recorded.')}</div>"
        '<div class="subsection"><h3>SNN and WNN integration candidates</h3>'
        f"{_table(integration_rows, empty='No integration candidates were eligible.')}</div>"
        '<div class="subsection"><h3>Model-authored comparisons</h3>'
        f"{_table(comparisons, empty='No candidate comparisons were required.')}</div>"
        '<div class="subsection"><h3>Bounded search plans</h3>'
        f"{_value(plans) if plans else '<p class="empty">No refinement plan was requested.</p>'}"
        "</div>"
    )


def _execution_rows(reports: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def visit(value: Any, stage: str, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            usage = value.get("usage")
            agent_name = value.get("agentName")
            if (
                isinstance(usage, Mapping)
                and isinstance(agent_name, str)
                and agent_name.strip()
            ):
                run_id = str(value.get("runId") or "")
                identity = (agent_name, run_id, str(value.get("modelName") or ""))
                if identity not in seen:
                    seen.add(identity)
                    rows.append(
                        {
                            "agent stage": _label(stage),
                            "execution": _label(path[-1]) if path else agent_name,
                            "agent": agent_name,
                            "run ID": run_id or "deterministic",
                            "model": value.get("modelName") or "not applicable",
                            "duration seconds": value.get("durationSeconds"),
                            "requests": usage.get("requests", 0),
                            "tool calls": usage.get("toolCalls", 0),
                            "input tokens": usage.get("inputTokens", 0),
                            "output tokens": usage.get("outputTokens", 0),
                            "total tokens": usage.get("totalTokens", 0),
                        }
                    )
            for key, item in value.items():
                visit(item, stage, (*path, str(key)))
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                visit(item, stage, (*path, str(index + 1)))

    for stage, records in reports.items():
        visit(records, str(stage), ())
    return rows


def _render_executions(reports: Mapping[str, Any]) -> str:
    rows = _execution_rows(reports)
    if not rows:
        return '<p class="empty">No provider execution metadata was recorded.</p>'
    totals = {
        "recorded executions": len(rows),
        "provider executions": sum(
            int(
                bool(row["model"] != "not applicable")
                or int(row["requests"] or 0) > 0
                or int(row["input tokens"] or 0) > 0
                or int(row["output tokens"] or 0) > 0
            )
            for row in rows
        ),
        "requests": sum(int(row["requests"] or 0) for row in rows),
        "tool calls": sum(int(row["tool calls"] or 0) for row in rows),
        "input tokens": sum(int(row["input tokens"] or 0) for row in rows),
        "output tokens": sum(int(row["output tokens"] or 0) for row in rows),
        "total tokens": sum(int(row["total tokens"] or 0) for row in rows),
    }
    return (
        '<div class="callout"><h3>Recorded totals</h3>'
        f'{_chips(totals)}</div><div class="subsection">{_table(rows)}</div>'
    )


def _render_timeline(
    attempts: Sequence[Mapping[str, Any]],
    resumes: Sequence[Mapping[str, Any]],
) -> str:
    artifacts: list[dict[str, Any]] = []
    for attempt in attempts:
        for name, reference in _mapping(attempt.get("artifacts")).items():
            artifact = _mapping(reference)
            artifacts.append(
                {
                    "stage": attempt.get("stage"),
                    "attempt": attempt.get("attemptId"),
                    "name": name,
                    "scope": artifact.get("scope"),
                    "assay": artifact.get("assay"),
                    "kind": artifact.get("kind"),
                    "artifact ID": artifact.get("artifactId"),
                }
            )
    return (
        "<h3>Stage attempts</h3>"
        + _table(
            attempts,
            columns=(
                "stage",
                "status",
                "durationSeconds",
                "actions",
                "reportCount",
                "artifactCount",
                "parentAttempts",
                "questionIds",
                "noteCount",
                "errorType",
            ),
        )
        + '<div class="subsection"><h3>Stage artifact inventory</h3>'
        + _table(artifacts, empty="No stage artifacts were recorded.")
        + "</div>"
        + '<div class="subsection"><h3>Resume lineage</h3>'
        + _table(
            resumes,
            columns=(
                "resumeId",
                "answeredStage",
                "answeredAttemptId",
                "questionIds",
            ),
            empty="No resume was required.",
        )
        + "</div>"
    )


def _render_document(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    request = _mapping(payload.get("request"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    enrichment = _latest(reports, "data_enrichment")
    experimental = _latest(reports, "experimental_context")
    parameter = _latest(reports, "parameter_tuning")
    biology = _latest(reports, "biological_interpretation")
    cluster_counts = {
        str(key): int(value)
        for key, value in _mapping(payload.get("clusterCounts")).items()
    }
    top_markers = _mappings(payload.get("topMarkers"))
    plots = {
        str(key): str(value)
        for key, value in _mapping(payload.get("plotFiles")).items()
    }
    plot_notes = [str(item) for item in payload.get("plotNotes", [])]
    attempts = _mappings(payload.get("stageAttempts"))
    resumes = _mappings(payload.get("workflowResumes"))
    workflow = _mapping(workflow_result.get("workflowRun"))
    workflow_id = str(workflow.get("workflowRunId") or "unavailable")
    total_cells = sum(cluster_counts.values())
    assay_plans = _mappings(plan.get("assays"))
    assays = [str(item.get("assay")) for item in assay_plans if item.get("assay")]
    metrics = [
        ("Final cells", total_cells or None),
        ("Final clusters", len(cluster_counts) or None),
        ("Assays", ", ".join(assays) or None),
        ("Candidates", parameter.get("totalCandidates")),
        ("Selected graph", final.get("graphMethod")),
        ("Marker assay", final.get("markerAssay")),
    ]
    metric_markup = "".join(
        '<span class="metric">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<span class="metric-value">{html.escape(_scalar(value))}</span></span>'
        for label, value in metrics
        if _present(value)
    )
    interpretation = {
        "status": biology.get("status"),
        "clusterInterpretations": biology.get("clusterInterpretations"),
        "evidenceIds": biology.get("evidenceIds"),
        "stopReason": biology.get("stopReason"),
    }
    study = enrichment.get("studyContextSummary") or {
        "originalContext": request.get("studyContext")
    }
    enrichment_summary = {
        "status": enrichment.get("status"),
        "policies": enrichment.get("policies"),
        "inspections": enrichment.get("inspections"),
        "evidenceIds": enrichment.get("evidenceIds"),
        "unresolvedQuestions": enrichment.get("unresolvedQuestions"),
    }
    experimental_summary = {
        "status": experimental.get("status"),
        "decision": experimental.get("decision"),
        "cellQc": experimental.get("cellQc"),
        "qcProfiles": experimental.get("qcProfiles"),
        "batchSafety": experimental.get("batchSafety"),
        "characterization": experimental.get("characterization"),
    }
    preprocessing_summary = {
        "primaryAssay": plan.get("primaryAssay"),
        "markerAssay": plan.get("markerAssay"),
        "pairedAssays": plan.get("pairedAssays"),
        "cellQc": plan.get("cellQc"),
        "assays": plan.get("assays"),
        "planChecksum": plan.get("planChecksum"),
    }
    limitations = {
        "Data Enrichment": enrichment.get("limitations"),
        "Experimental Context": experimental.get("notes"),
        "Parameter Tuning": parameter.get("limitations"),
        "Biological Interpretation": biology.get("limitations"),
        "Final analysis": final.get("limitations"),
        "Workflow": workflow_result.get("notes"),
        "Plots": plot_notes,
    }
    limitations = {key: value for key, value in limitations.items() if _present(value)}
    marker_columns = [
        key
        for key in (
            "group_id",
            "feature_name",
            "feature_id",
            "score",
            "frac_exp",
            "fold_change",
            "p_value",
        )
        if any(key in row for row in top_markers)
    ]
    provenance: list[dict[str, Any]] = [
        {"field": "Workflow run ID", "value": workflow_id},
        {"field": "Scarf version", "value": __version__},
        {"field": "Workspace", "value": workflow.get("workspace")},
        {"field": "Analysis store", "value": workflow.get("analysisStore")},
        {"field": "Dataset fingerprints", "value": workflow.get("datasetFingerprints")},
        {"field": "Generated at", "value": payload.get("generatedAt")},
        {"field": "Source path", "value": request.get("sourcePath")},
    ]
    raw_json = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, default=str
    )
    title = f"Scarf agent report {workflow_id}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{REPORT_STYLES}</style>
</head>
<body>
<header>
  <a class="brand" href="https://www.nygen.io/" target="_blank" rel="noopener noreferrer">Nygen Analytics</a>
  <span class="eyebrow">Scarf agent workflow</span>
</header>
<main>
  <p class="eyebrow">Completed analysis</p>
  <h1>Evidence from an automated analysis.</h1>
  <p class="lead">The workflow completed and its selected artifacts, decisions, and biological interpretation are summarized here.</p>
  <div class="pill-row">
    <span class="pill">Completed</span>
    <span class="pill pill-outline">{html.escape(_label(workflow_result.get("currentStage") or "completed"))}</span>
  </div>
  <div class="metric-grid">{metric_markup}</div>
  <nav class="pill-row" aria-label="Report sections">
    <a class="pill pill-outline" href="#visuals">Visual results</a>
    <a class="pill pill-outline" href="#biology">Biology</a>
    <a class="pill pill-outline" href="#context">Context</a>
    <a class="pill pill-outline" href="#tuning">Tuning</a>
    <a class="pill pill-outline" href="#workflow">Workflow</a>
  </nav>

  <aside class="product-callout">
    <p><strong>ScarfWeb</strong><br>Distributed, secure infrastructure for intuitive secondary analysis, browser-native.</p>
    <a class="pill" href="https://www.nygen.io/products/scarfweb" target="_blank" rel="noopener noreferrer">Explore ScarfWeb</a>
  </aside>

  <section class="section" id="visuals">
    <div class="section-heading"><h2>Visual results</h2><span class="pill pill-outline">Persisted artifacts</span></div>
    {_render_plots(plots, plot_notes)}
  </section>

  <section class="section" id="biology">
    <h2>Biological interpretation</h2>
    {_value(interpretation)}
    <div class="subsection"><h3>Final cluster sizes</h3>{_render_clusters(cluster_counts)}</div>
    <div class="subsection"><h3>Top marker evidence</h3>{_table(top_markers, columns=marker_columns, empty="No marker table was available.")}</div>
  </section>

  <section class="section"><h2>Treatment observations</h2>{_value(biology.get("treatmentObservations"))}</section>
  <section class="section"><h2>Follow-up recommendations</h2>{_value(biology.get("followUps"))}</section>

  <section class="section" id="context"><h2>Study context</h2>{_value(study)}</section>
  <section class="section"><h2>Data enrichment</h2>{_value(enrichment_summary)}</section>
  <section class="section"><h2>Experimental design</h2>{_value(experimental_summary)}</section>
  <section class="section"><h2>Preprocessing plan</h2>{_value(preprocessing_summary)}</section>

  <section class="section" id="tuning"><h2>Parameter tuning and graph selection</h2>{_render_parameter_tuning(parameter)}</section>
  <section class="section" id="workflow"><h2>Workflow execution</h2>{_render_timeline(attempts, resumes)}</section>
  <section class="section"><h2>Agent execution</h2>{_render_executions(reports)}</section>
  <section class="section"><h2>Limitations and workflow notes</h2>{_value(limitations) if limitations else '<p class="empty">No limitations were recorded.</p>'}</section>

  <section class="section">
    <h2>Technical provenance</h2>
    {_table(provenance)}
    <details><summary>Final immutable artifact references</summary>{_value(final)}</details>
    <details><summary>Structured report data</summary><pre>{html.escape(raw_json)}</pre></details>
  </section>
</main>
<footer>
  <p>Generated locally by Scarf agents. <a href="https://www.nygen.io/">Nygen Analytics</a></p>
</footer>
</body>
</html>
"""


def generate_agent_report(
    target: str | Path | DataStore,
    workflow_run_id: str,
    *,
    workspace: str | None = None,
) -> Path:
    """Generate a local HTML report for one completed automated workflow.

    The returned path points to ``index.html`` beneath the workflow's report
    directory. Existing derived report files may be replaced; immutable agent
    and orchestration records are only read.
    """
    root = _local_root(target)
    resolved_workspace = (
        target.workspace if isinstance(target, DataStore) else workspace
    )
    if (
        isinstance(target, DataStore)
        and workspace is not None
        and workspace != target.workspace
    ):
        raise ValueError("workspace does not match the DataStore workspace")
    workflow = load_agent_workflow(
        target,
        workflow_run_id,
        workspace=resolved_workspace,
    )
    store = _open_datastore(target, root, workflow)
    prefix, result, request = _load_completed_result(store, workflow)
    reports = _collect_reports(store, result)
    stage_attempts, resumes = _collect_history(store, prefix, workflow, request)

    active_root = (
        root if workflow.workspace is None else (root / workflow.workspace).resolve()
    )
    if not active_root.is_relative_to(root):
        raise ValueError("Workflow workspace resolves outside the analysis store")
    report_dir = (
        active_root / "agents" / "runs" / workflow_run_id / "report"
    ).resolve()
    if not report_dir.is_relative_to(active_root):
        raise ValueError("Agent report path resolves outside the analysis store")
    plot_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    cluster_counts, top_markers, plot_files, plot_notes = _collect_final_artifacts(
        store,
        result,
        plot_dir,
    )
    payload: dict[str, Any] = {
        "status": result.status,
        "currentStage": result.currentStage,
        "workflowRunId": workflow_run_id,
        "generatedAt": datetime.now(UTC).isoformat(),
        "request": request.request.model_dump(mode="json"),
        "effectiveConfig": request.config.model_dump(mode="json"),
        "workflowResult": result.model_dump(mode="json"),
        "reports": reports,
        "stageAttempts": stage_attempts,
        "workflowResumes": resumes,
        "clusterCounts": cluster_counts,
        "topMarkers": top_markers,
        "plotFiles": plot_files,
        "plotNotes": plot_notes,
    }
    document = _render_document(payload)
    destination = report_dir / "index.html"
    temporary = report_dir / f".index.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(document, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info(
        f"Generated HTML report for agent workflow {workflow_run_id}: {destination}"
    )
    return destination


__all__ = ["generate_agent_report"]
