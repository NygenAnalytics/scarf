import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, cast

import numpy as np

from ...metadata.rows import iter_metadata_column_blocks, metadata_missing_mask
from ...metrics import graph_connectivity
from ...storage.refs import ArtifactRef
from ...storage.types import as_zarr_array
from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..tools import artifact_reference, core_artifact_reference
from .contracts import (
    ArtifactRecord,
    IntegrationCandidateEvaluation,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
    ParameterTuningDependencies,
    ParameterTuningReport,
)

try:
    from pydantic_ai import RunContext
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


_RANDOM_SEED = 4444
_PCA_RANDOM_SEED = 4466

_METRIC_CACHE: ContextVar[dict[tuple[Any, ...], Any] | None] = ContextVar(
    "scarf_candidate_metric_cache", default=None
)
_DIAGNOSTIC_WORK: ContextVar[dict[str, dict[str, int]] | None] = ContextVar(
    "scarf_diagnostic_work", default=None
)


@contextmanager
def diagnostic_work() -> Iterator[dict[str, dict[str, int]]]:
    """Count agent operation calls, not internal core numerical rebuilds."""
    counts: dict[str, dict[str, int]] = {}
    token = _DIAGNOSTIC_WORK.set(counts)
    try:
        yield counts
    finally:
        _DIAGNOSTIC_WORK.reset(token)


def _diagnostic_count(name: str, event: str, count: int = 1) -> None:
    counts = _DIAGNOSTIC_WORK.get()
    if counts is not None:
        row = counts.setdefault(
            name,
            dict.fromkeys(
                (
                    "attempted",
                    "completed",
                    "failed",
                    "cacheHits",
                    "restored",
                    "artifactReuses",
                ),
                0,
            ),
        )
        row[event] += count


def diagnostic_call[**P, T](
    name: str, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Record an invoked operation while preserving its result and exception."""
    _diagnostic_count(name, "attempted")
    try:
        result = operation(*args, **kwargs)
    except BaseException:
        _diagnostic_count(name, "failed")
        raise
    _diagnostic_count(name, "completed")
    return result


def diagnostic_reuse(
    name: str,
    kind: Literal["cacheHits", "restored", "artifactReuses"] = "cacheHits",
    count: int = 1,
) -> None:
    """Record reuse only when the agent has directly established it."""
    _diagnostic_count(name, kind, count)


@contextmanager
def candidate_metric_cache() -> Iterator[None]:
    """Reuse exact-input metrics only for the current orchestration stage."""
    token = _METRIC_CACHE.set({})
    try:
        yield
    finally:
        _METRIC_CACHE.reset(token)


def _cached_candidate_metric[T](key: tuple[Any, ...], compute: Callable[[], T]) -> T:
    name = f"metric.{key[1]}"
    cache = _METRIC_CACHE.get()
    if cache is None:
        return diagnostic_call(name, compute)
    if key not in cache:
        cache[key] = diagnostic_call(name, compute)
    else:
        diagnostic_reuse(name)
    return cast(T, cache[key])


def _metric_metadata_key(store: Any, column: str) -> str | None:
    """Fingerprint live metric inputs so edits cannot reuse stale evidence."""
    if _METRIC_CACHE.get() is None:
        return None
    return _metadata_column_fingerprint(store.cells, column)


def _metadata_column_fingerprint(metadata: Any, column: str) -> str:
    """Hash metadata values in bounded blocks, including scalar type identity."""
    digest = hashlib.sha256()
    for block in iter_metadata_column_blocks(metadata, column):
        digest.update(str(block.dtype).encode())
        digest.update(str(block.shape).encode())
        digest.update(
            json.dumps(
                [(type(value).__name__, repr(value)) for value in block.tolist()]
            ).encode()
            if block.dtype.hasobject
            else block.tobytes()
        )
    missing = metadata_missing_mask(metadata, column)
    digest.update(b"missing:none" if missing is None else b"missing:present")
    if missing is not None:
        for start in range(0, len(missing), 65_536):
            digest.update(
                np.asarray(missing[start : start + 65_536], dtype=bool).tobytes()
            )
    return digest.hexdigest()


def _final_graph_options(
    report: ParameterTuningReport,
    integration_evaluations: Sequence[IntegrationCandidateEvaluation],
) -> dict[str, dict[str, Any]]:
    """Return the exact eligible graph options and option-scoped evidence."""

    assay_reports = report.assayReports or {report.fromAssay: report}
    report_cell_selection = core_artifact_reference(report.cellSelection)
    if not isinstance(report_cell_selection, ArtifactRef):
        raise ValueError("Parameter tuning report lacks an exact cell selection")
    options: dict[str, dict[str, Any]] = {}
    for assay, assay_report in assay_reports.items():
        candidate_id = assay_report.recommendedCandidateId
        if candidate_id is None:
            continue
        native_evaluation = next(
            (
                item
                for item in assay_report.evaluations
                if item.candidateId == candidate_id
            ),
            None,
        )
        if (
            native_evaluation is None
            or native_evaluation.status != "done"
            or not native_evaluation.eligible
            or "clusters" not in native_evaluation.artifacts
            or "connectivityMap" not in native_evaluation.artifacts
            or not native_evaluation.evidenceIds
        ):
            continue
        if (
            core_artifact_reference(native_evaluation.cellSelection)
            != report_cell_selection
        ):
            raise ValueError("Native graph option uses a different cell selection")
        option_id = f"native:{assay}:{candidate_id}"
        option_evidence = [
            f"native:{assay}:{evidence_id}"
            for evidence_id in native_evaluation.evidenceIds
        ]
        evaluation_payload = native_evaluation.model_dump()
        evaluation_payload["evidenceIds"] = option_evidence
        options[option_id] = {
            "optionId": option_id,
            "graphMethod": "native",
            "nativeAssay": assay,
            "nativeCandidateId": candidate_id,
            "evaluation": evaluation_payload,
            "evidenceIds": option_evidence,
        }
    for integration_evaluation in integration_evaluations:
        if (
            integration_evaluation.status != "done"
            or not integration_evaluation.eligible
        ):
            continue
        if (
            integration_evaluation.clusterArtifact is None
            or integration_evaluation.graphArtifact is None
        ):
            continue
        if not integration_evaluation.evidenceIds:
            continue
        if (
            core_artifact_reference(integration_evaluation.cellSelection)
            != report_cell_selection
        ):
            raise ValueError("Integrated graph option uses a different cell selection")
        if not integration_evaluation.integrationId:
            raise ValueError("Eligible integration evaluations require integrationId")
        if (
            integration_evaluation.graphArtifact.scope != "datastore"
            or integration_evaluation.graphArtifact.assay is not None
            or integration_evaluation.clusterArtifact.scope != "datastore"
            or integration_evaluation.clusterArtifact.assay is not None
            or integration_evaluation.graphArtifact.kind != "integrated_graph"
            or integration_evaluation.clusterArtifact.kind
            not in {"cluster_labels", "cluster_cut"}
        ):
            raise ValueError(
                "Integrated graph and cluster artifacts must be datastore-scoped "
                "without an assay"
            )
        if (
            integration_evaluation.method == "wnn"
            and integration_evaluation.metrics.modalityWeightsValid is not True
        ):
            continue
        option_id = f"integration:{integration_evaluation.integrationId}"
        if option_id in options:
            raise ValueError(
                f"Duplicate integration id {integration_evaluation.integrationId!r}"
            )
        options[option_id] = {
            "optionId": option_id,
            "graphMethod": integration_evaluation.method,
            "integrationId": integration_evaluation.integrationId,
            "evaluation": integration_evaluation.model_dump(),
            "evidenceIds": list(integration_evaluation.evidenceIds),
        }
    return options


def normalized_artifact_shape(store: Any, normalized: Any) -> tuple[int, int]:
    """Return the exact cell-by-feature shape of a normalized artifact."""

    group = store.load_artifact(normalized)
    if "data" not in group:
        raise ValueError("Normalized artifact does not contain a data matrix")
    shape = getattr(group["data"], "shape", None)
    if not isinstance(shape, tuple | list) or len(shape) != 2:
        raise ValueError("Normalized artifact data must be two-dimensional")
    n_cells, n_features = map(int, shape)
    if n_cells < 2 or n_features < 2:
        raise ValueError(
            "Parameter tuning requires at least two cells and two selected features"
        )
    return n_cells, n_features


def validate_parameter_candidate_rank(
    candidate: ParameterCandidate,
    normalized_shape: tuple[int, int],
    *,
    identity_feature_limit: int = 64,
) -> int:
    """Validate a candidate before any branch operation and return output rank."""

    n_cells, n_features = normalized_shape
    if candidate.neighborsK >= n_cells:
        raise ValueError(
            f"neighborsK={candidate.neighborsK} requires more than "
            f"{candidate.neighborsK} selected cells; observed {n_cells}"
        )
    if candidate.reductionMethod == "pca":
        if candidate.dimensions + 1 > min(n_cells, n_features):
            raise ValueError(
                f"PCA dimensions={candidate.dimensions} requires at least "
                f"{candidate.dimensions + 1} cells and selected features; "
                f"observed shape {normalized_shape}"
            )
        return candidate.dimensions
    if candidate.reductionMethod == "lsi":
        required_rank = candidate.dimensions + 1
        if required_rank > min(n_cells, n_features):
            raise ValueError(
                "LSI dimensions, including the skipped component, exceed the "
                f"normalized matrix rank for shape {normalized_shape}"
            )
        return candidate.dimensions
    if n_features > identity_feature_limit:
        raise ValueError(
            f"Identity reduction supports at most {identity_feature_limit} selected "
            f"features; observed {n_features}"
        )
    if candidate.dimensions != n_features:
        raise ValueError(
            "Identity reduction dimensions must equal the exact normalized feature "
            f"count {n_features}; received {candidate.dimensions}"
        )
    return n_features


def run_candidate_reduction(
    store: Any,
    *,
    normalized: Any,
    candidate: ParameterCandidate,
    normalized_shape: tuple[int, int],
    identity_feature_limit: int = 64,
) -> tuple[Any, str, int]:
    """Run one validated modality-aware reduction with public Scarf methods."""

    effective_dimensions = validate_parameter_candidate_rank(
        candidate,
        normalized_shape,
        identity_feature_limit=identity_feature_limit,
    )
    if candidate.reductionMethod == "pca":
        ref = diagnostic_call(
            "core.pca",
            store.run_pca,
            normalized,
            dims=candidate.dimensions,
            feat_scaling=True,
            show_elbow_plot=False,
            invalidate_cache=False,
        )
        return ref, "pca", effective_dimensions
    if candidate.reductionMethod == "lsi":
        ref = diagnostic_call(
            "core.lsi",
            store.run_lsi,
            normalized,
            dims=candidate.dimensions,
            skip_first=True,
            rand_state=_PCA_RANDOM_SEED,
            invalidate_cache=False,
        )
        return ref, "lsi", effective_dimensions
    loadings = np.eye(normalized_shape[1], dtype=np.float64)
    ref = diagnostic_call(
        "core.customReduction",
        store.run_custom_reduction,
        loadings,
        normalized,
        invalidate_cache=False,
    )
    return ref, "identity", effective_dimensions


def _bounded_membership_summary(
    values: Any,
    labels: np.ndarray,
    *,
    maximum_sample_size: int = 65_536,
) -> tuple[float, float, float, dict[str, float], int]:
    if len(values.shape) != 1 or values.shape != labels.shape:
        raise ValueError("Membership strengths must align with cluster labels")
    n_values = int(values.shape[0])
    if n_values < 1:
        raise ValueError("Membership strengths cannot be empty")
    stride = max(1, (n_values + maximum_sample_size - 1) // maximum_sample_size)
    total = 0.0
    sampled_values: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []
    for start in range(0, n_values, 65_536):
        block = np.asarray(values[start : start + 65_536], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Membership strengths must be finite")
        total += float(block.sum())
        offset = (-start) % stride
        sampled_values.append(block[offset::stride])
        sampled_labels.append(labels[start + offset : start + len(block) : stride])
    sample = np.concatenate(sampled_values)
    sample_labels = np.concatenate(sampled_labels)
    by_cluster = {
        str(cluster): float(np.median(sample[sample_labels == cluster]))
        for cluster in np.unique(sample_labels)
    }
    return (
        total / n_values,
        float(np.median(sample)),
        float(np.quantile(sample, 0.1)),
        by_cluster,
        int(len(sample)),
    )


def _collect_cluster_structure_metrics(
    store: Any,
    *,
    cluster_ref: Any,
    graph_ref: Any,
    cluster_values: np.ndarray,
    candidate_id: str,
    metrics: ParameterMetrics,
    evidence_ids: list[str],
    warnings: list[str],
) -> ArtifactRef | None:
    calculate_membership = getattr(store, "calc_membership_strength", None)
    if not callable(calculate_membership):
        return None
    membership_ref: ArtifactRef | None = None
    try:
        membership_ref = diagnostic_call(
            "core.membershipStrength",
            calculate_membership,
            cluster_ref,
            graph_ref,
            invalidate_cache=False,
        )
        membership_group = store.load_artifact(membership_ref)
        membership_values = as_zarr_array(
            membership_group["values"],
            name="values",
        )
        mean, median, p10, by_cluster, sample_size = _cached_candidate_metric(
            (id(store), "membership_summary", membership_ref, cluster_ref),
            lambda: _bounded_membership_summary(membership_values, cluster_values),
        )
        metrics.membershipStrengthMean = mean
        metrics.membershipStrengthMedian = median
        metrics.membershipStrengthP10 = p10
        metrics.membershipStrengthByCluster = by_cluster
        metrics.membershipStrengthSampleSize = sample_size
        evidence_ids.append(f"candidate:{candidate_id}:membershipStrength")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        warnings.append(f"Cluster membership strength unavailable: {exc}")
        membership_ref = None

    try:
        graph_group = store.load_artifact(graph_ref)
        graph_edges = as_zarr_array(graph_group["edges"], name="edges")
        connectivity = _cached_candidate_metric(
            (id(store), "cluster_connectivity", graph_ref, cluster_ref),
            lambda: float(graph_connectivity(graph_edges, cluster_values)),
        )
        if np.isfinite(connectivity):
            metrics.clusterConnectivity = connectivity
            evidence_ids.append(f"candidate:{candidate_id}:clusterConnectivity")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        warnings.append(f"Cluster connectivity unavailable: {exc}")
    return membership_ref


def _collect_parameter_candidate_metrics(
    deps: ParameterTuningDependencies,
    *,
    candidate: ParameterCandidate,
    candidate_id: str,
    reduction_ref: Any,
    neighbors_ref: Any,
    graph_ref: Any,
    cluster_ref: Any,
    cluster_column: str,
    evidence_ids: list[str],
    warnings: list[str],
) -> tuple[ParameterMetrics, list[str], ArtifactRef | None]:
    store = deps.store
    cluster_group = store.load_artifact(cluster_ref)
    cluster_data = cluster_group["values"]
    cluster_values = np.asarray(cluster_data[:])
    if cluster_values.ndim != 1 or len(cluster_values) == 0:
        raise ValueError("Cluster artifact must contain one non-empty label vector")
    if np.any(cluster_values < 0):
        raise ValueError("Cluster artifact contains invalid negative labels")
    _, cluster_counts = np.unique(cluster_values, return_counts=True)
    n_clusters = int(len(cluster_counts))
    min_cluster_cells = int(cluster_counts.min())
    min_cluster_fraction = float(min_cluster_cells / len(cluster_values))
    metrics = ParameterMetrics(
        nClusters=n_clusters,
        minClusterCells=min_cluster_cells,
        minClusterFraction=min_cluster_fraction,
    )
    evidence_ids.append(f"candidate:{candidate_id}:clusters")
    membership_ref = _collect_cluster_structure_metrics(
        store,
        cluster_ref=cluster_ref,
        graph_ref=graph_ref,
        cluster_values=cluster_values,
        candidate_id=candidate_id,
        metrics=metrics,
        evidence_ids=evidence_ids,
        warnings=warnings,
    )

    try:
        graph_scores = _cached_candidate_metric(
            (
                id(store),
                "graph_silhouette",
                neighbors_ref,
                cluster_ref,
                _RANDOM_SEED,
                11,
            ),
            lambda: diagnostic_call(
                "core.graphSilhouette",
                store.metric_graph_silhouette,
                neighbors_ref,
                cluster_ref,
                random_seed=_RANDOM_SEED,
                sample_size=11,
            ),
        )
        if graph_scores is not None:
            finite_scores = np.asarray(graph_scores, dtype=float)
            finite_scores = finite_scores[np.isfinite(finite_scores)]
            if len(finite_scores):
                metrics.graphSilhouetteMedian = float(np.median(finite_scores))
                evidence_ids.append(f"candidate:{candidate_id}:graphSilhouette")
    except (KeyError, TypeError, ValueError) as exc:
        warnings.append(f"Graph silhouette unavailable: {exc}")

    if candidate.reductionMethod == "pca":
        try:

            def separability_values() -> dict[str, Any]:
                separability = diagnostic_call(
                    "core.clusterSeparability",
                    store.metric_cluster_separability,
                    reduction_ref,
                    {cluster_column: cluster_ref},
                    random_seed=_RANDOM_SEED,
                )
                table = separability.clustering_scores
                rows = table.loc[table["clustering"] == cluster_column]
                return dict(rows.iloc[0]) if len(rows) else {}

            row = _cached_candidate_metric(
                (
                    id(store),
                    "cluster_separability",
                    reduction_ref,
                    cluster_ref,
                    _RANDOM_SEED,
                ),
                separability_values,
            )
            if row:
                for field_name, column_name, evidence_name in (
                    ("pcaSilhouette", "silhouette_score", "pcaSilhouette"),
                    ("macroF1", "macro_f1_mean", "macroF1"),
                    ("weightedF1", "weighted_f1_mean", "weightedF1"),
                ):
                    value = row[column_name]
                    if value is not None and np.isfinite(float(value)):
                        setattr(metrics, field_name, float(value))
                        evidence_ids.append(f"candidate:{candidate_id}:{evidence_name}")
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"PCA cluster separability unavailable: {exc}")

    _collect_covariate_metrics(
        deps,
        candidate=candidate,
        candidate_id=candidate_id,
        neighbors_ref=neighbors_ref,
        graph_ref=graph_ref,
        metrics=metrics,
        evidence_ids=evidence_ids,
        warnings=warnings,
    )

    eligibility_reasons: list[str] = []
    if n_clusters < 2:
        eligibility_reasons.append("fewer than two clusters")
    if min_cluster_cells < deps.minClusterCells:
        eligibility_reasons.append(
            f"smallest cluster has {min_cluster_cells} cells; "
            f"minimum is {deps.minClusterCells}"
        )
    return metrics, eligibility_reasons, membership_ref


def _collect_covariate_metrics(
    deps: ParameterTuningDependencies,
    *,
    candidate: ParameterCandidate,
    candidate_id: str,
    neighbors_ref: ArtifactRef,
    graph_ref: ArtifactRef,
    metrics: ParameterMetrics,
    evidence_ids: list[str],
    warnings: list[str],
) -> None:
    """Measure exact typed design effects on an existing graph and neighborhood."""
    store = deps.store
    perplexity = max(1.0, float(candidate.neighborsK // 3))
    for column in deps.batchColumns:
        try:
            score = float(
                _cached_candidate_metric(
                    (
                        id(store),
                        "batch_mixing",
                        neighbors_ref,
                        column,
                        _metric_metadata_key(store, column),
                        perplexity,
                    ),
                    lambda: diagnostic_call(
                        "core.batchMixing",
                        store.metric_proportional_batch_mixing,
                        column,
                        neighbors_ref,
                        perplexity=perplexity,
                    ),
                )
            )
            if np.isfinite(score):
                metrics.batchMixing[column] = score
                evidence_ids.append(f"candidate:{candidate_id}:batchMixing:{column}")
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Batch mixing for {column!r} unavailable: {exc}")

    for column in deps.preservationColumns:
        if deps.columnKinds.get(column) == "continuous":
            warnings.append(
                f"Matched graph preservation for continuous column {column!r} is unsupported; "
                "PCA association is descriptive evidence only."
            )
            continue
        scores: dict[str, float] = {}
        try:
            clisi = float(
                _cached_candidate_metric(
                    (
                        id(store),
                        "clisi",
                        neighbors_ref,
                        column,
                        _metric_metadata_key(store, column),
                        None,
                        True,
                    ),
                    lambda: diagnostic_call(
                        "core.clisi",
                        store.metric_clisi,
                        column,
                        neighbors_ref,
                        perplexity=None,
                        scale=True,
                    ),
                )
            )
            if np.isfinite(clisi):
                scores["clisi"] = clisi
                evidence_ids.append(f"candidate:{candidate_id}:clisi:{column}")
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"cLISI for {column!r} unavailable: {exc}")
        try:
            connectivity = float(
                _cached_candidate_metric(
                    (
                        id(store),
                        "protected_connectivity",
                        graph_ref,
                        column,
                        _metric_metadata_key(store, column),
                    ),
                    lambda: diagnostic_call(
                        "core.graphConnectivity",
                        store.metric_graph_connectivity,
                        column,
                        graph_ref,
                    ),
                )
            )
            if np.isfinite(connectivity):
                scores["graphConnectivity"] = connectivity
                evidence_ids.append(
                    f"candidate:{candidate_id}:graphConnectivity:{column}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Graph connectivity for {column!r} unavailable: {exc}")
        if scores:
            metrics.biologicalPreservation[column] = scores

    if deps.protectedCombinations:
        from ...metrics import clisi_knn
        from ..experimental_context.characterization import _SelectionBoundCells
        from ..experimental_context.comparisons import combination_labels

        bound_cells = _SelectionBoundCells(store.zw, store.cells, deps.cellSelection)
        neighbor_group = store.load_artifact(neighbors_ref)
        graph_group = store.load_artifact(graph_ref)
        for columns in deps.protectedCombinations:
            name = "joint:" + json.dumps(list(columns), separators=(",", ":"))
            metadata_key = tuple(
                _metric_metadata_key(store, column) for column in columns
            )
            labels = combination_labels(bound_cells, columns)
            scores = _cached_candidate_metric(
                (
                    id(store),
                    "protected_combination",
                    neighbors_ref,
                    graph_ref,
                    columns,
                    metadata_key,
                ),
                lambda: {
                    "clisi": float(
                        clisi_knn(
                            as_zarr_array(
                                neighbor_group["distances"], name="distances"
                            ),
                            as_zarr_array(neighbor_group["indices"], name="indices"),
                            labels,
                            perplexity=None,
                            scale=True,
                        )
                    ),
                    "graphConnectivity": float(
                        graph_connectivity(
                            as_zarr_array(graph_group["edges"], name="edges"),
                            labels,
                        )
                    ),
                },
            )
            if not all(np.isfinite(value) for value in scores.values()):
                raise ValueError("Protected combination metrics must be finite")
            metrics.biologicalPreservation[name] = scores
            evidence_ids.append(f"candidate:{candidate_id}:{name}")


def refresh_candidate_design_evidence(
    deps: ParameterTuningDependencies,
    evaluation: ParameterCandidateEvaluation,
) -> ParameterCandidateEvaluation:
    """Reassess a revised design without repeating primary analysis or doublets."""
    evaluation = evaluation.model_copy(deep=True)
    evaluation.metrics.batchMixing = {}
    evaluation.metrics.biologicalPreservation = {}
    identifiers = (":batchMixing:", ":clisi:", ":graphConnectivity:", ":joint:")
    evaluation.evidenceIds = [
        item
        for item in evaluation.evidenceIds
        if not any(identifier in item for identifier in identifiers)
    ]
    prefixes = (
        "Batch mixing for ",
        "cLISI for ",
        "Graph connectivity for ",
        "Matched graph preservation for continuous column ",
    )
    evaluation.warnings = [
        item for item in evaluation.warnings if not item.startswith(prefixes)
    ]
    _collect_covariate_metrics(
        deps,
        candidate=evaluation.parameters,
        candidate_id=evaluation.candidateId,
        neighbors_ref=core_artifact_reference(evaluation.artifacts["neighbors"]),
        graph_ref=core_artifact_reference(evaluation.artifacts["connectivityMap"]),
        metrics=evaluation.metrics,
        evidence_ids=evaluation.evidenceIds,
        warnings=evaluation.warnings,
    )
    return evaluation


def execute_parameter_candidate(
    deps: ParameterTuningDependencies,
    candidate_id: str,
) -> ParameterCandidateEvaluation:
    """Execute one allowlisted candidate without model involvement."""

    with deps.executionLock:
        if candidate_id in deps.evaluations:
            diagnostic_reuse("candidate.evaluation")
            logger.debug(
                f"Parameter candidate {candidate_id!r} for assay "
                f"{deps.fromAssay!r} reused its completed evaluation"
            )
            return deps.evaluations[candidate_id]
        if candidate_id not in deps.candidates:
            logger.warning(
                f"Parameter candidate {candidate_id!r} is not authorized for "
                f"assay {deps.fromAssay!r}"
            )
            return ParameterCandidateEvaluation(
                candidateId=candidate_id,
                status="failed",
                error=(
                    f"Unknown candidate id {candidate_id!r}; allowed ids are "
                    f"{sorted(deps.candidates)}"
                ),
            )
        if len(deps.executionOrder) >= deps.maxCandidates:
            logger.warning(
                f"Parameter candidate {candidate_id!r} was not executed because "
                f"assay {deps.fromAssay!r} reached its limit of "
                f"{deps.maxCandidates} candidates"
            )
            return ParameterCandidateEvaluation(
                candidateId=candidate_id,
                phase=deps.candidatePhases.get(candidate_id, "initial"),
                harmonyBatchColumns=(
                    list(deps.batchColumns)
                    if deps.candidates[candidate_id].useHarmony
                    else []
                ),
                status="failed",
                parameters=deps.candidates[candidate_id],
                error=f"Candidate execution limit {deps.maxCandidates} reached",
            )

        candidate = deps.candidates[candidate_id]
        deps.executionOrder.append(candidate_id)
        logger.debug(f"Executing candidate {candidate_id!r} for {deps.fromAssay!r}")
        logger.info(
            f"Comparing settings for {deps.fromAssay}: {candidate.reductionMethod.upper()} "
            f"dimensions={candidate.dimensions}, neighbors={candidate.neighborsK}, "
            f"resolution={candidate.leidenResolution}, "
            f"harmony={candidate.useHarmony}"
        )
        if candidate.useHarmony and not deps.batchColumns:
            logger.warning(
                f"Parameter candidate {candidate_id!r} cannot run Harmony because "
                "no batch columns were authorized"
            )
            evaluation = ParameterCandidateEvaluation(
                candidateId=candidate_id,
                phase=deps.candidatePhases.get(candidate_id, "initial"),
                harmonyBatchColumns=[],
                status="failed",
                parameters=candidate,
                error="Harmony candidate requires at least one authorized batch column",
            )
            deps.evaluations[candidate_id] = evaluation
            return evaluation

        store = deps.store
        artifacts: dict[str, ArtifactRecord] = {}
        warnings: list[str] = []
        evidence_ids: list[str] = []
        cluster_label = f"agent_tuning_{candidate_id}"
        cluster_column = f"{deps.fromAssay}_{cluster_label}"

        try:
            normalized_shape = deps.normalizedShape or normalized_artifact_shape(
                store,
                deps.normalized,
            )
            effective_dimensions = validate_parameter_candidate_rank(
                candidate,
                normalized_shape,
                identity_feature_limit=deps.identityFeatureLimit,
            )
            reduction_ref, reduction_key, _ = run_candidate_reduction(
                store,
                normalized=deps.normalized,
                candidate=candidate,
                normalized_shape=normalized_shape,
                identity_feature_limit=deps.identityFeatureLimit,
            )
            artifacts[reduction_key] = ArtifactRecord.from_ref(reduction_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed "
                f"{reduction_key} reduction"
            )

            coordinates_ref = reduction_ref
            if candidate.useHarmony:
                coordinates_ref = diagnostic_call(
                    "core.harmony",
                    store.run_harmony,
                    reduction_ref,
                    list(deps.batchColumns),
                    invalidate_cache=False,
                )
                artifacts["harmony"] = ArtifactRecord.from_ref(coordinates_ref)
                logger.debug(
                    f"Parameter candidate {candidate_id!r}: completed Harmony "
                    f"using {len(deps.batchColumns)} batch column(s)"
                )

            ann_ref = diagnostic_call(
                "core.ann",
                store.build_ann_index,
                coordinates_ref,
                ann_metric="l2",
                ann_parallel=False,
                rand_state=_PCA_RANDOM_SEED,
                invalidate_cache=False,
            )
            artifacts["annIndex"] = ArtifactRecord.from_ref(ann_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed ANN indexing"
            )

            neighbors_ref = diagnostic_call(
                "core.neighbors",
                store.query_neighbors,
                ann_ref,
                coordinates=coordinates_ref,
                k=candidate.neighborsK,
                invalidate_cache=False,
            )
            artifacts["neighbors"] = ArtifactRecord.from_ref(neighbors_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed neighbor query"
            )

            graph_ref = diagnostic_call(
                "core.graph",
                store.build_connectivity_map,
                neighbors_ref,
                local_connectivity=1.0,
                bandwidth=1.5,
                invalidate_cache=False,
            )
            artifacts["connectivityMap"] = ArtifactRecord.from_ref(graph_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed connectivity map"
            )

            cluster_ref = diagnostic_call(
                "core.partition",
                store.run_leiden_clustering,
                graph_ref,
                resolution=candidate.leidenResolution,
                backend="igraph",
                symmetric_graph=False,
                graph_upper_only=False,
                random_seed=_RANDOM_SEED,
                invalidate_cache=False,
            )
            artifacts["clusters"] = ArtifactRecord.from_ref(cluster_ref)
            logger.debug(
                f"Parameter candidate {candidate_id!r}: completed Leiden clustering"
            )

            (
                metrics,
                eligibility_reasons,
                membership_ref,
            ) = _collect_parameter_candidate_metrics(
                deps,
                candidate=candidate,
                candidate_id=candidate_id,
                reduction_ref=reduction_ref,
                neighbors_ref=neighbors_ref,
                graph_ref=graph_ref,
                cluster_ref=cluster_ref,
                cluster_column=cluster_column,
                evidence_ids=evidence_ids,
                warnings=warnings,
            )
            if membership_ref is not None:
                artifacts["membershipStrength"] = ArtifactRecord.from_ref(
                    membership_ref
                )

            evaluation = ParameterCandidateEvaluation(
                candidateId=candidate_id,
                phase=deps.candidatePhases.get(candidate_id, "initial"),
                harmonyBatchColumns=(
                    list(deps.batchColumns) if candidate.useHarmony else []
                ),
                status="done",
                eligible=not eligibility_reasons,
                parameters=candidate,
                artifacts=artifacts,
                cellSelection=artifact_reference(deps.cellSelection),
                clusterColumn=cluster_column,
                clusterLabel=cluster_label,
                effectiveDimensions=effective_dimensions,
                metrics=metrics,
                evidenceIds=evidence_ids,
                eligibilityReasons=eligibility_reasons,
                warnings=warnings,
            )
            logger.info(
                f"Compared settings: {metrics.nClusters} clusters; "
                f"smallest population has {metrics.minClusterCells} cells; "
                f"{'ready for assessment' if evaluation.eligible else 'candidate checks unresolved'}."
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            evaluation = ParameterCandidateEvaluation(
                candidateId=candidate_id,
                phase=deps.candidatePhases.get(candidate_id, "initial"),
                harmonyBatchColumns=(
                    list(deps.batchColumns) if candidate.useHarmony else []
                ),
                status="failed",
                parameters=candidate,
                artifacts=artifacts,
                cellSelection=(
                    artifact_reference(deps.cellSelection)
                    if deps.cellSelection is not None
                    else None
                ),
                evidenceIds=evidence_ids,
                warnings=warnings,
                error=str(exc),
            )
            logger.warning(
                f"Parameter candidate {candidate_id!r} for assay "
                f"{deps.fromAssay!r} failed: {exc}"
            )

        deps.evaluations[candidate_id] = evaluation
        return evaluation


async def evaluate_parameter_candidate(
    ctx: RunContext[ParameterTuningDependencies],
    candidate_id: str,
) -> ParameterCandidateEvaluation:
    """Expose deterministic candidate execution as a bounded agent tool."""

    return execute_parameter_candidate(ctx.deps, candidate_id)
