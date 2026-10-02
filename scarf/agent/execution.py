"""Small adapters from agent choices to public, immutable PipelineRun APIs."""

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .models import AnalysisConfig, AnalysisInputError, Candidate
from .diagnostics import finalist_composition, representation_diagnostics

_SILHOUETTE_CELLS = 2_000
_DIAGNOSTIC_CELLS = 10_000


def execute_pipeline(
    store: Any,
    prepared: dict[str, Any],
    candidate: Candidate,
    config: AnalysisConfig,
    *,
    label: str,
    resolution: float | None = None,
    markers: bool = False,
    final: bool = False,
) -> Any:
    """Execute one declared candidate with the core pipeline's durable ledger."""
    if candidate.useHarmony and not prepared["correctionEligible"]:
        raise AnalysisInputError(
            "Harmony requires an authorized, crossed experimental design"
        )
    if resolution is not None and resolution not in config.resolutions:
        raise AnalysisInputError(
            "The selected resolution is outside the configured panel"
        )
    if candidate.pcaDims >= min(
        candidate.hvgCount, prepared["availableFeatures"], prepared["retainedCells"]
    ):
        raise AnalysisInputError(
            "PCA dimensions must be smaller than selected cells and features"
        )
    if candidate.neighborsK >= prepared["retainedCells"]:
        raise AnalysisInputError(
            "Neighbor count must be smaller than the retained cohort"
        )
    seed = config.randomSeed
    leiden: dict[str, Any] = {
        "partitions": list(config.resolutions) if resolution is None else [resolution],
        "random_seed": seed,
        # The core selector has a larger fixed sample. A provisional selected
        # partition skips it; bounded agent evidence scores every offered row.
        "selected": config.resolutions[0] if resolution is None else resolution,
    }
    if resolution is not None:
        leiden["selected"] = resolution
    params: dict[str, Any] = {
        "hvg": {"blacklist": prepared["blacklist"]},
        "ann_index": {"rand_state": seed, "ann_parallel": False},
        "leiden": leiden,
        "tsne": False,
        "membership_strength": False,
    }
    if candidate.useHarmony:
        params["harmony"] = {
            "batch_columns": prepared["technicalBatchColumns"],
            "harmony_params": {"random_state": seed},
        }
    if final:
        params["embedding_initialization"] = {"rand_state": seed}
        params["umap"] = {"random_seed": seed}
    score_doublets = config.scoreDoublets and (markers or final)
    if score_doublets:
        params["doublets"] = {"random_seed": seed}
    # Put batches first even for native runs. Core snapshot order participates
    # in filtering identity, so matched branches must freeze the same snapshot.
    snapshots = list(
        dict.fromkeys(
            [
                *prepared["technicalBatchColumns"],
                *prepared["snapshotColumns"],
            ]
        )
    )
    return store.pipeline.run(
        assay=prepared["assay"],
        label=label,
        cell_key=prepared["cellKey"],
        filtering=prepared["filtering"],
        hvg_count=candidate.hvgCount,
        pca_dims=candidate.pcaDims,
        neighbors_k=candidate.neighborsK,
        umap=final,
        cell_cycle=False,
        paris=False,
        doublets=score_doublets,
        markers=markers or final,
        snapshot_columns=snapshots,
        params=params,
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) else None


def _parameters(candidate: Candidate) -> dict[str, Any]:
    return candidate.model_dump(exclude={"candidateId", "parentId"})


def summarize_candidate(
    store: Any,
    run: Any,
    candidate: Candidate,
    prepared: dict[str, Any],
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Read saved scalar diagnostics and bounded cluster-size summaries."""
    if run.status != "completed":
        raise AnalysisInputError(
            "Only completed pipeline runs supply candidate evidence"
        )
    from scarf.metrics.cluster_selection import select_clusters_by_silhouette

    scores: dict[str, float | None] = {}
    limitations: list[str] = []
    coordinates = store.load_artifact(
        run["harmony"] if candidate.useHarmony else run["pca"]
    )["data"]
    partition_arrays = [
        (key, store.load_artifact(run[key])["values"])
        for key in run
        if key.startswith("leiden_")
    ]
    sample_size = min(coordinates.shape[0], _SILHOUETTE_CELLS)
    try:
        selection = select_clusters_by_silhouette(
            coordinates,
            partition_arrays,
            seed=config.randomSeed,
            max_sample_size=_SILHOUETTE_CELLS,
            working_memory_mib=max(1, min(128, store.memoryBytes // (4 * 1024**2))),
        )
    except ValueError as error:
        if not str(error).startswith(
            "No clustering candidate is silhouette-scoreable:"
        ):
            raise
        limitations.append(str(error))
    else:
        scores = dict(
            zip(selection.candidate_keys, map(_finite, selection.scores), strict=True)
        )
        limitations.extend(
            str(reason) for reason in selection.invalid_reasons if reason
        )
    partitions = []
    for key in run:
        if not key.startswith("leiden_"):
            continue
        resolution = float(key.removeprefix("leiden_"))
        labels = run.cells.fetch(key)
        counts = pd.Series(labels).astype(str).value_counts().sort_index()
        partitions.append(
            {
                "optionId": f"{candidate.candidateId}:r{resolution:g}",
                "candidateId": candidate.candidateId,
                "resolution": resolution,
                "score": scores.get(key),
                "count": int(len(labels)),
                "clusterCount": int(len(counts)),
                "clusterCounts": {
                    str(name): int(count) for name, count in counts.head(64).items()
                },
                "clusterCountsTruncated": len(counts) > 64,
            }
        )
    result = {
        "candidateId": candidate.candidateId,
        "runId": run.run_id,
        "parameters": _parameters(candidate),
        "selection": run["analysis_cell_selection"].to_dict(),
        "partitions": partitions,
        "silhouetteSampleCells": int(sample_size),
        "limitations": list(dict.fromkeys(limitations)),
    }
    result.update(representation_diagnostics(store, run, prepared, config.randomSeed))
    return result


def compare_candidates(
    store: Any,
    parent_run: Any,
    run: Any,
    parent_candidate: Candidate,
    candidate: Candidate,
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    """Compare full-cohort partitions only at the same registered resolution."""
    from sklearn.metrics import adjusted_rand_score

    if parent_run.status != "completed" or run.status != "completed":
        raise AnalysisInputError("Candidate comparisons require completed pipelines")
    if parent_run["analysis_cell_selection"] != run["analysis_cell_selection"]:
        raise AnalysisInputError(
            "Candidate comparisons must share the exact frozen cohort"
        )
    if not np.array_equal(parent_run.cells.fetch("ids"), run.cells.fetch("ids")):
        raise AnalysisInputError("Candidate comparison cell ordering differs")

    def overlaps(source: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
        result = []
        # Label identifiers have no meaning across partitions; use maximum cell overlap.
        for group in np.unique(source):
            counts = pd.Series(target[source == group]).value_counts().sort_index()
            best = str(counts.idxmax())
            count = int(counts[best])
            size = int((source == group).sum())
            result.append(
                {
                    "clusterId": str(group),
                    "matchedClusterId": best,
                    "sourceCells": size,
                    "intersectionCells": count,
                    "fraction": count / size,
                }
            )
        return result

    result = []
    for compared_resolution in config.resolutions:
        key = f"leiden_{compared_resolution}"
        if key not in parent_run or key not in run:
            raise AnalysisInputError(
                "Candidate comparison is missing a matched resolution"
            )
        left = np.asarray(parent_run.cells.fetch(key)).astype(str)
        right = np.asarray(run.cells.fetch(key)).astype(str)
        if left.shape != right.shape:
            raise AnalysisInputError("Candidate partition lengths differ")
        result.append(
            {
                "parentCandidateId": parent_candidate.candidateId,
                "candidateId": candidate.candidateId,
                "parentRunId": parent_run.run_id,
                "runId": run.run_id,
                "selection": run["analysis_cell_selection"].to_dict(),
                "resolution": compared_resolution,
                "adjustedRandIndex": float(adjusted_rand_score(left, right)),
                "cellCount": len(left),
                "parentToCandidate": overlaps(left, right),
                "candidateToParent": overlaps(right, left),
            }
        )
    return result


def _measure(name: str, function: Any, limitations: list[str]) -> float | None:
    try:
        result = _finite(function())
    except (ValueError, KeyError, RuntimeError, TypeError) as error:
        limitations.append(f"{name} unavailable: {type(error).__name__}: {error}")
        return None
    if result is None:
        limitations.append(f"{name} is non-finite or unavailable.")
    return result


def _doublet_concentration(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Describe concentration of the cohort's top-decile advisory scores."""
    values = np.asarray(scores, dtype=float)
    if (
        values.shape != labels.shape
        or not np.isfinite(values).all()
        or values.size == 0
    ):
        return None
    high = values >= np.quantile(values, 0.9)
    fraction = float(high.mean())
    if fraction == 0:
        return None
    return float(
        max(high[labels == group].mean() / fraction for group in np.unique(labels))
    )


def _diagnostic_indices(size: int, seed: int) -> np.ndarray:
    return np.sort(
        np.random.default_rng(seed).choice(
            size, size=min(size, _DIAGNOSTIC_CELLS), replace=False
        )
    )


def _diagnostic_graph(
    store: Any, run: Any, candidate: Candidate, rows: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a bounded KNN diagnostic graph on frozen coordinate rows."""
    from scarf.neighbors import fix_knn_query, instantiate_knn_index
    from scarf.neighbors.graph import build_connectivity_arrays

    if len(rows) > _DIAGNOSTIC_CELLS:
        raise AnalysisInputError("Diagnostic graph is limited to 10000 coordinate rows")
    source = store.load_artifact(
        run["harmony"] if candidate.useHarmony else run["pca"]
    )["data"]
    coordinates = np.asarray(
        source.get_orthogonal_selection((rows, slice(None))), dtype=np.float64
    )
    if not np.isfinite(coordinates).all():
        raise AnalysisInputError("Diagnostic coordinates contain non-finite values")
    k = min(candidate.neighborsK, len(rows) - 1)
    if k < 3:
        raise AnalysisInputError(
            "Correction diagnostics require at least three neighbors"
        )
    index = instantiate_knn_index(
        "l2", coordinates.shape[1], len(rows), 100, 16, seed, max(50, k + 1), 1
    )
    index.add_items(coordinates, np.arange(len(rows)), num_threads=1)
    indices, distances = index.knn_query(coordinates, k=k + 1, num_threads=1)
    indices, distances, _ = fix_knn_query(indices, distances, np.arange(len(rows)))
    distances = np.sqrt(np.maximum(np.asarray(distances, dtype=float), 0))
    edges, weights = build_connectivity_arrays(
        indices, distances, local_connectivity=1.0, bandwidth=1.5
    )
    return distances, indices, edges, weights


def finalist_evidence(
    store: Any,
    run: Any,
    candidate: Candidate,
    prepared: dict[str, Any],
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Summarize each observed cluster and the matched correction safeguards."""
    if run.status != "completed" or "markers" not in run:
        raise AnalysisInputError(
            "Finalist evidence requires a completed run with markers"
        )
    labels = np.asarray(run.cells.fetch("clusters")).astype(str)
    counts = pd.Series(labels).value_counts().sort_index()
    limitations: list[str] = []
    rows = _diagnostic_indices(len(labels), config.randomSeed)
    sample_labels = labels[rows]
    sample_ids = [str(value) for value in run.cells.fetch("ids")[rows]]
    diagnostic_scope = {
        "method": "sampledCoordinatesKnn",
        "seed": config.randomSeed,
        "populationCells": len(labels),
        "sampleCells": len(rows),
        "cellIdsSha256": hashlib.sha256(
            json.dumps(sample_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "neighborsK": min(candidate.neighborsK, len(rows) - 1),
    }
    if len(rows) < len(labels):
        limitations.append(
            "Correction and doublet diagnostics estimate the cohort on at most 10000 deterministic coordinate rows; rare populations may be absent from that sample. Sample composition uses the full retained cohort."
        )
    clusters: list[dict[str, Any]] = []
    composition = finalist_composition(run, prepared, labels)
    specificities = []
    supported = 0
    for group, count in counts.items():
        # One group table at a time avoids loading all feature-by-cluster
        # results together. Empty supported-marker lists remain explicit.
        frame = store.get_markers(
            run["markers"], group_id=str(group), min_score=-1, min_frac_exp=-1
        )
        qualifying = frame[(frame["score"] >= 0.25) & (frame["frac_exp"] >= 0.2)]
        supported += int(not qualifying.empty)
        if not qualifying.empty:
            specificities.append(float(qualifying.head(10)["score"].median()))
        markers = []
        for row in frame.head(12).to_dict(orient="records"):
            markers.append(
                {
                    "gene": str(row["feature_name"]),
                    "score": _finite(row["score"]),
                    "fracExp": _finite(row["frac_exp"]),
                    "fracExpDelta": _finite(row["frac_exp"] - row["frac_exp_rest"]),
                    **(
                        {"fracExpRest": _finite(row["frac_exp_rest"])}
                        if "frac_exp_rest" in row
                        else {}
                    ),
                }
            )
        weak = []
        for row in (
            frame.loc[(frame["score"] < 0.25) | (frame["frac_exp"] < 0.2)]
            .head(6)
            .to_dict(orient="records")
        ):
            weak.append(
                {
                    "gene": str(row["feature_name"]),
                    "score": _finite(row["score"]),
                    "fracExp": _finite(row["frac_exp"]),
                    "fracExpRest": _finite(row["frac_exp_rest"]),
                    "fracExpDelta": _finite(row["frac_exp"] - row["frac_exp_rest"]),
                }
            )
        clusters.append(
            {
                "clusterId": str(group),
                "count": int(count),
                "markers": markers,
                "weakMarkers": weak,
                "qualifyingMarkerCount": len(qualifying),
                **composition[str(group)],
            }
        )
    metrics: dict[str, Any] = {
        "mixing": {},
        "protection": {},
        "markerSupportFraction": supported / len(clusters) if clusters else None,
        "markerSpecificityMedian": float(np.median(specificities))
        if specificities
        else None,
        "doubletHighScoreConcentration": None,
        "crossUnitSupport": None,
    }
    diagnostic_graph = None
    if prepared["technicalBatchColumns"] or prepared["protectedColumns"]:
        try:
            diagnostic_graph = _diagnostic_graph(
                store, run, candidate, rows, config.randomSeed
            )
        except (ValueError, RuntimeError, TypeError) as error:
            limitations.append(
                f"Bounded correction diagnostics unavailable: {type(error).__name__}: {error}"
            )
    from scarf.metrics import (
        clisi_knn,
        compute_lisi,
        graph_connectivity,
        lisi_batch_mixing_score,
    )

    def labels_for(column: str) -> np.ndarray:
        full = run.cells.to_pandas_dataframe([column])[column]
        sample = full.iloc[rows]
        if full.isna().any() or sample.nunique() != full.nunique():
            raise AnalysisInputError(
                "Diagnostic sample lacks complete coverage of the frozen metadata levels"
            )
        return np.asarray(sample.to_numpy())

    def mixing(column: str) -> float | None:
        if diagnostic_graph is None:
            return None
        distances, indices, _, _ = diagnostic_graph
        values = labels_for(column)
        lisi = compute_lisi(
            distances, indices, pd.DataFrame({column: values}), [column]
        )[:, 0]
        return lisi_batch_mixing_score(lisi, values)

    def protection(column: str, *, connectivity: bool) -> float | None:
        if diagnostic_graph is None:
            return None
        distances, indices, edges, weights = diagnostic_graph
        values = labels_for(column)
        return (
            graph_connectivity(edges, values, weights=weights)
            if connectivity
            else clisi_knn(distances, indices, values, scale=True)
        )

    for column in prepared["technicalBatchColumns"]:
        metrics["mixing"][column] = _measure(
            f"Batch mixing for {column}",
            lambda column=column: mixing(column),
            limitations,
        )
    for column in prepared["protectedColumns"]:
        metrics["protection"][column] = {
            "cLISI": _measure(
                f"Biological preservation for {column}",
                lambda column=column: protection(column, connectivity=False),
                limitations,
            ),
            "graphConnectivity": _measure(
                f"Graph connectivity for {column}",
                lambda column=column: protection(column, connectivity=True),
                limitations,
            ),
        }
    if "doublet_score" in run.cells.columns:
        values = np.asarray(run.cells.fetch("doublet_score"), dtype=float)[rows]
        metrics["doubletHighScoreConcentration"] = _doublet_concentration(
            values, sample_labels
        )
        for cluster in clusters:
            subset = values[sample_labels == cluster["clusterId"]]
            cluster["doubletScoreMedian"] = (
                _finite(np.median(subset)) if subset.size else None
            )
    else:
        limitations.append(
            "Doublet scoring was not requested; doublets remain an unresolved annotation limitation."
        )
    sample_column = prepared.get("sampleColumn")
    if sample_column:
        series = run.cells.to_pandas_dataframe([sample_column])[sample_column]
        if len(series) != len(labels):
            raise AnalysisInputError(
                "Frozen sample labels do not align with finalist cells"
            )
        if series.isna().any() or series.astype(str).str.strip().eq("").any():
            limitations.append(
                "Cross-unit support is unavailable because sample labels are missing."
            )
        elif series.nunique() < 2:
            limitations.append("Cross-unit support is unavailable for a single sample.")
        else:
            units = series.astype(str).to_numpy()
            metrics["crossUnitSupport"] = float(
                np.mean(
                    [
                        len(np.unique(units[labels == group])) >= 2
                        for group in counts.index
                    ]
                )
            )
            for cluster in clusters:
                cluster["sampleCount"] = int(
                    len(np.unique(units[labels == cluster["clusterId"]]))
                )
    selected_key = next(
        (
            key
            for key in run
            if key.startswith("leiden_") and run[key] == run["clusters"]
        ),
        None,
    )
    resolution = float(selected_key.removeprefix("leiden_")) if selected_key else None
    return {
        "candidateId": candidate.candidateId,
        "runId": run.run_id,
        "parameters": {**_parameters(candidate), "resolution": resolution},
        "selection": run["analysis_cell_selection"].to_dict(),
        "features": run["highly_variable_features"].to_dict(),
        "clusters": clusters,
        "diagnosticRoles": prepared.get("resolvedRoles", []),
        "compositionScope": "fullRetainedCohort",
        "metrics": metrics,
        "requiredSampleSupport": sample_column is not None,
        "diagnosticScope": diagnostic_scope,
        "limitations": limitations,
    }


def validate_harmony(
    native_evidence: dict[str, Any], corrected_evidence: dict[str, Any]
) -> list[str]:
    """Reject unmatched or incomplete correction evidence and measured harm."""
    reasons = []
    left = native_evidence.get("parameters", {})
    right = corrected_evidence.get("parameters", {})
    if left.get("useHarmony") is not False or right.get("useHarmony") is not True:
        reasons.append("Correction requires one native and one Harmony finalist.")
    if {key: value for key, value in left.items() if key != "useHarmony"} != {
        key: value for key, value in right.items() if key != "useHarmony"
    }:
        reasons.append("Correction finalists differ in settings or resolution.")
    for field in ("selection", "features"):
        if not native_evidence.get(field) or native_evidence.get(
            field
        ) != corrected_evidence.get(field):
            reasons.append(
                f"Correction finalists do not share the exact {field} artifact."
            )
    if not native_evidence.get("diagnosticScope") or native_evidence.get(
        "diagnosticScope"
    ) != corrected_evidence.get("diagnosticScope"):
        reasons.append(
            "Correction diagnostics do not share the same frozen sample and method."
        )
    native = native_evidence.get("metrics", {})
    corrected = corrected_evidence.get("metrics", {})
    before_mixing = native.get("mixing", {})
    after_mixing = corrected.get("mixing", {})
    gains = []
    if not before_mixing or set(before_mixing) != set(after_mixing):
        reasons.append("Matched technical batch mixing is missing.")
    for column in sorted(set(before_mixing) | set(after_mixing)):
        before, after = (
            _finite(before_mixing.get(column)),
            _finite(after_mixing.get(column)),
        )
        if before is None or after is None:
            reasons.append(f"Batch mixing evidence is unavailable for {column}.")
        else:
            gains.append(after - before)
            if after < before - 0.05:
                reasons.append(f"Batch mixing worsened by more than 0.05 for {column}.")
    if not gains or max(gains) <= 0.05:
        reasons.append("No batch mixing measure improved by more than 0.05.")
    before_protection = native.get("protection", {})
    after_protection = corrected.get("protection", {})
    if not before_protection or set(before_protection) != set(after_protection):
        reasons.append("Matched biological protection evidence is missing.")
    for column in sorted(set(before_protection) | set(after_protection)):
        for metric in ("cLISI", "graphConnectivity"):
            before = _finite(before_protection.get(column, {}).get(metric))
            after = _finite(after_protection.get(column, {}).get(metric))
            if before is None or after is None:
                reasons.append(
                    f"Biological protection {metric} is unavailable for {column}."
                )
            elif after < before - 0.05:
                reasons.append(
                    f"Biological protection {metric} worsened by more than 0.05 for {column}."
                )
    required = ["markerSupportFraction"]
    if native_evidence.get("requiredSampleSupport") or corrected_evidence.get(
        "requiredSampleSupport"
    ):
        required.append("crossUnitSupport")
    for metric in required:
        before, after = _finite(native.get(metric)), _finite(corrected.get(metric))
        if before is None or after is None:
            reasons.append(f"Matched {metric} evidence is unavailable.")
        elif after < before - 0.05:
            reasons.append(f"{metric} worsened by more than 0.05 after correction.")
    before = _finite(native.get("doubletHighScoreConcentration"))
    after = _finite(corrected.get("doubletHighScoreConcentration"))
    if before is None or after is None:
        reasons.append("Matched doubletHighScoreConcentration evidence is unavailable.")
    elif after > before + 0.05:
        reasons.append(
            "Doublet high-score concentration increased by more than 0.05 after correction."
        )
    before = _finite(native.get("markerSpecificityMedian"))
    after = _finite(corrected.get("markerSpecificityMedian"))
    if before is not None or after is not None:
        if before is None or after is None:
            reasons.append("Matched marker specificity evidence is unavailable.")
        elif after < before - 0.05:
            reasons.append(
                "Marker specificity worsened by more than 0.05 after correction."
            )
    return reasons
