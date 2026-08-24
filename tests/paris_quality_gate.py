import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import csr_matrix, triu
from sklearn.metrics import adjusted_rand_score

from scarf.clustering._paris_core import ParisHierarchy
from scarf.clustering._paris_modularity import modularity_split_gains
from scarf.clustering.paris import (
    fit_paris_hierarchy,
    hierarchy_to_dendrogram,
    straight_cut,
)
from scarf.clustering.paris_multiscale import (
    PlateauForest,
    adaptive_cut,
    collapse_equal_height_plateaus,
)

DEFAULT_SEED_COUNT = 8
MIN_RARE_CLUSTER_F1 = 0.9
MAX_ADAPTIVE_CLUSTERS = 5
MIN_ACCEPTED_SEED_FRACTION = 0.8
CPM_RESOLUTIONS = tuple(np.geomspace(0.002, 0.2, 12))
# The guarded cut must recover structure as well as the oracle horizontal cut
# that scans every cluster count, otherwise it has silently degraded.
ORACLE_ARI_TOLERANCE = 0.05
PAIR_F1_ORACLE_TOLERANCE = 1e-12
MAX_DINKELBACH_ITERATIONS = 100
MAX_EXHAUSTIVE_ORACLE_LEAVES = 20
SUPPORTED_ALPHA_BETA_EXPONENTS = tuple(range(-12, 13))
SUPPORTED_ALPHA_BETA_RATIOS = tuple(
    2.0**exponent for exponent in SUPPORTED_ALPHA_BETA_EXPONENTS
)


@dataclass(frozen=True, slots=True)
class PairMetrics:
    true_positive_pairs: int
    false_positive_pairs: int
    false_negative_pairs: int
    true_negative_pairs: int
    precision: float
    recall: float
    f1: float

    @property
    def predicted_positive_pairs(self) -> int:
        return self.true_positive_pairs + self.false_positive_pairs

    @property
    def truth_positive_pairs(self) -> int:
        return self.true_positive_pairs + self.false_negative_pairs

    @property
    def total_pairs(self) -> int:
        return (
            self.true_positive_pairs
            + self.false_positive_pairs
            + self.false_negative_pairs
            + self.true_negative_pairs
        )


@dataclass(frozen=True, slots=True)
class PairCut:
    selected_nodes: tuple[int, ...]
    metrics: PairMetrics

    @property
    def n_clusters(self) -> int:
        return len(self.selected_nodes)


@dataclass(frozen=True, slots=True)
class PairF1OracleResult:
    cut: PairCut
    iterations: int
    tolerance: float
    residual: float


@dataclass(frozen=True, slots=True)
class PairErrorOracleResult:
    alpha: int | float
    beta: int | float
    weighted_error: int | float
    cut: PairCut

    @property
    def alpha_beta_ratio(self) -> float:
        return float(self.alpha) / float(self.beta)


@dataclass(frozen=True, slots=True)
class HierarchyHeadroomEvaluation:
    min_cluster_size: int
    baseline: PairMetrics
    maximum_pair_f1: PairF1OracleResult
    supported_cuts: tuple[PairErrorOracleResult, ...]
    supported_frontier: tuple[PairErrorOracleResult, ...]
    pair_f1_headroom: float
    has_measurable_headroom: bool


@dataclass(frozen=True, slots=True)
class _EventPairStatistics:
    same_pairs: tuple[int, ...]
    cross_pairs: tuple[int, ...]
    all_pairs: tuple[int, ...]
    truth_pairs: int
    total_pairs: int
    evaluated_items: int


def _choose_two(size: int) -> int:
    return size * (size - 1) // 2


def _label_array(values: np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if labels.size == 0:
        raise ValueError(f"{name} must not be empty")
    return labels


def _normalized_label_strings(values: np.ndarray) -> np.ndarray:
    normalized = np.empty(values.size, dtype=object)
    for index, raw_value in enumerate(values):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        normalized[index] = str(value)
    return normalized.astype(str)


def _encoded_labels(values: np.ndarray) -> np.ndarray:
    labels = _label_array(values, "labels")
    normalized = (
        _normalized_label_strings(labels)
        if labels.dtype.kind == "O"
        else labels.astype(str)
    )
    _unique, codes = np.unique(normalized, return_inverse=True)
    return codes.astype(np.int32, copy=False)


def _encoded_truth(values: np.ndarray) -> tuple[np.ndarray, int]:
    truth = _label_array(values, "truth")
    valid = np.ones(truth.size, dtype=bool)
    normalized = np.empty(truth.size, dtype=object)
    for index, raw_value in enumerate(truth):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        missing = value is None
        if not missing:
            try:
                missing = bool(value != value)
            except (TypeError, ValueError):
                missing = False
        label = "" if missing else str(value).strip()
        if missing or not label or label.casefold() == "unknown":
            valid[index] = False
            normalized[index] = ""
        else:
            normalized[index] = label

    codes = np.full(truth.size, -1, dtype=np.int32)
    if np.any(valid):
        _unique, valid_codes = np.unique(
            normalized[valid].astype(str),
            return_inverse=True,
        )
        codes[valid] = valid_codes.astype(np.int32, copy=False)
    return codes, int(np.count_nonzero(valid))


def _pair_metrics_from_counts(
    *,
    n_items: int,
    true_positive_pairs: int,
    predicted_positive_pairs: int,
    truth_positive_pairs: int,
) -> PairMetrics:
    false_positive_pairs = predicted_positive_pairs - true_positive_pairs
    false_negative_pairs = truth_positive_pairs - true_positive_pairs
    true_negative_pairs = (
        _choose_two(n_items)
        - true_positive_pairs
        - false_positive_pairs
        - false_negative_pairs
    )
    if (
        min(
            true_positive_pairs,
            false_positive_pairs,
            false_negative_pairs,
            true_negative_pairs,
        )
        < 0
    ):
        raise ValueError("pair counts are inconsistent")
    precision = (
        true_positive_pairs / predicted_positive_pairs
        if predicted_positive_pairs
        else 0.0
    )
    recall = true_positive_pairs / truth_positive_pairs if truth_positive_pairs else 0.0
    denominator = predicted_positive_pairs + truth_positive_pairs
    f1 = 2 * true_positive_pairs / denominator if denominator else 0.0
    return PairMetrics(
        true_positive_pairs=true_positive_pairs,
        false_positive_pairs=false_positive_pairs,
        false_negative_pairs=false_negative_pairs,
        true_negative_pairs=true_negative_pairs,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def exact_pair_metrics(labels: np.ndarray, truth: np.ndarray) -> PairMetrics:
    predicted = _label_array(labels, "labels")
    expected = _label_array(truth, "truth")
    if predicted.size != expected.size:
        raise ValueError("labels and truth must have the same length")

    truth_codes, evaluated_items = _encoded_truth(expected)
    valid = truth_codes >= 0
    if evaluated_items == 0:
        return _pair_metrics_from_counts(
            n_items=0,
            true_positive_pairs=0,
            predicted_positive_pairs=0,
            truth_positive_pairs=0,
        )
    predicted_codes = _encoded_labels(predicted[valid])
    valid_truth_codes = truth_codes[valid]
    predicted_counts = np.bincount(predicted_codes).astype(np.int64, copy=False)
    truth_counts = np.bincount(valid_truth_codes).astype(np.int64, copy=False)
    n_predicted = max(1, predicted_counts.size)
    joint_keys = valid_truth_codes.astype(
        np.int64
    ) * n_predicted + predicted_codes.astype(np.int64)
    _joint_values, joint_counts = np.unique(joint_keys, return_counts=True)
    true_positive_pairs = int(
        np.sum(joint_counts * (joint_counts - 1) // 2, dtype=np.int64)
    )
    predicted_positive_pairs = int(
        np.sum(predicted_counts * (predicted_counts - 1) // 2, dtype=np.int64)
    )
    truth_positive_pairs = int(
        np.sum(truth_counts * (truth_counts - 1) // 2, dtype=np.int64)
    )
    return _pair_metrics_from_counts(
        n_items=evaluated_items,
        true_positive_pairs=true_positive_pairs,
        predicted_positive_pairs=predicted_positive_pairs,
        truth_positive_pairs=truth_positive_pairs,
    )


def _validate_min_cluster_size(min_cluster_size: int) -> None:
    if isinstance(min_cluster_size, bool) or not isinstance(min_cluster_size, int):
        raise TypeError("min_cluster_size must be an integer")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")


def _event_pair_statistics(
    forest: PlateauForest,
    truth: np.ndarray,
) -> _EventPairStatistics:
    expected = _label_array(truth, "truth")
    if expected.size != forest.n_leaves:
        raise ValueError("truth and plateau forest have different leaf counts")
    truth_codes, evaluated_items = _encoded_truth(expected)
    n_classes = int(truth_codes.max()) + 1 if evaluated_items else 0
    dense_bytes = (
        int(forest.representatives.size) * n_classes * np.dtype(np.int64).itemsize
    )
    if dense_bytes < 64 * 1024**2:
        return _dense_event_pair_statistics(
            forest,
            truth_codes,
            evaluated_items,
            n_classes,
        )
    return _sparse_event_pair_statistics(
        forest,
        truth_codes,
        evaluated_items,
    )


def _statistics_from_event_counts(
    event_counts: np.ndarray,
    evaluated_items: int,
) -> _EventPairStatistics:
    event_sizes = event_counts.sum(axis=1, dtype=np.int64)
    same_pairs_array = np.sum(
        event_counts * (event_counts - 1) // 2,
        axis=1,
        dtype=np.int64,
    )
    all_pairs_array = event_sizes * (event_sizes - 1) // 2
    truth_counts = event_counts[-1] if event_counts.shape[0] else np.empty(0)
    return _EventPairStatistics(
        same_pairs=tuple(int(value) for value in same_pairs_array),
        cross_pairs=tuple(int(value) for value in all_pairs_array - same_pairs_array),
        all_pairs=tuple(int(value) for value in all_pairs_array),
        truth_pairs=int(np.sum(truth_counts * (truth_counts - 1) // 2, dtype=np.int64)),
        total_pairs=_choose_two(evaluated_items),
        evaluated_items=evaluated_items,
    )


def _dense_event_pair_statistics(
    forest: PlateauForest,
    truth_codes: np.ndarray,
    evaluated_items: int,
    n_classes: int,
) -> _EventPairStatistics:
    counts = np.zeros(
        (forest.representatives.size, n_classes),
        dtype=np.int64,
    )
    for event in range(forest.representatives.size):
        for offset in range(
            int(forest.child_offsets[event]),
            int(forest.child_offsets[event + 1]),
        ):
            child_ref = int(forest.child_refs[offset])
            if child_ref < 0:
                leaf = -child_ref - 1
                if leaf < 0 or leaf >= forest.n_leaves:
                    raise ValueError(
                        "plateau forest contains an invalid leaf reference"
                    )
                code = int(truth_codes[leaf])
                if code >= 0:
                    counts[event, code] += 1
            else:
                if child_ref >= event:
                    raise ValueError("plateau forest events must be in postorder")
                counts[event] += counts[child_ref]
    root_counts = np.zeros(n_classes, dtype=np.int64)
    for root_ref_value in forest.component_roots:
        root_ref = int(root_ref_value)
        if root_ref >= 0:
            root_counts += counts[root_ref]
        else:
            code = int(truth_codes[-root_ref - 1])
            if code >= 0:
                root_counts[code] += 1
    statistics = _statistics_from_event_counts(counts, evaluated_items)
    return _EventPairStatistics(
        same_pairs=statistics.same_pairs,
        cross_pairs=statistics.cross_pairs,
        all_pairs=statistics.all_pairs,
        truth_pairs=int(np.sum(root_counts * (root_counts - 1) // 2, dtype=np.int64)),
        total_pairs=statistics.total_pairs,
        evaluated_items=evaluated_items,
    )


def _merge_sparse_counts(
    left_ids: np.ndarray,
    left_counts: np.ndarray,
    right_ids: np.ndarray,
    right_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    merged_ids = np.empty(left_ids.size + right_ids.size, dtype=np.int32)
    merged_counts = np.empty(left_ids.size + right_ids.size, dtype=np.int64)
    left = 0
    right = 0
    output = 0
    while left < left_ids.size and right < right_ids.size:
        left_id = int(left_ids[left])
        right_id = int(right_ids[right])
        if left_id < right_id:
            merged_ids[output] = left_id
            merged_counts[output] = left_counts[left]
            left += 1
        elif right_id < left_id:
            merged_ids[output] = right_id
            merged_counts[output] = right_counts[right]
            right += 1
        else:
            merged_ids[output] = left_id
            merged_counts[output] = left_counts[left] + right_counts[right]
            left += 1
            right += 1
        output += 1
    if left < left_ids.size:
        count = left_ids.size - left
        merged_ids[output : output + count] = left_ids[left:]
        merged_counts[output : output + count] = left_counts[left:]
        output += count
    if right < right_ids.size:
        count = right_ids.size - right
        merged_ids[output : output + count] = right_ids[right:]
        merged_counts[output : output + count] = right_counts[right:]
        output += count
    return merged_ids[:output], merged_counts[:output]


def _sparse_event_pair_statistics(
    forest: PlateauForest,
    truth_codes: np.ndarray,
    evaluated_items: int,
) -> _EventPairStatistics:
    event_ids: list[np.ndarray | None] = []
    event_counts: list[np.ndarray | None] = []
    same_pairs: list[int] = []
    cross_pairs: list[int] = []
    all_pairs: list[int] = []

    for event in range(forest.representatives.size):
        child_events: list[int] = []
        leaf_codes: list[int] = []
        for offset in range(
            int(forest.child_offsets[event]),
            int(forest.child_offsets[event + 1]),
        ):
            child_ref = int(forest.child_refs[offset])
            if child_ref < 0:
                leaf = -child_ref - 1
                if leaf < 0 or leaf >= forest.n_leaves:
                    raise ValueError(
                        "plateau forest contains an invalid leaf reference"
                    )
                code = int(truth_codes[leaf])
                if code >= 0:
                    leaf_codes.append(code)
                continue
            if child_ref >= event:
                raise ValueError("plateau forest events must be in postorder")
            child_events.append(child_ref)

        if child_events:
            largest_child = max(
                child_events,
                key=lambda child: len(
                    event_ids[child] if event_ids[child] is not None else ()
                ),
            )
            ids = event_ids[largest_child]
            counts = event_counts[largest_child]
            if ids is None or counts is None:
                raise ValueError("plateau event has more than one parent")
            event_ids[largest_child] = None
            event_counts[largest_child] = None
            for child in child_events:
                if child == largest_child:
                    continue
                child_ids = event_ids[child]
                child_counts = event_counts[child]
                if child_ids is None or child_counts is None:
                    raise ValueError("plateau event has more than one parent")
                ids, counts = _merge_sparse_counts(
                    ids,
                    counts,
                    child_ids,
                    child_counts,
                )
                event_ids[child] = None
                event_counts[child] = None
        else:
            ids = np.empty(0, dtype=np.int32)
            counts = np.empty(0, dtype=np.int64)
        if leaf_codes:
            leaf_ids, leaf_counts = np.unique(
                np.asarray(leaf_codes, dtype=np.int32),
                return_counts=True,
            )
            ids, counts = _merge_sparse_counts(
                ids,
                counts,
                leaf_ids,
                leaf_counts.astype(np.int64, copy=False),
            )
        event_ids.append(ids)
        event_counts.append(counts)
        event_same_pairs = int(np.sum(counts * (counts - 1) // 2, dtype=np.int64))
        size = int(counts.sum(dtype=np.int64))
        event_all_pairs = _choose_two(size)
        same_pairs.append(event_same_pairs)
        cross_pairs.append(event_all_pairs - event_same_pairs)
        all_pairs.append(event_all_pairs)

    valid_truth = truth_codes[truth_codes >= 0]
    truth_counts = np.bincount(valid_truth).astype(np.int64, copy=False)
    return _EventPairStatistics(
        same_pairs=tuple(same_pairs),
        cross_pairs=tuple(cross_pairs),
        all_pairs=tuple(all_pairs),
        truth_pairs=int(np.sum(truth_counts * (truth_counts - 1) // 2, dtype=np.int64)),
        total_pairs=_choose_two(evaluated_items),
        evaluated_items=evaluated_items,
    )


def _selected_ref_key(forest: PlateauForest, selected_ref: int) -> int:
    return (
        -selected_ref - 1 if selected_ref < 0 else int(forest.min_leaves[selected_ref])
    )


def _selected_nodes(
    forest: PlateauForest,
    selected_refs: tuple[int, ...],
) -> tuple[int, ...]:
    ordered_refs = sorted(
        selected_refs,
        key=lambda selected_ref: _selected_ref_key(forest, selected_ref),
    )
    return tuple(
        -selected_ref - 1
        if selected_ref < 0
        else int(forest.representatives[selected_ref])
        for selected_ref in ordered_refs
    )


def _pair_cut_from_refs(
    forest: PlateauForest,
    statistics: _EventPairStatistics,
    selected_refs: tuple[int, ...],
) -> PairCut:
    true_positive_pairs = sum(
        statistics.same_pairs[selected_ref]
        for selected_ref in selected_refs
        if selected_ref >= 0
    )
    predicted_positive_pairs = sum(
        statistics.all_pairs[selected_ref]
        for selected_ref in selected_refs
        if selected_ref >= 0
    )
    return PairCut(
        selected_nodes=_selected_nodes(forest, selected_refs),
        metrics=_pair_metrics_from_counts(
            n_items=statistics.evaluated_items,
            true_positive_pairs=true_positive_pairs,
            predicted_positive_pairs=predicted_positive_pairs,
            truth_positive_pairs=statistics.truth_pairs,
        ),
    )


def _optimize_additive_antichain(
    forest: PlateauForest,
    node_values: tuple[int | float, ...],
    min_cluster_size: int,
    *,
    maximize: bool,
) -> tuple[int | float, tuple[int, ...]]:
    _validate_min_cluster_size(min_cluster_size)
    n_events = forest.representatives.size
    if len(node_values) != n_events:
        raise ValueError("node_values must have one value per plateau event")

    best_values: list[int | float | None] = [None] * n_events
    split_events = [False] * n_events
    for event in range(n_events):
        if int(forest.sizes[event]) < min_cluster_size:
            continue
        keep_value = node_values[event]
        split_value: int | float = 0
        split_feasible = True
        for offset in range(
            int(forest.child_offsets[event]),
            int(forest.child_offsets[event + 1]),
        ):
            child_ref = int(forest.child_refs[offset])
            if child_ref < 0 or best_values[child_ref] is None:
                split_feasible = False
                break
            split_value += best_values[child_ref]

        prefer_split = split_feasible and (
            split_value > keep_value if maximize else split_value < keep_value
        )
        if prefer_split:
            best_values[event] = split_value
            split_events[event] = True
        else:
            best_values[event] = keep_value

    total_value: int | float = 0
    selected_refs: list[int] = []
    stack: list[int] = []
    for root_ref_value in forest.component_roots:
        root_ref = int(root_ref_value)
        if root_ref < 0:
            selected_refs.append(root_ref)
            continue
        if root_ref >= n_events:
            raise ValueError("plateau forest contains an invalid component root")
        if best_values[root_ref] is None:
            total_value += node_values[root_ref]
            selected_refs.append(root_ref)
            continue
        total_value += best_values[root_ref]
        stack.append(root_ref)
        while stack:
            event = stack.pop()
            if not split_events[event]:
                selected_refs.append(event)
                continue
            for offset in range(
                int(forest.child_offsets[event + 1]) - 1,
                int(forest.child_offsets[event]) - 1,
                -1,
            ):
                child_ref = int(forest.child_refs[offset])
                if child_ref < 0:
                    raise RuntimeError("a feasible oracle split contains a leaf")
                stack.append(child_ref)

    selected_refs.sort(key=lambda selected_ref: _selected_ref_key(forest, selected_ref))
    return total_value, tuple(selected_refs)


def _maximum_pair_f1_from_statistics(
    forest: PlateauForest,
    statistics: _EventPairStatistics,
    min_cluster_size: int,
    *,
    tolerance: float,
    max_iterations: int,
    initial_numerator: int = 0,
    initial_denominator: int = 1,
) -> PairF1OracleResult:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite number")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if initial_numerator < 0 or initial_denominator < 1:
        raise ValueError("initial pair-F1 quotient is invalid")

    quotient_numerator = initial_numerator
    quotient_denominator = initial_denominator
    for iteration in range(1, max_iterations + 1):
        node_values = tuple(
            2 * same_pairs * quotient_denominator - quotient_numerator * all_pairs
            for same_pairs, all_pairs in zip(
                statistics.same_pairs,
                statistics.all_pairs,
                strict=True,
            )
        )
        _objective, selected_refs = _optimize_additive_antichain(
            forest,
            node_values,
            min_cluster_size,
            maximize=True,
        )
        cut = _pair_cut_from_refs(forest, statistics, selected_refs)
        numerator = 2 * cut.metrics.true_positive_pairs
        denominator = cut.metrics.predicted_positive_pairs + statistics.truth_pairs
        if denominator == 0:
            return PairF1OracleResult(
                cut=cut,
                iterations=iteration,
                tolerance=tolerance,
                residual=0.0,
            )
        residual_numerator = (
            numerator * quotient_denominator - quotient_numerator * denominator
        )
        residual = residual_numerator / quotient_denominator
        if abs(residual) <= tolerance * denominator:
            return PairF1OracleResult(
                cut=cut,
                iterations=iteration,
                tolerance=tolerance,
                residual=max(0.0, residual),
            )
        divisor = math.gcd(numerator, denominator)
        quotient_numerator = numerator // divisor
        quotient_denominator = denominator // divisor
    raise RuntimeError("pair-F1 Dinkelbach iterations did not converge")


def maximum_pair_f1_cut(
    forest: PlateauForest,
    truth: np.ndarray,
    min_cluster_size: int = 2,
    *,
    tolerance: float = PAIR_F1_ORACLE_TOLERANCE,
    max_iterations: int = MAX_DINKELBACH_ITERATIONS,
) -> PairF1OracleResult:
    statistics = _event_pair_statistics(forest, truth)
    return _maximum_pair_f1_from_statistics(
        forest,
        statistics,
        min_cluster_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _minimum_pair_error_from_statistics(
    forest: PlateauForest,
    statistics: _EventPairStatistics,
    *,
    alpha: int | float,
    beta: int | float,
    min_cluster_size: int,
) -> PairErrorOracleResult:
    for name, value in (("alpha", alpha), ("beta", beta)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    node_values = tuple(
        alpha * cross_pairs - beta * same_pairs
        for cross_pairs, same_pairs in zip(
            statistics.cross_pairs,
            statistics.same_pairs,
            strict=True,
        )
    )
    _objective, selected_refs = _optimize_additive_antichain(
        forest,
        node_values,
        min_cluster_size,
        maximize=False,
    )
    cut = _pair_cut_from_refs(forest, statistics, selected_refs)
    weighted_error = (
        alpha * cut.metrics.false_positive_pairs
        + beta * cut.metrics.false_negative_pairs
    )
    return PairErrorOracleResult(
        alpha=alpha,
        beta=beta,
        weighted_error=weighted_error,
        cut=cut,
    )


def minimum_pair_error_cut(
    forest: PlateauForest,
    truth: np.ndarray,
    *,
    alpha: int | float,
    beta: int | float,
    min_cluster_size: int = 2,
) -> PairErrorOracleResult:
    statistics = _event_pair_statistics(forest, truth)
    return _minimum_pair_error_from_statistics(
        forest,
        statistics,
        alpha=alpha,
        beta=beta,
        min_cluster_size=min_cluster_size,
    )


def _supported_pair_cuts_from_statistics(
    forest: PlateauForest,
    statistics: _EventPairStatistics,
    min_cluster_size: int,
) -> tuple[PairErrorOracleResult, ...]:
    cuts: list[PairErrorOracleResult] = []
    for exponent in SUPPORTED_ALPHA_BETA_EXPONENTS:
        if exponent < 0:
            alpha, beta = 1, 1 << -exponent
        else:
            alpha, beta = 1 << exponent, 1
        cuts.append(
            _minimum_pair_error_from_statistics(
                forest,
                statistics,
                alpha=alpha,
                beta=beta,
                min_cluster_size=min_cluster_size,
            )
        )
    return tuple(cuts)


def supported_pair_cuts(
    forest: PlateauForest,
    truth: np.ndarray,
    min_cluster_size: int = 2,
) -> tuple[PairErrorOracleResult, ...]:
    statistics = _event_pair_statistics(forest, truth)
    return _supported_pair_cuts_from_statistics(
        forest,
        statistics,
        min_cluster_size,
    )


def nondominated_supported_cuts(
    cuts: tuple[PairErrorOracleResult, ...],
) -> tuple[PairErrorOracleResult, ...]:
    frontier: list[PairErrorOracleResult] = []
    seen_metrics: set[tuple[int, int, int]] = set()
    for candidate in cuts:
        metrics = candidate.cut.metrics
        key = (
            metrics.true_positive_pairs,
            metrics.false_positive_pairs,
            metrics.false_negative_pairs,
        )
        if key in seen_metrics:
            continue
        seen_metrics.add(key)
        dominated = any(
            other.cut.metrics.precision >= metrics.precision
            and other.cut.metrics.recall >= metrics.recall
            and (
                other.cut.metrics.precision > metrics.precision
                or other.cut.metrics.recall > metrics.recall
            )
            for other in cuts
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)


def _enumerate_feasible_refs(
    forest: PlateauForest,
    min_cluster_size: int,
) -> tuple[tuple[int, ...], ...]:
    _validate_min_cluster_size(min_cluster_size)
    if forest.n_leaves > MAX_EXHAUSTIVE_ORACLE_LEAVES:
        raise ValueError(
            "exhaustive antichain enumeration is limited to "
            f"{MAX_EXHAUSTIVE_ORACLE_LEAVES} leaves"
        )

    event_leaf_masks: list[int] = []
    for event in range(forest.representatives.size):
        leaf_mask = 0
        for offset in range(
            int(forest.child_offsets[event]),
            int(forest.child_offsets[event + 1]),
        ):
            child_ref = int(forest.child_refs[offset])
            if child_ref < 0:
                leaf = -child_ref - 1
                if leaf < 0 or leaf >= forest.n_leaves:
                    raise ValueError(
                        "plateau forest contains an invalid leaf reference"
                    )
                child_mask = 1 << leaf
            else:
                if child_ref >= event:
                    raise ValueError("plateau forest events must be in postorder")
                child_mask = event_leaf_masks[child_ref]
            if leaf_mask & child_mask:
                raise ValueError("plateau event children overlap")
            leaf_mask |= child_mask
        if leaf_mask.bit_count() != int(forest.sizes[event]):
            raise ValueError("plateau event size does not match its children")
        event_leaf_masks.append(leaf_mask)

    root_events: set[int] = set()
    fixed_refs: list[int] = []
    fixed_mask = 0
    for root_ref_value in forest.component_roots:
        root_ref = int(root_ref_value)
        if root_ref < 0:
            leaf = -root_ref - 1
            if leaf < 0 or leaf >= forest.n_leaves:
                raise ValueError("plateau forest contains an invalid component root")
            leaf_mask = 1 << leaf
            if fixed_mask & leaf_mask:
                raise ValueError("plateau forest component roots overlap")
            fixed_mask |= leaf_mask
            fixed_refs.append(root_ref)
        else:
            if root_ref >= forest.representatives.size:
                raise ValueError("plateau forest contains an invalid component root")
            root_events.add(root_ref)
    if not forest.component_roots.size:
        raise ValueError("plateau forest must contain a component root")

    candidate_events = tuple(
        event
        for event in range(forest.representatives.size)
        if int(forest.sizes[event]) >= min_cluster_size or event in root_events
    )
    target_mask = (1 << forest.n_leaves) - 1
    antichains: list[tuple[int, ...]] = []
    for subset in range(1 << len(candidate_events)):
        coverage = fixed_mask
        selected_refs = fixed_refs.copy()
        for candidate_index, event in enumerate(candidate_events):
            if not subset & (1 << candidate_index):
                continue
            leaf_mask = event_leaf_masks[event]
            if coverage & leaf_mask:
                break
            coverage |= leaf_mask
            selected_refs.append(event)
        else:
            if coverage != target_mask:
                continue
            antichains.append(
                tuple(
                    sorted(
                        selected_refs,
                        key=lambda selected_ref: _selected_ref_key(
                            forest,
                            selected_ref,
                        ),
                    )
                )
            )
    if not antichains:
        raise RuntimeError("plateau forest has no feasible antichain")
    return tuple(antichains)


def enumerate_feasible_antichains(
    forest: PlateauForest,
    min_cluster_size: int = 2,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        _selected_nodes(forest, selected_refs)
        for selected_refs in _enumerate_feasible_refs(forest, min_cluster_size)
    )


def exhaustive_pair_cuts(
    forest: PlateauForest,
    truth: np.ndarray,
    min_cluster_size: int = 2,
) -> tuple[PairCut, ...]:
    statistics = _event_pair_statistics(forest, truth)
    return tuple(
        _pair_cut_from_refs(forest, statistics, selected_refs)
        for selected_refs in _enumerate_feasible_refs(forest, min_cluster_size)
    )


def exhaustive_maximum_pair_f1_cut(
    forest: PlateauForest,
    truth: np.ndarray,
    min_cluster_size: int = 2,
) -> PairCut:
    cuts = exhaustive_pair_cuts(forest, truth, min_cluster_size)
    return max(cuts, key=lambda cut: cut.metrics.f1)


def exhaustive_minimum_pair_error_cut(
    forest: PlateauForest,
    truth: np.ndarray,
    *,
    alpha: int | float,
    beta: int | float,
    min_cluster_size: int = 2,
) -> PairCut:
    cuts = exhaustive_pair_cuts(forest, truth, min_cluster_size)
    return min(
        cuts,
        key=lambda cut: (
            alpha * cut.metrics.false_positive_pairs
            + beta * cut.metrics.false_negative_pairs
        ),
    )


def oracle_gap_recovery(
    scorer_pair_f1: float,
    baseline_pair_f1: float,
    maximum_pair_f1: float,
    *,
    tolerance: float = PAIR_F1_ORACLE_TOLERANCE,
) -> float:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite number")
    values = (scorer_pair_f1, baseline_pair_f1, maximum_pair_f1)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pair-F1 values must be finite")
    headroom = maximum_pair_f1 - baseline_pair_f1
    if headroom <= tolerance:
        return 0.0
    return (scorer_pair_f1 - baseline_pair_f1) / headroom


def evaluate_hierarchy_headroom(
    hierarchy: ParisHierarchy,
    truth: np.ndarray,
    min_cluster_size: int = 2,
    *,
    plateau_forest: PlateauForest | None = None,
    graph: csr_matrix | None = None,
    baseline_labels: np.ndarray | None = None,
    tolerance: float = PAIR_F1_ORACLE_TOLERANCE,
) -> HierarchyHeadroomEvaluation:
    forest = (
        collapse_equal_height_plateaus(hierarchy)
        if plateau_forest is None
        else plateau_forest
    )
    if forest.n_leaves != hierarchy.n_leaves:
        raise ValueError("plateau forest and hierarchy have different leaf counts")
    statistics = _event_pair_statistics(forest, truth)
    if baseline_labels is None:
        if graph is None:
            raise ValueError("graph is required when baseline_labels are not provided")
        split_gate = modularity_split_gains(hierarchy, forest, graph)
        baseline_labels = adaptive_cut(
            hierarchy,
            min_cluster_size,
            plateau_forest=forest,
            split_gate=split_gate,
        ).labels
    baseline = exact_pair_metrics(baseline_labels, truth)
    maximum = _maximum_pair_f1_from_statistics(
        forest,
        statistics,
        min_cluster_size,
        tolerance=tolerance,
        max_iterations=MAX_DINKELBACH_ITERATIONS,
        initial_numerator=2 * baseline.true_positive_pairs,
        initial_denominator=max(
            1,
            baseline.predicted_positive_pairs + statistics.truth_pairs,
        ),
    )
    supported = _supported_pair_cuts_from_statistics(
        forest,
        statistics,
        min_cluster_size,
    )
    headroom = maximum.cut.metrics.f1 - baseline.f1
    return HierarchyHeadroomEvaluation(
        min_cluster_size=min_cluster_size,
        baseline=baseline,
        maximum_pair_f1=maximum,
        supported_cuts=supported,
        supported_frontier=nondominated_supported_cuts(supported),
        pair_f1_headroom=headroom,
        has_measurable_headroom=headroom > tolerance,
    )


@dataclass(frozen=True)
class SeedQuality:
    seed: int
    adaptive_clusters: int
    adaptive_ari: float
    adaptive_rare_f1: float
    unguarded_clusters: int
    unguarded_ari: float
    best_global_ari: float
    best_global_clusters: int
    best_cpm_ari: float
    best_cpm_resolution: float
    accepted: bool


@dataclass(frozen=True)
class QualityGateReport:
    seed_count: int
    accepted_seed_fraction: float
    passed: bool
    seeds: tuple[SeedQuality, ...]


def unequal_depth_block_graph(seed: int) -> tuple[csr_matrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    subgroup_sizes = (45, 45, 35, 35, 20)
    subgroup = np.repeat(np.arange(len(subgroup_sizes)), subgroup_sizes)
    labels = np.where(subgroup < 2, 0, np.where(subgroup < 4, 1, 2))
    probabilities = np.full((5, 5), 0.006)
    np.fill_diagonal(probabilities, (0.19, 0.19, 0.15, 0.15, 0.38))
    probabilities[0, 1] = probabilities[1, 0] = 0.10
    probabilities[2, 3] = probabilities[3, 2] = 0.115
    probabilities[:2, 2:4] = probabilities[2:4, :2] = 0.012
    probabilities[4, :4] = probabilities[:4, 4] = 0.003

    rows, columns = np.triu_indices(labels.size, k=1)
    present = rng.random(rows.size) < probabilities[subgroup[rows], subgroup[columns]]
    weights = rng.uniform(0.7, 1.3, int(present.sum()))
    upper = csr_matrix(
        (weights, (rows[present], columns[present])),
        shape=(labels.size, labels.size),
    )
    return csr_matrix(upper + upper.T), labels


def _rare_cluster_f1(labels: np.ndarray, truth: np.ndarray) -> float:
    rare = truth == 2
    best = 0.0
    for cluster in np.unique(labels):
        predicted = labels == cluster
        score = (
            2
            * int(np.count_nonzero(predicted & rare))
            / (int(np.count_nonzero(predicted)) + int(np.count_nonzero(rare)))
        )
        best = max(best, score)
    return best


def _best_global_cut(
    dendrogram: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, int]:
    best_ari = -1.0
    best_clusters = 0
    for n_clusters in range(2, len(truth) + 1):
        labels = straight_cut(dendrogram, n_clusters)
        ari = float(adjusted_rand_score(truth, labels))
        if ari > best_ari:
            best_ari = ari
            best_clusters = n_clusters
    return best_ari, best_clusters


def _best_cpm_partition(
    graph: csr_matrix,
    truth: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    import igraph
    import leidenalg

    upper = triu(graph, k=1).tocoo()
    igraph_graph = igraph.Graph(n=graph.shape[0])
    igraph_graph.add_edges(
        list(zip(upper.row.tolist(), upper.col.tolist(), strict=True))
    )
    igraph_graph.es["weight"] = upper.data.tolist()
    best_ari = -1.0
    best_resolution = 0.0
    for resolution in CPM_RESOLUTIONS:
        partition = leidenalg.find_partition(
            igraph_graph,
            leidenalg.CPMVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=seed,
        )
        ari = float(adjusted_rand_score(truth, partition.membership))
        if ari > best_ari:
            best_ari = ari
            best_resolution = float(resolution)
    return best_ari, best_resolution


def evaluate_quality_gate(
    seed_count: int = DEFAULT_SEED_COUNT,
    *,
    nthreads: int = 4,
) -> QualityGateReport:
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    outcomes: list[SeedQuality] = []
    for seed in range(seed_count):
        graph, truth = unequal_depth_block_graph(seed)
        hierarchy = fit_paris_hierarchy(graph, nthreads=nthreads)
        forest = collapse_equal_height_plateaus(hierarchy)
        split_gate = modularity_split_gains(hierarchy, forest, graph)
        adaptive = adaptive_cut(
            hierarchy,
            min_cluster_size=8,
            plateau_forest=forest,
            split_gate=split_gate,
        )
        unguarded = adaptive_cut(
            hierarchy,
            min_cluster_size=8,
            plateau_forest=forest,
        )
        dendrogram = hierarchy_to_dendrogram(hierarchy)
        adaptive_ari = float(adjusted_rand_score(truth, adaptive.labels))
        adaptive_rare_f1 = _rare_cluster_f1(adaptive.labels, truth)
        unguarded_ari = float(adjusted_rand_score(truth, unguarded.labels))
        global_ari, global_clusters = _best_global_cut(dendrogram, truth)
        cpm_ari, cpm_resolution = _best_cpm_partition(graph, truth, seed)
        accepted = (
            adaptive_rare_f1 >= MIN_RARE_CLUSTER_F1
            and adaptive.n_clusters <= MAX_ADAPTIVE_CLUSTERS
            and adaptive_ari >= global_ari - ORACLE_ARI_TOLERANCE
            and adaptive.n_clusters <= unguarded.n_clusters
            and adaptive_ari >= unguarded_ari - ORACLE_ARI_TOLERANCE
        )
        outcomes.append(
            SeedQuality(
                seed=seed,
                adaptive_clusters=adaptive.n_clusters,
                adaptive_ari=adaptive_ari,
                adaptive_rare_f1=adaptive_rare_f1,
                unguarded_clusters=unguarded.n_clusters,
                unguarded_ari=unguarded_ari,
                best_global_ari=global_ari,
                best_global_clusters=global_clusters,
                best_cpm_ari=cpm_ari,
                best_cpm_resolution=cpm_resolution,
                accepted=accepted,
            )
        )
    accepted_fraction = sum(item.accepted for item in outcomes) / seed_count
    return QualityGateReport(
        seed_count=seed_count,
        accepted_seed_fraction=accepted_fraction,
        passed=accepted_fraction >= MIN_ACCEPTED_SEED_FRACTION,
        seeds=tuple(outcomes),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    report = evaluate_quality_gate(arguments.seeds, nthreads=arguments.threads)
    print(json.dumps(asdict(report), indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
