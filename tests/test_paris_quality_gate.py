import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tests.paris_quality_gate import (
    MAX_ADAPTIVE_CLUSTERS,
    MIN_RARE_CLUSTER_F1,
    ORACLE_ARI_TOLERANCE,
    SUPPORTED_ALPHA_BETA_RATIOS,
    _dense_event_pair_statistics,
    _encoded_truth,
    _event_pair_statistics,
    _sparse_event_pair_statistics,
    evaluate_hierarchy_headroom,
    evaluate_quality_gate,
    exact_pair_metrics,
    exhaustive_pair_cuts,
    maximum_pair_f1_cut,
    oracle_gap_recovery,
    supported_pair_cuts,
)
from scarf.clustering._paris_core import ParisHierarchy
from scarf.clustering.paris_multiscale import (
    collapse_equal_height_plateaus,
    labels_from_selected_nodes,
)


def _hierarchy(
    children: list[tuple[int, int]],
    heights: list[float],
    *,
    component_roots: list[int] | None = None,
    synthetic_joins: list[bool] | None = None,
) -> ParisHierarchy:
    n_leaves = len(children) + 1
    child_array = np.asarray(children, dtype=np.int32)
    sizes = np.ones(2 * n_leaves - 1, dtype=np.int32)
    merge_sizes = np.empty(n_leaves - 1, dtype=np.int32)
    for merge_index, (left, right) in enumerate(children):
        size = int(sizes[left]) + int(sizes[right])
        merge_sizes[merge_index] = size
        sizes[n_leaves + merge_index] = size
    roots = (
        np.asarray([2 * n_leaves - 2], dtype=np.int32)
        if component_roots is None
        else np.asarray(component_roots, dtype=np.int32)
    )
    synthetic = (
        np.isinf(heights)
        if synthetic_joins is None
        else np.asarray(synthetic_joins, dtype=bool)
    )
    return ParisHierarchy(
        children=child_array,
        heights=np.asarray(heights, dtype=np.float64),
        sizes=merge_sizes,
        component_roots=roots,
        synthetic_joins=np.asarray(synthetic, dtype=bool),
        n_leaves=n_leaves,
        total_weight=1.0,
    )


def _tiny_oracle_cases() -> tuple[
    tuple[ParisHierarchy, np.ndarray, int],
    ...,
]:
    balanced_children: list[tuple[int, int]] = []
    balanced_heights: list[float] = []
    level = list(range(16))
    height = 1.0
    while len(level) > 1:
        next_level: list[int] = []
        for offset in range(0, len(level), 2):
            balanced_children.append((level[offset], level[offset + 1]))
            balanced_heights.append(height)
            next_level.append(16 + len(balanced_children) - 1)
        level = next_level
        height *= 2
    balanced = _hierarchy(balanced_children, balanced_heights)
    equal_height_plateau = _hierarchy(
        [(0, 1), (2, 3), (6, 7), (4, 5), (8, 9)],
        [1, 1, 1, 2, 10],
    )
    minimum_size_folding = _hierarchy(
        [(0, 1), (2, 3), (6, 7), (4, 5), (8, 9)],
        [0.5, 0.5, 1, 2, 10],
    )
    disconnected_with_isolate = _hierarchy(
        [(0, 1), (2, 3), (5, 6), (7, 4)],
        [1, 1, 4, np.inf],
        component_roots=[7, 4],
        synthetic_joins=[False, False, False, True],
    )
    return (
        (
            balanced,
            np.asarray([0, 0, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5]),
            2,
        ),
        (
            equal_height_plateau,
            np.asarray([0, 0, 1, 1, 2, 2]),
            2,
        ),
        (
            minimum_size_folding,
            np.asarray([0, 0, 1, 1, 2, 2]),
            3,
        ),
        (
            disconnected_with_isolate,
            np.asarray([0, 0, 1, 1, 0]),
            2,
        ),
    )


def test_exact_pair_metrics_uses_integer_pair_counts() -> None:
    metrics = exact_pair_metrics(
        np.asarray([1, 1, 1, 2]),
        np.asarray(["a", "a", "b", "b"]),
    )

    assert metrics.true_positive_pairs == 1
    assert metrics.false_positive_pairs == 2
    assert metrics.false_negative_pairs == 1
    assert metrics.true_negative_pairs == 2
    assert metrics.predicted_positive_pairs == 3
    assert metrics.truth_positive_pairs == 2
    assert metrics.total_pairs == 6
    assert metrics.precision == pytest.approx(1 / 3)
    assert metrics.recall == pytest.approx(1 / 2)
    assert metrics.f1 == pytest.approx(2 / 5)


def test_pair_metrics_excludes_missing_and_unknown_truth() -> None:
    metrics = exact_pair_metrics(
        np.asarray([1, 1, 2, 2]),
        np.asarray(["a", "unknown", None, "a"], dtype=object),
    )

    assert metrics.true_positive_pairs == 0
    assert metrics.false_positive_pairs == 0
    assert metrics.false_negative_pairs == 1
    assert metrics.true_negative_pairs == 0


def test_sparse_and_dense_event_histograms_are_identical() -> None:
    hierarchy, truth, _min_cluster_size = _tiny_oracle_cases()[0]
    truth = truth.astype(object)
    truth[3] = "unknown"
    forest = collapse_equal_height_plateaus(hierarchy)
    codes, evaluated_items = _encoded_truth(truth)
    n_classes = int(codes.max()) + 1

    dense = _dense_event_pair_statistics(
        forest,
        codes,
        evaluated_items,
        n_classes,
    )
    sparse = _sparse_event_pair_statistics(
        forest,
        codes,
        evaluated_items,
    )

    assert dense == sparse
    assert dense == _event_pair_statistics(forest, truth)


@pytest.mark.parametrize(
    ("hierarchy", "truth", "min_cluster_size"),
    _tiny_oracle_cases(),
)
def test_oracles_match_every_exhaustive_tiny_tree_objective(
    hierarchy: ParisHierarchy,
    truth: np.ndarray,
    min_cluster_size: int,
) -> None:
    forest = collapse_equal_height_plateaus(hierarchy)
    exhaustive = exhaustive_pair_cuts(forest, truth, min_cluster_size)
    exhaustive_nodes = {cut.selected_nodes for cut in exhaustive}

    for cut in exhaustive:
        labels = labels_from_selected_nodes(
            hierarchy,
            np.asarray(cut.selected_nodes, dtype=np.int32),
        )
        assert exact_pair_metrics(labels, truth) == cut.metrics

    maximum = maximum_pair_f1_cut(forest, truth, min_cluster_size)
    best_f1 = max(cut.metrics.f1 for cut in exhaustive)
    assert maximum.cut.metrics.f1 == pytest.approx(best_f1)
    assert maximum.cut.selected_nodes in exhaustive_nodes

    supported = supported_pair_cuts(forest, truth, min_cluster_size)
    assert [cut.alpha_beta_ratio for cut in supported] == pytest.approx(
        SUPPORTED_ALPHA_BETA_RATIOS
    )
    for supported_cut in supported:
        exact_minimum = min(
            supported_cut.alpha * cut.metrics.false_positive_pairs
            + supported_cut.beta * cut.metrics.false_negative_pairs
            for cut in exhaustive
        )
        assert supported_cut.weighted_error == exact_minimum
        assert supported_cut.cut.selected_nodes in exhaustive_nodes


def test_minimum_size_folding_forces_the_nearest_valid_ancestor() -> None:
    hierarchy, truth, min_cluster_size = _tiny_oracle_cases()[2]
    forest = collapse_equal_height_plateaus(hierarchy)

    exhaustive = exhaustive_pair_cuts(forest, truth, min_cluster_size)
    maximum = maximum_pair_f1_cut(forest, truth, min_cluster_size)

    assert [cut.selected_nodes for cut in exhaustive] == [(10,)]
    assert maximum.cut.selected_nodes == (10,)


def test_headroom_evaluation_keeps_the_existing_cut_as_baseline() -> None:
    hierarchy = _hierarchy(
        [(0, 1), (2, 3), (4, 5)],
        [1, 1, 10],
    )
    truth = np.asarray(["same"] * 4)
    graph = csr_matrix(
        (
            np.ones(6, dtype=np.float64),
            (
                np.asarray([0, 1, 2, 3, 0, 2]),
                np.asarray([1, 0, 3, 2, 2, 0]),
            ),
        ),
        shape=(4, 4),
    )

    evaluation = evaluate_hierarchy_headroom(hierarchy, truth, graph=graph)

    assert evaluation.baseline.f1 == pytest.approx(0.5)
    assert evaluation.maximum_pair_f1.cut.selected_nodes == (6,)
    assert evaluation.maximum_pair_f1.cut.metrics.f1 == pytest.approx(1.0)
    assert evaluation.pair_f1_headroom == pytest.approx(0.5)
    assert evaluation.has_measurable_headroom
    assert len(evaluation.supported_frontier) == 1


@pytest.mark.parametrize(
    ("scorer", "baseline", "maximum", "expected"),
    [
        (0.75, 0.5, 1.0, 0.5),
        (0.5, 0.5, 0.5, 0.0),
    ],
)
def test_oracle_gap_recovery(
    scorer: float,
    baseline: float,
    maximum: float,
    expected: float,
) -> None:
    assert oracle_gap_recovery(scorer, baseline, maximum) == pytest.approx(expected)


def test_unequal_depth_quality_gate_preserves_the_rare_branch() -> None:
    report = evaluate_quality_gate(seed_count=5, nthreads=2)

    assert report.passed
    accepted = [result for result in report.seeds if result.accepted]
    assert accepted, "no seed satisfied the discriminative acceptance criteria"
    for result in accepted:
        assert result.adaptive_rare_f1 >= MIN_RARE_CLUSTER_F1
        assert result.adaptive_clusters <= MAX_ADAPTIVE_CLUSTERS
        # The guarded cut must match the oracle horizontal cut that scans every
        # cluster count, so a silent regression to a worse partition fails here.
        assert result.adaptive_ari >= result.best_global_ari - ORACLE_ARI_TOLERANCE
        # The modularity guard may only coarsen the unguarded cut and must not
        # sacrifice recovered structure while doing so.
        assert result.adaptive_clusters <= result.unguarded_clusters
        assert result.adaptive_ari >= result.unguarded_ari - ORACLE_ARI_TOLERANCE
