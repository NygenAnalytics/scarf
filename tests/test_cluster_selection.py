from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from sklearn import get_config
from sklearn.metrics import silhouette_score

from scarf.metrics.cluster_selection import (
    ClusterSelectionResult,
    select_clusters_by_silhouette,
    shared_cluster_quota_sample_indices,
)


def test_cluster_selection_uses_one_deterministic_sample_and_first_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_cells = 30
    coordinates = np.column_stack(
        (
            np.arange(n_cells, dtype=np.float64),
            np.arange(n_cells, dtype=np.float64) * 2,
        )
    )
    labels = (
        np.zeros(n_cells, dtype=np.int64),
        np.arange(n_cells, dtype=np.int64) % 2,
        (np.arange(n_cells, dtype=np.int64) // 2) % 2,
        np.arange(n_cells, dtype=np.int64) % 3,
    )
    expected_indices = np.arange(n_cells, dtype=np.int64)
    score_results = iter((0.4, 0.4, float("nan")))
    score_calls: list[tuple[np.ndarray, np.ndarray, str, int]] = []

    def fake_silhouette_score(
        sampled_coordinates: np.ndarray,
        sampled_labels: np.ndarray,
        *,
        metric: str,
    ) -> float:
        score_calls.append(
            (
                sampled_coordinates.copy(),
                sampled_labels.copy(),
                metric,
                int(get_config()["working_memory"]),
            )
        )
        return next(score_results)

    monkeypatch.setattr(
        "sklearn.metrics.silhouette_score",
        fake_silhouette_score,
    )
    checkpoints: list[None] = []
    result = select_clusters_by_silhouette(
        coordinates,
        (
            ("single", labels[0]),
            ("first", labels[1]),
            ("second", labels[2]),
            ("nonfinite", labels[3]),
        ),
        max_sample_size=n_cells,
        working_memory_mib=17,
        checkpoint=lambda: checkpoints.append(None),
    )

    np.testing.assert_array_equal(result.sample_indices, expected_indices)
    np.testing.assert_allclose(result.scores, [np.nan, 0.4, 0.4, np.nan])
    assert result.invalid_reasons == (
        "sample contains fewer than two clusters",
        None,
        None,
        "silhouette score is not finite",
    )
    assert result.selected_key == "first"
    assert result.tie_order == ("single", "first", "second", "nonfinite")
    assert result.metric == "euclidean"
    assert result.sample_strategy == "sharedClusterQuota"
    assert result.min_cluster_quota == 2
    assert len(checkpoints) == 10
    assert len(score_calls) == 3
    for index, call in enumerate(score_calls, start=1):
        sampled_coordinates, sampled_labels, metric, working_memory = call
        np.testing.assert_array_equal(
            sampled_coordinates,
            coordinates[expected_indices],
        )
        np.testing.assert_array_equal(sampled_labels, labels[index][expected_indices])
        assert metric == "euclidean"
        assert working_memory == 17


def test_cluster_selection_reads_only_sampled_zarr_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_size = 20
    n_cells = 100

    class ZarrLikeArray:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.shape = values.shape
            self.selections: list[tuple[Any, ...]] = []
            self.full_reads = 0

        def __getitem__(self, index: object) -> np.ndarray:
            if index != slice(None):
                raise AssertionError("Scoring reads must use orthogonal sampled rows")
            self.full_reads += 1
            return self.values

        def get_orthogonal_selection(self, indexer: tuple[Any, ...]) -> np.ndarray:
            self.selections.append(indexer)
            return self.values[indexer]

    coordinates = ZarrLikeArray(
        np.column_stack(
            (
                np.arange(n_cells, dtype=np.float64),
                np.arange(n_cells, dtype=np.float64) * 2,
            )
        )
    )
    first = ZarrLikeArray(np.arange(n_cells, dtype=np.int64) % 2)
    second = ZarrLikeArray(np.arange(n_cells, dtype=np.int64) % 2)
    monkeypatch.setattr("zarr.Array", ZarrLikeArray)

    result = select_clusters_by_silhouette(
        coordinates,  # type: ignore[arg-type]
        (
            ("first", first),  # type: ignore[arg-type]
            ("second", second),  # type: ignore[arg-type]
        ),
        max_sample_size=sample_size,
    )

    expected_indices = shared_cluster_quota_sample_indices(
        (
            ("first", first.values),
            ("second", second.values),
        ),
        n_cells=n_cells,
        seed=4466,
        max_sample_size=sample_size,
    )
    np.testing.assert_array_equal(result.sample_indices, expected_indices)
    assert coordinates.full_reads == 0
    assert len(coordinates.selections) == 1
    np.testing.assert_array_equal(coordinates.selections[0][0], expected_indices)
    assert coordinates.selections[0][1] == slice(None)
    for labels in (first, second):
        assert labels.full_reads == 1
        assert len(labels.selections) == 1
        np.testing.assert_array_equal(labels.selections[0][0], expected_indices)


def test_cluster_selection_reports_each_invalid_candidate() -> None:
    coordinates = np.asarray([[0.0], [1.0], [2.0], [3.0]])

    with pytest.raises(
        ValueError,
        match=(
            "No clustering candidate is silhouette-scoreable: "
            "single: sample contains fewer than two clusters; "
            "unique: every sampled cell has a distinct cluster; "
            "short: candidate has 3 labels for 4 coordinate rows"
        ),
    ):
        select_clusters_by_silhouette(
            coordinates,
            (
                ("single", np.zeros(4, dtype=np.int64)),
                ("unique", np.arange(4, dtype=np.int64)),
                ("short", np.asarray([0, 0, 1], dtype=np.int64)),
            ),
        )


def test_shared_sample_covers_rare_clusters_and_fails_when_the_cap_is_too_small() -> (
    None
):
    n_cells = 200
    labels = np.zeros(n_cells, dtype=np.int64)
    labels[-3:] = 1
    coordinates = np.column_stack(
        (
            np.arange(n_cells, dtype=np.float64),
            np.zeros(n_cells, dtype=np.float64),
        )
    )
    result = select_clusters_by_silhouette(
        coordinates,
        (("imbalanced", labels),),
        max_sample_size=20,
    )
    sampled = labels[result.sample_indices]
    assert set(sampled) == {0, 1}
    assert int(np.count_nonzero(sampled == 1)) >= 2

    crowded = np.repeat(np.arange(8), 5).astype(np.int64)
    with pytest.raises(ValueError, match="exceeds max_sample_size"):
        select_clusters_by_silhouette(
            np.zeros((40, 2), dtype=np.float64),
            (("crowded", crowded),),
            max_sample_size=10,
        )


def test_select_clusters_prefers_separable_labels_and_matches_sklearn() -> None:
    rng = np.random.default_rng(12)
    separable_coordinates = np.vstack(
        (
            rng.normal(loc=0.0, scale=0.15, size=(80, 2)),
            rng.normal(loc=6.0, scale=0.15, size=(80, 2)),
        )
    )
    separable = np.concatenate((np.zeros(80), np.ones(80))).astype(np.int64)
    shuffled = rng.integers(0, 2, size=160)
    result = select_clusters_by_silhouette(
        separable_coordinates,
        (
            ("separable", separable),
            ("shuffled", shuffled),
        ),
    )

    sampled = result.sample_indices
    expected_separable = float(
        silhouette_score(
            separable_coordinates[sampled],
            separable[sampled],
            metric="euclidean",
        )
    )
    expected_shuffled = float(
        silhouette_score(
            separable_coordinates[sampled],
            shuffled[sampled],
            metric="euclidean",
        )
    )
    np.testing.assert_allclose(result.scores, [expected_separable, expected_shuffled])
    assert result.selected_key == "separable"
    assert expected_separable > expected_shuffled


def test_cluster_selection_result_is_strictly_immutable() -> None:
    sample_indices = np.asarray([0, 2], dtype=np.int64)
    scores = np.asarray([0.3, np.nan], dtype=np.float64)
    result = ClusterSelectionResult(
        candidate_keys=("selected", "invalid"),
        sample_indices=sample_indices,
        scores=scores,
        invalid_reasons=(None, "not scoreable"),
        selected_key="selected",
        seed=4466,
        population_size=3,
        max_sample_size=2,
        working_memory_mib=8,
    )
    sample_indices[0] = 1
    scores[0] = 0.9

    np.testing.assert_array_equal(result.sample_indices, [0, 2])
    np.testing.assert_allclose(result.scores, [0.3, np.nan])
    assert result.sample_definition == {
        "seed": 4466,
        "populationSize": 3,
        "sampleSize": 2,
        "maxSampleSize": 2,
        "sampleStrategy": "sharedClusterQuota",
        "minClusterQuota": 2,
    }
    with pytest.raises(ValueError, match="read-only"):
        result.scores[0] = 1.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        result.sample_indices.setflags(write=True)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        result.scores.setflags(write=True)
    with pytest.raises(TypeError):
        result.sample_definition["sampleSize"] = 1
    with pytest.raises(FrozenInstanceError):
        result.selected_key = "invalid"


def test_shared_sample_reserves_one_cell_for_a_singleton_cluster() -> None:
    n_cells = 50
    labels = np.zeros(n_cells, dtype=np.int64)
    labels[-1] = 1
    checkpoints: list[None] = []
    indices = shared_cluster_quota_sample_indices(
        (("labels", labels),),
        n_cells=n_cells,
        seed=7,
        max_sample_size=10,
        checkpoint=lambda: checkpoints.append(None),
    )
    sampled = labels[indices]
    assert checkpoints == [None]
    assert (n_cells - 1) in set(indices)
    assert int(np.count_nonzero(sampled == 1)) == 1
    assert len(indices) == 10

    exact = np.repeat(np.arange(5), 20).astype(np.int64)
    exact_indices = shared_cluster_quota_sample_indices(
        (("labels", exact),),
        n_cells=100,
        seed=3,
        max_sample_size=10,
    )
    assert len(exact_indices) == 10
    assert set(exact[exact_indices]) == {0, 1, 2, 3, 4}


def test_shared_sample_rejects_malformed_candidates_and_2d_labels() -> None:
    with pytest.raises(TypeError, match=r"\(key, labels\) tuples"):
        shared_cluster_quota_sample_indices(
            ["not-a-tuple"],  # type: ignore[arg-type]
            n_cells=20,
            seed=0,
            max_sample_size=5,
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        shared_cluster_quota_sample_indices(
            (("labels", np.zeros((20, 1), dtype=np.int64)),),
            n_cells=20,
            seed=0,
            max_sample_size=5,
        )
    with pytest.raises(ValueError, match="labels for 20 coordinate rows"):
        shared_cluster_quota_sample_indices(
            (("labels", np.zeros(19, dtype=np.int64)),),
            n_cells=20,
            seed=0,
            max_sample_size=5,
        )


def test_select_clusters_rejects_invalid_coordinates_and_sample_strategy() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    coordinates = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="two-dimensional"):
        select_clusters_by_silhouette(np.arange(4), (("a", labels),))
    with pytest.raises(ValueError, match="no coordinate rows"):
        select_clusters_by_silhouette(np.zeros((0, 2)), (("a", labels[:0]),))
    with pytest.raises(ValueError, match="at least one dimension"):
        select_clusters_by_silhouette(np.zeros((4, 0)), (("a", labels),))
    with pytest.raises(ValueError, match="at least one candidate"):
        select_clusters_by_silhouette(coordinates, ())
    with pytest.raises(ValueError, match="must be unique"):
        select_clusters_by_silhouette(
            coordinates,
            (("a", labels), ("a", labels)),
        )
    with pytest.raises(TypeError, match=r"\(key, labels\) tuples"):
        select_clusters_by_silhouette(coordinates, ["a"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one-dimensional"):
        select_clusters_by_silhouette(
            coordinates,
            (("a", np.zeros((4, 1), dtype=np.int64)),),
        )
    with pytest.raises(ValueError, match="sample_strategy must be"):
        ClusterSelectionResult(
            candidate_keys=("selected",),
            sample_indices=np.asarray([0, 1], dtype=np.int64),
            scores=np.asarray([0.3], dtype=np.float64),
            invalid_reasons=(None,),
            selected_key="selected",
            seed=0,
            population_size=2,
            max_sample_size=2,
            working_memory_mib=8,
            sample_strategy="uniform",
        )


def test_cluster_selection_result_rejects_an_inconsistent_winner() -> None:
    with pytest.raises(
        ValueError,
        match="selected_key must be the first candidate with the maximum score",
    ):
        ClusterSelectionResult(
            candidate_keys=("first", "second"),
            sample_indices=np.asarray([0, 1], dtype=np.int64),
            scores=np.asarray([0.5, 0.5], dtype=np.float64),
            invalid_reasons=(None, None),
            selected_key="second",
            seed=4466,
            population_size=2,
            max_sample_size=2,
            working_memory_mib=8,
        )
