from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from sklearn import get_config

from scarf.metrics.cluster_selection import (
    ClusterSelectionResult,
    select_clusters_by_silhouette,
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
    expected_indices = np.sort(
        np.random.default_rng(4466)
        .choice(n_cells, size=8, replace=False)
        .astype(np.int64)
    )
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
        max_sample_size=8,
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
    sample_size = 7
    n_cells = 100

    class ZarrLikeArray:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.shape = values.shape
            self.selections: list[tuple[Any, ...]] = []

        def __getitem__(self, _index: object) -> np.ndarray:
            raise AssertionError("Zarr inputs must use orthogonal sampled reads")

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
    second = ZarrLikeArray(np.arange(n_cells, dtype=np.int64) % 3)
    monkeypatch.setattr("zarr.Array", ZarrLikeArray)
    monkeypatch.setattr("sklearn.metrics.silhouette_score", lambda *_a, **_k: 0.2)

    result = select_clusters_by_silhouette(
        coordinates,  # type: ignore[arg-type]
        (
            ("first", first),  # type: ignore[arg-type]
            ("second", second),  # type: ignore[arg-type]
        ),
        max_sample_size=sample_size,
    )

    expected_indices = np.sort(
        np.random.default_rng(4466)
        .choice(n_cells, size=sample_size, replace=False)
        .astype(np.int64)
    )
    np.testing.assert_array_equal(result.sample_indices, expected_indices)
    assert len(coordinates.selections) == 1
    np.testing.assert_array_equal(coordinates.selections[0][0], expected_indices)
    assert coordinates.selections[0][1] == slice(None)
    for labels in (first, second):
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
            "short: candidate has 3 labels for 4 PCA rows"
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
