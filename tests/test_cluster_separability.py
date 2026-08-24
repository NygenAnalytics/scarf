import numpy as np
import pytest

from scarf.metrics import (
    ClusterSeparabilityResult,
    evaluate_cluster_separability,
)


def _score_by_name(result: ClusterSeparabilityResult, name: str):
    rows = result.clustering_scores.set_index("clustering")
    return rows.loc[name]


def test_sampling_is_deterministic_stratified_and_shared():
    rng = np.random.default_rng(12)
    coordinates = rng.normal(size=(240, 4))
    finest = np.repeat(np.arange(12), 20)
    coarser = finest // 3
    clusterings = {"finest": finest, "coarser": coarser}

    first = evaluate_cluster_separability(
        coordinates,
        clusterings,
        n_folds=3,
        max_sample_cells=60,
        max_silhouette_cells=40,
        random_seed=91,
    )
    second = evaluate_cluster_separability(
        coordinates,
        clusterings,
        n_folds=3,
        max_sample_cells=60,
        max_silhouette_cells=40,
        random_seed=91,
    )

    np.testing.assert_array_equal(first.sample_indices, second.sample_indices)
    np.testing.assert_array_equal(
        np.bincount(finest[first.sample_indices], minlength=12),
        np.full(12, 5),
    )
    assert set(first.clustering_scores["n_sampled_cells"]) == {60}
    assert (
        first.cluster_scores.groupby("clustering")["n_sampled_cells"].sum() == 60
    ).all()


def test_scores_are_held_out_and_cover_every_sampled_cell():
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(7)
    labels = np.tile([0, 1, 2], 50)
    coordinates = rng.normal(size=(len(labels), 120))

    result = evaluate_cluster_separability(
        coordinates,
        {"noise": labels},
        max_silhouette_cells=len(labels),
    )
    in_sample = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=4444,
        ),
    ).fit(coordinates, labels)
    in_sample_f1 = f1_score(
        labels,
        in_sample.predict(coordinates),
        average="macro",
    )
    score = _score_by_name(result, "noise")

    # These coordinates carry no cluster signal, so a model fitted and scored on
    # the same rows reaches perfect separation while held-out folds cannot.
    assert in_sample_f1 > 0.95
    assert score["macro_f1_mean"] < 0.5
    assert result.confusion["n_cells"].sum() == len(labels)
    assert result.cluster_scores["n_sampled_cells"].sum() == len(labels)


def test_separable_labels_score_higher_than_overlapping_labels():
    rng = np.random.default_rng(34)
    separable = np.repeat(np.arange(3), 80)
    coordinates = np.column_stack(
        (
            separable * 5 + rng.normal(scale=0.35, size=len(separable)),
            rng.normal(scale=0.5, size=len(separable)),
        )
    )
    overlapping = np.tile(np.arange(3), 80)

    result = evaluate_cluster_separability(
        coordinates,
        {"separable": separable, "overlapping": overlapping},
        max_silhouette_cells=240,
    )

    separable_score = _score_by_name(result, "separable")
    overlapping_score = _score_by_name(result, "overlapping")
    assert separable_score["macro_f1_mean"] > 0.9
    assert separable_score["macro_f1_mean"] > overlapping_score["macro_f1_mean"] + 0.4
    assert separable_score["silhouette_score"] > overlapping_score["silhouette_score"]


def test_macro_and_weighted_f1_reflect_cluster_imbalance():
    rng = np.random.default_rng(52)
    labels = np.concatenate((np.zeros(120), np.ones(30), np.full(10, 2))).astype(int)
    coordinates = np.concatenate(
        (
            rng.normal(loc=(-3, 0), scale=0.5, size=(120, 2)),
            rng.normal(loc=(3, 0), scale=0.5, size=(30, 2)),
            rng.normal(loc=(-3, 0), scale=0.5, size=(10, 2)),
        )
    )

    result = evaluate_cluster_separability(
        coordinates,
        {"imbalanced": labels},
        max_silhouette_cells=len(labels),
    )
    score = _score_by_name(result, "imbalanced")

    assert score["weighted_f1_mean"] > score["macro_f1_mean"]
    cluster_scores = result.cluster_scores.set_index("cluster_label")
    assert cluster_scores.loc[0, "f1_score"] > cluster_scores.loc[2, "f1_score"]


def test_confusion_fractions_use_the_true_cluster_as_denominator():
    rng = np.random.default_rng(11)
    labels = np.asarray(["big"] * 200 + ["small"] * 25)
    coordinates = np.concatenate(
        (
            rng.normal(loc=(0, 0), scale=1.0, size=(200, 2)),
            rng.normal(loc=(2.2, 0), scale=1.0, size=(25, 2)),
        )
    )

    result = evaluate_cluster_separability(
        coordinates,
        {"named": labels},
        max_silhouette_cells=len(labels),
    )
    confusion = result.confusion
    true_totals = confusion.groupby("true_cluster")["n_cells"].transform("sum")
    predicted_totals = confusion.groupby("predicted_cluster")["n_cells"].transform(
        "sum"
    )

    np.testing.assert_allclose(
        confusion["fraction_of_true_cluster"].to_numpy(),
        (confusion["n_cells"] / true_totals).to_numpy(),
    )
    np.testing.assert_allclose(
        confusion.groupby("true_cluster")["fraction_of_true_cluster"].sum().to_numpy(),
        1,
    )
    assert set(result.cluster_scores["cluster_label"]) == {"big", "small"}
    # The two clusters differ in size, so normalising by the predicted cluster
    # instead would report different off-diagonal fractions.
    off_diagonal = confusion["true_cluster"] != confusion["predicted_cluster"]
    assert (confusion["n_cells"][off_diagonal] > 0).any()
    assert not np.allclose(
        confusion["fraction_of_true_cluster"][off_diagonal].to_numpy(),
        (confusion["n_cells"] / predicted_totals)[off_diagonal].to_numpy(),
    )


def test_arbitrary_cluster_labels_round_trip():
    coordinates = np.concatenate(
        (
            np.linspace(-3, -1, 18),
            np.linspace(1, 3, 18),
        )
    )[:, None]
    one_based = np.repeat([1, 4], 18)
    strings = np.repeat(["alpha", "omega"], 18)

    result = evaluate_cluster_separability(
        coordinates,
        {"one_based": one_based, "strings": strings},
        n_folds=3,
        max_silhouette_cells=len(coordinates),
    )

    assert set(
        result.cluster_scores.query("clustering == 'one_based'")["cluster_label"]
    ) == {1, 4}
    assert set(result.confusion.query("clustering == 'strings'")["true_cluster"]) == {
        "alpha",
        "omega",
    }


def test_unscorable_clusterings_keep_silhouette_and_other_scores():
    coordinates = np.concatenate(
        (
            np.linspace(-4, -2, 6),
            np.linspace(2, 4, 4),
        )
    )[:, None]
    too_small = np.asarray([1] * 6 + [2] * 4)
    single = np.ones(10, dtype=int)
    scorable = np.asarray([1] * 5 + [2] * 5)

    result = evaluate_cluster_separability(
        coordinates,
        {"too_small": too_small, "single": single, "scorable": scorable},
        n_folds=5,
        max_silhouette_cells=len(coordinates),
    )
    too_small_score = _score_by_name(result, "too_small")
    single_score = _score_by_name(result, "single")
    scored = _score_by_name(result, "scorable")

    assert too_small_score["status"] == "unscorable"
    assert "fewer than 5" in too_small_score["status_reason"]
    assert np.isnan(too_small_score["macro_f1_mean"])
    assert np.isfinite(too_small_score["silhouette_score"])
    assert single_score["status"] == "unscorable"
    assert single_score["status_reason"] == "fewer than two clusters"
    assert np.isnan(single_score["silhouette_score"])
    assert set(
        result.cluster_scores.query("clustering == 'single'")["n_sampled_cells"]
    ) == {10}
    assert scored["status"] == "scored"
    assert np.isfinite(scored["macro_f1_mean"])
    assert set(result.confusion["clustering"]) == {"scorable"}


@pytest.mark.parametrize("max_silhouette_cells", [5, 10])
def test_silhouette_is_skipped_when_the_cap_starves_clusters(max_silhouette_cells):
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(10), 3)
    coordinates = rng.normal(size=(len(labels), 2))

    capped = evaluate_cluster_separability(
        coordinates,
        {"clusters": labels},
        n_folds=3,
        max_silhouette_cells=max_silhouette_cells,
    )
    generous = evaluate_cluster_separability(
        coordinates,
        {"clusters": labels},
        n_folds=3,
        max_silhouette_cells=len(labels),
    )
    capped_score = _score_by_name(capped, "clusters")

    assert np.isnan(capped_score["silhouette_score"])
    assert capped_score["status"] == "scored"
    assert np.isfinite(capped_score["macro_f1_mean"])
    assert np.isfinite(_score_by_name(generous, "clusters")["silhouette_score"])


@pytest.mark.parametrize(
    ("coordinates", "clusterings", "kwargs", "message"),
    [
        (np.ones(4), {"labels": np.arange(4)}, {}, "two-dimensional"),
        (np.ones((4, 2)), {"labels": np.arange(3)}, {}, "coordinate rows"),
        (
            np.ones((4, 2)),
            {"labels": np.asarray([0, 0, 1, np.nan])},
            {},
            "missing values",
        ),
        (
            np.asarray([[0.0], [1.0], [np.inf], [2.0]]),
            {"labels": np.asarray([0, 0, 1, 1])},
            {},
            "finite values",
        ),
        (
            np.ones((4, 2)),
            {"labels": np.arange(4)},
            {"n_folds": 1},
            "at least 2",
        ),
    ],
)
def test_invalid_inputs_are_rejected(coordinates, clusterings, kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_cluster_separability(coordinates, clusterings, **kwargs)


def test_datastore_wrapper_uses_explicit_pca_without_writes(
    datastore,
    graph_artifacts,
    leiden_clustering,
):
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    columns_before = set(datastore.cells.columns)
    artifacts_before = set(datastore.list_artifacts())

    result = datastore.metric_cluster_separability(
        state.reduction,
        ["RNA_leiden_cluster"],
        n_folds=3,
        max_sample_cells=300,
        max_silhouette_cells=100,
    )

    assert isinstance(result, ClusterSeparabilityResult)
    assert list(result.clustering_scores["clustering"]) == ["RNA_leiden_cluster"]
    assert int(result.clustering_scores["n_sampled_cells"].iloc[0]) == len(
        result.sample_indices
    )
    assert len(result.sample_indices) == min(
        300,
        len(datastore.cells.fetch("RNA_leiden_cluster", key="I")),
    )
    assert set(datastore.cells.columns) == columns_before
    assert set(datastore.list_artifacts()) == artifacts_before
    with pytest.raises(KeyError):
        datastore.metric_cluster_separability(
            state.reduction,
            ["missing_cluster_column"],
        )
    assert state.normalized is not None
    with pytest.raises(ValueError, match="reduction"):
        datastore.metric_cluster_separability(
            state.normalized,
            ["RNA_leiden_cluster"],
        )


def test_datastore_wrapper_rejects_a_foreign_cell_selection(
    datastore,
    graph_artifacts,
    leiden_clustering,
):
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    selected = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    subset = selected.copy()
    subset[np.flatnonzero(selected)[::2]] = False
    datastore.cells.insert(
        column_name="separability_subset",
        values=subset,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="does not match the cell selection"):
        datastore.metric_cluster_separability(
            state.reduction,
            ["RNA_leiden_cluster"],
            cell_key="separability_subset",
        )
