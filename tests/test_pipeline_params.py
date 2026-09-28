from typing import Any

import numpy as np
import pytest

from scarf.storage.feature_selection import read_feature_selection_indices


def _params(**sections: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "filtering": False,
        "cell_cycle": False,
        "hvg": {"top_n": 50},
        "pca": {"dims": 3},
        "neighbors": {"k": 3},
        "umap": False,
        "leiden": False,
        "paris": False,
        "doublets": False,
        "markers": False,
    }
    params.update(sections)
    return params


def _run_ids(store) -> list[str]:
    return [run.run_id for run in store.pipeline.list_runs()]


def test_params_forward_stage_settings_and_record_them(datastore_ephemeral):
    store = datastore_ephemeral

    run = store.pipeline.run(
        params=_params(
            hvg={"top_n": 50, "lowess_frac": 0.2},
            pca={"dims": 3, "feat_scaling": False},
            connectivity={"bandwidth": 1.2},
            species="homo_sapiens",
        )
    )

    assert run.status == "completed"
    config = run.report()["run"]["config"]
    assert (config["hvgCount"], config["pcaDims"], config["neighborsK"]) == (50, 3, 3)
    assert config["species"] == "homo_sapiens"
    assert config["params"] == {
        "hvg": {"lowess_frac": 0.2},
        "pca": {"feat_scaling": False},
        "connectivity": {"bandwidth": 1.2},
    }
    # The stage functions called with the same settings reuse the run's artifacts.
    assert store.run_pca(run["normalized"], dims=3, feat_scaling=False) == run["pca"]
    assert (
        store.build_connectivity_map(run["neighbors"], bandwidth=1.2)
        == run["connectivity_map"]
    )
    assert store.build_connectivity_map(run["neighbors"]) != run["connectivity_map"]


def test_params_pca_dims_zero_builds_the_graph_on_normalized_values(
    datastore_ephemeral,
):
    store = datastore_ephemeral

    run = store.pipeline.run(params=_params(pca={"dims": 0}))

    outputs = list(run)
    assert "pca" not in outputs
    assert {"reduction", "ann_index", "neighbors", "connectivity_map"} <= set(outputs)
    n_features = len(
        read_feature_selection_indices(store.zw, "RNA", run["highly_variable_features"])
    )
    assert store.inspect_artifact(run["reduction"]).parameters["dims"] == n_features
    with pytest.raises(ValueError, match="dims above 0"):
        store.pipeline.run(params=_params(pca={"dims": 0, "feat_scaling": False}))


def test_params_selected_resolution_is_the_saved_clustering(datastore_ephemeral):
    store = datastore_ephemeral

    run = store.pipeline.run(
        params=_params(
            leiden={"partitions": [0.5, 1.0], "selected": 1.0},
            membership_strength=True,
        )
    )

    outputs = list(run)
    assert run["clusters"] == run["leiden_1.0"]
    assert "cluster_selection" not in outputs
    assert "membership_strength" in outputs
    assert {"clusters", "membership_strength"} <= set(run.cells.columns)
    assert run.report()["run"]["config"]["leiden"] == {
        "partitions": [0.5, 1.0],
        "selected": 1.0,
    }


def test_params_tsne_stage_forwards_its_settings(datastore_ephemeral, monkeypatch):
    store = datastore_ephemeral
    calls: list[dict[str, Any]] = []

    def run_sgtsne(_graph, initial, **kwargs):
        calls.append(kwargs)
        return np.asarray(initial, dtype=np.float64).T

    monkeypatch.setattr("scarf.embeddings.sgtsne.run_sgtsne", run_sgtsne)

    run = store.pipeline.run(params=_params(tsne={"max_iter": 60, "early_iter": 20}))

    assert (calls[0]["max_iter"], calls[0]["early_iter"]) == (60, 20)
    outputs = list(run)
    assert "embedding_initialization" in outputs
    assert "umap" not in outputs
    assert {"tsne_1", "tsne_2"} <= set(run.cells.columns)


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"clustering": {}}, ValueError, "Unknown params section"),
        ({"umap": {"epochs": 400}}, ValueError, "Unknown umap parameters"),
        ({"pca": True}, TypeError, "always runs"),
        ({"hvg": {"min_mean": float("inf")}}, ValueError, "must be finite"),
        ({"species": "human"}, ValueError, "species"),
        ({"harmony": True}, ValueError, "batch_columns"),
        ({"leiden": {"partitions": [0.5], "selected": 1.0}}, ValueError, "selected"),
    ],
)
def test_params_reject_invalid_sections_before_writing(
    datastore_ephemeral, params, error, match
):
    store = datastore_ephemeral
    before = _run_ids(store)

    with pytest.raises(error, match=match):
        store.pipeline.run(params=params)

    assert _run_ids(store) == before


def test_params_and_shortcuts_cannot_set_the_same_setting(datastore_ephemeral):
    store = datastore_ephemeral
    before = _run_ids(store)

    with pytest.raises(ValueError, match="not both"):
        store.pipeline.run(hvg_count=500, params={"hvg": {"top_n": 50}})
    with pytest.raises(ValueError, match="not both"):
        store.pipeline.run(umap=False, params={"umap": {"n_epochs": 10}})

    assert _run_ids(store) == before
