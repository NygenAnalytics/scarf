from typing import Any

import numpy as np
import pytest

from scarf.datastore._pipeline_recipe import (
    _parameter_value,
    _resolve_params,
    resolve_pipeline_recipe,
)
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


_SHORTCUT_DEFAULTS: dict[str, Any] = {
    "assay": None,
    "label": None,
    "cell_key": "I",
    "filtering": False,
    "harmony_batch_columns": None,
    "hvg_count": 1000,
    "pca_dims": 21,
    "neighbors_k": 11,
    "umap": True,
    "leiden": True,
    "cell_cycle": True,
    "paris": True,
    "doublets": True,
    "markers": True,
    "snapshot_columns": (),
}


@pytest.fixture
def readonly_store(datastore_zarr_root):
    from scarf.datastore.datastore import DataStore

    return DataStore(datastore_zarr_root, default_assay="RNA", zarr_mode="r")


def _resolve(store, **overrides: Any):
    return resolve_pipeline_recipe(store, **{**_SHORTCUT_DEFAULTS, **overrides})


def test_params_values_are_recorded_as_json():
    assert type(_parameter_value(np.int64(3), "x")) is int
    assert _parameter_value(np.float32(0.5), "x") == 0.5
    assert _parameter_value((1, 2), "x") == [1, 2]
    assert _parameter_value(np.array([1.0, 2.0]), "x") == [1.0, 2.0]
    assert _parameter_value({"a": [np.bool_(True)]}, "x") == {"a": [True]}
    with pytest.raises(TypeError, match="keys must be strings"):
        _parameter_value({1: 2}, "x")
    for value in (object(), b"raw"):
        with pytest.raises(TypeError, match="must be a number"):
            _parameter_value(value, "x")
    with pytest.raises(ValueError, match="must be finite"):
        _parameter_value([1.0, float("nan")], "x")


def test_params_sections_are_validated():
    assert _resolve_params(None) == ({}, None)
    with pytest.raises(TypeError, match="mapping of stage names"):
        _resolve_params([("pca", {})])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="filtering"):
        _resolve_params({"filtering": "mad"})
    with pytest.raises(TypeError, match="mapping or bool"):
        _resolve_params({"umap": 400})
    with pytest.raises(ValueError, match="Unknown markers parameters"):
        _resolve_params({"markers": {"nthreads": 2}})

    sections, species = _resolve_params(
        {"markers": False, "species": "mus_musculus", "filtering": {"method": "mad"}}
    )
    assert sections == {"markers": False, "filtering": {"method": "mad"}}
    assert species == "mus_musculus"


def test_params_resolve_into_stage_settings(readonly_store):
    recipe = _resolve(
        readonly_store,
        params={
            "cell_cycle": {"n_bins": 30},
            "hvg": {"top_n": 500, "keep_bounds": True},
            "pca": {"dims": 10},
            "neighbors": {"k": 7, "batch_size": 1000},
            "umap": {"n_epochs": 400, "umap_dims": 3},
            "leiden": {"partitions": [0.5, 1.0], "selected": 0.5, "random_seed": 1},
            "paris": False,
            "doublets": {"save_k": 3},
            "tsne": True,
            "membership_strength": True,
        },
    )

    assert (recipe.hvg_count, recipe.pca_dims, recipe.neighbors_k) == (500, 10, 7)
    assert recipe.params_for("cell_cycle") == {"n_bins": 30}
    assert recipe.params_for("hvg") == {"keep_bounds": True}
    assert recipe.params_for("neighbors") == {"batch_size": 1000}
    assert recipe.params_for("leiden") == {"random_seed": 1}
    assert recipe.params_for("tsne") == {}
    assert recipe.leiden_selected == "0.5"
    assert (recipe.tsne, recipe.membership_strength, recipe.paris) == (
        True,
        True,
        False,
    )
    assert {"membership_strength", "tsne"} <= set(recipe.stage_order)
    config = recipe.to_config()
    assert config["leiden"] == {"partitions": [0.5, 1.0], "selected": 0.5}
    assert config["params"]["umap"] == {"n_epochs": 400, "umap_dims": 3}
    assert config["params"]["doublets"] == {"save_k": 3}


@pytest.mark.parametrize(
    ("shortcut", "params"),
    [
        ({"pca_dims": 10}, {"pca": {"dims": 5}}),
        ({"neighbors_k": 5}, {"neighbors": {"k": 7}}),
        ({"harmony_batch_columns": ["sample"]}, {"harmony": False}),
        ({"filtering": {"method": "mad"}}, {"filtering": False}),
        ({"paris": False}, {"paris": {"n_clusters": 5}}),
        ({"leiden": False}, {"leiden": {"partitions": [1.0]}}),
    ],
)
def test_params_conflict_with_changed_shortcuts(readonly_store, shortcut, params):
    with pytest.raises(ValueError, match="not both"):
        _resolve(readonly_store, **shortcut, params=params)


def test_params_stage_rules_are_checked(readonly_store):
    with pytest.raises(ValueError, match="collide"):
        _resolve(
            readonly_store,
            snapshot_columns=("umap_3",),
            params={"umap": {"umap_dims": 3}},
        )
    with pytest.raises(ValueError, match="needs batch_columns"):
        _resolve(readonly_store, params={"harmony": {"batch_size": 100}})
    with pytest.raises(ValueError, match="membership_strength requires"):
        _resolve(
            readonly_store,
            leiden=False,
            doublets=False,
            markers=False,
            params={"membership_strength": True},
        )
    assert (
        _resolve(readonly_store, params={"harmony": False}).harmony_batch_columns == ()
    )
    harmony = _resolve(
        readonly_store,
        params={"harmony": {"batch_columns": ["ids"], "batch_size": 100}},
    )
    assert harmony.harmony_batch_columns == ("ids",)
    assert harmony.params_for("harmony") == {"batch_size": 100}
    assert _resolve(readonly_store, params={"pca": {"dims": 0}}).pca_dims == 0
