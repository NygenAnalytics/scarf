import inspect
import json
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from scarf import DataStore, PipelineExecutionError
from scarf.datastore._pipeline_recipe import (
    _STAGE_PARAMETERS,
    _parameter_value,
    _resolve_params,
    resolve_pipeline_recipe,
)
from scarf.storage.artifacts import ArtifactRef, provenance_hash
from scarf.storage.feature_selection import read_feature_selection_indices
from scarf.utils.logging import logger


@contextmanager
def _captured_warnings(level: str = "WARNING") -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level=level,
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


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


@pytest.fixture(scope="module")
def params_base(datastore_zarr_root: str, tmp_path_factory) -> str:
    """A PBMC store holding the artifacts of one run with the default params."""
    location = tmp_path_factory.mktemp("params_base") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    DataStore(str(location), default_assay="RNA").pipeline.run(params=_params())
    return str(location)


@pytest.fixture
def params_store(params_base: str, tmp_path) -> DataStore:
    """A writable copy of the base; runs reuse the stages they share with it."""
    location = tmp_path / "data.zarr"
    shutil.copytree(params_base, location)
    return DataStore(str(location), default_assay="RNA")


@pytest.fixture(scope="module")
def rejection_store(datastore_zarr_root: str, tmp_path_factory) -> DataStore:
    """One writable store for calls that must be refused before any write."""
    location = tmp_path_factory.mktemp("params_rejections") / "data.zarr"
    shutil.copytree(datastore_zarr_root, location)
    return DataStore(str(location), default_assay="RNA")


def test_params_forward_stage_settings_and_record_them(params_store):
    store = params_store

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
    hvg_parameters = store.inspect_artifact(run["highly_variable_features"]).parameters
    assert (hvg_parameters["top_n"], hvg_parameters["lowess_frac"]) == (50, 0.2)
    # The stage functions called with the same settings reuse the run's artifacts.
    assert store.run_pca(run["normalized"], dims=3, feat_scaling=False) == run["pca"]
    assert (
        store.build_connectivity_map(run["neighbors"], bandwidth=1.2)
        == run["connectivity_map"]
    )
    assert store.build_connectivity_map(run["neighbors"]) != run["connectivity_map"]


def test_params_pca_dims_zero_builds_the_graph_on_normalized_values(
    params_store, monkeypatch
):
    store = params_store
    n_features = len(
        read_feature_selection_indices(
            store.zw, "RNA", store.pipeline.list_runs()[0]["highly_variable_features"]
        )
    )
    assert n_features == 50
    reductions = store.list_artifacts(kind="reduction", from_assay="RNA")

    def no_custom_reduction(*_args, **_kwargs):
        raise AssertionError("pca_dims=0 registered a custom reduction")

    square_matrices: list[int] = []
    eye = np.eye

    def recorded_eye(n, *args, **kwargs):
        if n >= n_features:
            square_matrices.append(n)
        return eye(n, *args, **kwargs)

    monkeypatch.setattr(type(store), "run_custom_reduction", no_custom_reduction)
    monkeypatch.setattr(np, "eye", recorded_eye)

    run = store.pipeline.run(params=_params(pca={"dims": 0}, leiden=True))

    outputs = list(run)
    assert "pca" not in outputs and "reduction" not in outputs
    assert {"normalized", "ann_index", "neighbors", "connectivity_map"} <= set(outputs)
    stages = {stage["stage"]: stage["status"] for stage in run.report()["stages"]}
    assert stages["pca"] == "skipped"
    # Each selected feature is one graph coordinate: the ANN index and the
    # neighbors read the normalized artifact itself.
    for key in ("ann_index", "neighbors"):
        status = store.inspect_artifact(run[key])
        assert status.input_ref("coordinates") == run["normalized"]
    assert store.load_artifact(run["normalized"])["data"].shape[1] == n_features
    # No identity loadings: nothing registered a reduction or built an
    # F x F matrix.
    assert store.list_artifacts(kind="reduction", from_assay="RNA") == reductions
    assert square_matrices == []
    # Silhouette selection scores the candidates on the normalized values,
    # the coordinates of the graph they partition.
    decision = store.inspect_artifact(run["cluster_selection"])
    assert decision.input_ref("coordinates") == run["normalized"]
    assert decision.input_ref("connectivityMap") == run["connectivity_map"]
    assert run["clusters"] in {
        run[f"leiden_{key}"] for key in ("0.5", "0.75", "1.0", "1.25")
    }
    assert "clusters" in run.cells.columns
    with pytest.raises(ValueError, match="dims above 0"):
        store.pipeline.run(params=_params(pca={"dims": 0, "feat_scaling": False}))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"params": _params(pca={"dims": 0}, harmony={"batch_columns": ["ids"]})},
            "pca_dims=0 builds the graph on normalized values, which Harmony cannot "
            "correct; set pca_dims above 0 or omit harmony_batch_columns",
        ),
        (
            # Doublet scoring is on by default.
            {"pca_dims": 0},
            "pca_dims=0 builds the graph on normalized values, but doublet scoring "
            "needs a PCA graph; pass doublets=False or set pca_dims above 0",
        ),
    ],
    ids=["harmony", "doublets-default"],
)
def test_recipe_rejects_pca_dims_zero_with_harmony_or_doublets(
    rejection_store, arguments, message
):
    store = rejection_store
    before = _run_ids(store)

    with pytest.raises(ValueError) as caught:
        store.pipeline.run(**arguments)

    assert str(caught.value) == message
    # The recipe raised before the run record or any artifact was written.
    assert _run_ids(store) == before
    assert store.list_artifacts(kind="normalized", from_assay="RNA") == []


def test_params_selected_resolution_is_the_saved_clustering(params_store):
    store = params_store

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
    # Membership strength is computed on the saved clustering and run graph.
    inputs = store.inspect_artifact(run["membership_strength"]).inputs
    assert ArtifactRef.from_dict(inputs["clusters"]) == run["leiden_1.0"]
    assert ArtifactRef.from_dict(inputs["connectivity_map"]) == run["connectivity_map"]
    strength = run.cells.fetch("membership_strength")
    assert strength.shape == run.cells.fetch("clusters").shape
    assert np.all((strength >= 0) & (strength <= 1))
    assert run.report()["run"]["config"]["leiden"] == {
        "partitions": [0.5, 1.0],
        "selected": 1.0,
    }


def test_params_doublets_on_one_selected_cluster_name_the_pipeline_settings(
    params_store,
):
    store = params_store
    # At this resolution the k=15 graph of the fixture is one Leiden cluster.
    params = _params(
        neighbors={"k": 15},
        leiden={"partitions": [0.0001], "selected": 0.0001},
        doublets=True,
    )

    with pytest.raises(PipelineExecutionError) as caught:
        store.pipeline.run(params=params)

    assert caught.value.stage == "doublets"
    assert isinstance(caught.value.__cause__, ValueError)
    message = str(caught.value)
    assert "every selected cell has the same cluster label" in message
    for setting in (
        "params['leiden']['selected']",
        "params['doublets']['heterotypic_fraction'] to 0",
        "doublets=False",
    ):
        assert setting in message
    assert store.pipeline.open(run_id=caught.value.run_id).status == "failed"


def test_params_tsne_stage_forwards_its_settings(params_store, monkeypatch):
    store = params_store
    calls: list[dict[str, Any]] = []

    def run_sgtsne(_graph, initial, **kwargs):
        calls.append(kwargs)
        return np.asarray(initial, dtype=np.float64).T

    monkeypatch.setattr("scarf.embeddings.sgtsne.run_sgtsne", run_sgtsne)
    monkeypatch.setattr("scarf.embeddings.sgtsne.require_sgtsnepi", lambda: run_sgtsne)

    run = store.pipeline.run(params=_params(tsne={"max_iter": 60, "early_iter": 20}))

    assert (calls[0]["max_iter"], calls[0]["early_iter"]) == (60, 20)
    # The pipeline runs t-SNE quietly and never passes thread or file options.
    assert calls[0]["verbose"] is False
    assert not {"parallel", "nthreads", "temp_file_loc"} & set(calls[0])
    outputs = list(run)
    assert "embedding_initialization" in outputs
    assert "umap" not in outputs
    assert {"tsne_1", "tsne_2"} <= set(run.cells.columns)


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"clustering": {}}, ValueError, "Unknown params section"),
        ({"umap": {"epochs": 400}}, ValueError, "Unknown umap parameters"),
        # t-SNE has one single-threaded backend and no thread setting.
        ({"tsne": {"parallel": True}}, ValueError, "Unknown tsne parameters"),
        # The embeddings run late, so their settings are checked before any
        # stage runs, as are the select_hvgs checks of the hvg settings.
        ({"tsne": {"box_h": -1.0}}, ValueError, "box_h must be positive"),
        ({"umap": {"symmetric_graph": None}}, TypeError, "symmetric_graph must be"),
        ({"pca": True}, TypeError, "always runs"),
        ({"hvg": {"min_mean": float("inf")}}, ValueError, "must be finite"),
        ({"hvg": {"min_cells": 2.5}}, TypeError, "min_cells must be an integer"),
        ({"species": "human"}, ValueError, "species"),
        ({"harmony": True}, ValueError, "batch_columns"),
        ({"leiden": {"partitions": [0.5], "selected": 1.0}}, ValueError, "selected"),
    ],
)
def test_params_reject_invalid_sections_before_writing(
    rejection_store, params, error, match
):
    store = rejection_store
    before = _run_ids(store)

    with pytest.raises(error, match=match):
        store.pipeline.run(params=params)

    assert _run_ids(store) == before


# Identity digests of the HVG parameters that select_hvgs(cells, top_n=50)
# records on this fixture, as earlier releases recorded them: with the default
# max_cells=None (selected cells - 20) and with max_cells=np.inf (no cutoff).
_AUTO_DIGEST = "a4c2782984b3fd2199c1b6a25e1b58f7f2012a2277009f60bb8cdf4a49b2ae0a"
_NO_CUTOFF_DIGEST = "b7baa288f07fcf36e08b016f615744545f525fc14f7d0857bd5ff16f926f2203"


def _parameter_digest(store: DataStore, ref: ArtifactRef) -> str:
    status = store.inspect_artifact(ref)
    return provenance_hash(
        {"operation": status.operation, "parameters": status.parameters, "inputs": {}}
    )


def test_hvg_cutoffs_keep_the_identities_of_earlier_releases(params_store):
    store = params_store
    (base,) = store.pipeline.list_runs()

    # The base run omits max_cells, so its default None sets the automatic cutoff.
    assert _parameter_digest(store, base["highly_variable_features"]) == _AUTO_DIGEST
    uncut = store.select_hvgs(
        base["analysis_cell_selection"], top_n=50, max_cells=np.inf, show_plot=False
    )
    assert _parameter_digest(store, uncut) == _NO_CUTOFF_DIGEST


def test_params_and_shortcuts_cannot_set_the_same_setting(rejection_store):
    store = rejection_store
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


def test_checked_settings_are_recorded_as_given():
    sections, _species = _resolve_params(
        {
            "hvg": {
                "min_cells": np.int64(3),
                "n_bins": 50,
                "lowess_frac": 1,
                "keep_bounds": np.bool_(True),
                "bin_strategy": "fixed",
            },
            "tsne": {
                "box_h": 1,
                "max_iter": np.int64(800),
                "early_iter": 0,
                "symmetric_graph": np.bool_(True),
            },
        }
    )

    # Checked settings keep their JSON values, so 1 stays an integer.
    hvg = sections["hvg"]
    assert isinstance(hvg, dict)
    assert hvg == {
        "min_cells": 3,
        "n_bins": 50,
        "lowess_frac": 1,
        "keep_bounds": True,
        "bin_strategy": "fixed",
    }
    assert type(hvg["lowess_frac"]) is int
    assert sections["tsne"] == {
        "box_h": 1,
        "max_iter": 800,
        "early_iter": 0,
        "symmetric_graph": True,
    }
    # An omitted setting is checked as the method's default, and a switch
    # needs no settings.
    assert _resolve_params({"hvg": {}})[0] == {"hvg": {}}
    assert _resolve_params({"tsne": {}})[0] == {"tsne": {}}
    assert _resolve_params({"tsne": True})[0] == {"tsne": True}


def test_a_tsne_recipe_warns_at_resolution_when_sgtsnepi_is_missing(
    readonly_store, monkeypatch
):
    # An entry of None makes every import of sgtsnepi raise ImportError.
    monkeypatch.setitem(sys.modules, "sgtsnepi", None)

    with _captured_warnings() as messages:
        recipe = _resolve(readonly_store, params={"tsne": {"max_iter": 800}})
    assert recipe.tsne
    (message,) = [message for message in messages if "sgtsnepi" in message]
    assert "tsne extra" in message and "scarf[tsne]" in message
    # The run is not refused: an embedding the store holds is reused.
    assert "reuse" in message

    with _captured_warnings() as messages:
        _resolve(readonly_store, params={"tsne": False})
        _resolve(readonly_store)
    assert not [message for message in messages if "sgtsnepi" in message]


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


# Settings that choose the recipe's Leiden candidates rather than a keyword.
_RECIPE_SETTINGS = {"leiden": {"partitions", "selected"}}
# Settings whose default, an infinity, is no JSON value; a run omits them.
_INFINITE_DEFAULTS = {"hvg": {"min_var", "max_var", "min_mean", "max_mean"}}
# The public method whose keywords each stage's settings are, then the method
# that the stage passes them to where that is another one.
_STAGE_METHODS: dict[str, tuple[str, ...]] = {
    "cell_cycle": ("run_cell_cycle_scoring", "_run_cell_cycle_scoring_artifact"),
    "hvg": ("select_hvgs", "_select_hvgs_artifact"),
    "normalization": ("run_normalization",),
    "pca": ("run_pca",),
    "harmony": ("run_harmony", "_run_harmony_artifact"),
    "ann_index": ("build_ann_index",),
    "neighbors": ("query_neighbors",),
    "connectivity": ("build_connectivity_map",),
    "embedding_initialization": ("build_embedding_initialization",),
    "umap": ("run_umap", "_run_umap_artifact"),
    "tsne": ("run_tsne",),
    "leiden": ("run_leiden_clustering", "_run_leiden_artifact"),
    "paris": ("run_paris_clustering", "_run_paris_artifact"),
    "doublets": ("run_doublet_detection", "_run_doublet_detection_artifact"),
}


def _is_json_value(value: Any) -> bool:
    try:
        return json.loads(json.dumps(value, allow_nan=False)) == value
    except (TypeError, ValueError):
        return False


def _defaults(method: Any) -> dict[str, Any]:
    return {
        name: parameter.default
        for name, parameter in inspect.signature(method).parameters.items()
    }


def test_stage_settings_are_keywords_of_the_methods_they_configure(readonly_store):
    settings = {
        stage: keys - _RECIPE_SETTINGS.get(stage, set())
        for stage, keys in _STAGE_PARAMETERS.items()
    }
    # Membership strength and markers are switches only.
    assert {stage for stage, keys in settings.items() if keys} == set(_STAGE_METHODS)
    for stage, names in _STAGE_METHODS.items():
        public, *called = [_defaults(getattr(DataStore, name)) for name in names]
        for key in settings[stage]:
            assert all(key in keywords for keywords in (public, *called)), (stage, key)
            # Omitting a setting means the public method's default.
            default = public[key]
            assert all(keywords[key] == default for keywords in called), (stage, key)
            if key in _INFINITE_DEFAULTS.get(stage, set()):
                assert default in (-np.inf, np.inf), (stage, key, default)
            elif default is not inspect.Parameter.empty:
                assert _is_json_value(default), (stage, key, default)
    # The settings that the run's shortcuts pass default to the same values.
    run = _defaults(readonly_store.pipeline.run)
    assert (run["hvg_count"], run["pca_dims"], run["neighbors_k"]) == (
        _defaults(DataStore.select_hvgs)["top_n"],
        _defaults(DataStore.run_pca)["dims"],
        _defaults(DataStore.query_neighbors)["k"],
    )
    # Automatic filtering is the MAD method of auto_filter_cells, at its defaults.
    defaults = _defaults(DataStore.auto_filter_cells)
    filtering = _resolve(readonly_store, filtering=True).filtering
    assert filtering["method"] == defaults["method"] == "mad"
    assert (
        filtering["minP"],
        filtering["maxP"],
        filtering["nMads"],
        filtering["minCellsPerSample"],
    ) == tuple(
        defaults[name] for name in ("min_p", "max_p", "n_mads", "min_cells_per_sample")
    )


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
    # Without PCA there is no PCA graph to score doublets on.
    assert (
        _resolve(readonly_store, doublets=False, params={"pca": {"dims": 0}}).pca_dims
        == 0
    )
