import inspect
import re

import numpy as np
import pytest

from scarf.assay import ATACassay, RNAassay
from scarf.datastore.datastore import DataStore
from scarf.storage.artifacts import ArtifactRef, artifact_path, inspect_artifact
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import snapshot_run_metadata


def test_hvg_public_contract_removed_assay_persistence_methods() -> None:
    signature = inspect.signature(DataStore.select_hvgs)
    assert "hvg_key_name" not in signature.parameters
    assert "cell_key" not in signature.parameters
    assert "label" not in signature.parameters
    assert signature.return_annotation in {ArtifactRef, "ArtifactRef"}
    assert not hasattr(DataStore, "mark_hvgs")
    assert not hasattr(DataStore, "set_hvgs")
    assert not hasattr(RNAassay, "set_hvgs")
    assert not hasattr(RNAassay, "set_summary_stats")
    assert not hasattr(RNAassay, "set_feature_stats")
    assert not hasattr(ATACassay, "set_feature_stats")
    all_features = inspect.signature(DataStore.select_all_features)
    assert tuple(all_features.parameters) == ("self", "from_assay")
    assert all_features.return_annotation in {ArtifactRef, "ArtifactRef"}
    assert not hasattr(DataStore, "_ensure_all_features")


def test_hvg_regex_correction_recomputes_without_rewriting_saved_selection(
    datastore_ephemeral, monkeypatch
) -> None:
    import scarf.datastore._operations.features as operations
    import scarf.features.variability as variability

    store = datastore_ephemeral
    indices = np.flatnonzero(store.RNA.feats.fetch_all("nCells") > 2)[:2]
    names = np.array([f"GENE_{index}" for index in range(store.RNA.feats.N)])
    names[indices] = ["RPS3", "RPSX"]
    store.RNA.feats.insert("names", names, overwrite=True)
    cells = store.snapshot_cell_selection()
    options = dict(
        min_cells=0,
        max_cells=np.inf,
        top_n=store.RNA.feats.N,
        n_bins=20,
        blacklist=r"^RPS\d+$",
        keep_bounds=True,
        show_plot=False,
    )
    plan_selection = operations._feature_selection_plan

    def plan_without_blacklist_fingerprint(*args, **kwargs):
        kwargs["parameters"] = dict(kwargs["parameters"])
        kwargs["parameters"].pop("blacklist_fingerprint", None)
        return plan_selection(*args, **kwargs)

    def uppercase_matches(values, pattern):
        expression = re.compile(pattern.upper())
        return np.array(
            [expression.match(str(value).upper()) is not None for value in values]
        )

    with monkeypatch.context() as context:
        context.setattr(
            operations, "_feature_selection_plan", plan_without_blacklist_fingerprint
        )
        context.setattr(variability, "regex_match_mask", uppercase_matches)
        original = store.select_hvgs(cells, **options)
    old_group = store.load_artifact(original)
    old_attributes = dict(old_group.attrs)
    old_values = np.asarray(old_group["values"][:])
    np.testing.assert_array_equal(old_values[indices], [True, False])

    corrected = store.select_hvgs(cells, **options)

    assert corrected != original
    np.testing.assert_array_equal(
        store.load_artifact(corrected)["values"][:][indices], [False, True]
    )
    assert store.select_hvgs(cells, **options) == corrected
    assert store.resolve_features("RNA", original) == original
    assert dict(old_group.attrs) == old_attributes
    np.testing.assert_array_equal(old_group["values"][:], old_values)


def test_select_hvgs_returns_ref_without_creating_alias(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    cell_selection = store.snapshot_cell_selection()
    columns_before = set(store.RNA.feats.columns)
    ref = store.select_hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=np.inf,
        n_bins=20,
        lowess_frac=0.2,
        blacklist="",
        keep_bounds=True,
        show_plot=False,
        max_cells=np.inf,
        bin_strategy="adaptive",
    )

    assert isinstance(ref, ArtifactRef)
    assert ref.kind == "feature_selection"
    assert store.resolve_features("RNA", ref) == ref
    assert set(store.RNA.feats.columns) == columns_before
    status = inspect_artifact(store.zw, ref)
    assert status.operation == "select_hvgs"
    assert set(status.inputs or {}) == {"feature_summary", "feature_snapshot"}
    assert status.parameters == {
        "min_cells": 0,
        "max_cells": {"special_float": "inf"},
        "top_n": 5,
        "min_var": {"special_float": "-inf"},
        "max_var": {"special_float": "inf"},
        "min_mean": {"special_float": "-inf"},
        "max_mean": {"special_float": "inf"},
        "n_bins": 20,
        "lowess_frac": 0.2,
        "blacklist": "",
        "keep_bounds": True,
        "bin_strategy": "adaptive",
        "variance_estimator": "regularized_local_quantile",
        "variance_quantile": 0.25,
    }
    group = store.load_artifact(ref)
    values = np.asarray(group["values"][:])
    corrected = np.asarray(group["corrected_variance"][:])
    assert values.dtype == np.dtype(bool)
    assert values.shape == corrected.shape == (store.RNA.feats.N,)

    reused = store.select_hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        min_var=-np.inf,
        max_var=np.inf,
        min_mean=-np.inf,
        max_mean=np.inf,
        n_bins=20,
        lowess_frac=0.2,
        blacklist="",
        keep_bounds=True,
        show_plot=False,
        max_cells=np.inf,
        bin_strategy="adaptive",
    )
    assert reused == ref
    assert set(store.RNA.feats.columns) == columns_before


@pytest.mark.parametrize("bin_strategy", ["adaptive", "fixed"])
def test_select_hvgs_reuse_accounts_for_variance_estimator(
    datastore_ephemeral, bin_strategy
) -> None:
    store = datastore_ephemeral
    cell_selection = store.snapshot_cell_selection()
    options = {
        "min_cells": 0,
        "top_n": 5,
        "n_bins": 20,
        "lowess_frac": 0.2,
        "blacklist": "",
        "show_plot": False,
        "max_cells": np.inf,
        "bin_strategy": bin_strategy,
    }
    existing = store.select_hvgs(cell_selection, **options)
    group = store.zw[artifact_path(existing)]
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    if bin_strategy == "adaptive":
        parameters.pop("variance_estimator")
        parameters.pop("variance_quantile")
        group.attrs["provenance"] = {**provenance, "parameters": parameters}
    else:
        assert "variance_estimator" not in parameters
    stored_attributes = dict(group.attrs)
    stored_values = np.asarray(group["values"][:])
    stored_scores = np.asarray(group["corrected_variance"][:])

    selected = store.select_hvgs(cell_selection, **options)

    assert (selected != existing) == (bin_strategy == "adaptive")
    assert store.select_hvgs(cell_selection, **options) == selected
    assert store.resolve_features("RNA", existing) == existing
    preserved = store.load_artifact(existing)
    assert dict(preserved.attrs) == stored_attributes
    np.testing.assert_array_equal(preserved["values"][:], stored_values)
    np.testing.assert_array_equal(preserved["corrected_variance"][:], stored_scores)
    assert inspect_artifact(store.zw, selected).inputs == provenance["inputs"]


@pytest.mark.parametrize("stored_quantile", [None, 0.1])
def test_select_hvgs_recomputes_when_background_quantile_changes(
    datastore_ephemeral, stored_quantile
) -> None:
    store = datastore_ephemeral
    cells = store.snapshot_cell_selection()
    existing = store.select_hvgs(cells, show_plot=False)
    group = store.zw[artifact_path(existing)]
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    if stored_quantile is None:
        parameters.pop("variance_quantile")
    else:
        parameters["variance_quantile"] = stored_quantile
    group.attrs["provenance"] = {**provenance, "parameters": parameters}
    stored_attributes = dict(group.attrs)
    stored_scores = np.asarray(group["corrected_variance"][:])

    selected = store.select_hvgs(cells, show_plot=False)

    assert selected != existing
    assert store.select_hvgs(cells, show_plot=False) == selected
    assert store.resolve_features("RNA", existing) == existing
    assert dict(group.attrs) == stored_attributes
    np.testing.assert_array_equal(group["corrected_variance"][:], stored_scores)
    assert inspect_artifact(store.zw, selected).parameters["variance_quantile"] == 0.25


@pytest.mark.parametrize("quantile", [True, 0, 1, -0.1, "0.25"])
def test_hvg_artifact_rejects_invalid_background_quantile(
    datastore_ephemeral, quantile
) -> None:
    store = datastore_ephemeral
    ref = store.select_hvgs(store.snapshot_cell_selection(), show_plot=False)
    group = store.zw[artifact_path(ref)]
    provenance = dict(group.attrs["provenance"])
    parameters = {**provenance["parameters"], "variance_quantile": quantile}
    group.attrs["provenance"] = {**provenance, "parameters": parameters}

    with pytest.raises(ArtifactResolutionError, match="variance quantile"):
        store.resolve_features("RNA", ref)


def test_select_hvgs_default_calibrates_pbmc_malat1(datastore_ephemeral) -> None:
    store = datastore_ephemeral
    ref = store.select_hvgs(store.snapshot_cell_selection(), show_plot=False)
    names = np.asarray(store.RNA.feats.fetch_all("names"))
    index = int(np.flatnonzero(names == "MALAT1")[0])
    corrected = np.asarray(store.load_artifact(ref)["corrected_variance"][:])

    assert 0.9 < corrected[index] < 1.1


@pytest.mark.parametrize(
    ("estimator", "bin_strategy"),
    [("unknown", "adaptive"), ("regularized_local_quantile", "fixed")],
)
def test_hvg_artifact_rejects_incompatible_variance_estimator(
    datastore_ephemeral, estimator, bin_strategy
) -> None:
    store = datastore_ephemeral
    ref = store.select_hvgs(
        store.snapshot_cell_selection(),
        min_cells=0,
        max_cells=np.inf,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
    )
    group = store.zw[artifact_path(ref)]
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters.update(variance_estimator=estimator, bin_strategy=bin_strategy)
    group.attrs["provenance"] = {**provenance, "parameters": parameters}

    with pytest.raises(ArtifactResolutionError, match="variance estimator"):
        store.resolve_features("RNA", ref)


def test_select_hvgs_persists_effective_default_max_cells(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    n_selected = int(np.asarray(store.cells.fetch_all("I"), dtype=bool).sum())
    expected: int | float = n_selected - 20
    if expected <= 0:
        expected = np.inf
    cell_selection = store.snapshot_cell_selection()

    implicit = store.select_hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
    )
    explicit = store.select_hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
        max_cells=expected,
    )

    assert explicit == implicit
    assert inspect_artifact(store.zw, implicit).parameters["max_cells"] == expected


def test_select_hvgs_recomputes_when_feature_names_change(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    names = np.asarray(
        [f"GENE_{index}" for index in range(store.RNA.feats.N)],
    )
    store.RNA.feats.insert("names", names, overwrite=True)
    cell_selection = store.snapshot_cell_selection()
    first = store.select_hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="^MT-",
        show_plot=False,
        max_cells=np.inf,
    )

    store.RNA.feats.insert(
        "names",
        np.asarray([f"MT-{name}" for name in names]),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="No features passed HVG candidate filters"):
        store.select_hvgs(
            cell_selection,
            from_assay="RNA",
            min_cells=0,
            top_n=5,
            n_bins=20,
            blacklist="^MT-",
            show_plot=False,
            max_cells=np.inf,
        )

    assert inspect_artifact(store.zw, first).complete


def test_select_hvgs_rejects_empty_result_without_metadata_mutation(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.select_all_features(from_assay="RNA")
    cell_selection = store.snapshot_cell_selection()
    before = set(store.list_artifacts(kind="feature_selection", from_assay="RNA"))
    columns_before = set(store.RNA.feats.columns)

    with pytest.raises(ValueError, match="HVG selection contains no features"):
        store.select_hvgs(
            cell_selection,
            from_assay="RNA",
            min_cells=0,
            max_cells=np.inf,
            top_n=5,
            min_var=np.inf,
            n_bins=20,
            blacklist="",
            show_plot=False,
        )

    after = set(store.list_artifacts(kind="feature_selection", from_assay="RNA"))
    assert after == before
    assert set(store.RNA.feats.columns) == columns_before


def test_select_hvgs_rejects_non_rna_assay(datastore_ephemeral) -> None:
    cell_selection = datastore_ephemeral.snapshot_cell_selection()
    with pytest.raises(TypeError, match="RNAassay"):
        datastore_ephemeral.select_hvgs(
            cell_selection,
            from_assay="assay2",
            show_plot=False,
        )


def test_select_hvgs_read_only_guard_precedes_snapshot_planning(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    cell_selection = store.snapshot_cell_selection()
    store.zarr_mode = "r"

    with pytest.raises(PermissionError, match=r"zarr_mode='r\+'"):
        store.select_hvgs(cell_selection, show_plot=False)


def test_select_hvgs_requires_a_feature_name_snapshot(datastore_ephemeral) -> None:
    store = datastore_ephemeral
    cell_selection = store.snapshot_cell_selection()
    ref = store.select_hvgs(
        cell_selection,
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
        max_cells=np.inf,
    )
    unrelated_snapshot = snapshot_run_metadata(
        store.zw,
        table_path="RNA/featureData",
        id_column="ids",
        columns=("I",),
        axis="feature",
        assay="RNA",
    )
    group = store.zw[artifact_path(ref)]
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    inputs["feature_snapshot"] = unrelated_snapshot.to_dict()
    group.attrs["provenance"] = {**provenance, "inputs": inputs}

    with pytest.raises(ArtifactResolutionError) as caught:
        store.resolve_features("RNA", ref)
    assert caught.value.code == "snapshot_contract_mismatch"
