import inspect

import numpy as np
import pytest

from scarf.assay import ATACassay, RNAassay
from scarf.datastore.datastore import DataStore
from scarf.storage.artifacts import ArtifactRef, inspect_artifact


def test_hvg_public_contract_removed_assay_persistence_methods() -> None:
    signature = inspect.signature(DataStore.mark_hvgs)
    assert "hvg_key_name" not in signature.parameters
    assert signature.parameters["label"].default == "hvgs"
    assert signature.return_annotation in {ArtifactRef, "ArtifactRef"}
    assert not hasattr(DataStore, "set_hvgs")
    assert not hasattr(RNAassay, "set_hvgs")
    assert not hasattr(RNAassay, "set_summary_stats")
    assert not hasattr(RNAassay, "set_feature_stats")
    assert not hasattr(ATACassay, "set_feature_stats")


def test_mark_hvgs_publishes_ref_and_exact_identity(datastore_ephemeral) -> None:
    store = datastore_ephemeral
    ref = store.mark_hvgs(
        from_assay="RNA",
        cell_key="I",
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
        label="test_hvgs",
        max_cells=np.inf,
        bin_strategy="adaptive",
    )

    assert isinstance(ref, ArtifactRef)
    assert ref.kind == "feature_selection"
    assert store.resolve_features("RNA", "test_hvgs") == ref
    status = inspect_artifact(store.zw, ref)
    assert status.operation == "mark_hvgs"
    assert set(status.inputs or {}) == {"feature_summary"}
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
    }
    group = store.load_artifact(ref)
    values = np.asarray(group["values"][:])
    corrected = np.asarray(group["corrected_variance"][:])
    assert values.dtype == np.dtype(bool)
    assert values.shape == corrected.shape == (store.RNA.feats.N,)

    reused = store.mark_hvgs(
        from_assay="RNA",
        cell_key="I",
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
        label="test_hvgs_alias",
        max_cells=np.inf,
        bin_strategy="adaptive",
    )
    assert reused == ref
    assert store.resolve_features("RNA", "test_hvgs_alias") == ref


def test_mark_hvgs_persists_effective_default_max_cells(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    n_selected = int(np.asarray(store.cells.fetch_all("I"), dtype=bool).sum())
    expected: int | float = n_selected - 20
    if expected <= 0:
        expected = np.inf

    implicit = store.mark_hvgs(
        from_assay="RNA",
        cell_key="I",
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
        label="implicit_max",
    )
    explicit = store.mark_hvgs(
        from_assay="RNA",
        cell_key="I",
        min_cells=0,
        top_n=5,
        n_bins=20,
        blacklist="",
        show_plot=False,
        label="explicit_max",
        max_cells=expected,
    )

    assert explicit == implicit
    assert inspect_artifact(store.zw, implicit).parameters["max_cells"] == expected


def test_mark_hvgs_rejects_empty_result_before_publication(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store._ensure_all_features(store.RNA)
    before = set(store.list_artifacts(kind="feature_selection", from_assay="RNA"))

    with pytest.raises(ValueError, match="HVG selection contains no features"):
        store.mark_hvgs(
            from_assay="RNA",
            cell_key="I",
            min_cells=0,
            max_cells=np.inf,
            top_n=5,
            min_var=np.inf,
            n_bins=20,
            blacklist="",
            show_plot=False,
            label="empty_hvgs",
        )

    after = set(store.list_artifacts(kind="feature_selection", from_assay="RNA"))
    assert after == before
    assert "empty_hvgs" not in store.RNA.feats.columns


def test_mark_hvgs_rejects_non_rna_assay(datastore_ephemeral) -> None:
    with pytest.raises(TypeError, match="RNAassay"):
        datastore_ephemeral.mark_hvgs(
            from_assay="assay2",
            show_plot=False,
        )
