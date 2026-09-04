import inspect

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
