import numba
import numpy as np
import pandas as pd
import pytest
import zarr

from scarf.assay import norm_lib_size_log
from scarf.datastore.datastore import DataStore
from scarf.matrix import ChunkedArray
from scarf.storage.artifacts import ArtifactRef
from scarf.storage.budget import ResourceBudget
from scarf.storage.sharding import write_counts_t
from scarf.utils.arrays import array_digest
from scarf.writers import create_cell_data, create_zarr_count_assay

pytestmark = pytest.mark.slow


def _configure_enrichment_inputs(
    datastore,
) -> tuple[np.ndarray, ArtifactRef, ArtifactRef, np.ndarray, pd.DataFrame]:
    assay = datastore.RNA
    active_cells = datastore.cells.active_index("I")[:8]
    cell_mask = np.zeros(datastore.cells.N, dtype=bool)
    cell_mask[active_cells] = True
    datastore.cells.insert("enrichment_cells", cell_mask, overwrite=True)
    cell_selection = datastore.snapshot_cell_selection("enrichment_cells")

    all_names = np.asarray(assay.feats.fetch_all("names"))
    selected_features: list[int] = []
    seen: set[str] = set()
    for index, name in enumerate(all_names):
        normalized = str(name).upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected_features.append(index)
        if len(selected_features) == 12:
            break
    selected = np.asarray(selected_features, dtype=np.int64)
    feature_selection = datastore.set_feature_selection(
        from_assay="RNA",
        feature_indexes=selected,
    )
    target_index = selected[:6]
    net = pd.DataFrame(
        {
            "source": ["Alpha"] * 3 + ["βeta"] * 3,
            "target": all_names[target_index].astype(str),
            "weight": [1.0, -1.0, 2.0, 0.5, 1.5, -0.5],
        }
    )
    return active_cells, cell_selection, feature_selection, target_index, net


def test_waggr_returns_ref_streams_and_loads_explicitly(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    cells, cell_selection, features, targets, net = _configure_enrichment_inputs(
        datastore_ephemeral
    )
    cell_columns = set(datastore_ephemeral.cells.columns)
    feature_columns = set(datastore_ephemeral.RNA.feats.columns)

    def fail_if_computed(*args, **kwargs):
        raise AssertionError("WAGGR materialized the selected input matrix")

    with monkeypatch.context() as compute_patch:
        compute_patch.setattr(ChunkedArray, "compute", fail_if_computed)
        ref = datastore_ephemeral.run_waggr(
            net,
            cell_selection,
            features=features,
            tmin=3,
        )

    assert isinstance(ref, ArtifactRef)
    assert ref.kind == "enrichment_scores"
    assert set(datastore_ephemeral.cells.columns) == cell_columns
    assert set(datastore_ephemeral.RNA.feats.columns) == feature_columns
    assert "enrichment" not in datastore_ephemeral.RNA.z

    result = datastore_ephemeral.get_enrichment(ref)
    raw = datastore_ephemeral.RNA.rawData[cells, :][:, targets].compute()
    scalars = np.asarray(
        datastore_ephemeral.cells.fetch_all("RNA_nCounts")[cells],
        dtype=np.float64,
    )
    scalars[scalars == 0] = 1
    normalized = float(datastore_ephemeral.RNA.sf) * raw / scalars[:, None]
    expected = np.column_stack(
        [
            normalized[:, :3] @ np.array([1.0, -1.0, 2.0]) / 4.0,
            normalized[:, 3:] @ np.array([0.5, 1.5, -0.5]) / 2.5,
        ]
    )
    np.testing.assert_allclose(result.data.compute(), expected, rtol=1e-5)
    np.testing.assert_array_equal(result.source_names, ["Alpha", "βeta"])
    np.testing.assert_array_equal(result.source_sizes, [3, 3])
    np.testing.assert_array_equal(result.cell_index, cells)
    assert result.artifact == ref
    assert result.cell_selection == cell_selection
    assert result.feature_selection == features
    assert result.method == "waggr"
    assert result.data.dtype == np.dtype(np.float32)
    slot = datastore_ephemeral.load_artifact(ref)
    assert slot.attrs["cell_digest"] == array_digest(cells.astype(np.int64))
    assert "cell_key" not in slot.attrs

    subset = datastore_ephemeral.get_enrichment(
        ref,
        sources=["βeta", "Alpha"],
    )
    np.testing.assert_array_equal(subset.source_names, ["βeta", "Alpha"])
    np.testing.assert_allclose(subset.data.compute(), expected[:, [1, 0]], rtol=1e-5)
    with pytest.raises(ValueError, match="non-empty"):
        datastore_ephemeral.get_enrichment(ref, sources=[])
    with pytest.raises(ValueError, match="duplicate"):
        datastore_ephemeral.get_enrichment(ref, sources=["Alpha", "Alpha"])
    with pytest.raises(KeyError, match="not found"):
        datastore_ephemeral.get_enrichment(ref, sources=["missing"])


def test_enrichment_cache_identity_and_invalidation(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    _cells, selection, features, _targets, net = _configure_enrichment_inputs(
        datastore_ephemeral
    )
    first = datastore_ephemeral.run_waggr(
        net,
        selection,
        features=features,
        tmin=3,
    )

    def fail_if_streamed(*args, **kwargs):
        raise AssertionError("cache hit streamed counts")

    with monkeypatch.context() as cache_patch:
        cache_patch.setattr(ChunkedArray, "stream_blocks", fail_if_streamed)
        assert (
            datastore_ephemeral.run_waggr(
                net,
                selection,
                features=features,
                tmin=3,
            )
            == first
        )

    invalidated = datastore_ephemeral.run_waggr(
        net,
        selection,
        features=features,
        tmin=3,
        invalidate_cache=True,
    )
    assert invalidated != first
    assert (
        datastore_ephemeral.run_waggr(
            net,
            selection,
            features=features,
            tmin=3,
        )
        == invalidated
    )
    changed = net.copy()
    changed.loc[0, "weight"] = 4.0
    changed_ref = datastore_ephemeral.run_waggr(
        changed,
        selection,
        features=features,
        tmin=3,
    )
    assert changed_ref not in {first, invalidated}


def test_waggr_modes_and_explicit_contract_errors(datastore_ephemeral) -> None:
    cells, selection, features, targets, net = _configure_enrichment_inputs(
        datastore_ephemeral
    )
    raw = datastore_ephemeral.RNA.rawData[cells, :][:, targets].compute()
    scalars = np.asarray(
        datastore_ephemeral.cells.fetch_all("RNA_nCounts")[cells],
        dtype=np.float64,
    )
    scalars[scalars == 0] = 1
    normalized = float(datastore_ephemeral.RNA.sf) * raw / scalars[:, None]

    logged_ref = datastore_ephemeral.run_waggr(
        net,
        selection,
        features=features,
        tmin=3,
        log_transform=True,
    )
    logged = datastore_ephemeral.get_enrichment(logged_ref)
    logged_values = np.log1p(normalized)
    logged_expected = np.column_stack(
        [
            logged_values[:, :3] @ np.array([1.0, -1.0, 2.0]) / 4.0,
            logged_values[:, 3:] @ np.array([0.5, 1.5, -0.5]) / 2.5,
        ]
    )
    np.testing.assert_allclose(logged.data.compute(), logged_expected, rtol=1e-5)

    summed_ref = datastore_ephemeral.run_waggr(
        net,
        selection,
        features=features,
        mode="wsum",
        tmin=3,
    )
    summed = datastore_ephemeral.get_enrichment(summed_ref)
    summed_expected = np.column_stack(
        [
            normalized[:, :3] @ np.array([1.0, -1.0, 2.0]),
            normalized[:, 3:] @ np.array([0.5, 1.5, -0.5]),
        ]
    )
    np.testing.assert_allclose(summed.data.compute(), summed_expected, rtol=1e-5)

    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        datastore_ephemeral.run_waggr(
            net,
            selection,
            features="enrichment_features",  # type: ignore[arg-type]
            tmin=3,
        )
    with pytest.raises(TypeError, match="enrichment must be an ArtifactRef"):
        datastore_ephemeral.get_enrichment("waggr")  # type: ignore[arg-type]
    original_norm = datastore_ephemeral.RNA.normMethod
    datastore_ephemeral.RNA.normMethod = norm_lib_size_log
    try:
        with pytest.raises(ValueError, match="norm_lib_size"):
            datastore_ephemeral.run_waggr(
                net,
                selection,
                features=features,
                tmin=3,
            )
    finally:
        datastore_ephemeral.RNA.normMethod = original_norm
    with pytest.raises(TypeError, match="RNAassay"):
        datastore_ephemeral.run_waggr(
            net,
            selection,
            from_assay="assay2",
            features=features,
            tmin=3,
        )


def test_aucell_returns_ref_ignores_weights_and_restores_threads(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    _cells, selection, features, _targets, net = _configure_enrichment_inputs(
        datastore_ephemeral
    )
    original_threads = numba.get_num_threads()
    ref = datastore_ephemeral.run_aucell(
        net,
        selection,
        features=features,
        tmin=3,
        n_up=4,
        tie_seed=7,
    )
    assert numba.get_num_threads() == original_threads
    result = datastore_ephemeral.get_enrichment(ref)
    scores = result.data.compute()
    assert scores.shape == (8, 2)
    assert np.all((0 <= scores) & (scores <= 1))
    slot = datastore_ephemeral.load_artifact(ref)
    assert slot.attrs["method"] == "aucell"
    selected = np.flatnonzero(
        np.asarray(datastore_ephemeral.load_artifact(features)["values"][:], dtype=bool)
    )
    assert sorted(np.asarray(slot["rank_feature_index"][:]).tolist()) == sorted(
        selected.tolist()
    )

    weighted_differently = net.assign(weight=np.arange(1, 7, dtype=np.float64))

    def fail_if_streamed(*args, **kwargs):
        raise AssertionError("weight-only AUCell rerun streamed counts")

    with monkeypatch.context() as cache_patch:
        cache_patch.setattr(ChunkedArray, "stream_blocks", fail_if_streamed)
        cached = datastore_ephemeral.run_aucell(
            weighted_differently.sample(frac=1.0, random_state=3),
            selection,
            features=features,
            tmin=3,
            n_up=4,
            tie_seed=7,
        )
    assert cached == ref


def test_enrichment_loader_is_read_only_and_validates_artifact(
    datastore_ephemeral,
) -> None:
    _cells, selection, features, _targets, net = _configure_enrichment_inputs(
        datastore_ephemeral
    )
    ref = datastore_ephemeral.run_waggr(
        net,
        selection,
        features=features,
        tmin=3,
    )
    group = datastore_ephemeral.zw[datastore_ephemeral.inspect_artifact(ref).path]
    group.attrs["method"] = "mystery"
    with pytest.raises(ValueError, match="unknown method"):
        datastore_ephemeral.get_enrichment(ref)
    group.attrs["method"] = "waggr"

    read_only = DataStore(
        datastore_ephemeral.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    loaded = read_only.get_enrichment(ref, sources=["Alpha"])
    assert loaded.data.shape == (8, 1)
    with pytest.raises(ValueError, match=r"zarr_mode='r\+'"):  # producer only
        read_only.run_waggr(net, selection, features=features, tmin=3)


def test_enrichment_rebuilds_invalid_payload_and_keeps_refs_immutable(
    datastore_ephemeral,
) -> None:
    _cells, selection, features, _targets, net = _configure_enrichment_inputs(
        datastore_ephemeral
    )
    original = datastore_ephemeral.run_waggr(
        net,
        selection,
        features=features,
        tmin=3,
    )
    original_group = datastore_ephemeral.zw[
        datastore_ephemeral.inspect_artifact(original).path
    ]
    scores = np.asarray(original_group["scores"][:], dtype=np.float64)
    chunks = original_group["scores"].chunks
    del original_group["scores"]
    original_group.create_array("scores", data=scores, chunks=chunks)
    with pytest.raises(ValueError, match="invalid score dtype"):
        datastore_ephemeral.get_enrichment(original)

    replacement = datastore_ephemeral.run_waggr(
        net,
        selection,
        features=features,
        tmin=3,
    )
    assert replacement != original
    replacement_group = datastore_ephemeral.zw[
        datastore_ephemeral.inspect_artifact(replacement).path
    ]
    expected_network = replacement_group.attrs["network_digest"]
    replacement_group.attrs["network_digest"] = "corrupt"
    with pytest.raises(ValueError, match="network input"):
        datastore_ephemeral.get_enrichment(replacement)
    replacement_group.attrs["network_digest"] = expected_network


def test_workspace_results_live_in_the_assay_artifact_tree(tmp_path) -> None:
    path = tmp_path / "workspace.zarr"
    root = zarr.open_group(str(path), mode="w")
    create_cell_data(
        root,
        "ws",
        ids=np.array([f"c{i}" for i in range(5)]),
        names=np.array([f"c{i}" for i in range(5)]),
    )
    create_zarr_count_assay(
        z=root,
        assay_name="RNA",
        workspace="ws",
        n_cells=5,
        feat_ids=np.array([f"f{i}" for i in range(6)]),
        feat_names=np.array([f"g{i}" for i in range(6)]),
        dtype="uint32",
    )
    counts = root["matrices/RNA/counts"]
    counts[:] = np.array(
        [
            [6, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 6],
            [6, 1, 5, 2, 4, 3],
            [3, 6, 1, 5, 2, 4],
            [2, 5, 6, 1, 4, 3],
        ],
        dtype=np.uint32,
    )
    write_counts_t(
        counts,
        root["matrices/RNA"],
        resources=ResourceBudget(1024**3, 2),
    )
    datastore = DataStore(
        str(path),
        default_assay="RNA",
        workspace="ws",
        min_features_per_cell=0,
    )
    net = pd.DataFrame(
        {
            "source": ["A", "A", "A", "B", "B", "B"],
            "target": [f"g{i}" for i in range(6)],
        }
    )
    cell_selection = datastore.snapshot_cell_selection()
    feature_selection = datastore.set_feature_selection(
        from_assay="RNA",
        mask=np.ones(6, dtype=bool),
    )

    waggr_ref = datastore.run_waggr(
        net,
        cell_selection,
        features=feature_selection,
        tmin=3,
    )
    aucell_ref = datastore.run_aucell(
        net,
        cell_selection,
        features=feature_selection,
        tmin=3,
        n_up=4,
        tie_seed=7,
    )
    waggr = datastore.get_enrichment(waggr_ref)
    aucell = datastore.get_enrichment(aucell_ref)

    assert waggr.storage_path.startswith("ws/RNA/artifacts/enrichment_scores/")
    assert aucell.storage_path.startswith("ws/RNA/artifacts/enrichment_scores/")
    np.testing.assert_allclose(
        aucell.data.compute(),
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0 / 6.0, 1.0 / 6.0],
            [0.5, 0.5],
            [5.0 / 6.0, 1.0 / 6.0],
        ],
    )
    assert "artifacts" in root["ws/RNA"]
    assert "enrichment" not in root["ws/RNA"]
    assert "enrichment" not in root["matrices/RNA"]
