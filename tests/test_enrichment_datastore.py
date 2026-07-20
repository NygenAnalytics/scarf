import numba
import numpy as np
import pandas as pd
import pytest
import zarr

from scarf.datastore.datastore import DataStore
from scarf.assay import norm_lib_size_log
from scarf.matrix import ChunkedArray
from scarf.utils.arrays import array_digest
from scarf.writers import (
    create_cell_data,
    create_zarr_count_assay,
    finalize_writer_counts,
)


def _configure_enrichment_keys(datastore):
    assay = datastore.RNA
    active_cells = datastore.cells.active_index("I")[:8]
    cell_mask = np.zeros(datastore.cells.N, dtype=bool)
    cell_mask[active_cells] = True
    datastore.cells.insert("enrichment_cells", cell_mask, overwrite=True)

    all_names = np.asarray(assay.feats.fetch_all("names"))
    selected_features = []
    seen = set()
    for index in assay.feats.active_index("I"):
        normalized = str(all_names[index]).upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected_features.append(int(index))
        if len(selected_features) == 12:
            break
    selected_features = np.asarray(selected_features, dtype=np.int64)
    feature_mask = np.zeros(assay.feats.N, dtype=bool)
    feature_mask[selected_features] = True
    assay.feats.insert("enrichment_features", feature_mask, overwrite=True)

    target_index = selected_features[:6]
    target_names = all_names[target_index].astype(str)
    net = pd.DataFrame(
        {
            "source": ["Alpha"] * 3 + ["βeta"] * 3,
            "target": target_names,
            "weight": [1.0, -1.0, 2.0, 0.5, 1.5, -0.5],
        }
    )
    return active_cells, selected_features, target_index, net


def test_waggr_streaming_persistence_cache_and_lazy_subset(
    datastore_ephemeral,
    monkeypatch,
):
    cells, _, targets, net = _configure_enrichment_keys(datastore_ephemeral)

    def fail_if_computed(*args, **kwargs):
        raise AssertionError("WAGGR materialized the selected input matrix")

    with monkeypatch.context() as compute_patch:
        compute_patch.setattr(ChunkedArray, "compute", fail_if_computed)
        result = datastore_ephemeral.run_waggr(
            net,
            "waggr_unicode",
            cell_key="enrichment_cells",
            feat_key="enrichment_features",
            tmin=3,
        )

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
    assert result.data.dtype == np.dtype(np.float32)
    assert result.storage_path.endswith("RNA/enrichment/waggr_unicode")
    slot = datastore_ephemeral.RNA.z["enrichment/waggr_unicode"]
    assert slot.attrs["complete"] is True
    assert slot["scores"].dtype == np.dtype(np.float32)
    assert slot.attrs["cell_digest"] == array_digest(cells.astype(np.int64))

    def fail_if_streamed(*args, **kwargs):
        raise AssertionError("cache hit streamed counts")

    with monkeypatch.context() as cache_patch:
        cache_patch.setattr(ChunkedArray, "stream_blocks", fail_if_streamed)
        cached = datastore_ephemeral.run_waggr(
            net,
            "waggr_unicode",
            cell_key="enrichment_cells",
            feat_key="enrichment_features",
            tmin=3,
        )
    assert cached.storage_path == result.storage_path

    subset = datastore_ephemeral.get_enrichment(
        "waggr_unicode",
        sources=["βeta", "Alpha"],
    )
    np.testing.assert_array_equal(subset.source_names, ["βeta", "Alpha"])
    np.testing.assert_allclose(subset.data.compute(), expected[:, [1, 0]], rtol=1e-5)
    with pytest.raises(ValueError, match="non-empty"):
        datastore_ephemeral.get_enrichment("waggr_unicode", sources=[])
    with pytest.raises(ValueError, match="duplicate"):
        datastore_ephemeral.get_enrichment(
            "waggr_unicode",
            sources=["Alpha", "Alpha"],
        )
    with pytest.raises(KeyError, match="not found"):
        datastore_ephemeral.get_enrichment(
            "waggr_unicode",
            sources=["missing"],
        )

    logged = datastore_ephemeral.run_waggr(
        net,
        "waggr_logged",
        cell_key="enrichment_cells",
        feat_key="enrichment_features",
        tmin=3,
        log_transform=True,
    )
    logged_values = np.log1p(normalized)
    logged_expected = np.column_stack(
        [
            logged_values[:, :3] @ np.array([1.0, -1.0, 2.0]) / 4.0,
            logged_values[:, 3:] @ np.array([0.5, 1.5, -0.5]) / 2.5,
        ]
    )
    np.testing.assert_allclose(logged.data.compute(), logged_expected, rtol=1e-5)
    assert (
        datastore_ephemeral.RNA.z["enrichment/waggr_logged"].attrs["execution_digest"]
        != slot.attrs["execution_digest"]
    )

    summed = datastore_ephemeral.run_waggr(
        net,
        "waggr_sum",
        cell_key="enrichment_cells",
        feat_key="enrichment_features",
        mode="wsum",
        tmin=3,
    )
    summed_expected = np.column_stack(
        [
            normalized[:, :3] @ np.array([1.0, -1.0, 2.0]),
            normalized[:, 3:] @ np.array([0.5, 1.5, -0.5]),
        ]
    )
    np.testing.assert_allclose(summed.data.compute(), summed_expected, rtol=1e-5)

    original_norm = datastore_ephemeral.RNA.normMethod
    datastore_ephemeral.RNA.normMethod = norm_lib_size_log
    try:
        with pytest.raises(ValueError, match="norm_lib_size"):
            datastore_ephemeral.run_waggr(
                net,
                "wrong_normalization",
                cell_key="enrichment_cells",
                feat_key="enrichment_features",
                tmin=3,
            )
    finally:
        datastore_ephemeral.RNA.normMethod = original_norm
    with pytest.raises(TypeError, match="RNAassay"):
        datastore_ephemeral.run_waggr(
            net,
            "wrong_assay",
            from_assay="assay2",
            tmin=3,
        )
    for invalid_label in ("", ".", "..", "a/b", "a\\b", "bad\nlabel"):
        with pytest.raises(ValueError):
            datastore_ephemeral.run_waggr(net, invalid_label, tmin=3)

    changed = net.copy()
    changed.loc[0, "weight"] = 4.0
    with pytest.raises(ValueError, match="different"):
        datastore_ephemeral.run_waggr(
            changed,
            "waggr_unicode",
            cell_key="enrichment_cells",
            feat_key="enrichment_features",
            tmin=3,
        )

    old_cell_index = result.cell_index.copy()
    mutated_mask = np.zeros(datastore_ephemeral.cells.N, dtype=bool)
    mutated_mask[cells[:4]] = True
    datastore_ephemeral.cells.insert(
        "enrichment_cells",
        mutated_mask,
        overwrite=True,
    )
    historical = datastore_ephemeral.get_enrichment("waggr_unicode")
    np.testing.assert_array_equal(historical.cell_index, old_cell_index)
    with pytest.raises(ValueError, match="different"):
        datastore_ephemeral.run_waggr(
            net,
            "waggr_unicode",
            cell_key="enrichment_cells",
            feat_key="enrichment_features",
            tmin=3,
        )


def test_aucell_ignores_weights_and_can_explicitly_replace_a_method(
    datastore_ephemeral,
    monkeypatch,
):
    _, features, _, net = _configure_enrichment_keys(datastore_ephemeral)
    original_threads = numba.get_num_threads()
    result = datastore_ephemeral.run_aucell(
        net,
        "activity",
        cell_key="enrichment_cells",
        feat_key="enrichment_features",
        tmin=3,
        n_up=4,
        tie_seed=7,
    )
    assert numba.get_num_threads() == original_threads

    scores = result.data.compute()
    assert scores.shape == (8, 2)
    assert np.all((0 <= scores) & (scores <= 1))
    slot = datastore_ephemeral.RNA.z["enrichment/activity"]
    assert slot.attrs["method"] == "aucell"
    assert "rank_feature_index" in slot
    assert sorted(np.asarray(slot["rank_feature_index"][:]).tolist()) == sorted(
        features.tolist()
    )
    with pytest.raises(ValueError, match="different"):
        datastore_ephemeral.run_aucell(
            net,
            "activity",
            cell_key="enrichment_cells",
            feat_key="enrichment_features",
            tmin=3,
            n_up=4,
            tie_seed=8,
        )

    weighted_differently = net.assign(weight=np.arange(1, 7, dtype=np.float64))

    def fail_if_streamed(*args, **kwargs):
        raise AssertionError("weight-only AUCell rerun streamed counts")

    with monkeypatch.context() as cache_patch:
        cache_patch.setattr(ChunkedArray, "stream_blocks", fail_if_streamed)
        cached = datastore_ephemeral.run_aucell(
            weighted_differently.sample(frac=1.0, random_state=3),
            "activity",
            cell_key="enrichment_cells",
            feat_key="enrichment_features",
            tmin=3,
            n_up=4,
            tie_seed=7,
        )
    np.testing.assert_array_equal(cached.source_names, result.source_names)

    with pytest.raises(ValueError, match="different"):
        datastore_ephemeral.run_waggr(
            net,
            "activity",
            cell_key="enrichment_cells",
            feat_key="enrichment_features",
            tmin=3,
        )
    replacement = datastore_ephemeral.run_waggr(
        net,
        "activity",
        cell_key="enrichment_cells",
        feat_key="enrichment_features",
        tmin=3,
        overwrite=True,
    )
    assert replacement.method == "waggr"
    activity_group = datastore_ephemeral.RNA.z["enrichment/activity"]
    active_slot = activity_group[activity_group.attrs["_active_slot"]]
    assert "rank_feature_index" not in active_slot


def test_incomplete_slots_and_read_only_access(
    datastore_ephemeral,
    monkeypatch,
):
    _, _, _, net = _configure_enrichment_keys(datastore_ephemeral)
    original = datastore_ephemeral.run_waggr(
        net,
        "complete",
        cell_key="enrichment_cells",
        feat_key="enrichment_features",
        tmin=3,
    )
    original_scores = original.data.compute()
    original_digest = datastore_ephemeral.RNA.z["enrichment/complete"].attrs[
        "execution_digest"
    ]

    read_only = DataStore(
        datastore_ephemeral.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    loaded = read_only.get_enrichment("complete", sources=["Alpha"])
    assert loaded.data.shape == (8, 1)
    with pytest.raises(ValueError, match="zarr_mode='r\\+'"):
        read_only.run_waggr(net, "read_only", tmin=3)

    import scarf.datastore._operations.features as feature_operations

    def interrupted_write(slot, **kwargs):
        slot.attrs["method"] = "waggr"
        raise RuntimeError("interrupted")

    changed = net.copy()
    changed.loc[0, "weight"] = 2.0
    with monkeypatch.context() as fault:
        fault.setattr(
            feature_operations,
            "_write_enrichment_slot",
            interrupted_write,
        )
        with pytest.raises(RuntimeError, match="interrupted"):
            datastore_ephemeral.run_waggr(
                changed,
                "complete",
                cell_key="enrichment_cells",
                feat_key="enrichment_features",
                tmin=3,
                overwrite=True,
            )
        with pytest.raises(RuntimeError, match="interrupted"):
            datastore_ephemeral.run_waggr(
                net,
                "broken",
                cell_key="enrichment_cells",
                feat_key="enrichment_features",
                tmin=3,
            )

    preserved = datastore_ephemeral.get_enrichment("complete")
    np.testing.assert_array_equal(preserved.data.compute(), original_scores)
    assert (
        datastore_ephemeral.RNA.z["enrichment/complete"].attrs["execution_digest"]
        == original_digest
    )
    broken = datastore_ephemeral.RNA.z["enrichment/broken"]
    assert broken.attrs["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        datastore_ephemeral.get_enrichment("broken")

    rebuilt = datastore_ephemeral.run_waggr(
        net,
        "broken",
        cell_key="enrichment_cells",
        feat_key="enrichment_features",
        tmin=3,
    )
    assert rebuilt.data.shape == (8, 2)

    replacement = datastore_ephemeral.run_waggr(
        changed,
        "complete",
        cell_key="enrichment_cells",
        feat_key="enrichment_features",
        tmin=3,
        overwrite=True,
    )
    assert replacement.data.shape == (8, 2)
    complete_group = datastore_ephemeral.RNA.z["enrichment/complete"]
    active_name = complete_group.attrs["_active_slot"]
    active_slot = complete_group[active_name]
    assert active_slot.attrs["execution_digest"] != original_digest
    active_slot["cell_index"][0] = (
        int(active_slot["cell_index"][0]) + datastore_ephemeral.cells.N
    )
    with pytest.raises(ValueError, match="cell digest"):
        datastore_ephemeral.get_enrichment("complete")


def test_workspace_results_are_written_to_the_assay_shell(tmp_path):
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
        chunk_size=(2, 3),
        n_cells=5,
        feat_ids=np.array([f"f{i}" for i in range(6)]),
        feat_names=np.array([f"g{i}" for i in range(6)]),
        dtype="uint32",
    )
    root["matrices/RNA/counts"][:] = np.array(
        [
            [6, 5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5, 6],
            [6, 1, 5, 2, 4, 3],
            [3, 6, 1, 5, 2, 4],
            [2, 5, 6, 1, 4, 3],
        ],
        dtype=np.uint32,
    )
    finalize_writer_counts(root, "RNA", "ws")
    datastore = DataStore(
        str(path),
        default_assay="RNA",
        workspace="ws",
        min_features_per_cell=0,
        min_cells_per_feature=0,
    )
    net = pd.DataFrame(
        {
            "source": ["A", "A", "A", "B", "B", "B"],
            "target": [f"g{i}" for i in range(6)],
        }
    )

    result = datastore.run_waggr(net, "workspace", tmin=3)
    aucell = datastore.run_aucell(
        net,
        "workspace_aucell",
        tmin=3,
        n_up=4,
        tie_seed=7,
    )

    assert result.storage_path == "ws/RNA/enrichment/workspace"
    assert aucell.storage_path == "ws/RNA/enrichment/workspace_aucell"
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
    assert "enrichment" in root["ws/RNA"]
    assert "enrichment" not in root["matrices/RNA"]
