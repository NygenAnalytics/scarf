import numba
import numpy as np
import pandas as pd
import pytest
import zarr

from scarf.datastore.datastore import DataStore
from scarf.assay import norm_lib_size_log
from scarf.matrix import ChunkedArray
from scarf.storage.artifacts import ArtifactRef
from scarf.storage.errors import ArtifactResolutionError
from scarf.utils.arrays import array_digest
from scarf.storage.budget import ResourceBudget
from scarf.storage.sharding import write_counts_t
from scarf.writers import (
    create_cell_data,
    create_zarr_count_assay,
)

pytestmark = pytest.mark.slow


def _configure_enrichment_keys(datastore):
    assay = datastore.RNA
    active_cells = datastore.cells.active_index("I")[:8]
    cell_mask = np.zeros(datastore.cells.N, dtype=bool)
    cell_mask[active_cells] = True
    datastore.cells.insert("enrichment_cells", cell_mask, overwrite=True)

    all_names = np.asarray(assay.feats.fetch_all("names"))
    selected_features = []
    seen = set()
    for index in range(assay.feats.N):
        normalized = str(all_names[index]).upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected_features.append(int(index))
        if len(selected_features) == 12:
            break
    selected_features = np.asarray(selected_features, dtype=np.int64)
    feature_ref = datastore.set_feature_selection(
        from_assay="RNA",
        feature_indexes=selected_features,
        label="enrichment_features",
    )
    alias_ref = datastore.set_feature_selection(
        from_assay="RNA",
        feature_indexes=selected_features,
        label="enrichment_features_alias",
    )
    assert alias_ref == feature_ref

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


def _enrichment_artifact(datastore, label):
    index = datastore.RNA.z["enrichment"].attrs["artifact_results"]
    ref = ArtifactRef.from_dict(index[label]["artifact"])
    return ref, datastore.load_artifact(ref)


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
            features="enrichment_features",
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
    assert "/RNA/artifacts/enrichment_scores/" in f"/{result.storage_path}"
    waggr_ref, slot = _enrichment_artifact(
        datastore_ephemeral,
        "waggr_unicode",
    )
    assert waggr_ref.kind == "enrichment_scores"
    assert slot.attrs["complete"] is True
    assert slot["scores"].dtype == np.dtype(np.float32)
    assert slot.attrs["cell_digest"] == array_digest(cells.astype(np.int64))

    def fail_if_streamed(*args, **kwargs):
        raise AssertionError("cache hit streamed counts")

    datastore_ephemeral.cells.insert(
        "enrichment_cells_alias",
        datastore_ephemeral.cells.fetch_all("enrichment_cells"),
        overwrite=True,
    )
    with monkeypatch.context() as cache_patch:
        cache_patch.setattr(ChunkedArray, "stream_blocks", fail_if_streamed)
        cached = datastore_ephemeral.run_waggr(
            net,
            "waggr_unicode",
            cell_key="enrichment_cells",
            features="enrichment_features",
            tmin=3,
        )
        alias = datastore_ephemeral.run_waggr(
            net,
            "waggr_alias",
            cell_key="enrichment_cells",
            features="enrichment_features",
            tmin=3,
        )
        key_alias = datastore_ephemeral.run_waggr(
            net,
            "waggr_key_alias",
            cell_key="enrichment_cells_alias",
            features="enrichment_features_alias",
            tmin=3,
        )
    assert cached.storage_path == result.storage_path
    assert alias.storage_path == result.storage_path
    assert key_alias.storage_path == result.storage_path
    assert key_alias.cell_key == "enrichment_cells_alias"
    assert key_alias.feature_selection == datastore_ephemeral.resolve_features(
        "RNA",
        "enrichment_features_alias",
    )
    enrichment_group = datastore_ephemeral.RNA.z["enrichment"]
    artifact_results = dict(enrichment_group.attrs["artifact_results"])
    alias_entry = dict(artifact_results["waggr_key_alias"])
    alias_entry["cell_key"] = "missing_enrichment_cells"
    artifact_results["waggr_key_alias"] = alias_entry
    enrichment_group.attrs["artifact_results"] = artifact_results
    with pytest.raises(ArtifactResolutionError) as error:
        datastore_ephemeral.get_enrichment("waggr_key_alias")
    assert error.value.code == "selection_column_missing"
    alias_entry["cell_key"] = "enrichment_cells_alias"
    artifact_results["waggr_key_alias"] = alias_entry
    enrichment_group.attrs["artifact_results"] = artifact_results
    assert (
        _enrichment_artifact(datastore_ephemeral, "waggr_alias")[0]
        == _enrichment_artifact(datastore_ephemeral, "waggr_unicode")[0]
    )
    original_ref = _enrichment_artifact(datastore_ephemeral, "waggr_unicode")[0]
    invalidated = datastore_ephemeral.run_waggr(
        net,
        "waggr_unicode",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
        invalidate_cache=True,
    )
    assert invalidated.storage_path != result.storage_path
    assert _enrichment_artifact(datastore_ephemeral, "waggr_unicode")[0] != original_ref
    preferred = datastore_ephemeral.run_waggr(
        net,
        "waggr_unicode",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
    )
    assert preferred.storage_path == invalidated.storage_path

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
        features="enrichment_features",
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
    logged_ref, logged_slot = _enrichment_artifact(
        datastore_ephemeral,
        "waggr_logged",
    )
    assert datastore_ephemeral.inspect_artifact(logged_ref).parameters["log_transform"]
    assert "execution_digest" not in logged_slot.attrs
    assert "execution_digest" not in slot.attrs

    summed = datastore_ephemeral.run_waggr(
        net,
        "waggr_sum",
        cell_key="enrichment_cells",
        features="enrichment_features",
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
                features="enrichment_features",
                tmin=3,
            )
    finally:
        datastore_ephemeral.RNA.normMethod = original_norm
    with pytest.raises(TypeError, match="RNAassay"):
        datastore_ephemeral.run_waggr(
            net,
            "wrong_assay",
            from_assay="assay2",
            features="all_features",
            tmin=3,
        )
    for invalid_label in ("", ".", "..", "a/b", "a\\b", "bad\nlabel"):
        with pytest.raises(ValueError):
            datastore_ephemeral.run_waggr(
                net,
                invalid_label,
                features="enrichment_features",
                tmin=3,
            )

    changed = net.copy()
    changed.loc[0, "weight"] = 4.0
    with pytest.raises(ValueError, match="different"):
        datastore_ephemeral.run_waggr(
            changed,
            "waggr_unicode",
            cell_key="enrichment_cells",
            features="enrichment_features",
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
    with pytest.raises(ArtifactResolutionError) as error:
        datastore_ephemeral.get_enrichment("waggr_unicode")
    assert error.value.code == "selection_values_changed"
    np.testing.assert_array_equal(result.cell_index, old_cell_index)
    with pytest.raises(ValueError, match="different"):
        datastore_ephemeral.run_waggr(
            net,
            "waggr_unicode",
            cell_key="enrichment_cells",
            features="enrichment_features",
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
        features="enrichment_features",
        tmin=3,
        n_up=4,
        tie_seed=7,
    )
    assert numba.get_num_threads() == original_threads

    scores = result.data.compute()
    assert scores.shape == (8, 2)
    assert np.all((0 <= scores) & (scores <= 1))
    _, slot = _enrichment_artifact(datastore_ephemeral, "activity")
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
            features="enrichment_features",
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
            features="enrichment_features",
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
            features="enrichment_features",
            tmin=3,
        )
    replacement = datastore_ephemeral.run_waggr(
        net,
        "activity",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
        overwrite=True,
    )
    assert replacement.method == "waggr"
    _, active_slot = _enrichment_artifact(datastore_ephemeral, "activity")
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
        features="enrichment_features",
        tmin=3,
    )
    original_scores = original.data.compute()
    original_ref, _original_slot = _enrichment_artifact(
        datastore_ephemeral,
        "complete",
    )

    read_only = DataStore(
        datastore_ephemeral.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    loaded = read_only.get_enrichment("complete", sources=["Alpha"])
    assert loaded.data.shape == (8, 1)
    with pytest.raises(ValueError, match="zarr_mode='r\\+'"):
        read_only.run_waggr(
            net,
            "read_only",
            features="enrichment_features",
            tmin=3,
        )

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
                features="enrichment_features",
                tmin=3,
                overwrite=True,
            )
        with pytest.raises(RuntimeError, match="interrupted"):
            datastore_ephemeral.run_waggr(
                changed,
                "broken",
                cell_key="enrichment_cells",
                features="enrichment_features",
                tmin=3,
            )

    preserved = datastore_ephemeral.get_enrichment("complete")
    np.testing.assert_array_equal(preserved.data.compute(), original_scores)
    preserved_ref, preserved_slot = _enrichment_artifact(
        datastore_ephemeral,
        "complete",
    )
    assert preserved_ref == original_ref
    assert "execution_digest" not in preserved_slot.attrs
    with pytest.raises(KeyError, match="not found"):
        datastore_ephemeral.get_enrichment("broken")

    rebuilt = datastore_ephemeral.run_waggr(
        changed,
        "broken",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
    )
    assert rebuilt.data.shape == (8, 2)

    replacement = datastore_ephemeral.run_waggr(
        changed,
        "complete",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
        overwrite=True,
    )
    assert replacement.data.shape == (8, 2)
    replacement_ref, _ = _enrichment_artifact(
        datastore_ephemeral,
        "complete",
    )
    assert replacement_ref != original_ref
    active_slot = datastore_ephemeral.zw[
        datastore_ephemeral.inspect_artifact(replacement_ref).path
    ]
    active_slot["cell_index"][0] = (
        int(active_slot["cell_index"][0]) + datastore_ephemeral.cells.N
    )
    with pytest.raises(ValueError, match="cell digest"):
        datastore_ephemeral.get_enrichment("complete")


def test_get_enrichment_rejects_unindexed_legacy_slots(datastore_ephemeral):
    enrichment = datastore_ephemeral.RNA.z.create_group("enrichment")
    enrichment.create_group("legacy_waggr")

    with pytest.raises(KeyError, match="was not found"):
        datastore_ephemeral.get_enrichment("legacy_waggr")


def test_enrichment_rebuilds_wrong_score_dtype_and_validates_provenance(
    datastore_ephemeral,
):
    _, _, _, net = _configure_enrichment_keys(datastore_ephemeral)
    datastore_ephemeral.run_waggr(
        net,
        "validated",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
    )
    original_ref, _ = _enrichment_artifact(datastore_ephemeral, "validated")
    original_path = datastore_ephemeral.inspect_artifact(original_ref).path
    original_group = datastore_ephemeral.zw[original_path]
    scores = np.asarray(original_group["scores"][:], dtype=np.float64)
    chunks = original_group["scores"].chunks
    del original_group["scores"]
    original_group.create_array("scores", data=scores, chunks=chunks)

    datastore_ephemeral.run_waggr(
        net,
        "validated",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
    )
    replacement_ref, _ = _enrichment_artifact(datastore_ephemeral, "validated")
    assert replacement_ref != original_ref

    replacement_path = datastore_ephemeral.inspect_artifact(replacement_ref).path
    replacement_group = datastore_ephemeral.zw[replacement_path]
    expected_network = replacement_group.attrs["network_digest"]
    replacement_group.attrs["network_digest"] = "corrupt"
    with pytest.raises(ValueError, match="network input"):
        datastore_ephemeral.get_enrichment("validated")
    replacement_group.attrs["network_digest"] = expected_network

    provenance = dict(replacement_group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    normalization_method = dict(parameters["normalization_method"])
    normalization_method["module"] = "corrupt.module"
    parameters["normalization_method"] = normalization_method
    provenance["parameters"] = parameters
    replacement_group.attrs["provenance"] = provenance
    with pytest.raises(ValueError, match="normalization provenance"):
        datastore_ephemeral.get_enrichment("validated")


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
    feature_ref = datastore.set_feature_selection(
        from_assay="RNA",
        mask=np.ones(6, dtype=bool),
        label="workspace_features",
    )

    result = datastore.run_waggr(
        net,
        "workspace",
        features=feature_ref,
        tmin=3,
    )
    aucell = datastore.run_aucell(
        net,
        "workspace_aucell",
        features=feature_ref,
        tmin=3,
        n_up=4,
        tie_seed=7,
    )

    assert result.storage_path.startswith("ws/RNA/artifacts/enrichment_scores/")
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
    assert "enrichment" in root["ws/RNA"]
    assert "artifacts" in root["ws/RNA"]
    assert "enrichment" not in root["matrices/RNA"]


def test_get_enrichment_rejects_unknown_method_and_missing_arrays(datastore_ephemeral):
    _, _, _, net = _configure_enrichment_keys(datastore_ephemeral)
    datastore_ephemeral.run_waggr(
        net,
        "method_poison",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
    )
    method_ref, _ = _enrichment_artifact(datastore_ephemeral, "method_poison")
    method_path = datastore_ephemeral.inspect_artifact(method_ref).path
    method_group = datastore_ephemeral.zw[method_path]
    method_group.attrs["method"] = "mystery"
    with pytest.raises(ValueError, match="unknown method"):
        datastore_ephemeral.get_enrichment("method_poison")

    datastore_ephemeral.run_waggr(
        net,
        "array_poison",
        cell_key="enrichment_cells",
        features="enrichment_features",
        tmin=3,
    )
    array_ref, _ = _enrichment_artifact(datastore_ephemeral, "array_poison")
    array_path = datastore_ephemeral.inspect_artifact(array_ref).path
    array_group = datastore_ephemeral.zw[array_path]
    del array_group["source_sizes"]
    with pytest.raises(ValueError, match="missing required arrays"):
        datastore_ephemeral.get_enrichment("array_poison")
