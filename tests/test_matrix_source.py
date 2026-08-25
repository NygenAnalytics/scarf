from pathlib import Path

import numpy as np
import pytest
import zarr
from obstore.store import MemoryStore as ObjectMemoryStore
from zarr.abc.store import Store
from zarr.codecs import BloscCodec, ZstdCodec
from zarr.storage import ObjectStore

from scarf.datastore.datastore import DataStore, mount_datastore
from scarf.storage.artifacts import ArtifactRef, artifact_group
from scarf.storage.budget import ResourceBudget
from scarf.storage.sharding import write_counts_t
from scarf.storage.stores import (
    MATRIX_SOURCE_ATTR,
    create_matrix_source,
    is_remote_datastore,
    resolve_matrix_source,
)
from tests.fixtures_datastore import build_neighbourhood_graph
from scarf.writers import (
    create_cell_data,
    create_zarr_count_assay,
    create_zarr_obj_array,
)


def _write_assay(
    root: zarr.Group,
    workspace: str | None,
    assay_name: str,
    values: np.ndarray,
    *,
    dataset_fingerprint: str | None = None,
) -> None:
    n_cells, n_feats = values.shape
    create_zarr_count_assay(
        z=root,
        assay_name=assay_name,
        workspace=workspace,
        n_cells=n_cells,
        feat_ids=np.array([f"{assay_name.lower()}-f{i}" for i in range(n_feats)]),
        feat_names=np.array([f"{assay_name.lower()}-g{i}" for i in range(n_feats)]),
        dtype="uint32",
    )
    if workspace is None:
        counts = root[f"{assay_name}/counts"]
        assay = root[assay_name]
    else:
        counts = root[f"matrices/{assay_name}/counts"]
        assay = root[f"{workspace}/{assay_name}"]
    counts[:] = values
    matrix_group = (
        root[assay_name] if workspace is None else root[f"matrices/{assay_name}"]
    )
    write_counts_t(
        counts,
        matrix_group,
        resources=ResourceBudget(1024**2, 2),
    )
    if dataset_fingerprint is not None:
        assay.attrs["dataset_fingerprint"] = dataset_fingerprint


def _write_source_store(
    path: str | Store,
    *,
    workspace: str | None,
    values: np.ndarray | None = None,
    dataset_fingerprint: str | None = None,
) -> np.ndarray:
    if values is None:
        values = np.arange(40, dtype=np.uint32).reshape(10, 4)
    n_cells, n_feats = values.shape
    root = zarr.open_group(path, mode="w")
    create_cell_data(
        root,
        workspace,
        ids=np.array([f"c{i}" for i in range(n_cells)]),
        names=np.array([f"c{i}" for i in range(n_cells)]),
    )
    _write_assay(
        root,
        workspace,
        "RNA",
        values,
        dataset_fingerprint=dataset_fingerprint,
    )
    if workspace is None:
        zw = root
    else:
        zw = root[workspace]
    zw.attrs["defaultAssay"] = "RNA"
    zw.attrs["assayTypes"] = {"RNA": "RNA"}
    return values


def _snapshot_store_files(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


@pytest.mark.parametrize("workspace", [None, "analysis"])
def test_mount_datastore_creates_and_reopens(tmp_path, workspace):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    values = _write_source_store(source, workspace=workspace)

    ds = mount_datastore(
        source,
        at=target,
        workspace=workspace,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    assert ds.workspace == workspace
    assert "counts" not in ds.z["RNA" if workspace is None else f"{workspace}/RNA"]
    np.testing.assert_array_equal(ds.RNA.rawData.compute(), values)

    reopened = DataStore(
        target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    assert reopened.workspace == workspace
    np.testing.assert_array_equal(reopened.RNA.rawData.compute(), values)
    assert MATRIX_SOURCE_ATTR in zarr.open_group(target, mode="r").attrs


def test_mount_datastore_multiple_assays(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    rna_values = _write_source_store(source, workspace=None)
    adt_values = np.arange(30, dtype=np.uint32).reshape(10, 3)
    source_root = zarr.open_group(source, mode="r+")
    _write_assay(source_root, None, "ADT", adt_values)
    source_root.attrs["assayTypes"] = {"RNA": "RNA", "ADT": "ADT"}

    ds = mount_datastore(
        source,
        at=target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    np.testing.assert_array_equal(ds.RNA.rawData.compute(), rna_values)
    np.testing.assert_array_equal(ds.ADT.rawData.compute(), adt_values)
    manifest = zarr.open_group(target, mode="r").attrs[MATRIX_SOURCE_ATTR]["assays"]
    assert (
        manifest["RNA"]["cellIdsFingerprint"] == manifest["ADT"]["cellIdsFingerprint"]
    )
    assert "counts" not in zarr.open_group(target, mode="r")["ADT"]


def test_mounted_store_reopens_from_another_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    values = _write_source_store("source.zarr", workspace=None)
    mount_datastore(
        "source.zarr",
        at="target.zarr",
        default_assay="RNA",
        min_features_per_cell=1,
    )
    target = str(tmp_path / "target.zarr")
    location = zarr.open_group(target, mode="r").attrs[MATRIX_SOURCE_ATTR]["location"]
    assert Path(location).resolve() == (tmp_path / "source.zarr").resolve()

    monkeypatch.chdir(tmp_path.parent)
    reopened = DataStore(
        target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    np.testing.assert_array_equal(reopened.RNA.rawData.compute(), values)


@pytest.mark.parametrize("target_kind", ["path", "store"])
def test_failed_mount_discards_target_and_allows_retry(
    monkeypatch,
    tmp_path,
    target_kind,
):
    source = str(tmp_path / "source.zarr")
    _write_source_store(source, workspace=None)
    target: str | Store = (
        str(tmp_path / "target.zarr")
        if target_kind == "path"
        else ObjectStore(store=ObjectMemoryStore())
    )

    from scarf.storage import copy as copy_module

    def fail_copy(*args, **kwargs):
        raise OSError("injected metadata copy failure")

    monkeypatch.setattr(copy_module, "copy_zarr_group_tree", fail_copy)
    with pytest.raises(OSError, match="injected metadata copy failure"):
        create_matrix_source(source, target, workspace=None)
    if isinstance(target, str):
        assert not Path(target).exists()
    monkeypatch.undo()

    retried = create_matrix_source(source, target, workspace=None)
    assert MATRIX_SOURCE_ATTR in retried.attrs
    assert "ids" in retried["cellData"]


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"zarr_mode": "r"}, ValueError),
        ({"zarr_loc": "elsewhere.zarr"}, TypeError),
    ],
)
def test_mount_datastore_rejects_conflicting_options(tmp_path, options, error):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)

    with pytest.raises(error):
        mount_datastore(
            source,
            at=target,
            default_assay="RNA",
            min_features_per_cell=1,
            **options,
        )
    assert not Path(target).exists()


def test_mount_datastore_rejects_existing_target(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)
    mount_datastore(
        source,
        at=target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    with pytest.raises(FileExistsError):
        mount_datastore(
            source,
            at=target,
            default_assay="RNA",
            min_features_per_cell=1,
        )


def test_mounted_store_writes_only_to_target(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)
    source_before = _snapshot_store_files(source)

    ds = mount_datastore(
        source,
        at=target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    ds.cells.insert("mounted_flag", np.ones(ds.cells.N, dtype=bool), overwrite=True)
    mask = np.zeros(ds.cells.N, dtype=bool)
    mask[:3] = True
    ds.cells.update_key(mask, key="I")

    assert _snapshot_store_files(source) == source_before
    assert "mounted_flag" in zarr.open_group(target, mode="r")["cellData"]
    assert int(np.asarray(ds.cells.fetch_all("I")).sum()) == 3


def test_mount_copies_literal_feature_metadata_but_not_artifact_aliases(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)
    source_ds = DataStore(source, default_assay="RNA", min_features_per_cell=1)
    source_ds.RNA.feats.insert(
        "literal_flag",
        np.array([True, False, True, False]),
        overwrite=True,
    )
    selected = source_ds.set_feature_selection(
        mask=np.array([True, True, False, False]),
        label="selected_features",
    )
    source_features = source_ds.zw["RNA/featureData"]
    source_features["I"][:] = np.array([True, False, True, False])
    source_features["I"].attrs["source_artifact"] = selected.to_dict()
    source_features.create_array(
        "half_published",
        data=np.array([True, False, False, True]),
    )
    source_features.attrs["pending_feature_selection_aliases"] = {
        "half_published": selected.to_dict()
    }

    mounted = mount_datastore(source, at=target, default_assay="RNA")

    assert "literal_flag" in mounted.RNA.feats.columns
    np.testing.assert_array_equal(
        mounted.RNA.feats.fetch_all("literal_flag"),
        np.array([True, False, True, False]),
    )
    assert "all_features" not in mounted.RNA.feats.columns
    assert "selected_features" not in mounted.RNA.feats.columns
    assert "half_published" not in mounted.RNA.feats.columns
    np.testing.assert_array_equal(
        mounted.RNA.feats.fetch_all("I"),
        np.ones(mounted.RNA.feats.N, dtype=bool),
    )
    assert (
        "pending_feature_selection_aliases"
        not in mounted.RNA.feats.locations["primary"].attrs
    )

    created = mounted.set_feature_selection(
        mask=np.array([True, False, True, False]),
        label="mounted_features",
    )
    assert mounted.resolve_features("RNA", "mounted_features") == created


def test_mounted_store_loads_assays_written_to_the_target(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    values = np.arange(1, 41, dtype=np.uint32).reshape(10, 4)
    _write_source_store(source, workspace=None, values=values)
    source_before = _snapshot_store_files(source)

    mounted = mount_datastore(
        source,
        at=target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    mounted.RNA.feats.insert(
        "modules",
        np.array([0, 0, 1, 1]),
        overwrite=True,
    )
    mounted.add_grouped_assay(
        from_assay="RNA",
        group_key="modules",
        assay_label="MODULES",
    )

    assert mounted.MODULES.rawData.shape == (10, 2)
    np.testing.assert_array_equal(
        mounted.MODULES.feats.fetch_all("I"),
        np.ones(2, dtype=bool),
    )
    assert "MODULES" not in zarr.open_group(source, mode="r")

    reopened = DataStore(
        target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    assert reopened.RNA.rawData.shape == (10, 4)
    assert reopened.MODULES.rawData.shape == (10, 2)
    assert _snapshot_store_files(source) == source_before


def test_workspace_mounted_store_loads_assays_written_to_the_target(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    workspace = "analysis"
    values = np.arange(1, 41, dtype=np.uint32).reshape(10, 4)
    _write_source_store(source, workspace=workspace, values=values)
    source_before = _snapshot_store_files(source)

    mounted = mount_datastore(
        source,
        at=target,
        workspace=workspace,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    mounted.RNA.feats.insert(
        "modules",
        np.array([0, 0, 1, 1]),
        overwrite=True,
    )
    mounted.add_grouped_assay(
        from_assay="RNA",
        group_key="modules",
        assay_label="MODULES",
    )

    assert mounted.RNA.rawData.shape == (10, 4)
    assert mounted.MODULES.rawData.shape == (10, 2)
    np.testing.assert_array_equal(
        mounted.MODULES.feats.fetch_all("I"),
        np.ones(2, dtype=bool),
    )
    source_root = zarr.open_group(source, mode="r")
    assert "MODULES" not in source_root[workspace]
    assert "MODULES" not in source_root["matrices"]

    reopened = DataStore(
        target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    assert reopened.workspace == workspace
    assert reopened.RNA.rawData.shape == (10, 4)
    assert reopened.MODULES.rawData.shape == (10, 2)
    assert _snapshot_store_files(source) == source_before


def test_mount_rejects_malformed_pending_feature_alias_journal(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)
    source_root = zarr.open_group(source, mode="r+")
    source_root["RNA/featureData"].attrs["pending_feature_selection_aliases"] = [
        "not",
        "a",
        "mapping",
    ]

    with pytest.raises(
        ValueError,
        match="pending_feature_selection_aliases must be a mapping",
    ):
        create_matrix_source(source, target)
    assert not Path(target).exists()


def test_mounted_store_computes_markers_without_writing_source(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    values = np.array(
        [
            [4, 0, 1, 0],
            [3, 0, 1, 0],
            [0, 5, 1, 2],
            [0, 6, 1, 2],
        ],
        dtype=np.uint32,
    )
    _write_source_store(source, workspace=None, values=values)
    source_before = _snapshot_store_files(source)
    ds = mount_datastore(
        source,
        at=target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    ds.cells.insert(
        "marker_groups",
        np.array(["a", "a", "b", "b"]),
        overwrite=True,
    )

    markers = ds.run_marker_search(
        from_assay="RNA",
        cell_key="I",
        features=ds.set_feature_selection(
            mask=np.ones(ds.RNA.feats.N, dtype=bool),
            label="marker_features",
        ),
        group_key="marker_groups",
        nthreads=1,
        skip_save=True,
    )

    assert set(markers) == {"a", "b"}
    assert _snapshot_store_files(source) == source_before
    assert "markers" not in zarr.open_group(target, mode="r")["RNA"]


@pytest.mark.parametrize(
    ("group_path", "prefix"),
    [
        ("cellData", "cell"),
        ("RNA/featureData", "feature"),
    ],
)
def test_matrix_source_id_mismatch_fails_closed(tmp_path, group_path, prefix):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)
    create_matrix_source(source, target, workspace=None)

    source_root = zarr.open_group(source, mode="r+")
    group = source_root[group_path]
    n_rows = group["ids"].shape[0]
    create_zarr_obj_array(
        group,
        "ids",
        np.array([f"{prefix}-{i}" for i in range(n_rows)]),
        overwrite=True,
    )
    with pytest.raises(ValueError, match="cell or feature identifiers"):
        DataStore(target, default_assay="RNA")


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((9, 4), "uint32"),
        ((10, 4), "uint16"),
    ],
)
def test_matrix_source_count_identity_mismatch_fails_closed(
    tmp_path,
    shape,
    dtype,
):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)
    create_matrix_source(source, target, workspace=None)

    source_root = zarr.open_group(source, mode="r+")
    source_root["RNA"].create_array(
        "counts",
        data=np.zeros(shape, dtype=dtype),
        chunks=(5, 2),
        overwrite=True,
    )
    with pytest.raises(ValueError, match="count matrix identity"):
        DataStore(target, default_assay="RNA")


@pytest.mark.parametrize(
    ("parent_path", "name", "error_type"),
    [
        ("", "cellData", KeyError),
        ("RNA", "featureData", KeyError),
        ("RNA", "counts", ValueError),
    ],
)
def test_invalid_source_does_not_create_target(tmp_path, parent_path, name, error_type):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)
    source_root = zarr.open_group(source, mode="r+")
    parent = source_root if not parent_path else source_root[parent_path]
    del parent[name]

    with pytest.raises(error_type):
        create_matrix_source(source, target, workspace=None)
    assert not Path(target).exists()


def test_matrix_source_dataset_fingerprint_fast_path(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(
        source,
        workspace=None,
        dataset_fingerprint="abc123",
    )
    create_matrix_source(source, target, workspace=None)
    manifest = zarr.open_group(target, mode="r").attrs[MATRIX_SOURCE_ATTR]
    assert manifest["assays"]["RNA"]["datasetFingerprint"] == "abc123"
    assert manifest["assays"]["RNA"]["cellIdsFingerprint"] is None

    source_root = zarr.open_group(source, mode="r+")
    source_root["RNA"].attrs["dataset_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="dataset fingerprint"):
        resolve_matrix_source(zarr.open_group(target, mode="r"))


def test_dataset_fingerprint_fast_path_reads_no_identifiers(monkeypatch, tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None, dataset_fingerprint="abc123")
    create_matrix_source(source, target, workspace=None)

    read_array = zarr.Array.__getitem__

    def reject_identifier_reads(array, selection):
        if array.path.endswith(("cellData/ids", "featureData/ids")):
            raise AssertionError(f"Fast path read identifiers at {array.path}")
        return read_array(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", reject_identifier_reads)
    _, workspace = resolve_matrix_source(zarr.open_group(target, mode="r"))
    assert workspace is None


def test_source_open_does_not_change_target_profile(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace=None)

    create_matrix_source(source, target, workspace=None)
    target_ids = zarr.open_group(target, mode="r")["cellData/ids"]
    assert isinstance(target_ids.compressors[0], BloscCodec)


def test_mount_profile_applies_to_store_target(tmp_path):
    source = str(tmp_path / "source.zarr")
    _write_source_store(source, workspace=None)
    target_store = ObjectStore(store=ObjectMemoryStore())

    ds = mount_datastore(
        source,
        at=target_store,
        default_assay="RNA",
        min_features_per_cell=1,
        zarrProfile="cloud",
    )

    target_ids = ds.z["cellData/ids"]
    assert isinstance(target_ids.compressors[0], ZstdCodec)


def test_mounted_datastore_reads_remote_counts_and_persists_summary_locally(
    monkeypatch,
    tmp_path,
):
    values = np.arange(1, 41, dtype=np.uint32).reshape(10, 4)
    reference_path = str(tmp_path / "reference.zarr")
    _write_source_store(reference_path, workspace=None, values=values)
    remote_store = ObjectStore(store=ObjectMemoryStore())
    _write_source_store(remote_store, workspace=None, values=values)
    location = "s3://atlas/pbmc.zarr"

    from scarf.storage import stores as stores_module

    real_make_store = stores_module.make_store

    def fake_make_store(loc, *, storage_options=None, read_only=False):
        if loc == location:
            return remote_store
        return real_make_store(
            loc,
            storage_options=storage_options,
            read_only=read_only,
        )

    monkeypatch.setattr(stores_module, "make_store", fake_make_store)
    ds = mount_datastore(
        location,
        at=str(tmp_path / "target.zarr"),
        default_assay="RNA",
        min_features_per_cell=1,
    )
    assert is_remote_datastore(None, ds.RNA.rawData._backing) is True

    cell_idx = ds.cells.active_index("I")
    feat_idx = ds.RNA.feats.active_index("I")
    blocks = list(
        ds.RNA.iter_raw_column_blocks(
            cell_idx,
            feat_idx,
            batch_size=1,
        )
    )
    observed = np.concatenate([raw for _, raw, _, _, _ in blocks], axis=1)
    np.testing.assert_array_equal(observed, values[np.ix_(cell_idx, feat_idx)])

    selection = ds.select_detected_features(min_cells=1)
    selection_status = ds.inspect_artifact(selection)
    summary_ref = ArtifactRef.from_dict(selection_status.inputs["feature_summary"])
    summary = artifact_group(ds.zw, summary_ref)
    assert set(summary.array_keys()) == {"normed_tot", "normed_n", "sigmas"}
    assert summary.attrs["complete"] is True
    np.testing.assert_array_equal(
        np.asarray(summary["normed_n"][:]),
        (values > 0).sum(axis=0),
    )
    target_assay = zarr.open_group(str(tmp_path / "target.zarr"))["RNA"]
    assert not any(name.startswith("summary_stats_") for name in target_assay.keys())


def test_workspace_mismatch_raises(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _write_source_store(source, workspace="analysis")
    create_matrix_source(source, target, workspace="analysis")
    with pytest.raises(ValueError, match="workspace does not match"):
        DataStore(target, workspace="other", default_assay="RNA")


def test_mounted_store_normalization_and_graph(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    rng = np.random.default_rng(0)
    values = rng.integers(1, 20, size=(50, 30), dtype=np.uint32)
    _write_source_store(source, workspace=None, values=values)

    ds = mount_datastore(
        source,
        at=target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    # Guarantee enough selected features for IncrementalPCA(dims).
    feature_mask = np.zeros(ds.RNA.feats.N, dtype=bool)
    feature_mask[:12] = True
    features = ds.set_feature_selection(mask=feature_mask, label="hvgs")
    build_neighbourhood_graph(
        ds,
        features=features,
        k=3,
        dims=3,
        batch_size=25,
        local_cache=False,
    )
    state = ds.get_assay_state("RNA")
    assert state is not None
    assert state.connectivity_map is not None
    assert "counts" not in zarr.open_group(target, mode="r")["RNA"]


def test_mounted_store_build_mapping_reference(datastore_zarr_root, tmp_path):
    target = str(tmp_path / "mounted.zarr")
    source_before = _snapshot_store_files(datastore_zarr_root)
    ds = mount_datastore(
        datastore_zarr_root,
        at=target,
        default_assay="RNA",
    )
    features = ds.mark_hvgs(top_n=50, show_plot=False, bin_strategy="fixed")
    build_neighbourhood_graph(
        ds,
        features=features,
        k=3,
        dims=5,
        n_centroids=10,
    )
    state = ds.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors is not None
    reference = ds.build_mapping_reference(state.neighbors)
    assert reference.method == "pca"
    assert reference.dataset_fingerprint
    assert "counts" not in zarr.open_group(target, mode="r")["RNA"]
    assert _snapshot_store_files(datastore_zarr_root) == source_before
