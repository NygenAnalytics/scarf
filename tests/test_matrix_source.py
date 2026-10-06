import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from obstore.store import MemoryStore as ObjectMemoryStore
from zarr.abc.store import Store
from zarr.codecs import BloscCodec, ZstdCodec
from zarr.storage import ObjectStore

from scarf.datastore.datastore import DataStore, mount_datastore
from scarf.storage.artifacts import ArtifactRef, artifact_group, artifact_path
from scarf.storage.budget import ResourceBudget
from scarf.storage.sharding import write_counts_t
from scarf.storage.stores import (
    MATRIX_SOURCE_ATTR,
    create_matrix_source,
    is_remote_datastore,
    resolve_matrix_source,
)
from tests.fixtures_datastore import build_neighbourhood_graph
from scarf.storage.schema import create_cell_data
from scarf.writers import create_zarr_count_assay


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
    from tests.storage_helpers import finalize_test_counts

    finalize_test_counts(counts)
    matrix_group = (
        root[assay_name] if workspace is None else root[f"matrices/{assay_name}"]
    )
    write_counts_t(
        counts,
        matrix_group,
        resources=ResourceBudget(1024**2, 2),
    )
    from scarf.assay.classification import preset_assay_types
    from scarf.metadata import MetaData

    table = root["cellData"] if workspace is None else root[f"{workspace}/cellData"]
    instance = preset_assay_types().get(assay_name, preset_assay_types()["Assay"])(
        z=root,
        workspace=workspace,
        name=assay_name,
        cell_data=MetaData(table),
        nthreads=1,
    )
    instance.prepare({})
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


_DEFAULT_VALUES = np.arange(40, dtype=np.uint32).reshape(10, 4)


@pytest.fixture(scope="module")
def default_sources(tmp_path_factory) -> dict[str | None, Path]:
    """Prepared sources of the default counts, one per workspace layout."""
    directory = tmp_path_factory.mktemp("default_sources")
    sources: dict[str | None, Path] = {}
    for workspace in (None, "analysis"):
        path = directory / ("root.zarr" if workspace is None else f"{workspace}.zarr")
        _write_source_store(str(path), workspace=workspace)
        sources[workspace] = path
    return sources


def _copy_source(
    default_sources: dict[str | None, Path],
    location: str | Path,
    *,
    workspace: str | None = None,
) -> np.ndarray:
    """Copy a prepared default source to ``location`` and return its counts.

    A copy holds the same bytes as a freshly written source, so each test owns
    a source it may change without the cost of preparing one.
    """
    shutil.copytree(default_sources[workspace], location)
    return _DEFAULT_VALUES.copy()


@pytest.mark.parametrize("workspace", [None, "analysis"])
def test_mount_datastore_creates_and_reopens(default_sources, tmp_path, workspace):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    values = _copy_source(default_sources, source, workspace=workspace)

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


def test_mount_datastore_multiple_assays(default_sources, tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    rna_values = _copy_source(default_sources, source)
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
        manifest["RNA"]["datasetFingerprint"]
        == source_root["RNA"].attrs["dataset_fingerprint"]
    )
    assert "counts" not in zarr.open_group(target, mode="r")["ADT"]


def test_mount_datastore_does_not_copy_source_pipeline_runs(default_sources, tmp_path):
    source = str(tmp_path / "source_with_run.zarr")
    target = str(tmp_path / "target_without_run.zarr")
    _copy_source(default_sources, source)
    source_root = zarr.open_group(source, mode="r+")
    source_root.create_group(f"pipeline/runs/{'a' * 64}/stages")

    mount_datastore(
        source,
        at=target,
        default_assay="RNA",
        min_features_per_cell=1,
    )

    mounted_root = zarr.open_group(target, mode="r")
    assert "pipeline" in source_root
    assert "pipeline" not in mounted_root


def test_mounted_store_reopens_from_another_directory(
    default_sources, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    values = _copy_source(default_sources, tmp_path / "source.zarr")
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
    default_sources,
    monkeypatch,
    tmp_path,
    target_kind,
):
    source = str(tmp_path / "source.zarr")
    _copy_source(default_sources, source)
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
        create_matrix_source(
            source, target, required_transposes=frozenset({"RNA"}), workspace=None
        )
    if isinstance(target, str):
        assert not Path(target).exists()
    monkeypatch.undo()

    retried = create_matrix_source(
        source, target, required_transposes=frozenset({"RNA"}), workspace=None
    )
    assert MATRIX_SOURCE_ATTR in retried.attrs
    assert "ids" in retried["cellData"]


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        (
            {"zarr_mode": "r"},
            ValueError,
            "A mounted datastore is writable and needs zarr_mode 'r\\+', got 'r'",
        ),
        (
            {"zarr_loc": "elsewhere.zarr"},
            TypeError,
            "mount_datastore takes the target location through 'at'",
        ),
    ],
)
def test_mount_datastore_rejects_conflicting_options(
    default_sources, tmp_path, options, error, message
):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)

    with pytest.raises(error, match=message):
        mount_datastore(
            source,
            at=target,
            default_assay="RNA",
            min_features_per_cell=1,
            **options,
        )
    assert not Path(target).exists()


def test_mount_datastore_checks_its_options_before_creating_the_target(
    default_sources, monkeypatch, tmp_path
):
    # Before, options such as an unknown type raised only when the mounted
    # target was opened, and the target that was left behind made a retry fail
    # with Zarr's bare "already contains data".
    import scarf.datastore.datastore as datastore_module

    def create(*_args, **_kwargs):
        raise AssertionError("the target was created before the options were checked")

    source = str(tmp_path / "source.zarr")
    _copy_source(default_sources, source)
    monkeypatch.setattr(datastore_module, "create_matrix_source", create)
    target = str(tmp_path / "target.zarr")
    for options, error, message in (
        ({"assay_types": {"RNA": "rna"}}, ValueError, "assay_type 'rna' of assay"),
        ({"min_features_per_cell": "1"}, TypeError, "must be an integer"),
        ({"mem_budget": "lots"}, ValueError, "Invalid memory spec: 'lots'"),
        ({"nthread": 2}, TypeError, "unexpected keyword argument 'nthread'"),
    ):
        with pytest.raises(error, match=message):
            mount_datastore(source, at=target, **options)


def test_mount_datastore_transposes_follow_the_declared_types(monkeypatch, tmp_path):
    from scipy.sparse import csr_matrix

    import scarf.storage.stores as stores_module
    from scarf.writers import SparseToZarr

    def discard(*_args, **_kwargs):
        raise AssertionError("the target was created before the types were checked")

    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    SparseToZarr(
        csr_matrix(_DEFAULT_VALUES),
        source,
        [f"c{i}" for i in range(10)],
        [f"g{i}" for i in range(4)],
        assay_name="GEX",
        nthreads=1,
    ).dump()
    DataStore(source, min_features_per_cell=-1, nthreads=1)
    assert "countsT" not in zarr.open_group(source, mode="r")["GEX"]

    # Declaring GEX as RNA on the target needs the gene-major counts that the
    # generic source assay lacks, which is found before the target exists.
    with monkeypatch.context() as patch:
        patch.setattr(stores_module, "discard_mount_target", discard)
        with pytest.raises(ValueError, match="countsT"):
            mount_datastore(source, at=target, assay_types={"GEX": "RNA"})
    assert not Path(target).exists()

    mounted = mount_datastore(source, at=target, assay_types={"GEX": "CRISPR"})
    assert mounted.get_assay("GEX").assayType == "CRISPR"
    manifest = zarr.open_group(target, mode="r").attrs[MATRIX_SOURCE_ATTR]
    assert manifest["assays"]["GEX"]["requiresTranspose"] is False


def test_a_mount_whose_first_open_fails_leaves_no_target(
    default_sources, monkeypatch, tmp_path
):
    from scarf.datastore.base_datastore import BaseDataStore

    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)

    def fail(*_args, **_kwargs):
        raise OSError("injected failure of the first writable open")

    # Only the writable open of the new target filters cells.
    monkeypatch.setattr(BaseDataStore, "_filter_cells", fail)
    with pytest.raises(OSError, match="injected failure"):
        mount_datastore(source, at=target, min_features_per_cell=1)
    assert not Path(target).exists()
    monkeypatch.undo()

    mounted = mount_datastore(source, at=target, min_features_per_cell=1)
    assert mounted.assay_names == ["RNA"]


def test_mount_datastore_rejects_existing_target(default_sources, tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)
    mount_datastore(
        source,
        at=target,
        default_assay="RNA",
        min_features_per_cell=1,
    )
    with pytest.raises(FileExistsError, match="already contains data"):
        mount_datastore(
            source,
            at=target,
            default_assay="RNA",
            min_features_per_cell=1,
        )


def test_mount_rejects_overlap_chained_mounts_and_old_contracts(
    default_sources, tmp_path
):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    chained = str(tmp_path / "chained.zarr")
    required = frozenset({"RNA"})
    _copy_source(default_sources, source)
    with pytest.raises(ValueError, match="must not overlap"):
        create_matrix_source(source, f"{source}/nested", required_transposes=required)
    assert not Path(f"{source}/nested").exists()

    create_matrix_source(source, target, required_transposes=required)
    with pytest.raises(ValueError, match="repacking"):
        create_matrix_source(target, chained, required_transposes=required)
    assert not Path(chained).exists()

    root = zarr.open_group(target, mode="r+")
    manifest = dict(root.attrs[MATRIX_SOURCE_ATTR])
    assert set(manifest) == {"location", "workspace", "assays"}
    # The top level is exact, so a field a later release adds fails closed.
    for changed in (
        {**manifest, "artifacts": "read_only"},
        {key: value for key, value in manifest.items() if key != "workspace"},
    ):
        root.attrs[MATRIX_SOURCE_ATTR] = changed
        with pytest.raises(
            ValueError,
            match="unsupported matrix source contract; create a fresh target "
            "with mount_datastore$",
        ):
            resolve_matrix_source(zarr.open_group(target, mode="r"))
        with pytest.raises(ValueError, match="unsupported matrix source contract"):
            DataStore(target, default_assay="RNA")
    fingerprint = manifest["assays"]["RNA"]["datasetFingerprint"]
    root.attrs[MATRIX_SOURCE_ATTR] = {
        **manifest,
        "assays": {"RNA": {"datasetFingerprint": fingerprint}},
    }
    with pytest.raises(
        ValueError,
        match="unsupported identity contract for assay 'RNA'; create a fresh "
        "target with mount_datastore$",
    ):
        resolve_matrix_source(zarr.open_group(target, mode="r"))


def test_mounted_store_writes_only_to_target(default_sources, tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)
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


def test_mount_copies_literal_feature_metadata_and_resets_feature_selection(
    default_sources,
    tmp_path,
):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)
    source_ds = DataStore(source, default_assay="RNA", min_features_per_cell=1)
    source_ds.RNA.feats.insert(
        "literal_flag",
        np.array([True, False, True, False]),
        overwrite=True,
    )
    selected = source_ds.set_feature_selection(
        mask=np.array([True, True, False, False]),
    )
    source_features = source_ds.zw["RNA/featureData"]
    source_features["I"][:] = np.array([True, False, True, False])

    mounted = mount_datastore(source, at=target, default_assay="RNA")

    assert "literal_flag" in mounted.RNA.feats.columns
    np.testing.assert_array_equal(
        mounted.RNA.feats.fetch_all("literal_flag"),
        np.array([True, False, True, False]),
    )
    assert "all_features" not in mounted.RNA.feats.columns
    assert "selected_features" not in mounted.RNA.feats.columns
    np.testing.assert_array_equal(
        mounted.RNA.feats.fetch_all("I"),
        np.ones(mounted.RNA.feats.N, dtype=bool),
    )
    assert selected.kind == "feature_selection"

    created = mounted.set_feature_selection(
        mask=np.array([True, False, True, False]),
    )
    assert mounted.resolve_features("RNA", created) == created


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
        "modules",
        from_assay="RNA",
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
        "modules",
        from_assay="RNA",
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

    clusters = ds.snapshot_cluster_labels(
        "marker_groups", cell_selection=ds.snapshot_cell_selection()
    )
    markers = ds.run_marker_search(
        clusters,
        from_assay="RNA",
        features=ds.set_feature_selection(
            mask=np.ones(ds.RNA.feats.N, dtype=bool),
        ),
        nthreads=1,
    )

    assert markers.kind == "marker_table"
    assert ds.inspect_artifact(markers).complete
    assert set(ds.get_markers(markers)["group_id"]) == {"a", "b"}
    assert _snapshot_store_files(source) == source_before
    assert "markers" not in zarr.open_group(target, mode="r")["RNA"]


@pytest.mark.parametrize(
    ("group_path", "prefix"),
    [
        ("cellData", "cell"),
        ("RNA/featureData", "feature"),
    ],
)
def test_matrix_source_id_mismatch_fails_closed(
    default_sources, tmp_path, group_path, prefix
):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)
    create_matrix_source(
        source, target, required_transposes=frozenset({"RNA"}), workspace=None
    )

    source_root = zarr.open_group(source, mode="r+")
    group = source_root[group_path]
    n_rows = group["ids"].shape[0]
    from scarf.metadata import MetaData

    with pytest.raises(ValueError, match="prepared data"):
        MetaData(group).insert(
            "ids",
            np.array([f"{prefix}-{i}" for i in range(n_rows)]),
            overwrite=True,
            force=True,
        )
    source_root["RNA"].attrs["dataset_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="mounted identity"):
        DataStore(target, default_assay="RNA")


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((9, 4), "uint32"),
        ((10, 4), "uint16"),
    ],
)
def test_matrix_source_count_identity_mismatch_fails_closed(
    default_sources,
    tmp_path,
    shape,
    dtype,
):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)
    create_matrix_source(
        source, target, required_transposes=frozenset({"RNA"}), workspace=None
    )

    source_root = zarr.open_group(source, mode="r+")
    source_root["RNA"].create_array(
        "counts",
        data=np.zeros(shape, dtype=dtype),
        chunks=(5, 2),
        overwrite=True,
    )
    with pytest.raises(ValueError, match="not finalized"):
        DataStore(target, default_assay="RNA")


@pytest.mark.parametrize(
    ("parent_path", "name", "error_type", "message"),
    [
        ("", "cellData", KeyError, "'cellData'"),
        ("RNA", "featureData", KeyError, "'featureData'"),
        ("RNA", "counts", ValueError, "Raw counts are missing"),
    ],
)
def test_invalid_source_does_not_create_target(
    default_sources, tmp_path, parent_path, name, error_type, message
):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)
    source_root = zarr.open_group(source, mode="r+")
    parent = source_root if not parent_path else source_root[parent_path]
    del parent[name]

    with pytest.raises(error_type, match=message):
        create_matrix_source(
            source, target, required_transposes=frozenset({"RNA"}), workspace=None
        )
    assert not Path(target).exists()


def test_matrix_source_dataset_fingerprint_fast_path(default_sources, tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)
    create_matrix_source(
        source, target, required_transposes=frozenset({"RNA"}), workspace=None
    )
    manifest = zarr.open_group(target, mode="r").attrs[MATRIX_SOURCE_ATTR]
    assert (
        manifest["assays"]["RNA"]["datasetFingerprint"]
        == zarr.open_group(source, mode="r")["RNA"].attrs["dataset_fingerprint"]
    )
    assert manifest["assays"]["RNA"]["countsFingerprint"]

    source_root = zarr.open_group(source, mode="r+")
    source_root["RNA"].attrs["dataset_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="mounted identity"):
        resolve_matrix_source(zarr.open_group(target, mode="r"))


def test_dataset_fingerprint_fast_path_reads_no_identifiers(
    default_sources, monkeypatch, tmp_path
):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)
    create_matrix_source(
        source, target, required_transposes=frozenset({"RNA"}), workspace=None
    )

    read_array = zarr.Array.__getitem__

    def reject_identifier_reads(array, selection):
        if array.path.endswith(("cellData/ids", "featureData/ids")):
            raise AssertionError(f"Fast path read identifiers at {array.path}")
        return read_array(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", reject_identifier_reads)
    _, workspace = resolve_matrix_source(zarr.open_group(target, mode="r"))
    assert workspace is None


def test_source_open_does_not_change_target_profile(default_sources, tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source)

    create_matrix_source(
        source, target, required_transposes=frozenset({"RNA"}), workspace=None
    )
    target_ids = zarr.open_group(target, mode="r")["cellData/ids"]
    assert isinstance(target_ids.compressors[0], BloscCodec)


def test_mount_profile_applies_to_store_target(default_sources, tmp_path):
    source = str(tmp_path / "source.zarr")
    _copy_source(default_sources, source)
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
    counts_t = ds.RNA.rawDataT
    assert counts_t is not None
    observed = np.asarray(counts_t.get_orthogonal_selection((feat_idx, cell_idx))).T
    np.testing.assert_array_equal(observed, values[np.ix_(cell_idx, feat_idx)])

    selection = ds.select_detected_features(
        ds.snapshot_cell_selection(),
        min_cells=1,
    )
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


def test_workspace_mismatch_raises(default_sources, tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "target.zarr")
    _copy_source(default_sources, source, workspace="analysis")
    create_matrix_source(
        source, target, required_transposes=frozenset({"RNA"}), workspace="analysis"
    )
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
    features = ds.set_feature_selection(mask=feature_mask)
    graph = build_neighbourhood_graph(
        ds,
        features=features,
        k=3,
        dims=3,
        batch_size=25,
        local_cache=False,
    )
    assert ds.inspect_artifact(graph).complete
    assert "counts" not in zarr.open_group(target, mode="r")["RNA"]


def test_mounted_store_build_mapping_reference(tmp_path):
    source = str(tmp_path / "source.zarr")
    target = str(tmp_path / "mounted.zarr")
    # Three groups of 40 cells, each with 20 highly expressed genes of its own.
    groups = np.arange(120) % 3
    rates = np.ones((3, 200))
    for group in range(3):
        rates[group, group * 20 : group * 20 + 20] = 6.0
    values = np.random.default_rng(0).poisson(rates[groups]).astype(np.uint32)
    _write_source_store(source, workspace=None, values=values)
    source_before = _snapshot_store_files(source)
    ds = mount_datastore(
        source,
        at=target,
        default_assay="RNA",
    )
    features = ds.select_hvgs(
        ds.snapshot_cell_selection(),
        top_n=50,
        show_plot=False,
        bin_strategy="fixed",
    )
    graph = build_neighbourhood_graph(
        ds,
        features=features,
        k=3,
        dims=5,
        n_centroids=10,
    )
    neighbors = ArtifactRef.from_dict(ds.inspect_artifact(graph).inputs["neighbors"])
    reference_ref = ds.build_mapping_reference(neighbors)
    reference = ds.get_mapping_reference(reference_ref)
    assert reference.method == "pca"
    # The reference binds the source's identity and is written to the target.
    assert (
        reference.dataset_fingerprint
        == zarr.open_group(source, mode="r")["RNA"].attrs["dataset_fingerprint"]
    )
    assert (Path(target) / artifact_path(reference_ref)).is_dir()
    assert "counts" not in zarr.open_group(target, mode="r")["RNA"]
    assert _snapshot_store_files(source) == source_before
