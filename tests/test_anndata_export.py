import dataclasses
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scarf
import scarf.assay.base as assay_base
from scarf.assay.base import raw_csr
from scarf.datastore._operations import presentation as presentation_module
from scarf.datastore.datastore import DataStore
from scarf.datastore._operations.presentation import _PresentationOperationsMixin
from scarf.datastore.pipeline_run import PipelineRun
from scarf.storage.pipeline_runs import PipelineOutputRecord
from tests.storage_helpers import insert_nullable_cell_column, write_count_store
from tests.test_writers import _TOY_RNA_COUNTS


def _open_toy_copy(toy_crdir_writer, destination) -> DataStore:
    shutil.copytree(toy_crdir_writer, destination)
    return DataStore(
        str(destination),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )


@pytest.fixture
def export_store(toy_crdir_writer, tmp_path):
    """A copy of the toy store that a test may change."""
    return _open_toy_copy(toy_crdir_writer, tmp_path / "toy.zarr")


@pytest.fixture(scope="module")
def shared_export_store(toy_crdir_writer, tmp_path_factory):
    """A copy of the toy store for tests that only read it."""
    return _open_toy_copy(
        toy_crdir_writer, tmp_path_factory.mktemp("shared_export") / "toy.zarr"
    )


def test_to_anndata_requires_the_optional_anndata_dependency(
    shared_export_store,
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anndata", None)

    with pytest.raises(ImportError) as raised:
        shared_export_store.to_anndata()

    assert str(raised.value) == (
        "DataStore.to_anndata requires anndata. "
        "Install it with: pip install 'scarf[extra]'"
    )
    assert isinstance(raised.value.__cause__, ImportError)


class _RawMatrix:
    """Raw counts that a fake assay slices and streams in one block."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.dtype = self.values.dtype
        self.shape = self.values.shape

    def __getitem__(self, item):
        return _RawMatrix(self.values[item])

    def _stream_blocks(self, **_kwargs):
        # An export charges its conversion to CSR as resident bytes.
        assert _kwargs["resident_bytes"] > 0
        yield self.values


class _FrozenView:
    """A run axis view over a table whose ``I`` column selects its rows."""

    def __init__(self, frame):
        self._frame = frame
        self._selected = frame["I"].to_numpy(dtype=bool)
        self.columns = tuple(frame)

    def fetch_all(self, column):
        if column == "I":
            return self._selected.copy()
        return self._frame[column].to_numpy(copy=True)

    def _selected_field(self, column):
        return self.fetch_all(column)[self._selected], None


def _fake_run_store(monkeypatch, cell_frame: pd.DataFrame):
    """A presentation mixin and a fake completed run of three cells."""
    from scarf.datastore import pipeline_run as pipeline_run_module

    class FakePipelineRun:
        def __init__(self, owner):
            self._owner = owner
            self.assay = "RNA"
            self.cells = _FrozenView(cell_frame)
            self.features = _FrozenView(
                pd.DataFrame(
                    {
                        "I": [False, True],
                        "ids": ["g0", "g1"],
                        "names": ["frozen-g0", "frozen-g1"],
                    }
                )
            )

    store = _PresentationOperationsMixin()
    assay = SimpleNamespace(
        name="RNA",
        nthreads=1,
        rawData=_RawMatrix([[1, 2], [3, 4], [5, 6]]),
    )
    store._get_assay = lambda name: assay
    monkeypatch.setattr(pipeline_run_module, "PipelineRun", FakePipelineRun)
    return store, FakePipelineRun(store)


def test_to_anndata_materializes_frozen_pipeline_run_views(monkeypatch) -> None:
    store, run = _fake_run_store(
        monkeypatch,
        pd.DataFrame(
            {
                "I": [True, False, True],
                "ids": ["c0", "c1", "c2"],
                "names": ["frozen-a", "frozen-b", "frozen-c"],
                "clusters": [1, 2, 1],
            }
        ),
    )

    exported = store.to_anndata(run=run)

    np.testing.assert_array_equal(exported.X.toarray(), [[2], [6]])
    assert list(exported.obs_names) == ["c0", "c2"]
    assert exported.obs["names"].tolist() == ["frozen-a", "frozen-c"]
    assert exported.obs["clusters"].tolist() == [1, 1]
    assert "umap_1" not in exported.obs
    assert "X_umap" not in exported.obsm
    assert list(exported.var_names) == ["g1"]
    assert exported.var["names"].tolist() == ["frozen-g1"]

    with pytest.raises(ValueError, match="frozen run selection"):
        store.to_anndata(run=run, cell_key="I")
    with pytest.raises(ValueError, match="feature selection"):
        store.to_anndata(run=run, feature_indexes=[0])


def test_to_anndata_exports_empty_normed_cell_selection(export_store) -> None:
    export_store.cells.insert(
        "empty_export",
        np.zeros(export_store.cells.N, dtype=bool),
        overwrite=True,
    )

    adata = export_store.to_anndata(
        cell_key="empty_export",
        matrix="normed",
    )

    assert sparse.isspmatrix_csr(adata.X)
    assert adata.shape == (0, export_store.RNA.feats.N)
    assert adata.obs.empty


def test_to_anndata_exports_empty_raw_cell_selection(export_store) -> None:
    export_store.cells.insert(
        "empty_export",
        np.zeros(export_store.cells.N, dtype=bool),
        overwrite=True,
    )
    n_features = export_store.RNA.feats.N

    raw = raw_csr(export_store.RNA, export_store.cells.active_index("empty_export"))
    adata = export_store.to_anndata(
        cell_key="empty_export",
        layers={"raw": "RNA"},
    )
    subset = export_store.to_anndata(
        cell_key="empty_export",
        feature_indexes=[1, 0],
    )

    assert sparse.isspmatrix_csr(raw)
    assert raw.shape == (0, n_features)
    assert raw.dtype == export_store.RNA.rawData.dtype
    assert sparse.isspmatrix_csr(adata.X)
    assert adata.shape == (0, n_features)
    assert adata.layers["raw"].shape == (0, n_features)
    assert adata.obs.empty
    assert subset.shape == (0, 2)


def test_live_exports_write_masked_metadata_as_missing(export_store, tmp_path) -> None:
    anndata = pytest.importorskip("anndata")
    from scarf.writers import to_h5ad
    from tests.storage_helpers import insert_nullable_cell_column

    n_cells = export_store.cells.N
    missing = np.arange(n_cells) % 3 == 1
    columns = {
        "donor": np.where(missing, 0, np.arange(n_cells) % 2 + 1).astype(np.int64),
        "site": np.where(missing, "", np.where(np.arange(n_cells) % 2, "A", "B")),
        "flag": np.where(missing, False, np.arange(n_cells) % 2 == 0),
    }
    for name, values in columns.items():
        insert_nullable_cell_column(export_store, name, values, missing)
    active = export_store.cells.active_index("I")
    path = tmp_path / "live.h5ad"
    to_h5ad(export_store.RNA, str(path))

    for obs, rows in (
        (export_store.to_anndata().obs, active),
        (anndata.read_h5ad(path).obs, np.arange(n_cells)),
    ):
        for name, values in columns.items():
            present = ~missing[rows]
            np.testing.assert_array_equal(obs[name].isna().to_numpy(), ~present)
            assert obs[name].to_numpy()[present].tolist() == (
                values[rows][present].tolist()
            )
    assert anndata.read_h5ad(path).obs["flag"].dtype == "boolean"


def test_raw_feature_subset_matches_full_raw_columns(shared_export_store) -> None:
    adata = shared_export_store.to_anndata(feature_indexes=[3, 1])

    assert sparse.isspmatrix_csr(adata.X)
    assert list(adata.var_names) == ["g4", "g2"]
    np.testing.assert_array_equal(adata.X.toarray(), _TOY_RNA_COUNTS[:, [3, 1]])


def test_to_anndata_exports_normed_csr_with_ordered_feature_indexes(export_store):
    feature_indexes = np.array([3, 0])
    export_store.cells.insert("picked", np.array([True, False, True]), overwrite=True)

    adata = export_store.to_anndata(
        from_assay="RNA",
        cell_key="picked",
        matrix="normed",
        feature_indexes=feature_indexes,
    )

    # Library-size normalization scales each cell to 1,000 counts over all of
    # its RNA features, not only the exported ones.
    totals = _TOY_RNA_COUNTS.sum(axis=1, keepdims=True)
    expected = (_TOY_RNA_COUNTS * 1000 / totals)[[0, 2]][:, feature_indexes]
    assert sparse.isspmatrix_csr(adata.X)
    np.testing.assert_allclose(adata.X.toarray(), expected, rtol=1e-12)
    assert list(adata.var_names) == ["g4", "g1"]
    assert list(adata.obs_names) == ["b1", "b3"]


def test_to_anndata_selects_names_and_aligns_raw_layers(shared_export_store):
    all_names = shared_export_store.RNA.feats.fetch_all("names").astype(str)
    requested = [all_names[2], all_names[0]]

    adata = shared_export_store.to_anndata(
        from_assay="RNA",
        feature_names=requested,
        layers={"raw": "RNA"},
    )

    expected = _TOY_RNA_COUNTS[:, [2, 0]]
    assert sparse.isspmatrix_csr(adata.X)
    np.testing.assert_array_equal(adata.X.toarray(), expected)
    np.testing.assert_array_equal(adata.layers["raw"].toarray(), expected)
    assert list(adata.var["names"].astype(str)) == requested == ["g3", "g1"]


def test_to_anndata_preserves_legacy_layer_behavior_without_subset(shared_export_store):
    adata = shared_export_store.to_anndata(layers={"raw": "RNA"})

    assert sparse.isspmatrix_csr(adata.X)
    np.testing.assert_array_equal(adata.X.toarray(), _TOY_RNA_COUNTS)
    np.testing.assert_array_equal(adata.layers["raw"].toarray(), _TOY_RNA_COUNTS)
    assert list(adata.var_names) == ["g1", "g2", "g3", "g4"]


def test_to_anndata_aligns_reordered_layer_ids_without_subset(
    shared_export_store,
    monkeypatch,
):
    primary = shared_export_store.RNA
    primary_ids = primary.feats.fetch_all("ids").astype(str)
    order = np.array([2, 0, 3, 1])
    reordered_assay = SimpleNamespace(
        feats=SimpleNamespace(
            fetch_all=lambda column: primary_ids[order] if column == "ids" else None
        ),
        rawData=primary.rawData[:, order],
        nthreads=1,
        name="reordered",
    )
    original_get_assay = shared_export_store._get_assay

    def get_assay(name):
        if name == "reordered":
            return reordered_assay
        return original_get_assay(name)

    monkeypatch.setattr(shared_export_store, "_get_assay", get_assay)
    adata = shared_export_store.to_anndata(layers={"reordered": "reordered"})

    # The layer's columns are put back in the order of the primary features.
    np.testing.assert_array_equal(adata.layers["reordered"].toarray(), _TOY_RNA_COUNTS)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"feature_indexes": "0"},
            TypeError,
            "sequence of integer feature indexes",
        ),
        (
            {"feature_indexes": np.asarray([[0]])},
            ValueError,
            "one-dimensional",
        ),
        (
            {"feature_indexes": [0.5]},
            TypeError,
            "only integers",
        ),
        (
            {"feature_indexes": [-1]},
            IndexError,
            "out-of-range",
        ),
        (
            {"feature_names": "g1"},
            TypeError,
            "sequence of feature names",
        ),
        (
            {"feature_names": [1]},
            TypeError,
            "only strings",
        ),
    ],
)
def test_to_anndata_rejects_malformed_selector_types(
    shared_export_store,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        shared_export_store.to_anndata(**kwargs)


def test_to_anndata_rejects_invalid_or_ambiguous_selectors(export_store):
    names = export_store.RNA.feats.fetch_all("names").astype(str)

    with pytest.raises(ValueError, match="mutually exclusive"):
        export_store.to_anndata(feature_indexes=[0], feature_names=[names[0]])
    with pytest.raises(ValueError, match="unique indexes"):
        export_store.to_anndata(feature_indexes=[0, 0])
    with pytest.raises(IndexError, match="out-of-range"):
        export_store.to_anndata(feature_indexes=[export_store.RNA.feats.N])
    with pytest.raises(ValueError, match="unique names"):
        export_store.to_anndata(feature_names=[names[0], names[0]])
    with pytest.raises(KeyError, match="not found"):
        export_store.to_anndata(feature_names=["not-a-feature"])
    with pytest.raises(ValueError, match="matrix must"):
        export_store.to_anndata(matrix="scaled")

    names[1] = names[0]
    export_store.RNA.feats.insert("names", names, overwrite=True)
    with pytest.raises(ValueError, match="not unique"):
        export_store.to_anndata(feature_names=[names[0]])


def test_to_anndata_rejects_duplicate_primary_ids_when_exporting_layers(
    export_store,
) -> None:
    feature_ids = export_store.RNA.feats.fetch_all("ids").astype(str)
    feature_ids[1] = feature_ids[0]
    export_store.RNA.feats._get_array("ids")[:] = feature_ids

    # Layers are checked before any value is read or any AnnData is built.
    with pytest.raises(ValueError, match="Selected feature IDs must be unique"):
        export_store.to_anndata(
            feature_indexes=[0, 1],
            layers={"raw": "RNA"},
        )


def test_to_anndata_rejects_ambiguous_layer_feature_ids(
    shared_export_store,
    monkeypatch,
) -> None:
    primary = shared_export_store.RNA
    primary_ids = primary.feats.fetch_all("ids").astype(str)
    ambiguous_ids = primary_ids.copy()
    ambiguous_ids[1] = ambiguous_ids[0]
    ambiguous_assay = SimpleNamespace(
        feats=SimpleNamespace(
            fetch_all=lambda column: ambiguous_ids if column == "ids" else None
        ),
        rawData=primary.rawData,
        nthreads=1,
        name="ambiguous",
    )
    original_get_assay = shared_export_store._get_assay

    def get_assay(name):
        if name == "ambiguous":
            return ambiguous_assay
        return original_get_assay(name)

    monkeypatch.setattr(shared_export_store, "_get_assay", get_assay)

    with pytest.raises(ValueError, match="ambiguous"):
        shared_export_store.to_anndata(
            feature_indexes=[0],
            layers={"ambiguous": "ambiguous"},
        )


def test_to_anndata_rejects_unaligned_subset_layer(shared_export_store):
    columns_before = set(shared_export_store.cells.columns)
    artifacts_before = set(shared_export_store.list_artifacts())

    with pytest.raises(ValueError, match="cannot align selected feature IDs"):
        shared_export_store.to_anndata(
            feature_indexes=[0],
            layers={"adt": "ADT"},
        )

    assert set(shared_export_store.cells.columns) == columns_before
    assert set(shared_export_store.list_artifacts()) == artifacts_before


def test_to_anndata_run_export_lifts_consecutive_umap_into_obsm(monkeypatch) -> None:
    store, run = _fake_run_store(
        monkeypatch,
        pd.DataFrame(
            {
                "I": [True, False, True],
                "ids": ["c0", "c1", "c2"],
                "names": ["frozen-a", "frozen-b", "frozen-c"],
                "clusters": [1, 2, 1],
                "umap_1": [1.5, 9.0, 3.5],
                "umap_2": [10.5, 99.0, 30.5],
            }
        ),
    )
    exported = store.to_anndata(run=run)

    assert list(exported.obs_names) == ["c0", "c2"]
    assert exported.obs["clusters"].tolist() == [1, 1]
    assert "umap_1" not in exported.obs
    assert "umap_2" not in exported.obs
    np.testing.assert_allclose(
        exported.obsm["X_umap"],
        np.asarray([[1.5, 10.5], [3.5, 30.5]]),
    )


def test_to_anndata_run_export_rejects_gapped_umap_components(monkeypatch) -> None:
    store, run = _fake_run_store(
        monkeypatch,
        pd.DataFrame(
            {
                "I": [True, True],
                "ids": ["c0", "c1"],
                "names": ["a", "b"],
                "umap_1": [1.0, 2.0],
                "umap_3": [3.0, 4.0],
            }
        ),
    )
    with pytest.raises(ValueError, match="consecutively numbered"):
        store.to_anndata(run=run)


def test_to_anndata_run_export_ignores_non_positive_umap_suffixes(
    monkeypatch,
) -> None:
    store, run = _fake_run_store(
        monkeypatch,
        pd.DataFrame(
            {
                "I": [True, True],
                "ids": ["c0", "c1"],
                "names": ["a", "b"],
                "umap_0": [0.0, 1.0],
                "clusters": [1, 2],
            }
        ),
    )
    exported = store.to_anndata(run=run)

    assert "X_umap" not in exported.obsm
    assert "umap_0" in exported.obs
    assert exported.obs["clusters"].tolist() == [1, 2]


_RUN_CELLS = 120
_RUN_FEATURES = 40


def _exported_run_counts() -> np.ndarray:
    rng = np.random.default_rng(11)
    rates = rng.uniform(0.5, 4.0, size=_RUN_FEATURES)
    return rng.poisson(rates, size=(_RUN_CELLS, _RUN_FEATURES)).astype(np.uint32)


_EXPORTED_RUN_COUNTS = _exported_run_counts()
# Every eighth cell is left out of the run, so its cells are a strict subset.
_EXPORTED_RUN_CELLS = np.arange(_RUN_CELLS) % 8 != 5
_MISSING = np.arange(_RUN_CELLS) % 7 == 3
_SNAPSHOT_COLUMNS = {
    # Repeated text becomes a categorical whose categories are in natural
    # order: d2 before d10.
    "donor": np.asarray([f"d{index % 12}" for index in range(_RUN_CELLS)]),
    # natsort, which AnnData sorts categories with, reads a superscript digit
    # as a number; Scarf's natural order keeps it as text.
    "marker": np.asarray(
        [("a²", "a10", "a3")[index % 3] for index in range(_RUN_CELLS)]
    ),
    "barcode": np.asarray([f"u{index}" for index in range(_RUN_CELLS)]),
    "score": np.linspace(-1.0, 1.0, _RUN_CELLS),
    "count": np.arange(_RUN_CELLS, dtype=np.int64),
    "flag": np.arange(_RUN_CELLS) % 2 == 0,
}
# Missing rows hold placeholders; the boolean placeholder True is one that
# AnnData does not store.
_MASKED_COLUMNS = {
    "m_text": np.where(_MISSING, "", np.where(np.arange(_RUN_CELLS) % 2, "A", "B")),
    "m_int": np.where(_MISSING, 0, np.arange(_RUN_CELLS) % 5).astype(np.int64),
    "m_bool": np.where(_MISSING, True, np.arange(_RUN_CELLS) % 3 == 0),
}


@pytest.fixture(scope="module")
def exported_run(tmp_path_factory) -> tuple[DataStore, PipelineRun]:
    """A completed run with UMAP, clusters, and masked snapshot fields."""
    location = str(tmp_path_factory.mktemp("exported_run") / "data.zarr")
    write_count_store(location, {"RNA": _EXPORTED_RUN_COUNTS}, np.uint32)
    datastore = DataStore(
        location, default_assay="RNA", min_features_per_cell=1, nthreads=1
    )
    datastore.cells.insert("keep", _EXPORTED_RUN_CELLS, overwrite=True)
    for name, values in _SNAPSHOT_COLUMNS.items():
        datastore.cells.insert(name, values, overwrite=True)
    for name, values in _MASKED_COLUMNS.items():
        insert_nullable_cell_column(datastore, name, values, _MISSING)
    # Text missing in every cell, which AnnData cannot write as text.
    insert_nullable_cell_column(
        datastore,
        "m_none",
        np.full(_RUN_CELLS, "", dtype="<U1"),
        np.ones(_RUN_CELLS, dtype=bool),
    )
    datastore = DataStore(
        location, default_assay="RNA", min_features_per_cell=1, nthreads=1
    )
    run = datastore.pipeline.run(
        label="export",
        cell_key="keep",
        filtering=False,
        cell_cycle=False,
        paris=False,
        doublets=False,
        markers=False,
        hvg_count=12,
        pca_dims=3,
        neighbors_k=5,
        snapshot_columns=[*_SNAPSHOT_COLUMNS, *_MASKED_COLUMNS, "m_none"],
    )
    assert run.status == "completed"
    return datastore, run


def _forbid(*_args, **_kwargs):
    raise AssertionError("A run export must stream its matrix, not materialize it")


def _stored(node) -> tuple[str, str]:
    """Return a dataset's dtype and values, in a form where NaN equals NaN."""
    import h5py

    if h5py.check_string_dtype(node.dtype):
        return "text", repr(node.asstr()[()].tolist())
    return str(node.dtype), repr(node[()].tolist())


def _dataframe_elements(path, table: str) -> dict[str, object]:
    """Return the attributes and the stored elements of an ``obs`` or ``var``."""
    import h5py

    with h5py.File(path, "r") as h5:
        group = h5[table]
        elements: dict[str, object] = {
            "_index": group.attrs["_index"],
            "column-order": list(group.attrs["column-order"]),
        }
        for name, node in group.items():
            encoding = node.attrs["encoding-type"]
            if encoding == "nullable-string-array":
                # AnnData under pandas 3 stores distinct text so; it reads
                # back as the same strings as a string array.
                assert not node["mask"][()].any()
                elements[name] = ("string-array", _stored(node["values"]))
            elif isinstance(node, h5py.Group):
                elements[name] = (
                    encoding,
                    dict(node.attrs),
                    {key: _stored(child) for key, child in node.items()},
                )
            else:
                elements[name] = (encoding, _stored(node))
    return elements


def _text_plan(values, missing=None):
    """A one-gene run export plan whose cells hold one text field ``t``."""
    from scarf.writers.export import H5adColumn, H5adExportPlan, H5adMatrix

    values = np.asarray(values)
    mask = None if missing is None else np.asarray(missing, dtype=bool)
    n_cells = len(values)
    ids = np.asarray([f"c{index}" for index in range(n_cells)], dtype=str)

    def frozen(name, column, column_missing=None):
        return H5adColumn(name, lambda: (column, column_missing), "categorical")

    return H5adExportPlan(
        x=H5adMatrix(
            shape=(n_cells, 1),
            dtype=np.dtype(np.float32),
            blocks=lambda: iter([np.ones((n_cells, 1), dtype=np.float32)]),
        ),
        obs_index=frozen("ids", ids),
        obs=(frozen("t", values, mask),),
        var_index=frozen("gene_ids", np.asarray(["g0"])),
        var=(),
    )


@pytest.mark.parametrize(
    ("values", "missing", "categories"),
    [
        # Scarf's natural order keeps a superscript digit as text, where
        # natsort reads it as a number.
        (["a²", "a²", "a10", "a10", "a3", "a3"], None, ["a3", "a10", "a²"]),
        ([b"b", b"b", b"a", b"a"], None, ["a", "b"]),
        (["d10", "d2", "d10", "d2"], None, ["d2", "d10"]),
        # A label that starts with a number sorts before one that starts with
        # text, and accented text sorts by its decomposition.
        (["x", "10", "9", "x", "10", "9"], None, ["9", "10", "x"]),
        (["f", "é", "e", "f", "é", "e"], None, ["e", "é", "f"]),
        (["a", "b", "c", ""], [False, False, False, True], ["a", "b", "c"]),
        # Text missing in every row is a categorical without categories.
        (["", "", "", ""], [True, True, True, True], []),
    ],
)
def test_run_text_categoricals_agree_in_memory_and_in_both_files(
    tmp_path, values, missing, categories
) -> None:
    anndata = pytest.importorskip("anndata")
    from scarf.writers.export import write_h5ad_plan

    plan = _text_plan(values, missing)
    written = tmp_path / "scarf.h5ad"
    reference = tmp_path / "anndata.h5ad"

    write_h5ad_plan(plan, written)
    expected = presentation_module._anndata_from_plan(anndata.AnnData, plan)
    # The plan decides the categories, so to_anndata returns them, and both
    # files store them, whatever order AnnData would choose. AnnData's writer
    # changes text columns of the object that it writes, so it writes a copy.
    column = expected.obs["t"]
    expected.copy().write_h5ad(reference)

    assert isinstance(column.dtype, pd.CategoricalDtype)
    assert column.cat.categories.tolist() == categories
    np.testing.assert_array_equal(
        column.isna().to_numpy(),
        np.zeros(len(values), bool) if missing is None else missing,
    )
    assert _dataframe_elements(written, "obs") == _dataframe_elements(reference, "obs")
    for path in (written, reference):
        stored = anndata.read_h5ad(path).obs["t"]
        assert stored.cat.categories.tolist() == categories
        assert stored.astype(object).where(stored.notna(), None).tolist() == (
            column.astype(object).where(column.notna(), None).tolist()
        )


def test_run_text_that_neither_repeats_nor_is_missing_stays_text(tmp_path) -> None:
    anndata = pytest.importorskip("anndata")
    from scarf.writers.export import _categorical_column, write_h5ad_plan

    plan = _text_plan([b"a", b"b", b"c"])
    write_h5ad_plan(plan, tmp_path / "scarf.h5ad")
    expected = presentation_module._anndata_from_plan(anndata.AnnData, plan)

    assert not isinstance(expected.obs["t"].dtype, pd.CategoricalDtype)
    assert expected.obs["t"].tolist() == ["a", "b", "c"]
    assert anndata.read_h5ad(tmp_path / "scarf.h5ad").obs["t"].tolist() == [
        "a",
        "b",
        "c",
    ]
    # Repeated values that are neither text nor missing, such as dates, are
    # not categorical either.
    dates = np.asarray(["2020-01-01", "2020-01-01"], dtype="datetime64[D]")
    assert _categorical_column(dates, None) is None


@pytest.mark.slow
def test_run_h5ad_needs_no_anndata_and_matches_to_anndata(
    exported_run, tmp_path, monkeypatch
) -> None:
    anndata = pytest.importorskip("anndata")
    datastore, run = exported_run
    path = tmp_path / "run.h5ad"

    with monkeypatch.context() as patched:
        # Without AnnData, and without materializing the counts.
        patched.setitem(sys.modules, "anndata", None)
        patched.setattr(presentation_module, "raw_csr", _forbid)
        patched.setattr(assay_base, "raw_csr", _forbid)
        scarf.to_h5ad(datastore.RNA, str(path), run=run)

    expected = datastore.to_anndata(run=run)
    reference = tmp_path / "reference.h5ad"
    # AnnData's own encoding of the same object.
    expected.copy().write_h5ad(reference)
    written = anndata.read_h5ad(path)
    encoded = anndata.read_h5ad(reference)

    # The file stores obs and var as AnnData stores the same object.
    for table in ("obs", "var"):
        assert _dataframe_elements(path, table) == _dataframe_elements(reference, table)
    pd.testing.assert_frame_equal(written.obs, encoded.obs)
    pd.testing.assert_frame_equal(written.var, encoded.var)
    assert sparse.isspmatrix_csr(written.X)
    assert written.X.dtype == expected.X.dtype
    np.testing.assert_array_equal(written.X.toarray(), expected.X.toarray())
    assert set(written.obsm) == {"X_umap"}
    assert written.obsm["X_umap"].dtype == expected.obsm["X_umap"].dtype
    np.testing.assert_array_equal(written.obsm["X_umap"], expected.obsm["X_umap"])
    assert written.obs_names.tolist() == expected.obs_names.tolist()
    assert written.var_names.tolist() == expected.var_names.tolist()
    assert written.obs["donor"].cat.categories.tolist() == [
        f"d{index}" for index in range(12)
    ]
    # The run plan decides the categories, so the object that to_anndata
    # returns holds them as well.
    for name, categories in (
        ("donor", [f"d{index}" for index in range(12)]),
        ("marker", ["a3", "a10", "a²"]),
        ("m_none", []),
    ):
        assert expected.obs[name].cat.categories.tolist() == categories
        assert written.obs[name].cat.categories.tolist() == categories
        assert encoded.obs[name].cat.categories.tolist() == categories
    assert written.obs["m_none"].isna().all()
    present = ~_MISSING[_EXPORTED_RUN_CELLS]
    for name, values in _MASKED_COLUMNS.items():
        column = written.obs[name]
        np.testing.assert_array_equal(column.isna().to_numpy(), ~present)
        assert column.to_numpy()[present].tolist() == (
            values[_EXPORTED_RUN_CELLS][present].tolist()
        )
    # The run's own tables show the stored placeholders as missing too.
    frame = run.cells.to_pandas_dataframe(["ids", "m_text"])
    np.testing.assert_array_equal(frame["m_text"].isna(), ~present)


@pytest.mark.slow
def test_normed_run_export_is_the_normalized_artifact(exported_run) -> None:
    datastore, run = exported_run

    exported = datastore.to_anndata(run=run, matrix="normed", layers={"raw": "RNA"})

    stored = np.asarray(datastore.load_artifact(run["normalized"])["data"][:])
    hvg = np.flatnonzero(run.features.fetch_all("highly_variable_features"))
    cells = np.flatnonzero(run.cells.fetch_all("I"))
    feature_ids = datastore.RNA.feats.fetch_all("ids").astype(str)
    assert sparse.isspmatrix_csr(exported.X)
    assert exported.X.dtype == np.float32
    np.testing.assert_array_equal(exported.X.toarray(), stored)
    assert exported.var_names.tolist() == feature_ids[hvg].tolist()
    assert exported.var["highly_variable_features"].all()
    assert exported.obs_names.tolist() == run.cells.fetch("ids").astype(str).tolist()
    np.testing.assert_array_equal(
        exported.layers["raw"].toarray(), _EXPORTED_RUN_COUNTS[np.ix_(cells, hvg)]
    )


def _with_outputs(run: PipelineRun, **replaced) -> PipelineRun:
    """Open ``run``'s record with some outputs replaced, or removed by None."""
    outputs = tuple(
        PipelineOutputRecord(
            key=output.key, artifact=replaced.get(output.key, output.artifact)
        )
        for output in run._record.outputs
        if replaced.get(output.key, output.artifact) is not None
    )
    return PipelineRun(run._owner, dataclasses.replace(run._record, outputs=outputs))


@pytest.mark.slow
def test_normed_run_export_requires_the_runs_own_normalized_values(
    exported_run, tmp_path
) -> None:
    datastore, run = exported_run
    # Normalized values of other cells, and the run's feature universe in
    # place of its highly variable features.
    totals = datastore.cells.fetch_all("RNA_nCounts")
    other_cells = datastore.filter_cells(
        ["RNA_nCounts"],
        [float(np.median(totals))],
        [None],
        cell_selection=run["input_cell_selection"],
    )
    foreign = datastore.run_normalization(other_cells, run["highly_variable_features"])
    path = tmp_path / "rejected.h5ad"
    cases = {
        "does not cover the run's cells": _with_outputs(run, normalized=foreign),
        "does not cover the run's highly variable features": _with_outputs(
            run, highly_variable_features=run["feature_universe"]
        ),
    }
    for message, swapped in cases.items():
        expected = f"The normalized artifact of pipeline run {run.run_id} {message}"
        with pytest.raises(ValueError, match=expected):
            datastore.to_anndata(run=swapped, matrix="normed")
        with pytest.raises(ValueError, match=expected):
            scarf.to_h5ad(datastore.RNA, str(path), run=swapped, matrix="normed")
        assert not path.exists()
    # As many cells as the run but other ones: only the selection check sees it.
    datastore.cells.insert("shifted", np.roll(_EXPORTED_RUN_CELLS, 1), overwrite=True)
    try:
        same_size = datastore.run_normalization(
            datastore.snapshot_cell_selection("shifted"),
            run["highly_variable_features"],
        )
    finally:
        datastore.cells.drop("shifted")
    with pytest.raises(
        ValueError,
        match=f"The normalized artifact of pipeline run {run.run_id} does not cover "
        "the run's cells",
    ):
        datastore.to_anndata(
            run=_with_outputs(run, normalized=same_size), matrix="normed"
        )

    reduction = _with_outputs(run, normalized=run["pca"])
    with pytest.raises(ValueError, match="is not a normalized artifact of assay 'RNA'"):
        datastore.to_anndata(run=reduction, matrix="normed")
    without = _with_outputs(run, normalized=None)
    with pytest.raises(
        ValueError, match=f"Pipeline run {run.run_id} has no normalized"
    ):
        datastore.to_anndata(run=without, matrix="normed")
    # Raw counts do not need the normalized values.
    assert datastore.to_anndata(run=without).shape == (
        int(_EXPORTED_RUN_CELLS.sum()),
        _RUN_FEATURES,
    )
