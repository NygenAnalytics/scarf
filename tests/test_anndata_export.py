import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scarf.datastore._operations.presentation as presentation_operations
from scarf.assay.base import raw_csr
from scarf.datastore.datastore import DataStore
from scarf.datastore._operations.presentation import _PresentationOperationsMixin
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


def test_to_anndata_handles_missing_optional_dependency(
    shared_export_store,
    monkeypatch,
) -> None:
    messages: list[str] = []

    class RecordingLogger:
        def error(self, message: str) -> None:
            messages.append(message)

    monkeypatch.setitem(sys.modules, "anndata", None)
    monkeypatch.setattr(
        presentation_operations,
        "logger",
        RecordingLogger(),
    )

    assert shared_export_store.to_anndata() is None
    assert len(messages) == 1
    assert "anndata is not installed" in messages[0]
    assert "optional dependency" in messages[0]


def test_to_anndata_materializes_frozen_pipeline_run_views(monkeypatch) -> None:
    from scarf.datastore import pipeline_run as pipeline_run_module

    class RawMatrix:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.dtype = self.values.dtype

        def __getitem__(self, item):
            return RawMatrix(self.values[item])

        def stream_blocks(self, **_kwargs):
            yield self.values

    class FrozenView:
        def __init__(self, frame, selected):
            self._frame = frame
            self._selected = np.asarray(selected, dtype=bool)
            self.columns = tuple(frame)

        def fetch_all(self, column):
            if column == "I":
                return self._selected.copy()
            return self._frame[column].to_numpy(copy=True)

        def fetch(self, column):
            return self.fetch_all(column)[self._selected]

        def to_pandas_dataframe(self, columns):
            return self._frame.loc[self._selected, list(columns)].reset_index(drop=True)

    class FakePipelineRun:
        def __init__(self, owner):
            self._owner = owner
            self.assay = "RNA"
            self.cells = FrozenView(
                pd.DataFrame(
                    {
                        "I": [True, False, True],
                        "ids": ["c0", "c1", "c2"],
                        "names": ["frozen-a", "frozen-b", "frozen-c"],
                        "clusters": [1, 2, 1],
                    }
                ),
                [True, False, True],
            )
            self.features = FrozenView(
                pd.DataFrame(
                    {
                        "I": [False, True],
                        "ids": ["g0", "g1"],
                        "names": ["frozen-g0", "frozen-g1"],
                    }
                ),
                [False, True],
            )

    store = _PresentationOperationsMixin()
    assay = SimpleNamespace(
        name="RNA",
        nthreads=1,
        rawData=RawMatrix([[1, 2], [3, 4], [5, 6]]),
    )
    store._get_assay = lambda name: assay
    monkeypatch.setattr(pipeline_run_module, "PipelineRun", FakePipelineRun)
    run = FakePipelineRun(store)

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

    with pytest.warns(UserWarning, match="Variable names are not unique"):
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


def _run_export_store(monkeypatch, cell_frame: pd.DataFrame):
    from scarf.datastore import pipeline_run as pipeline_run_module

    class RawMatrix:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.dtype = self.values.dtype

        def __getitem__(self, item):
            return RawMatrix(self.values[item])

        def stream_blocks(self, **_kwargs):
            yield self.values

    class FrozenView:
        def __init__(self, frame, selected):
            self._frame = frame
            self._selected = np.asarray(selected, dtype=bool)
            self.columns = tuple(frame)

        def fetch_all(self, column):
            if column == "I":
                return self._selected.copy()
            return self._frame[column].to_numpy(copy=True)

        def fetch(self, column):
            return self.fetch_all(column)[self._selected]

        def to_pandas_dataframe(self, columns):
            return self._frame.loc[self._selected, list(columns)].reset_index(drop=True)

    selected = cell_frame["I"].to_numpy(dtype=bool)

    class FakePipelineRun:
        def __init__(self, owner):
            self._owner = owner
            self.assay = "RNA"
            self.cells = FrozenView(cell_frame, selected)
            self.features = FrozenView(
                pd.DataFrame(
                    {
                        "I": [False, True],
                        "ids": ["g0", "g1"],
                        "names": ["frozen-g0", "frozen-g1"],
                    }
                ),
                [False, True],
            )

    store = _PresentationOperationsMixin()
    assay = SimpleNamespace(
        name="RNA",
        nthreads=1,
        rawData=RawMatrix([[1, 2], [3, 4], [5, 6]]),
    )
    store._get_assay = lambda name: assay
    monkeypatch.setattr(pipeline_run_module, "PipelineRun", FakePipelineRun)
    return store, FakePipelineRun(store)


def test_to_anndata_run_export_lifts_consecutive_umap_into_obsm(monkeypatch) -> None:
    store, run = _run_export_store(
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
    store, run = _run_export_store(
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
    store, run = _run_export_store(
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
