import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import scarf.datastore._operations.presentation as presentation_operations
from scarf.datastore.datastore import DataStore


@pytest.fixture
def export_store(toy_crdir_writer, tmp_path):
    destination = tmp_path / "toy.zarr"
    shutil.copytree(toy_crdir_writer, destination)
    return DataStore(
        str(destination),
        default_assay="RNA",
        min_features_per_cell=0,
        min_cells_per_feature=0,
        nthreads=1,
    )


def test_to_anndata_handles_missing_optional_dependency(
    export_store,
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

    assert export_store.to_anndata() is None
    assert len(messages) == 1
    assert "anndata is not installed" in messages[0]
    assert "optional dependency" in messages[0]


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


def test_to_anndata_exports_normed_csr_with_ordered_feature_indexes(export_store):
    feature_indexes = np.array([3, 0])
    cell_indexes = export_store.cells.active_index("I")

    adata = export_store.to_anndata(
        from_assay="RNA",
        cell_key="I",
        matrix="normed",
        feature_indexes=feature_indexes,
    )

    expected = export_store.RNA.normed(
        cell_idx=cell_indexes,
        feat_idx=feature_indexes,
    ).compute()
    assert sparse.isspmatrix_csr(adata.X)
    np.testing.assert_allclose(adata.X.toarray(), expected)
    assert list(adata.var_names) == ["g4", "g1"]
    assert list(adata.obs_names) == list(export_store.cells.fetch("ids", key="I"))


def test_to_anndata_selects_names_and_aligns_raw_layers(export_store):
    all_names = export_store.RNA.feats.fetch_all("names").astype(str)
    requested = [all_names[2], all_names[0]]

    adata = export_store.to_anndata(
        from_assay="RNA",
        feature_names=requested,
        layers={"raw": "RNA"},
    )

    expected = export_store.RNA.to_raw_sparse("I")[:, [2, 0]]
    assert sparse.isspmatrix_csr(adata.X)
    np.testing.assert_array_equal(adata.X.toarray(), expected.toarray())
    np.testing.assert_array_equal(adata.layers["raw"].toarray(), expected.toarray())
    assert list(adata.var["names"].astype(str)) == requested


def test_to_anndata_preserves_legacy_layer_behavior_without_subset(export_store):
    adata = export_store.to_anndata(layers={"raw": "RNA"})

    assert sparse.isspmatrix_csr(adata.X)
    np.testing.assert_array_equal(
        adata.layers["raw"].toarray(),
        adata.X.toarray(),
    )
    assert adata.n_vars == export_store.RNA.feats.N


def test_to_anndata_aligns_reordered_layer_ids_without_subset(
    export_store,
    monkeypatch,
):
    primary = export_store.RNA
    primary_ids = primary.feats.fetch_all("ids").astype(str)
    order = np.array([2, 0, 3, 1])
    reordered_matrix = primary.to_raw_sparse("I")[:, order]
    reordered_assay = SimpleNamespace(
        feats=SimpleNamespace(
            fetch_all=lambda column: primary_ids[order] if column == "ids" else None
        ),
        to_raw_sparse=lambda cell_key: reordered_matrix,
    )
    original_get_assay = export_store._get_assay

    def get_assay(name):
        if name == "reordered":
            return reordered_assay
        return original_get_assay(name)

    monkeypatch.setattr(export_store, "_get_assay", get_assay)
    adata = export_store.to_anndata(layers={"reordered": "reordered"})

    np.testing.assert_array_equal(
        adata.layers["reordered"].toarray(),
        primary.to_raw_sparse("I").toarray(),
    )


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
    export_store,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        export_store.to_anndata(**kwargs)


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
    export_store.RNA.feats.insert("ids", feature_ids, overwrite=True, force=True)

    with pytest.warns(UserWarning, match="Variable names are not unique"):
        with pytest.raises(ValueError, match="Selected feature IDs must be unique"):
            export_store.to_anndata(
                feature_indexes=[0, 1],
                layers={"raw": "RNA"},
            )


def test_to_anndata_rejects_ambiguous_layer_feature_ids(
    export_store,
    monkeypatch,
) -> None:
    primary = export_store.RNA
    primary_ids = primary.feats.fetch_all("ids").astype(str)
    ambiguous_ids = primary_ids.copy()
    ambiguous_ids[1] = ambiguous_ids[0]
    ambiguous_assay = SimpleNamespace(
        feats=SimpleNamespace(
            fetch_all=lambda column: ambiguous_ids if column == "ids" else None
        ),
        to_raw_sparse=lambda cell_key: primary.to_raw_sparse(cell_key),
    )
    original_get_assay = export_store._get_assay

    def get_assay(name):
        if name == "ambiguous":
            return ambiguous_assay
        return original_get_assay(name)

    monkeypatch.setattr(export_store, "_get_assay", get_assay)

    with pytest.raises(ValueError, match="ambiguous"):
        export_store.to_anndata(
            feature_indexes=[0],
            layers={"ambiguous": "ambiguous"},
        )


def test_to_anndata_rejects_unaligned_subset_layer(export_store):
    columns_before = set(export_store.cells.columns)
    artifacts_before = set(export_store.list_artifacts())

    with pytest.raises(ValueError, match="cannot align selected feature IDs"):
        export_store.to_anndata(
            feature_indexes=[0],
            layers={"adt": "ADT"},
        )

    assert set(export_store.cells.columns) == columns_before
    assert set(export_store.list_artifacts()) == artifacts_before
