from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from profiling.datasets import (
    _load_string_column,
    prepare_fixture_datasets,
    write_fixture_h5ad,
)


def test_load_string_column_reads_categorical_feature_name(tmp_path: Path) -> None:
    import anndata as ad
    import pandas as pd

    matrix = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    adata = ad.AnnData(matrix)
    adata.obs_names = ["c0", "c1"]
    adata.var_names = ["g0", "g1"]
    adata.var["feature_name"] = pd.Categorical(["GeneA", "GeneB"])
    path = tmp_path / "cat.h5ad"
    adata.write_h5ad(path)

    with h5py.File(path, "r") as h5:
        assert isinstance(h5["var/feature_name"], h5py.Group)
        names = _load_string_column(h5, "var/feature_name", expectedLength=2)
    assert list(names) == ["GeneA", "GeneB"]


def test_write_fixture_h5ad_writes_a_reproducible_anndata_csr_file(
    tmp_path: Path,
) -> None:
    import anndata as ad

    path = tmp_path / "1000.h5ad"
    artifact = write_fixture_h5ad(path, nRows=100, nColumns=60, seed=1)
    feature_names = ["MT-ND1" if i % 50 == 0 else f"GENE{i}" for i in range(60)]

    adata = ad.read_h5ad(path)
    matrix = adata.X
    assert adata.shape == (100, 60)
    assert adata.obs_names.tolist() == [f"cell-{i}" for i in range(100)]
    assert adata.var_names.tolist() == [f"ENSG{i:011d}" for i in range(60)]
    assert adata.var["feature_name"].tolist() == feature_names
    assert isinstance(matrix, csr_matrix)
    assert matrix.has_sorted_indices and matrix.has_canonical_format
    assert np.diff(matrix.indptr).min() >= 1
    np.testing.assert_array_equal(matrix.data, np.round(matrix.data))
    assert 1 <= matrix.data.min() and matrix.data.max() <= 19
    assert (artifact.targetRows, artifact.nColumns, artifact.nnz) == (
        100,
        60,
        matrix.nnz,
    )
    assert artifact.fileBytes == path.stat().st_size
    with h5py.File(path, "r") as h5:
        names = _load_string_column(h5, "var/feature_name", expectedLength=60)
    assert names.tolist() == feature_names

    # The seed fixes the matrix; another seed draws another one.
    again = write_fixture_h5ad(tmp_path / "again.h5ad", nRows=100, nColumns=60, seed=1)
    other = write_fixture_h5ad(tmp_path / "other.h5ad", nRows=100, nColumns=60, seed=2)
    assert (ad.read_h5ad(again.localPath).X != matrix).nnz == 0
    assert (ad.read_h5ad(other.localPath).X != matrix).nnz > 0


def test_prepare_fixture_datasets_writes_requested_sizes(tmp_path: Path) -> None:
    announced = []
    artifacts = prepare_fixture_datasets(
        tmp_path, targetRows=(10, 25), nColumns=20, onArtifact=announced.append
    )
    assert [item.targetRows for item in artifacts] == [10, 25]
    assert announced == list(artifacts)
    for artifact, rows in zip(artifacts, (10, 25), strict=True):
        assert artifact.localPath == tmp_path / f"{rows}.h5ad"
        with h5py.File(artifact.localPath, "r") as h5:
            assert h5["X"].attrs["shape"].tolist() == [rows, 20]
            assert h5["X/indptr"].shape == (rows + 1,)
            assert int(h5["X/indptr"][-1]) == artifact.nnz
