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


def test_write_fixture_h5ad_is_readable_by_string_loader(tmp_path: Path) -> None:
    path = tmp_path / "1000.h5ad"
    artifact = write_fixture_h5ad(path, nRows=100, nColumns=50, seed=1)
    assert artifact.targetRows == 100
    assert path.is_file()
    with h5py.File(path, "r") as h5:
        names = _load_string_column(h5, "var/feature_name", expectedLength=50)
        assert names.shape == (50,)
        assert "MT-ND1" in set(names.tolist())


def test_prepare_fixture_datasets_writes_requested_sizes(tmp_path: Path) -> None:
    artifacts = prepare_fixture_datasets(tmp_path, targetRows=(10, 25), nColumns=20)
    assert [item.targetRows for item in artifacts] == [10, 25]
    assert (tmp_path / "10.h5ad").is_file()
    assert (tmp_path / "25.h5ad").is_file()
