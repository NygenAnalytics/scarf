import inspect

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix


def test_export_knn_to_mtx(datastore, graph_artifacts, tmp_path):
    from scarf.embeddings.sgtsne import export_knn_to_mtx

    graph = datastore.load_graph(
        from_assay="RNA",
        cell_key="I",
        symmetric=False,
        upper_only=False,
    )
    fn = str(tmp_path / "test_export_mtx_from_graph.mtx")
    ret_val = export_knn_to_mtx(fn, graph)
    assert ret_val is None

    read_options = (
        {"spmatrix": True} if "spmatrix" in inspect.signature(mmread).parameters else {}
    )
    actual = csr_matrix(mmread(fn, **read_options))
    expected = graph.tocsr(copy=True)
    actual.sort_indices()
    expected.sort_indices()
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-7, atol=1e-8)
