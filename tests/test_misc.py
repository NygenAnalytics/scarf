import numpy as np
from scipy.io import mmread


def test_export_knn_to_mtx(datastore, make_graph, tmp_path):
    from scarf.embeddings.sgtsne import export_knn_to_mtx

    graph = datastore.load_graph(
        from_assay="RNA",
        cell_key="I",
        feat_key="hvgs",
        symmetric=False,
        upper_only=False,
    )
    fn = str(tmp_path / "test_export_mtx_from_graph.mtx")
    ret_val = export_knn_to_mtx(fn, graph)
    assert ret_val is None

    actual = mmread(fn, spmatrix=True).tocsr()
    expected = graph.tocsr(copy=True)
    actual.sort_indices()
    expected.sort_indices()
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.data, expected.data, rtol=0, atol=1e-12)
