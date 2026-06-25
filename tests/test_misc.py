def test_export_knn_to_mtx(datastore, make_graph, tmp_path):
    from scarf.knn_utils import export_knn_to_mtx

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
