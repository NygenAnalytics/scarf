import pcst_fast


def test_pcst_fast_sanity_on_numpy2() -> None:
    """PyPI wheels for pcst-fast 1.0.10 need a source build with pybind11>=2.11."""
    edges = [[0, 1], [1, 2], [2, 3]]
    prizes = [1.0, 1.0, 1.0, 1.0]
    costs = [0.8, 1.8, 2.8]
    nodes, edge_ids = pcst_fast.pcst_fast(edges, prizes, costs, -1, 1, "strong", 0)
    assert list(nodes) == [0, 1]
    assert list(edge_ids) == [0]
