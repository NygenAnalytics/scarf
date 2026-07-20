import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

from profiling.config import WorkflowParameters
from profiling.stages import _run_native_igraph_leiden
from scarf.clustering.leiden import leiden_membership


class _Cells:
    def __init__(self) -> None:
        self.name: str | None = None
        self.values: np.ndarray | None = None

    def insert(
        self,
        name: str,
        values: np.ndarray,
        *,
        fill_value: int,
        key: str,
        overwrite: bool,
    ) -> None:
        assert fill_value == -1
        assert key == "I"
        assert overwrite is True
        self.name = name
        self.values = values


class _Store:
    def __init__(self, graph: csr_matrix) -> None:
        self.graph = graph
        self.cells = _Cells()

    def load_graph(self, **_: object) -> csr_matrix:
        return self.graph


def _two_cliques() -> csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for start in (0, 8):
        for source in range(start, start + 8):
            for target in range(start, start + 8):
                if source != target:
                    rows.append(source)
                    columns.append(target)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, columns)), shape=(16, 16))


def test_native_igraph_leiden_matches_historical_partition() -> None:
    graph = _two_cliques()
    expected = leiden_membership(graph, resolution=1.0, random_seed=4444)
    store = _Store(graph)

    _run_native_igraph_leiden(store, WorkflowParameters())

    assert store.cells.name == "RNA_leiden_cluster"
    assert store.cells.values is not None
    assert adjusted_rand_score(expected, store.cells.values) == 1.0
