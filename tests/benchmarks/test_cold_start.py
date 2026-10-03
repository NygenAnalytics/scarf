"""Benchmark of the fixed cost every new process pays before its first stage.

A pipeline worker that builds a graph, embeds it, and clusters it first
imports the libraries those stages use and compiles the Numba kernels that
are not cached on disk. Each process pays this again, so it adds to every
stage that runs in a fresh process and to every test worker. The benchmark
times a new interpreter running the smallest such pipeline.
"""

import subprocess
import sys

import pytest

from .harness import REPOSITORY_ROOT, Ladder, Thresholds

pytestmark = pytest.mark.benchmark

COLD_PIPELINE = """
import numpy as np
from scipy import sparse

from scarf.clustering.leiden import leiden_membership
from scarf.embeddings.umap import fit_transform
from scarf.neighbors.graph import build_connectivity_arrays

rng = np.random.default_rng(0)
n_cells, k = 300, 11
indices = np.stack(
    [rng.choice(np.delete(np.arange(n_cells), cell), k, replace=False)
     for cell in range(n_cells)]
).astype(np.int64)
distances = np.sort(rng.random((n_cells, k)).astype(np.float32), axis=1)
edges, weights = build_connectivity_arrays(
    indices, distances, local_connectivity=1.0, bandwidth=1.5
)
graph = sparse.csr_matrix(
    (weights, (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells)
)
embedding, _a, _b = fit_transform(
    graph.tocoo(), rng.random((n_cells, 2)).astype(np.float32), spread=2.0,
    min_dist=1.0, n_epochs=5, random_seed=1, repulsion_strength=1.0,
    initial_alpha=1.0, negative_sample_rate=5, densmap_kwds={}, parallel=False,
    nthreads=1, verbose=False,
)
labels = leiden_membership(graph, 1.0, 1)
assert embedding.shape == (n_cells, 2) and np.isfinite(embedding).all()
assert labels.shape == (n_cells,)
"""

COLD_START = Ladder(sizes=(1,), smoke=1, unit="process", targets=(1,), model="fixed")


def test_cold_start_of_graph_embedding_and_clustering(bench) -> None:
    if not bench.timed:
        pytest.skip("The cold start is a timing benchmark; set SCARF_RUN_BENCHMARKS=1")

    def make(_processes: int):
        return lambda: subprocess.run(
            [sys.executable, "-c", COLD_PIPELINE],
            check=True,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
        )

    # A second spent by every new process is worth failing on.
    bench(
        "process.cold_start",
        make,
        COLD_START,
        thresholds=Thresholds(meaningful_delay=1.0),
        min_repeats=2,
    )
