"""Scaling benchmarks of the plotting adapters whose cost grows with the data.

Each benchmark calls the function a plot calls, on inputs shaped like a
production store, and checks its value against an independent computation, so
it doubles as a test of that function at the smoke size. Figures are never
drawn inside the timed call: the timed work is the data preparation that grows
with cells or features.
"""

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

# Every dotplot, matrixplot, and distribution resolves each requested feature
# with its own call, which reads the assay's whole feature table. Projections
# count a 20-feature dotplot.
FEATURES = Ladder(
    sizes=(16_000, 32_000, 64_000, 128_000),
    smoke=2_000,
    unit="features",
    targets=(30_000, 60_000),
    work=20.0,
)
RASTER_CELLS = Ladder(sizes=(125_000, 250_000, 500_000, 1_000_000), smoke=2_000)
GRAPH_CELLS = Ladder(sizes=(62_500, 125_000, 250_000, 500_000), smoke=2_000)
SUMMARY_CELLS = Ladder(sizes=(62_500, 125_000, 250_000, 500_000), smoke=2_000)

# embedding_raster's defaults: a 400-pixel square canvas, 1%/99% color limits
# from a uniform sample of at most 50,000 values.
RASTER_PIXELS = 400
RASTER_QUANTILES = (0.01, 0.99)
RASTER_SAMPLE = 50_000
# Leiden at a million cells finds a few dozen clusters; most kNN edges stay
# inside a cluster.
N_CLUSTERS = 30
WITHIN_CLUSTER_SHARE = 0.85
# A typical dotplot panel, streamed in the row blocks of a 1M-cell, 33k-gene
# count matrix (its Zarr chunk length).
N_SUMMARY_FEATURES = 30
COUNT_BLOCK_ROWS = 15_000


@lru_cache(maxsize=4)
def _feature_table(n_features: int) -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    """Return a store whose RNA feature table holds ``n_features`` genes."""
    import zarr
    from zarr.storage import MemoryStore

    from scarf.metadata.table import MetaData
    from scarf.storage.schema import create_zarr_count_assay

    prefixes = ("AC", "LINC", "Gm", "MT-ND", "Rpl", "CD")
    names = np.array(
        [f"{prefixes[index % len(prefixes)]}{index}" for index in range(n_features)]
    )
    ids = np.array([f"ENSG{index:011d}" for index in range(n_features)])
    root = zarr.open_group(store=MemoryStore(), mode="w")
    create_zarr_count_assay(
        root, "RNA", None, 2, feat_ids=ids, feat_names=names, dtype="uint16"
    )
    assay = SimpleNamespace(feats=MetaData(root["RNA"]["featureData"]))
    store = SimpleNamespace(_defaultAssay="RNA", _get_assay=lambda _name: assay)
    return store, names, ids


def _feature_query(n_features: int) -> str:
    """A gene from the middle of the table, typed in the other letter case."""
    _store, names, _ids = _feature_table(n_features)
    return str(names[n_features // 2]).swapcase()


def test_feature_name_lookup(bench) -> None:
    from scarf.features.values import resolve_feature

    def make(n_features: int):
        store, _names, _ids = _feature_table(n_features)
        name = _feature_query(n_features)
        return lambda: resolve_feature(store, name)

    def check(n_features: int, resolved) -> None:
        _store, names, ids = _feature_table(n_features)
        name = _feature_query(n_features)
        matches = np.flatnonzero(np.char.upper(names.astype(str)) == name.upper())
        assert matches.tolist() == [n_features // 2]
        (index,) = matches
        assert resolved.assay == "RNA"
        assert resolved.indices == (index,)
        assert resolved.names == (str(names[index]),)
        assert resolved.ids == (str(ids[index]),)
        assert resolved.label == str(names[index])
        assert resolved.reduction is None

    bench("plotting.feature_name_lookup", make, FEATURES, check=check)


@lru_cache(maxsize=4)
def _cell_table(n_cells: int) -> tuple[object, np.ndarray, np.ndarray]:
    """Return cell metadata with a clustered 2-D layout and library sizes."""
    import zarr
    from zarr.storage import MemoryStore

    from scarf.metadata.table import MetaData
    from scarf.storage.schema import create_cell_data

    rng = np.random.default_rng(inputs.SEED)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ids = np.array([f"BARCODE{index:010d}-1" for index in range(n_cells)])
    create_cell_data(root, None, ids=ids, names=ids)
    cells = MetaData(root["cellData"])
    centers = rng.normal(0.0, 6.0, size=(inputs.N_GROUPS, 2))
    labels = rng.integers(0, inputs.N_GROUPS, size=n_cells)
    layout = centers[labels] + rng.normal(size=(n_cells, 2))
    totals = np.round(rng.lognormal(np.log(1_500.0), 0.5, size=n_cells))
    cells.insert("RNA_UMAP1", layout[:, 0], overwrite=True)
    cells.insert("RNA_UMAP2", layout[:, 1], overwrite=True)
    cells.insert("RNA_nCounts", totals, overwrite=True)
    return cells, layout, totals


def _square_extent(x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
    """Pad each axis by 1% of its span, then square the window around it."""
    pad_x, pad_y = 0.01 * np.ptp(x), 0.01 * np.ptp(y)
    x0, x1 = x.min() - pad_x, x.max() + pad_x
    y0, y1 = y.min() - pad_y, y.max() + pad_y
    half = max(x1 - x0, y1 - y0) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - half, cx + half, cy - half, cy + half)


def test_embedding_raster_canvas(bench) -> None:
    from scarf.plotting._raster import raster_from_metadata

    def make(n_cells: int):
        cells, _layout, _totals = _cell_table(n_cells)
        return lambda: raster_from_metadata(
            cells,
            x_key="RNA_UMAP1",
            y_key="RNA_UMAP2",
            color_key="RNA_nCounts",
            pixels=RASTER_PIXELS,
            quantiles=RASTER_QUANTILES,
            sample_capacity=RASTER_SAMPLE,
        )

    def check(n_cells: int, canvas) -> None:
        _cells, layout, totals = _cell_table(n_cells)
        x, y = layout[:, 0], layout[:, 1]
        extent = _square_extent(x, y)
        assert canvas.extent == pytest.approx(extent)
        assert canvas.n_cells == n_cells
        window = ((extent[0], extent[1]), (extent[2], extent[3]))
        # Image row 0 is the top of the canvas.
        counts = np.flipud(np.histogram2d(x, y, bins=RASTER_PIXELS, range=window)[0].T)
        sums = np.flipud(
            np.histogram2d(x, y, bins=RASTER_PIXELS, range=window, weights=totals)[0].T
        )
        np.testing.assert_array_equal(canvas.counts, counts)
        np.testing.assert_allclose(
            canvas.image,
            np.where(counts > 0, sums / np.maximum(counts, 1), np.nan),
            rtol=1e-9,
            equal_nan=True,
        )
        if n_cells <= RASTER_SAMPLE:
            # The sample holds every value, so the limits are exact quantiles.
            assert (canvas.vmin, canvas.vmax) == pytest.approx(
                tuple(np.quantile(totals, RASTER_QUANTILES))
            )
        else:
            # A uniform sample of 50,000 values puts each limit within a few
            # standard errors (0.04% of the cells) of its rank.
            for limit, rank in zip(
                (canvas.vmin, canvas.vmax), RASTER_QUANTILES, strict=True
            ):
                assert np.mean(totals < limit) <= rank + 0.004
                assert np.mean(totals <= limit) >= rank - 0.004

    bench("plotting.raster_canvas", make, RASTER_CELLS, check=check)


@lru_cache(maxsize=4)
def _cluster_graph(n_cells: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return a symmetric kNN-like graph and the cluster code of each cell.

    Each cell links to ``N_NEIGHBORS`` cells, 85% of them in its own cluster,
    and the graph is prepared exactly as cluster_connectivity loads it.
    """
    from scarf.plotting.cluster_connectivity import _load_graph
    from scarf.storage import ArtifactRef

    rng = np.random.default_rng(inputs.SEED)
    labels = rng.integers(0, N_CLUSTERS, size=n_cells)
    members = np.argsort(labels, kind="stable")
    starts = np.searchsorted(labels[members], np.arange(N_CLUSTERS))
    sizes = np.bincount(labels, minlength=N_CLUSTERS)
    rows = np.repeat(np.arange(n_cells), inputs.N_NEIGHBORS)
    own = labels[rows]
    local = members[starts[own] + (rng.random(rows.size) * sizes[own]).astype(np.int64)]
    within = rng.random(rows.size) < WITHIN_CLUSTER_SHARE
    columns = np.where(within, local, rng.integers(0, n_cells, size=rows.size))
    columns = np.where(columns == rows, (rows + 1) % n_cells, columns)
    weights = rng.uniform(0.05, 1.0, size=rows.size)
    directed = sparse.csr_matrix((weights, (rows, columns)), shape=(n_cells, n_cells))
    # DataStore.load_graph(symmetric=True) adds the graph to its transpose.
    symmetric = (directed + directed.T).tocsr()
    store = SimpleNamespace(load_graph=lambda *_args, **_kwargs: symmetric)
    graph = _load_graph(
        store,
        n_cells=n_cells,
        graph=ArtifactRef(
            scope="assay", assay="RNA", kind="connectivity_map", artifact_id="1" * 64
        ),
    )
    return graph, labels.astype(np.intp)


def test_cluster_connectivity_edges(bench) -> None:
    from scarf.plotting.cluster_connectivity import _aggregate_intercluster_edges

    def make(n_cells: int):
        graph, codes = _cluster_graph(n_cells)
        return lambda: _aggregate_intercluster_edges(graph, codes, N_CLUSTERS)

    def check(n_cells: int, value) -> None:
        source, target, raw, normalized = value
        graph, codes = _cluster_graph(n_cells)
        membership = sparse.csr_matrix(
            (np.ones(n_cells), (np.arange(n_cells), codes)),
            shape=(n_cells, N_CLUSTERS),
        )
        # For a symmetric graph, entry (a, b) of M' G M sums each edge between
        # clusters a and b once; row sums give each cluster's incident weight.
        between = (membership.T @ graph @ membership).toarray()
        incident = membership.T @ np.asarray(graph.sum(axis=1)).ravel()
        first, second = np.nonzero(np.triu(between, k=1))
        np.testing.assert_array_equal(source, first)
        np.testing.assert_array_equal(target, second)
        np.testing.assert_allclose(raw, between[first, second], rtol=1e-10)
        np.testing.assert_allclose(
            normalized,
            between[first, second] / np.sqrt(incident[first] * incident[second]),
            rtol=1e-10,
        )

    bench("plotting.cluster_connectivity_edges", make, GRAPH_CELLS, check=check)


@lru_cache(maxsize=4)
def _expression(n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return log-normalized expression of a feature panel and cluster labels.

    Cluster-specific Poisson rates leave most values at zero, as in droplet
    RNA data.
    """
    rng = np.random.default_rng(inputs.SEED)
    labels = rng.integers(0, N_CLUSTERS, size=n_cells)
    rates = rng.gamma(0.5, 2.0, size=(N_CLUSTERS, N_SUMMARY_FEATURES))
    values = np.log1p(0.7 * rng.poisson(rates[labels]))
    return values, labels


def _summary_store(values: np.ndarray) -> SimpleNamespace:
    """A store whose assay streams normalized values in count-matrix row blocks."""
    from scarf.matrix.chunked import ChunkedArray

    class Assay:
        @staticmethod
        def normed(*, cell_idx, feat_idx):
            return ChunkedArray(
                values,
                rows=cell_idx,
                cols=feat_idx,
                block_size=COUNT_BLOCK_ROWS,
                nthreads=1,
                is_numpy=True,
            )

    assay = Assay()
    return SimpleNamespace(
        _defaultAssay="RNA", nthreads=1, _get_assay=lambda _name: assay
    )


def test_dotplot_group_summary(bench) -> None:
    import pandas as pd

    from scarf.features.values import ResolvedFeature
    from scarf.plotting._data import _summarize_resolved_features

    features = [f"GENE{index}" for index in range(N_SUMMARY_FEATURES)]
    resolved = [
        ResolvedFeature(
            assay="RNA",
            by="name",
            indices=(index,),
            ids=(f"ENSG{index:011d}",),
            names=(name,),
            label=name,
            reduction=None,
            raw=name,
        )
        for index, name in enumerate(features)
    ]

    def make(n_cells: int):
        values, labels = _expression(n_cells)
        store = _summary_store(values)
        grouping = (("cluster",), np.arange(n_cells), [labels], None)
        return lambda: _summarize_resolved_features(
            store, resolved, [None] * len(resolved), grouping
        )

    def check(n_cells: int, value) -> None:
        aggregate, per_sample = value
        assert per_sample is None
        values, labels = _expression(n_cells)
        counts = np.bincount(labels, minlength=N_CLUSTERS)
        rows = []
        for index, name in enumerate(features):
            column = values[:, index]
            sums = np.bincount(labels, weights=column, minlength=N_CLUSTERS)
            squares = np.bincount(labels, weights=column**2, minlength=N_CLUSTERS)
            expressing = np.bincount(labels, weights=column > 0, minlength=N_CLUSTERS)
            means = sums / counts
            for cluster in range(N_CLUSTERS):
                rows.append(
                    {
                        "cluster": cluster,
                        "feature": name,
                        "mean": means[cluster],
                        "fraction": expressing[cluster] / counts[cluster],
                        "n_cells": counts[cluster],
                        "variance": (
                            squares[cluster] - counts[cluster] * means[cluster] ** 2
                        )
                        / (counts[cluster] - 1),
                    }
                )
        expected = pd.DataFrame(rows).set_index(["cluster", "feature"]).sort_index()
        observed = aggregate.set_index(["cluster", "feature"]).sort_index()
        assert observed.index.equals(expected.index)
        np.testing.assert_array_equal(observed["n_cells"], expected["n_cells"])
        np.testing.assert_allclose(
            observed[["mean", "fraction", "variance"]],
            expected[["mean", "fraction", "variance"]],
            rtol=1e-9,
        )

    bench("plotting.dotplot_group_summary", make, SUMMARY_CELLS, check=check)
