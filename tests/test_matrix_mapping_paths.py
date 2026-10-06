"""ChunkedArray operator and nesting semantics, aligned mapping totals, and
UMAP thread limits."""

import numpy as np
import pytest
import scipy.special
import zarr
from scipy.sparse import coo_matrix
from zarr.storage import MemoryStore

from scarf.assay import norm_lib_size
from scarf.datastore.datastore import DataStore
from scarf.embeddings import simplicial_set_embedding
from scarf.embeddings.umap import densmap_distance_graph
from scarf.mapping.features import AlignedFeatureStream
from scarf.matrix import ChunkedArray
from scarf.storage.artifacts import callable_identity
from scarf.storage.budget import ResourceBudget
from scarf.storage.execution import execution_report_scope
from scarf.storage.parallel import stream_shards
from tests.storage_helpers import write_count_store


def _values() -> np.ndarray:
    return np.arange(1, 13, dtype=np.float64).reshape(4, 3)


def test_empty_row_selection_has_no_blocks() -> None:
    empty = ChunkedArray.from_numpy(_values(), block_size=2)[
        np.array([], dtype=np.int64), :
    ]

    assert empty.numblocks == (0, 1)
    assert list(empty.stream_blocks()) == []
    assert "numblocks=0" in repr(empty)


def test_reflected_operators_and_comparisons_match_numpy() -> None:
    values = _values()
    ca = ChunkedArray.from_numpy(values, block_size=3)
    column = np.array([[1.0], [2.0], [3.0], [4.0]])
    cases = {
        "rtruediv": (2.0 / ca, 2.0 / values),
        "radd": (1 + ca, 1 + values),
        "rsub": (10 - ca, 10 - values),
        "array_rsub": (column - ca, column - values),
        "ufunc_right_operand": (np.subtract(10.0, ca), 10.0 - values),
        "ge": (ca >= 6, values >= 6),
        "le": (ca <= 6, values <= 6),
    }

    for name, (lazy, expected) in cases.items():
        assert isinstance(lazy, ChunkedArray), name
        actual = lazy.compute()
        assert actual.dtype == expected.dtype, name
        np.testing.assert_array_equal(actual, expected, err_msg=name)


def test_ufuncs_with_three_operands_are_not_lazy() -> None:
    ca = ChunkedArray.from_numpy(_values() / 13.0)

    for arguments in ((ca, 1.0, 0.5), (1.0, 1.0, ca)):
        with pytest.raises(TypeError, match="NotImplemented"):
            scipy.special.betainc(*arguments)


def test_indexing_rejects_more_than_two_axes() -> None:
    ca = ChunkedArray.from_numpy(_values())

    with pytest.raises(IndexError, match="at most 2D indexing"):
        ca[0, 1, 2]


def test_streams_and_reductions_inside_a_shard_worker_read_inline() -> None:
    values = np.arange(120, dtype=np.float64).reshape(40, 3)
    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts = root.create_array("counts", shape=values.shape, chunks=(8, 3), dtype="f8")
    counts[:] = values
    ca = ChunkedArray(counts, nthreads=4, resources=ResourceBudget(64 * 1024**2, 4))

    def readers(compute) -> tuple[list[np.ndarray], list[int]]:
        with execution_report_scope() as reports:
            results = list(stream_shards([0], lambda _: compute(), workers=1))
        return results, [report.actualReadWorkers for report in reports]

    with execution_report_scope() as reports:
        expected_sums = ca.sum(axis=0).compute()
    assert reports[-1].actualReadWorkers > 1

    sums, sum_readers = readers(lambda: ca.sum(axis=0).compute())
    blocks, block_readers = readers(lambda: np.vstack(list(ca.stream_blocks())))

    np.testing.assert_array_equal(expected_sums, values.sum(axis=0))
    np.testing.assert_array_equal(sums[0], expected_sums)
    np.testing.assert_array_equal(blocks[0], values)
    assert sum_readers == block_readers == [1]


def _normalization() -> dict:
    return {
        "normalization_method": callable_identity(norm_lib_size),
        "size_factor": 10.0,
        "log_transform": False,
        "renormalize_subset": False,
    }


def test_aligned_stream_rejects_selected_cells_with_negative_library_sizes(
    tmp_path,
) -> None:
    # Pre-normalized values may be negative, so a cell's library size can be.
    values = np.array([[1.5, 2.0, 0.5], [-4.0, 1.0, 0.0], [0.0, 3.0, 2.0]])
    zarr_loc = str(tmp_path / "query.zarr")
    write_count_store(zarr_loc, {"RNA": values}, "float32")
    query = DataStore(zarr_loc, default_assay="RNA", min_features_per_cell=0)
    reference_ids = np.array(["RNA0", "RNA1"])

    def stream(cells: np.ndarray) -> AlignedFeatureStream:
        return AlignedFeatureStream(
            query.RNA,
            cells,
            reference_ids,
            np.zeros(len(reference_ids)),
            _normalization(),
            "zero",
            ResourceBudget(16 * 1024**2, 1),
        )

    with pytest.raises(
        ValueError, match="Query assay 'RNA_nCounts' holds negative or non-finite"
    ):
        stream(np.arange(3))
    aligned = np.concatenate(
        [block.values for block in stream(np.array([0, 2])).iter_blocks()]
    )

    np.testing.assert_allclose(
        aligned,
        10.0 * values[np.ix_([0, 2], [0, 1])] / values[[0, 2]].sum(axis=1)[:, None],
    )


def test_aligned_stream_normalizes_bool_query_counts_as_zeros_and_ones(
    tmp_path,
) -> None:
    detected = np.array(
        [[True, False, True], [False, False, False], [True, True, True]]
    )
    reference_ids = np.array(["RNA0", "RNA2"])
    zarr_loc = str(tmp_path / "bool.zarr")
    write_count_store(zarr_loc, {"RNA": detected}, "bool")
    query = DataStore(zarr_loc, default_assay="RNA", min_features_per_cell=0)
    assert query.RNA.rawData.dtype == np.dtype(bool)
    stream = AlignedFeatureStream(
        query.RNA,
        np.arange(3),
        reference_ids,
        np.zeros(len(reference_ids)),
        _normalization(),
        "zero",
        ResourceBudget(16 * 1024**2, 1),
    )
    aligned = np.concatenate([block.values for block in stream.iter_blocks()])

    # Each detected feature counts once, as uint8 counts of one would.
    totals = np.maximum(detected.sum(axis=1), 1)[:, None]
    np.testing.assert_allclose(aligned, 10.0 * detected[:, [0, 2]] / totals)


def _ring_graph(n_cells: int) -> coo_matrix:
    rows = np.repeat(np.arange(n_cells), 2)
    cols = np.stack(
        [(np.arange(n_cells) - 1) % n_cells, (np.arange(n_cells) + 1) % n_cells],
        axis=1,
    ).ravel()
    return coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n_cells, n_cells))


# The layout is replaced, but importing umap-learn compiles pynndescent's
# kernels, which takes several seconds in a fresh process.
@pytest.mark.slow
def test_umap_layout_uses_the_requested_threads_up_to_the_numba_pool(
    monkeypatch,
) -> None:
    import numba
    import umap.layouts

    observed: list[int] = []

    def layout(**kwargs):
        observed.append(numba.get_num_threads())
        return kwargs["head_embedding"]

    monkeypatch.setattr(umap.layouts, "optimize_layout_euclidean", layout)
    pool = numba.config.NUMBA_NUM_THREADS
    graph = _ring_graph(6)
    previous = numba.get_num_threads()
    # The caller runs on fewer threads than it requests, so a request for the
    # whole pool must raise the count, not keep the caller's.
    numba.set_num_threads(1)
    try:
        for parallel, requested in (
            (True, 1),
            (True, pool),
            (True, pool + 3),
            (False, pool),
        ):
            simplicial_set_embedding(
                graph,
                np.zeros((6, 2), dtype=np.float32),
                2,
                1.0,
                1.0,
                1,
                1.0,
                1.0,
                5,
                {},
                parallel,
                requested,
                False,
            )
        restored = numba.get_num_threads()
    finally:
        numba.set_num_threads(previous)

    # Numba refuses more threads than its pool, so a larger request runs the
    # layout on the whole pool, and a serial layout runs on one thread.
    assert observed == [1, pool, pool, 1]
    assert restored == 1


@pytest.mark.parametrize(
    ("indices", "distances"),
    [
        (np.zeros((3, 2), dtype=np.int64), np.zeros((3, 3))),
        (np.zeros(3, dtype=np.int64), np.zeros(3)),
    ],
)
def test_densmap_distances_require_matching_neighbor_matrices(
    indices: np.ndarray,
    distances: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="matching matrices"):
        densmap_distance_graph(indices, distances)
