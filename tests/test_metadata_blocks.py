"""Milestone C: MetaData.iter_row_blocks parity and make_bulk cell_key semantics."""

import numpy as np
import pandas as pd

from scarf.datastore._operations.features import _aligned_feature_labels
from scarf.utils.logging import logger

_PSEUDO_REP_WARNING = (
    "make_bulk with pseudo_reps > 1 randomly splits cells within each "
    "group into descriptive resamples"
)


def test_iter_row_blocks_matches_active_index_and_fetch(datastore):
    cells = datastore.cells
    col = "ids"
    expected_idx = cells.active_index("I")
    expected_vals = cells.fetch(col, key="I")

    for block_rows in (None, 7, 1, cells.default_block_rows("I")):
        blocks = list(
            cells.iter_row_blocks(cell_key="I", columns=[col], block_rows=block_rows)
        )
        got_idx = (
            np.concatenate([b.active_global_indices for b in blocks])
            if blocks
            else np.array([], dtype=np.int64)
        )
        got_vals = (
            np.concatenate([b.values[col] for b in blocks]) if blocks else np.array([])
        )
        np.testing.assert_array_equal(got_idx, expected_idx)
        np.testing.assert_array_equal(got_vals, expected_vals)
        assert all(b.start < b.stop for b in blocks)
        assert blocks[0].start == 0
        assert blocks[-1].stop == cells.N


def test_iter_row_blocks_chunk_edges_cover_full_range(datastore):
    cells = datastore.cells
    chunk = cells.default_block_rows("I")
    blocks = list(cells.iter_row_blocks(cell_key="I", block_rows=chunk))
    assert sum(b.stop - b.start for b in blocks) == cells.N
    for b in blocks:
        assert b.stop - b.start <= chunk


def test_iter_row_blocks_respects_subset_cell_key(datastore):
    cells = datastore.cells
    keep = np.zeros(cells.N, dtype=bool)
    keep[::3] = True
    cells.insert("block_subset", keep, overwrite=True)
    expected = cells.active_index("block_subset")
    got = np.concatenate(
        [
            b.active_global_indices
            for b in cells.iter_row_blocks(
                cell_key="block_subset", columns=["ids"], block_rows=11
            )
        ]
    )
    np.testing.assert_array_equal(got, expected)


def test_make_bulk_respects_non_default_cell_key(leiden_clustering, datastore):
    """Inactive cells that share a group label must not enter the bulk profile."""
    ds = datastore
    active = ds.cells.fetch_all("I").copy()
    assert active.dtype == bool
    active_idx = np.flatnonzero(active)
    drop = np.zeros(ds.cells.N, dtype=bool)
    drop[active_idx[::2]] = True
    subset = active & ~drop
    ds.cells.insert("bulk_subset", subset, overwrite=True)

    full = ds.make_bulk(
        group_key="RNA_leiden_cluster",
        cell_key="I",
        aggr_type="sum",
        remove_empty_features=False,
        feature_label="index",
    )
    sub = ds.make_bulk(
        group_key="RNA_leiden_cluster",
        cell_key="bulk_subset",
        aggr_type="sum",
        remove_empty_features=False,
        feature_label="index",
    )
    shared = [c for c in sub.columns if c in full.columns]
    assert shared
    for c in shared:
        assert (sub[c].to_numpy() <= full[c].to_numpy() + 1e-6).all()

    from scarf.utils import controlled_compute

    clusters = ds.cells.fetch_all("RNA_leiden_cluster")
    subset_idx = ds.cells.active_index("bulk_subset")
    g_val = clusters[subset_idx[0]]
    col = str(g_val)
    assert col in sub.columns
    idx_g = subset_idx[clusters[subset_idx] == g_val]
    expected = controlled_compute(ds.RNA.rawData[idx_g].sum(axis=0), ds.nthreads)
    np.testing.assert_allclose(sub[col].to_numpy(), expected, rtol=1e-5, atol=1e-6)


def test_make_bulk_pseudo_reps_warns_without_changing_values(
    leiden_clustering, datastore
):
    """Pseudo-replicate splits warn once; aggregation stays numerically identical."""
    ds = datastore
    kwargs = {
        "group_key": "RNA_leiden_cluster",
        "aggr_type": "sum",
        "remove_empty_features": False,
        "feature_label": "index",
        "random_seed": 4466,
    }

    default_messages: list[str] = []
    sink = logger.add(
        lambda message: default_messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        default = ds.make_bulk(pseudo_reps=1, **kwargs)
    finally:
        logger.remove(sink)
    assert not any(_PSEUDO_REP_WARNING in msg for msg in default_messages)

    rep_messages: list[str] = []
    sink = logger.add(
        lambda message: rep_messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        with_reps = ds.make_bulk(pseudo_reps=2, **kwargs)
        again = ds.make_bulk(pseudo_reps=2, **kwargs)
    finally:
        logger.remove(sink)

    assert sum(_PSEUDO_REP_WARNING in msg for msg in rep_messages) == 2
    assert any("not independent biological replicates" in msg for msg in rep_messages)
    pd_values_equal = np.allclose(with_reps.to_numpy(), again.to_numpy())
    assert pd_values_equal
    assert with_reps.shape[1] == 2 * default.shape[1]
    # Each group's two pseudo-reps partition the cells; their sums recover the
    # unsplit group total (aggregation path unchanged by the warning).
    for col in default.columns:
        rep1 = f"{col}_Rep1"
        rep2 = f"{col}_Rep2"
        assert rep1 in with_reps.columns and rep2 in with_reps.columns
        np.testing.assert_allclose(
            with_reps[rep1].to_numpy() + with_reps[rep2].to_numpy(),
            default[col].to_numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


def test_aligned_feature_labels_accepts_pandas_string_array() -> None:
    values = pd.array(["gene_a", "gene_b", "gene_c"], dtype="string")
    labels = _aligned_feature_labels(np.asarray(values), pd.Index([0, 2]))
    frame = pd.DataFrame([[1.0], [2.0]])
    frame.set_index(labels, inplace=True)
    assert list(frame.index) == ["gene_a", "gene_c"]


def test_make_bulk_feature_name_index_is_hashable(leiden_clustering, datastore):
    bulk = datastore.make_bulk(
        group_key="RNA_leiden_cluster",
        feature_label="name",
        aggr_type="sum",
        remove_empty_features=True,
    )
    names = np.asarray(datastore.RNA.feats.fetch_all("names"), dtype=object)
    assert not bulk.empty
    assert set(bulk.index).issubset(set(names))


def test_column_partition_digest_matches_factorization(datastore):
    from scarf.metadata.queries import column_partition_digest, columns_same_partition

    cells = datastore.cells
    ids = cells.fetch_all("ids")
    # Renamed labels, same partition as ids under a fresh codebook.
    alias = np.array([f"alias-{value}" for value in ids], dtype=object)
    cells.insert("digest_alias_ids", alias, overwrite=True)

    digest_ids = column_partition_digest(cells, "ids")
    digest_alias = column_partition_digest(cells, "digest_alias_ids")
    assert digest_ids.digest == digest_alias.digest
    assert digest_ids.nLevels == digest_alias.nLevels
    assert digest_ids.nRows == int(cells.active_index("I").size)

    same, shown = columns_same_partition(cells, "ids", "digest_alias_ids")
    assert same is True
    assert " = " in shown

    # Missing values use a numeric column so NaN survives storage. Place them on
    # active rows so the default cell_key="I" scan observes them.
    with_missing = np.arange(cells.N, dtype=float)
    active = cells.active_index("I")
    assert active.size >= 2
    with_missing[int(active[0])] = np.nan
    with_missing[int(active[1])] = np.nan
    cells.insert("digest_alias_missing", with_missing, overwrite=True)
    digest_missing = column_partition_digest(cells, "digest_alias_missing")
    assert digest_missing.nMissing == 2
    assert digest_missing.nLevels == digest_missing.nRows - 1
    assert digest_missing.digest != digest_alias.digest


def test_partition_digest_is_global_across_blocks(datastore):
    from scarf.metadata.queries import column_partition_digest, columns_same_partition

    cells = datastore.cells
    n = cells.N
    # Two columns with identical per-block patterns but different global partitions.
    left = np.array(["a", "b"] * (n // 2) + ["a"] * (n % 2), dtype=object)
    right = left.copy()
    mid = n // 2
    if mid + 1 < n:
        right[mid], right[mid + 1] = right[mid + 1], right[mid]
    cells.insert("digest_part_left", left, overwrite=True)
    cells.insert("digest_part_right", right, overwrite=True)

    left_digest = column_partition_digest(cells, "digest_part_left")
    right_digest = column_partition_digest(cells, "digest_part_right")
    if np.array_equal(left, right):
        assert left_digest.digest == right_digest.digest
    else:
        assert left_digest.digest != right_digest.digest
        same, _ = columns_same_partition(cells, "digest_part_left", "digest_part_right")
        assert same is False


def test_reduce_observation_units_respects_cell_key(datastore):
    from scarf.metadata.queries import reduce_observation_units

    cells = datastore.cells
    keep = np.zeros(cells.N, dtype=bool)
    keep[::2] = True
    cells.insert("digest_unit_subset", keep, overwrite=True)
    sample = np.array([f"s{i % 3}" for i in range(cells.N)], dtype=object)
    cells.insert("digest_unit_sample", sample, overwrite=True)
    disease = np.array(["case" if i % 2 == 0 else "ctrl" for i in range(cells.N)])
    cells.insert("digest_unit_disease", disease, overwrite=True)

    design = reduce_observation_units(
        cells,
        "digest_unit_sample",
        ["digest_unit_disease"],
        cell_key="digest_unit_subset",
    )
    assert len(design) == len(set(sample[keep]))
    assert set(design.columns) >= {"digest_unit_sample", "digest_unit_disease"}
