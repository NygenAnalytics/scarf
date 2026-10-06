"""Milestone C: MetaData.iter_row_blocks parity and make_bulk cell_key semantics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import zarr
from zarr.storage import MemoryStore

from scarf.features.aggregation import aligned_feature_labels
from scarf.metadata import MetaData
from scarf.metadata.artifacts import artifact_values
from scarf.metadata.queries import (
    column_partition_digest,
    columns_same_partition,
    reduce_observation_units,
)
from scarf.metadata.rows import metadata_column_fingerprint
from scarf.storage.artifacts import ArtifactRef, artifact_group
from scarf.utils.logging import logger

_PSEUDO_REP_WARNING = (
    "make_bulk with pseudo_reps > 1 randomly splits cells within each "
    "group into descriptive resamples"
)


def test_metric_fingerprint_changes_when_only_missing_mask_changes() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    missing = np.asarray([False, False, False])
    metadata = SimpleNamespace(
        N=3,
        _get_array=lambda _column: values,
        default_block_rows=lambda _column: 2,
        _get_missing_mask_array=lambda _column: missing,
    )
    first = metadata_column_fingerprint(metadata, "age")
    missing[1] = True
    assert metadata_column_fingerprint(metadata, "age") != first
    missing[1] = False
    assert metadata_column_fingerprint(metadata, "age") == first


def test_iter_row_blocks_matches_active_index_and_fetch(datastore):
    cells = datastore.cells
    col = "ids"
    expected_idx = cells.active_index("I")
    expected_vals = cells.fetch(col, key="I")

    # Single-row blocks are checked on a small table in test_metadata_contracts.
    for block_rows in (None, 7, cells.default_block_rows("I")):
        blocks = list(
            cells.iter_row_blocks(cell_key="I", columns=[col], block_rows=block_rows)
        )
        got_idx = np.concatenate([b.active_global_indices for b in blocks])
        got_vals = np.concatenate([b.values[col] for b in blocks])
        np.testing.assert_array_equal(got_idx, expected_idx)
        np.testing.assert_array_equal(got_vals, expected_vals)
        size = cells.default_block_rows("I") if block_rows is None else block_rows
        # Blocks tile [0, N) in order with the requested size.
        assert [(b.start, b.stop) for b in blocks] == [
            (start, min(start + size, cells.N)) for start in range(0, cells.N, size)
        ]
        for block in blocks:
            assert np.all(block.active_global_indices >= block.start)
            assert np.all(block.active_global_indices < block.stop)


def test_iter_row_blocks_chunk_edges_cover_full_range(datastore):
    cells = datastore.cells
    chunk = cells.default_block_rows("I")
    # The default block size is the chunk length of the key column.
    assert chunk == datastore.zw["cellData"]["I"].chunks[0]
    blocks = list(cells.iter_row_blocks(cell_key="I", block_rows=chunk))
    assert len(blocks) == -(-cells.N // chunk)
    assert sum(b.stop - b.start for b in blocks) == cells.N
    for b in blocks:
        assert b.stop - b.start <= chunk
        assert b.values == {}


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


def test_make_bulk_respects_explicit_subset_selection(leiden_clustering, datastore):
    """Inactive cells that share a group label must not enter the bulk profile."""
    ds = datastore
    selection = ArtifactRef.from_dict(
        ds.inspect_artifact(leiden_clustering).inputs["cell_selection"]
    )
    active = artifact_values(
        artifact_group(ds.zw, selection),
        "values",
    ).astype(bool)
    active_idx = np.flatnonzero(active)
    labels = artifact_values(artifact_group(ds.zw, leiden_clustering), "values")
    drop = np.zeros(ds.cells.N, dtype=bool)
    drop[active_idx[::2]] = True
    subset = active & ~drop
    ds.cells.insert("bulk_subset", subset, overwrite=True)
    subset_selection = ds.snapshot_cell_selection("bulk_subset")

    full = ds.make_bulk(
        leiden_clustering,
        aggr_type="sum",
        remove_empty_features=False,
        feature_label="index",
    )
    sub = ds.make_bulk(
        leiden_clustering,
        cell_selection=subset_selection,
        aggr_type="sum",
        remove_empty_features=False,
        feature_label="index",
    )

    groups = [str(label) for label in np.unique(labels)]
    assert list(full.columns) == list(sub.columns) == groups
    # Group totals over every feature equal the per-cell totals recorded at
    # import, and a sample of features equals the raw counts of the members.
    n_counts = np.asarray(ds.cells.fetch_all("RNA_nCounts"))
    in_subset = subset[active_idx]
    features = np.sort(np.argsort(ds.RNA.feats.fetch_all("nCells"))[-25:])
    raw = np.asarray(ds.RNA.rawData[active_idx][:, features].compute())
    for group in groups:
        members = labels.astype(str) == group
        for frame, rows in ((full, members), (sub, members & in_subset)):
            assert frame[group].sum() == n_counts[active_idx[rows]].sum()
            np.testing.assert_array_equal(
                frame[group].to_numpy()[features], raw[rows].sum(axis=0)
            )
        assert sub[group].sum() < full[group].sum()


def test_make_bulk_pseudo_reps_warns_without_changing_values(
    leiden_clustering, datastore
):
    """Pseudo-replicate splits warn once; aggregation stays numerically identical."""
    ds = datastore
    kwargs = {
        "groups": leiden_clustering,
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
    labels = aligned_feature_labels(np.asarray(values), pd.Index([0, 2]))
    frame = pd.DataFrame([[1.0], [2.0]])
    frame.set_index(labels, inplace=True)
    assert list(frame.index) == ["gene_a", "gene_c"]


def test_make_bulk_feature_name_index_is_hashable(leiden_clustering, datastore):
    bulk = datastore.make_bulk(
        leiden_clustering,
        feature_label="name",
        aggr_type="sum",
        remove_empty_features=True,
    )
    indexed = datastore.make_bulk(
        leiden_clustering,
        feature_label="index",
        aggr_type="sum",
        remove_empty_features=True,
    )
    names = np.asarray(datastore.RNA.feats.fetch_all("names"), dtype=object)

    # Names replace the positions of the expressed features, row for row.
    assert len(bulk) == len(indexed) > 0
    assert (indexed.sum(axis=1) > 0).all()
    assert bulk.index.tolist() == names[indexed.index.to_numpy()].tolist()
    assert all(isinstance(name, str) for name in bulk.index)
    np.testing.assert_array_equal(bulk.to_numpy(), indexed.to_numpy())
    assert bulk.loc[bulk.index[0]].shape == (bulk.shape[1],)


def _chunked_table(columns: dict[str, list]) -> MetaData:
    """A table whose two-row chunks make each row block hold two rows."""
    group = zarr.open_group(store=MemoryStore(), mode="w")
    n_rows = len(next(iter(columns.values())))
    group.create_array("I", data=np.ones(n_rows, dtype=bool), chunks=(2,))
    group.create_array(
        "ids", data=np.array([f"c{i}" for i in range(n_rows)]), chunks=(2,)
    )
    for name, values in columns.items():
        group.create_array(name, data=np.asarray(values), chunks=(2,))
    return MetaData(group)


def test_column_partition_digest_matches_factorization():
    ids = [f"id{index}" for index in range(12)]
    with_missing = np.arange(12, dtype=float)
    with_missing[[0, 5]] = np.nan
    cells = _chunked_table(
        {
            "label": ids,
            # Renamed labels, same partition as label under a fresh codebook.
            "alias": [f"alias-{value}" for value in ids],
            "with_missing": with_missing,
        }
    )

    digest_ids = column_partition_digest(cells, "label")
    digest_alias = column_partition_digest(cells, "alias")
    assert digest_ids == digest_alias
    assert (digest_ids.nLevels, digest_ids.nMissing, digest_ids.nRows) == (12, 0, 12)

    same, shown = columns_same_partition(cells, "label", "alias")
    assert same is True
    # The first eight correspondences are listed and the rest elided.
    assert shown == "; ".join(f"{value} = alias-{value}" for value in ids[:8]) + (
        "; ..."
    )

    # NaN rows join one missing level that every other value differs from.
    digest_missing = column_partition_digest(cells, "with_missing")
    assert (digest_missing.nMissing, digest_missing.nLevels) == (2, 11)
    assert digest_missing.nRows == 12
    assert digest_missing.digest != digest_alias.digest


def test_partition_digest_is_global_across_blocks():
    cells = _chunked_table(
        {
            "left": ["a", "b", "a", "b"],
            # Each two-row block of right repeats one of left's patterns, but
            # the second block swaps the labels.
            "right": ["a", "b", "b", "a"],
            "renamed": ["x", "y", "x", "y"],
        }
    )
    assert [len(block.active_global_indices) for block in cells.iter_row_blocks()] == [
        2,
        2,
    ]

    left_digest = column_partition_digest(cells, "left")
    # Per-block codebooks would code both columns 0, 1, 0, 1.
    assert column_partition_digest(cells, "right").digest != left_digest.digest
    assert columns_same_partition(cells, "left", "right") == (False, "")
    assert column_partition_digest(cells, "renamed") == left_digest
    assert columns_same_partition(cells, "left", "renamed") == (True, "a = x; b = y")


def test_reduce_observation_units_respects_cell_key():
    n_cells = 9
    keep = np.arange(n_cells) % 2 == 0
    cells = _chunked_table(
        {
            "digest_unit_subset": keep,
            "digest_unit_sample": [f"s{i % 3}" for i in range(n_cells)],
            "digest_unit_disease": [
                "case" if i % 2 == 0 else "ctrl" for i in range(n_cells)
            ],
        }
    )

    design = reduce_observation_units(
        cells,
        "digest_unit_sample",
        ["digest_unit_disease"],
        cell_key="digest_unit_subset",
    )
    unfiltered = reduce_observation_units(
        cells, "digest_unit_sample", ["digest_unit_disease"]
    )

    # The first kept row of each sample is 0 (s0), 2 (s2), and 4 (s1).
    assert design.to_dict(orient="list") == {
        "digest_unit_sample": ["s0", "s2", "s1"],
        "digest_unit_disease": ["case", "case", "case"],
    }
    # Without the key, row 1 is the first row of s1, which is a control.
    assert unfiltered.to_dict(orient="list") == {
        "digest_unit_sample": ["s0", "s1", "s2"],
        "digest_unit_disease": ["case", "ctrl", "case"],
    }
