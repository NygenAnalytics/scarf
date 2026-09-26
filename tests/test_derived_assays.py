"""Derived-assay creation: atomic publication, rollback, and grouped means."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.assay.normalization import iter_feature_group_means
from scarf.datastore.datastore import DataStore
from scarf.storage.count_matrix import CountMatrixPolicy
from scarf.storage.identity import finalize_counts
from scarf.storage.schema import (
    PENDING_ASSAY_ATTR,
    create_cell_data,
    derived_assay_transaction,
    discard_pending_assay,
    pending_assays,
    validate_new_assay,
)
from scarf.tools.repack_zarr import repack_store
from scarf.writers import SparseToZarr, create_zarr_count_assay

# Four count shards of ten cells each, so one shard can fail on its own.
_SMALL_SHARDS = CountMatrixPolicy(unitBytes=480, chunkBytes=96)


def _counts(n_cells: int = 40, n_features: int = 12, seed: int = 7) -> np.ndarray:
    counts = np.random.default_rng(seed).integers(0, 6, (n_cells, n_features))
    counts[:, 0] += 1
    return counts.astype(np.uint32)


def _write_store(
    path: Path,
    counts: np.ndarray,
    *,
    assay: str = "RNA",
    feature_ids: list[str] | None = None,
) -> DataStore:
    SparseToZarr(
        csr_matrix(counts),
        zarr_loc=str(path),
        cell_ids=[f"c{i}" for i in range(counts.shape[0])],
        feature_ids=feature_ids or [f"g{i}" for i in range(counts.shape[1])],
        assay_name=assay,
        nthreads=1,
        policy=_SMALL_SHARDS,
    ).dump(batch_size=10)
    return DataStore(str(path), default_assay=assay, min_features_per_cell=0)


def _lib_size_group_means(
    counts: np.ndarray, groups: list[np.ndarray], size_factor: float
) -> np.ndarray:
    totals = counts.sum(axis=1).astype(np.float64)
    totals[totals == 0] = 1
    normalized = size_factor * counts / totals[:, None]
    return np.column_stack([normalized[:, group].mean(axis=1) for group in groups])


def _stored_counts(store: DataStore, assay: str) -> np.ndarray:
    return np.vstack(
        list(store.get_assay(assay).rawData.stream_blocks(nthreads=1, msg=None))
    )


def _assert_openable_without(path: Path, assay: str) -> None:
    for mode in ("r", "r+"):
        reopened = DataStore(str(path), zarr_mode=mode, min_features_per_cell=0)
        assert assay not in reopened.assay_names


def _last_count_shard(path: Path, assay: str) -> Path:
    shards = sorted(
        (path / assay / "counts" / "c").glob("*/0"),
        key=lambda shard: int(shard.parent.name),
    )
    assert len(shards) > 1
    return shards[-1]


def _corrupt(shard: Path) -> bytes:
    original = shard.read_bytes()
    shard.write_bytes(b"not a shard" * 16)
    return original


def test_add_grouped_assay_failure_leaves_store_openable_and_retryable(tmp_path):
    path = tmp_path / "rna.zarr"
    counts = _counts()
    store = _write_store(path, counts)
    modules = np.repeat([1, 2, 3], 4)
    store.RNA.feats.insert("module", modules, overwrite=True)
    shard = _last_count_shard(path, "RNA")
    original = _corrupt(shard)

    # A damaged count shard fails the read after the new assay was created.
    with pytest.raises(ValueError):
        store.add_grouped_assay("module", assay_label="MODULES")

    root = zarr.open_group(str(path), mode="r")
    assert "MODULES" not in root
    assert pending_assays(root) == []
    assert "MODULES" not in store.assay_names
    _assert_openable_without(path, "MODULES")

    shard.write_bytes(original)
    store.add_grouped_assay("module", assay_label="MODULES")

    assert "MODULES" in store.assay_names
    assert store.MODULES.z.attrs["is_assay"] is True
    assert PENDING_ASSAY_ATTR not in store.MODULES.z.attrs
    assert store.MODULES.z.attrs["grouped_group_column"] == "module"
    expected = _lib_size_group_means(
        counts,
        [np.flatnonzero(modules == value) for value in (1, 2, 3)],
        store.RNA.sf,
    )
    np.testing.assert_allclose(_stored_counts(store, "MODULES"), expected)
    assert list(store.MODULES.feats.fetch_all("ids")) == [
        "group_1",
        "group_2",
        "group_3",
    ]


def test_add_melded_assay_failure_is_rolled_back(tmp_path):
    path = tmp_path / "atac.zarr"
    counts = _counts(n_features=6)
    peaks = [
        "chr1:100-200",
        "chr1:250-350",
        "chr1:400-500",
        "chr2:100-200",
        "chr2:300-400",
        "chr2:600-700",
    ]
    store = _write_store(path, counts, assay="ATAC", feature_ids=peaks)
    bed = tmp_path / "genes.bed"
    pd.DataFrame(
        [
            ("chr1", 120, 300, "gene_a", "GENE_A", "+"),
            ("chr1", 420, 480, "gene_b", "GENE_B", "+"),
            ("chr2", 150, 650, "gene_c", "GENE_C", "+"),
        ]
    ).to_csv(bed, sep="\t", header=False, index=False)
    shard = _last_count_shard(path, "ATAC")
    original = _corrupt(shard)
    arguments = {
        "from_assay": "ATAC",
        "external_bed_fn": str(bed),
        "assay_label": "GeneScores",
        "assay_type": "RNA",
        "renormalization": False,
    }

    with pytest.raises(ValueError):
        store.add_melded_assay(**arguments)

    root = zarr.open_group(str(path), mode="r")
    assert "GeneScores" not in root
    assert "GeneScores" not in dict(root.attrs.get("assayTypes", {}))
    assert pending_assays(root) == []
    _assert_openable_without(path, "GeneScores")

    shard.write_bytes(original)
    store.add_melded_assay(**arguments)

    assert store.GeneScores.z.attrs["is_assay"] is True
    assert store.GeneScores.z.attrs["sourceAssay"] == "ATAC"
    assert store.z["GeneScores"]["countsT"].attrs["complete"] is True
    assert DataStore(str(path), zarr_mode="r").GeneScores.rawData.shape == (40, 3)


def test_repack_refuses_interrupted_derived_assay(tmp_path):
    path = tmp_path / "rna.zarr"
    counts = _counts()
    store = _write_store(path, counts)
    store.RNA.feats.insert("module", np.repeat([1, 2], 6), overwrite=True)
    # A hard kill inside the transaction never runs its cleanup; keep the
    # context alive so garbage collection does not run it either.
    context = derived_assay_transaction(
        store.z, "MODULES", None, operation="add_grouped_assay"
    )
    transaction = context.__enter__()
    partial = transaction.create_counts(store.cells.N, ["group_1"], ["group_1"])
    partial[:10] = 1.0

    for destination, data_only in (("data.zarr", True), ("full.zarr", False)):
        with pytest.raises(ValueError, match="discard_interrupted_assay"):
            repack_store(str(path), str(tmp_path / destination), data_only=data_only)
        assert not (tmp_path / destination).exists()
    _assert_openable_without(path, "MODULES")

    reopened = DataStore(str(path), min_features_per_cell=0)
    with pytest.raises(ValueError, match=r"interrupted add_grouped_assay"):
        reopened.add_grouped_assay("module", assay_label="MODULES")
    with pytest.raises(ValueError, match="not an interrupted derived assay"):
        reopened.discard_interrupted_assay("RNA")
    assert "RNA" in zarr.open_group(str(path), mode="r")

    reopened.discard_interrupted_assay("MODULES")
    reopened.add_grouped_assay("module", assay_label="MODULES")
    repack_store(str(path), str(tmp_path / "after.zarr"), data_only=True)

    repacked = DataStore(str(tmp_path / "after.zarr"), zarr_mode="r")
    np.testing.assert_allclose(
        _stored_counts(repacked, "MODULES"),
        _stored_counts(reopened, "MODULES"),
    )
    del context


def test_repack_refuses_unfinalized_writer_counts(tmp_path):
    path = tmp_path / "rna.zarr"
    _write_store(path, _counts())
    root = zarr.open_group(str(path), mode="r+")
    extra = create_zarr_count_assay(root, "EXTRA", None, 40, ["a", "b"], ["a", "b"])
    extra[:20] = 1

    assert extra.attrs["complete"] is False
    with pytest.raises(ValueError, match="incomplete count matrix"):
        repack_store(str(path), str(tmp_path / "data.zarr"), data_only=True)

    finalize_counts(extra)
    assert extra.attrs["complete"] is True


def _memory_root(workspace: str | None = None) -> zarr.Group:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    if workspace is not None:
        root.create_group(workspace)
    create_cell_data(
        root,
        workspace,
        ids=np.array(["c0", "c1", "c2"]),
        names=np.array(["c0", "c1", "c2"]),
    )
    return root


def test_derived_assay_transaction_discards_on_keyboard_interrupt():
    root = _memory_root("ws")
    root["ws"].attrs["assayTypes"] = {"OTHER": "RNA"}

    with pytest.raises(KeyboardInterrupt):
        with derived_assay_transaction(
            root, "SCORES", "ws", operation="add_melded_assay"
        ) as transaction:
            counts = transaction.create_counts(3, ["f0"], ["f0"])
            assert "is_assay" not in transaction.group.attrs
            assert pending_assays(root) == [("SCORES", "ws", "add_melded_assay")]
            root["ws"].attrs["assayTypes"] = {"OTHER": "RNA", "SCORES": "RNA"}
            counts[:] = 1.0
            raise KeyboardInterrupt

    assert "SCORES" not in root["ws"]
    assert "SCORES" not in root["matrices"]
    assert root["ws"].attrs["assayTypes"] == {"OTHER": "RNA"}


def test_derived_assay_transaction_publishes_only_finalized_counts():
    root = _memory_root()

    with pytest.raises(RuntimeError, match="not finalized"):
        with derived_assay_transaction(
            root, "SCORES", None, operation="add_grouped_assay"
        ) as transaction:
            transaction.create_counts(3, ["f0"], ["f0"])
    assert "SCORES" not in root

    with derived_assay_transaction(
        root, "SCORES", None, operation="add_grouped_assay"
    ) as transaction:
        counts = transaction.create_counts(3, ["f0"], ["f0"])
        counts[:] = 2.0
        finalize_counts(counts)
        transaction.group.attrs["provenance"] = "kept"

    attrs = dict(root["SCORES"].attrs)
    assert attrs["is_assay"] is True
    assert attrs["provenance"] == "kept"
    assert PENDING_ASSAY_ATTR not in attrs
    with pytest.raises(ValueError, match="already has metadata"):
        validate_new_assay(root, "SCORES", None)
    with pytest.raises(ValueError, match="not an interrupted derived assay"):
        discard_pending_assay(root, "SCORES", None)
    assert "SCORES" in root


def test_pending_assay_names_its_discard_path():
    root = _memory_root()
    context = derived_assay_transaction(root, "SCORES", None, operation="op")
    context.__enter__().create_counts(3, ["f0"], ["f0"])

    with pytest.raises(ValueError, match=r"discard_interrupted_assay\('SCORES'\)"):
        validate_new_assay(root, "SCORES", None)
    assert discard_pending_assay(root, "SCORES", None) is True
    assert "SCORES" not in root
    assert discard_pending_assay(root, "SCORES", None, missing_ok=True) is False
    del context

    root = _memory_root("ws1")
    context = derived_assay_transaction(root, "SCORES", "ws1", operation="op")
    context.__enter__().create_counts(3, ["f0"], ["f0"])
    with pytest.raises(ValueError, match=r"opened with workspace='ws1'"):
        validate_new_assay(root, "SCORES", "ws1")
    del context


def test_grouped_assay_uses_one_read_and_matches_per_group_means(tmp_path):
    counts = _counts(n_features=40)
    counts[5] = 0
    store = _write_store(tmp_path / "rna.zarr", counts)
    modules = np.arange(40) % 8
    store.RNA.feats.insert("module", modules, overwrite=True)

    store.add_grouped_assay("module", assay_label="MODULES", exclude_values=[])

    groups = [np.flatnonzero(modules == value) for value in range(8)]
    actual = _stored_counts(store, "MODULES")
    np.testing.assert_allclose(
        actual, _lib_size_group_means(counts, groups, store.RNA.sf)
    )
    # A zero-total cell has zero group means, not NaN.
    np.testing.assert_array_equal(actual[5], 0.0)
    assert np.isfinite(store.cells.fetch_all("MODULES_nCounts")).all()


def test_grouped_assay_skips_missing_group_values(tmp_path):
    counts = _counts()
    store = _write_store(tmp_path / "rna.zarr", counts)
    modules = np.array([1, 1, np.nan, 2, 2, np.nan, 1, 2, -1, 3, 3, np.nan])
    store.RNA.feats.insert("module", modules, overwrite=True)

    store.add_grouped_assay("module", assay_label="MODULES")

    assert list(store.MODULES.feats.fetch_all("ids")) == [
        "group_1.0",
        "group_2.0",
        "group_3.0",
    ]
    expected = _lib_size_group_means(
        counts,
        [np.flatnonzero(modules == value) for value in (1, 2, 3)],
        store.RNA.sf,
    )
    np.testing.assert_allclose(_stored_counts(store, "MODULES"), expected)

    labels = np.array(["b", "a", None, "a", "b", "b", "a", "a", "b", "b", "a", "a"])
    store.RNA.feats.insert("labels", labels, overwrite=True)
    store.add_grouped_assay("labels", assay_label="LABELS")
    assert list(store.LABELS.feats.fetch_all("ids")) == ["group_a", "group_b"]


def _tfidf_group_means(counts: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    totals = counts.sum(axis=1).astype(np.float64)
    totals[totals == 0] = 1
    idf = np.log2(1 + counts.shape[0] / (np.count_nonzero(counts, axis=0) + 1))
    normalized = counts / totals[:, None] * idf
    return np.column_stack([normalized[:, group].mean(axis=1) for group in groups])


def test_atac_grouped_means_fit_idf_once_and_ignore_band_size(tmp_path):
    counts = _counts(n_features=10)
    peaks = [f"chr1:{100 * i}-{100 * i + 50}" for i in range(1, 11)]
    store = _write_store(
        tmp_path / "atac.zarr", counts, assay="ATAC", feature_ids=peaks
    )
    groups = [np.array([0, 3, 7]), np.array([1, 2]), np.array([4, 5, 6, 8, 9])]
    cells = np.arange(store.cells.N)

    whole = np.vstack(list(iter_feature_group_means(store.ATAC, cells, groups)))
    banded = np.vstack(
        list(iter_feature_group_means(store.ATAC, cells, groups, block_rows=7))
    )
    per_group = np.column_stack(
        [
            store.ATAC.normed(cell_idx=cells, feat_idx=group).mean(axis=1).compute()
            for group in groups
        ]
    )

    np.testing.assert_array_equal(banded, whole)
    np.testing.assert_allclose(whole, per_group)
    np.testing.assert_allclose(whole, _tfidf_group_means(counts, groups))

    labels = np.full(10, -1)
    for value, group in enumerate(groups):
        labels[group] = value
    store.ATAC.feats.insert("module", labels, overwrite=True)
    store.add_grouped_assay("module", assay_label="MODULES")
    np.testing.assert_allclose(_stored_counts(store, "MODULES"), whole)


def test_adt_grouped_means_fit_clr_once_and_ignore_band_size(tmp_path):
    counts = _counts(n_features=6)
    store = _write_store(tmp_path / "adt.zarr", counts, assay="ADT")
    groups = [np.array([0, 5]), np.array([1, 2, 3]), np.array([4])]
    cells = np.arange(store.cells.N)

    whole = np.vstack(list(iter_feature_group_means(store.ADT, cells, groups)))
    banded = np.vstack(
        list(iter_feature_group_means(store.ADT, cells, groups, block_rows=3))
    )
    geometric = np.exp(np.log1p(counts).mean(axis=0))
    normalized = np.log1p(counts / geometric)
    expected = np.column_stack([normalized[:, group].mean(axis=1) for group in groups])

    np.testing.assert_array_equal(banded, whole)
    np.testing.assert_allclose(whole, expected)


def test_iter_feature_group_means_rejects_empty_groups(tmp_path):
    store = _write_store(tmp_path / "rna.zarr", _counts())
    with pytest.raises(ValueError, match="non-empty"):
        list(iter_feature_group_means(store.RNA, np.arange(3), [np.array([], int)]))
    with pytest.raises(ValueError, match="non-empty"):
        list(store.RNA._iter_feature_group_means(np.arange(3), []))


def test_iter_feature_group_means_yields_nothing_for_no_cells(tmp_path):
    store = _write_store(tmp_path / "rna.zarr", _counts())
    groups = [np.array([0, 1])]
    empty = np.array([], dtype=np.int64)
    assert list(store.RNA._iter_feature_group_means(empty, groups)) == []
    assert list(iter_feature_group_means(store.RNA, empty, groups)) == []


def test_grouped_assay_from_aggregation_ignores_unclustered_features(
    datastore, pseudotime_aggregation, tmp_path
):
    source = datastore.load_pseudotime_aggregation(pseudotime_aggregation)
    aggregation = datastore.run_pseudotime_aggregation(
        source.pseudotime,
        features=source.feature_selection,
        n_clusters=15,
        window_size=50,
        chunk_size=10,
        nan_cluster_value=0,
    )
    copy = tmp_path / "copy.zarr"
    shutil.copytree(datastore.zarr_loc, copy)
    store = DataStore(str(copy), default_assay="RNA")
    loaded = store.load_pseudotime_aggregation(aggregation)
    clusters = sorted(set(loaded.feature_clusters.tolist()))
    assert 0 not in clusters

    with pytest.raises(ValueError, match="exclude_values"):
        store.add_grouped_assay(aggregation, assay_label="MODULES", exclude_values=[0])
    store.add_grouped_assay(aggregation, assay_label="MODULES")

    # Unclustered features carry nan_cluster_value=0 and must not form group_0.
    assert list(store.MODULES.feats.fetch_all("ids")) == [
        f"group_{cluster}" for cluster in clusters
    ]
    cells = np.arange(store.cells.N)
    first = np.sort(loaded.feature_indices[loaded.feature_clusters == clusters[0]])
    expected = store.RNA.normed(cell_idx=cells, feat_idx=first).mean(axis=1).compute()
    np.testing.assert_allclose(_stored_counts(store, "MODULES")[:, 0], expected)
    assert store.MODULES.z.attrs["grouped_group_artifact"] == aggregation.to_dict()
