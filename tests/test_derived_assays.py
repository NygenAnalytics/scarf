"""Derived-assay creation: atomic publication, rollback, and grouped means."""

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

from scarf.assay.classification import declared_assay_type
from scarf.assay.normalization import iter_feature_group_means
from scarf.datastore.datastore import DataStore
from scarf.metadata.membership import membership_attributes
from scarf.storage.count_matrix import CountMatrixPolicy
from tests.storage_helpers import finalize_test_counts
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
from scarf.writers._store import write_membership_column

# Four count shards of ten cells each, so one shard can fail on its own. The
# counts store as uint8, one byte for each of the twelve features of a row.
_SMALL_SHARDS = CountMatrixPolicy(unitBytes=120, chunkBytes=24)


def _counts(n_cells: int = 40, n_features: int = 12, seed: int = 7) -> np.ndarray:
    counts = np.random.default_rng(seed).integers(0, 6, (n_cells, n_features))
    counts[:, 0] += 1
    return counts.astype(np.uint32)


def _write_counts(
    path: Path,
    counts: np.ndarray,
    *,
    assay: str = "RNA",
    feature_ids: list[str] | None = None,
) -> None:
    SparseToZarr(
        csr_matrix(counts),
        zarr_loc=str(path),
        cell_ids=[f"c{i}" for i in range(counts.shape[0])],
        feature_ids=feature_ids or [f"g{i}" for i in range(counts.shape[1])],
        assay_name=assay,
        nthreads=1,
        policy=_SMALL_SHARDS,
    ).dump(batch_size=10)


def _write_store(
    path: Path,
    counts: np.ndarray,
    *,
    assay: str = "RNA",
    feature_ids: list[str] | None = None,
) -> DataStore:
    _write_counts(path, counts, assay=assay, feature_ids=feature_ids)
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
    partial = transaction.create_counts(
        store.cells.N, ["group_1"], ["group_1"], np.float64
    )
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


# Every fourth cell was not measured, so its counts are zero.
_MEMBERS = np.arange(40) % 4 != 0


def _partial_store(
    path: Path, counts: np.ndarray, *, assay: str = "RNA", **options
) -> DataStore:
    """Write a store whose assay measured only the cells of ``_MEMBERS``.

    The membership column is written as an import writes it.
    """
    counts = counts * _MEMBERS[:, None].astype(counts.dtype)
    _write_counts(path, counts, assay=assay, **options)
    write_membership_column(
        zarr.open_group(str(path), mode="r+")["cellData"], assay, _MEMBERS
    )
    return DataStore(str(path), default_assay=assay, min_features_per_cell=-1)


def _melded_inputs(tmp_path: Path) -> tuple[list[str], dict[str, object]]:
    """Return peak IDs and the arguments of a melded gene-score assay."""
    peaks = [
        "chr1:100-200",
        "chr1:250-350",
        "chr1:400-500",
        "chr2:100-200",
        "chr2:300-400",
        "chr2:600-700",
    ]
    bed = tmp_path / "genes.bed"
    pd.DataFrame(
        [
            ("chr1", 120, 300, "gene_a", "GENE_A", "+"),
            ("chr1", 420, 480, "gene_b", "GENE_B", "+"),
            ("chr2", 150, 650, "gene_c", "GENE_C", "+"),
        ]
    ).to_csv(bed, sep="\t", header=False, index=False)
    arguments = {
        "from_assay": "ATAC",
        "external_bed_fn": str(bed),
        "assay_label": "GeneScores",
        "assay_type": "RNA",
        "renormalization": False,
    }
    return peaks, arguments


@pytest.mark.slow
@pytest.mark.parametrize("partial", [True, False])
def test_derived_assays_measure_the_cells_of_their_source_assay(tmp_path, partial):
    write = _partial_store if partial else _write_store
    grouped = write(tmp_path / "rna.zarr", _counts())
    grouped.RNA.feats.insert("module", np.repeat([1, 2, 3], 4), overwrite=True)
    peaks, arguments = _melded_inputs(tmp_path)
    melded = write(
        tmp_path / "atac.zarr", _counts(n_features=6), assay="ATAC", feature_ids=peaks
    )

    grouped.add_grouped_assay("module", assay_label="MODULES")
    melded.add_melded_assay(**arguments)

    for store, label in ((grouped, "MODULES"), (melded, "GeneScores")):
        reopened = DataStore(store.zarr_loc, zarr_mode="r")
        column = f"{label}_I"
        if not partial:
            # A source that measured every cell gives an assay that does too.
            assert column not in reopened.cells.columns
            continue
        np.testing.assert_array_equal(reopened.cells.fetch_all(column), _MEMBERS)
        attributes = reopened.cells._get_array(column).attrs.asdict()
        assert attributes == membership_attributes(label)
        # The cells outside the source assay hold none of the derived counts.
        assert not _stored_counts(reopened, label)[~_MEMBERS].any()


@pytest.mark.parametrize("writer", ["subset", "merge", "mount"])
def test_writers_refuse_a_source_with_a_pending_derived_assay(tmp_path, writer):
    from scarf.datastore.datastore import mount_datastore
    from scarf.merge import DataStoreMerge
    from scarf.writers import SubsetZarr

    path = tmp_path / "rna.zarr"
    store = _write_store(path, _counts())
    store.RNA.feats.insert("module", np.repeat([1, 2], 6), overwrite=True)
    # A hard kill inside the transaction never runs its cleanup; keep the
    # context alive so garbage collection does not run it either.
    context = derived_assay_transaction(
        store.z, "MODULES", None, operation="add_grouped_assay"
    )
    context.__enter__().create_counts(
        store.cells.N, ["group_1"], ["group_1"], np.float64
    )
    destination = tmp_path / "out.zarr"
    operation, subject = {
        "subset": ("subset", "The source store"),
        "merge": ("merged", "Source 'first'"),
        "mount": ("mounted", "The source store"),
    }[writer]
    message = (
        rf"^{subject} holds a pending derived assay and cannot be {operation}\. "
        r"Assay 'MODULES' is pending: .*discard_interrupted_assay\('MODULES'\)"
    )
    # Before, subset, merge, and mount copied the pending assay's membership
    # column with the cell metadata as finished data.
    with pytest.raises(ValueError, match=message):
        if writer == "subset":
            SubsetZarr(str(destination), [store.RNA], cell_idx=np.arange(4), nthreads=1)
        elif writer == "merge":
            DataStoreMerge(
                [store, store], str(destination), ["first", "second"], nthreads=1
            ).plan()
        else:
            mount_datastore(str(path), at=str(destination), min_features_per_cell=0)
    assert not destination.exists()
    assert pending_assays(zarr.open_group(str(path), mode="r")) == [
        ("MODULES", None, "add_grouped_assay")
    ]


def test_an_interrupted_derived_assay_is_discarded_with_its_membership(tmp_path):
    path = tmp_path / "rna.zarr"
    store = _partial_store(path, _counts())
    store.RNA.feats.insert("module", np.repeat([1, 2], 6), overwrite=True)
    # A hard kill inside the transaction never runs its cleanup; keep the
    # context alive so garbage collection does not run it either.
    context = derived_assay_transaction(
        store.z, "MODULES", None, operation="add_grouped_assay", membership="RNA_I"
    )
    transaction = context.__enter__()
    transaction.create_counts(store.cells.N, ["group_1"], ["group_1"], np.float64)

    reopened = DataStore(str(path), min_features_per_cell=-1)
    np.testing.assert_array_equal(reopened.cells.fetch_all("MODULES_I"), _MEMBERS)
    # The pending assay owns the column, so it cannot be rewritten or dropped.
    with pytest.raises(ValueError, match="reserved for the membership of assay"):
        reopened.cells.insert("MODULES_I", ~_MEMBERS, overwrite=True)
    with pytest.raises(ValueError, match="records which cells assay 'MODULES'"):
        reopened.cells.drop("MODULES_I")

    reopened.discard_interrupted_assay("MODULES")
    assert "MODULES_I" not in reopened.cells.columns
    reopened.add_grouped_assay("module", assay_label="MODULES")
    np.testing.assert_array_equal(reopened.cells.fetch_all("MODULES_I"), _MEMBERS)
    # Registering the assay leaves I unchanged, although the cells outside
    # _MEMBERS have no RNA counts.
    assert reopened.cells.fetch_all("I").all()
    assert DataStore(str(path), min_features_per_cell=-1).cells.fetch_all("I").all()
    del context


def test_an_interrupted_membership_write_keeps_its_pending_assay(tmp_path, monkeypatch):
    """A membership write that Ctrl-C stopped may still land, so the name stays held.

    Before, cleanup removed the pending assay at once, and a retry of the same
    name could receive the late requests of the interrupted write: the retried
    assay then held its own counts beside the first attempt's membership.
    """
    import scarf.storage.schema as schema

    path = tmp_path / "rna.zarr"
    store = _partial_store(path, _counts())
    store.RNA.feats.insert("module", np.repeat([1, 2], 6), overwrite=True)

    def stopped(*_args, **_kwargs):
        raise KeyboardInterrupt("stopped while writing the membership column")

    monkeypatch.setattr(schema, "create_streamed_metadata_column", stopped)
    with pytest.raises(KeyboardInterrupt):
        store.add_grouped_assay("module", assay_label="MODULES")
    monkeypatch.undo()
    assert pending_assays(zarr.open_group(str(path), mode="r")) == [
        ("MODULES", None, "add_grouped_assay")
    ]
    store.discard_interrupted_assay("MODULES")
    store.add_grouped_assay("module", assay_label="MODULES")
    np.testing.assert_array_equal(store.cells.fetch_all("MODULES_I"), _MEMBERS)


def test_repack_refuses_unfinalized_writer_counts(tmp_path):
    path = tmp_path / "rna.zarr"
    _write_store(path, _counts())
    root = zarr.open_group(str(path), mode="r+")
    extra = create_zarr_count_assay(
        root, "EXTRA", None, 40, ["a", "b"], ["a", "b"], np.uint8
    )
    extra[:20] = 1

    assert extra.attrs["complete"] is False
    with pytest.raises(ValueError, match="incomplete count matrix"):
        repack_store(str(path), str(tmp_path / "data.zarr"), data_only=True)

    finalize_test_counts(extra)
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


def test_derived_assay_transaction_keeps_the_pending_assay_on_keyboard_interrupt():
    root = _memory_root("ws")
    root["ws"].attrs["assayTypes"] = {"OTHER": "RNA"}

    with _warnings() as warnings, pytest.raises(KeyboardInterrupt):
        with derived_assay_transaction(
            root, "SCORES", "ws", operation="add_melded_assay"
        ) as transaction:
            counts = transaction.create_counts(3, ["f0"], ["f0"], np.float64)
            assert "is_assay" not in transaction.group.attrs
            assert pending_assays(root) == [("SCORES", "ws", "add_melded_assay")]
            root["ws"].attrs["assayTypes"] = {"OTHER": "RNA", "SCORES": "RNA"}
            counts[:] = 1.0
            raise KeyboardInterrupt

    # An interrupted write may still land, so the pending assay stays.
    assert pending_assays(root) == [("SCORES", "ws", "add_melded_assay")]
    assert len(warnings) == 1
    assert "discard_interrupted_assay('SCORES')" in warnings[0]
    assert discard_pending_assay(root, "SCORES", "ws") is True
    assert "SCORES" not in root["ws"]
    assert "SCORES" not in root["matrices"]
    assert root["ws"].attrs["assayTypes"] == {"OTHER": "RNA"}


def test_derived_assay_transaction_owns_the_membership_column_it_copies():
    root = _memory_root("ws")
    cells = root["ws/cellData"]
    members = np.array([True, False, True])
    write_membership_column(cells, "OTHER", members)

    with pytest.raises(RuntimeError, match="body failed"):
        with derived_assay_transaction(
            root, "SCORES", "ws", operation="add_melded_assay", membership="OTHER_I"
        ) as transaction:
            transaction.create_counts(3, ["f0"], ["f0"], np.float64)
            # The copy is written with the pending assay, attributes first.
            assert cells["SCORES_I"][:].tolist() == members.tolist()
            assert cells["SCORES_I"].attrs.asdict() == membership_attributes("SCORES")
            raise RuntimeError("body failed")
    assert "SCORES_I" not in cells
    assert "SCORES" not in root["ws"]

    with derived_assay_transaction(
        root, "SCORES", "ws", operation="add_melded_assay", membership="OTHER_I"
    ) as transaction:
        _finalized(transaction.create_counts(3, ["f0"], ["f0"], np.float64), 1.0)
    assert root["ws/SCORES"].attrs["is_assay"] is True
    assert cells["SCORES_I"][:].tolist() == members.tolist()
    np.testing.assert_array_equal(cells["OTHER_I"][:], members)

    # A column that holds the membership name of a new assay blocks it.
    write_membership_column(cells, "NEXT", members)
    with pytest.raises(ValueError, match="Cell column 'NEXT_I' already exists"):
        with derived_assay_transaction(root, "NEXT", "ws", operation="op"):
            pass
    assert "NEXT" not in root["ws"]
    assert cells["NEXT_I"].attrs.asdict() == membership_attributes("NEXT")


def test_discard_keeps_a_column_that_is_not_the_pending_assays_membership():
    root = _memory_root()
    context = derived_assay_transaction(root, "SCORES", None, operation="op")
    context.__enter__().create_counts(3, ["f0"], ["f0"], np.float64)
    # A writer outside Scarf added a plain column with the membership name.
    root["cellData"].create_array("SCORES_I", data=np.array([True, False, True]))

    assert discard_pending_assay(root, "SCORES", None) is True
    assert "SCORES" not in root
    assert root["cellData/SCORES_I"][:].tolist() == [True, False, True]
    del context


def test_derived_assay_transaction_publishes_only_finalized_counts():
    root = _memory_root()

    with pytest.raises(RuntimeError, match="not finalized"):
        with derived_assay_transaction(
            root, "SCORES", None, operation="add_grouped_assay"
        ) as transaction:
            transaction.create_counts(3, ["f0"], ["f0"], np.float64)
    assert "SCORES" not in root

    with derived_assay_transaction(
        root, "SCORES", None, operation="add_grouped_assay"
    ) as transaction:
        counts = transaction.create_counts(3, ["f0"], ["f0"], np.float64)
        counts[:] = 2.0
        finalize_test_counts(counts)
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
    # An array at an assay's path is not a pending assay either.
    root.create_array("ARRAY", data=np.zeros(3))
    with pytest.raises(ValueError, match="not an interrupted derived assay"):
        discard_pending_assay(root, "ARRAY", None)
    assert discard_pending_assay(root, "ARRAY", None, missing_ok=True) is False
    assert "ARRAY" in root


def test_empty_derived_assay_transaction_leaves_no_assay():
    root = _memory_root()

    with pytest.raises(RuntimeError, match="no counts to publish"):
        with derived_assay_transaction(
            root, "SCORES", None, operation="add_grouped_assay"
        ):
            pass

    assert "SCORES" not in root
    assert pending_assays(root) == []


def test_derived_assay_cannot_create_counts_twice():
    root = _memory_root()

    with pytest.raises(RuntimeError, match="already created"):
        with derived_assay_transaction(
            root, "SCORES", None, operation="add_grouped_assay"
        ) as transaction:
            transaction.create_counts(3, ["f0"], ["f0"], np.float64)
            transaction.create_counts(3, ["f1"], ["f1"], np.float64)

    assert "SCORES" not in root
    assert pending_assays(root) == []


def test_failed_rollback_keeps_pending_assay_recoverable(monkeypatch):
    from scarf.storage import schema

    root = _memory_root()

    def denied_cleanup(*_args, **_kwargs):
        raise PermissionError("storage temporarily unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(schema, "discard_pending_assay", denied_cleanup)
        with pytest.raises(RuntimeError, match="count write failed"):
            with derived_assay_transaction(
                root, "SCORES", None, operation="add_grouped_assay"
            ) as transaction:
                transaction.create_counts(3, ["f0"], ["f0"], np.float64)
                raise RuntimeError("count write failed")

    assert pending_assays(root) == [("SCORES", None, "add_grouped_assay")]
    assert root["SCORES"].attrs.get("is_assay") is not True
    assert discard_pending_assay(root, "SCORES", None)
    assert "SCORES" not in root


def test_pending_assay_names_its_discard_path():
    root = _memory_root()
    context = derived_assay_transaction(root, "SCORES", None, operation="op")
    context.__enter__().create_counts(3, ["f0"], ["f0"], np.float64)

    with pytest.raises(ValueError, match=r"discard_interrupted_assay\('SCORES'\)"):
        validate_new_assay(root, "SCORES", None)
    assert discard_pending_assay(root, "SCORES", None) is True
    assert "SCORES" not in root
    assert discard_pending_assay(root, "SCORES", None, missing_ok=True) is False
    del context

    root = _memory_root("ws1")
    context = derived_assay_transaction(root, "SCORES", "ws1", operation="op")
    context.__enter__().create_counts(3, ["f0"], ["f0"], np.float64)
    with pytest.raises(
        ValueError,
        match=r"Assay 'SCORES' of workspace 'ws1' is pending.*workspace='ws1'",
    ):
        validate_new_assay(root, "SCORES", "ws1")
    del context


def _finalized(counts: zarr.Array, value: float) -> None:
    counts[:] = value
    finalize_test_counts(counts)


def test_a_failed_writer_never_removes_another_writers_assay():
    root = _memory_root("ws")

    with derived_assay_transaction(
        root, "SCORES", "ws", operation="add_grouped_assay"
    ) as first:
        with pytest.raises(ValueError) as raised:
            with derived_assay_transaction(
                root, "SCORES", "ws", operation="add_grouped_assay"
            ) as second:
                # Both writers checked the free name before either created it.
                counts = first.create_counts(3, ["f0"], ["f0"], np.float64)
                second.create_counts(3, ["f1"], ["f1"], np.float64)
        # The second writer neither removed the first one's assay nor called
        # it interrupted.
        assert pending_assays(root) == [("SCORES", "ws", "add_grouped_assay")]
        assert "another process may still be running add_grouped_assay" in str(
            raised.value
        )
        _finalized(counts, 1.0)

    attrs = dict(root["ws/SCORES"].attrs)
    assert attrs["is_assay"] is True
    assert PENDING_ASSAY_ATTR not in attrs
    np.testing.assert_array_equal(root["matrices/SCORES/counts"][:], np.ones((3, 1)))


@pytest.mark.parametrize("stop", [KeyboardInterrupt, OSError])
def test_an_interrupted_publication_leaves_the_assay_pending(monkeypatch, stop):
    root = _memory_root()
    original = zarr.Group.update_attributes

    def interrupted(group: zarr.Group, attributes: dict) -> zarr.Group:
        if attributes.get("is_assay") is True:
            # Ctrl-C, or a lost acknowledgement, stops the caller while Zarr's
            # I/O thread may still run the write.
            raise stop
        return original(group, attributes)

    try:
        with pytest.raises(stop):
            with derived_assay_transaction(
                root, "SCORES", None, operation="add_grouped_assay"
            ) as transaction:
                counts = transaction.create_counts(3, ["f0"], ["f0"], np.float64)
                _finalized(counts, 2.0)
                monkeypatch.setattr(zarr.Group, "update_attributes", interrupted)
    finally:
        monkeypatch.undo()

    # The publication may still land, so the complete counts stay in place.
    assert pending_assays(root) == [("SCORES", None, "add_grouped_assay")]
    assert root["SCORES/counts"].attrs["complete"] is True
    # An operator who confirmed that no process is writing it removes it.
    assert discard_pending_assay(root, "SCORES", None)
    assert "SCORES" not in root


def test_derived_assay_publishes_on_a_store_without_atomic_creates(tmp_path):
    from zarr.storage import FsspecStore

    store = FsspecStore.from_url(f"memory://scarf-derived/{tmp_path.name}")
    root = zarr.open_group(store=store, mode="w")

    with derived_assay_transaction(
        root, "SCORES", None, operation="add_grouped_assay"
    ) as transaction:
        _finalized(transaction.create_counts(3, ["f0"], ["f0"], np.float64), 1.0)

    assert root["SCORES"].attrs["is_assay"] is True
    np.testing.assert_array_equal(root["SCORES/counts"][:], np.ones((3, 1)))


def _workspaces_root(*workspaces: str) -> zarr.Group:
    """Return a store whose workspaces share one ``matrices`` group."""
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_group("matrices")
    for workspace in workspaces:
        root.create_group(workspace)
        create_cell_data(
            root,
            workspace,
            ids=np.array(["c0", "c1", "c2"]),
            names=np.array(["c0", "c1", "c2"]),
        )
    return root


@contextmanager
def _warnings() -> Iterator[list[str]]:
    from scarf.utils import logger

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _derive(root: zarr.Group, workspace: str, value: float) -> None:
    with derived_assay_transaction(
        root, "SCORES", workspace, operation="add_grouped_assay"
    ) as transaction:
        _finalized(transaction.create_counts(3, ["f0"], ["f0"], np.float64), value)


def test_a_name_pending_in_one_workspace_is_taken_in_every_workspace():
    root = _workspaces_root("ws1", "ws2")
    # A write in ws2 was interrupted before it created its matrix group.
    root.create_group(
        "ws2/SCORES",
        attributes={PENDING_ASSAY_ATTR: "add_grouped_assay", "prepared": False},
    )

    # Discarding it deletes matrices/SCORES, so ws1 must not publish there.
    with pytest.raises(ValueError, match=r"opened with workspace='ws2'"):
        _derive(root, "ws1", 1.0)
    assert "SCORES" not in root["ws1"] and "SCORES" not in root["matrices"]

    assert discard_pending_assay(root, "SCORES", "ws2") is True
    _derive(root, "ws1", 1.0)
    assert root["ws1/SCORES"].attrs["is_assay"] is True


def test_a_failed_writer_keeps_a_matrix_group_it_did_not_create(monkeypatch):
    from zarr.errors import ContainsGroupError

    from scarf.storage import schema

    root = _workspaces_root("ws1", "ws2")
    _derive(root, "ws1", 1.0)
    # The ws2 writer checked the free name before ws1 published SCORES.
    monkeypatch.setattr(schema, "validate_new_assay", lambda *_args: None)

    with pytest.raises(ContainsGroupError):
        _derive(root, "ws2", 2.0)

    assert "SCORES" not in root["ws2"]
    assert root["ws1/SCORES"].attrs["is_assay"] is True
    np.testing.assert_array_equal(root["matrices/SCORES/counts"][:], np.ones((3, 1)))


def test_derived_assays_of_a_workspace_publish_their_matrices(tmp_path):
    path = tmp_path / "ws.zarr"
    peaks, arguments = _melded_inputs(tmp_path)
    SparseToZarr(
        csr_matrix(_counts(n_features=6)),
        zarr_loc=str(path),
        cell_ids=[f"c{i}" for i in range(40)],
        feature_ids=peaks,
        assay_name="ATAC",
        workspace="ws1",
        nthreads=1,
        policy=_SMALL_SHARDS,
    ).dump(batch_size=10)
    store = DataStore(
        str(path), workspace="ws1", default_assay="ATAC", min_features_per_cell=0
    )
    store.ATAC.feats.insert("module", np.repeat([1, 2], 3), overwrite=True)

    store.add_melded_assay(**{**arguments, "assay_type": "GeneActivity"})
    store.add_grouped_assay("module", from_assay="ATAC", assay_label="MODULES")

    # Every assay of the session carries its type, including the gene scores
    # that were rebuilt when the grouped assay was registered.
    types = {"ATAC": "ATAC", "MODULES": "Assay", "GeneScores": "GeneActivity"}
    assert {
        name: declared_assay_type(store.get_assay(name)) for name in store.assay_names
    } == types
    root = zarr.open_group(str(path), mode="r")
    assert root["ws1"].attrs["assayTypes"] == types
    for name in ("MODULES", "GeneScores"):
        assert root[f"ws1/{name}"].attrs["is_assay"] is True
        for group in (root[f"ws1/{name}"], root[f"matrices/{name}"]):
            assert PENDING_ASSAY_ATTR not in group.attrs
    # RNA-class gene scores keep countsT beside their counts in the matrix group.
    assert root["matrices/GeneScores/countsT"].attrs["complete"] is True
    assert pending_assays(root) == []
    reopened = DataStore(str(path), workspace="ws1", zarr_mode="r")
    assert {"MODULES", "GeneScores"} <= set(reopened.assay_names)


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


def test_rna_grouped_means_read_totals_once_across_bands(tmp_path, monkeypatch):
    from scarf.assay import RNAassay

    counts = _counts(n_features=12)
    counts[5] = 0
    store = _write_store(tmp_path / "rna.zarr", counts)
    groups = [np.array([0, 3, 7]), np.array([1, 2]), np.array([4, 5, 6, 8, 9, 11])]
    cells = np.arange(store.cells.N)
    whole = np.vstack(list(store.RNA._iter_feature_group_means(cells, groups)))
    keyed = {str(index): group for index, group in enumerate(groups)}
    per_band = np.vstack(
        [
            np.column_stack(
                list(
                    store.RNA._mean_normed_feature_groups(
                        cells[start : start + 7], keyed
                    ).values()
                )
            )
            for start in range(0, len(cells), 7)
        ]
    )
    reads: list[int] = []
    original = RNAassay._cell_count_totals

    def counted(self: RNAassay, cell_idx: np.ndarray) -> np.ndarray:
        reads.append(len(cell_idx))
        return original(self, cell_idx)

    monkeypatch.setattr(RNAassay, "_cell_count_totals", counted)
    banded = np.vstack(
        list(store.RNA._iter_feature_group_means(cells, groups, block_rows=7))
    )

    assert reads == [len(cells)]
    np.testing.assert_array_equal(banded, whole)
    np.testing.assert_array_equal(banded, per_band)
    np.testing.assert_allclose(
        whole, _lib_size_group_means(counts, groups, store.RNA.sf)
    )


@pytest.mark.parametrize("block_rows", [1, 7])
def test_rna_grouped_means_honor_a_different_normalizer(
    tmp_path, monkeypatch, block_rows
):
    from scarf.assay.normalization import norm_clr

    counts = _counts(n_features=6)
    store = _write_store(tmp_path / "rna.zarr", counts)
    monkeypatch.setattr(store.RNA, "normMethod", norm_clr)
    cells = np.arange(1, store.cells.N, 2)
    groups = [np.array([0, 3, 5]), np.array([1, 2, 4])]
    selected = counts[cells].astype(np.float64)
    geometric_means = np.exp(np.log1p(selected).mean(axis=0))
    normalized = np.log1p(selected / geometric_means)
    expected = np.column_stack([normalized[:, group].mean(axis=1) for group in groups])

    actual = np.vstack(
        list(store.RNA._iter_feature_group_means(cells, groups, block_rows=block_rows))
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


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
    # A copy keeps the shared session store free of this test's results.
    copy = tmp_path / "copy.zarr"
    shutil.copytree(datastore.zarr_loc, copy)
    store = DataStore(str(copy), default_assay="RNA")
    aggregation = store.run_pseudotime_aggregation(
        source.pseudotime,
        features=source.feature_selection,
        n_clusters=15,
        window_size=50,
        chunk_size=10,
        nan_cluster_value=0,
    )
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


def test_add_grouped_assay_fits_the_count_layout_to_the_budget(tmp_path):
    from scarf.storage.count_matrix import (
        DEFAULT_COUNT_MATRIX_POLICY,
        load_count_matrix_plan,
        policy_from_payload,
    )

    path = tmp_path / "rna.zarr"
    counts = _counts(n_cells=4_000, n_features=200, seed=3)
    # Source chunks of 100 cells keep the reads of the group means small.
    SparseToZarr(
        csr_matrix(counts),
        zarr_loc=str(path),
        cell_ids=[f"c{i}" for i in range(4_000)],
        feature_ids=[f"g{i}" for i in range(200)],
        nthreads=1,
        policy=CountMatrixPolicy(unitBytes=100_000, chunkBytes=20_000),
    ).dump()
    # The 3.2 MB of group means fit one default band, but the write of that
    # band does not fit this budget.
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, mem_budget="8M"
    )
    modules = np.repeat(np.arange(100), 2)
    store.RNA.feats.insert("module", modules, overwrite=True)
    store.add_grouped_assay("module", assay_label="MODULES")

    grouped = zarr.open_group(str(path), mode="r")["MODULES/counts"]
    policy = policy_from_payload(load_count_matrix_plan(grouped))
    assert policy.unitBytes < DEFAULT_COUNT_MATRIX_POLICY.unitBytes
    assert grouped.metadata.shards[0] < 4_000
    expected = _lib_size_group_means(
        counts,
        [np.flatnonzero(modules == value) for value in range(100)],
        store.RNA.sf,
    )
    np.testing.assert_allclose(grouped[:], expected)
