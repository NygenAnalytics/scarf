"""Merge, repack, assay schema, and count identity paths."""

import runpy
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.merge.datasets as merge_datasets
from scarf import DataStore, mount_datastore
from scarf.assay import ADTassay, ATACassay, RNAassay
from scarf.merge import DataStoreMerge
from scarf.merge.features import align_features
from scarf.metadata import MetaData
from scarf.storage.arrays import create_metadata_column
from scarf.storage.count_matrix import CountMatrixPolicy, create_product_counts_array
from scarf.storage.identity import CountSummary, finalize_counts
from scarf.storage.stores import MATRIX_SOURCE_ATTR, load_zarr
from scarf.tools import repack_zarr
from scarf.tools.repack_zarr import repack_store
from scarf.utils import digest
from scarf.utils.logging import logger
from scarf.writers import create_zarr_count_assay
from scarf.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts

_NAMES = ["left", "right"]
_COUNTS = np.random.default_rng(7).poisson(2.0, size=(6, 3)).astype(np.uint16)
_MOUNTED = {MATRIX_SOURCE_ATTR: {"location": "elsewhere"}}


def _write_cells(root: zarr.Group, n_cells: int) -> None:
    """Write cell metadata whose chunks span the cells, as a small import has."""
    cells = root.create_group("cellData")
    ids = np.asarray([f"c{index}" for index in range(n_cells)])
    for column, values in (
        ("ids", ids),
        ("names", ids),
        ("I", np.ones(n_cells, dtype=bool)),
    ):
        create_metadata_column(
            cells, column, data=values, dtype=values.dtype, chunkSize=n_cells
        )


def _write_source(
    location: str | MemoryStore,
    counts: Mapping[str, np.ndarray],
    *,
    ids: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
    dtype: str = "uint16",
) -> str | MemoryStore:
    """Write a fresh import whose assays share one set of cells.

    ``ids`` and ``names`` describe the features of every assay; by default
    they are ``f"{assay}{index}"``.
    """
    root = load_zarr(location, mode="w")
    (n_cells,) = {len(values) for values in counts.values()}
    _write_cells(root, n_cells)
    for assay, values in counts.items():
        feature_ids = (
            list(ids)
            if ids is not None
            else [f"{assay}{index}" for index in range(values.shape[1])]
        )
        array = create_zarr_count_assay(
            root,
            assay,
            None,
            n_cells,
            feature_ids,
            feature_ids if names is None else list(names),
            dtype,
        )
        array[:] = values
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, None)
    return location


def _open(location: str | MemoryStore) -> DataStore:
    return DataStore(location, default_assay="RNA", min_features_per_cell=0, nthreads=1)


def _named_sources(values: np.ndarray, dtype: str) -> list[DataStore]:
    """Open sources whose features share names: A, A, B on the left, A, B, B."""
    return [
        _open(_write_source(MemoryStore(), {"RNA": values}, names=names, dtype=dtype))
        for names in (["A", "A", "B"], ["A", "B", "B"])
    ]


def _merge(sources: Sequence[Any], destination: Any, **options: Any) -> DataStoreMerge:
    return DataStoreMerge(
        list(sources), destination, list(_NAMES), seed=0, nthreads=1, **options
    )


def _copy(location: str, tmp_path: Path) -> str:
    destination = str(tmp_path / Path(location).name)
    shutil.copytree(location, destination)
    return destination


def _merged_counts(location: str, path: str = "RNA/counts") -> np.ndarray:
    return np.asarray(zarr.open_group(location, mode="r")[path][:])


def _rows_by_cell(location: str, path: str = "RNA/counts") -> dict[str, list]:
    """Return merged rows keyed by merged cell ID."""
    root = zarr.open_group(location, mode="r")
    ids = np.asarray(root["cellData/ids"][:]).astype(str)
    return {
        cell: row.tolist()
        for cell, row in zip(ids, np.asarray(root[path][:]), strict=True)
    }


def _source_rows(values_by_name: Mapping[str, np.ndarray]) -> dict[str, list]:
    """Return the expected merged rows of sources whose cells are c0, c1, ..."""
    return {
        f"{name}__c{index}": row.tolist()
        for name, values in values_by_name.items()
        for index, row in enumerate(values)
    }


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _before_opening(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[zarr.Group], None]
) -> None:
    """Change the destination between inspection and opening, as another writer could."""
    original = DataStoreMerge._open_destination

    def open_destination(self, manifest, inspection):
        change(zarr.open_group(self.zarr_path, mode="a"))
        return original(self, manifest, inspection)

    monkeypatch.setattr(DataStoreMerge, "_open_destination", open_destination)


def _empty(root: zarr.Group) -> None:
    for key in list(root.keys()):
        del root[key]
    root.attrs.put({})


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> list[DataStore]:
    """Two prepared sources of the same cells, which merges only read."""
    root = tmp_path_factory.mktemp("sources")
    return [
        _open(_write_source(str(root / f"{name}.zarr"), {"RNA": _COUNTS + offset}))
        for offset, name in enumerate(_NAMES)
    ]


@pytest.fixture(scope="module")
def merged(tmp_path_factory: pytest.TempPathFactory, sources) -> str:
    """A completed merge of ``sources``; tests change copies of it."""
    location = str(tmp_path_factory.mktemp("merged") / "merged.zarr")
    _merge(sources, location).dump()
    return location


@pytest.fixture(scope="module")
def merged_in_workspace(tmp_path_factory: pytest.TempPathFactory, sources) -> str:
    """A completed merge of ``sources`` into workspace ``ws``."""
    location = str(tmp_path_factory.mktemp("merged_ws") / "merged.zarr")
    _merge(sources, location, out_workspace="ws").dump()
    return location


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A prepared single-assay store; tests change copies of it."""
    location = str(tmp_path_factory.mktemp("prepared") / "prepared.zarr")
    _open(_write_source(location, {"RNA": _COUNTS}))
    return location


# Count identity and assay schema


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_row_summary_kernel_rejects_non_finite_counts(value):
    values = np.array([[value, 1.0], [0.0, 2.0]])
    outputs = (
        np.zeros((2, 2), dtype=np.uint64),
        np.zeros(2),
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
    )
    # The first nonzero value is checked before it is mixed into a digest.
    with pytest.raises(ValueError, match="must hold finite values"):
        digest.summarize_rows.py_func(values, values.view(np.uint64), *outputs)


def test_count_summary_refuses_blocks_and_windows_it_cannot_digest():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts = root.create_array("counts", shape=(4, 2), dtype=np.uint16)
    summary = CountSummary(counts)
    # The compiled kernel indexes rows and columns without bounds checks.
    with pytest.raises(ValueError, match="block has the wrong shape"):
        summary.update(0, np.ones((1, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="rows are outside their window"):
        summary.update(3, np.ones((2, 2), dtype=np.uint16))

    # A fingerprint over rows that were never summarized would be wrong.
    summary.update(0, np.ones((2, 2), dtype=np.uint16))
    with pytest.raises(RuntimeError, match="does not cover its row window"):
        finalize_counts(counts, summary=summary)
    window = CountSummary(counts, rows=(0, 2))
    window.update(0, np.ones((2, 2), dtype=np.uint16))
    with pytest.raises(RuntimeError, match="covers only part of the matrix"):
        finalize_counts(counts, summary=window)
    assert "complete" not in counts.attrs


@pytest.mark.parametrize(
    ("name", "n_cells", "message"),
    [
        ("", 2, "Assay names must be non-empty"),
        ("  ", 2, "Assay names must be non-empty"),
        ("a/b", 2, "must not contain path separators"),
        ("a\\b", 2, "must not contain path separators"),
        ("RNA", -1, "Assay dimensions must be non-negative"),
    ],
)
def test_count_assay_creation_rejects_invalid_names_and_cell_counts(
    name, n_cells, message
):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ValueError, match=message):
        create_zarr_count_assay(root, name, None, n_cells, ["g0"], ["g0"], "uint8")
    assert list(root.group_keys()) == []


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "Count summaries do not match the counts"),
        ("fingerprint", "Count summaries do not match the counts"),
        ("length", "Count summaries are malformed"),
    ],
)
def test_first_preparation_refuses_count_summaries_of_other_counts(
    tmp_path, damage, message
):
    location = str(tmp_path / "import.zarr")
    _write_source(location, {"RNA": _COUNTS})
    matrix = zarr.open_group(location, mode="r+")["RNA"]
    if damage == "missing":
        del matrix["countSummaries"]
    elif damage == "fingerprint":
        matrix["countSummaries"].attrs["source_fingerprint"] = "0" * 64
    else:
        del matrix["countSummaries/rowSums"]
        matrix["countSummaries"].create_array("rowSums", data=np.zeros(2))

    with pytest.raises(ValueError, match=message):
        _open(location)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("unprepared", "is not prepared"),
        ("percentages", "Percentage definitions are malformed"),
        ("pattern", "Required percentage 'RNA_percentX' is missing"),
        ("column", "Required column 'RNA_nCounts' is inconsistent"),
        ("identity", "Copied dataset identity differs from the source"),
    ],
)
def test_repack_refuses_a_source_with_a_damaged_preparation_record(
    tmp_path, prepared, damage, message
):
    if damage == "unprepared":
        source = str(tmp_path / "import.zarr")
        _write_source(source, {"RNA": _COUNTS})
    else:
        source = _copy(prepared, tmp_path)
        root = zarr.open_group(source, mode="r+")
        if damage == "percentages":
            root["RNA"].attrs["percentFeatures"] = ["RNA_percentX"]
        elif damage == "pattern":
            root["RNA"].attrs["percentFeatures"] = {"RNA_percentX": 5}
        elif damage == "column":
            del root["cellData/RNA_nCounts"]
            root["cellData"].create_array(
                "RNA_nCounts", data=np.asarray(["x"] * _COUNTS.shape[0])
            )
        else:
            # The recorded identity is not recomputed until the copy publishes.
            root["RNA"].attrs["dataset_fingerprint"] = "0" * 64
    output = tmp_path / "repacked.zarr"

    with pytest.raises(ValueError, match=message):
        repack_store(source, str(output))
    assert not output.exists()


def test_repack_refuses_counts_stored_in_a_non_numeric_dtype(tmp_path):
    source = str(tmp_path / "foreign.zarr")
    root = load_zarr(source, mode="w")
    _write_cells(root, 2)
    counts = create_zarr_count_assay(root, "RNA", None, 2, ["g0"], ["g0"], "complex64")
    counts[:] = 1
    # A foreign writer marked them finalized; no Scarf writer can summarize them.
    counts.attrs.update({"complete": True, "content_fingerprint": "0" * 64})
    output = tmp_path / "rebuilt.zarr"

    with pytest.raises(ValueError, match="two-dimensional numeric array"):
        repack_store(source, str(output), data_only=True)
    assert not output.exists()


def test_only_cell_and_feature_tables_protect_prepared_columns(tmp_path, prepared):
    root = zarr.open_group(_copy(prepared, tmp_path), mode="r+")
    n_cells = _COUNTS.shape[0]
    with pytest.raises(ValueError, match="belongs to prepared data"):
        MetaData(root["cellData"]).insert(
            "RNA_nCounts", np.zeros(n_cells), overwrite=True
        )

    annotations = root.create_group("annotations")
    for column, values in (
        ("ids", np.asarray([f"c{index}" for index in range(n_cells)])),
        ("I", np.ones(n_cells, dtype=bool)),
        ("RNA_nCounts", np.zeros(n_cells)),
    ):
        create_metadata_column(annotations, column, data=values, dtype=values.dtype)
    table = MetaData(annotations)
    table.insert("RNA_nCounts", np.ones(n_cells), overwrite=True)
    np.testing.assert_array_equal(table.fetch_all("RNA_nCounts"), np.ones(n_cells))


# Merge planning and writing


def test_module_merges_hold_the_rows_of_both_sources(merged, merged_in_workspace):
    # Later tests compare resumed and rebuilt merges with these two.
    expected = _source_rows({"left": _COUNTS, "right": _COUNTS + 1})
    assert _rows_by_cell(merged) == expected
    workspace = zarr.open_group(merged_in_workspace, mode="r")
    ids = np.asarray(workspace["ws/cellData/ids"][:]).astype(str)
    counts = np.asarray(workspace["matrices/RNA/counts"][:])
    assert dict(zip(ids, counts.tolist(), strict=True)) == expected


def test_merge_keeps_the_assay_class_of_each_source_assay(tmp_path):
    counts = {"RNA": _COUNTS, "ADT": _COUNTS[:, :2] * 3, "ATAC": _COUNTS % 2}
    sources = [_open(_write_source(MemoryStore(), counts)) for _ in _NAMES]
    destination = str(tmp_path / "merged.zarr")

    _merge(sources, destination).dump()
    root = zarr.open_group(destination, mode="r")
    assert root.attrs["assayTypes"] == {"RNA": "RNA", "ADT": "ADT", "ATAC": "ATAC"}
    for name, values in counts.items():
        assert _rows_by_cell(destination, f"{name}/counts") == _source_rows(
            {"left": values, "right": values}
        )
    assert {name for name in counts if "countsT" in root[name]} == {"RNA"}
    merged = _open(destination)
    # ADT keeps CLR and ATAC keeps TF-IDF normalization.
    assert type(merged.get_assay("RNA")) is RNAassay
    assert type(merged.get_assay("ADT")) is ADTassay
    assert type(merged.get_assay("ATAC")) is ATACassay


def test_merge_reads_sources_that_live_in_memory(tmp_path):
    sources = [
        _open(_write_source(MemoryStore(), {"RNA": _COUNTS + offset}))
        for offset in range(2)
    ]
    destination = str(tmp_path / "merged.zarr")
    merger = _merge(sources, destination)

    # An in-memory source has no filesystem path that could overlap.
    assert merger.plan().canDump is True
    merger.dump()
    assert _rows_by_cell(destination) == _source_rows(
        {"left": _COUNTS, "right": _COUNTS + 1}
    )


def test_merge_requires_at_least_one_assay(tmp_path, sources):
    with pytest.raises(ValueError, match="No assays available to merge"):
        _merge(sources, str(tmp_path / "merged.zarr"), assays=[])


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("mounted", "Destination is a mounted matrix-source store"),
        ("workspace_array", "Destination workspace 'ws' is not a Zarr group"),
        ("workspace_matrices", "already uses workspace matrix storage"),
        ("foreign_overwrite", "overwrite=True can replace only a merge-owned"),
        ("foreign", "does not contain a matching DataStoreMerge manifest"),
    ],
)
def test_merge_refuses_destinations_it_cannot_own(
    tmp_path, sources, merged_in_workspace, case, reason
):
    destination = str(tmp_path / "merged.zarr")
    options: dict[str, Any] = {}
    if case == "mounted":
        mount_datastore(
            sources[0].zarr_loc,
            at=destination,
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=1,
        )
    elif case == "workspace_array":
        zarr.open_group(destination, mode="w").create_array(
            "ws", shape=(1,), dtype="uint8"
        )
        options["out_workspace"] = "ws"
    elif case == "workspace_matrices":
        destination = _copy(merged_in_workspace, tmp_path)
    else:
        zarr.open_group(destination, mode="w").create_group("sentinel")
        options["overwrite"] = case == "foreign_overwrite"
    merger = _merge(sources, destination, **options)

    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None and reason in plan.blockedReason
    with pytest.raises(ValueError, match=reason):
        merger.dump()


@pytest.mark.parametrize("workspace", [None, "ws"])
def test_merge_writes_into_an_empty_destination_shell(tmp_path, sources, workspace):
    destination = str(tmp_path / "shell.zarr")
    zarr.open_group(destination, mode="w")
    merger = _merge(sources, destination, out_workspace=workspace)

    plan = merger.plan()
    assert plan.canDump is True
    assert plan.cellDataAction == "write"
    result = merger.dump()
    assert {component.action for component in result.components} == {"write"}
    assert result.resumed is False


def test_merge_validates_the_preparation_of_a_prepared_destination(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    _open(destination)

    plan = _merge(sources, destination).plan()
    assert plan.canDump is True
    assert {plan.cellDataAction} | {
        action
        for assay in plan.assays
        for action in (assay.countsAction, assay.countsTAction)
    } == {"skip"}

    zarr.open_group(destination, mode="r+")["RNA"].attrs["counts_fingerprint"] = (
        "0" * 64
    )
    plan = _merge(sources, destination).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "inconsistent dataset identity" in plan.blockedReason


def test_merge_plan_refuses_to_resume_the_counts_of_a_prepared_assay(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    _open(destination)
    root = zarr.open_group(destination, mode="r+")
    root["RNA"].attrs["complete"] = False
    root.attrs.update({"scarf:import_complete": False, "complete": False})
    merger = _merge(sources, destination)

    # The preview agrees with dump, which must not delete the prepared assay.
    plan = merger.plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "damaged prepared assay requires a fresh destination" in plan.blockedReason
    with pytest.raises(ValueError, match="damaged prepared assay"):
        merger.dump()
    assert zarr.open_group(destination, mode="r")["RNA"].attrs["prepared"] is True


@pytest.mark.parametrize("overwrite", [False, True])
def test_merge_interrupted_before_cell_metadata_is_completed(
    tmp_path, monkeypatch, sources, merged, overwrite
):
    destination = str(tmp_path / "merged.zarr")

    def interrupt(*args, **kwargs):
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(merge_datasets, "write_cell_metadata", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            _merge(sources, destination).dump()
    interrupted = zarr.open_group(destination, mode="r")
    assert "cellData" not in interrupted
    assert "assayTypes" not in interrupted.attrs

    merger = _merge(sources, destination, overwrite=overwrite)
    assert merger.plan().cellDataAction == "write"
    result = merger.dump()
    actions = {component.name: component.action for component in result.components}
    assert actions["cellData"] == "write"
    assert actions["counts:RNA"] == ("write" if overwrite else "resume")
    np.testing.assert_array_equal(_merged_counts(destination), _merged_counts(merged))


@pytest.mark.parametrize("manifest", ["missing", "assays_not_a_list"])
def test_merge_overwrite_restarts_without_a_usable_stored_manifest(
    tmp_path, sources, merged, manifest
):
    destination = _copy(merged, tmp_path)
    root = zarr.open_group(destination, mode="r+")
    if manifest == "missing":
        del root.attrs["scarf:merge_manifest"]
    else:
        root.attrs["scarf:merge_manifest"] = {
            **root.attrs["scarf:merge_manifest"],
            "assays": "RNA",
        }

    result = _merge(sources, destination, overwrite=True).dump()
    assert {component.action for component in result.components} == {"write"}
    np.testing.assert_array_equal(_merged_counts(destination), _merged_counts(merged))


def test_merge_overwrite_removes_a_pipeline_group_it_leaves_empty(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    zarr.open_group(destination, mode="r+").create_group(
        f"pipeline/runs/{'a' * 64}/stages"
    )

    _merge(sources, destination, overwrite=True).dump()
    assert "pipeline" not in zarr.open_group(destination, mode="r")


def test_merge_refuses_a_destination_mounted_after_planning(
    tmp_path, monkeypatch, sources
):
    _before_opening(monkeypatch, lambda root: root.attrs.update(_MOUNTED))

    with pytest.raises(ValueError, match="mounted matrix-source store"):
        _merge(sources, str(tmp_path / "merged.zarr")).dump()


def test_merge_overwrite_refuses_a_destination_disowned_after_planning(
    tmp_path, monkeypatch, sources, merged
):
    destination = _copy(merged, tmp_path)

    def disown(root: zarr.Group) -> None:
        del root.attrs["scarf:import_source"]

    _before_opening(monkeypatch, disown)
    with pytest.raises(ValueError, match="changed after planning"):
        _merge(sources, destination, overwrite=True).dump()
    assert "RNA" in zarr.open_group(destination, mode="r")


@pytest.mark.parametrize("workspace", [None, "ws"])
def test_merge_overwrite_rebuilds_a_destination_emptied_after_planning(
    tmp_path, monkeypatch, sources, merged, merged_in_workspace, workspace
):
    template = merged if workspace is None else merged_in_workspace
    destination = _copy(template, tmp_path)
    path = "RNA/counts" if workspace is None else "matrices/RNA/counts"

    _before_opening(monkeypatch, _empty)
    _merge(sources, destination, out_workspace=workspace, overwrite=True).dump()
    np.testing.assert_array_equal(
        _merged_counts(destination, path), _merged_counts(template, path)
    )
    root = zarr.open_group(destination, mode="r")
    attributes = root.attrs if workspace is None else root[workspace].attrs
    assert attributes["complete"] is True


def test_merge_overwrite_rechecks_containment_after_clearing(
    tmp_path, monkeypatch, sources, merged
):
    destination = _copy(merged, tmp_path)
    original = DataStoreMerge._clear_merge_components

    def clear(self, root, stored_manifest):
        original(self, root, stored_manifest)
        root.attrs.update(_MOUNTED)

    monkeypatch.setattr(DataStoreMerge, "_clear_merge_components", clear)
    with pytest.raises(ValueError, match="mounted matrix-source store"):
        _merge(sources, destination, overwrite=True).dump()


def test_merge_finalization_rechecks_containment(
    tmp_path, monkeypatch, sources, merged
):
    destination = _copy(merged, tmp_path)
    # Every component is complete, but the final markers were not written.
    zarr.open_group(destination, mode="r+").attrs.update(
        {"scarf:import_complete": False, "complete": False}
    )
    original = merge_datasets.load_zarr

    def load(zarr_loc, mode, storage_options=None):
        if mode == "r+":
            zarr.open_group(zarr_loc, mode="r+").attrs.update(_MOUNTED)
        return original(zarr_loc, mode=mode, storage_options=storage_options)

    monkeypatch.setattr(merge_datasets, "load_zarr", load)
    with pytest.raises(ValueError, match="mounted matrix-source store"):
        _merge(sources, destination).dump()
    assert zarr.open_group(destination, mode="r").attrs["complete"] is False


def test_merge_resume_refuses_counts_of_another_feature_union(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    # The right counts under other feature IDs keep the source fingerprints.
    renamed = _open(
        _write_source(MemoryStore(), {"RNA": _COUNTS + 1}, ids=["RNA0", "RNA1", "x"])
    )

    plan = _merge([sources[0], renamed], destination).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "counts shape for 'RNA' is (12, 3), expected (12, 4)" in plan.blockedReason


def test_merge_resume_refuses_counts_narrower_than_the_names_now_summed(
    tmp_path, sources
):
    destination = str(tmp_path / "merged.zarr")
    _merge(sources, destination, feature_key="names").dump()
    # The right counts, now with two features of one name to sum.
    renamed = _open(
        _write_source(
            MemoryStore(), {"RNA": _COUNTS + 1}, names=["RNA0", "RNA0", "RNA1"]
        )
    )

    plan = _merge([sources[0], renamed], destination, feature_key="names").plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "counts dtype for 'RNA' is uint16, expected uint32" in plan.blockedReason


def test_merge_resume_refuses_counts_rewritten_in_another_layout(tmp_path, sources):
    destination = str(tmp_path / "merged.zarr")
    policy = CountMatrixPolicy(unitBytes=64, chunkBytes=16)
    _merge(sources, destination, policy=policy).dump()
    # Hand-written counts in another layout, finalized with a matching record.
    matrix = zarr.open_group(destination, mode="r+")["RNA"]
    values = np.asarray(matrix["counts"][:])
    del matrix["counts"]
    counts = create_product_counts_array(
        matrix,
        *values.shape,
        values.dtype,
        profile="fast_local",
        policy=CountMatrixPolicy(unitBytes=128, chunkBytes=16),
    )
    counts[:] = values
    finalize_test_counts(counts)

    plan = _merge(sources, destination, policy=policy).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "counts chunks for 'RNA' are (12, 1), expected (10, 1)" in plan.blockedReason


def test_merge_resume_refuses_feature_metadata_without_its_selection(
    tmp_path, sources, merged
):
    destination = _copy(merged, tmp_path)
    del zarr.open_group(destination, mode="r+")["RNA/featureData/I"]

    plan = _merge(sources, destination).plan()
    assert plan.canDump is False
    assert plan.blockedReason is not None
    assert "featureData/I is missing for 'RNA'" in plan.blockedReason


def test_merge_refuses_a_feature_profile_beyond_its_budget(tmp_path):
    counts = np.zeros((4, 25_000), dtype=np.uint16)
    counts[:, ::997] = 1
    sources = [_open(_write_source(MemoryStore(), {"RNA": counts})) for _ in _NAMES]
    destination = tmp_path / "merged.zarr"
    # Every phase holds the feature alignment. Count summaries and the
    # nonzero profile of the sources add about 40 bytes per feature, or 1 MB.
    alignment = align_features([source.get_assay("RNA") for source in sources], _NAMES)
    budget = alignment.resident_bytes() + 500_000

    with pytest.raises(MemoryError, match="Merged assay NNZ profile needs about"):
        _merge(sources, str(destination), mem_budget=budget).plan()
    assert not destination.exists()


def test_merge_warns_when_few_features_overlap(tmp_path):
    counts = np.ones((3, 11), dtype=np.uint16)
    left = _open(
        _write_source(
            MemoryStore(), {"RNA": counts}, ids=[f"x{index}" for index in range(11)]
        )
    )
    right = _open(
        _write_source(
            MemoryStore(),
            {"RNA": counts},
            ids=["x0", *(f"y{index}" for index in range(1, 11))],
        )
    )

    with _captured_warnings() as messages:
        plan = _merge([left, right], str(tmp_path / "merged.zarr")).plan()
    assert plan.assays[0].featureOverlapFraction == pytest.approx(1 / 21)
    assert any("Fewer than 10% of features overlap" in text for text in messages)


def test_merge_by_name_sums_signed_counts_in_a_wider_dtype(tmp_path):
    sources = _named_sources(np.array([[3, -2, 1], [0, 4, 2]]), "int16")
    destination = str(tmp_path / "merged.zarr")
    merger = _merge(sources, destination, feature_key="names")

    assert merger.plan().assays[0].dtype == "int32"
    merger.dump()
    root = zarr.open_group(destination, mode="r")
    assert root["RNA/counts"].dtype == np.int32
    # The left source sums features 0 and 1 into A, the right 1 and 2 into B.
    assert _rows_by_cell(destination) == {
        "left__c0": [1, 1],
        "left__c1": [4, 2],
        "right__c0": [3, -1],
        "right__c1": [0, 6],
    }


def test_merge_by_name_keeps_uint64_counts_it_cannot_widen(tmp_path):
    sources = _named_sources(np.array([[3, 2, 1], [0, 4, 2]]), "uint64")
    merger = _merge(sources, str(tmp_path / "merged.zarr"), feature_key="names")

    # No wider integer exists; a sum that overflows raises while it is written.
    assert merger.plan().assays[0].dtype == "uint64"


# Repacking


def test_repack_copies_arrays_of_any_dimensionality(tmp_path, prepared):
    source = _copy(prepared, tmp_path)
    extra = zarr.open_group(source, mode="r+").create_group("extra")
    cube = extra.create_array(
        "cube", data=np.arange(24, dtype=np.float32).reshape(2, 3, 4), chunks=(1, 3, 4)
    )
    cube.attrs["kind"] = "kernel"
    scalar = extra.create_array("scalar", data=np.array(7, dtype=np.int64))
    scalar.attrs["kind"] = "scalar"
    output = str(tmp_path / "repacked.zarr")

    repack_store(source, output)
    copied = zarr.open_group(output, mode="r")["extra"]
    np.testing.assert_array_equal(copied["cube"][:], cube[:])
    assert copied["cube"].chunks == (1, 3, 4)
    assert copied["cube"].attrs["kind"] == "kernel"
    assert copied["scalar"][...] == 7
    assert copied["scalar"].attrs["kind"] == "scalar"


@pytest.mark.parametrize("record", ["elsewhere", {"location": 5}, {}])
def test_repack_refuses_a_malformed_matrix_source_record(tmp_path, prepared, record):
    source = _copy(prepared, tmp_path)
    zarr.open_group(source, mode="r+").attrs[MATRIX_SOURCE_ATTR] = record
    output = tmp_path / "repacked.zarr"

    with pytest.raises(ValueError, match="Malformed matrix source location"):
        repack_store(source, str(output))
    assert not output.exists()


def test_repack_refuses_a_destination_inside_the_mounted_count_owner(
    tmp_path, prepared
):
    source = _copy(prepared, tmp_path)
    target = str(tmp_path / "mount.zarr")
    mount_datastore(
        source, at=target, default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    output = Path(source) / "copy.zarr"

    with pytest.raises(ValueError, match="overlaps the mounted count owner"):
        repack_store(target, str(output))
    assert not output.exists()


def test_repack_refuses_a_store_without_assays(tmp_path):
    source = str(tmp_path / "empty.zarr")
    zarr.open_group(source, mode="w")

    with pytest.raises(ValueError, match="No logical assays found"):
        repack_store(source, str(tmp_path / "repacked.zarr"))


def test_repack_refuses_an_interrupted_import(tmp_path, prepared):
    source = _copy(prepared, tmp_path)
    zarr.open_group(source, mode="r+").attrs["scarf:import_complete"] = False

    with pytest.raises(ValueError, match="An incomplete import cannot be repacked"):
        repack_store(source, str(tmp_path / "repacked.zarr"))


def test_repack_module_repacks_a_store_from_the_command_line(
    tmp_path, monkeypatch, capsys, prepared
):
    source = _copy(prepared, tmp_path)
    output = str(tmp_path / "repacked.zarr")
    monkeypatch.setattr(sys, "argv", ["repack_zarr", source, output, "--nthreads", "1"])
    # Executed as a script, the module must not shadow its imported copy.
    monkeypatch.delitem(sys.modules, "scarf.tools.repack_zarr")

    runpy.run_module("scarf.tools.repack_zarr", run_name="__main__")
    assert f"Repacked {source} -> {output}" in capsys.readouterr().out
    np.testing.assert_array_equal(_merged_counts(output), _COUNTS)


@pytest.mark.parametrize(
    ("options", "message"),
    [("{not json", "must be valid JSON"), ("[1]", "must be a JSON object")],
)
def test_repack_command_line_rejects_invalid_storage_options(
    monkeypatch, options, message
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["repack_zarr", "in.zarr", "out.zarr", "--storage-options", options],
    )
    with pytest.raises(SystemExit, match=message):
        repack_zarr.main()
