import asyncio
from typing import Any

import numpy as np
import pytest
import zarr
from scipy.sparse import coo_matrix
from zarr.storage import MemoryStore

import scarf.storage.selections as selections
import scarf.storage.sharding as sharding
from scarf.storage.arrays import create_metadata_column
from scarf.storage.artifacts import artifact_group
from scarf.storage.budget import ResourceBudget
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.geometry import ArrayGeometry
from scarf.storage.layout import ZarrArraySpec
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import ValidatedStoredSelection


def _selection_root(
    values: np.ndarray | None = None,
) -> tuple[zarr.Group, ArtifactRef]:
    mask = (
        np.asarray([True, False, True, False]) if values is None else np.asarray(values)
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    create_metadata_column(
        table,
        "ids",
        data=np.asarray([f"cell_{index}" for index in range(len(mask))]),
        dtype=str,
        chunkSize=2,
    )
    create_metadata_column(table, "I", data=mask, dtype=mask.dtype, chunkSize=2)
    ref = selections.resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="manual_selection",
        parameters={},
        inputs={},
        invalidate_cache=True,
    )
    return root, ref


def _selection_kwargs() -> dict[str, Any]:
    return {
        "kind": "cell_selection",
        "scope": "datastore",
        "assay": None,
        "table_path": "cellData",
    }


def _snapshot_root() -> tuple[zarr.Group, ArtifactRef]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    create_metadata_column(
        table,
        "ids",
        data=np.asarray(["c1", "c2", "c3"]),
        dtype=str,
    )
    create_metadata_column(
        table,
        "names",
        data=np.asarray(["A", "B", "C"]),
        dtype=str,
    )
    ref = selections.snapshot_run_metadata(
        root,
        table_path="cellData",
        id_column="ids",
        columns=("names",),
        axis="cell",
        invalidate_cache=True,
    )
    return root, ref


def _validate_snapshot(root: zarr.Group, ref: ArtifactRef) -> zarr.Group:
    return selections.validate_run_metadata_snapshot(
        root,
        ref,
        axis="cell",
        assay=None,
        table_path="cellData",
        ordered_columns=("names",),
    )


def test_selection_summary_reuse_and_block_guards() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    wrong = root.create_array("wrong", data=np.arange(3))
    with pytest.raises(TypeError, match="booleans"):
        selections._stored_selection_summary(wrong)

    validator = selections._selection_reuse_validator("fingerprint")
    assert (
        validator(ArtifactRef("datastore", "cell_selection", "1" * 64), root) is False
    )

    root, ref = _selection_root()
    validated = selections.validate_stored_selection_integrity(
        root,
        ref,
        **_selection_kwargs(),
    )
    with pytest.raises(ValueError, match="block_rows"):
        selections._selection_block_rows(validated, 0)

    inconsistent = ValidatedStoredSelection(
        ref=validated.ref,
        values=validated.values,
        row_ids=validated.row_ids,
        selected_count=validated.selected_count + 1,
        table_path=validated.table_path,
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        list(selections._iter_validated_selection_blocks(inconsistent, block_rows=2))
    assert caught.value.code == "selection_values_changed"


def test_selection_integrity_wraps_dependency_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, ref = _selection_root()
    monkeypatch.setattr(
        selections,
        "inspect_artifact",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad status")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        selections.validate_stored_selection_integrity(
            root,
            ref,
            **_selection_kwargs(),
        )
    assert caught.value.code == "artifact_missing"

    monkeypatch.undo()
    monkeypatch.setattr(
        selections,
        "artifact_group",
        lambda *_args: (_ for _ in ()).throw(KeyError("missing")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        selections.validate_stored_selection_integrity(
            root,
            ref,
            **_selection_kwargs(),
        )
    assert caught.value.code == "artifact_missing"

    monkeypatch.undo()
    original = selections.as_zarr_array

    def malformed_values(value: Any, *, name: str) -> zarr.Array:
        if name == "values":
            raise TypeError("not an array")
        return original(value, name=name)

    monkeypatch.setattr(selections, "as_zarr_array", malformed_values)
    with pytest.raises(ArtifactResolutionError) as caught:
        selections.validate_stored_selection_integrity(
            root,
            ref,
            **_selection_kwargs(),
        )
    assert caught.value.code == "selection_values_changed"

    monkeypatch.undo()
    monkeypatch.setattr(
        selections,
        "_stored_selection_summary",
        lambda *_args: (_ for _ in ()).throw(TypeError("bad mask")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        selections.validate_stored_selection_integrity(
            root,
            ref,
            **_selection_kwargs(),
        )
    assert caught.value.code == "selection_values_changed"


def test_selection_detects_late_row_and_live_alias_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, ref = _selection_root()
    expected = selections.fingerprint_stored_strings(root["cellData/ids"])
    fingerprints = iter((expected, "changed"))
    monkeypatch.setattr(
        selections,
        "fingerprint_stored_strings",
        lambda *_args: next(fingerprints),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        selections.validate_stored_selection_integrity(
            root,
            ref,
            **_selection_kwargs(),
        )
    assert caught.value.code == "row_identity_mismatch"

    monkeypatch.undo()
    del root["cellData/I"]
    root["cellData"].create_group("I")
    with pytest.raises(ArtifactResolutionError) as caught:
        selections.validate_stored_selection_live_alias(
            root,
            ref,
            column="I",
            **_selection_kwargs(),
        )
    assert caught.value.code == "selection_values_changed"

    del root["cellData/I"]
    root["cellData"].create_array("I", data=np.arange(4))
    with pytest.raises(ArtifactResolutionError) as caught:
        selections.validate_stored_selection_live_alias(
            root,
            ref,
            column="I",
            **_selection_kwargs(),
        )
    assert caught.value.code == "selection_values_changed"


def test_selection_alignment_rejects_bad_source_lengths() -> None:
    root, ref = _selection_root()
    with pytest.raises(ValueError, match="Compact values"):
        list(
            selections.iter_full_axis_selection_blocks(
                root,
                ref,
                np.arange(3),
                fill_value=-1,
                **_selection_kwargs(),
            )
        )
    with pytest.raises(ValueError, match="Full-axis values"):
        list(
            selections.iter_selected_axis_selection_blocks(
                root,
                ref,
                np.arange(3),
                **_selection_kwargs(),
            )
        )


class _ObjectStrings:
    ndim = 1
    dtype = np.dtype(object)
    chunks = None

    def __init__(self, values: list[Any]) -> None:
        self.values = np.asarray(values, dtype=object)
        self.shape = self.values.shape

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.values[key]


class _BooleanMask:
    ndim = 1
    dtype = np.dtype(bool)
    chunks = None

    def __init__(self, values: list[bool]) -> None:
        self.values = np.asarray(values, dtype=bool)
        self.shape = self.values.shape

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.values[key]


def test_selected_string_fingerprint_supports_object_backed_ids() -> None:
    digest, count = selections.fingerprint_selected_stored_strings(
        _ObjectStrings(["a", "long", "c"]),  # type: ignore[arg-type]
        _BooleanMask([True, True, False]),  # type: ignore[arg-type]
    )
    assert count == 2
    assert isinstance(digest, str) and len(digest) == 64


def test_selection_producers_reject_drift_and_invalid_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    table.create_array("ids", data=np.asarray(["a", "b"]))
    table.create_array("bad", data=np.arange(2))
    with pytest.raises(TypeError, match="booleans"):
        selections.resolve_stored_selection_artifact(
            root,
            table_path="cellData",
            id_column="ids",
            source_column="bad",
            scope="datastore",
            kind="cell_selection",
            operation="manual_selection",
            parameters={},
            inputs={},
        )
    table.create_array("mask", data=np.asarray([True]))
    with pytest.raises(ValueError, match="align"):
        selections.resolve_stored_selection_artifact(
            root,
            table_path="cellData",
            id_column="ids",
            source_column="mask",
            scope="datastore",
            kind="cell_selection",
            operation="manual_selection",
            parameters={},
            inputs={},
        )

    root, _ref = _selection_root()
    calls = 0
    original = selections._stored_selection_fingerprint

    def drifting_fingerprint(array: zarr.Array) -> str:
        nonlocal calls
        calls += 1
        return original(array) if calls == 1 else "f" * 64

    monkeypatch.setattr(
        selections, "_stored_selection_fingerprint", drifting_fingerprint
    )
    with pytest.raises(RuntimeError, match="changed"):
        selections.resolve_stored_selection_artifact(
            root,
            table_path="cellData",
            id_column="ids",
            source_column="I",
            scope="datastore",
            kind="cell_selection",
            operation="copy_again",
            parameters={},
            inputs={},
            invalidate_cache=True,
        )

    monkeypatch.undo()
    calls = 0

    def generated_drift(array: zarr.Array) -> str:
        nonlocal calls
        calls += 1
        return "0" * 64 if calls == 1 else "1" * 64

    monkeypatch.setattr(selections, "_stored_selection_fingerprint", generated_drift)
    with pytest.raises(RuntimeError, match="changed"):
        selections.resolve_generated_selection_artifact(
            root,
            scope="datastore",
            kind="cell_selection",
            values=np.asarray([True, False]),
            row_ids=np.asarray(["x", "y"]),
            operation="generated",
            parameters={},
            inputs={},
            source_column="generated",
            invalidate_cache=True,
        )


def test_snapshot_scalar_and_source_column_contracts() -> None:
    assert selections._snapshot_text(b"caf\xc3\xa9") == "caf\u00e9"
    assert selections._snapshot_text(None) == ""
    assert selections._snapshot_values_dtype(_ObjectStrings(["a", "long"])) == np.dtype(
        "U4"
    )

    class MatrixValues(_ObjectStrings):
        ndim = 2

        def __init__(self) -> None:
            self.values = np.asarray([["a"], ["b"]], dtype=object)
            self.shape = self.values.shape

    with pytest.raises(ValueError, match="one-dimensional"):
        selections._snapshot_values_dtype(MatrixValues())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one-dimensional"):
        selections._snapshot_values_block(
            MatrixValues(),  # type: ignore[arg-type]
            0,
            2,
            np.dtype("U1"),
        )
    with pytest.raises(RuntimeError, match="changed"):
        selections._snapshot_values_block(
            _ObjectStrings(["too-long"]),  # type: ignore[arg-type]
            0,
            1,
            np.dtype("U2"),
        )

    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("table")
    table.create_array("value", data=np.arange(3))
    cases: tuple[Any, ...] = (
        "value",
        (),
        ("",),
        ("value", "value"),
        ("path/name",),
        ("missing",),
    )
    for columns in cases:
        with pytest.raises((TypeError, ValueError, KeyError)):
            selections._snapshot_source_columns(
                table,
                table_path="table",
                columns=columns,
                row_count=3,
            )
    with pytest.raises(ValueError, match="align"):
        selections._snapshot_source_columns(
            table,
            table_path="table",
            columns=("value",),
            row_count=4,
        )
    table["value"].attrs["missing_mask"] = "absent"
    with pytest.raises(ValueError, match="invalid missing mask"):
        selections._snapshot_source_columns(
            table,
            table_path="table",
            columns=("value",),
            row_count=3,
        )
    table.create_array("absent", data=np.arange(3))
    with pytest.raises(ValueError, match="malformed missing mask"):
        selections._snapshot_source_columns(
            table,
            table_path="table",
            columns=("value",),
            row_count=3,
        )


def test_snapshot_scope_and_record_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="cannot set an assay"):
        selections._snapshot_scope("cell", "RNA")
    with pytest.raises(ValueError, match="require an assay"):
        selections._snapshot_scope("feature", None)
    with pytest.raises(ValueError, match="axis"):
        selections._snapshot_scope("rows", None)

    root, ref = _snapshot_root()
    wrong = ArtifactRef("datastore", "cell_selection", ref.artifact_id)
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, wrong)
    assert caught.value.code == "artifact_reference_mismatch"

    monkeypatch.setattr(
        selections,
        "inspect_artifact",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, ref)
    assert caught.value.code == "artifact_missing"

    monkeypatch.undo()
    missing = ArtifactRef("datastore", "metadata_snapshot", "f" * 64)
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, missing)
    assert caught.value.code == "artifact_missing"
    artifact_group(root, ref).attrs["complete"] = False
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, ref)
    assert caught.value.code == "artifact_incomplete"


def test_snapshot_detects_contract_and_payload_corruption() -> None:
    root, ref = _snapshot_root()
    group = artifact_group(root, ref)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters["axis"] = "feature"
    provenance["parameters"] = parameters
    group.attrs["provenance"] = provenance
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, ref)
    assert caught.value.code == "snapshot_contract_mismatch"

    root, ref = _snapshot_root()
    group = artifact_group(root, ref)
    provenance = dict(group.attrs["provenance"])
    inputs = dict(provenance["inputs"])
    inputs["column_fingerprints"] = []
    provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, ref)
    assert caught.value.code == "snapshot_contract_mismatch"

    root, ref = _snapshot_root()
    artifact_group(root, ref)["names"].attrs["unexpected"] = True
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, ref)
    assert caught.value.code == "snapshot_values_changed"

    root, ref = _snapshot_root()
    artifact_group(root, ref).create_array("unexpected", data=np.arange(3))
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, ref)
    assert caught.value.code == "snapshot_values_changed"

    root, ref = _snapshot_root()
    del root["cellData/ids"]
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, ref)
    assert caught.value.code == "selection_row_ids_missing"

    root, ref = _snapshot_root()
    del root["cellData"]
    with pytest.raises(ArtifactResolutionError) as caught:
        _validate_snapshot(root, ref)
    assert caught.value.code == "selection_table_missing"


def test_snapshot_copy_detects_source_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    table.create_array("ids", data=np.asarray(["a", "b"]))
    table.create_array("names", data=np.asarray(["A", "B"]))

    fingerprints = iter(("first", "second"))
    monkeypatch.setattr(
        selections,
        "_fingerprint_snapshot_column",
        lambda *_args, **_kwargs: next(fingerprints),
    )
    with pytest.raises(RuntimeError, match="changed"):
        selections.snapshot_run_metadata(
            root,
            table_path="cellData",
            id_column="ids",
            columns=("names",),
            axis="cell",
            invalidate_cache=True,
        )

    monkeypatch.undo()
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    table.create_array("ids", data=np.asarray(["a", "b"]))
    table.create_array("names", data=np.asarray(["A", "B"]))
    row_fingerprints = iter(("first", "second"))
    monkeypatch.setattr(
        selections,
        "fingerprint_stored_strings",
        lambda *_args: next(row_fingerprints),
    )
    with pytest.raises(RuntimeError, match="row IDs changed"):
        selections.snapshot_run_metadata(
            root,
            table_path="cellData",
            id_column="ids",
            columns=("names",),
            axis="cell",
            invalidate_cache=True,
        )


def test_resolve_metadata_snapshot_rejects_unaligned_values() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ValueError, match="align"):
        selections.resolve_metadata_snapshot(
            root,
            values=np.arange(3),
            row_ids=np.asarray(["a", "b"]),
            operation="snapshot",
            parameters={},
            inputs={},
            source_columns=[],
        )


def _spec(
    shape: tuple[int, ...] = (4, 3),
    chunks: tuple[int, ...] = (2, 3),
    *,
    shards: tuple[int, ...] | None = None,
) -> ZarrArraySpec:
    return ZarrArraySpec(
        shape=shape,
        chunks=chunks,
        dtype=np.float64,
        compressors=(),
        shards=shards,
    )


def test_run_async_uses_worker_thread_inside_running_loop() -> None:
    seen: list[str] = []

    async def success() -> None:
        seen.append("done")

    async def outer() -> None:
        sharding._run_async(success)

    asyncio.run(outer())
    assert seen == ["done"]

    async def failure() -> None:
        raise ValueError("thread failure")

    async def failing_outer() -> None:
        with pytest.raises(ValueError, match="thread failure"):
            sharding._run_async(failure)

    asyncio.run(failing_outer())


def test_sparse_geometry_and_batch_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        sharding._destination_geometry(_spec((3,), (3,)))
    assert sharding.row_band_task_count(0, 2) == 0
    with pytest.raises(ValueError, match="positive"):
        sharding.row_band_task_count(3, 0)
    assert sharding.sparse_write_task_count((), 3) == 0

    resources = ResourceBudget(memoryBytes=10**8, workers=2)
    geometry = ArrayGeometry((4, 3), (2, 3), None, 8)
    base = {
        "resources": resources,
        "maxWindowNnz": lambda width: width,
        "sourceDtype": np.float64,
    }
    with pytest.raises(ValueError, match="At least one"):
        sharding._resolve_sparse_import_geometries((), nRows=4, **base)
    with pytest.raises(ValueError, match="negative"):
        sharding._resolve_sparse_import_geometries(
            ((geometry, np.dtype(np.float64)),), nRows=-1, **base
        )
    with pytest.raises(ValueError, match="row count"):
        sharding._resolve_sparse_import_geometries(
            ((geometry, np.dtype(np.float64)),), nRows=3, **base
        )
    with pytest.raises(ValueError, match="batch_size"):
        sharding._resolve_sparse_import_geometries(
            ((geometry, np.dtype(np.float64)),), nRows=4, batchRows=0, **base
        )
    empty = ArrayGeometry((0, 3), (2, 3), None, 8)
    plan = sharding._resolve_sparse_import_geometries(
        ((empty, np.dtype(np.float64)),), nRows=0, **base
    )
    assert plan.batchRows == 1 and plan.writeTasks == 0

    monkeypatch.setattr(
        sharding,
        "admitted_worker_split",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("small")),
    )
    with pytest.raises(MemoryError, match="cannot fit"):
        sharding._resolve_sparse_import_geometries(
            ((geometry, np.dtype(np.float64)),), nRows=4, **base
        )


def test_sparse_buffer_and_band_geometry_errors() -> None:
    spec = _spec(shards=(2, 3))
    buffer = sharding.SparseShardBuffer(spec)
    with pytest.raises(ValueError, match="columns"):
        list(buffer.add(coo_matrix((1, 2))))
    with pytest.raises(ValueError, match="more rows"):
        list(buffer.add(coo_matrix((5, 3))))
    with pytest.raises(ValueError, match="expected"):
        list(buffer.finish())
    with pytest.raises(ValueError, match="chunk geometry"):
        sharding._band_geometry(np.zeros((2, 2)))


def test_dense_and_sparse_empty_destination_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    empty = root.create_array("empty", shape=(0, 2), chunks=(1, 2), dtype=np.float64)
    resources = ResourceBudget(memoryBytes=10**8, workers=1)
    with pytest.raises(ValueError, match="invalid shape"):
        sharding.write_dense_from_row_batches(
            empty,
            iter((np.empty((0, 3)),)),
            resources=resources,
        )
    with pytest.raises(ValueError, match="empty destination"):
        sharding.write_dense_from_row_batches(
            empty,
            iter((np.ones((1, 2)),)),
            resources=resources,
        )
    with pytest.raises(ValueError, match="invalid shape"):
        sharding.accumulate_sparse_to_shards(
            empty,
            iter((coo_matrix((0, 3)),)),
            resources=resources,
            producerReserveBytes=0,
        )
    with pytest.raises(ValueError, match="empty destination"):
        sharding.accumulate_sparse_to_shards(
            empty,
            iter((coo_matrix((1, 2)),)),
            resources=resources,
            producerReserveBytes=0,
        )

    destination = root.create_array(
        "destination", shape=(3, 2), chunks=(2, 2), dtype=np.float64
    )
    monkeypatch.setattr(
        sharding,
        "_writer_count",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("budget")),
    )
    with pytest.raises(MemoryError, match="budget"):
        sharding.write_dense_from_row_batches(
            destination,
            iter((np.ones((3, 2)),)),
            resources=resources,
        )


def test_dense_shard_writer_contract_errors() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    resources = ResourceBudget(memoryBytes=10**8, workers=1)
    empty = root.create_array("empty", shape=(0, 2), chunks=(1, 2), dtype=np.float64)
    assert (
        sharding.write_dense_in_shard_rows(
            empty,
            lambda start, end: np.empty((end - start, 2)),
            resources=resources,
        )
        is None
    )
    with pytest.raises(ValueError, match="provided together"):
        sharding.write_dense_in_shard_rows(
            empty,
            lambda start, end: np.empty((end - start, 2)),
            resources=resources,
            summarize=np.sum,
        )

    destination = root.create_array(
        "dst", shape=(3, 2), chunks=(2, 2), dtype=np.float64
    )
    mirror = root.create_array("mirror", shape=(4, 2), chunks=(2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="Mirror"):
        sharding.write_dense_in_shard_rows(
            destination,
            lambda start, end: np.ones((end - start, 2)),
            also_write_to=mirror,
            resources=resources,
        )
    with pytest.raises(ValueError, match="returned shape"):
        sharding.write_dense_in_shard_rows(
            destination,
            lambda _start, _end: np.ones((1, 1)),
            resources=resources,
        )


def test_counts_t_shape_and_layout_contracts() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        sharding.counts_t_spec(_spec((3,), (3,)), profile="local")
    assert not sharding.is_paired_counts_t_layout(
        shape=(-1, 2),
        chunks=(1, 1),
        shards=(1, 1),
        dtype=np.float64,
    )
    assert not sharding.is_paired_counts_t_layout(
        shape=(2, 2),
        chunks=(0, 1),
        shards=(1, 1),
        dtype=np.float64,
    )
    assert not sharding.is_paired_counts_t_layout(
        shape=(2, 2),
        chunks=(2, 2),
        shards=(3, 4),
        dtype=np.float64,
    )
