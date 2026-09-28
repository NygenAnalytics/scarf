import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    artifact_plan_scope,
    artifact_transaction,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from scarf.storage.artifacts import artifact_path, inspect_artifact


def test_artifact_writer_streams_to_random_path_then_reuses_provenance() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "normalized",
        "operation": "run_normalization",
        "parameters": {"log_transform": True},
        "inputs": {"selection": {"artifact_id": "a" * 64}},
        "execution_options": {"batch_size": 100},
    }
    planned = plan_artifact(root, **arguments)
    assert not planned.reused
    group = start_artifact(root, planned)
    assert not inspect_artifact(root, planned.ref).complete
    group.create_array("data", data=np.array([1.0, 2.0, 3.0]))
    finish_artifact(group, planned)
    status = inspect_artifact(root, planned.ref)
    assert status.complete
    assert status.created_at_ns is not None
    assert status.scarf_version is not None
    completed_attrs = dict(group.attrs)

    reused = plan_artifact(root, **arguments)
    assert reused.reused
    assert reused.ref == planned.ref
    assert reused_artifact_group(root, reused).path == group.path
    assert dict(group.attrs) == completed_attrs

    invalidated = plan_artifact(
        root,
        **arguments,
        invalidate_cache=True,
    )
    assert not invalidated.reused
    assert invalidated.ref != planned.ref
    assert artifact_path(invalidated.ref) not in root
    refreshed_group = start_artifact(root, invalidated)
    refreshed_group.create_array("data", data=np.array([4.0, 5.0, 6.0]))
    finish_artifact(refreshed_group, invalidated)
    preferred = plan_artifact(root, **arguments)
    assert preferred.reused
    assert preferred.ref == invalidated.ref


def test_artifact_plan_scope_records_nested_created_and_reused_decisions() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "normalized",
        "operation": "run_normalization",
        "parameters": {},
        "inputs": {},
        "execution_options": {},
    }
    with artifact_plan_scope() as outer:
        with artifact_plan_scope() as inner:
            created = plan_artifact(root, **arguments)
        group = start_artifact(root, created)
        finish_artifact(group, created)
        reused = plan_artifact(root, **arguments)

    assert [(item.operation, item.ref, item.disposition) for item in inner] == [
        ("run_normalization", created.ref, "created")
    ]
    assert [(item.operation, item.ref, item.disposition) for item in outer] == [
        ("run_normalization", created.ref, "created"),
        ("run_normalization", reused.ref, "reused"),
    ]


def test_incomplete_artifact_is_not_reused() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "ann_index",
        "operation": "build_ann_index",
        "parameters": {"ann_parallel": True},
        "inputs": {"coordinates": {"artifact_id": "b" * 64}},
        "execution_options": {},
    }
    first = plan_artifact(root, **arguments)
    start_artifact(root, first)

    second = plan_artifact(root, **arguments)
    assert not second.reused
    assert second.ref != first.ref


def test_execution_options_use_artifact_value_serialization() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
        execution_options={
            "nan": float("nan"),
            "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
            "bytes": b"\x00\xff",
            "set": {2, 1},
            "numpy_scalar": np.int64(3),
        },
    )

    start_artifact(root, planned)
    status = inspect_artifact(root, planned.ref)

    assert status.execution_options == {
        "nan": {"special_float": "nan"},
        "positive_infinity": {"special_float": "inf"},
        "negative_infinity": {"special_float": "-inf"},
        "bytes": {"bytes_hex": "00ff"},
        "set": [1, 2],
        "numpy_scalar": 3,
    }


def test_missing_required_attribute_prevents_reuse() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "mapping_reference",
        "operation": "build_mapping_reference",
        "parameters": {"method": "symphony"},
        "inputs": {"reduction": {"artifact_id": "c" * 64}},
        "execution_options": {},
    }
    first = plan_artifact(root, **arguments)
    group = start_artifact(root, first)
    group.create_array("data", data=np.array([1.0]))
    finish_artifact(group, first)

    second = plan_artifact(
        root,
        **arguments,
        required_arrays=("data",),
        required_attributes=("reference_metadata",),
    )

    assert not second.reused
    assert second.ref != first.ref


def test_invalid_required_attribute_type_prevents_reuse() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "mapping_reference",
        "operation": "build_mapping_reference",
        "parameters": {"method": "symphony"},
        "inputs": {"reduction": {"artifact_id": "d" * 64}},
        "execution_options": {},
    }
    first = plan_artifact(root, **arguments)
    group = start_artifact(root, first)
    group.attrs["reference_metadata"] = "invalid"
    finish_artifact(group, first)

    second = plan_artifact(
        root,
        **arguments,
        required_attributes=(
            AttributeRequirement(
                "reference_metadata",
                expected_types=(dict,),
            ),
        ),
    )

    assert not second.reused
    assert second.ref != first.ref


def test_finish_rejects_payload_that_violates_declared_shape() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
        execution_options={},
        required_arrays=(ArrayRequirement("data", shape=(3,), dtype_kind="f"),),
    )
    group = start_artifact(root, planned)
    group.create_array("data", data=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="does not satisfy"):
        finish_artifact(group, planned)

    assert not inspect_artifact(root, planned.ref).complete


def test_exact_dtype_requirement_prevents_reuse() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "reduction",
        "operation": "run_pca",
        "parameters": {},
        "inputs": {},
        "execution_options": {},
    }
    first = plan_artifact(root, **arguments)
    group = start_artifact(root, first)
    group.create_array("data", data=np.array([1.0], dtype=np.float64))
    finish_artifact(group, first)

    same_kind = plan_artifact(
        root,
        **arguments,
        required_arrays=(ArrayRequirement("data", dtype_kind="f"),),
    )
    exact = plan_artifact(
        root,
        **arguments,
        required_arrays=(ArrayRequirement("data", dtype=np.float32),),
    )

    assert same_kind.reused
    assert not exact.reused


def _planned_normalized(root: zarr.Group, **extra: object):
    return plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={"log_transform": True},
        inputs={},
        execution_options={},
        required_arrays=(ArrayRequirement("data", shape=(3,)),),
        **extra,
    )


def test_artifact_transaction_finishes_the_written_slot() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _planned_normalized(root)
    with artifact_transaction(root, planned) as group:
        assert not inspect_artifact(root, planned.ref).complete
        group.create_array("data", data=np.arange(3.0))
    assert inspect_artifact(root, planned.ref).complete
    assert _planned_normalized(root).ref == planned.ref


@pytest.mark.parametrize("error", [RuntimeError("write failed"), KeyboardInterrupt()])
def test_artifact_transaction_removes_a_failed_slot(error: BaseException) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _planned_normalized(root)
    with pytest.raises(type(error)):
        with artifact_transaction(root, planned) as group:
            group.create_array("data", data=np.arange(3.0))
            raise error
    assert artifact_path(planned.ref) not in root
    assert not inspect_artifact(root, planned.ref).exists


def test_artifact_transaction_removes_a_slot_that_fails_its_contract() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = _planned_normalized(root)
    with pytest.raises(ValueError, match="does not satisfy its contract"):
        with artifact_transaction(root, planned) as group:
            group.create_array("data", data=np.arange(4.0))
    assert not inspect_artifact(root, planned.ref).exists


def test_start_artifact_refuses_a_read_only_root_before_writing() -> None:
    store = MemoryStore()
    zarr.open_group(store=store, mode="w")
    read_only = zarr.open_group(store=store.with_read_only(True), mode="r")
    planned = _planned_normalized(read_only)
    with pytest.raises(PermissionError, match=r"run_normalization.*zarr_mode='r\+'"):
        start_artifact(read_only, planned)
    with pytest.raises(PermissionError):
        with artifact_transaction(read_only, planned):
            raise AssertionError("the body must not run")
    assert not inspect_artifact(read_only, planned.ref).exists
