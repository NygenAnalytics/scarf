import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
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
    assert inspect_artifact(root, planned.ref).complete

    reused = plan_artifact(root, **arguments)
    assert reused.reused
    assert reused.ref == planned.ref
    assert reused_artifact_group(root, reused).path == group.path

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


def test_planned_artifact_invalidated_forces_new_ref() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "normalized",
        "operation": "run_normalization",
        "parameters": {"log_transform": True},
        "inputs": {"selection": {"artifact_id": "e" * 64}},
        "execution_options": {"batch_size": 50},
    }
    planned = plan_artifact(root, **arguments)
    group = start_artifact(root, planned)
    group.create_array("data", data=np.array([1.0, 2.0, 3.0]))
    finish_artifact(group, planned)

    reused = plan_artifact(root, **arguments)
    assert reused.reused
    invalidated = reused.invalidated(root)
    assert not invalidated.reused
    assert invalidated.ref != reused.ref
    assert invalidated.provenance == reused.provenance
    assert invalidated.execution_options == reused.execution_options
