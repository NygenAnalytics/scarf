from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.assay import norm_lib_size, norm_tf_idf
from scarf.datastore.datastore import DataStore
from scarf.datastore.base_datastore import BaseDataStore
from scarf.storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    reused_artifact_group,
    start_artifact,
)
from scarf.storage.artifacts import (
    ARTIFACT_KINDS,
    ArtifactRef,
    ArtifactScope,
    ExternalArtifactRef,
    ValueFingerprintBuilder,
    artifact_exists,
    artifact_group,
    artifact_path,
    canonical_bytes,
    callable_identity,
    find_reusable_artifacts,
    fingerprint_array,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    fingerprint_strings,
    group_at,
    inspect_artifact,
    list_artifacts,
    make_provenance,
    new_artifact_id,
    parse_artifact_path,
    provenance_hash,
    require_complete_artifact,
    serialize_artifact_value,
)


def _ref(
    kind: str = "normalized",
    artifact_id: str = "a" * 64,
) -> ArtifactRef:
    return ArtifactRef(
        scope="assay",
        assay="RNA",
        kind=kind,
        artifact_id=artifact_id,
    )


def test_callable_identity_requires_dynamic_explicit_identity() -> None:
    with pytest.raises(ValueError, match="artifact_identity"):
        callable_identity(lambda values: values)

    def configured(values):
        return values

    configured.artifact_identity = "configured"  # type: ignore[attr-defined]
    assert callable_identity(configured) == {"identity": "configured"}


def test_tfidf_uses_semantic_artifact_identity_only_for_atac() -> None:
    assert callable_identity(norm_tf_idf) == {
        "identity": "scarf.assay.norm_tf_idf:selected-cell-df:total-count-tf"
    }
    assert callable_identity(norm_lib_size) == {
        "module": "scarf.assay",
        "qualname": "norm_lib_size",
    }


def test_datastore_load_artifact_resolves_workspace_path() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    workspace = root.create_group("analysis")
    planned = plan_artifact(
        workspace,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
        execution_options={},
    )
    group = start_artifact(workspace, planned)
    group.create_array("data", data=np.ones((2, 2)))
    finish_artifact(group, planned)
    datastore = BaseDataStore.__new__(BaseDataStore)
    datastore.z = root
    datastore.workspace = "analysis"

    loaded = datastore.load_artifact(planned.ref)

    assert loaded.path == f"analysis/{artifact_path(planned.ref)}"
    np.testing.assert_array_equal(loaded["data"][:], np.ones((2, 2)))


def _complete_artifact(
    root: zarr.Group,
    *,
    kind: str,
    operation: str,
    parameters: dict,
    inputs: dict,
    scope: ArtifactScope = "assay",
) -> ArtifactRef:
    ref = ArtifactRef(
        scope=scope,
        assay="RNA" if scope == "assay" else None,
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    group = root.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": ref.kind,
            "provenance": make_provenance(
                operation=operation,
                parameters=parameters,
                inputs=inputs,
            ),
            "execution_options": {},
            "complete": True,
        }
    )
    return ref


def test_artifact_group_helpers_and_require_complete() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
    )
    path = artifact_path(ref)
    assert group_at(root, path).path == root[path].path
    assert artifact_group(root, ref).path == root[path].path
    status = require_complete_artifact(root, ref)
    assert status.complete
    assert status.path == path

    missing = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="normalized",
        artifact_id="f" * 64,
    )
    with pytest.raises(KeyError, match="does not exist"):
        require_complete_artifact(root, missing)

    incomplete = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="normalized",
        artifact_id="c" * 64,
    )
    incomplete_path = artifact_path(incomplete)
    group = root.create_group(incomplete_path)
    group.attrs.update(
        {
            "artifact_id": incomplete.artifact_id,
            "kind": incomplete.kind,
            "provenance": make_provenance(
                operation="run_normalization",
                parameters={},
                inputs={},
            ),
            "execution_options": {},
            "complete": False,
        }
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        require_complete_artifact(root, incomplete)


def test_artifact_kinds_cover_the_reviewed_taxonomy() -> None:
    assert {
        "cell_selection",
        "feature_selection",
        "metadata_snapshot",
        "normalized",
        "feature_scaling",
        "reduction",
        "batch_correction",
        "ann_index",
        "neighbors",
        "connectivity_map",
        "embedding_initialization",
        "embedding",
        "cluster_labels",
        "marker_table",
        "enrichment_scores",
        "diffusion_operator",
        "mapping_reference",
        "projection",
        "integrated_graph",
        "imported_coordinates",
        "wnn_coordinates",
    } <= ARTIFACT_KINDS


def test_artifact_ids_are_random_storage_addresses() -> None:
    first = new_artifact_id()
    second = new_artifact_id()
    assert first != second
    assert len(first) == len(second) == 64

    assay_ref = _ref(artifact_id=first)
    datastore_ref = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id=second,
    )
    assert artifact_path(assay_ref) == f"RNA/artifacts/normalized/{first}"
    assert artifact_path(datastore_ref) == f"artifacts/integrated_graph/{second}"
    assert ArtifactRef.from_dict(assay_ref.to_dict()) == assay_ref
    assert parse_artifact_path(artifact_path(assay_ref)) == assay_ref
    assert parse_artifact_path(artifact_path(datastore_ref)) == datastore_ref


def test_artifact_ref_repr_truncates_the_artifact_id() -> None:
    assay_ref = _ref(artifact_id="ab" * 32)
    datastore_ref = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id="cd" * 32,
    )

    assert repr(assay_ref) == (
        "ArtifactRef(assay='RNA', kind='normalized', artifact_id='abababababab...')"
    )
    assert repr(datastore_ref) == (
        "ArtifactRef(datastore, kind='integrated_graph', artifact_id='cdcdcdcdcdcd...')"
    )
    assert assay_ref.to_dict()["artifact_id"] == "ab" * 32


def test_artifact_ref_rejects_malformed_values() -> None:
    with pytest.raises(ValueError, match="require an assay"):
        ArtifactRef(scope="assay", kind="normalized", artifact_id="a" * 64)
    with pytest.raises(ValueError, match="cannot set assay"):
        ArtifactRef(
            scope="datastore",
            assay="RNA",
            kind="integrated_graph",
            artifact_id="a" * 64,
        )
    with pytest.raises(ValueError, match="Unknown artifact kind"):
        ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="normalized__log_true",
            artifact_id="a" * 64,
        )
    with pytest.raises(ValueError, match="Not an artifact path"):
        parse_artifact_path("RNA/normed__I__hvgs")
    missing_type = _ref().to_dict()
    del missing_type["type"]
    with pytest.raises(ValueError, match="type must be"):
        ArtifactRef.from_dict(missing_type)
    with pytest.raises(ValueError, match="fields do not match"):
        ArtifactRef.from_dict({**_ref().to_dict(), "path": "RNA/legacy"})
    datastore_ref = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id="b" * 64,
    )
    with pytest.raises(ValueError, match="fields do not match"):
        ArtifactRef.from_dict({**datastore_ref.to_dict(), "assay": "RNA"})


def test_artifact_path_parser_normalizes_boundaries_and_rejects_corruption() -> None:
    ref = _ref(artifact_id="1" * 64)
    assert parse_artifact_path(f"/{artifact_path(ref)}/") == ref

    invalid_paths = [
        (f"artifacts/normalized/{'A' * 64}", "lowercase hex token"),
        (f"RNA/artifacts/not_a_kind/{'a' * 64}", "Unknown artifact kind"),
        (f"RNA/artifacts/normalized/{'a' * 64}/payload", "Not an artifact path"),
        (f"RNA/results/normalized/{'a' * 64}", "Not an artifact path"),
    ]
    for path, message in invalid_paths:
        with pytest.raises(ValueError, match=message):
            parse_artifact_path(path)


def test_external_artifact_ref_round_trips_strictly() -> None:
    ref = _ref(kind="embedding", artifact_id="b" * 64)
    external = ExternalArtifactRef(
        dataset_fingerprint="reference-dataset",
        ref=ref,
    )
    serialized = {
        "type": "external_artifact",
        "dataset_fingerprint": "reference-dataset",
        "ref": ref.to_dict(),
    }

    assert external.to_dict() == serialized
    assert ExternalArtifactRef.from_dict(serialized) == external
    assert serialize_artifact_value(external) == serialized
    assert canonical_bytes(external) == canonical_bytes(serialized)
    assert not {
        "path",
        "uri",
        "workspace",
        "storage_options",
        "credentials",
        "schema",
        "version",
    } & set(serialized)


def test_external_artifact_ref_rejects_malformed_values() -> None:
    assay_ref = _ref()
    datastore_ref = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id="c" * 64,
    )
    with pytest.raises(ValueError, match="non-empty"):
        ExternalArtifactRef(dataset_fingerprint="", ref=assay_ref)
    with pytest.raises(ValueError, match="assay-scoped"):
        ExternalArtifactRef(
            dataset_fingerprint="reference-dataset",
            ref=datastore_ref,
        )

    serialized = ExternalArtifactRef(
        dataset_fingerprint="reference-dataset",
        ref=assay_ref,
    ).to_dict()
    with pytest.raises(ValueError, match="contain exactly"):
        ExternalArtifactRef.from_dict({**serialized, "path": "/tmp/reference"})
    with pytest.raises(ValueError, match="contain exactly"):
        ExternalArtifactRef.from_dict(
            {
                "type": "external_artifact",
                "dataset_fingerprint": "reference-dataset",
            }
        )
    with pytest.raises(ValueError, match="type must be"):
        ExternalArtifactRef.from_dict({**serialized, "type": "artifact"})
    with pytest.raises(ValueError, match="complete assay artifact"):
        ExternalArtifactRef.from_dict(
            {
                **serialized,
                "ref": {**assay_ref.to_dict(), "path": "reference.zarr"},
            }
        )
    with pytest.raises(ValueError, match="declared scope"):
        ExternalArtifactRef.from_dict(
            {
                **serialized,
                "ref": {**datastore_ref.to_dict(), "assay": None},
            }
        )


def test_external_artifact_ref_has_stable_provenance_identity() -> None:
    ref = _ref(kind="mapping_reference", artifact_id="d" * 64)
    external = ExternalArtifactRef("reference-dataset", ref)
    round_tripped = ExternalArtifactRef.from_dict(external.to_dict())
    same = make_provenance(
        operation="map_query",
        parameters={},
        inputs={"mapping_reference": round_tripped},
    )
    original = make_provenance(
        operation="map_query",
        parameters={},
        inputs={"mapping_reference": external},
    )
    other_dataset = make_provenance(
        operation="map_query",
        parameters={},
        inputs={
            "mapping_reference": ExternalArtifactRef(
                "other-dataset",
                ref,
            )
        },
    )
    other_artifact = make_provenance(
        operation="map_query",
        parameters={},
        inputs={
            "mapping_reference": ExternalArtifactRef(
                "reference-dataset",
                _ref(kind="mapping_reference", artifact_id="e" * 64),
            )
        },
    )

    assert original["inputs"]["mapping_reference"] == external.to_dict()
    assert provenance_hash(original) == provenance_hash(same)
    assert provenance_hash(original) != provenance_hash(other_dataset)
    assert provenance_hash(original) != provenance_hash(other_artifact)


def test_provenance_hash_is_typed_order_independent_and_not_persisted() -> None:
    normalized = _ref()
    first = make_provenance(
        operation="run_reduction",
        parameters={
            "dims": np.int64(12),
            "feat_scaling": True,
            "method": "pca",
            "zero": -0.0,
        },
        inputs={"normalized": normalized},
    )
    second = make_provenance(
        operation="run_reduction",
        parameters={
            "zero": -0.0,
            "method": "pca",
            "feat_scaling": True,
            "dims": 12,
        },
        inputs={"normalized": normalized},
    )
    integer = make_provenance(
        operation="run_reduction",
        parameters={"value": 1},
        inputs={},
    )
    floating = make_provenance(
        operation="run_reduction",
        parameters={"value": 1.0},
        inputs={},
    )

    assert provenance_hash(first) == provenance_hash(second)
    assert provenance_hash(integer) != provenance_hash(floating)
    assert canonical_bytes(Path("/tmp/cache")) == canonical_bytes("/tmp/cache")
    assert "provenance_hash" not in first


def test_canonical_bytes_cover_supported_scalar_and_container_types() -> None:
    assert canonical_bytes({"b": 2, "a": 1}) == canonical_bytes({"a": 1, "b": 2})
    assert canonical_bytes([1, "two"]) == canonical_bytes((1, "two"))
    assert canonical_bytes({1, "two"}) == canonical_bytes(frozenset({"two", 1}))
    assert canonical_bytes(np.bool_(True)) == canonical_bytes(True)
    assert canonical_bytes(np.int64(4)) == canonical_bytes(4)
    assert canonical_bytes(np.float32(1.5)) == canonical_bytes(1.5)

    typed_values = [
        canonical_bytes(None),
        canonical_bytes(False),
        canonical_bytes(0),
        canonical_bytes(0.0),
        canonical_bytes("0"),
        canonical_bytes(b"0"),
    ]
    assert len(set(typed_values)) == len(typed_values)


def test_canonical_bytes_normalize_numpy_scalar_widths() -> None:
    integer_values = (
        np.int8(7),
        np.int16(7),
        np.int32(7),
        np.int64(7),
        np.uint8(7),
        np.uint16(7),
        np.uint32(7),
        np.uint64(7),
    )
    floating_values = (
        np.float16(1.5),
        np.float32(1.5),
        np.float64(1.5),
    )

    assert {canonical_bytes(value) for value in integer_values} == {canonical_bytes(7)}
    assert {canonical_bytes(value) for value in floating_values} == {
        canonical_bytes(1.5)
    }
    assert canonical_bytes(np.float32(-0.0)) == canonical_bytes(-0.0)
    assert canonical_bytes(np.str_("μ")) == canonical_bytes("μ")
    assert canonical_bytes(np.bytes_(b"\x00\xff")) == canonical_bytes(b"\x00\xff")


def test_canonical_bytes_nested_mapping_and_set_order_is_stable() -> None:
    first = {
        "outer": {
            "steps": [{"z": 3, "a": 1}, {"enabled": True}],
            "values": {None, 1, "one", b"one"},
        },
        "name": "analysis",
    }
    reordered = {
        "name": "analysis",
        "outer": {
            "values": frozenset({b"one", "one", 1, None}),
            "steps": [{"a": 1, "z": 3}, {"enabled": True}],
        },
    }
    changed = {
        **reordered,
        "outer": {
            **reordered["outer"],
            "steps": [{"a": 2, "z": 3}, {"enabled": True}],
        },
    }

    assert canonical_bytes(first) == canonical_bytes(reordered)
    assert canonical_bytes(first) != canonical_bytes(changed)


def test_canonical_bytes_reject_unsupported_values_and_mapping_keys() -> None:
    with pytest.raises(TypeError, match="string keys"):
        canonical_bytes({1: "one"})
    with pytest.raises(TypeError, match="Unsupported provenance value"):
        canonical_bytes(object())


@pytest.mark.parametrize(
    "value",
    [
        np.complex64(1 + 2j),
        np.datetime64("2024-01-01"),
        np.timedelta64(2, "D"),
    ],
)
def test_canonical_bytes_reject_unsupported_numpy_scalar_types(value: object) -> None:
    with pytest.raises(TypeError, match="Unsupported provenance value"):
        canonical_bytes(value)


def test_serialize_artifact_value_normalizes_external_value_types() -> None:
    assert serialize_artifact_value(
        {
            1: np.int64(7),
            "path": Path("/tmp/cache"),
            "payload": b"\x00\xff",
            "sequence": ("a", np.float32(2.5)),
            "set": {"b", "a"},
            "nan": np.nan,
            "positive": np.inf,
            "negative": -np.inf,
        }
    ) == {
        "1": 7,
        "path": "/tmp/cache",
        "payload": {"bytes_hex": "00ff"},
        "sequence": ["a", 2.5],
        "set": ["a", "b"],
        "nan": {"special_float": "nan"},
        "positive": {"special_float": "inf"},
        "negative": {"special_float": "-inf"},
    }


def test_provenance_replaces_array_values_with_fingerprints() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    provenance = make_provenance(
        operation="run_reduction",
        parameters={},
        inputs={"custom_loadings": values},
    )

    assert provenance["inputs"]["custom_loadings"] == {
        "value_fingerprint": fingerprint_array(values)
    }
    with pytest.raises(ValueError, match="non-finite"):
        canonical_bytes({"value": np.inf})
    with pytest.raises(TypeError, match="value_fingerprint"):
        canonical_bytes({"values": values})


def test_streamed_value_fingerprint_is_block_size_independent() -> None:
    values = np.arange(24, dtype=np.float64).reshape(6, 4)
    whole = ValueFingerprintBuilder()
    whole.update_array("values", values)
    streamed = ValueFingerprintBuilder()
    streamed.begin_array("values", values.shape, values.dtype)
    streamed.update_array_block("values", (0, 0), values[:2])
    streamed.update_array_block("values", (2, 0), values[2:5])
    streamed.update_array_block("values", (5, 0), values[5:])
    streamed.end_array("values")

    assert streamed.hexdigest() == whole.hexdigest()


@pytest.mark.parametrize(
    ("narrow_dtype", "wide_dtype", "values"),
    [
        ("i2", "i8", [1, -2, 513]),
        ("f4", "f8", [0.0, -0.0, 1.5]),
        ("c8", "c16", [1 + 2j, complex(0.0, -0.0)]),
    ],
)
def test_array_fingerprint_preserves_width_and_normalizes_endian(
    narrow_dtype: str,
    wide_dtype: str,
    values: list[object],
) -> None:
    little_endian = np.asarray(values, dtype=np.dtype(narrow_dtype).newbyteorder("<"))
    big_endian = np.asarray(values, dtype=np.dtype(narrow_dtype).newbyteorder(">"))
    wider = np.asarray(values, dtype=wide_dtype)

    assert fingerprint_array(little_endian) == fingerprint_array(big_endian)
    assert fingerprint_array(little_endian) != fingerprint_array(wider)


def test_array_fingerprint_normalizes_structured_subarray_endian() -> None:
    little_dtype = np.dtype([("coordinates", "<f4", (2,)), ("cluster", "<u2")])
    big_dtype = np.dtype([("coordinates", ">f4", (2,)), ("cluster", ">u2")])
    little = np.zeros(3, dtype=little_dtype)
    little["coordinates"] = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    little["cluster"] = [1, 2, 3]
    big = np.zeros(3, dtype=big_dtype)
    big["coordinates"] = little["coordinates"]
    big["cluster"] = little["cluster"]

    assert fingerprint_array(little) == fingerprint_array(big)


def test_value_fingerprint_builder_validates_array_declarations() -> None:
    with pytest.raises(TypeError, match="Object arrays"):
        ValueFingerprintBuilder().begin_array("values", (1,), object)
    with pytest.raises(ValueError, match="non-negative dimensions"):
        ValueFingerprintBuilder().begin_array("values", (), np.float64)
    with pytest.raises(ValueError, match="non-negative dimensions"):
        ValueFingerprintBuilder().begin_array("values", (2, -1), np.float64)


def test_value_fingerprint_builder_enforces_lifecycle_order() -> None:
    with pytest.raises(RuntimeError, match="begin_array"):
        ValueFingerprintBuilder().update_array_block(
            "values",
            (0,),
            np.array([1.0]),
        )
    with pytest.raises(RuntimeError, match="No active array"):
        ValueFingerprintBuilder().end_array("values")

    builder = ValueFingerprintBuilder()
    builder.begin_array("values", (2,), np.float64)
    with pytest.raises(RuntimeError, match="active array"):
        builder.update_bytes("metadata", b"value")
    with pytest.raises(RuntimeError, match="active array"):
        builder.begin_array("other", (1,), np.float64)
    with pytest.raises(RuntimeError, match="array is incomplete"):
        builder.hexdigest()
    with pytest.raises(ValueError, match="Expected block for 'values'"):
        builder.update_array_block("other", (0,), np.array([1.0]))
    with pytest.raises(ValueError, match="Expected to finish 'values'"):
        builder.end_array("other")
    with pytest.raises(ValueError, match="is incomplete"):
        builder.end_array("values")

    builder.update_array_block("values", (0,), np.array([1.0, 2.0]))
    builder.end_array("values")
    assert len(builder.hexdigest()) == 64


def test_value_fingerprint_builder_validates_block_layout_and_dtype() -> None:
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", (2, 2), np.float64)

    with pytest.raises(TypeError, match="Expected dtype"):
        builder.update_array_block(
            "values",
            (0, 0),
            np.ones((1, 2), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="Expected block offset"):
        builder.update_array_block("values", (1, 0), np.ones((1, 2)))
    with pytest.raises(ValueError, match="incompatible with array shape"):
        builder.update_array_block("values", (0, 0), np.ones((1, 3)))
    with pytest.raises(ValueError, match="exceeds declared shape"):
        builder.update_array_block("values", (0, 0), np.ones((3, 2)))

    builder.update_array_block("values", (0, 0), np.ones((2, 2)))
    builder.end_array("values")


def test_stored_string_fingerprint_supports_variable_length_strings() -> None:
    values = np.array(["a", "longer", "中"], dtype=np.dtypes.StringDType())
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array("ids", data=values, chunks=(1,))

    expected = fingerprint_strings(values.astype("U6"))
    assert fingerprint_stored_strings(array) == expected


def test_stored_array_fingerprint_matches_ordered_in_memory_values() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = {
        "counts": np.arange(12, dtype=np.int16).reshape(6, 2),
        "flags": np.array([True, False, True, True], dtype=np.bool_),
        "empty": np.empty((0, 2), dtype=np.float32),
    }
    root.create_array("counts", data=values["counts"], chunks=(2, 2))
    root.create_array("flags", data=values["flags"], chunks=(3,))
    root.create_array("empty", data=values["empty"], chunks=(1, 2))

    expected = ValueFingerprintBuilder()
    for name, array in values.items():
        expected.update_array(name, array)

    assert fingerprint_stored_arrays(root, tuple(values)) == expected.hexdigest()


def test_stored_string_fingerprint_normalizes_fixed_string_dtypes() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=2)
    unicode_values = np.array(["a", "xy"], dtype="U4")
    byte_values = np.array([b"a", b"xy"], dtype="S4")
    unicode_array = root.create_array(
        "unicode_ids",
        data=unicode_values,
        chunks=(1,),
    )
    byte_array = root.create_array("byte_ids", data=byte_values, chunks=(1,))

    expected = fingerprint_strings(unicode_values)
    assert fingerprint_stored_strings(unicode_array) == expected
    assert fingerprint_stored_strings(byte_array) == expected


def test_stored_string_fingerprint_handles_empty_values_and_rejects_matrices() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    empty = np.array([], dtype=np.dtypes.StringDType())
    empty_array = root.create_array("empty", data=empty, chunks=(1,))
    matrix = root.create_array(
        "matrix",
        data=np.array([["a"], ["b"]], dtype="U1"),
    )

    assert fingerprint_stored_strings(empty_array) == fingerprint_strings(
        empty.astype("U1")
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        fingerprint_stored_strings(matrix)


def test_value_fingerprint_normalizes_endian_alignment_and_padding() -> None:
    little_dtype = np.dtype([("count", "<i2"), ("score", "<f8")], align=True)
    big_dtype = np.dtype([("count", ">i2"), ("score", ">f8")], align=True)
    left = np.empty(6, dtype=little_dtype)
    right = np.empty(6, dtype=little_dtype)
    left.view(np.uint8)[:] = 0
    right.view(np.uint8)[:] = 255
    for values in (left, right):
        values["count"] = np.arange(6)
        values["score"] = np.linspace(0.0, 1.0, 6)
    big_endian = np.empty(6, dtype=big_dtype)
    big_endian["count"] = left["count"]
    big_endian["score"] = left["score"]

    assert fingerprint_array(left) == fingerprint_array(right)
    assert fingerprint_array(left) == fingerprint_array(big_endian)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_artifact_inspection_listing_and_reuse_are_read_only(
    zarr_format: int,
) -> None:
    root = zarr.open_group(
        store=MemoryStore(),
        mode="w",
        zarr_format=zarr_format,
    )
    provenance = make_provenance(
        operation="run_normalization",
        parameters={"log_transform": True},
        inputs={"cell_selection": _ref("cell_selection", "c" * 64)},
    )
    complete_ref = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters=provenance["parameters"],
        inputs=provenance["inputs"],
    )
    incomplete_ref = _ref(artifact_id="e" * 64)
    incomplete = root.create_group(artifact_path(incomplete_ref))
    incomplete.attrs.update(
        {
            "artifact_id": incomplete_ref.artifact_id,
            "kind": incomplete_ref.kind,
            "provenance": provenance,
            "execution_options": {},
            "complete": False,
        }
    )
    datastore_ref = _complete_artifact(
        root,
        scope="datastore",
        kind="integrated_graph",
        operation="integrate_assays",
        parameters={"method": "wnn"},
        inputs={},
    )
    before = dict(root[artifact_path(complete_ref)].attrs)

    status = inspect_artifact(root, complete_ref)
    assert status.exists
    assert status.complete
    assert status.parameters == {"log_transform": True}
    assert status.inputs == provenance["inputs"]
    assert artifact_exists(root, complete_ref)
    assert not artifact_exists(root, incomplete_ref)
    assert list_artifacts(root, scope="assay", assay="RNA") == sorted(
        [complete_ref, incomplete_ref],
        key=lambda ref: ref.artifact_id,
    )
    assert list_artifacts(root, scope="datastore") == [datastore_ref]
    assert find_reusable_artifacts(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        provenance=provenance,
    ) == [complete_ref]
    assert (
        find_reusable_artifacts(
            root,
            scope="assay",
            assay="RNA",
            kind="normalized",
            provenance=provenance,
            invalidate_cache=True,
        )
        == []
    )
    assert dict(root[artifact_path(complete_ref)].attrs) == before


def test_artifact_inspection_rejects_mismatched_stored_identity() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
    )
    group = root[artifact_path(ref)]

    group.attrs["artifact_id"] = "0" * 64
    with pytest.raises(ValueError, match="mismatched artifact_id"):
        inspect_artifact(root, ref)

    group.attrs["artifact_id"] = ref.artifact_id
    group.attrs["kind"] = "embedding"
    with pytest.raises(ValueError, match="mismatched kind"):
        inspect_artifact(root, ref)


def test_artifact_status_handles_missing_and_corrupt_storage_nodes() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    missing = _ref(artifact_id="3" * 64)
    missing_status = inspect_artifact(root, missing)
    assert not missing_status.exists
    assert missing_status.operation is None
    assert missing_status.parameters is None
    assert missing_status.inputs is None

    array_ref = _ref(artifact_id="4" * 64)
    root.create_array(artifact_path(array_ref), data=np.array([1.0]))
    with pytest.raises(TypeError, match="Expected Zarr group"):
        inspect_artifact(root, array_ref)

    corrupt_ref = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
    )
    root[artifact_path(corrupt_ref)].attrs["provenance"] = {
        "operation": "run_normalization",
        "parameters": {"threshold": float("inf")},
        "inputs": {},
    }
    with pytest.raises(ValueError, match="non-finite"):
        inspect_artifact(root, corrupt_ref)


def test_artifact_inspection_rejects_malformed_mapping_attrs() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
    )
    group = root[artifact_path(ref)]

    group.attrs["execution_options"] = []
    with pytest.raises(TypeError, match="must be a mapping"):
        inspect_artifact(root, ref)

    group.attrs["execution_options"] = {}
    group.attrs["provenance"] = {
        "operation": "run_normalization",
        "parameters": [],
        "inputs": {},
    }
    with pytest.raises(TypeError, match="provenance.*malformed"):
        inspect_artifact(root, ref)

    group.attrs["provenance"] = {
        "operation": "Run normalization",
        "parameters": {},
        "inputs": {},
    }
    with pytest.raises(ValueError, match="snake_case identifier"):
        inspect_artifact(root, ref)


def test_artifact_listing_validates_filters_and_malformed_paths() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    complete = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
    )
    incomplete = _ref(artifact_id="1" * 64)
    root.create_group(artifact_path(incomplete)).attrs["complete"] = False
    root.create_group("RNA/artifacts/normalized/not-an-artifact-id")

    listed = list_artifacts(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
    )
    assert listed == sorted([complete, incomplete], key=lambda ref: ref.artifact_id)
    assert list_artifacts(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        complete_only=True,
    ) == [complete]
    assert artifact_exists(root, incomplete, require_complete=False)
    assert (
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            kind="embedding",
        )
        == []
    )

    with pytest.raises(ValueError, match="Invalid artifact scope"):
        list_artifacts(root, scope="workspace")  # type: ignore[arg-type]
    for invalid_assay in (None, "", "RNA/subset"):
        with pytest.raises(ValueError, match="assay is required"):
            list_artifacts(root, scope="assay", assay=invalid_assay)
    with pytest.raises(ValueError, match="assay cannot be set"):
        list_artifacts(root, scope="datastore", assay="RNA")
    with pytest.raises(ValueError, match="Unknown artifact kind"):
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            kind="not_a_kind",
        )

    root.create_group("RNA/artifacts/not_a_kind")
    assert (
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            kind="normalized",
        )
        == listed
    )
    with pytest.raises(ValueError, match="Unknown artifact kind"):
        list_artifacts(root, scope="assay", assay="RNA")


def test_artifact_listing_matches_exact_serialized_provenance_predicates() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    source = _ref(kind="neighbors", artifact_id="2" * 64)
    expected = [
        _complete_artifact(
            root,
            kind="normalized",
            operation="run_normalization",
            parameters={"scale": 10_000, "options": {"log": True}},
            inputs={"source": source},
        )
        for _ in range(2)
    ]
    log_disabled = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={"scale": 10_000, "options": {"log": False}},
        inputs={"source": source},
    )
    _complete_artifact(
        root,
        kind="normalized",
        operation="import_normalization",
        parameters={"scale": 10_000, "options": {"log": True}},
        inputs={"source": source},
    )
    incomplete = _ref(artifact_id="3" * 64)
    root.create_group(artifact_path(incomplete)).attrs.update(
        {
            "artifact_id": incomplete.artifact_id,
            "kind": incomplete.kind,
            "provenance": make_provenance(
                operation="run_normalization",
                parameters={"scale": 10_000, "options": {"log": True}},
                inputs={"source": source},
            ),
            "execution_options": {},
            "complete": False,
        }
    )

    query = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "normalized",
        "operation": "run_normalization",
        "parameters": {"options": {"log": True}, "scale": 10_000},
        "inputs": {"source": source},
    }
    assert list_artifacts(root, **query) == sorted(
        expected,
        key=lambda ref: ref.artifact_id,
    )
    assert list_artifacts(
        root,
        **{**query, "inputs": {"source": source.to_dict()}},
    ) == sorted(expected, key=lambda ref: ref.artifact_id)
    assert list_artifacts(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={"scale": 10_000},
    ) == sorted(
        expected + [log_disabled],
        key=lambda ref: ref.artifact_id,
    )
    assert (
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            kind="normalized",
            parameters={},
        )
        == []
    )
    assert list_artifacts(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameters={"options": {"log": True}},
        inputs={"source": source.to_dict()},
    ) == sorted(expected, key=lambda ref: ref.artifact_id)
    with pytest.raises(TypeError, match="parameters must be a mapping"):
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            parameters=[],  # type: ignore[arg-type]
        )


def test_artifact_listing_fails_on_malformed_complete_provenance() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    malformed = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
    )
    root[artifact_path(malformed)].attrs["provenance"] = {
        "operation": "run_normalization",
        "parameters": [],
        "inputs": {},
    }

    with pytest.raises(TypeError, match="provenance.*malformed"):
        list_artifacts(
            root,
            scope="assay",
            assay="RNA",
            operation="run_normalization",
        )


def test_reusable_artifacts_skip_incomplete_malformed_and_mismatched_records() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    provenance = make_provenance(
        operation="run_normalization",
        parameters={"log_transform": True},
        inputs={},
    )
    older = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters=provenance["parameters"],
        inputs=provenance["inputs"],
    )
    newer = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters=provenance["parameters"],
        inputs=provenance["inputs"],
    )
    root[artifact_path(older)].attrs["created_at_ns"] = 10
    root[artifact_path(newer)].attrs["created_at_ns"] = 20

    _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={"log_transform": False},
        inputs={},
    )
    malformed = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters=provenance["parameters"],
        inputs=provenance["inputs"],
    )
    root[artifact_path(malformed)].attrs["provenance"] = {
        "operation": "run_normalization",
        "parameters": [],
        "inputs": {},
    }
    mismatched = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters=provenance["parameters"],
        inputs=provenance["inputs"],
    )
    root[artifact_path(mismatched)].attrs["artifact_id"] = "0" * 64
    incomplete = _ref(artifact_id="2" * 64)
    root.create_group(artifact_path(incomplete)).attrs.update(
        {
            "provenance": provenance,
            "complete": False,
        }
    )

    assert find_reusable_artifacts(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        provenance=provenance,
    ) == [newer, older]


def test_reusable_artifacts_require_matching_operation_and_inputs() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    selection = _ref(kind="cell_selection", artifact_id="5" * 64)
    ref = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={"log_transform": True},
        inputs={"cell_selection": selection},
    )
    matching = make_provenance(
        operation="run_normalization",
        parameters={"log_transform": True},
        inputs={"cell_selection": selection},
    )
    wrong_operation = make_provenance(
        operation="normalize_query",
        parameters={"log_transform": True},
        inputs={"cell_selection": selection},
    )
    wrong_input = make_provenance(
        operation="run_normalization",
        parameters={"log_transform": True},
        inputs={
            "cell_selection": _ref(
                kind="cell_selection",
                artifact_id="6" * 64,
            )
        },
    )

    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "normalized",
    }
    assert find_reusable_artifacts(
        root,
        **arguments,
        provenance=matching,
    ) == [ref]
    assert (
        find_reusable_artifacts(
            root,
            **arguments,
            provenance=wrong_operation,
        )
        == []
    )
    assert (
        find_reusable_artifacts(
            root,
            **arguments,
            provenance=wrong_input,
        )
        == []
    )


def test_nondeterministic_artifacts_use_normal_provenance_reuse() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    provenance = make_provenance(
        operation="build_ann_index",
        parameters={"ann_parallel": True, "ann_metric": "l2"},
        inputs={"coordinates": _ref("reduction", "b" * 64)},
    )
    ref = _complete_artifact(
        root,
        kind="ann_index",
        operation="build_ann_index",
        parameters=provenance["parameters"],
        inputs=provenance["inputs"],
    )

    assert find_reusable_artifacts(
        root,
        scope="assay",
        assay="RNA",
        kind="ann_index",
        provenance=provenance,
    ) == [ref]


def test_artifact_writer_rejects_invalid_lifecycle_transitions() -> None:
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
    planned = plan_artifact(root, **arguments)
    group = start_artifact(root, planned)

    with pytest.raises(FileExistsError, match="already exists"):
        start_artifact(root, planned)

    group.attrs["kind"] = "embedding"
    group.attrs["complete"] = True
    with pytest.raises(ValueError, match="does not match"):
        finish_artifact(group, planned)
    assert group.attrs["complete"] is False

    group.attrs["kind"] = planned.ref.kind
    finish_artifact(group, planned)
    reused = plan_artifact(root, **arguments)
    assert reused.reused
    with pytest.raises(ValueError, match="start a reused"):
        start_artifact(root, reused)
    with pytest.raises(ValueError, match="finish a reused"):
        finish_artifact(group, reused)
    with pytest.raises(ValueError, match="is not reused"):
        reused_artifact_group(root, planned)


def test_artifact_writer_requires_mapping_execution_options() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(TypeError, match="serialize to a mapping"):
        plan_artifact(
            root,
            scope="assay",
            assay="RNA",
            kind="normalized",
            operation="run_normalization",
            parameters={},
            inputs={},
            execution_options=[],  # type: ignore[arg-type]
        )


def test_artifact_writer_validates_attribute_and_payload_contracts() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")

    def validator(_ref: ArtifactRef, group: zarr.Group) -> bool:
        return group.attrs.get("approved") is True

    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "mapping_reference",
        "operation": "build_mapping_reference",
        "parameters": {},
        "inputs": {},
        "execution_options": {},
        "required_attributes": (
            AttributeRequirement(
                "quality",
                expected_types=(int,),
                predicate=lambda value: value > 0,
            ),
        ),
    }
    planned = plan_artifact(
        root,
        **arguments,
        reuse_validator=validator,
    )
    group = start_artifact(root, planned)
    group.attrs.update({"quality": 0, "approved": True})

    with pytest.raises(ValueError, match="attribute 'quality'"):
        finish_artifact(group, planned)
    assert group.attrs["complete"] is False

    group.attrs.update({"quality": 1, "approved": False})
    with pytest.raises(ValueError, match="reuse contract"):
        finish_artifact(group, planned)
    assert group.attrs["complete"] is False

    group.attrs["approved"] = True
    finish_artifact(group, planned)
    reused = plan_artifact(
        root,
        **arguments,
        reuse_validator=validator,
    )
    rejected = plan_artifact(
        root,
        **arguments,
        reuse_validator=lambda _ref, _group: False,
    )
    assert reused.reused
    assert reused.ref == planned.ref
    assert not rejected.reused


def test_artifact_writer_reuse_skips_invalid_array_candidates() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "reduction",
        "operation": "run_reduction",
        "parameters": {},
        "inputs": {},
        "execution_options": {},
    }
    valid = plan_artifact(root, **arguments)
    valid_group = start_artifact(root, valid)
    valid_group.create_array(
        "data",
        data=np.arange(6, dtype=np.float32).reshape(3, 2),
    )
    finish_artifact(valid_group, valid)

    invalid = plan_artifact(root, **arguments, invalidate_cache=True)
    invalid_group = start_artifact(root, invalid)
    invalid_group.create_group("data")
    finish_artifact(invalid_group, invalid)
    valid_group.attrs["created_at_ns"] = 10
    invalid_group.attrs["created_at_ns"] = 20

    reusable = plan_artifact(
        root,
        **arguments,
        required_arrays=(ArrayRequirement("data", shape=(None, 2), dtype_kind="f"),),
    )
    missing = plan_artifact(
        root,
        **arguments,
        required_arrays=("missing",),
    )
    wrong_rank = plan_artifact(
        root,
        **arguments,
        required_arrays=(ArrayRequirement("data", shape=(None,)),),
    )
    wrong_kind = plan_artifact(
        root,
        **arguments,
        required_arrays=(ArrayRequirement("data", dtype_kind="i"),),
    )

    assert reusable.reused
    assert reusable.ref == valid.ref
    assert not missing.reused
    assert not wrong_rank.reused
    assert not wrong_kind.reused


def test_datastore_exposes_artifact_inspection_without_writing() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _complete_artifact(
        root,
        kind="normalized",
        operation="run_normalization",
        parameters={},
        inputs={},
    )
    datastore = DataStore.__new__(DataStore)
    datastore.z = root
    datastore.workspace = None
    datastore._defaultAssay = "RNA"

    assert datastore.inspect_artifact(ref).complete
    assert datastore.list_artifacts() == [ref]
    assert datastore.list_artifacts(
        operation="run_normalization",
        parameters={},
        inputs={},
    ) == [ref]


def test_completed_artifact_requires_full_provenance_record() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _ref(artifact_id="4" * 64)
    group = root.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": ref.kind,
            "complete": True,
        }
    )
    with pytest.raises(KeyError, match="missing attrs"):
        inspect_artifact(root, ref)

    group.attrs.update(
        {
            "provenance": {
                "operation": "run_normalization",
                "parameters": {},
                "inputs": {},
            },
            "execution_options": {},
        }
    )
    assert inspect_artifact(root, ref).complete

    group.attrs["complete"] = "false"
    with pytest.raises(TypeError, match="must be boolean"):
        inspect_artifact(root, ref)
