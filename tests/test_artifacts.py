from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.assay import norm_lib_size, norm_tf_idf
from scarf.datastore.datastore import DataStore
from scarf.datastore.base_datastore import BaseDataStore
from scarf.storage.artifact_writer import (
    finish_artifact,
    plan_artifact,
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
    with pytest.raises(ValueError, match="assay-scoped"):
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


def test_stored_string_fingerprint_supports_variable_length_strings() -> None:
    values = np.array(["a", "longer", "中"], dtype=np.dtypes.StringDType())
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array("ids", data=values, chunks=(1,))

    expected = fingerprint_strings(values.astype("U6"))
    assert fingerprint_stored_strings(array) == expected


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
