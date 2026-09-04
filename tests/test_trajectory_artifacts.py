import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.arrays import create_metadata_column
from scarf.storage.artifact_writer import finish_artifact, plan_artifact, start_artifact
from scarf.storage.artifacts import (
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
)
from scarf.storage.selections import (
    resolve_metadata_snapshot,
    resolve_stored_selection_artifact,
)
from scarf.trajectory.artifacts import (
    AGGREGATION_PAYLOAD,
    FATE_PAYLOAD,
    MARKER_PAYLOAD,
    PSEUDOTIME_PAYLOAD,
    aggregation_payload_is_valid,
    fate_payload_is_valid,
    labels_with_missing_mask,
    load_cell_artifact_values,
    marker_payload_is_valid,
    pseudotime_payload_is_valid,
    validate_aggregation_parameters,
    validate_fate_parameters,
    validate_marker_parameters,
    validate_pseudotime_parameters,
    validate_resolved_ann_parameters,
)
from scarf.trajectory.parameters import resolve_aggregation_ann_params


def _payload_group(payload: dict[str, np.ndarray]) -> zarr.Group:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("candidate")
    for name, values in payload.items():
        group.create_array(name, data=np.asarray(values))
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        group,
        tuple(payload),
    )
    group.attrs.update(
        {
            "artifact_id": "a" * 64,
            "kind": "test_payload",
            "provenance": {},
            "execution_options": {},
            "created_at_ns": 1,
            "scarf_version": "test",
            "complete": True,
        }
    )
    return group


def _refresh_payload_fingerprint(group: zarr.Group, names: tuple[str, ...]) -> None:
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, names)


def _selection_store() -> tuple[zarr.Group, object]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    table = root.create_group("cellData")
    create_metadata_column(
        table,
        "ids",
        data=np.array(["c0", "c1", "c2"]),
        dtype=str,
    )
    create_metadata_column(
        table,
        "I",
        data=np.array([True, False, True]),
        dtype=bool,
    )
    selection = resolve_stored_selection_artifact(
        root,
        table_path="cellData",
        id_column="ids",
        source_column="I",
        scope="datastore",
        kind="cell_selection",
        operation="manual_selection",
        parameters={},
        inputs={},
    )
    return root, selection


def test_pseudotime_payload_rejects_same_shape_semantic_corruption() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 0.5, 1.0, np.nan]),
            "valid": np.array([True, True, True, False]),
        }
    )
    assert pseudotime_payload_is_valid(
        group,
        n_cells=4,
        min_max_normalized=True,
        expected_valid=np.array([True, True, True, False]),
    )

    group["pseudotime"][1] = np.nan
    _refresh_payload_fingerprint(group, PSEUDOTIME_PAYLOAD)
    assert not pseudotime_payload_is_valid(
        group,
        n_cells=4,
        min_max_normalized=True,
        expected_valid=np.array([True, True, True, False]),
    )


def test_pseudotime_payload_requires_a_nonconstant_exact_normalized_range() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.25, 0.5, 0.75]),
            "valid": np.array([True, True, True]),
        }
    )
    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=True,
        expected_valid=np.ones(3, dtype=bool),
    )

    group["pseudotime"][:] = np.array([0.5, 0.5, 0.5])
    _refresh_payload_fingerprint(group, PSEUDOTIME_PAYLOAD)
    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=False,
        expected_valid=np.ones(3, dtype=bool),
    )


def test_pseudotime_payload_requires_the_exact_component_validity_mask() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 1.0, np.nan]),
            "valid": np.array([True, True, False]),
        }
    )
    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=True,
        expected_valid=np.array([True, False, True]),
    )


def test_pseudotime_payload_rejects_array_attribute_tampering() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 0.5, 1.0]),
            "valid": np.array([True, True, True]),
        }
    )
    group["pseudotime"].attrs["missing_mask"] = "valid"

    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=True,
        expected_valid=np.ones(3, dtype=bool),
    )


def test_pseudotime_payload_rejects_unknown_artifact_attributes() -> None:
    group = _payload_group(
        {
            "pseudotime": np.array([0.0, 0.5, 1.0]),
            "valid": np.array([True, True, True]),
        }
    )
    group.attrs["schema_version"] = 1

    assert not pseudotime_payload_is_valid(
        group,
        n_cells=3,
        min_max_normalized=True,
        expected_valid=np.ones(3, dtype=bool),
    )


def test_fate_payload_rejects_invalid_rows_and_boundary_columns() -> None:
    labels = np.array(["A", "middle", "B", "missing"], dtype=object)
    group = _payload_group(
        {
            "probabilities": np.array(
                [[1.0, 0.0], [0.4, 0.6], [0.0, 1.0], [np.nan, np.nan]],
                dtype=np.float32,
            ),
            "valid": np.array([True, True, True, False]),
        }
    )
    assert fate_payload_is_valid(
        group,
        n_cells=4,
        n_sinks=2,
        pseudotime_valid=np.array([True, True, True, False]),
        sink_values=labels,
        sink_labels=["A", "B"],
    )

    group["probabilities"][0] = np.array([0.0, 1.0], dtype=np.float32)
    _refresh_payload_fingerprint(group, FATE_PAYLOAD)
    assert not fate_payload_is_valid(
        group,
        n_cells=4,
        n_sinks=2,
        pseudotime_valid=np.array([True, True, True, False]),
        sink_values=labels,
        sink_labels=["A", "B"],
    )


def test_marker_payload_freezes_feature_identity_for_reuse() -> None:
    names = np.array(["g0", "g1", "g2"])
    ids = np.array(["id0", "id1", "id2"])
    group = _payload_group(
        {
            "r_value": np.array([0.5, np.nan, -0.4]),
            "p_value": np.array([0.01, np.nan, 0.02]),
            "p_value_adjusted": np.array([0.02, np.nan, 0.03]),
            "feature_names": names,
            "feature_ids": ids,
        }
    )
    expected_ids = fingerprint_stored_strings(group["feature_ids"])
    expected_names = fingerprint_stored_strings(group["feature_names"])
    assert marker_payload_is_valid(
        group,
        n_features=3,
        selected_features=np.array([0, 2]),
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
    )

    group["feature_names"][0] = "renamed"
    _refresh_payload_fingerprint(group, MARKER_PAYLOAD)
    assert not marker_payload_is_valid(
        group,
        n_features=3,
        selected_features=np.array([0, 2]),
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
    )

    group["feature_names"][:] = names
    group["p_value"][0] = np.inf
    _refresh_payload_fingerprint(group, MARKER_PAYLOAD)
    assert not marker_payload_is_valid(
        group,
        n_features=3,
        selected_features=np.array([0, 2]),
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
    )


def test_aggregation_payload_rejects_reordered_feature_rows() -> None:
    names = np.array(["g0", "g1", "g2", "g3"])
    ids = np.array(["id0", "id1", "id2", "id3"])
    group = _payload_group(
        {
            "data": np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
            "feature_indices": np.array([0, 2, 3], dtype=np.uint64),
            "valid_features": np.array([True, False, True]),
            "feature_clusters": np.array([1, -1, 2]),
            "cluster_values": np.array([1, -1, -1, 2]),
            "feature_names": names,
            "feature_ids": ids,
        }
    )
    group.attrs["input_fingerprints"] = ["cells", "features", "ordering"]
    group.attrs["nan_cluster_value"] = -1
    group.attrs["effective_window"] = 2
    group.attrs["effective_bins"] = 2
    expected_ids = fingerprint_stored_strings(group["feature_ids"])
    expected_names = fingerprint_stored_strings(group["feature_names"])
    resolved_ann = resolve_aggregation_ann_params(None, dim=2)
    assert aggregation_payload_is_valid(
        group,
        n_features=4,
        selected_features=np.array([0, 2, 3]),
        n_bins=2,
        n_clusters=2,
        n_neighbours=1,
        nan_cluster_value=-1,
        ann_params=resolved_ann,
        expected_input_fingerprints=["cells", "features", "ordering"],
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
        effective_window=2,
    )
    assert not aggregation_payload_is_valid(
        group,
        n_features=4,
        selected_features=np.array([0, 2, 3]),
        n_bins=2,
        n_clusters=2,
        n_neighbours=1,
        nan_cluster_value=-1,
        ann_params={},
        expected_input_fingerprints=["cells", "features", "ordering"],
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
        effective_window=2,
    )
    assert not aggregation_payload_is_valid(
        group,
        n_features=4,
        selected_features=np.array([0, 2, 3]),
        n_bins=2,
        n_clusters=2,
        n_neighbours=1,
        nan_cluster_value=-1,
        ann_params={**resolved_ann, "max_elements": 1},
        expected_input_fingerprints=["cells", "features", "ordering"],
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
        effective_window=2,
    )

    group["feature_indices"][:] = np.array([2, 0, 3], dtype=np.uint64)
    _refresh_payload_fingerprint(group, AGGREGATION_PAYLOAD)
    assert not aggregation_payload_is_valid(
        group,
        n_features=4,
        selected_features=np.array([0, 2, 3]),
        n_bins=2,
        n_clusters=2,
        n_neighbours=1,
        nan_cluster_value=-1,
        ann_params=resolved_ann,
        expected_input_fingerprints=["cells", "features", "ordering"],
        expected_feature_ids_fingerprint=expected_ids,
        expected_feature_names_fingerprint=expected_names,
        effective_window=2,
    )


def test_raw_source_sink_snapshot_is_replayable_and_selection_aligned() -> None:
    root, selection = _selection_store()
    values = np.array([-1.0, 1.0])
    snapshot = resolve_metadata_snapshot(
        root,
        values=values,
        row_ids=np.array(["c0", "c2"]),
        operation="snapshot_pseudotime_source_sink",
        parameters={},
        inputs={"cell_selection": selection},
        source_columns=["ss_vec"],
    )

    loaded, loaded_selection, missing = load_cell_artifact_values(root, snapshot)
    np.testing.assert_array_equal(loaded, values)
    assert loaded_selection == selection
    assert missing is None
    assert (inspect_artifact(root, snapshot).inputs or {})["cell_selection"] == (
        selection.to_dict()
    )


def test_nullable_zero_label_is_not_treated_as_an_endpoint() -> None:
    root, selection = _selection_store()
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        operation="import_cluster_labels",
        parameters={},
        inputs={"cell_selection": selection},
        execution_options={},
    )
    group = start_artifact(root, planned)
    values = group.create_array("values", data=np.array([0, 1]))
    group.create_array("__scarf_missing__values", data=np.array([True, False]))
    values.attrs["missing_mask"] = "__scarf_missing__values"
    finish_artifact(group, planned)

    loaded, loaded_selection, missing = load_cell_artifact_values(root, planned.ref)
    assert loaded_selection == selection
    assert missing is not None
    masked = labels_with_missing_mask(loaded, missing, "labels")
    np.testing.assert_array_equal(masked == 0, np.array([False, False]))
    np.testing.assert_array_equal(masked == 1, np.array([False, True]))


def test_trajectory_parameter_contracts_reject_ambiguous_or_coerced_values() -> None:
    pseudotime = {
        "n_singular_vals": np.int64(3),
        "sources": [0],
        "sinks": [1],
        "min_max_norm_ptime": np.bool_(True),
        "random_seed": np.int64(7),
        "component_policy": "largest",
    }
    validated = validate_pseudotime_parameters(pseudotime)
    assert validated["n_singular_vals"] == 3
    assert type(validated["n_singular_vals"]) is int
    assert validated["min_max_norm_ptime"] is True
    with pytest.raises(ValueError, match="disjoint"):
        validate_pseudotime_parameters({**pseudotime, "sources": [1], "sinks": [True]})
    with pytest.raises(TypeError, match="n_singular_vals"):
        validate_pseudotime_parameters({**pseudotime, "n_singular_vals": True})

    fate = {
        "sinks": ["terminal"],
        "beta": 10.0,
        "solver_tol": 1e-6,
        "max_iterations": 1000,
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_fate_parameters({**fate, "sinks": [1, 1.0]})
    with pytest.raises(ValueError, match="solver_tol"):
        validate_fate_parameters({**fate, "solver_tol": 0.0})

    marker = {
        "normalization": {
            "log_transform": False,
            "renormalize_subset": False,
        },
        "normalization_method": {"module": "m", "qualname": "normalize"},
        "size_factor": 1000.0,
        "association_method": "pearson",
        "p_value_method": "student_t",
        "adjustment_method": "fdr_bh",
        "adjustment_scope": "tested_features",
        "min_cells": 10,
    }
    with pytest.raises(TypeError, match="log_transform"):
        validate_marker_parameters(
            {
                **marker,
                "normalization": {
                    **marker["normalization"],
                    "log_transform": 1,
                },
            }
        )
    with pytest.raises(ValueError, match="association_method"):
        validate_marker_parameters({**marker, "association_method": "spearman"})

    aggregation = {
        "normalization": marker["normalization"],
        "normalization_method": marker["normalization_method"],
        "size_factor": marker["size_factor"],
        "min_exp": 1e-3,
        "window_size": 20,
        "chunk_size": 10,
        "smoothen": True,
        "z_scale": True,
        "n_neighbours": 2,
        "n_clusters": 3,
        "ann_params": {},
        "nan_cluster_value": -1,
    }
    with pytest.raises(TypeError, match="smoothen"):
        validate_aggregation_parameters({**aggregation, "smoothen": 1})
    with pytest.raises(ValueError, match="conflicts"):
        validate_aggregation_parameters({**aggregation, "nan_cluster_value": 1})
    with pytest.raises(ValueError, match="unsupported"):
        validate_aggregation_parameters({**aggregation, "ann_params": {"ignored": 1}})
    with pytest.raises(ValueError, match="resolved ANN contract"):
        validate_resolved_ann_parameters({}, dim=10)
    with pytest.raises(ValueError, match="effective bin count"):
        resolve_aggregation_ann_params({"dim": 9}, dim=10)
    with pytest.raises(TypeError, match="mapping or None"):
        resolve_aggregation_ann_params([("M", 20)], dim=10)  # type: ignore[arg-type]
