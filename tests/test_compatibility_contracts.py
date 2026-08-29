import inspect
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scarf
import scarf.datastore.summary as datastore_summary
import scarf.features as feature_algorithms
import scarf.mapping as mapping
import scarf.trajectory as trajectory


_EXPECTED_TRAJECTORY_EXPORTS = (
    "FateMappingResult",
    "PseudotimeAggregationResult",
    "PseudotimeMarkerResult",
    "PseudotimeScoreResult",
    "aggregate_feature_profiles",
    "make_source_sink_vector",
    "random_walk_laplacian_transpose",
    "select_pseudotime_component",
    "scatter_feature_clusters",
    "truncated_pba_potential",
    "validate_source_sink_labels",
    "validate_source_sink_vector",
    "validate_pseudotime_regressor",
)

_EXPECTED_RESULT_FIELDS = {
    datastore_summary.ResourceSummary: (
        "memory_bytes",
        "workers",
        "storage_profile",
    ),
    datastore_summary.ArtifactSummary: (
        "ref",
        "operation",
        "complete",
    ),
    datastore_summary.AssaySummary: (
        "name",
        "assay_type",
        "total_features",
        "active_features",
        "feature_columns",
        "dataset_fingerprint",
        "artifacts",
    ),
    datastore_summary.DataStoreSummary: (
        "zarr_mode",
        "workspace",
        "default_assay",
        "scarf_version",
        "resources",
        "total_cells",
        "active_cells",
        "cell_columns",
        "assays",
        "artifacts",
        "pipeline_run_counts",
        "labeled_pipeline_runs",
    ),
    feature_algorithms.EnrichmentResult: (
        "data",
        "source_names",
        "source_sizes",
        "cell_index",
        "artifact",
        "storage_path",
        "assay",
        "cell_selection",
        "feature_selection",
        "method",
    ),
    mapping.MappingResult: (
        "ref",
        "n_cells",
        "correction_method",
        "diagnostics",
        "reference",
        "indices",
        "distances",
        "uninformative",
    ),
    trajectory.FateMappingResult: (
        "ref",
        "graph",
        "pseudotime",
        "sink_labels_artifact",
        "cell_selection",
        "sink_labels",
        "values",
        "valid",
    ),
    trajectory.PseudotimeScoreResult: (
        "ref",
        "graph",
        "cell_selection",
        "values",
        "valid",
    ),
    trajectory.PseudotimeMarkerResult: (
        "ref",
        "table",
        "assay",
        "cell_selection",
        "feature_selection",
        "pseudotime",
    ),
    trajectory.PseudotimeAggregationResult: (
        "ref",
        "data",
        "feature_indices",
        "feature_clusters",
        "assay",
        "cell_selection",
        "feature_selection",
        "pseudotime",
    ),
}


def test_result_facades_and_constructor_fields_are_stable():
    assert scarf.DataStoreSummary is datastore_summary.DataStoreSummary
    assert scarf.EnrichmentResult is feature_algorithms.EnrichmentResult
    assert scarf.MappingResult is mapping.MappingResult
    assert tuple(trajectory.__all__) == _EXPECTED_TRAJECTORY_EXPORTS
    for name in (
        "FateMappingResult",
        "PseudotimeAggregationResult",
        "PseudotimeMarkerResult",
        "PseudotimeScoreResult",
    ):
        assert getattr(scarf, name) is getattr(trajectory, name)

    for result_class, expected_fields in _EXPECTED_RESULT_FIELDS.items():
        assert tuple(field.name for field in fields(result_class)) == expected_fields
        assert tuple(inspect.signature(result_class).parameters) == expected_fields
        assert result_class.__dataclass_params__.frozen

    mapping_parameters = inspect.signature(mapping.MappingResult).parameters
    for name in (
        "indices",
        "distances",
        "uninformative",
    ):
        assert mapping_parameters[name].default is None
    assert mapping_parameters["reference"].default is inspect.Parameter.empty


def test_result_records_reject_attribute_assignment():
    feature_selection = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        artifact_id="f" * 64,
    )
    graph = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="c" * 64,
    )
    enrichment = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="enrichment_scores",
        artifact_id="e" * 64,
    )
    cell_selection = scarf.ArtifactRef(
        scope="datastore",
        assay=None,
        kind="cell_selection",
        artifact_id="d" * 64,
    )
    pseudotime = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="pseudotime",
        artifact_id="b" * 64,
    )
    sink_labels = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        artifact_id="9" * 64,
    )
    records = (
        feature_algorithms.EnrichmentResult(
            SimpleNamespace(shape=(2, 1)),
            np.array(["set"]),
            np.array([2]),
            np.array([0, 1]),
            enrichment,
            "_artifacts/assays/RNA/enrichment_scores/result",
            "RNA",
            cell_selection,
            feature_selection,
            "waggr",
        ),
        mapping.MappingResult(
            scarf.ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="projection",
                artifact_id="a" * 64,
            ),
            2,
            "none",
            {},
            SimpleNamespace(),
        ),
        trajectory.FateMappingResult(
            scarf.ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="fate_map",
                artifact_id="8" * 64,
            ),
            graph,
            pseudotime,
            sink_labels,
            cell_selection,
            ("A", "B"),
            np.array([[0.25, 0.75], [1.0, 0.0]]),
            np.array([True, True]),
        ),
        trajectory.PseudotimeScoreResult(
            pseudotime,
            graph,
            cell_selection,
            np.array([0.0, 1.0]),
            np.array([True, True]),
        ),
        trajectory.PseudotimeMarkerResult(
            scarf.ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="pseudotime_markers",
                artifact_id="7" * 64,
            ),
            pd.DataFrame(
                {
                    "feature_index": [0],
                    "feature_name": ["gene"],
                    "r_value": [1.0],
                    "p_value": [0.0],
                }
            ),
            "RNA",
            cell_selection,
            feature_selection,
            pseudotime,
        ),
        trajectory.PseudotimeAggregationResult(
            scarf.ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="pseudotime_aggregation",
                artifact_id="6" * 64,
            ),
            SimpleNamespace(shape=(2, 3)),
            np.array([0, 1]),
            np.array([0, 1]),
            "RNA",
            cell_selection,
            feature_selection,
            pseudotime,
        ),
    )

    for record in records:
        first_field = fields(record)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(record, first_field, None)

    aggregation = records[-1]
    assert isinstance(aggregation, trajectory.PseudotimeAggregationResult)
    assert not hasattr(aggregation, "__dict__")
    aggregation._attach_feature_identity(
        np.array(["g0", "g1"]),
        np.array(["id0", "id1"]),
    )
    with pytest.raises(RuntimeError, match="already attached"):
        aggregation._attach_feature_identity(
            np.array(["g0", "g1"]),
            np.array(["id0", "id1"]),
        )


def test_pseudotime_result_shape_validation_is_stable():
    feature_selection = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        artifact_id="f" * 64,
    )
    graph = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="c" * 64,
    )
    cell_selection = scarf.ArtifactRef(
        scope="datastore",
        assay=None,
        kind="cell_selection",
        artifact_id="d" * 64,
    )
    pseudotime = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="pseudotime",
        artifact_id="b" * 64,
    )
    sink_labels = scarf.ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        artifact_id="9" * 64,
    )
    with pytest.raises(ValueError, match="rows must align"):
        trajectory.FateMappingResult(
            scarf.ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="fate_map",
                artifact_id="8" * 64,
            ),
            graph,
            pseudotime,
            sink_labels,
            cell_selection,
            ("A", "B"),
            np.array([[0.25, 0.75], [1.0, 0.0]]),
            np.array([True]),
        )

    with pytest.raises(ValueError, match="same shape"):
        trajectory.PseudotimeScoreResult(
            pseudotime,
            graph,
            cell_selection,
            np.array([0.0, 1.0]),
            np.array([True]),
        )

    with pytest.raises(ValueError, match="missing columns"):
        trajectory.PseudotimeMarkerResult(
            scarf.ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="pseudotime_markers",
                artifact_id="7" * 64,
            ),
            pd.DataFrame({"feature_index": [0]}),
            "RNA",
            cell_selection,
            feature_selection,
            pseudotime,
        )

    with pytest.raises(ValueError, match="Feature clusters"):
        trajectory.PseudotimeAggregationResult(
            scarf.ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="pseudotime_aggregation",
                artifact_id="6" * 64,
            ),
            SimpleNamespace(shape=(2, 3)),
            np.array([0, 1]),
            np.array([0]),
            "RNA",
            cell_selection,
            feature_selection,
            pseudotime,
        )


def test_mapping_execution_contract_is_query_owned():
    signature = inspect.signature(scarf.DataStore.run_mapping)
    assert tuple(signature.parameters) == (
        "self",
        "reference",
        "cell_selection",
        "query_assay",
        "save_k",
        "missing_feature_policy",
        "query_batches",
        "invalidate_cache",
    )
    params = signature.parameters
    assert params["query_assay"].default is None
    assert params["save_k"].default == 3
    assert params["missing_feature_policy"].default == "reference_mean"
    assert params["query_batches"].default is None
    assert params["invalidate_cache"].default is False
    # pandas 2 stringifies as pandas.core.frame.DataFrame; pandas 3 as pandas.DataFrame.
    assert "DataFrame | None" in str(params["query_batches"].annotation)
    return_annotation = signature.return_annotation
    assert getattr(return_annotation, "__name__", str(return_annotation)).endswith(
        "ArtifactRef"
    )
    with pytest.raises(TypeError, match="unexpected keyword argument 'target_assay'"):
        scarf.DataStore.run_mapping(
            object(),
            object(),
            "legacy",
            target_assay=object(),
        )


def test_retired_unified_mapping_contract_is_absent():
    for name in (
        "_load_unified_layout_data",
        "load_unified_graph",
        "run_unified_tsne",
        "run_unified_umap",
    ):
        assert not hasattr(scarf.DataStore, name)
