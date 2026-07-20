import inspect
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scarf
import scarf.mapping as mapping
import scarf.trajectory as trajectory
from scarf.merge import ZarrMerge


_EXPECTED_TRAJECTORY_EXPORTS = (
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
    mapping.MappingResult: (
        "projection_path",
        "n_cells",
        "correction_method",
        "diagnostics",
        "indices",
        "distances",
        "uncorrected_latent",
        "corrected_latent",
        "uninformative",
    ),
    trajectory.PseudotimeScoreResult: (
        "pseudotime_key",
        "validity_key",
        "assay",
        "graph_cell_key",
        "result_cell_key",
        "feature_key",
        "values",
        "valid",
    ),
    trajectory.PseudotimeMarkerResult: (
        "table",
        "correlation_key",
        "p_value_key",
        "assay",
        "cell_key",
        "feature_key",
        "pseudotime_key",
    ),
    trajectory.PseudotimeAggregationResult: (
        "data",
        "feature_indices",
        "feature_clusters",
        "cluster_key",
        "storage_path",
        "assay",
        "cell_key",
        "feature_key",
        "pseudotime_key",
    ),
}


def test_result_facades_and_constructor_fields_are_stable():
    assert scarf.MappingResult is mapping.MappingResult
    assert tuple(trajectory.__all__) == _EXPECTED_TRAJECTORY_EXPORTS
    for name in (
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
        "uncorrected_latent",
        "corrected_latent",
        "uninformative",
    ):
        assert mapping_parameters[name].default is None


def test_result_records_reject_attribute_assignment():
    records = (
        mapping.MappingResult("projection", 2, "none", {}),
        trajectory.PseudotimeScoreResult(
            "pseudotime",
            "valid",
            "RNA",
            "I",
            "I",
            "hvgs",
            np.array([0.0, 1.0]),
            np.array([True, True]),
        ),
        trajectory.PseudotimeMarkerResult(
            pd.DataFrame(
                {
                    "feature_index": [0],
                    "feature_name": ["gene"],
                    "r_value": [1.0],
                    "p_value": [0.0],
                }
            ),
            "correlation",
            "p_value",
            "RNA",
            "I",
            "hvgs",
            "pseudotime",
        ),
        trajectory.PseudotimeAggregationResult(
            SimpleNamespace(shape=(2, 3)),
            np.array([0, 1]),
            np.array([0, 1]),
            "clusters",
            "aggregated",
            "RNA",
            "I",
            "hvgs",
            "pseudotime",
        ),
    )

    for record in records:
        first_field = fields(record)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(record, first_field, None)


def test_pseudotime_result_shape_validation_is_stable():
    with pytest.raises(ValueError, match="same shape"):
        trajectory.PseudotimeScoreResult(
            "pseudotime",
            "valid",
            "RNA",
            "I",
            "I",
            "hvgs",
            np.array([0.0, 1.0]),
            np.array([True]),
        )

    with pytest.raises(ValueError, match="missing columns"):
        trajectory.PseudotimeMarkerResult(
            pd.DataFrame({"feature_index": [0]}),
            "correlation",
            "p_value",
            "RNA",
            "I",
            "hvgs",
            "pseudotime",
        )

    with pytest.raises(ValueError, match="Feature clusters"):
        trajectory.PseudotimeAggregationResult(
            SimpleNamespace(shape=(2, 3)),
            np.array([0, 1]),
            np.array([0]),
            "clusters",
            "aggregated",
            "RNA",
            "I",
            "hvgs",
            "pseudotime",
        )


def test_zarr_merge_uses_standard_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="Scarf 2.0") as caught:
        with pytest.raises(TypeError):
            ZarrMerge()

    assert caught[0].filename == __file__


def test_metric_integration_uses_standard_deprecation_warning():
    class Store:
        @staticmethod
        def metric_label_concordance(labels, metric):
            assert labels == ["first", "second"]
            assert metric == "ari"
            return 1.0

    with pytest.warns(DeprecationWarning, match="Scarf 2.0") as caught:
        result = scarf.DataStore.metric_integration(
            Store(),
            ["first", "second"],
        )

    assert result == 1.0
    assert caught[0].filename == __file__


def test_metric_batch_mixing_uses_standard_deprecation_warning():
    class Store:
        @staticmethod
        def metric_proportional_batch_mixing(
            label_colname,
            use_latest_knn,
            from_assay,
            knn_loc,
            perplexity,
        ):
            assert (
                label_colname,
                use_latest_knn,
                from_assay,
                knn_loc,
                perplexity,
            ) == ("batch", False, "RNA", "knn", 7)
            return 0.5

    with pytest.warns(DeprecationWarning, match="Scarf 2.0") as caught:
        result = scarf.DataStore.metric_batch_mixing(
            Store(),
            "batch",
            False,
            "RNA",
            "knn",
            7,
        )

    assert result == 0.5
    assert caught[0].filename == __file__
    assert inspect.signature(scarf.DataStore.metric_batch_mixing) == inspect.signature(
        scarf.DataStore.metric_proportional_batch_mixing
    )


def test_metric_silhouette_uses_standard_deprecation_warning():
    class Store:
        @staticmethod
        def metric_graph_silhouette(
            use_latest_knn,
            res_label,
            from_assay,
            knn_loc,
            random_seed,
            sample_size,
        ):
            assert (
                use_latest_knn,
                res_label,
                from_assay,
                knn_loc,
                random_seed,
                sample_size,
            ) == (False, "clusters", "RNA", "knn", 42, 5)
            return np.array([0.25])

    with pytest.warns(DeprecationWarning, match="Scarf 2.0") as caught:
        result = scarf.DataStore.metric_silhouette(
            Store(),
            False,
            "clusters",
            "RNA",
            "knn",
            42,
            5,
        )

    assert np.array_equal(result, np.array([0.25]))
    assert caught[0].filename == __file__
    assert inspect.signature(scarf.DataStore.metric_silhouette) == inspect.signature(
        scarf.DataStore.metric_graph_silhouette
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ref_mu": False}, "ref_mu"),
        ({"run_coral": True}, "CORAL"),
        ({"exclude_missing": True}, "exclude_missing"),
    ],
)
def test_mapping_deprecations_warn_before_validation(kwargs, message):
    class Store:
        @staticmethod
        def _get_latest_keys(*args):
            raise RuntimeError("stop after compatibility warnings")

    with pytest.warns(DeprecationWarning, match=message) as caught:
        with pytest.raises(RuntimeError, match="stop after"):
            scarf.DataStore.run_mapping(
                Store(),
                target_assay=object(),
                target_name="query",
                target_feat_key="I",
                **kwargs,
            )

    assert caught[0].filename == __file__
