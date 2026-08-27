from dataclasses import fields
from importlib.util import find_spec
from inspect import signature

from scarf.datastore.datastore import DataStore
from scarf.embeddings.sgtsne import run_sgtsne
from scarf.neighbors.stream import AnnStream
from scarf.trajectory.feature_dynamics import validate_pseudotime_regressor
from scarf.trajectory.results import (
    FateMappingResult,
    PseudotimeAggregationResult,
    PseudotimeMarkerResult,
    PseudotimeScoreResult,
)
from tests.signature_contracts import signature_digest


def test_embedding_and_trajectory_entry_point_signatures_are_stable() -> None:
    methods = {
        "AnnStream.__init__": AnnStream.__init__,
        "run_sgtsne": run_sgtsne,
        "validate_pseudotime_regressor": validate_pseudotime_regressor,
    }
    assert signature_digest(methods) == (
        "b963e7139e72eea4843182060350a8c907e973014df48a84a13806ec6b435dfd"
    )


def test_trajectory_results_describe_artifacts_not_metadata_columns() -> None:
    assert [field.name for field in fields(PseudotimeScoreResult)] == [
        "ref",
        "graph",
        "cell_selection",
        "values",
        "valid",
    ]
    assert [field.name for field in fields(FateMappingResult)] == [
        "ref",
        "graph",
        "pseudotime",
        "sink_labels_artifact",
        "cell_selection",
        "sink_labels",
        "values",
        "valid",
    ]
    assert [field.name for field in fields(PseudotimeMarkerResult)] == [
        "ref",
        "table",
        "assay",
        "cell_selection",
        "feature_selection",
        "pseudotime",
    ]
    assert [field.name for field in fields(PseudotimeAggregationResult)] == [
        "ref",
        "data",
        "feature_indices",
        "feature_clusters",
        "assay",
        "cell_selection",
        "feature_selection",
        "pseudotime",
    ]


def test_artifact_only_producers_drop_metadata_output_arguments() -> None:
    forbidden = {
        "cell_key",
        "output_assay",
        "label",
        "cluster_key",
        "cluster_label",
        "new_col_name",
        "pseudotime_key",
        "sink_key",
    }
    producers = (
        DataStore.run_umap,
        DataStore.run_tsne,
        DataStore.run_leiden_clustering,
        DataStore.run_paris_clustering,
        DataStore.run_topacedo_sampler,
        DataStore.calc_membership_strength,
        DataStore.smart_label,
        DataStore.run_pseudotime_scoring,
        DataStore.run_fate_mapping,
        DataStore.run_pseudotime_marker_search,
        DataStore.run_pseudotime_aggregation,
    )
    for producer in producers:
        assert forbidden.isdisjoint(signature(producer).parameters)


def test_trajectory_loaders_are_explicit_public_datastore_methods() -> None:
    for name in (
        "load_paris_clustering",
        "load_pseudotime_scoring",
        "load_fate_mapping",
        "load_pseudotime_markers",
        "load_pseudotime_aggregation",
    ):
        parameters = signature(getattr(DataStore, name)).parameters
        assert tuple(parameters) == ("self", "ref")


def test_run_consumers_take_explicit_artifacts() -> None:
    silhouette = signature(DataStore.metric_graph_silhouette).parameters
    assert tuple(silhouette) == (
        "self",
        "neighbors",
        "clusters",
        "random_seed",
        "sample_size",
    )
    assert "run" in signature(DataStore.to_anndata).parameters


def test_moved_symbols_are_absent_from_old_hybrid_modules() -> None:
    from scarf.datastore import datastore, graph_datastore
    from scarf.features import markers

    assert find_spec("scarf.knn_utils") is None
    retired = {
        markers: {"knn_clustering"},
        datastore: {
            "_scatter_feature_clusters",
            "_validated_pseudotime_regressor",
        },
        graph_datastore: {
            "_make_source_sink_vector",
            "_random_walk_laplacian_transpose",
            "_select_pseudotime_component",
            "_truncated_pba_potential",
            "_validate_source_sink_labels",
            "_validate_source_sink_vector",
        },
    }
    for module, names in retired.items():
        assert names.isdisjoint(vars(module))
