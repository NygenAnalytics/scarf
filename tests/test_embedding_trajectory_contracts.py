import inspect
from importlib.util import find_spec

from scarf.embeddings.sgtsne import run_sgtsne
from scarf.neighbors.stream import AnnStream
from scarf.trajectory.feature_dynamics import validate_pseudotime_regressor
from scarf.trajectory.results import PseudotimeScoreResult


def test_embedding_and_trajectory_entry_point_signatures_are_stable():
    assert list(inspect.signature(AnnStream).parameters) == [
        "data",
        "k",
        "n_cluster",
        "reduction_method",
        "dims",
        "loadings",
        "use_for_pca",
        "mu",
        "sigma",
        "ann_metric",
        "ann_efc",
        "ann_ef",
        "ann_m",
        "nthreads",
        "ann_parallel",
        "rand_state",
        "do_kmeans_fit",
        "disable_scaling",
        "ann_idx",
        "lsi_skip_first",
        "lsi_params",
        "harmonize",
        "harmonized_data",
        "batches",
        "cache_embeddings",
        "harmony_params",
    ]
    assert list(inspect.signature(run_sgtsne).parameters) == [
        "graph",
        "ini_embed",
        "tsne_dims",
        "max_iter",
        "early_iter",
        "alpha",
        "lambda_scale",
        "box_h",
        "temp_file_loc",
        "verbose",
        "parallel",
        "nthreads",
    ]
    assert list(inspect.signature(PseudotimeScoreResult).parameters) == [
        "pseudotime_key",
        "validity_key",
        "assay",
        "graph_cell_key",
        "result_cell_key",
        "feature_key",
        "values",
        "valid",
    ]
    assert list(inspect.signature(validate_pseudotime_regressor).parameters) == [
        "values",
        "expected_size",
        "pseudotime_key",
        "cell_key",
        "has_validity_column",
    ]


def test_moved_symbols_are_absent_from_old_hybrid_modules():
    from scarf.features import markers
    from scarf.datastore import datastore, graph_datastore

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
