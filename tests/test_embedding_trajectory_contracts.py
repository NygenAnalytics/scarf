from dataclasses import fields
from importlib.util import find_spec

from scarf.embeddings.sgtsne import run_sgtsne
from scarf.neighbors.stream import AnnStream
from scarf.trajectory.feature_dynamics import validate_pseudotime_regressor
from scarf.trajectory.results import PseudotimeScoreResult
from tests.signature_contracts import signature_digest


def test_embedding_and_trajectory_entry_point_signatures_are_stable():
    methods = {
        "AnnStream.__init__": AnnStream.__init__,
        "run_sgtsne": run_sgtsne,
        "validate_pseudotime_regressor": validate_pseudotime_regressor,
    }
    assert signature_digest(methods) == (
        "b963e7139e72eea4843182060350a8c907e973014df48a84a13806ec6b435dfd"
    )
    assert [field.name for field in fields(PseudotimeScoreResult)] == [
        "pseudotime_key",
        "validity_key",
        "assay",
        "graph_cell_key",
        "result_cell_key",
        "feature_key",
        "values",
        "valid",
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
