from scarf.features.markers.search import (
    find_markers_by_rank,
    find_markers_by_regression,
)
from scarf.metrics.lisi import compute_lisi
from scarf.metrics.silhouette import silhouette_scoring
from tests.signature_contracts import signature_digest


def test_feature_mapping_and_metric_entry_point_signatures_are_stable():
    methods = {
        "compute_lisi": compute_lisi,
        "find_markers_by_rank": find_markers_by_rank,
        "find_markers_by_regression": find_markers_by_regression,
        "silhouette_scoring": silhouette_scoring,
    }

    assert signature_digest(methods) == (
        "40fa971874bff7e877c3484708a175f9ae7824d21a6f2a0ddeab96122015cd74"
    )
