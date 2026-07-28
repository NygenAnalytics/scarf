from scarf.features.genomic.gff import GffReader
from scarf.features.genomic.melding import coordinate_melding
from scarf.mapping.reference import MappingReference
from scarf.features.markers.search import (
    find_markers_by_rank,
    find_markers_by_regression,
)
from scarf.metrics.lisi import compute_lisi
from scarf.metrics.silhouette import silhouette_scoring
from tests.signature_contracts import signature_digest


def test_feature_mapping_and_metric_entry_point_signatures_are_stable():
    methods = {
        "GffReader.__init__": GffReader.__init__,
        "MappingReference.map_query": MappingReference.map_query,
        "compute_lisi": compute_lisi,
        "coordinate_melding": coordinate_melding,
        "find_markers_by_rank": find_markers_by_rank,
        "find_markers_by_regression": find_markers_by_regression,
        "silhouette_scoring": silhouette_scoring,
    }

    assert signature_digest(methods) == (
        "0aa9b3a5fe3e450a625be826fa16b05ae1e797367641d70a8509ac19c37f2d01"
    )
