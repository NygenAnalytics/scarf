import inspect

from scarf.features.genomic.gff import GffReader
from scarf.features.genomic.melding import coordinate_melding
from scarf.mapping.reference import MappingReference
from scarf.features.markers.search import (
    find_markers_by_rank,
    find_markers_by_regression,
)
from scarf.metrics.lisi import compute_lisi
from scarf.metrics.silhouette import silhouette_scoring


def test_feature_mapping_and_metric_entry_point_signatures_are_stable():
    assert list(inspect.signature(GffReader).parameters) == [
        "gff_fn",
        "up_offset",
        "down_offset",
        "chunk_size",
    ]
    assert list(inspect.signature(MappingReference.map_query).parameters) == [
        "self",
        "target_assay",
        "target_name",
        "target_feat_key",
        "target_cell_key",
        "save_k",
        "query_batches",
        "correction_method",
        "missing_feature_policy",
        "result_store",
    ]
    assert list(inspect.signature(compute_lisi).parameters) == [
        "distances",
        "indices",
        "metadata",
        "label_colnames",
        "perplexity",
    ]
    assert list(inspect.signature(silhouette_scoring).parameters) == [
        "ds",
        "ann_obj",
        "graph",
        "hvg_data",
        "assay_type",
        "res_label",
        "cell_key",
        "random_seed",
        "sample_size",
        "data_is_reduced",
        "distance_metric",
        "neighbor_indices",
        "neighbor_distances",
    ]
    assert list(inspect.signature(find_markers_by_rank).parameters) == [
        "assay",
        "group_key",
        "cell_key",
        "feat_key",
        "batch_size",
        "n_threads",
        "prefetch_depth",
        "norm_params",
    ]
    assert list(inspect.signature(find_markers_by_regression).parameters) == [
        "assay",
        "cell_key",
        "feat_key",
        "regressor",
        "min_cells",
        "batch_size",
        "norm_params",
    ]
    assert list(inspect.signature(coordinate_melding).parameters) == [
        "assay",
        "workspace",
        "feature_bed",
        "new_assay_name",
        "peaks_col",
        "scalar_coeff",
        "renormalization",
        "peaks_coords",
    ]
