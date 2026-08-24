"""
Methods and classes for evaluation
"""

from ._types import MatrixData, NeighborMetric, ZarrArray
from .association import (
    association_pair,
    coefficient_estimability,
    cramers_v,
    directional_mapping,
    eta_squared,
    report_confounding,
    report_technical_nesting,
    spearman_rho,
)
from .cluster_separability import (
    ClusterSeparabilityResult,
    evaluate_cluster_separability,
)
from .connectivity import graph_connectivity
from .concordance import label_concordance_score
from .graph import (
    calculate_knn_cluster_similarity,
    calculate_top_k_neighbor_distances,
    calculate_weighted_cluster_similarity,
    knn_to_csr_matrix,
)
from .lisi import (
    clisi_knn,
    compute_lisi,
    compute_simpson,
    ilisi_knn,
    lisi_batch_mixing_score,
)
from .silhouette import process_cluster, silhouette_scoring

__all__ = [
    "ClusterSeparabilityResult",
    "MatrixData",
    "NeighborMetric",
    "ZarrArray",
    "association_pair",
    "calculate_knn_cluster_similarity",
    "calculate_top_k_neighbor_distances",
    "calculate_weighted_cluster_similarity",
    "clisi_knn",
    "coefficient_estimability",
    "compute_lisi",
    "compute_simpson",
    "cramers_v",
    "directional_mapping",
    "eta_squared",
    "evaluate_cluster_separability",
    "graph_connectivity",
    "ilisi_knn",
    "knn_to_csr_matrix",
    "label_concordance_score",
    "lisi_batch_mixing_score",
    "process_cluster",
    "report_confounding",
    "report_technical_nesting",
    "silhouette_scoring",
    "spearman_rho",
]
