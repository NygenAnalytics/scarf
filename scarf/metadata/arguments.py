from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from ..graph.arguments import (
    OperationArguments,
    artifact_input,
    execution,
    parameter,
)
from ..storage.artifacts import ArtifactRef


@dataclass(frozen=True, slots=True)
class UmapArguments(OperationArguments):
    operation: ClassVar[str] = "run_umap"
    artifact_kind: ClassVar[str] = "embedding"

    graph: ArtifactRef = artifact_input()
    initialization: Any = artifact_input()
    symmetric_graph: bool | None = parameter()
    graph_upper_only: bool | None = parameter()
    umap_dims: int = parameter()
    spread: float = parameter()
    min_dist: float = parameter()
    n_epochs: int = parameter()
    repulsion_strength: float = parameter()
    initial_alpha: float = parameter()
    negative_sample_rate: float = parameter()
    use_density_map: bool = parameter()
    dens_lambda: float = parameter()
    dens_frac: float = parameter()
    dens_var_shift: float = parameter()
    random_seed: int = parameter()
    parallel: bool = parameter()
    parallel_threads: int | None = parameter()
    label: str = execution()
    from_assay: str = execution()
    cell_key: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class TsneArguments(OperationArguments):
    operation: ClassVar[str] = "run_tsne"
    artifact_kind: ClassVar[str] = "embedding"

    graph: ArtifactRef = artifact_input()
    initialization: Any = artifact_input()
    symmetric_graph: bool = parameter()
    graph_upper_only: bool = parameter()
    tsne_dims: int = parameter()
    lambda_scale: float = parameter()
    max_iter: int = parameter()
    early_iter: int = parameter()
    alpha: int = parameter()
    box_h: float = parameter()
    parallel: bool = parameter()
    parallel_threads: int = parameter()
    label: str = execution()
    from_assay: str = execution()
    cell_key: str = execution()
    temp_file_loc: str = execution()
    verbose: bool = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class LeidenArguments(OperationArguments):
    operation: ClassVar[str] = "run_leiden_clustering"
    artifact_kind: ClassVar[str] = "cluster_labels"

    graph: ArtifactRef = artifact_input()
    resolution: float = parameter()
    backend: Literal["igraph", "leidenalg"] = parameter()
    symmetric_graph: bool = parameter()
    graph_upper_only: bool = parameter()
    random_seed: int = parameter()
    label: str = execution()
    from_assay: str = execution()
    cell_key: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class TopacedoArguments(OperationArguments):
    operation: ClassVar[str] = "run_topacedo_sampler"
    artifact_kind: ClassVar[str] = "sampling"

    graph: ArtifactRef = artifact_input()
    clusters: Any = artifact_input()
    dendrogram: Any = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    use_k: int | None = parameter()
    density_depth: int = parameter()
    density_bandwidth: float = parameter()
    max_sampling_rate: float = parameter()
    min_sampling_rate: float = parameter()
    min_cells_per_group: int = parameter()
    snn_bandwidth: float = parameter()
    seed_reward: float = parameter()
    non_seed_reward: float = parameter()
    edge_cost_multiplier: float = parameter()
    edge_cost_bandwidth: float = parameter()
    rand_state: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    cluster_key: str = execution()
    save_sampling_key: str = execution()
    save_density_key: str = execution()
    save_mean_snn_key: str = execution()
    save_seeds_key: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class DoubletScoreArguments(OperationArguments):
    operation: ClassVar[str] = "run_doublet_detection"
    artifact_kind: ClassVar[str] = "doublet_score"

    clusters: Any = artifact_input()
    connectivity_map: ArtifactRef = artifact_input()
    neighbors: ArtifactRef = artifact_input()
    cluster_sample_fraction: float = parameter()
    max_cells_per_cluster: int = parameter()
    simulation_ratio: float = parameter()
    heterotypic_fraction: float = parameter()
    save_k: int = parameter()
    smoothing_t: int = parameter()
    normalize_scores: bool = parameter()
    random_seed: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    label: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class CellCycleArguments(OperationArguments):
    operation: ClassVar[str] = "run_cell_cycle_scoring"
    artifact_kind: ClassVar[str] = "cell_cycle"

    feature_summary: ArtifactRef = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    s_gene_indices: tuple[int, ...] = parameter()
    g2m_gene_indices: tuple[int, ...] = parameter()
    control_size: int = parameter()
    n_bins: int = parameter()
    rand_seed: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    s_score_label: str = execution()
    g2m_score_label: str = execution()
    phase_label: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class MarkerTableArguments(OperationArguments):
    operation: ClassVar[str] = "run_marker_search"
    artifact_kind: ClassVar[str] = "marker_table"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    clusters: Any = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    method: str = parameter()
    alternative: str = parameter()
    tie_correction: bool = parameter()
    continuity_correction: bool = parameter()
    adjustment_method: str = parameter()
    adjustment_scope: str = parameter()
    group_key: str = execution()
    cell_key: str = execution()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PseudotimeScoringArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_scoring"
    artifact_kind: ClassVar[str] = "pseudotime"

    connectivity_map: ArtifactRef = artifact_input()
    source_sink: Any = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    n_singular_vals: int = parameter()
    sources: tuple[Any, ...] = parameter()
    sinks: tuple[Any, ...] = parameter()
    min_max_norm_ptime: bool = parameter()
    random_seed: int = parameter()
    component_policy: str = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    subset_cell_key: str = execution()
    label: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class FateMappingArguments(OperationArguments):
    operation: ClassVar[str] = "run_fate_mapping"
    artifact_kind: ClassVar[str] = "fate_map"

    connectivity_map: ArtifactRef = artifact_input()
    pseudotime: Any = artifact_input()
    sink_labels: Any = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    sinks: tuple[Any, ...] = parameter()
    beta: float = parameter()
    solver_tol: float = parameter()
    max_iterations: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    subset_cell_key: str = execution()
    pseudotime_key: str = execution()
    sink_key: str = execution()
    label: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PseudotimeMarkerArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_marker_search"
    artifact_kind: ClassVar[str] = "pseudotime_markers"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    pseudotime: Any = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    association_method: str = parameter()
    p_value_method: str = parameter()
    adjustment_method: str = parameter()
    adjustment_scope: str = parameter()
    min_cells: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    pseudotime_key: str = execution()
    gene_batch_size: int | None = execution()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PseudotimeAggregationArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_aggregation"
    artifact_kind: ClassVar[str] = "pseudotime_aggregation"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    pseudotime: Any = artifact_input()
    normalization: dict[str, Any] = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float | None = parameter()
    min_exp: float = parameter()
    window_size: int = parameter()
    chunk_size: int = parameter()
    smoothen: bool = parameter()
    z_scale: bool = parameter()
    n_neighbours: int = parameter()
    n_clusters: int = parameter()
    ann_params: dict[str, Any] = parameter()
    nan_cluster_value: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    pseudotime_key: str = execution()
    cluster_label: str = execution()
    batch_size: int | None = execution()
    nthreads: int = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class PrevalentPeakArguments(OperationArguments):
    operation: ClassVar[str] = "mark_prevalent_peaks"
    artifact_kind: ClassVar[str] = "feature_selection"

    feature_summary: ArtifactRef = artifact_input()
    top_n: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    label: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class WaggrArguments(OperationArguments):
    operation: ClassVar[str] = "run_waggr"
    artifact_kind: ClassVar[str] = "enrichment_scores"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    network_digest: str = artifact_input()
    algorithm_version: int = parameter()
    mode: str = parameter()
    tmin: int = parameter()
    log_transform: bool = parameter()
    normalization_method: dict[str, str] = parameter()
    size_factor: float = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    label: str = execution()
    overwrite: bool = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class AucellArguments(OperationArguments):
    operation: ClassVar[str] = "run_aucell"
    artifact_kind: ClassVar[str] = "enrichment_scores"

    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    network_digest: str = artifact_input()
    algorithm_version: int = parameter()
    tmin: int = parameter()
    n_up: int = parameter()
    tie_seed: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    label: str = execution()
    overwrite: bool = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class HtoIdentityArguments(OperationArguments):
    operation: ClassVar[str] = "mark_hto_identities"
    artifact_kind: ClassVar[str] = "hto_identity"

    cell_selection: ArtifactRef = artifact_input()
    feature_ids_fingerprint: str = artifact_input()
    method: dict[str, object] = parameter()
    random_seed: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    label: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class MembershipStrengthArguments(OperationArguments):
    operation: ClassVar[str] = "calc_membership_strength"
    artifact_kind: ClassVar[str] = "membership_strength"

    connectivity_map: ArtifactRef = artifact_input()
    clusters: Any = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    algorithm_version: int = parameter()
    decimals: int = parameter()
    from_assay: str = execution()
    cell_key: str = execution()
    clust_key: str = execution()
    output_key: str = execution()
    invalidate_cache: bool = execution()


@dataclass(frozen=True, slots=True)
class SmartLabelArguments(OperationArguments):
    operation: ClassVar[str] = "smart_label"
    artifact_kind: ClassVar[str] = "smart_label"

    values: Any = artifact_input()
    base_labels: Any = artifact_input()
    cell_selection: ArtifactRef = artifact_input()
    algorithm_version: int = parameter()
    suffix_style: str = parameter()
    to_relabel: str = execution()
    base_label: str = execution()
    cell_key: str = execution()
    new_col_name: str = execution()
    invalidate_cache: bool = execution()
