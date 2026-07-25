from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..graph.arguments import OperationArguments
from ..storage.artifacts import ArtifactRef


@dataclass(frozen=True, slots=True)
class UmapArguments(OperationArguments):
    operation: ClassVar[str] = "run_umap"
    artifact_kind: ClassVar[str] = "embedding"

    graph: Any = field(metadata={"argument_role": "input"})
    initialization: Any = field(metadata={"argument_role": "input"})
    symmetric_graph: bool | None = field(metadata={"argument_role": "parameter"})
    graph_upper_only: bool | None = field(metadata={"argument_role": "parameter"})
    umap_dims: int = field(metadata={"argument_role": "parameter"})
    spread: float = field(metadata={"argument_role": "parameter"})
    min_dist: float = field(metadata={"argument_role": "parameter"})
    n_epochs: int = field(metadata={"argument_role": "parameter"})
    repulsion_strength: float = field(metadata={"argument_role": "parameter"})
    initial_alpha: float = field(metadata={"argument_role": "parameter"})
    negative_sample_rate: float = field(metadata={"argument_role": "parameter"})
    use_density_map: bool = field(metadata={"argument_role": "parameter"})
    dens_lambda: float = field(metadata={"argument_role": "parameter"})
    dens_frac: float = field(metadata={"argument_role": "parameter"})
    dens_var_shift: float = field(metadata={"argument_role": "parameter"})
    random_seed: int = field(metadata={"argument_role": "parameter"})
    parallel: bool = field(metadata={"argument_role": "parameter"})
    parallel_threads: int | None = field(metadata={"argument_role": "parameter"})
    label: str = field(metadata={"argument_role": "execution"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    integrated_graph: str | None = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class TsneArguments(OperationArguments):
    operation: ClassVar[str] = "run_tsne"
    artifact_kind: ClassVar[str] = "embedding"

    graph: Any = field(metadata={"argument_role": "input"})
    initialization: Any = field(metadata={"argument_role": "input"})
    symmetric_graph: bool = field(metadata={"argument_role": "parameter"})
    graph_upper_only: bool = field(metadata={"argument_role": "parameter"})
    tsne_dims: int = field(metadata={"argument_role": "parameter"})
    lambda_scale: float = field(metadata={"argument_role": "parameter"})
    max_iter: int = field(metadata={"argument_role": "parameter"})
    early_iter: int = field(metadata={"argument_role": "parameter"})
    alpha: int = field(metadata={"argument_role": "parameter"})
    box_h: float = field(metadata={"argument_role": "parameter"})
    parallel: bool = field(metadata={"argument_role": "parameter"})
    parallel_threads: int = field(metadata={"argument_role": "parameter"})
    label: str = field(metadata={"argument_role": "execution"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    temp_file_loc: str = field(metadata={"argument_role": "execution"})
    verbose: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class LeidenArguments(OperationArguments):
    operation: ClassVar[str] = "run_leiden_clustering"
    artifact_kind: ClassVar[str] = "cluster_labels"

    graph: Any = field(metadata={"argument_role": "input"})
    resolution: float = field(metadata={"argument_role": "parameter"})
    symmetric_graph: bool = field(metadata={"argument_role": "parameter"})
    graph_upper_only: bool = field(metadata={"argument_role": "parameter"})
    random_seed: int = field(metadata={"argument_role": "parameter"})
    label: str = field(metadata={"argument_role": "execution"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    integrated_graph: str | None = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class TopacedoArguments(OperationArguments):
    operation: ClassVar[str] = "run_topacedo_sampler"
    artifact_kind: ClassVar[str] = "sampling"

    graph: Any = field(metadata={"argument_role": "input"})
    clusters: Any = field(metadata={"argument_role": "input"})
    dendrogram: Any = field(metadata={"argument_role": "input"})
    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    use_k: int | None = field(metadata={"argument_role": "parameter"})
    density_depth: int = field(metadata={"argument_role": "parameter"})
    density_bandwidth: float = field(metadata={"argument_role": "parameter"})
    max_sampling_rate: float = field(metadata={"argument_role": "parameter"})
    min_sampling_rate: float = field(metadata={"argument_role": "parameter"})
    min_cells_per_group: int = field(metadata={"argument_role": "parameter"})
    snn_bandwidth: float = field(metadata={"argument_role": "parameter"})
    seed_reward: float = field(metadata={"argument_role": "parameter"})
    non_seed_reward: float = field(metadata={"argument_role": "parameter"})
    edge_cost_multiplier: float = field(metadata={"argument_role": "parameter"})
    edge_cost_bandwidth: float = field(metadata={"argument_role": "parameter"})
    rand_state: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    integrated_graph: str | None = field(metadata={"argument_role": "execution"})
    cluster_key: str = field(metadata={"argument_role": "execution"})
    save_sampling_key: str = field(metadata={"argument_role": "execution"})
    save_density_key: str = field(metadata={"argument_role": "execution"})
    save_mean_snn_key: str = field(metadata={"argument_role": "execution"})
    save_seeds_key: str = field(metadata={"argument_role": "execution"})
    return_edges: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class DoubletScoreArguments(OperationArguments):
    operation: ClassVar[str] = "run_doublet_detection"
    artifact_kind: ClassVar[str] = "doublet_score"

    clusters: Any = field(metadata={"argument_role": "input"})
    connectivity_map: Any = field(metadata={"argument_role": "input"})
    cluster_sample_fraction: float = field(metadata={"argument_role": "parameter"})
    max_cells_per_cluster: int = field(metadata={"argument_role": "parameter"})
    simulation_ratio: float = field(metadata={"argument_role": "parameter"})
    heterotypic_fraction: float = field(metadata={"argument_role": "parameter"})
    save_k: int = field(metadata={"argument_role": "parameter"})
    smoothing_t: int = field(metadata={"argument_role": "parameter"})
    normalize_scores: bool = field(metadata={"argument_role": "parameter"})
    random_seed: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    label: str = field(metadata={"argument_role": "execution"})
    batch_size: int = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class CellCycleArguments(OperationArguments):
    operation: ClassVar[str] = "run_cell_cycle_scoring"
    artifact_kind: ClassVar[str] = "cell_cycle"

    s_gene_indices: tuple[int, ...] = field(metadata={"argument_role": "input"})
    g2m_gene_indices: tuple[int, ...] = field(metadata={"argument_role": "input"})
    normalization_method: dict[str, str] = field(
        metadata={"argument_role": "parameter"}
    )
    size_factor: float | None = field(metadata={"argument_role": "parameter"})
    control_size: int = field(metadata={"argument_role": "parameter"})
    n_bins: int = field(metadata={"argument_role": "parameter"})
    rand_seed: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    s_score_label: str = field(metadata={"argument_role": "execution"})
    g2m_score_label: str = field(metadata={"argument_role": "execution"})
    phase_label: str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class MarkerTableArguments(OperationArguments):
    operation: ClassVar[str] = "run_marker_search"
    artifact_kind: ClassVar[str] = "marker_table"

    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    clusters: Any = field(metadata={"argument_role": "input"})
    normalization: dict[str, Any] = field(metadata={"argument_role": "parameter"})
    normalization_method: dict[str, str] = field(
        metadata={"argument_role": "parameter"}
    )
    size_factor: float | None = field(metadata={"argument_role": "parameter"})
    group_key: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    gene_batch_size: int = field(metadata={"argument_role": "execution"})
    n_threads: int = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class PseudotimeScoringArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_scoring"
    artifact_kind: ClassVar[str] = "pseudotime"

    connectivity_map: Any = field(metadata={"argument_role": "input"})
    source_sink: Any = field(metadata={"argument_role": "input"})
    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    n_singular_vals: int = field(metadata={"argument_role": "parameter"})
    sources: tuple[Any, ...] = field(metadata={"argument_role": "parameter"})
    sinks: tuple[Any, ...] = field(metadata={"argument_role": "parameter"})
    min_max_norm_ptime: bool = field(metadata={"argument_role": "parameter"})
    random_seed: int = field(metadata={"argument_role": "parameter"})
    component_policy: str = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    subset_cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    label: str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class FateMappingArguments(OperationArguments):
    operation: ClassVar[str] = "run_fate_mapping"
    artifact_kind: ClassVar[str] = "fate_map"

    connectivity_map: Any = field(metadata={"argument_role": "input"})
    pseudotime: Any = field(metadata={"argument_role": "input"})
    sink_labels: Any = field(metadata={"argument_role": "input"})
    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    sinks: tuple[Any, ...] = field(metadata={"argument_role": "parameter"})
    beta: float = field(metadata={"argument_role": "parameter"})
    solver_tol: float = field(metadata={"argument_role": "parameter"})
    max_iterations: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    subset_cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    pseudotime_key: str = field(metadata={"argument_role": "execution"})
    sink_key: str = field(metadata={"argument_role": "execution"})
    label: str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class PseudotimeMarkerArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_marker_search"
    artifact_kind: ClassVar[str] = "pseudotime_markers"

    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    pseudotime: Any = field(metadata={"argument_role": "input"})
    normalization: dict[str, Any] = field(metadata={"argument_role": "parameter"})
    normalization_method: dict[str, str] = field(
        metadata={"argument_role": "parameter"}
    )
    size_factor: float | None = field(metadata={"argument_role": "parameter"})
    min_cells: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    pseudotime_key: str = field(metadata={"argument_role": "execution"})
    gene_batch_size: int = field(metadata={"argument_role": "execution"})
    n_threads: int = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class PseudotimeAggregationArguments(OperationArguments):
    operation: ClassVar[str] = "run_pseudotime_aggregation"
    artifact_kind: ClassVar[str] = "pseudotime_aggregation"

    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    pseudotime: Any = field(metadata={"argument_role": "input"})
    normalization: dict[str, Any] = field(metadata={"argument_role": "parameter"})
    normalization_method: dict[str, str] = field(
        metadata={"argument_role": "parameter"}
    )
    size_factor: float | None = field(metadata={"argument_role": "parameter"})
    min_exp: float = field(metadata={"argument_role": "parameter"})
    window_size: int = field(metadata={"argument_role": "parameter"})
    chunk_size: int = field(metadata={"argument_role": "parameter"})
    smoothen: bool = field(metadata={"argument_role": "parameter"})
    z_scale: bool = field(metadata={"argument_role": "parameter"})
    n_neighbours: int = field(metadata={"argument_role": "parameter"})
    n_clusters: int = field(metadata={"argument_role": "parameter"})
    ann_params: dict[str, Any] = field(metadata={"argument_role": "parameter"})
    nan_cluster_value: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    pseudotime_key: str = field(metadata={"argument_role": "execution"})
    cluster_label: str = field(metadata={"argument_role": "execution"})
    batch_size: int = field(metadata={"argument_role": "execution"})
    n_threads: int = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class PrevalentPeakArguments(OperationArguments):
    operation: ClassVar[str] = "mark_prevalent_peaks"
    artifact_kind: ClassVar[str] = "feature_selection"

    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    normalization_method: dict[str, str] = field(
        metadata={"argument_role": "parameter"}
    )
    algorithm_version: int = field(metadata={"argument_role": "parameter"})
    top_n: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    prevalence_key_name: str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class WaggrArguments(OperationArguments):
    operation: ClassVar[str] = "run_waggr"
    artifact_kind: ClassVar[str] = "enrichment_scores"

    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    network_digest: str = field(metadata={"argument_role": "input"})
    algorithm_version: int = field(metadata={"argument_role": "parameter"})
    mode: str = field(metadata={"argument_role": "parameter"})
    tmin: int = field(metadata={"argument_role": "parameter"})
    log_transform: bool = field(metadata={"argument_role": "parameter"})
    normalization_method: dict[str, str] = field(
        metadata={"argument_role": "parameter"}
    )
    size_factor: float = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    label: str = field(metadata={"argument_role": "execution"})
    overwrite: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class AucellArguments(OperationArguments):
    operation: ClassVar[str] = "run_aucell"
    artifact_kind: ClassVar[str] = "enrichment_scores"

    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    network_digest: str = field(metadata={"argument_role": "input"})
    algorithm_version: int = field(metadata={"argument_role": "parameter"})
    tmin: int = field(metadata={"argument_role": "parameter"})
    n_up: int = field(metadata={"argument_role": "parameter"})
    tie_seed: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    label: str = field(metadata={"argument_role": "execution"})
    overwrite: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class HtoIdentityArguments(OperationArguments):
    operation: ClassVar[str] = "mark_hto_identities"
    artifact_kind: ClassVar[str] = "hto_identity"

    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_ids_fingerprint: str = field(metadata={"argument_role": "input"})
    algorithm_version: int = field(metadata={"argument_role": "parameter"})
    random_seed: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    label: str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class MembershipStrengthArguments(OperationArguments):
    operation: ClassVar[str] = "calc_membership_strength"
    artifact_kind: ClassVar[str] = "membership_strength"

    connectivity_map: Any = field(metadata={"argument_role": "input"})
    clusters: Any = field(metadata={"argument_role": "input"})
    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    algorithm_version: int = field(metadata={"argument_role": "parameter"})
    decimals: int = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    clust_key: str = field(metadata={"argument_role": "execution"})
    output_key: str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class SmartLabelArguments(OperationArguments):
    operation: ClassVar[str] = "smart_label"
    artifact_kind: ClassVar[str] = "smart_label"

    values: Any = field(metadata={"argument_role": "input"})
    base_labels: Any = field(metadata={"argument_role": "input"})
    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    algorithm_version: int = field(metadata={"argument_role": "parameter"})
    suffix_style: str = field(metadata={"argument_role": "parameter"})
    to_relabel: str = field(metadata={"argument_role": "execution"})
    base_label: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    new_col_name: str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})


@dataclass(frozen=True, slots=True)
class LisiArguments(OperationArguments):
    operation: ClassVar[str] = "metric_lisi"
    artifact_kind: ClassVar[str] = "lisi"

    neighbors: Any = field(metadata={"argument_role": "input"})
    labels: tuple[Any, ...] = field(metadata={"argument_role": "input"})
    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    algorithm_version: int = field(metadata={"argument_role": "parameter"})
    perplexity: float = field(metadata={"argument_role": "parameter"})
    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    label_colnames: tuple[str, ...] = field(metadata={"argument_role": "execution"})
    save_result: bool = field(metadata={"argument_role": "execution"})
    return_lisi: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(metadata={"argument_role": "execution"})
