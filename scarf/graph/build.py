"""Coarse graph-build plan and stage result types.

These are plain holders for orchestration in ``make_graph``. They do not
compute, write Zarr groups, or own AnnStream internals.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .paths import AssayGraphPaths


@dataclass(frozen=True, slots=True)
class GraphDataInputs:
    assay: Any  # Assay
    from_assay: str
    cell_key: str
    feat_key: str
    pca_cell_key: str
    batches: pd.DataFrame | None
    custom_loadings: np.ndarray | None


@dataclass(frozen=True, slots=True)
class ResolvedGraphParameters:
    log_transform: bool
    renormalize_subset: bool
    reduction_method: str
    dims: int
    pca_cell_key: str
    ann_metric: str
    ann_efc: int
    ann_ef: int
    ann_m: int
    rand_state: int
    k: int
    n_centroids: int
    local_connectivity: float
    bandwidth: float
    feat_scaling: bool
    lsi_skip_first: bool
    harmonize: bool
    batch_columns: list[str] | None
    harmony_params: dict[str, Any] | None
    harmony_contract_hash: str | None


@dataclass(frozen=True, slots=True)
class GraphExecutionOptions:
    batch_size: int
    update_keys: bool
    return_ann_object: bool
    show_elbow_plot: bool
    ann_parallel: bool
    ann_index_fetcher: Callable | None
    ann_index_saver: Callable | None
    local_cache: bool | str
    force_harmony_refit: bool


@dataclass(frozen=True, slots=True)
class GraphBuildPlan:
    data_inputs: GraphDataInputs
    parameters: ResolvedGraphParameters
    options: GraphExecutionOptions
    paths: AssayGraphPaths


@dataclass(frozen=True, slots=True)
class NormalizedMatrix:
    data: Any  # ChunkedArray


@dataclass(frozen=True, slots=True)
class FeatureMeansAndScales:
    mu: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True, slots=True)
class NearestNeighbors:
    knn_group_path: str
    recall: str | None
    graph_already_complete: bool


@dataclass(frozen=True, slots=True)
class CellGraph:
    cell_graph_group_path: str


@dataclass(frozen=True, slots=True)
class _GraphBuildOutcome:
    plan: GraphBuildPlan
    ann_stream: Any | None  # AnnStream | None
    cell_graph_group_path: str
    fresh_batch_correction: bool
