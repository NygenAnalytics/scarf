"""Graph-build plan types used by ``make_graph``."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GraphDataInputs:
    assay: Any  # Assay
    from_assay: str
    cell_key: str
    feat_key: str
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
    invalidate_cache: bool


@dataclass(frozen=True, slots=True)
class GraphBuildPlan:
    data_inputs: GraphDataInputs
    parameters: ResolvedGraphParameters
    options: GraphExecutionOptions
