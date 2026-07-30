from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, svds

from ..utils.logging import logger


def validate_source_sink_labels(
    labels: pd.Series,
    sources: list[Any],
    sinks: list[Any],
    context: str,
) -> None:
    overlap = sorted(set(sources) & set(sinks))
    if overlap:
        raise ValueError(f"Source and sink labels overlap in {context}: {overlap}")

    present = set(pd.unique(labels))
    missing_sources = [source for source in sources if source not in present]
    missing_sinks = [sink for sink in sinks if sink not in present]
    if missing_sources or missing_sinks:
        raise ValueError(
            f"Source/sink labels were not found in {context}. "
            f"Missing sources: {missing_sources}; missing sinks: {missing_sinks}"
        )


def make_source_sink_vector(
    labels: pd.Series,
    sources: list[Any],
    sinks: list[Any],
) -> np.ndarray:
    sink_mask = labels.isin(sinks).to_numpy(dtype=bool)
    source_mask = labels.isin(sources).to_numpy(dtype=bool)
    labelled = sink_mask | source_mask
    if labelled.all():
        raise ValueError(
            "All selected cells are labelled as sources or sinks, so the "
            "source/sink vector cannot be balanced over unlabelled cells"
        )

    vector = np.zeros(labels.shape[0], dtype=float)
    vector[sink_mask] = 1.0
    vector[source_mask] = -1.0
    vector[~labelled] = -vector.sum() / int((~labelled).sum())
    return vector


def validate_source_sink_vector(
    values: np.ndarray,
    n_cells: int,
    context: str,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must contain numeric values") from exc

    if vector.ndim == 2 and vector.shape == (n_cells, 1):
        vector = vector[:, 0]
    elif vector.ndim != 1:
        raise ValueError(
            f"{context} must be one-dimensional or have shape ({n_cells}, 1)"
        )

    if vector.shape[0] != n_cells:
        raise ValueError(
            f"Size mismatch between {context} ({vector.shape[0]}) and graph ({n_cells})"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{context} must contain only finite values")

    tolerance = 1e-10 * max(1.0, float(np.abs(vector).sum()))
    if not np.isclose(vector.sum(), 0.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"The values in {context} must sum to zero")
    return vector


def select_pseudotime_component(
    graph: csr_matrix,
    selected_cell_indices: np.ndarray,
    component_policy: Literal["largest", "error"],
) -> tuple[np.ndarray, list[int]]:
    if component_policy not in {"largest", "error"}:
        raise ValueError("component_policy must be either 'largest' or 'error'")

    n_components, component_labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    sizes = np.bincount(component_labels, minlength=n_components).astype(int).tolist()
    if n_components == 1:
        return np.ones(graph.shape[0], dtype=bool), sizes
    if component_policy == "error":
        raise ValueError(
            f"The selected graph has {n_components} connected components with sizes {sizes}"
        )

    retained_component = min(
        range(n_components),
        key=lambda component_id: (
            -sizes[component_id],
            int(selected_cell_indices[component_labels == component_id].min()),
        ),
    )
    return np.asarray(component_labels == retained_component, dtype=bool), sizes


def random_walk_laplacian_transpose(graph: csr_matrix) -> csr_matrix:
    degree = np.asarray(graph.sum(axis=1), dtype=float).ravel()
    if np.any(degree <= 0):
        raise ValueError("The retained graph contains isolated cells")
    n_cells = graph.shape[0]
    inverse_degree = csr_matrix(
        (1.0 / degree, (range(n_cells), range(n_cells))),
        shape=(n_cells, n_cells),
    )
    identity = csr_matrix(
        (np.ones(n_cells), (range(n_cells), range(n_cells))),
        shape=(n_cells, n_cells),
    )
    return identity - graph.dot(inverse_degree)


def truncated_pba_potential(
    laplacian_transpose: csr_matrix,
    n_singular_vals: int,
    random_seed: int,
    source_sink: np.ndarray,
) -> np.ndarray:
    random_state = np.random.RandomState(random_seed)
    initial_vector = random_state.rand(laplacian_transpose.shape[0])
    logger.debug(
        f"Pseudotime scoring: calculating SVD "
        f"(shape={laplacian_transpose.shape}, nnz={laplacian_transpose.nnz}, "
        f"k={n_singular_vals})"
    )
    try:
        left_vectors, singular_values, right_vectors_t = svds(
            laplacian_transpose,
            k=n_singular_vals,
            which="SM",
            v0=initial_vector,
        )
    except ArpackNoConvergence as exc:
        raise RuntimeError(
            "Pseudotime SVD did not converge. Try a smaller n_singular_vals value"
        ) from exc

    order = np.argsort(singular_values)
    singular_values = singular_values[order]
    left_vectors = left_vectors[:, order]
    right_vectors = right_vectors_t[order, :].T

    rank_tolerance = (
        max(laplacian_transpose.shape)
        * np.finfo(singular_values.dtype).eps
        * max(1.0, float(singular_values.max()))
    )
    if singular_values[0] > max(1e-8, rank_tolerance):
        raise ValueError("The graph Laplacian does not contain the expected null mode")
    if np.any(singular_values[1:] <= rank_tolerance):
        raise ValueError(
            "The graph Laplacian contains additional near-zero singular modes"
        )

    inverse_singular_values = 1.0 / singular_values[1:]
    left_vectors = left_vectors[:, 1:]
    right_vectors = right_vectors[:, 1:]
    return np.asarray(
        left_vectors
        @ (inverse_singular_values * (right_vectors.T @ source_sink).ravel())
    )
