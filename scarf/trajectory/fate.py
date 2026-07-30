import math
import re
from typing import Any

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, gmres

from ..utils.logging import logger


@njit(cache=True, inline="always")
def _log_biased_weight(
    weight: float,
    pseudotime_difference: float,
    beta: float,
) -> float:
    value = math.log(weight)
    if pseudotime_difference > 0.0 and beta > 0.0:
        exponent = beta * pseudotime_difference
        value += math.log(2.0) - exponent - math.log1p(math.exp(-exponent))
    return value


@njit(cache=True)
def _bias_and_normalize_rows(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    pseudotime: np.ndarray,
    absorbing: np.ndarray,
    beta: float,
) -> int:
    """Bias backward edges and normalize CSR rows in place."""
    isolated_transient_count = 0
    tiny = np.finfo(np.float64).tiny
    for row in range(pseudotime.shape[0]):
        maximum_log_weight = -math.inf
        for offset in range(indptr[row], indptr[row + 1]):
            col = indices[offset]
            if col == row:
                data[offset] = 0.0
                continue
            log_weight = _log_biased_weight(
                data[offset],
                pseudotime[row] - pseudotime[col],
                beta,
            )
            maximum_log_weight = max(maximum_log_weight, log_weight)

        if maximum_log_weight == -math.inf:
            if not absorbing[row]:
                isolated_transient_count += 1
            continue

        row_sum = 0.0
        for offset in range(indptr[row], indptr[row + 1]):
            col = indices[offset]
            if col == row:
                continue
            log_weight = _log_biased_weight(
                data[offset],
                pseudotime[row] - pseudotime[col],
                beta,
            )
            scaled_weight = math.exp(log_weight - maximum_log_weight)
            if scaled_weight == 0.0:
                scaled_weight = tiny
            data[offset] = scaled_weight
            row_sum += scaled_weight

        inverse_sum = 1.0 / row_sum
        for offset in range(indptr[row], indptr[row + 1]):
            data[offset] *= inverse_sum
    return isolated_transient_count


@njit(cache=True)
def _has_symmetric_support(
    indices: np.ndarray,
    indptr: np.ndarray,
) -> bool:
    for row in range(indptr.shape[0] - 1):
        for offset in range(indptr[row], indptr[row + 1]):
            col = indices[offset]
            if col == row:
                continue
            lower = indptr[col]
            upper = indptr[col + 1]
            while lower < upper:
                middle = (lower + upper) // 2
                candidate = indices[middle]
                if candidate < row:
                    lower = middle + 1
                else:
                    upper = middle
            if lower >= indptr[col + 1] or indices[lower] != row:
                return False
    return True


def _validate_sink_groups(
    labels: np.ndarray,
    sinks: list[Any],
) -> tuple[tuple[Any, ...], np.ndarray]:
    if not isinstance(sinks, list):
        raise TypeError("sinks must be a list")
    if not sinks:
        raise ValueError("At least one sink label must be provided")
    if any(np.ndim(sink) != 0 for sink in sinks):
        raise TypeError("Sink labels must be hashable scalar values")
    try:
        if len(set(sinks)) != len(sinks):
            raise ValueError("Sink labels must be unique")
    except TypeError as exc:
        raise TypeError("Sink labels must be hashable scalar values") from exc

    sink_labels = tuple(sinks)
    sink_groups = np.full(labels.shape[0], -1, dtype=np.int32)
    missing: list[Any] = []
    for group, sink in enumerate(sink_labels):
        try:
            matches = np.asarray(labels == sink)
        except (TypeError, ValueError) as exc:
            raise TypeError("Sink labels must be comparable scalar values") from exc
        if matches.ndim != 1 or matches.dtype.kind != "b":
            raise TypeError("Sink labels must be comparable scalar values")
        if not matches.any():
            missing.append(sink)
            continue
        if np.any(sink_groups[matches] >= 0):
            raise ValueError("Sink groups must be disjoint")
        sink_groups[matches] = group
    if missing:
        raise ValueError(f"Sink labels were not found in the selected cells: {missing}")
    return sink_labels, sink_groups


def make_sink_tokens(sinks: tuple[Any, ...]) -> tuple[str, ...]:
    """Create deterministic metadata-safe tokens for sink labels."""
    tokens: list[str] = []
    used: set[str] = set()
    for index, sink in enumerate(sinks):
        base = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(sink)).strip("_.-")
        if not base:
            base = f"sink_{index + 1}"
        token = base
        suffix = 2
        while token in used:
            token = f"{base}_{suffix}"
            suffix += 1
        used.add(token)
        tokens.append(token)
    return tuple(tokens)


def _normalize_pseudotime(values: np.ndarray) -> np.ndarray:
    try:
        pseudotime = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("Pseudotime values must be numeric") from exc
    if pseudotime.ndim != 1:
        raise ValueError("Pseudotime values must be one-dimensional")
    if not np.isfinite(pseudotime).all():
        raise ValueError("Pseudotime values must be finite")
    minimum = float(pseudotime.min())
    maximum = float(pseudotime.max())
    scale = max(1.0, abs(minimum), abs(maximum))
    scaled_minimum = minimum / scale
    value_range = (maximum / scale) - scaled_minimum
    if value_range <= np.finfo(np.float64).eps:
        raise ValueError("Pseudotime values must contain at least two distinct values")
    normalized = pseudotime / scale
    normalized -= scaled_minimum
    normalized /= value_range
    if not np.isfinite(normalized).all():
        raise ValueError("Pseudotime values could not be normalized safely")
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized


def _validate_graph(
    graph: csr_matrix,
    n_cells: int,
) -> None:
    if not isinstance(graph, csr_matrix):
        raise TypeError("graph must be a scipy.sparse.csr_matrix")
    try:
        graph.check_format(full_check=True)
    except ValueError as exc:
        raise ValueError("graph has invalid CSR structure") from exc
    if graph.shape != (n_cells, n_cells):
        raise ValueError(
            f"Graph shape {graph.shape} does not match {n_cells} selected cells"
        )
    if graph.data.dtype.kind not in "buif":
        raise TypeError("Graph weights must be real numeric values")
    if not np.isfinite(graph.data).all():
        raise ValueError("Graph weights must be finite")
    if np.any(graph.data < 0):
        raise ValueError("Graph weights must be non-negative")


def _make_transition(
    graph: csr_matrix,
    pseudotime: np.ndarray,
    absorbing: np.ndarray,
    beta: float,
) -> csr_matrix:
    transition = graph
    transition.sum_duplicates()
    transition.eliminate_zeros()
    if transition.dtype != np.float64:
        with np.errstate(over="ignore", invalid="ignore"):
            transition.data = transition.data.astype(np.float64)
    if not np.isfinite(transition.data).all():
        raise ValueError("Graph weights must remain finite when converted to float64")
    if np.any(transition.data <= 0.0):
        raise ValueError("Graph weights must remain positive when converted to float64")
    transition.sort_indices()
    isolated_transient_count = _bias_and_normalize_rows(
        transition.data,
        transition.indices,
        transition.indptr,
        pseudotime,
        absorbing,
        beta,
    )
    transition.eliminate_zeros()
    if isolated_transient_count:
        raise ValueError(
            "The directed graph contains "
            f"{isolated_transient_count} isolated transient cells"
        )
    return transition


def _dirichlet_operator(
    transition: csr_matrix,
    absorbing: np.ndarray,
) -> LinearOperator:
    n_cells = transition.shape[0]

    def matvec(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64) - transition.dot(values)
        result[absorbing] = values[absorbing]
        return np.asarray(result)

    return LinearOperator(
        shape=(n_cells, n_cells),
        matvec=matvec,
        dtype=np.dtype(np.float64),
    )


def _bellman_residual(
    transition: csr_matrix,
    probabilities: np.ndarray,
    sink_groups: np.ndarray,
    group: int,
) -> float:
    boundary = np.asarray(sink_groups == group, dtype=np.float64)
    residual = probabilities[:, group] - transition.dot(probabilities[:, group])
    absorbing = sink_groups >= 0
    residual[absorbing] = probabilities[absorbing, group] - boundary[absorbing]
    minimum = float(residual.min(initial=0.0))
    maximum = float(residual.max(initial=0.0))
    return max(-minimum, maximum)


def _solve_fates(
    transition: csr_matrix,
    sink_groups: np.ndarray,
    n_sinks: int,
    solver_tol: float,
    max_iterations: int,
) -> np.ndarray:
    n_cells = transition.shape[0]
    probabilities = np.zeros((n_cells, n_sinks), dtype=np.float32)
    last_probability = np.ones(n_cells, dtype=np.float64)
    absorbing = sink_groups >= 0

    operator = _dirichlet_operator(transition, absorbing)
    for group in range(n_sinks - 1):
        boundary = np.asarray(sink_groups == group, dtype=np.float64)
        iterations = 0

        def count_iteration(_residual: float) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = gmres(
            operator,
            boundary,
            x0=boundary,
            rtol=solver_tol,
            atol=0.0,
            restart=20,
            maxiter=max_iterations,
            callback=count_iteration,
            # This mode makes maxiter count inner iterations, not restart cycles.
            callback_type="legacy",
        )
        iteration_unit = "iteration" if iterations == 1 else "iterations"
        if info != 0:
            reason = "broke down" if info < 0 else "did not converge"
            raise RuntimeError(
                f"Fate probability solve for sink index {group} {reason} "
                f"after {iterations} {iteration_unit}"
            )
        if not np.isfinite(solution).all():
            raise RuntimeError(
                f"Fate probability solve for sink index {group} produced "
                "non-finite values"
            )
        probabilities[:, group] = solution
        last_probability -= solution
        logger.debug(
            f"Fate mapping: sink {group + 1}/{n_sinks} converged "
            f"in {iterations} {iteration_unit}"
        )

    probabilities[:, -1] = last_probability
    del boundary, last_probability, solution
    probabilities[absorbing] = 0.0
    probabilities[absorbing, sink_groups[absorbing]] = 1.0

    validation_scale = 10.0 * solver_tol * max(1, n_sinks - 1)
    float32_tolerance = 5.0 * float(np.finfo(np.float32).eps)
    check_tol = max(float32_tolerance, min(1e-3, validation_scale))
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Fate probability calculation produced non-finite values")
    minimum = float(probabilities.min())
    maximum = float(probabilities.max())
    if minimum < -check_tol or maximum > 1.0 + check_tol:
        raise RuntimeError(
            "Fate probabilities exceeded numerical bounds "
            f"(minimum={minimum:.3e}, maximum={maximum:.3e})"
        )
    row_sums = probabilities.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=check_tol):
        deviation = float(np.max(np.abs(row_sums - 1.0)))
        raise RuntimeError(
            f"Fate probabilities do not sum to one (maximum deviation={deviation:.3e})"
        )

    np.clip(probabilities, 0.0, 1.0, out=probabilities)
    row_sums = probabilities.sum(axis=1, dtype=np.float64)
    probabilities /= row_sums[:, None]
    probabilities[absorbing] = 0.0
    probabilities[absorbing, sink_groups[absorbing]] = 1.0

    residual_limit = max(float32_tolerance, min(1e-3, validation_scale))
    residuals = [
        _bellman_residual(transition, probabilities, sink_groups, group)
        for group in range(n_sinks)
    ]
    if max(residuals) > residual_limit:
        raise RuntimeError(
            "Fate probabilities failed the Bellman residual check "
            f"(maximum={max(residuals):.3e}, limit={residual_limit:.3e})"
        )
    return probabilities


def compute_fate_probabilities(
    graph: csr_matrix,
    pseudotime: np.ndarray,
    labels: np.ndarray,
    sinks: list[Any],
    *,
    beta: float = 10.0,
    solver_tol: float = 1e-6,
    max_iterations: int = 1000,
    _copy_graph: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    """Compute grouped absorption probabilities on a pseudotime-biased graph."""
    try:
        beta = float(beta)
    except (TypeError, ValueError) as exc:
        raise TypeError("beta must be numeric") from exc
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    try:
        solver_tol = float(solver_tol)
    except (TypeError, ValueError) as exc:
        raise TypeError("solver_tol must be numeric") from exc
    if not np.isfinite(solver_tol) or not 0.0 < solver_tol < 1.0:
        raise ValueError("solver_tol must be finite and between 0 and 1")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Sink labels must be one-dimensional")
    _validate_graph(graph, labels.shape[0])
    if graph.shape[0] == 0:
        raise ValueError("No cells were selected for fate mapping")
    try:
        pseudotime_values = np.asarray(pseudotime, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("Pseudotime values must be numeric") from exc
    if pseudotime_values.shape != (labels.shape[0],):
        raise ValueError("Pseudotime values must align with the selected cells")
    if not np.isfinite(pseudotime_values).all():
        raise ValueError("Pseudotime values must be finite")

    sink_labels, sink_groups = _validate_sink_groups(labels, sinks)
    component_graph = graph
    if not graph.has_canonical_format or np.count_nonzero(graph.data) != graph.nnz:
        component_graph = graph.copy()
        component_graph.sum_duplicates()
        component_graph.eliminate_zeros()
    if not np.isfinite(component_graph.data).all():
        raise ValueError("Graph weights became non-finite when duplicates were summed")
    if not _has_symmetric_support(
        component_graph.indices,
        component_graph.indptr,
    ):
        raise ValueError("Graph support must be symmetric")
    n_components, component_labels = connected_components(
        component_graph,
        directed=False,
        return_labels=True,
    )
    retained_components = np.zeros(n_components, dtype=bool)
    retained_components[component_labels[sink_groups >= 0]] = True
    valid = retained_components[component_labels]
    all_components_retained = bool(retained_components.all())
    if not all_components_retained:
        component_sizes = np.bincount(component_labels, minlength=n_components)
        omitted_sizes = component_sizes[~retained_components]
        displayed_sizes = omitted_sizes[:20].astype(int).tolist()
        display_suffix = (
            "" if omitted_sizes.size <= 20 else f" (showing 20 of {omitted_sizes.size})"
        )
        logger.warning(
            "Fate mapping: omitting sinkless graph components with sizes "
            f"{displayed_sizes}{display_suffix}; "
            f"{int(omitted_sizes.sum())} cells marked invalid"
        )

    retained_pseudotime = _normalize_pseudotime(pseudotime_values[valid])
    if len(sink_labels) == 1:
        probabilities = np.full((labels.shape[0], 1), np.nan, dtype=np.float32)
        probabilities[valid, 0] = 1.0
        return probabilities, valid, sink_labels

    retained_sink_groups = sink_groups[valid]
    if all_components_retained:
        retained_graph = component_graph
        if retained_graph is graph and _copy_graph:
            retained_graph = graph.copy()
    else:
        retained_graph = component_graph[valid][:, valid].tocsr()
    transition = _make_transition(
        retained_graph,
        retained_pseudotime,
        retained_sink_groups >= 0,
        beta,
    )
    retained_probabilities = _solve_fates(
        transition,
        retained_sink_groups,
        len(sink_labels),
        solver_tol,
        max_iterations,
    )

    if all_components_retained:
        probabilities = retained_probabilities
    else:
        probabilities = np.full(
            (labels.shape[0], len(sink_labels)),
            np.nan,
            dtype=np.float32,
        )
        probabilities[valid] = retained_probabilities
    return probabilities, valid, sink_labels
