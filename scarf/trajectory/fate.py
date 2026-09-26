import math
import re
from typing import Any

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, gmres, splu

from ..utils.logging import logger

_GMRES_RESTART = 20
# Coarse aggregates above this count are merged into fewer pseudotime bins so
# the coarse factorization stays small.
_MAX_COARSE_AGGREGATES = 8192


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


@njit(cache=True)
def _ordered_sweep(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    order: np.ndarray,
    rank: np.ndarray,
    residual: np.ndarray,
    out: np.ndarray,
    descending: bool,
) -> None:
    """Apply one Gauss-Seidel sweep of the Dirichlet operator.

    Transient cells are visited in ``order``, or in reverse, and use values
    already updated earlier in the sweep. Absorbing cells have rank -1 and
    keep their residual.
    """
    for row in range(rank.shape[0]):
        if rank[row] < 0:
            out[row] = residual[row]
    count = order.shape[0]
    for step in range(count):
        row = order[step] if descending else order[count - 1 - step]
        own = rank[row]
        value = residual[row]
        for offset in range(indptr[row], indptr[row + 1]):
            other = rank[indices[offset]]
            if other < 0 or (other < own if descending else other > own):
                value += data[offset] * out[indices[offset]]
        out[row] = value


@njit(cache=True)
def _pseudotime_bin_aggregates(
    indptr: np.ndarray,
    indices: np.ndarray,
    rank: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, int]:
    """Label transient cells by pseudotime bin and connected piece within it."""
    n_cells = rank.shape[0]
    n_transient = 0
    for row in range(n_cells):
        if rank[row] >= 0:
            n_transient += 1
    parent = np.arange(n_cells)
    for row in range(n_cells):
        if rank[row] < 0:
            continue
        row_bin = (rank[row] * n_bins) // n_transient
        for offset in range(indptr[row], indptr[row + 1]):
            col = indices[offset]
            if rank[col] < 0 or (rank[col] * n_bins) // n_transient != row_bin:
                continue
            first = row
            while parent[first] != first:
                parent[first] = parent[parent[first]]
                first = parent[first]
            second = col
            while parent[second] != second:
                parent[second] = parent[parent[second]]
                second = parent[second]
            if first < second:
                parent[second] = first
            elif second < first:
                parent[first] = second
    labels = np.full(n_cells, -1, dtype=np.int64)
    count = 0
    for row in range(n_cells):
        if rank[row] < 0:
            continue
        root = row
        while parent[root] != root:
            root = parent[root]
        if labels[root] < 0:
            labels[root] = count
            count += 1
        labels[row] = labels[root]
    return labels, count


def _fate_preconditioner(
    transition: csr_matrix,
    pseudotime: np.ndarray,
    absorbing: np.ndarray,
    operator: LinearOperator,
) -> LinearOperator:
    """Build a two-level preconditioner for the fate Dirichlet system.

    Symmetric Gauss-Seidel sweeps in pseudotime order resolve local coupling.
    A coarse correction over pseudotime bins, split into connected pieces,
    resolves the slow variation along long trajectories that restarted GMRES
    otherwise needs many iterations for. Besides four integer arrays per
    cell and two work vectors per application, it holds the sparse factor of
    a coarse system with at most ``_MAX_COARSE_AGGREGATES`` rows, unless the
    transient cells alone form more connected pieces.
    """
    n_cells = transition.shape[0]
    transient = np.flatnonzero(~absorbing)
    order = transient[np.argsort(-pseudotime[transient], kind="stable")]
    rank = np.full(n_cells, -1, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    n_bins = max(1, int(round(math.sqrt(order.size))))
    while True:
        labels, n_aggregates = _pseudotime_bin_aggregates(
            transition.indptr,
            transition.indices,
            rank,
            n_bins,
        )
        if n_aggregates <= _MAX_COARSE_AGGREGATES or n_bins == 1:
            break
        n_bins = max(1, n_bins // 2)
    aggregates = labels[transient]
    del labels
    restriction = csr_matrix(
        (np.ones(transient.size), (transient, aggregates)),
        shape=(n_cells, n_aggregates),
    )
    coarse_matrix = (
        diags(np.bincount(aggregates, minlength=n_aggregates).astype(np.float64))
        - restriction.T @ (transition @ restriction)
    ).tocsc()
    del restriction
    try:
        coarse_factor: Any = splu(coarse_matrix)
    except RuntimeError:
        # A numerically singular coarse system only weakens the
        # preconditioner, so the smoothing sweeps are used alone.
        coarse_factor = None
    del coarse_matrix

    def apply(residual: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=np.float64).ravel()
        correction = np.empty(n_cells, dtype=np.float64)
        _ordered_sweep(
            transition.data,
            transition.indices,
            transition.indptr,
            order,
            rank,
            residual,
            correction,
            True,
        )
        if coarse_factor is not None:
            remainder = residual - operator.matvec(correction)
            coarse = coarse_factor.solve(
                np.bincount(
                    aggregates,
                    weights=remainder[transient],
                    minlength=n_aggregates,
                )
            )
            if np.isfinite(coarse).all():
                correction[transient] += coarse[aggregates]
        remainder = residual - operator.matvec(correction)
        smoothed = np.empty(n_cells, dtype=np.float64)
        _ordered_sweep(
            transition.data,
            transition.indices,
            transition.indptr,
            order,
            rank,
            remainder,
            smoothed,
            False,
        )
        correction += smoothed
        return correction

    return LinearOperator(
        shape=(n_cells, n_cells),
        matvec=apply,
        dtype=np.dtype(np.float64),
    )


def fate_solver_bytes(n_cells: int, n_edges: int, n_sinks: int) -> int:
    """Estimate the working memory of :func:`compute_fate_probabilities`.

    Counts the biased transition matrix and its product with the coarse
    restriction, and per cell the GMRES Krylov basis, the solver's float64
    work vectors, the preconditioner's four int64 arrays, and the float32
    probability output. The coarse factor is bounded separately by
    ``_MAX_COARSE_AGGREGATES``.
    """
    sparse = 2 * int(n_edges) * (8 + 4)
    krylov = (_GMRES_RESTART + 2) * 8
    work = 8 * 8
    preconditioner = 4 * 8
    output = int(n_sinks) * 4
    return sparse + int(n_cells) * (krylov + work + preconditioner + output)


def _residual_limit(solver_tol: float, n_sinks: int) -> float:
    """Return the Bellman residual limit that validated probabilities meet."""
    validation_scale = 10.0 * solver_tol * max(1, n_sinks - 1)
    float32_tolerance = 5.0 * float(np.finfo(np.float32).eps)
    return max(float32_tolerance, min(1e-3, validation_scale))


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
    pseudotime: np.ndarray,
) -> np.ndarray:
    n_cells = transition.shape[0]
    probabilities = np.zeros((n_cells, n_sinks), dtype=np.float32)
    last_probability = np.ones(n_cells, dtype=np.float64)
    absorbing = sink_groups >= 0
    residual_limit = _residual_limit(solver_tol, n_sinks)
    # The last column is one minus the others, so its residual can add up the
    # residuals of every solved column. Each solve therefore stops at the
    # smaller of solver_tol and half of its share of the limit, measured as
    # the largest residual over all cells, so sink size cannot loosen it.
    target = min(solver_tol, 0.5 * residual_limit / max(1, n_sinks - 1))

    operator = _dirichlet_operator(transition, absorbing)
    preconditioner = _fate_preconditioner(transition, pseudotime, absorbing, operator)
    for group in range(n_sinks - 1):
        boundary = np.asarray(sink_groups == group, dtype=np.float64)
        solution = boundary
        iterations = 0

        def count_iteration(_residual: float) -> None:
            nonlocal iterations
            iterations += 1

        while True:
            completed = iterations
            # Run one restart cycle at a time. GMRES reports success only when
            # the two-norm residual, which bounds the maximum residual, meets
            # the target. Otherwise the true maximum residual decides.
            solution, info = gmres(
                operator,
                boundary,
                x0=solution,
                rtol=0.0,
                atol=target,
                restart=_GMRES_RESTART,
                maxiter=min(_GMRES_RESTART, max_iterations - iterations),
                M=preconditioner,
                callback=count_iteration,
                # This mode makes maxiter count inner iterations, not restart cycles.
                callback_type="legacy",
            )
            iteration_unit = "iteration" if iterations == 1 else "iterations"
            if info < 0:
                raise RuntimeError(
                    f"Fate probability solve for sink index {group} broke down "
                    f"after {iterations} {iteration_unit}"
                )
            if not np.isfinite(solution).all():
                raise RuntimeError(
                    f"Fate probability solve for sink index {group} produced "
                    "non-finite values"
                )
            if info == 0:
                break
            residual = float(np.max(np.abs(boundary - operator.matvec(solution))))
            if residual <= target:
                break
            if iterations >= max_iterations or iterations == completed:
                raise RuntimeError(
                    f"Fate probability solve for sink index {group} did not "
                    f"converge after {iterations} {iteration_unit} (maximum "
                    f"residual {residual:.3e}, target {target:.3e})"
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

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Fate probability calculation produced non-finite values")
    minimum = float(probabilities.min())
    maximum = float(probabilities.max())
    if minimum < -residual_limit or maximum > 1.0 + residual_limit:
        raise RuntimeError(
            "Fate probabilities exceeded numerical bounds "
            f"(minimum={minimum:.3e}, maximum={maximum:.3e})"
        )
    row_sums = probabilities.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=residual_limit):
        deviation = float(np.max(np.abs(row_sums - 1.0)))
        raise RuntimeError(
            f"Fate probabilities do not sum to one (maximum deviation={deviation:.3e})"
        )

    np.clip(probabilities, 0.0, 1.0, out=probabilities)
    row_sums = probabilities.sum(axis=1, dtype=np.float64)
    probabilities /= row_sums[:, None]
    probabilities[absorbing] = 0.0
    probabilities[absorbing, sink_groups[absorbing]] = 1.0

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
    """Compute grouped absorption probabilities on a pseudotime-biased graph.

    Each sink column except the last solves a Dirichlet system with
    preconditioned, restarted GMRES. ``solver_tol`` bounds the largest
    absolute Bellman residual of a solved column over all cells, so its meaning
    does not depend on sink size. ``max_iterations`` counts GMRES inner
    iterations per solved column. The finished probabilities are checked
    independently against a residual limit of
    ``10 * solver_tol * (len(sinks) - 1)``, bounded below by float32 precision
    and above by ``1e-3``.
    """
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
        retained_pseudotime,
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
