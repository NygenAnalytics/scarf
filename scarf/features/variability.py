import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils.logging import logger

__all__ = [
    "DEFAULT_HVG_BLACKLIST",
    "HVG_UBIQUITOUS_SLACK",
    "fit_lowess",
    "select_highly_variable_features",
]

_ADAPTIVE_MIN_SUPPORT = 50
_ADAPTIVE_QUANTILE = 0.25

# Case-insensitive via uppercasing in select_highly_variable_features / MetaData.grep.
DEFAULT_HVG_BLACKLIST = (
    "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST|"
    "^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$"
)
HVG_UBIQUITOUS_SLACK = 20


def _fit_local_quantile(
    log_means: np.ndarray,
    log_variances: np.ndarray,
    noise_residual: np.ndarray,
    center: float,
    bandwidth: float,
    minimum_bandwidth: float,
) -> float:
    from scipy.optimize import minimize

    reference: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(2):
        keep = np.abs(log_means - center) < bandwidth
        relative = (log_means[keep] - center) / bandwidth
        weight = 0.5 + 0.5 * (1 - np.abs(relative) ** 3) ** 3
        if reference is not None:
            weight *= np.interp(log_means[keep], reference[0], reference[1])
        weight /= weight.sum()
        degree = min(2, max(1, len(np.unique(relative)) - 3))
        design = np.stack([relative**power for power in range(degree + 1)], axis=1)

        # Limit the influence of isolated genes at the ends of a fitting window.
        for _ in range(3):
            inverse = np.linalg.pinv(design.T @ (weight[:, None] * design))
            leverage = weight * np.sum((design @ inverse) * design, axis=1)
            limit = 1.25 * (degree + 1) * (weight @ weight)
            weight *= np.minimum(1, limit / np.maximum(leverage, 1e-15))
            weight /= weight.sum()

        origin = np.median(log_variances[keep])
        observed = log_variances[keep] - origin
        initial = np.linalg.lstsq(
            design * np.sqrt(weight[:, None]), observed * np.sqrt(weight), rcond=None
        )[0]
        local_noise = noise_residual[keep]
        noise = np.median(local_noise[np.isfinite(local_noise)]) if degree == 2 else 0.0
        penalty = noise * np.sqrt(weight @ weight)

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            residual = observed - design @ coefficients
            absolute = np.sqrt(residual**2 + 1e-8)
            value = weight @ (0.5 * absolute + (_ADAPTIVE_QUANTILE - 0.5) * residual)
            gradient = -design.T @ (
                weight * (0.5 * residual / absolute + _ADAPTIVE_QUANTILE - 0.5)
            )
            if degree == 2:
                value += penalty * coefficients[2] ** 2
                gradient[2] += 2 * penalty * coefficients[2]
            return float(value), gradient

        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 300, "maxls": 50},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError(f"Adaptive variance trend fit failed: {result.message}")

        # Continue linearly outside the interior support instead of extrapolating
        # poorly constrained curvature across an isolated endpoint.
        edge = np.clip(0.0, *np.quantile(relative, [0.1, 0.9]))
        prediction = result.x[0] + origin
        if degree == 2:
            prediction -= result.x[2] * edge**2
        if reference is not None or minimum_bandwidth >= bandwidth or degree < 2:
            break

        residual = observed - design @ result.x
        model_noise = np.median(np.abs(residual - np.median(residual)))
        if model_noise <= max(2 * noise, 1e-4):
            break

        # Narrow only when smooth curvature exceeds local scatter. Carry an
        # outlier guard from the wider window into the smaller fit.
        unique, groups = np.unique(log_means[keep], return_inverse=True)
        values = np.array(
            [np.median(observed[groups == index]) for index in range(len(unique))]
        )
        slopes = np.diff(values) / np.diff(unique)
        curvature = np.diff(slopes) / (unique[2:] - unique[:-2])
        curvature = np.abs(np.r_[curvature[0], curvature, curvature[-1]])
        scale = max(3 * np.median(curvature), 1e-8)
        robust_weight = np.minimum(1, (scale / np.maximum(curvature, 1e-15)) ** 2)
        reference = (log_means[keep], robust_weight[groups])
        bandwidth = minimum_bandwidth

    return float(prediction)


def _fit_lowess_adaptive(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int,
    lowess_frac: float,
) -> np.ndarray:
    means = np.asarray(a, dtype=float)
    variances = np.asarray(b, dtype=float)
    if means.ndim != 1 or variances.ndim != 1 or means.shape != variances.shape:
        raise ValueError("LOWESS inputs must be one-dimensional arrays of equal length")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins,
        (int, np.integer),
    ):
        raise TypeError("n_bins must be an integer")
    if n_bins < 1:
        raise ValueError("n_bins must be greater than 0")
    if isinstance(lowess_frac, (bool, np.bool_)) or not isinstance(
        lowess_frac,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("lowess_frac must be numeric")
    if not np.isfinite(lowess_frac) or not 0 <= lowess_frac <= 1:
        raise ValueError("lowess_frac must be between 0 and 1")

    corrected = np.zeros(means.shape, dtype=float)
    valid = np.isfinite(means) & np.isfinite(variances) & (means > 0) & (variances > 0)
    if not valid.any():
        return corrected
    if valid.sum() < 3:
        raise ValueError(
            "At least three genes with positive finite means and variances "
            "are needed to estimate an adaptive variance trend"
        )

    log_means = np.log(means[valid])
    log_variances = np.log(variances[valid])
    order = np.argsort(log_means, kind="stable")
    sorted_means = log_means[order]
    sorted_variances = log_variances[order]

    unique, starts, counts = np.unique(
        sorted_means, return_index=True, return_counts=True
    )
    if len(unique) == 1:
        correction = np.full(
            log_means.shape, np.quantile(log_variances, _ADAPTIVE_QUANTILE)
        )
    else:
        span = np.ptp(unique)
        rank = (starts + (counts - 1) / 2) / (len(order) - 1)
        coordinate = 0.5 * rank + 0.5 * (unique - unique[0]) / span
        coordinate = (coordinate - coordinate[0]) / np.ptp(coordinate)
        labels = np.repeat(
            np.minimum(np.floor(n_bins * coordinate).astype(int), n_bins - 1), counts
        )
        boundaries = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(order)]
        centers = [
            np.median(sorted_means[start:end])
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        # Fit both endpoints instead of clamping the sparse tail to an inner bin.
        anchors = np.unique(np.r_[unique[0], centers, unique[-1]])

        median_variance = sorted_variances[starts].copy()
        for index in np.flatnonzero(counts > 1):
            start = starts[index]
            median_variance[index] = np.median(
                sorted_variances[start : start + counts[index]]
            )
        neighbor_prediction = np.full(len(unique), np.nan)
        if len(unique) > 2:
            position = (unique[1:-1] - unique[:-2]) / (unique[2:] - unique[:-2])
            neighbor_prediction[1:-1] = median_variance[:-2] + position * (
                median_variance[2:] - median_variance[:-2]
            )
        noise_residual = np.abs(
            sorted_variances - np.repeat(neighbor_prediction, counts)
        )

        fitted = np.empty(len(anchors))
        support = min(_ADAPTIVE_MIN_SUPPORT, len(order))
        minimum_support = min(8, len(order))
        for anchor_index, center in enumerate(anchors):
            distances = np.abs(sorted_means - center)
            bandwidth = max(
                lowess_frac * span,
                np.partition(distances, support - 1)[support - 1] * 1.05,
            )
            if bandwidth == 0:
                bandwidth = distances[distances > 0].min() * 1.05
            minimum_bandwidth = max(
                lowess_frac * span,
                np.partition(distances, minimum_support - 1)[minimum_support - 1]
                * 1.05,
            )
            if minimum_bandwidth == 0:
                minimum_bandwidth = distances[distances > 0].min() * 1.05
            fitted[anchor_index] = _fit_local_quantile(
                sorted_means,
                sorted_variances,
                noise_residual,
                center,
                bandwidth,
                minimum_bandwidth,
            )
        correction = np.interp(log_means, anchors, fitted)

    with np.errstate(over="ignore", under="ignore"):
        corrected[valid] = np.exp(log_variances - correction)
    if not np.all(np.isfinite(corrected[valid]) & (corrected[valid] > 0)):
        raise ValueError(
            "Adaptive variance correction produced nonfinite or zero scores"
        )
    return corrected


def fit_lowess(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int,
    lowess_frac: float,
    *,
    bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
) -> np.ndarray:
    """Divide variance by an expression-dependent background.

    Adaptive fits use local quantile regression: ``n_bins`` controls evaluation
    density and ``lowess_frac`` is the window radius as a fraction of log-mean
    range. Minimum support is 50 genes, or 8 where smooth curvature warrants it.
    Smaller inputs use all available genes. The background is the lower quartile.
    Fixed fits use LOWESS over the minimum-variance gene in each equal-width bin.
    """
    if bin_strategy == "adaptive":
        return _fit_lowess_adaptive(a, b, n_bins, lowess_frac)
    if bin_strategy != "fixed":
        raise ValueError("bin_strategy must be either 'fixed' or 'adaptive'")

    from statsmodels.nonparametric.smoothers_lowess import lowess

    stats = pd.DataFrame({"a": a, "b": b}).apply(np.log)
    bin_edges = np.histogram(stats.a, bins=n_bins)[1]
    bin_edges[-1] += 0.1
    bin_idx: list[list[Any]] = []
    for index in range(n_bins):
        idx = pd.Series(
            (stats.a >= bin_edges[index]) & (stats.a < bin_edges[index + 1])
        )
        if sum(idx) > 0:
            bin_idx.append(list(idx[idx].index))
    bin_vals: list[list[float]] = []
    for idx in bin_idx:
        temp_stat = stats.reindex(idx)
        temp_gene = temp_stat.idxmin().b
        bin_vals.append([temp_stat.b[temp_gene], temp_stat.a[temp_gene]])
    bin_array = np.array(bin_vals).T
    bin_cor_fac = lowess(
        bin_array[0],
        bin_array[1],
        return_sorted=False,
        frac=lowess_frac,
        it=100,
    ).T
    fixed_var: dict[Any, float] = {}
    for correction, indices in zip(bin_cor_fac, bin_idx):
        for idx in indices:
            fixed_var[idx] = np.e ** (stats.b[idx] - correction)
    return np.array([fixed_var[index] for index in range(len(a))])


def _bounded(
    values: np.ndarray,
    lower: float,
    upper: float,
    *,
    keep_bounds: bool,
) -> np.ndarray:
    if keep_bounds:
        return (values >= lower) & (values <= upper)
    return (values > lower) & (values < upper)


def _linear_threshold(value: float, unbounded_value: float) -> float:
    if value == unbounded_value:
        return value
    return float(2**value)


def select_highly_variable_features(
    corrected_variance: np.ndarray,
    normalized_cell_counts: np.ndarray,
    mean_nonzero: np.ndarray,
    active_features: np.ndarray,
    feature_names: np.ndarray,
    *,
    min_cells: int,
    max_cells: int | float,
    top_n: int,
    min_var: float,
    max_var: float,
    min_mean: float,
    max_mean: float,
    blacklist: str,
    keep_bounds: bool,
) -> np.ndarray:
    """Select highly variable features from precomputed feature statistics."""
    corrected_variance = np.asarray(corrected_variance)
    normalized_cell_counts = np.asarray(normalized_cell_counts)
    mean_nonzero = np.asarray(mean_nonzero)
    active_features = np.asarray(active_features, dtype=bool)
    feature_names = np.asarray(feature_names)
    size = corrected_variance.shape[0]
    if any(
        values.shape != (size,)
        for values in (
            normalized_cell_counts,
            mean_nonzero,
            active_features,
            feature_names,
        )
    ):
        raise ValueError("HVG inputs must be one-dimensional arrays of equal length")

    min_var = _linear_threshold(min_var, -np.inf)
    max_var = _linear_threshold(max_var, np.inf)
    min_mean = _linear_threshold(min_mean, -np.inf)
    max_mean = _linear_threshold(max_mean, np.inf)

    if blacklist:
        pattern = re.compile(blacklist.upper())
        allowed = np.fromiter(
            (pattern.match(str(name).upper()) is None for name in feature_names),
            dtype=bool,
            count=size,
        )
    else:
        allowed = np.ones(size, dtype=bool)

    cell_count_candidates = normalized_cell_counts >= min_cells
    cell_count_candidates &= (
        normalized_cell_counts <= max_cells
        if keep_bounds
        else normalized_cell_counts < max_cells
    )
    candidates = (
        cell_count_candidates
        & _bounded(mean_nonzero, min_mean, max_mean, keep_bounds=keep_bounds)
        & active_features
        & allowed
    )
    if min_var == -np.inf:
        if top_n < 1:
            raise ValueError(
                "ERROR: Please provide a value greater than 0 for `top_n` parameter"
            )
        n_valid_features = int(candidates.sum())
        if n_valid_features == 0:
            raise ValueError(
                "No features passed HVG candidate filters "
                f"(min_cells={min_cells}, max_cells={max_cells}, "
                f"min_mean={min_mean}, max_mean={max_mean})."
            )
        if top_n > n_valid_features:
            logger.warning(
                f"WARNING: Number of valid features is less than value "
                f"of parameter `top_n`: {top_n}. Resetting `top_n` to "
                f"{n_valid_features}"
            )
            top_n = n_valid_features
        # Deterministic tie-break: higher variance first, then lower feature index.
        candidate_idx = np.flatnonzero(candidates)
        order = np.lexsort(
            (
                candidate_idx,
                -corrected_variance[candidate_idx],
            )
        )
        ranked = candidate_idx[order]
        selected = np.zeros(size, dtype=bool)
        selected[ranked[:top_n]] = True
        return np.asarray(
            selected
            & _bounded(
                corrected_variance,
                -np.inf,
                max_var,
                keep_bounds=keep_bounds,
            ),
            dtype=bool,
        )

    return np.asarray(
        candidates
        & _bounded(
            corrected_variance,
            min_var,
            max_var,
            keep_bounds=keep_bounds,
        ),
        dtype=bool,
    )
