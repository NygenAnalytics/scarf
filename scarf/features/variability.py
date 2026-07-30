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

_ADAPTIVE_MIN_BIN_SIZE = 25
_ADAPTIVE_ANCHOR_QUANTILE = 0.1

# Case-insensitive via uppercasing in select_highly_variable_features / MetaData.grep.
DEFAULT_HVG_BLACKLIST = (
    "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST|"
    "^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$"
)
HVG_UBIQUITOUS_SLACK = 20


def _fit_lowess_adaptive(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int,
    lowess_frac: float,
) -> np.ndarray:
    from statsmodels.nonparametric.smoothers_lowess import lowess

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

    log_means = np.log(means[valid])
    log_variances = np.log(variances[valid])
    order = np.argsort(log_means, kind="stable")
    sorted_means = log_means[order]
    sorted_variances = log_variances[order]

    bin_slices: list[slice] = []
    bins_left = max(1, min(n_bins, len(order) // _ADAPTIVE_MIN_BIN_SIZE))
    start = 0
    while start < len(order):
        remaining = len(order) - start
        if bins_left == 1:
            end = len(order)
        else:
            target_size = (remaining + bins_left - 1) // bins_left
            end = start + target_size
        while end < len(order) and sorted_means[end] == sorted_means[end - 1]:
            end += 1
        bins_after = min(
            bins_left - 1,
            (len(order) - end) // _ADAPTIVE_MIN_BIN_SIZE,
        )
        if bins_after == 0:
            end = len(order)
        bin_slices.append(slice(start, end))
        start = end
        bins_left = bins_after

    anchor_means = np.fromiter(
        (np.median(sorted_means[indices]) for indices in bin_slices),
        dtype=float,
        count=len(bin_slices),
    )
    anchor_variances = np.fromiter(
        (
            np.quantile(
                sorted_variances[indices],
                _ADAPTIVE_ANCHOR_QUANTILE,
            )
            for indices in bin_slices
        ),
        dtype=float,
        count=len(bin_slices),
    )

    if len(anchor_means) == 1:
        correction = np.full(log_means.shape, anchor_variances[0], dtype=float)
    else:
        fitted = np.asarray(
            lowess(
                anchor_variances,
                anchor_means,
                return_sorted=False,
                frac=lowess_frac,
                it=100,
            ),
            dtype=float,
        )
        if fitted.shape != anchor_means.shape or not np.all(np.isfinite(fitted)):
            raise ValueError("LOWESS returned invalid adaptive trend values")
        correction = np.interp(log_means, anchor_means, fitted)

    corrected[valid] = np.exp(log_variances - correction)
    return corrected


def fit_lowess(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int,
    lowess_frac: float,
    *,
    bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
) -> np.ndarray:
    """Fit a LOWESS curve and return corrected variance estimates."""
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
        if top_n >= n_valid_features:
            logger.warning(
                f"WARNING: Number of valid features are less then value "
                f"of parameter `top_n`: {top_n}. Resetting `top_n` to "
                f"{n_valid_features - 1}"
            )
            top_n = n_valid_features - 1
        min_var = (
            pd.Series(corrected_variance)[candidates]
            .sort_values(ascending=False)
            .values[top_n]
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
