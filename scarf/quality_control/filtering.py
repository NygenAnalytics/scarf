import numpy as np
from scipy.stats import norm

__all__ = ["gaussian_quantile_bounds"]


def gaussian_quantile_bounds(
    values: np.ndarray,
    min_p: float = 0.01,
    max_p: float = 0.99,
) -> tuple[float, float]:
    dist = norm(np.median(values), np.std(values))
    return float(dist.ppf(min_p)), float(dist.ppf(max_p))
