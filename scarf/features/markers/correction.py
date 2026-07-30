"""Private multiple-testing helpers for marker statistics."""

import numpy as np
from numpy.typing import NDArray

__all__ = ["_bh_adjusted_pvalues"]


def _bh_adjusted_pvalues(p_values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return Benjamini-Hochberg adjusted p-values for finite entries."""
    from statsmodels.stats.multitest import multipletests

    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    if not np.any(mask):
        return adjusted
    _, corrected, _, _ = multipletests(values[mask], method="fdr_bh")
    adjusted[mask] = corrected
    return adjusted
