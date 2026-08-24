from typing import cast

import numpy as np
from numba import njit, prange
from scipy.stats import linregress, t as student_t

_LINREGRESS_TINY = 1.0e-20
_REG_OK = 0
_REG_SENTINEL = 1
_REG_NONFINITE = 2

__all__ = [
    "_REG_NONFINITE",
    "_REG_OK",
    "_REG_SENTINEL",
    "_regression_batch_results",
    "_regression_p_values",
    "_regression_r_batch",
]


@njit(parallel=True, cache=True)
def _regression_r_batch(
    data: np.ndarray,
    x_centered: np.ndarray,
    ssxm: float,
    min_cells: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Pearson r per feature and return per-feature status codes."""
    n_cells = data.shape[0]
    n_genes = data.shape[1]
    r_out = np.empty(n_genes, dtype=np.float64)
    status = np.empty(n_genes, dtype=np.int8)
    inv_n = 1.0 / n_cells
    for g in prange(n_genes):
        v = data[:, g]
        finite = True
        nz = 0
        vmin = v[0]
        vmax = v[0]
        y_sum = 0.0
        for c in range(n_cells):
            val = v[c]
            if not np.isfinite(val):
                finite = False
                break
            y_sum += val
            if val > 0.0:
                nz += 1
            if val < vmin:
                vmin = val
            if val > vmax:
                vmax = val
        if not finite:
            r_out[g] = 0.0
            status[g] = _REG_NONFINITE
            continue
        if nz < min_cells or (vmax - vmin) <= eps:
            r_out[g] = 0.0
            status[g] = _REG_SENTINEL
            continue
        y_mean = y_sum * inv_n
        ssym = 0.0
        ssxym = 0.0
        for c in range(n_cells):
            yd = v[c] - y_mean
            ssym += yd * yd
            ssxym += x_centered[c] * yd
        ssym *= inv_n
        ssxym *= inv_n
        if ssxm == 0.0 or ssym == 0.0:
            r_out[g] = 0.0
            status[g] = _REG_SENTINEL
            continue
        r = ssxym / np.sqrt(ssxm * ssym)
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
        r_out[g] = r
        status[g] = _REG_OK
    return r_out, status


def _regression_p_values(r: np.ndarray, n_cells: int) -> np.ndarray:
    """Calculate two-sided Student-t p-values matching `linregress`."""
    df = float(n_cells - 2)
    denom = (1.0 - r + _LINREGRESS_TINY) * (1.0 + r + _LINREGRESS_TINY)
    t_stat = r * np.sqrt(df / denom)
    return cast(np.ndarray, 2.0 * student_t.sf(np.abs(t_stat), df))


def _regression_batch_results(
    data: np.ndarray,
    x_centered: np.ndarray,
    ssxm: float,
    regressor: np.ndarray,
    min_cells: int,
    feature_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate r, p, and status for one feature batch."""
    n_cells = data.shape[0]
    eps = float(np.finfo(float).eps)
    if n_cells == 2:
        r_vals = np.empty(data.shape[1], dtype=np.float64)
        p_vals = np.full(data.shape[1], np.nan, dtype=np.float64)
        status = np.full(data.shape[1], _REG_SENTINEL, dtype=np.int8)
        for g in range(data.shape[1]):
            v = data[:, g]
            if not np.isfinite(v).all():
                raise ValueError(
                    f"Feature {feature_labels[g]!r} contains non-finite "
                    "normalized values"
                )
            if (v > 0).sum() >= min_cells and np.ptp(v) > eps:
                lin_obj = linregress(regressor, v)
                r_vals[g] = float(lin_obj.rvalue)
            else:
                r_vals[g] = 0.0
        return r_vals, p_vals, status

    r_vals, status = _regression_r_batch(data, x_centered, ssxm, int(min_cells), eps)
    bad = np.flatnonzero(status == _REG_NONFINITE)
    if bad.size:
        raise ValueError(
            f"Feature {feature_labels[bad[0]]!r} contains non-finite normalized values"
        )
    p_vals = np.full(data.shape[1], np.nan, dtype=np.float64)
    ok = status == _REG_OK
    if np.any(ok):
        p_vals[ok] = _regression_p_values(r_vals[ok], n_cells)
    r_vals = np.where(ok, r_vals, 0.0)
    return r_vals, p_vals, status
