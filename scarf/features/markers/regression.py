import math
from typing import cast

import numpy as np
from numba import njit, prange
from scipy.stats import t as student_t

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
    """Calculate Pearson r per feature and return per-feature status codes.

    ``x_centered`` is the centered regressor and ``ssxm`` its mean square.
    A feature whose values span no more than ``eps`` times their largest
    magnitude is constant and untested; a power-of-two scale of the feature
    does not change the outcome while its values stay normal floats. Each
    tested feature's values are scaled by a power of two below one in
    magnitude before they are summed: the scale is exact and r does not
    depend on it, and it keeps the centered sums finite.
    """
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
        for c in range(n_cells):
            val = v[c]
            if not np.isfinite(val):
                finite = False
                break
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
        magnitude = max(-vmin, vmax)
        if nz < min_cells or (vmax - vmin) <= eps * magnitude:
            r_out[g] = 0.0
            status[g] = _REG_SENTINEL
            continue
        # Values below 2**-1024 need a scale above 2**1023, the largest power
        # of two in float64, so the scale is applied in two steps. Both steps
        # then enlarge the values by powers of two, which rounds nothing;
        # for larger values the second step multiplies by one.
        exponent = -math.frexp(magnitude)[1]
        scale = math.ldexp(1.0, min(exponent, 1023))
        scale_rest = math.ldexp(1.0, max(exponent - 1023, 0))
        y_sum = 0.0
        for c in range(n_cells):
            y_sum += v[c] * scale * scale_rest
        y_mean = y_sum * inv_n
        ssym = 0.0
        ssxym = 0.0
        for c in range(n_cells):
            yd = v[c] * scale * scale_rest - y_mean
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
            # The constant rule of the kernel: a span within eps of the
            # values' magnitude is no variation.
            if (v > 0).sum() >= min_cells and np.ptp(v) > eps * np.abs(v).max():
                # Two distinct points correlate exactly along their slope.
                r_vals[g] = float(
                    np.sign(regressor[1] - regressor[0]) * np.sign(v[1] - v[0])
                )
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
