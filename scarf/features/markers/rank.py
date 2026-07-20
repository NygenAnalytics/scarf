import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import norm

_MARKER_SORT_BY = ("score", "p_value")
_MARKER_SORT_ASCENDING = (False, True)

__all__ = [
    "_batch_stats",
    "_marker_stats_batch",
    "mannwhitneyu_from_ranks",
    "sort_marker_results",
]


def sort_marker_results(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "feature_index" not in frame.columns:
        frame["feature_index"] = frame.index
    sort_by = list(_MARKER_SORT_BY)
    ascending = list(_MARKER_SORT_ASCENDING)
    if "feature_name" in frame.columns:
        sort_by.append("feature_name")
        ascending.append(True)
    else:
        sort_by.append("feature_index")
        ascending.append(True)
    return frame.sort_values(by=sort_by, ascending=ascending)


def mannwhitneyu_from_ranks(
    ranked_df: pd.DataFrame,
    groups: np.ndarray,
    group_set: np.ndarray,
) -> pd.DataFrame:
    """Calculate two-sided Mann-Whitney U p-values from precomputed ranks."""
    n_total = len(groups)
    rank_sums = ranked_df.groupby(groups).sum().reindex(group_set)
    group_counts = pd.Series(groups).value_counts().reindex(group_set).values
    tie_corrections = np.zeros(ranked_df.shape[1])

    for col_idx in range(ranked_df.shape[1]):
        ranks = ranked_df.iloc[:, col_idx].values
        _, counts = np.unique(ranks, return_counts=True)
        tied_counts = counts[counts > 1]
        if len(tied_counts) > 0:
            tie_sum = np.sum(tied_counts**3 - tied_counts)
            tie_corrections[col_idx] = tie_sum / (n_total * (n_total - 1))

    pvals = {}
    for idx, cluster in enumerate(group_set):
        n1 = group_counts[idx]
        n2 = n_total - n1
        r1 = rank_sums.iloc[idx].values
        u1 = r1 - (n1 * (n1 + 1)) / 2
        mu_u = (n1 * n2) / 2
        sigma_u = np.sqrt((n1 * n2 / 12) * ((n_total + 1) - tie_corrections))
        z = (u1 - mu_u - 0.5) / sigma_u
        pvals[cluster] = 2 * norm.sf(np.abs(z))

    return pd.DataFrame(pvals, index=ranked_df.columns).T


@njit(parallel=True, cache=True)
def _marker_stats_batch(
    data: np.ndarray,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
) -> np.ndarray:
    """Compute per-feature, per-group marker statistics for one batch."""
    n_cells = data.shape[0]
    n_genes = data.shape[1]
    n_groups = group_counts.shape[0]
    out = np.zeros((n_genes, n_groups, 7))
    for g in prange(n_genes):
        v = data[:, g]
        order = np.argsort(v)
        ar = np.empty(n_cells)
        dr = np.empty(n_cells)
        tie_sum = 0.0
        i = 0
        rank = 0.0
        while i < n_cells:
            j = i
            vi = v[order[i]]
            while j + 1 < n_cells and v[order[j + 1]] == vi:
                j += 1
            avg = (i + j + 2) / 2.0
            rank += 1.0
            t = j - i + 1
            for k in range(i, j + 1):
                ar[order[k]] = avg
                dr[order[k]] = rank
            if t > 1:
                tie_sum += t * t * t - t
            i = j + 1
        sum_g = np.zeros(n_groups)
        nz_g = np.zeros(n_groups)
        rank_g = np.zeros(n_groups)
        drank_g = np.zeros(n_groups)
        for c in range(n_cells):
            grp = int_indices[c]
            val = v[c]
            sum_g[grp] += val
            if val > 0:
                nz_g[grp] += 1.0
            rank_g[grp] += ar[c]
            drank_g[grp] += dr[c]
        total_sum = 0.0
        total_nz = 0.0
        for x in range(n_groups):
            total_sum += sum_g[x]
            total_nz += nz_g[x]
        r_sum = 0.0
        r_vals = np.empty(n_groups)
        for x in range(n_groups):
            cnt = group_counts[x]
            r_vals[x] = drank_g[x] / cnt if cnt > 0 else 0.0
            r_sum += r_vals[x]
        if n_total > 1:
            tie_corr = tie_sum / (n_total * (n_total - 1))
        else:
            tie_corr = 0.0
        for x in range(n_groups):
            cnt = group_counts[x]
            rest = n_total - cnt
            m = sum_g[x] / cnt if cnt > 0 else 0.0
            m_o = (total_sum - sum_g[x]) / rest if rest > 0 else 0.0
            e = nz_g[x] / cnt if cnt > 0 else 0.0
            e_o = (total_nz - nz_g[x]) / rest if rest > 0 else 0.0
            if m_o == 0.0:
                fc = 0.0 if m == 0.0 else 100.1
            else:
                fc = m / m_o
            r = r_vals[x] / r_sum if r_sum > 0 else 0.0
            n1 = cnt
            n2 = rest
            r1 = rank_g[x]
            u1 = r1 - (n1 * (n1 + 1.0)) / 2.0
            mu = (n1 * n2) / 2.0
            var = (n1 * n2 / 12.0) * ((n_total + 1.0) - tie_corr)
            if var > 0.0:
                z = (u1 - mu - 0.5) / np.sqrt(var)
            else:
                z = 0.0
            out[g, x, 0] = r
            out[g, x, 1] = m
            out[g, x, 2] = m_o
            out[g, x, 3] = e
            out[g, x, 4] = e_o
            out[g, x, 5] = fc
            out[g, x, 6] = z
    return out


def _batch_stats(
    data: np.ndarray,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: int,
) -> np.ndarray:
    """Run the marker kernel and convert z statistics to p-values."""
    out = _marker_stats_batch(
        np.ascontiguousarray(data, dtype=np.float32),
        int_indices,
        group_counts.astype(np.float32),
        np.float32(n_total),
    )
    out[:, :, 6] = 2.0 * norm.sf(np.abs(out[:, :, 6]))
    return np.asarray(out)
