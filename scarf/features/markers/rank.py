import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import norm

_MARKER_SORT_BY = ("score", "p_value")
_MARKER_SORT_ASCENDING = (False, True)

__all__ = [
    "_batch_stats",
    "_batch_stats_gene_major",
    "_marker_stats_batch",
    "_marker_stats_gene_major",
    "gene_major_rank_scratch_bytes",
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
        tied_counts = counts[counts > 1].astype(np.float64)
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
        delta = u1 - mu_u
        z = np.zeros_like(delta, dtype=float)
        np.divide(
            delta - 0.5 * np.sign(delta),
            sigma_u,
            out=z,
            where=sigma_u > 0,
        )
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
    out = np.zeros((n_genes, n_groups, 8))
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
            t_float = float(t)
            for k in range(i, j + 1):
                ar[order[k]] = avg
                dr[order[k]] = rank
            if t > 1:
                tie_sum += t_float * t_float * t_float - t_float
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
            delta = u1 - mu
            if var > 0.0:
                z = (delta - 0.5 * np.sign(delta)) / np.sqrt(var)
            else:
                z = 0.0
            if n1 > 0.0 and n2 > 0.0:
                auc = u1 / (n1 * n2)
            else:
                auc = np.nan
            out[g, x, 0] = r
            out[g, x, 1] = m
            out[g, x, 2] = m_o
            out[g, x, 3] = e
            out[g, x, 4] = e_o
            out[g, x, 5] = fc
            out[g, x, 6] = z
            out[g, x, 7] = auc
    return out


@njit(parallel=True, cache=True)
def _marker_stats_gene_major(
    raw: np.ndarray,
    scalar: np.ndarray,
    size_factor: np.float32,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: np.float32,
    destination_rows: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute marker statistics from non-negative feature-major raw counts."""
    n_genes = raw.shape[0]
    n_cells = raw.shape[1]
    n_groups = group_counts.shape[0]
    for g in prange(n_genes):
        nz_values = np.empty(n_cells, dtype=np.float32)
        nz_cells = np.empty(n_cells, dtype=np.int64)
        zero_g = np.zeros(n_groups)
        sum_g = np.zeros(n_groups)
        nz_g = np.zeros(n_groups)
        rank_g = np.zeros(n_groups)
        drank_g = np.zeros(n_groups)
        n_nz = 0
        for c in range(n_cells):
            grp = int_indices[c]
            value = (size_factor * np.float32(raw[g, c])) / scalar[c]
            if log_transform:
                value = np.log1p(value)
            if value > 0.0:
                nz_values[n_nz] = value
                nz_cells[n_nz] = c
                n_nz += 1
                sum_g[grp] += value
                nz_g[grp] += 1.0
            else:
                zero_g[grp] += 1.0

        n_zero = n_cells - n_nz
        tie_sum = 0.0
        if n_zero > 0:
            zero_rank = (n_zero + 1.0) / 2.0
            zero_t = float(n_zero)
            if n_zero > 1:
                tie_sum += zero_t * zero_t * zero_t - zero_t
            for x in range(n_groups):
                rank_g[x] = zero_g[x] * zero_rank
                drank_g[x] = zero_g[x]

        order = np.argsort(nz_values[:n_nz])
        i = 0
        dense_rank = 1.0 if n_zero > 0 else 0.0
        while i < n_nz:
            j = i
            value = nz_values[order[i]]
            while j + 1 < n_nz and nz_values[order[j + 1]] == value:
                j += 1
            dense_rank += 1.0
            average_rank = n_zero + (i + j + 2.0) / 2.0
            tied = j - i + 1
            tied_float = float(tied)
            if tied > 1:
                tie_sum += tied_float * tied_float * tied_float - tied_float
            for k in range(i, j + 1):
                cell = nz_cells[order[k]]
                grp = int_indices[cell]
                rank_g[grp] += average_rank
                drank_g[grp] += dense_rank
            i = j + 1

        total_sum = 0.0
        total_nz = 0.0
        for x in range(n_groups):
            total_sum += sum_g[x]
            total_nz += nz_g[x]
        rank_total = 0.0
        rank_values = np.empty(n_groups)
        for x in range(n_groups):
            count = group_counts[x]
            rank_values[x] = drank_g[x] / count if count > 0 else 0.0
            rank_total += rank_values[x]
        tie_correction = tie_sum / (n_total * (n_total - 1.0)) if n_total > 1 else 0.0

        row = destination_rows[g]
        for x in range(n_groups):
            count = group_counts[x]
            rest = n_total - count
            mean = sum_g[x] / count if count > 0 else 0.0
            mean_rest = (total_sum - sum_g[x]) / rest if rest > 0 else 0.0
            fraction = nz_g[x] / count if count > 0 else 0.0
            fraction_rest = (total_nz - nz_g[x]) / rest if rest > 0 else 0.0
            if mean_rest == 0.0:
                fold_change = 0.0 if mean == 0.0 else 100.1
            else:
                fold_change = mean / mean_rest
            score = rank_values[x] / rank_total if rank_total > 0 else 0.0
            n1 = count
            n2 = rest
            rank_sum = rank_g[x]
            u1 = rank_sum - (n1 * (n1 + 1.0)) / 2.0
            mu = (n1 * n2) / 2.0
            variance = (n1 * n2 / 12.0) * ((n_total + 1.0) - tie_correction)
            delta = u1 - mu
            z = (
                (delta - 0.5 * np.sign(delta)) / np.sqrt(variance)
                if variance > 0.0
                else 0.0
            )
            if n1 > 0.0 and n2 > 0.0:
                auc = u1 / (n1 * n2)
            else:
                auc = np.nan
            out[row, x, 0] = score
            out[row, x, 1] = mean
            out[row, x, 2] = mean_rest
            out[row, x, 3] = fraction
            out[row, x, 4] = fraction_rest
            out[row, x, 5] = fold_change
            out[row, x, 6] = z
            out[row, x, 7] = auc


def gene_major_rank_scratch_bytes(
    *,
    n_cells: int,
    n_groups: int,
    n_threads: int,
) -> int:
    """Return the worst-case scratch owned by active gene workers."""
    cells = max(0, int(n_cells))
    groups = max(0, int(n_groups))
    threads = max(1, int(n_threads))
    per_thread = (
        cells * (np.dtype(np.float32).itemsize + 2 * np.dtype(np.int64).itemsize)
        + groups * 6 * np.dtype(np.float64).itemsize
    )
    return threads * per_thread


def _batch_stats_gene_major(
    raw: np.ndarray,
    scalar: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: int,
) -> np.ndarray:
    """Run the feature-major kernel and convert z statistics to p-values."""
    n_genes = int(raw.shape[0])
    out = np.zeros((n_genes, len(group_counts), 8), dtype=np.float64)
    _marker_stats_gene_major(
        np.ascontiguousarray(raw),
        np.asarray(scalar, dtype=np.float32),
        np.float32(size_factor),
        bool(log_transform),
        np.asarray(int_indices, dtype=np.int64),
        np.asarray(group_counts, dtype=np.float32),
        np.float32(n_total),
        np.arange(n_genes, dtype=np.int64),
        out,
    )
    out[:, :, 6] = 2.0 * norm.sf(np.abs(out[:, :, 6]))
    return out


def _batch_stats(
    data: np.ndarray,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: int,
    feature_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Run the marker kernel and convert z statistics to p-values."""
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("Marker data must be a two-dimensional array")
    labels = (
        np.arange(values.shape[1])
        if feature_labels is None
        else np.asarray(feature_labels)
    )
    if labels.shape != (values.shape[1],):
        raise ValueError("Feature labels must match the marker data columns")
    for column in range(values.shape[1]):
        if not np.isfinite(values[:, column]).all():
            raise ValueError(
                f"Feature {labels[column]!r} contains non-finite normalized values"
            )
    kernel_dtype = np.float32 if values.dtype == np.float32 else np.float64
    out = _marker_stats_batch(
        np.ascontiguousarray(values, dtype=kernel_dtype),
        int_indices,
        group_counts.astype(np.float32),
        np.float32(n_total),
    )
    out[:, :, 6] = 2.0 * norm.sf(np.abs(out[:, :, 6]))
    return np.asarray(out)
