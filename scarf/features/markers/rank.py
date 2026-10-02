import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import ndtr

__all__ = [
    "_batch_stats",
    "_marker_stats_batch",
    "_marker_stats_gene_major",
    "batch_rank_scratch_bytes",
    "gene_major_rank_scratch_bytes",
    "mannwhitneyu_from_ranks",
    "sort_marker_results",
    "tie_sum",
]


def sort_marker_results(df: pd.DataFrame) -> pd.DataFrame:
    """Order markers by descending score, then p-value, then feature.

    A stable NumPy lexsort gives the same order as the equivalent pandas sort,
    including NaNs last, without factorizing every key column.
    """
    frame = df if "feature_index" in df.columns else df.assign(feature_index=df.index)
    tie = "feature_name" if "feature_name" in frame.columns else "feature_index"
    keys = frame[tie].to_numpy()
    if keys.dtype == object:
        keys = keys.astype(str)
    order = np.lexsort(
        (
            keys,
            frame["p_value"].to_numpy(dtype=np.float64),
            -frame["score"].to_numpy(dtype=np.float64),
        )
    )
    return frame.iloc[order]


def tie_sum(values: np.ndarray) -> float:
    """Return the rank tie term, the sum of ``t**3 - t`` over tied value groups.

    Group sizes are cubed in float64 because an int64 cube wraps once a group
    holds more than about 2.1 million equal values.
    """
    _, counts = np.unique(np.asarray(values), return_counts=True)
    tied = counts[counts > 1].astype(np.float64)
    return float(np.sum(tied**3 - tied))


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
        ties = tie_sum(ranked_df.iloc[:, col_idx].to_numpy())
        if ties > 0:
            tie_corrections[col_idx] = ties / (n_total * (n_total - 1))

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
        pvals[cluster] = 2 * ndtr(-np.abs(z))

    return pd.DataFrame(pvals, index=ranked_df.columns).T


@njit(cache=True, nogil=True)
def _write_group_statistics(
    out: np.ndarray,
    sum_g: np.ndarray,
    nz_g: np.ndarray,
    rank_g: np.ndarray,
    drank_g: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    tie_total: float,
) -> None:
    """Write one feature's groups-by-statistics row from its group sums.

    ``rank_g`` and ``drank_g`` hold each group's average-rank and dense-rank
    sums, and ``tie_total`` the feature's rank tie term. Column 6 holds the
    continuity- and tie-corrected Mann-Whitney z statistic.
    """
    n_groups = group_counts.shape[0]
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
    tie_correction = tie_total / (n_total * (n_total - 1.0)) if n_total > 1 else 0.0

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
        out[x, 0] = score
        out[x, 1] = mean
        out[x, 2] = mean_rest
        out[x, 3] = fraction
        out[x, 4] = fraction_rest
        out[x, 5] = fold_change
        out[x, 6] = z
        out[x, 7] = auc


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
        tie_total = 0.0
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
                tie_total += t_float * t_float * t_float - t_float
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
        _write_group_statistics(
            out[g],
            sum_g,
            nz_g,
            rank_g,
            drank_g,
            group_counts,
            n_total,
            tie_total,
        )
    return out


@njit(cache=True, nogil=True)
def _argsort_positive(
    values: np.ndarray,
    n: int,
    order: np.ndarray,
    scratch: np.ndarray,
    buckets: np.ndarray,
) -> None:
    """Stably argsort ``values[:n]`` into ``order[:n]``.

    Positive float32 values sort like their bit patterns read as unsigned
    integers, so three counting passes of 11, 11, and 10 bits replace a
    comparison sort.
    """
    keys = values.view(np.uint32)
    for index in range(n):
        order[index] = index
    source = order
    target = scratch
    shift = 0
    for bits in (11, 11, 10):
        mask = np.uint32((1 << bits) - 1)
        buckets[: mask + 1] = 0
        for index in range(n):
            buckets[(keys[source[index]] >> np.uint32(shift)) & mask] += 1
        total = 0
        for bucket in range(mask + 1):
            count = buckets[bucket]
            buckets[bucket] = total
            total += count
        for index in range(n):
            position = source[index]
            bucket = (keys[position] >> np.uint32(shift)) & mask
            target[buckets[bucket]] = position
            buckets[bucket] += 1
        source, target = target, source
        shift += bits
    if source is not order:
        order[:n] = source[:n]


@njit(cache=True, nogil=True)
def _gene_major_feature(
    counts: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    out: np.ndarray,
    nz_values: np.ndarray,
    nz_cells: np.ndarray,
    order: np.ndarray,
    order_scratch: np.ndarray,
    buckets: np.ndarray,
    zero_g: np.ndarray,
) -> bool:
    """Write one feature's library-size statistics from its raw counts.

    Each value is computed in float64 and ranked and summed as its float32
    rounding. Zero counts share the lowest rank, so only nonzero values are
    sorted, which holds only while no value is negative. Returns False,
    leaving ``out`` unwritten, when a value is negative or not finite.
    """
    n_cells = counts.shape[0]
    n_groups = group_counts.shape[0]
    sum_g = np.zeros(n_groups)
    nz_g = np.zeros(n_groups)
    rank_g = np.zeros(n_groups)
    drank_g = np.zeros(n_groups)
    n_nz = 0
    for c in range(n_cells):
        count = counts[c]
        if count == 0:
            continue
        exact = size_factor * np.float64(count) / totals[c]
        if log_transform:
            exact = np.log1p(exact)
        value = np.float32(exact)
        if value > 0.0:
            grp = int_indices[c]
            nz_values[n_nz] = value
            nz_cells[n_nz] = c
            n_nz += 1
            sum_g[grp] += value
            nz_g[grp] += 1.0
        elif value != 0.0:
            # Negative or NaN; a positive value that rounds to zero is a zero.
            return False
    for x in range(n_groups):
        # Only an infinite value makes a sum of positive float32 values infinite.
        if not sum_g[x] < np.inf:
            return False
        zero_g[x] = group_counts[x] - nz_g[x]

    n_zero = n_cells - n_nz
    tie_total = 0.0
    if n_zero > 0:
        zero_rank = (n_zero + 1.0) / 2.0
        zero_t = float(n_zero)
        if n_zero > 1:
            tie_total += zero_t * zero_t * zero_t - zero_t
        for x in range(n_groups):
            rank_g[x] = zero_g[x] * zero_rank
            drank_g[x] = zero_g[x]

    _argsort_positive(nz_values, n_nz, order, order_scratch, buckets)
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
            tie_total += tied_float * tied_float * tied_float - tied_float
        for k in range(i, j + 1):
            cell = nz_cells[order[k]]
            grp = int_indices[cell]
            rank_g[grp] += average_rank
            drank_g[grp] += dense_rank
        i = j + 1

    _write_group_statistics(
        out,
        sum_g,
        nz_g,
        rank_g,
        drank_g,
        group_counts,
        n_total,
        tie_total,
    )
    return True


@njit(cache=True, nogil=True)
def _gene_major_slot(
    raw: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    destination_rows: np.ndarray,
    rows: np.ndarray,
    slot: int,
    n_slots: int,
    out: np.ndarray,
) -> int:
    """Write the statistics of every ``n_slots``-th row of ``rows`` from ``slot``.

    The slot owns one set of per-cell scratch arrays. Returns the position in
    ``rows`` of the first row with a negative or non-finite normalized value,
    or the number of rows when every value is valid.
    """
    n_cells = raw.shape[1]
    nz_values = np.empty(n_cells, dtype=np.float32)
    nz_cells = np.empty(n_cells, dtype=np.int64)
    order = np.empty(n_cells, dtype=np.int64)
    order_scratch = np.empty(n_cells, dtype=np.int64)
    buckets = np.empty(2048, dtype=np.int64)
    zero_g = np.zeros(group_counts.shape[0])
    for position in range(slot, rows.shape[0], n_slots):
        row = rows[position]
        if not _gene_major_feature(
            raw[row],
            totals,
            size_factor,
            log_transform,
            int_indices,
            group_counts,
            n_total,
            out[destination_rows[row]],
            nz_values,
            nz_cells,
            order,
            order_scratch,
            buckets,
            zero_g,
        ):
            return position
    return int(rows.shape[0])


@njit(parallel=True, cache=True, nogil=True)
def _gene_major_slots(
    raw: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    destination_rows: np.ndarray,
    rows: np.ndarray,
    n_slots: int,
    out: np.ndarray,
) -> int:
    """Run ``n_slots`` slots of ``rows`` on Numba threads.

    Rows run in increasing order within a slot, so the smallest position a
    slot returns is that of the first invalid row.
    """
    first_invalid = np.empty(n_slots, dtype=np.int64)
    for slot in prange(n_slots):
        first_invalid[slot] = _gene_major_slot(
            raw,
            totals,
            size_factor,
            log_transform,
            int_indices,
            group_counts,
            n_total,
            destination_rows,
            rows,
            slot,
            n_slots,
            out,
        )
    return int(first_invalid.min())


@njit(cache=True, nogil=True)
def _marker_stats_gene_major(
    raw: np.ndarray,
    totals: np.ndarray,
    size_factor: float,
    log_transform: bool,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: float,
    destination_rows: np.ndarray,
    threads: int,
    out: np.ndarray,
) -> int:
    """Write library-size marker statistics from feature-major raw counts.

    Raw counts of any dtype are normalized by the float64 cell ``totals``.
    Up to ``threads`` slots each own one set of per-cell scratch arrays and
    take every slot-count-th selected row, and every row is computed alone,
    so results do not depend on ``threads``. One slot runs on the calling
    thread without Numba's parallel runtime, so concurrent callers can each
    run one. Rows with a negative ``destination_rows`` entry are skipped.
    Returns the local row of the first feature with a negative or non-finite
    normalized value, or -1 when every value is valid.
    """
    rows = np.flatnonzero(destination_rows >= 0)
    n_rows = rows.shape[0]
    n_slots = min(max(1, threads), n_rows)
    if n_slots > 1:
        first = _gene_major_slots(
            raw,
            totals,
            size_factor,
            log_transform,
            int_indices,
            group_counts,
            n_total,
            destination_rows,
            rows,
            n_slots,
            out,
        )
    else:
        first = _gene_major_slot(
            raw,
            totals,
            size_factor,
            log_transform,
            int_indices,
            group_counts,
            n_total,
            destination_rows,
            rows,
            0,
            1,
            out,
        )
    return int(rows[first]) if first < n_rows else -1


def gene_major_rank_scratch_bytes(
    *,
    n_cells: int,
    n_groups: int,
    n_features: int,
    nthreads: int,
) -> int:
    """Return the scratch of one gene-major call over ``n_features`` rows.

    Each of its ``nthreads`` threads holds the nonzero values, their cells,
    and the radix sort's order arrays for every cell.
    """
    cells = max(0, int(n_cells))
    groups = max(0, int(n_groups))
    threads = max(1, int(nthreads))
    int64 = np.dtype(np.int64).itemsize
    # Values, their cells, and the radix sort's order and scratch arrays.
    per_thread = (
        cells * (np.dtype(np.float32).itemsize + 3 * int64)
        + 2048 * int64
        + groups * 6 * np.dtype(np.float64).itemsize
    )
    # The selected rows and each thread's first invalid row.
    per_call = (max(0, int(n_features)) + threads) * int64
    return per_call + threads * per_thread


def batch_rank_scratch_bytes(
    *,
    n_cells: int,
    n_groups: int,
    n_features: int,
    nthreads: int,
) -> int:
    """Return the scratch of one dense rank call over ``n_features`` features.

    The call returns eight statistics per feature and group, and each of its
    ``nthreads`` threads ranks one feature at a time.
    """
    cells = max(0, int(n_cells))
    groups = max(0, int(n_groups))
    float64 = np.dtype(np.float64).itemsize
    # The argsort order and the average and dense ranks of every cell.
    per_thread = cells * (np.dtype(np.int64).itemsize + 2 * float64) + (
        groups * 5 * float64
    )
    statistics = max(0, int(n_features)) * groups * 8 * float64
    return statistics + max(1, int(nthreads)) * per_thread


def _batch_stats(
    data: np.ndarray,
    int_indices: np.ndarray,
    group_counts: np.ndarray,
    n_total: int,
    feature_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Run the dense marker kernel over cells-by-features normalized values.

    Column 6 of the result holds the Mann-Whitney z statistic.
    """
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
        group_counts.astype(np.float64),
        float(n_total),
    )
    return np.asarray(out)
