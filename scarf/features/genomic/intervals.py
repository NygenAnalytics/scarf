from collections.abc import Iterable

import numpy as np
import pandas as pd
from numba import jit
from scipy.sparse import coo_matrix, csc_matrix

from ...utils.logging import logger

__all__ = [
    "binary_search",
    "create_bed_from_coord_ids",
    "get_feature_mappings",
    "get_ranges",
]


def create_bed_from_coord_ids(ids: Iterable[str]) -> pd.DataFrame:
    """Create a three-column BED dataframe from `chrom:start-end` strings."""
    ser = pd.Series(list(ids), dtype="object")
    chrom_rest = ser.str.split(":", expand=True)
    start_end = chrom_rest[1].str.split("-", expand=True)
    df = pd.DataFrame(
        {
            0: chrom_rest[0].to_numpy(),
            1: start_end[0].astype(np.int64).to_numpy(),
            2: start_end[1].astype(np.int64).to_numpy(),
        }
    )
    return df.sort_values(by=[0, 1])


@jit(nopython=True)
def binary_search(ranges: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Find overlapping range positions for sorted query intervals."""
    max_len = (ranges[:, 1] - ranges[:, 0]).max()
    n = queries.shape[0]
    ret_val = np.full((n, 2), 0)
    for i in range(n):
        start = max(0, queries[i][0] - max_len)
        end = queries[i][1]

        lo = 0
        hi = ranges.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if ranges[mid][0] < start:
                lo = mid + 1
            else:
                hi = mid
        starts_after = lo

        lo = 0
        hi = ranges.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if end < ranges[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        ends_before = lo

        if starts_after == ends_before:
            ret_val[i][0] = -1
            ret_val[i][1] = -1
        else:
            start = queries[i][0]
            m_pos_s, m_pos_e = -1, -1
            for j in range(starts_after, ends_before):
                if start < ranges[j][1] and end > ranges[j][0]:
                    if m_pos_s == -1:
                        m_pos_s = j
                    m_pos_e = j + 1
            ret_val[i][0] = m_pos_s
            ret_val[i][1] = m_pos_e

    return ret_val


def get_ranges(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """Extract integer start and end positions from a BED dataframe."""
    return np.asarray(df[[1, 2]][idx].values, dtype=int)


def get_feature_mappings(
    peaks_bed_df: pd.DataFrame,
    features_bed_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, csc_matrix]:
    """Build a sparse peak-to-feature interval overlap mapping."""
    feats_ids: list[str] = []
    feats_names: list[str] = []
    id_counter: dict[str, int] = {}
    map_peak_rows: list[int] = []
    map_feat_cols: list[int] = []
    n_no_match = 0
    feat_col = 0
    peak_chroms = set(peaks_bed_df[0].unique())

    for chrom in features_bed_df[0].unique():
        feats_chrom_idx = (features_bed_df[0] == chrom).values
        chrom_names = features_bed_df[4][feats_chrom_idx].values
        chrom_ids = features_bed_df[3][feats_chrom_idx].values

        feats_names.extend(chrom_names)
        for feature_id in chrom_ids:
            if feature_id not in id_counter:
                id_counter[feature_id] = 0
            id_counter[feature_id] += 1
            if id_counter[feature_id] > 1:
                feature_id = feature_id + f"_{id_counter[feature_id]}"
            feats_ids.append(feature_id)

        if chrom not in peak_chroms:
            logger.warning(f"Chromosome {chrom} not in the input peak coordinates")
            n_no_match += len(chrom_ids)
            feat_col += len(chrom_ids)
            continue

        peaks_chrom_idx = (peaks_bed_df[0] == chrom).values
        match_indices = binary_search(
            get_ranges(peaks_bed_df, peaks_chrom_idx),
            get_ranges(features_bed_df, feats_chrom_idx),
        ).astype(int)

        peak_idx = np.array(peaks_bed_df.index[peaks_chrom_idx])
        for match in match_indices:
            if match[0] == -1:
                assert match[1] == -1
                n_no_match += 1
            else:
                peaks_for_feat = peak_idx[match[0] : match[1]]
                map_peak_rows.extend(peaks_for_feat.tolist())
                map_feat_cols.extend([feat_col] * peaks_for_feat.shape[0])
            feat_col += 1

    if len(feats_ids) == 0:
        raise ValueError(
            "ERROR: None of the features were found in the assay. Melding failed"
        )
    feats_ids_arr = np.array(feats_ids)
    feats_names_arr = np.array(feats_names)
    n_features = feats_ids_arr.shape[0]
    if n_no_match == n_features:
        raise ValueError(
            "None of the provided features overlap with the peak coordinates"
        )
    if n_no_match:
        logger.warning(
            f"{n_no_match}/{n_features} features did not overlap with any peak"
        )
    logger.info(
        f"Mapped {n_features - n_no_match}/{n_features} features to peak coordinates"
    )
    if len(set(feats_ids_arr)) != n_features:
        raise ValueError(
            "ERROR: encountered an unexpected error. Somehow the feature ids are not unique "
            "despite our attempt to make them unique by appending a suffix. Please report this "
            "bug on Github"
        )
    assert feats_ids_arr.shape[0] == feats_names_arr.shape[0]
    mapping = coo_matrix(
        (
            np.ones(len(map_peak_rows), dtype=np.float64),
            (
                np.asarray(map_peak_rows, dtype=np.int64),
                np.asarray(map_feat_cols, dtype=np.int64),
            ),
        ),
        shape=(peaks_bed_df.shape[0], n_features),
    ).tocsc()
    return feats_ids_arr, feats_names_arr, mapping
