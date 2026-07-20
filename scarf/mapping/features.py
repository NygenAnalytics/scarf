from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from ..utils.compute import controlled_compute
from ..utils.logging import logger

if TYPE_CHECKING:
    from ..assay import Assay


def _order_features(
    s_assay: "Assay",
    t_assay: "Assay",
    s_feat_ids: np.ndarray,
    filter_null: bool,
    missing_feature_policy: str,
    nthreads: int,
    target_cell_key: str = "I",
) -> tuple[np.ndarray, np.ndarray]:
    s_ids = pd.Series(s_assay.feats.fetch_all("ids"))
    t_ids = pd.Series(t_assay.feats.fetch_all("ids"))
    if s_ids.duplicated().any():
        duplicates = s_ids[s_ids.duplicated()].iloc[:5].tolist()
        raise ValueError(f"Reference feature identifiers must be unique: {duplicates}")
    if t_ids.duplicated().any():
        duplicates = t_ids[t_ids.duplicated()].iloc[:5].tolist()
        raise ValueError(f"Target feature identifiers must be unique: {duplicates}")
    selected_ids = pd.Series(s_feat_ids)
    if selected_ids.duplicated().any():
        duplicates = selected_ids[selected_ids.duplicated()].iloc[:5].tolist()
        raise ValueError(
            f"Selected reference feature identifiers must be unique: {duplicates}"
        )
    t_idx = t_ids.isin(s_feat_ids)
    if t_idx.sum() == 0:
        raise ValueError(
            "ERROR: None of the features from reference were found in the target data"
        )
    if filter_null:
        if missing_feature_policy != "intersection":
            logger.warning(
                "`filter_null` has no effect unless missing_feature_policy is 'intersection'"
            )
        else:
            t_idx[t_idx] = (
                controlled_compute(
                    t_assay.rawData[:, list(t_idx[t_idx].index)][
                        t_assay.cells.active_index(target_cell_key), :
                    ].sum(axis=0),
                    nthreads,
                )
                != 0
            )
    t_idx = t_idx[t_idx].index
    if len(t_idx) == 0:
        raise ValueError("No target features remain after applying the feature policy")
    if missing_feature_policy == "intersection":
        s_idx = s_ids.isin(t_ids.values[t_idx])
    else:
        s_idx = s_ids.isin(s_feat_ids)
    s_idx = s_idx[s_idx].index
    t_idx_map = {v: k for k, v in t_ids.to_dict().items()}
    t_re_idx = np.array(
        [t_idx_map[x] if x in t_idx_map else -1 for x in s_ids.values[s_idx]]
    )
    if len(s_idx) != len(t_re_idx):
        raise AssertionError(
            "ERROR: Feature ordering failed. Please report this issue. "
            f"This is an unexpected scenario. Source has {len(s_idx)} features while target has "
            f"{len(t_re_idx)} features"
        )
    return s_idx.values, t_re_idx


def align_features(
    source_assay: "Assay",
    target_assay: "Assay",
    source_cell_key: str,
    source_feat_key: str,
    target_feat_key: str,
    target_cell_key: str,
    filter_null: bool,
    exclude_missing: bool,
    nthreads: int,
    missing_feature_policy: str | None = None,
    missing_feature_values: np.ndarray | None = None,
) -> np.ndarray:
    """Aligns target features to source features."""
    from ..storage.arrays import create_zarr_dataset

    if missing_feature_policy is None:
        missing_feature_policy = "intersection" if exclude_missing else "zero"
    if missing_feature_policy not in {
        "zero",
        "intersection",
        "error",
        "reference_mean",
    }:
        raise ValueError(
            "missing_feature_policy must be one of 'zero', 'intersection', 'error', "
            "or 'reference_mean'"
        )
    if exclude_missing and missing_feature_policy != "intersection":
        raise ValueError(
            "exclude_missing=True is only compatible with missing_feature_policy='intersection'"
        )

    source_feature_key = (
        source_feat_key
        if source_feat_key == "I"
        else f"{source_cell_key}__{source_feat_key}"
    )
    source_feat_ids = source_assay.feats.fetch("ids", key=source_feature_key)
    s_idx, t_idx = _order_features(
        source_assay,
        target_assay,
        source_feat_ids,
        filter_null,
        missing_feature_policy,
        nthreads,
        target_cell_key,
    )
    n_missing = int((t_idx == -1).sum())
    if missing_feature_policy == "error" and n_missing:
        raise ValueError(
            f"Target data is missing {n_missing} required reference features"
        )
    logger.info(f"{n_missing} features missing in target data")
    if missing_feature_values is not None:
        missing_feature_values = np.asarray(missing_feature_values)
        if missing_feature_values.shape != (len(t_idx),):
            raise ValueError(
                "missing_feature_values must have one value per aligned reference feature"
            )
    if missing_feature_policy == "reference_mean" and n_missing:
        if missing_feature_values is None:
            raise ValueError(
                "reference_mean feature handling requires missing_feature_values"
            )
        if not np.all(np.isfinite(missing_feature_values)):
            raise ValueError("missing_feature_values must be finite")
    normed_loc = f"normed__{source_cell_key}__{source_feat_key}"
    norm_params = cast(
        dict[str, Any], source_assay.z[normed_loc].attrs["subset_params"]
    )
    sorted_t_idx = np.array(sorted(t_idx[t_idx != -1]))

    normed_data = target_assay.normed(
        target_assay.cells.active_index(target_cell_key),
        sorted_t_idx,
        **norm_params,
    )
    normed_loc = f"normed__{target_cell_key}__{target_feat_key}"
    og = create_zarr_dataset(
        target_assay.z,
        f"{normed_loc}/data",
        (1000, len(t_idx)),
        "float64",
        (normed_data.shape[0], len(t_idx)),
    )
    pos_start, pos_end = 0, 0
    unsorter_idx = np.argsort(np.argsort(t_idx[t_idx != -1]))
    for i in normed_data.stream_blocks(
        nthreads=nthreads,
        msg=f"({target_assay.name}) Writing aligned data to {normed_loc}",
    ):
        pos_end += i.shape[0]
        if missing_feature_values is None:
            a = np.zeros((i.shape[0], len(t_idx)), dtype=i.dtype)
        else:
            a = np.broadcast_to(
                missing_feature_values,
                (i.shape[0], len(t_idx)),
            ).copy()
        a[:, np.where(t_idx != -1)[0]] = i[:, unsorter_idx]
        og[pos_start:pos_end, :] = a
        pos_start = pos_end
    return s_idx
