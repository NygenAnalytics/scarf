import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import logger


@dataclass(frozen=True, slots=True)
class FeatureAlignment:
    """Resolved feature union and per-source remapping for one assay."""

    mergedFeatsMap: pd.DataFrame
    featOrderMap: list[np.ndarray]
    nFeats: int
    overlapFraction: float

    def resident_bytes(self) -> int:
        frame_bytes = int(self.mergedFeatsMap.memory_usage(index=True, deep=True).sum())
        array_bytes = sum(
            array.nbytes
            for array in {id(array): array for array in self.featOrderMap}.values()
        )
        return frame_bytes + array_bytes + sys.getsizeof(self.featOrderMap)


def _is_missing_source(assay: Any) -> bool:
    return bool(getattr(assay, "isMissing", False))


def _get_feat_ids(assays: list[Any], names: list[str]) -> list[dict[str, str]]:
    ret_val: list[dict[str, str]] = []
    for assay, source_name in zip(assays, names, strict=True):
        if _is_missing_source(assay) or int(assay.feats.N) == 0:
            ret_val.append({})
            continue
        frame = assay.feats.to_pandas_dataframe(["names", "ids"])
        if frame["ids"].duplicated().any():
            raise ValueError(
                f"Duplicate feature IDs in assay {assay.name!r} of source "
                f"{source_name!r}; assign unique feature IDs before merging"
            )
        ret_val.append(
            dict(zip(frame["ids"].to_numpy(), frame["names"].to_numpy(), strict=True))
        )
    return ret_val


def _merge_order_feats(
    feat_collection: list[dict[str, str]],
) -> tuple[pd.DataFrame, float]:
    union_set: dict[str, str] = {}
    source_presence: Counter[str] = Counter()
    for ids in feat_collection:
        source_presence.update(ids.keys())
        for feature_id, feature_name in ids.items():
            if feature_id not in union_set:
                union_set[feature_id] = feature_name
    ret_val = pd.DataFrame(
        {
            "idx": list(range(len(union_set))),
            "names": list(union_set.values()),
            "ids": list(union_set.keys()),
        }
    )
    non_empty = sum(1 for ids in feat_collection if ids)
    if non_empty < 2:
        # A modality present in only one source is zero-filled elsewhere; every
        # feature is unique by construction rather than a failed overlap check.
        overlap = 1.0 if union_set else 0.0
    else:
        shared = sum(count > 1 for count in source_presence.values())
        overlap = 0.0 if not union_set else shared / len(union_set)
        if overlap == 0:
            raise ValueError(
                "No overlapping features found! Will not merge the files. Please check "
                "the features ids are comparable across the assays"
            )
        if overlap < 0.1:
            logger.warning("Fewer than 10% of features overlap across the assays")
    return ret_val, float(overlap)


def _ref_order_feat_idx(
    feat_collection: list[dict[str, str]],
    merged_feats: pd.DataFrame,
) -> list[np.ndarray]:
    positions = dict(zip(merged_feats["ids"], merged_feats["idx"], strict=True))
    return [
        np.fromiter((positions[feature_id] for feature_id in mapping), dtype=np.int64)
        for mapping in feat_collection
    ]


def align_features(assays: list[Any], names: list[str]) -> FeatureAlignment:
    """Compute the merged feature table and remapping for one assay type."""
    present = [
        assay
        for assay in assays
        if not _is_missing_source(assay) and int(assay.feats.N) > 0
    ]
    if not present:
        empty = pd.DataFrame({"idx": [], "names": [], "ids": []})
        return FeatureAlignment(
            mergedFeatsMap=empty,
            featOrderMap=[np.asarray([], dtype=np.int64) for _ in assays],
            nFeats=0,
            overlapFraction=0.0,
        )

    feat_collection = _get_feat_ids(assays, names)
    merged_feats, overlap = _merge_order_feats(feat_collection)
    return FeatureAlignment(
        mergedFeatsMap=merged_feats,
        featOrderMap=_ref_order_feat_idx(feat_collection, merged_feats),
        nFeats=int(merged_feats.shape[0]),
        overlapFraction=overlap,
    )


def dtype_for_integer_sum(dtype: np.dtype[Any], copies: int) -> np.dtype[Any]:
    if copies <= 1 or dtype.kind not in "biu":
        return dtype
    if dtype.kind in "bu":
        lower = 0
        upper = (1 if dtype.kind == "b" else np.iinfo(dtype).max) * copies
        for candidate in (np.uint8, np.uint16, np.uint32, np.uint64):
            candidate_info = np.iinfo(candidate)
            if lower >= candidate_info.min and upper <= candidate_info.max:
                return np.dtype(candidate)
    else:
        info = np.iinfo(dtype)
        lower = info.min * copies
        upper = info.max * copies
        for signed_candidate in (np.int8, np.int16, np.int32, np.int64):
            candidate_info = np.iinfo(signed_candidate)
            if lower >= candidate_info.min and upper <= candidate_info.max:
                return np.dtype(signed_candidate)
    return np.dtype(np.uint64 if dtype.kind in "bu" else np.int64)


def resolve_merge_dtype(
    assays: list[Any],
    feat_order_map: list[np.ndarray],
    explicit: str | None,
) -> str:
    if explicit is not None:
        return explicit
    present = [
        assay
        for assay in assays
        if not _is_missing_source(assay) and int(getattr(assay.feats, "N", 0)) > 0
    ]
    if not present:
        return "uint32"
    dtypes = {str(assay.rawData.dtype) for assay in present}
    if len(dtypes) != 1:
        return "float"
    max_copies = max(
        (
            int(np.unique(order_map, return_counts=True)[1].max())
            for order_map in feat_order_map
            if order_map.size
        ),
        default=1,
    )
    return str(dtype_for_integer_sum(np.dtype(present[0].rawData.dtype), max_copies))
