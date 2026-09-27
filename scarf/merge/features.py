import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

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


type FeatureKey = Literal["ids", "names"]

_NO_OVERLAP = (
    "No overlapping features found! Will not merge the files. No feature {kind} "
    "are shared across the sources."
)
_NAME_HINT = (
    " If the sources use different ID schemes but comparable feature names (for "
    "example Ensembl IDs in one source and gene symbols in another), pass "
    "feature_key='names' to match features by name."
)


def _source_keys(
    assays: list[Any | None], names: list[str], key: FeatureKey
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return each source's merge keys and feature names in source order."""
    keyed: list[tuple[np.ndarray, np.ndarray]] = []
    for assay, source_name in zip(assays, names, strict=True):
        if assay is None or int(assay.feats.N) == 0:
            empty = np.asarray([], dtype=object)
            keyed.append((empty, empty))
            continue
        frame = assay.feats.to_pandas_dataframe(["names", "ids"])
        if key == "ids" and frame["ids"].duplicated().any():
            raise ValueError(
                f"Duplicate feature IDs in assay {assay.name!r} of source "
                f"{source_name!r}; assign unique feature IDs before merging"
            )
        keyed.append((frame[key].to_numpy(), frame["names"].to_numpy()))
    return keyed


def _merge_order_feats(
    keyed: list[tuple[np.ndarray, np.ndarray]], key: FeatureKey
) -> tuple[pd.DataFrame, float]:
    union: dict[Any, Any] = {}
    source_presence: Counter[Any] = Counter()
    for keys, feature_names in keyed:
        source_presence.update(set(keys.tolist()))
        for feature_key, feature_name in zip(
            keys.tolist(), feature_names.tolist(), strict=True
        ):
            union.setdefault(feature_key, feature_name)
    ret_val = pd.DataFrame(
        {
            "idx": list(range(len(union))),
            "names": list(union.values()),
            "ids": list(union.keys()),
        }
    )
    non_empty = sum(1 for keys, _names in keyed if keys.size)
    if non_empty < 2:
        # A modality present in only one source is zero-filled elsewhere; every
        # feature is unique by construction rather than a failed overlap check.
        overlap = 1.0 if union else 0.0
    else:
        shared = sum(count > 1 for count in source_presence.values())
        overlap = 0.0 if not union else shared / len(union)
        if overlap == 0:
            message = _NO_OVERLAP.format(kind="IDs" if key == "ids" else "names")
            raise ValueError(message + (_NAME_HINT if key == "ids" else ""))
        if overlap < 0.1:
            logger.warning("Fewer than 10% of features overlap across the assays")
    return ret_val, float(overlap)


def align_features(
    assays: list[Any | None], names: list[str], *, key: FeatureKey = "ids"
) -> FeatureAlignment:
    """Compute the merged feature table and remapping for one assay type.

    ``key="ids"`` matches features by exact ID. ``key="names"`` matches them by
    name, uses each name as the merged ID, and sums features that share a name
    within one source. A missing source is ``None`` and maps no features.
    """
    present = [
        assay for assay in assays if assay is not None and int(assay.feats.N) > 0
    ]
    if not present:
        empty = pd.DataFrame({"idx": [], "names": [], "ids": []})
        return FeatureAlignment(
            mergedFeatsMap=empty,
            featOrderMap=[np.asarray([], dtype=np.int64) for _ in assays],
            nFeats=0,
            overlapFraction=0.0,
        )

    keyed = _source_keys(assays, names, key)
    merged_feats, overlap = _merge_order_feats(keyed, key)
    positions = pd.Index(merged_feats["ids"])
    return FeatureAlignment(
        mergedFeatsMap=merged_feats,
        featOrderMap=[
            np.asarray(positions.get_indexer(keys), dtype=np.int64)
            for keys, _names in keyed
        ],
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
    assays: list[Any | None],
    feat_order_map: list[np.ndarray],
    explicit: str | None,
) -> str:
    if explicit is not None:
        return explicit
    present = [
        assay
        for assay in assays
        if assay is not None and int(getattr(assay.feats, "N", 0)) > 0
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
