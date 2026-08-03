import re
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


def _get_feat_ids(assays: list[Any]) -> list[dict[str, str]]:
    ret_val: list[dict[str, str]] = []
    for assay in assays:
        if _is_missing_source(assay) or int(assay.feats.N) == 0:
            ret_val.append({})
            continue
        frame = assay.feats.to_pandas_dataframe(["names", "ids"])
        ret_val.append(
            dict(zip(frame["ids"].to_numpy(), frame["names"].to_numpy(), strict=True))
        )
    return ret_val


def _check_feat_ids(
    feat_collection: list[dict[str, str]],
    assays: list[Any],
    names: list[str],
) -> bool:
    for i, mapping in enumerate(feat_collection):
        keys = np.array(list(mapping.keys()))
        values = np.array(list(mapping.values()))
        if keys.size and np.equal(keys, values).all():
            logger.warning(
                f"Feature names and IDs are identical for assay "
                f"{assays[i].name} in dataset {names[i]}; "
                "feature names will be used as IDs"
            )
            return True
    return False


def _feat_suffix(feat_collection: list[dict[str, str]]) -> dict[int, int]:
    feat_suffix: dict[int, int] = {}
    for i, mapping in enumerate(feat_collection):
        keys = np.array(list(mapping.keys()))
        ends_0 = np.array([x.endswith("_0") for x in keys]).sum()
        ends_1 = np.array([x.endswith("_1") for x in keys]).sum()
        ends_2 = np.array([x.endswith("_2") for x in keys]).sum()
        if ends_0 > 0:
            feat_suffix[i] = 0
        elif ends_1 > 0:
            feat_suffix[i] = 1
        elif ends_2 > 0:
            raise ValueError(
                "Feature Numbering starts with 2, this is erroneous. Kindly check the data"
            )
        else:
            feat_suffix[i] = -1
    return feat_suffix


def _update_feat_ids(
    feat_collection: list[dict[str, str]],
    feat_suffix: dict[int, int],
) -> list[dict[str, str]]:
    pattern = re.compile(r"_\d+$")
    vals = np.array(list(feat_suffix.values()))
    vals = vals[vals > -1]
    min_val = int(vals.min()) if len(vals) > 0 else 0
    new_feat_collection = []
    for i, mapping in enumerate(feat_collection):
        in_dict: dict[str, str] = {}
        counter = Counter(mapping.values())
        if feat_suffix[i] == -1:
            sum_counter = {x: 0 for x in np.unique(list(mapping.values()))}
            for val in mapping.values():
                if counter[val] == 1:
                    in_dict[val] = val
                else:
                    updated_val = f"{val}_{min_val + sum_counter[val]}"
                    in_dict[updated_val] = updated_val
                sum_counter[val] += 1
        else:
            for val in mapping.values():
                if pattern.search(val):
                    num = int(val.split("_")[-1])
                    updated_val = pattern.sub(
                        f"_{min_val - feat_suffix[i] + num}",
                        val,
                    )
                    in_dict[updated_val] = updated_val
                else:
                    in_dict[val] = val
        new_feat_collection.append(in_dict)
    return new_feat_collection


def _update_feat_ids_for_map(
    feat_collection: list[dict[str, str]],
) -> list[dict[str, str]]:
    pattern = re.compile(r"_\d+$")
    new_feat_collection = []
    for mapping in feat_collection:
        in_dict: dict[str, str] = {}
        for feat_val in mapping.values():
            if pattern.search(feat_val):
                base_val = "_".join(feat_val.split("_")[:-1])
                if base_val not in in_dict:
                    in_dict[base_val] = base_val
            else:
                in_dict[feat_val] = feat_val
        new_feat_collection.append(in_dict)
    return new_feat_collection


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
    ret_val = []
    for ids in feat_collection:
        ordered_ids = pd.DataFrame({"ids": list(ids.keys())})
        vals = ordered_ids.merge(merged_feats, on="ids", how="left")["idx"].to_numpy()
        ret_val.append(np.asarray(vals))
    return ret_val


def _ref_order_feat_idx_map(
    feat_collection: list[dict[str, str]],
    merged_feats_map: pd.DataFrame,
) -> list[np.ndarray]:
    pattern = re.compile(r"_\d+$")
    name_to_idx = dict(
        zip(merged_feats_map["names"], merged_feats_map["idx"], strict=True)
    )
    feat_order = []
    for mapping in feat_collection:
        values_list = []
        for val in mapping.values():
            if pattern.search(val):
                val = "_".join(val.split("_")[:-1])
            values_list.append(val)
        feat_order.append(np.asarray([name_to_idx[name] for name in values_list]))
    return feat_order


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

    feat_collection = _get_feat_ids(assays)
    feat_name_ids_same = _check_feat_ids(feat_collection, assays, names)
    if feat_name_ids_same:
        # Only consider non-empty collections for suffix logic.
        non_empty = [mapping for mapping in feat_collection if mapping]
        feat_suffix_raw = _feat_suffix(non_empty)
        feat_suffix = {}
        non_empty_idx = 0
        for i, mapping in enumerate(feat_collection):
            if mapping:
                feat_suffix[i] = feat_suffix_raw[non_empty_idx]
                non_empty_idx += 1
            else:
                feat_suffix[i] = -1
        feat_collection = _update_feat_ids(feat_collection, feat_suffix)
        feat_collection_map = _update_feat_ids_for_map(feat_collection)
    else:
        feat_collection_map = [mapping.copy() for mapping in feat_collection]

    # Overlap uses only real feature maps; missing carriers stay empty slots.
    real_maps = [mapping for mapping in feat_collection if mapping]
    real_maps_for_names = [mapping for mapping in feat_collection_map if mapping]
    merged_feats, overlap = _merge_order_feats(real_maps)
    merged_feats_map, _ = _merge_order_feats(real_maps_for_names)
    feat_order = _ref_order_feat_idx(feat_collection, merged_feats)
    if feat_name_ids_same:
        feat_order_map = _ref_order_feat_idx_map(feat_collection, merged_feats_map)
    else:
        feat_order_map = [array.copy() for array in feat_order]
    return FeatureAlignment(
        mergedFeatsMap=merged_feats_map,
        featOrderMap=feat_order_map,
        nFeats=int(merged_feats_map.shape[0]),
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
