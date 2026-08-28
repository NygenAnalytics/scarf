"""Resolve assay features and fetch their values without presentation dependencies."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metadata.selection import (
    FeatureReduction,
    FeatureRef,
    LookupBy,
    NormalizationSpec,
    Standardize,
)

__all__ = [
    "ResolvedFeature",
    "fetch_normalized_feature_matrix",
    "resolve_feature",
]


@dataclass(frozen=True, slots=True)
class ResolvedFeature:
    """Concrete feature identity resolved against an assay."""

    assay: str
    by: LookupBy
    indices: tuple[int, ...]
    ids: tuple[str, ...]
    names: tuple[str, ...]
    label: str
    reduction: FeatureReduction | None
    raw: FeatureRef | str

    def scale_key(
        self,
        normalization: NormalizationSpec,
        standardize: Standardize = "none",
    ) -> tuple[Any, ...]:
        return (
            self.assay,
            self.by,
            self.ids,
            self.reduction,
            normalization.source,
            normalization.transform,
            standardize,
        )


def resolve_feature(
    store: Any,
    feature: str | FeatureRef,
    *,
    from_assay: str | None = None,
) -> ResolvedFeature:
    """Resolve a feature against one assay. Case-sensitive. No silent averaging."""
    if isinstance(feature, FeatureRef):
        ref = feature
    else:
        ref = FeatureRef(value=feature, assay=from_assay)

    assay_name = ref.assay or from_assay or store._defaultAssay
    assay = store._get_assay(assay_name)

    if ref.by == "index":
        idx = int(ref.value)
        if idx < 0 or idx >= assay.feats.N:
            raise KeyError(
                f"Feature index {idx} out of range for assay {assay_name!r} "
                f"(N={assay.feats.N})"
            )
        indices = [idx]
    elif ref.by == "id":
        indices = list(assay.feats.get_index_by([str(ref.value)], "ids"))
    else:
        indices = list(assay.feats.get_index_by([str(ref.value)], "names"))

    if len(indices) == 0:
        raise KeyError(
            f"Feature {ref.value!r} not found in assay {assay_name!r} by {ref.by!r}"
        )
    if len(indices) > 1 and ref.reduction is None:
        raise ValueError(
            f"Feature {ref.value!r} matches {len(indices)} entries in assay "
            f"{assay_name!r} at indices {indices}. Pass reduction='mean' or "
            f"reduction='sum', or look up by id/index."
        )

    idx_arr = np.asarray(indices)
    names = tuple(str(x) for x in assay.feats.fetch_all("names")[idx_arr])
    ids = tuple(str(x) for x in assay.feats.fetch_all("ids")[idx_arr])
    label = ref.label or (
        names[0] if len(names) == 1 else f"{ref.value}:{ref.reduction}"
    )
    return ResolvedFeature(
        assay=assay_name,
        by=ref.by,
        indices=tuple(int(i) for i in indices),
        ids=ids,
        names=names,
        label=label,
        reduction=ref.reduction,
        raw=ref if isinstance(feature, FeatureRef) else feature,
    )


def fetch_normalized_feature_matrix(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    cell_idx: np.ndarray,
    normalization: NormalizationSpec | None = None,
) -> np.ndarray:
    """Return assay-native or raw feature values in requested feature order."""
    from ..utils.compute import controlled_compute

    normalization = normalization or NormalizationSpec()
    if not resolved:
        return np.empty((len(cell_idx), 0), dtype=np.float64)

    output = np.empty((len(cell_idx), len(resolved)), dtype=np.float64)
    assay_slots: dict[str, list[int]] = {}
    for slot, feat in enumerate(resolved):
        assay_slots.setdefault(feat.assay, []).append(slot)

    for assay_name, slots in assay_slots.items():
        assay = store._get_assay(assay_name)
        physical_indices = np.unique(
            np.concatenate(
                [np.asarray(resolved[slot].indices, dtype=np.int64) for slot in slots]
            )
        )
        if normalization.source == "raw":
            values = assay.rawData[:, physical_indices][cell_idx, :]
        else:
            values = assay.normed(
                cell_idx=cell_idx,
                feat_idx=physical_indices,
            )
        normalized = controlled_compute(values, store.nthreads).astype(np.float64)
        if normalized.ndim == 1:
            normalized = normalized.reshape(-1, 1)
        if normalization.transform == "log1p":
            normalized = np.log1p(normalized)

        for slot in slots:
            feat = resolved[slot]
            local = np.searchsorted(
                physical_indices, np.asarray(feat.indices, dtype=np.int64)
            )
            values = normalized[:, local]
            if values.shape[1] == 1:
                output[:, slot] = values[:, 0]
            elif feat.reduction == "sum":
                output[:, slot] = values.sum(axis=1)
            else:
                output[:, slot] = values.mean(axis=1)
    return output
