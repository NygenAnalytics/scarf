"""Resolve assay features and fetch their values without presentation dependencies."""

from collections.abc import Iterator, Sequence
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
    """Resolve exact feature IDs or case-insensitive names without implicit reduction."""
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
        indices = np.flatnonzero(
            assay.feats.fetch_all("ids") == str(ref.value)
        ).tolist()
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
    output = np.empty((len(cell_idx), len(resolved)), dtype=np.float64)
    for slots, start, values in iter_normalized_feature_blocks(
        store, resolved, cell_idx, normalization
    ):
        output[start : start + len(values), slots] = values
    return output


def iter_normalized_feature_blocks(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    cell_idx: np.ndarray,
    normalization: NormalizationSpec | None = None,
) -> Iterator[tuple[list[int], int, np.ndarray]]:
    """Yield feature slots, selected-row offsets, and normalized value blocks."""
    normalization = normalization or NormalizationSpec()
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
        blocks = (
            (values,)
            if isinstance(values, np.ndarray)
            else values.stream_blocks(nthreads=store.nthreads)
        )
        local_indices = [
            np.searchsorted(physical_indices, resolved[slot].indices) for slot in slots
        ]
        start = 0
        for block in blocks:
            normalized = np.asarray(block, dtype=np.float64)
            if normalized.ndim == 1:
                normalized = normalized.reshape(-1, 1)
            if normalization.transform == "log1p":
                normalized = np.log1p(normalized)
            output = np.empty((len(normalized), len(slots)), dtype=np.float64)
            for column, (slot, local) in enumerate(
                zip(slots, local_indices, strict=True)
            ):
                selected = normalized[:, local]
                if selected.shape[1] == 1:
                    output[:, column] = selected[:, 0]
                elif resolved[slot].reduction == "sum":
                    output[:, column] = selected.sum(axis=1)
                else:
                    output[:, column] = selected.mean(axis=1)
            yield slots, start, output
            start += len(normalized)
