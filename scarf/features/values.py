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
    "resolve_feature_batch",
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


class _AssayFeatureIndex:
    """Feature names, ids, and lookup maps read once for one assay."""

    __slots__ = ("_by_id", "_by_name", "ids", "n_features", "names")

    def __init__(self, assay: Any) -> None:
        self.n_features = int(assay.feats.N)
        self.names = np.asarray(assay.feats.fetch_all("names"))
        self.ids = np.asarray(assay.feats.fetch_all("ids"))
        self._by_name: dict[str, list[int]] | None = None
        self._by_id: dict[str, list[int]] | None = None

    def by_name(self, value: str) -> list[int]:
        if self._by_name is None:
            by_name: dict[str, list[int]] = {}
            for index, name in enumerate(self.names):
                by_name.setdefault(str(name).upper(), []).append(index)
            self._by_name = by_name
        return self._by_name.get(value.upper(), [])

    def by_id(self, value: str) -> list[int]:
        if self._by_id is None:
            by_id: dict[str, list[int]] = {}
            for index, identifier in enumerate(self.ids):
                by_id.setdefault(str(identifier), []).append(index)
            self._by_id = by_id
        return self._by_id.get(value, [])


def resolve_feature_batch(
    store: Any,
    features: Sequence[str | FeatureRef],
    *,
    from_assay: str | None = None,
) -> list[ResolvedFeature]:
    """Resolve several features in input order, reading each assay's index once.

    Lookups follow :func:`resolve_feature`: ids and indices are exact, names are
    case-insensitive, and a name or id matching several features requires an
    explicit reduction.
    """
    indexes: dict[str, _AssayFeatureIndex] = {}
    resolved: list[ResolvedFeature] = []
    for feature in features:
        if isinstance(feature, FeatureRef):
            ref = feature
        else:
            ref = FeatureRef(value=feature, assay=from_assay)

        assay_name = ref.assay or from_assay or store._defaultAssay
        index = indexes.get(assay_name)
        if index is None:
            index = _AssayFeatureIndex(store._get_assay(assay_name))
            indexes[assay_name] = index

        if ref.by == "index":
            idx = int(ref.value)
            if idx < 0 or idx >= index.n_features:
                raise KeyError(
                    f"Feature index {idx} out of range for assay {assay_name!r} "
                    f"(N={index.n_features})"
                )
            indices = [idx]
        elif ref.by == "id":
            indices = index.by_id(str(ref.value))
        else:
            indices = index.by_name(str(ref.value))

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
        names = tuple(str(x) for x in index.names[idx_arr])
        ids = tuple(str(x) for x in index.ids[idx_arr])
        label = ref.label or (
            names[0] if len(names) == 1 else f"{ref.value}:{ref.reduction}"
        )
        resolved.append(
            ResolvedFeature(
                assay=assay_name,
                by=ref.by,
                indices=tuple(int(i) for i in indices),
                ids=ids,
                names=names,
                label=label,
                reduction=ref.reduction,
                raw=ref if isinstance(feature, FeatureRef) else feature,
            )
        )
    return resolved


def resolve_feature(
    store: Any,
    feature: str | FeatureRef,
    *,
    from_assay: str | None = None,
) -> ResolvedFeature:
    """Resolve exact feature IDs or case-insensitive names without implicit reduction."""
    return resolve_feature_batch(store, [feature], from_assay=from_assay)[0]


def fetch_normalized_feature_matrix(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    cell_idx: np.ndarray,
    normalization: NormalizationSpec | None = None,
) -> np.ndarray:
    """Return assay-native or raw feature values in requested feature order.

    Row blocks are sized so that they fit the memory budget beside the output.
    """
    output = np.empty((len(cell_idx), len(resolved)), dtype=np.float64)
    for slots, start, values in iter_normalized_feature_blocks(
        store,
        resolved,
        cell_idx,
        normalization,
        resident_bytes=output.nbytes,
    ):
        output[start : start + len(values), slots] = values
    return output


def iter_normalized_feature_blocks(
    store: Any,
    resolved: Sequence[ResolvedFeature],
    cell_idx: np.ndarray,
    normalization: NormalizationSpec | None = None,
    *,
    resident_bytes: int = 0,
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
            else values._stream_blocks(
                nthreads=store.nthreads,
                msg=None,
                prefetch=None,
                row_mask=None,
                resident_bytes=resident_bytes,
            )
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
