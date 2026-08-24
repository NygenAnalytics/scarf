"""Feature resolution and bounded group reducers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._contracts import (
    FeatureRef,
    FeatureReduction,
    LookupBy,
    NormalizationSpec,
    Standardize,
    StudyDesign,
)
from ._style import sort_categories


@dataclass(frozen=True, slots=True)
class ResolvedFeature:
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


def coerce_feature_list(
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
) -> list[tuple[str | None, str | FeatureRef]]:
    """Return (group_label, feature) pairs preserving order."""
    if isinstance(features, Mapping):
        out: list[tuple[str | None, str | FeatureRef]] = []
        for group, items in features.items():
            for item in items:
                out.append((str(group), item))
        return out
    return [(None, item) for item in features]


def resolve_cell_selection(
    n: int,
    *,
    subset: np.ndarray | None = None,
    subset_name: str | None = None,
    category_values: np.ndarray | None = None,
    groups: Sequence[Any] | None = None,
) -> tuple[np.ndarray, list[Any] | None]:
    """Build a boolean mask from ``subset`` and optional category ``groups``.

    ``subset`` must be boolean and length ``n`` when provided. ``groups`` keeps
    only those categories from ``category_values`` and defines their order.
    When ``groups`` is omitted, category order is natural via
    :func:`sort_categories` over observed values (or ``None`` if no categories).
    """
    mask = np.ones(n, dtype=bool)
    if subset is not None:
        sub = np.asarray(subset)
        if sub.dtype != bool:
            label = subset_name or "subset_by"
            raise TypeError(f"{label!r} must be boolean; got {sub.dtype}")
        if len(sub) != n:
            raise ValueError("subset_by length must match selected cells")
        mask &= sub

    group_order: list[Any] | None = None
    if category_values is not None:
        cats = np.asarray(category_values)
        if len(cats) != n:
            raise ValueError("category values length must match selected cells")
        present = set(pd.unique(cats).tolist())
        if groups is not None:
            group_order = list(groups)
            if not group_order:
                raise ValueError("groups must be non-empty when provided")
            missing = [g for g in group_order if g not in present]
            if missing:
                raise ValueError(
                    "groups contains labels not present in the data: "
                    + ", ".join(map(str, missing[:10]))
                )
            mask &= np.isin(cats, group_order)
        elif mask.any():
            group_order = sort_categories(list(pd.unique(cats[mask])))
        else:
            group_order = []

    if not mask.any():
        raise ValueError("No cells remain after applying subset/groups filters")
    return mask, group_order


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


def summarize_features_by_group(
    store: Any,
    *,
    features: Sequence[str | FeatureRef] | Mapping[str, Sequence[str | FeatureRef]],
    group_by: str | tuple[str, ...],
    cell_key: str = "I",
    from_assay: str | None = None,
    sample_by: str | None = None,
    study_design: StudyDesign | None = None,
    normalization: NormalizationSpec | None = None,
    expression_cutoff: float = 0.0,
    max_groups: int = 500,
    max_features: int = 2000,
    max_samples: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Aggregate features by group. With sample_by, samples get equal weight.

    Missing group combinations are omitted (not filled with zeros).
    """
    condition_by: str | None = None
    if study_design is not None:
        sample_by = study_design.sample_by
        condition_by = study_design.condition_by

    pairs = coerce_feature_list(features)
    if len(pairs) > max_features:
        raise ValueError(
            f"Too many features ({len(pairs)} > {max_features}). "
            "Raise max_features explicitly if intentional."
        )
    resolved = [
        resolve_feature(store, feat, from_assay=from_assay) for _, feat in pairs
    ]
    group_labels = [g for g, _ in pairs]
    feature_labels = [r.label for r in resolved]

    if isinstance(group_by, str):
        group_keys: tuple[str, ...] = (group_by,)
    else:
        group_keys = tuple(group_by)
    if len(group_keys) == 0 or len(group_keys) > 2:
        raise ValueError("group_by must have 1 or 2 keys")

    cells = store.cells
    cell_idx = cells.active_index(cell_key)
    group_cols = [cells.fetch(k, key=cell_key) for k in group_keys]
    n_groups = int(
        pd.DataFrame({k: c for k, c in zip(group_keys, group_cols)})
        .drop_duplicates()
        .shape[0]
    )
    if n_groups > max_groups:
        raise ValueError(
            f"Too many groups ({n_groups} > {max_groups}). "
            "Raise max_groups explicitly if intentional."
        )

    expr = fetch_normalized_feature_matrix(
        store,
        resolved,
        cell_idx,
        normalization=normalization,
    )
    frac_mask = expr > expression_cutoff
    base = pd.DataFrame({gk: col for gk, col in zip(group_keys, group_cols)})

    if sample_by is not None:
        samples = cells.fetch(sample_by, key=cell_key)
        if condition_by is not None:
            conditions = cells.fetch(condition_by, key=cell_key)
            check = pd.DataFrame({"sample": samples, "condition": conditions})
            nunique = check.groupby("sample", observed=False)["condition"].nunique()
            bad = nunique[nunique > 1]
            if len(bad):
                raise ValueError(
                    "condition_by is not constant within sample(s): "
                    + ", ".join(map(str, list(bad.index[:10])))
                )
        valid = pd.notna(samples) & (np.asarray(samples, dtype=object) != "")
        if int(valid.sum()) == 0:
            raise ValueError("No cells with valid sample_by values")
        uniq_samples = pd.unique(np.asarray(samples)[valid])
        if len(uniq_samples) > max_samples:
            raise ValueError(
                f"Too many samples ({len(uniq_samples)} > {max_samples}). "
                "Raise max_samples explicitly if intentional."
            )
        parts: list[pd.DataFrame] = []
        base_v = base.loc[valid].copy()
        base_v["sample"] = np.asarray(samples)[valid]
        for j in range(len(resolved)):
            part = base_v.copy()
            part["feature"] = feature_labels[j]
            part["feature_group"] = group_labels[j]
            part["value"] = expr[valid, j]
            part["detected"] = frac_mask[valid, j]
            parts.append(part)
        long = pd.concat(parts, ignore_index=True)
        gb_keys = ["sample", *group_keys, "feature", "feature_group"]
        per_sample = (
            long.groupby(gb_keys, observed=False, dropna=False)
            .agg(
                mean=("value", "mean"),
                fraction=("detected", "mean"),
                n_cells=("value", "size"),
                variance=("value", "var"),
            )
            .reset_index()
        )
        agg_keys = [*group_keys, "feature", "feature_group"]
        aggregate = (
            per_sample.groupby(agg_keys, observed=False, dropna=False)
            .agg(
                mean=("mean", "mean"),
                fraction=("fraction", "mean"),
                n_cells=("n_cells", "sum"),
                n_samples=("sample", "nunique"),
                variance=("variance", "mean"),
            )
            .reset_index()
        )
        return aggregate, per_sample

    parts = []
    for j in range(len(resolved)):
        part = base.copy()
        part["feature"] = feature_labels[j]
        part["feature_group"] = group_labels[j]
        part["value"] = expr[:, j]
        part["detected"] = frac_mask[:, j]
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    agg_keys = [*group_keys, "feature", "feature_group"]
    aggregate = (
        long.groupby(agg_keys, observed=False, dropna=False)
        .agg(
            mean=("value", "mean"),
            fraction=("detected", "mean"),
            n_cells=("value", "size"),
            variance=("value", "var"),
        )
        .reset_index()
    )
    return aggregate, None
