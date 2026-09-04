from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["gaussian_quantile_bounds"]

_MAD_SCALE = 1.4826
_COUNT_SUFFIXES = ("nCounts", "nFeatures")
_PERCENT_SUFFIXES = ("percentMito", "percentRibo")


class _MadProvenance(TypedDict):
    mad_scale: float
    metric_policies: dict[str, dict[str, str]]
    sample_sizes: dict[str, int]
    skip_reasons: dict[str, str]
    resolved_bounds: dict[str, dict[str, dict[str, object]]]
    warnings: list[str]


def gaussian_quantile_bounds(
    values: np.ndarray,
    min_p: float = 0.01,
    max_p: float = 0.99,
) -> tuple[float, float]:
    dist = norm(np.median(values), np.std(values))
    return float(dist.ppf(min_p)), float(dist.ppf(max_p))


def _mad_bounds(values: np.ndarray, n_mads: float) -> tuple[float, float, float]:
    """Return ``(low, high, scaled_mad)`` for robust per-sample thresholds.

    ``scaled_mad`` is ``1.4826 * MAD``. Callers must treat a zero scaled MAD as a
    skip condition rather than applying a zero-width exclusive interval.
    """
    if not np.isfinite(n_mads) or n_mads <= 0:
        raise ValueError("n_mads must be finite and greater than 0")
    if not np.isfinite(values).all():
        raise ValueError("MAD input values must all be finite")
    median = float(np.median(values))
    scaled_mad = float(_MAD_SCALE * np.median(np.abs(values - median)))
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(n_mads * scaled_mad)
        low = float(median - distance)
        high = float(median + distance)
    if not np.isfinite([median, scaled_mad, low, high]).all():
        raise ValueError("n_mads produces non-finite MAD bounds")
    return low, high, scaled_mad


def _metric_policy(attr: str) -> dict[str, str]:
    """Return transform and bound direction for a QC metadata column."""
    for suffix in _COUNT_SUFFIXES:
        if attr == suffix or attr.endswith(f"_{suffix}"):
            return {"transform": "log1p", "bound_direction": "two_sided"}
    for suffix in _PERCENT_SUFFIXES:
        if attr == suffix or attr.endswith(f"_{suffix}"):
            return {"transform": "identity", "bound_direction": "upper"}
    return {"transform": "identity", "bound_direction": "two_sided"}


def _validated_work_scale(
    values: np.ndarray,
    *,
    attr: str,
    transform: str,
) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError(f"QC values in '{attr}' contain non-finite entries")
    if transform == "log1p":
        if (raw < 0).any():
            raise ValueError(f"QC values in '{attr}' must be non-negative before log1p")
        work = np.asarray(np.log1p(raw), dtype=float)
    else:
        work = raw
    if not np.isfinite(work).all():
        raise ValueError(
            f"QC values in '{attr}' contain non-finite entries after {transform}"
        )
    return work


def _from_work_scale(bound: float, transform: str) -> float:
    if transform == "log1p":
        with np.errstate(over="ignore", invalid="ignore"):
            resolved = float(np.expm1(bound))
    else:
        resolved = float(bound)
    if not np.isfinite(resolved):
        raise ValueError(
            f"MAD bound is non-finite after converting from the {transform} scale"
        )
    return resolved


def _clamp_metric_bound(
    bound: float,
    *,
    transform: str,
    is_percent: bool,
) -> float:
    if not np.isfinite(bound):
        raise ValueError("Resolved MAD bounds must be finite")
    if transform == "log1p":
        resolved = max(0.0, bound)
    elif is_percent:
        resolved = min(100.0, max(0.0, bound))
    else:
        resolved = bound
    if not np.isfinite(resolved):
        raise ValueError("Resolved MAD bounds must be finite")
    return resolved


def _validated_sample_labels(
    sample_labels: np.ndarray,
    active: np.ndarray,
    *,
    label_name: str = "sample labels",
) -> np.ndarray:
    labels = np.asarray(sample_labels)
    active_mask = np.asarray(active)
    if labels.ndim != 1 or active_mask.ndim != 1 or labels.shape != active_mask.shape:
        raise ValueError("Sample labels and active selection must be aligned vectors")
    normalized = labels.astype(object, copy=True)
    kinds: set[str] = set()
    for index in np.flatnonzero(active_mask):
        value = labels[index]
        if isinstance(value, np.generic):
            value = value.item()
        missing = pd.isna(value)
        if isinstance(missing, bool | np.bool_) and bool(missing):
            raise ValueError(f"{label_name} contains missing labels among active cells")
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{label_name} contains a non-UTF-8 bytes label"
                ) from exc
            if decoded.strip() == "":
                raise ValueError(
                    f"{label_name} contains missing labels among active cells"
                )
            kind = "bytes"
        elif isinstance(value, str):
            if value.strip() == "":
                raise ValueError(
                    f"{label_name} contains missing labels among active cells"
                )
            kind = "str"
        elif isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, int):
            kind = "int"
        elif isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError(f"{label_name} must contain finite labels")
            kind = "float"
        else:
            raise TypeError(
                f"{label_name} contains unsupported label type {type(value).__name__!r}"
            )
        normalized[index] = value
        kinds.add(kind)
    if len(kinds) > 1:
        raise ValueError(
            f"{label_name} must use one consistent label type among active cells"
        )
    return normalized


def _apply_bounds(
    values: np.ndarray,
    low: float | None,
    high: float | None,
    *,
    keep_bounds: bool = False,
) -> np.ndarray:
    """Return a boolean mask for one-dimensional values within numeric bounds."""
    resolved = np.asarray(values)
    if resolved.ndim != 1:
        raise ValueError("Filter values must be a one-dimensional array")
    lower = -np.inf if low is None else low
    upper = np.inf if high is None else high
    if keep_bounds:
        return np.asarray((resolved >= lower) & (resolved <= upper), dtype=bool)
    return np.asarray((resolved > lower) & (resolved < upper), dtype=bool)


def _sample_aware_mad_mask(
    *,
    values_by_attr: dict[str, np.ndarray],
    sample_labels: np.ndarray,
    active: np.ndarray,
    n_mads: float,
    min_cells_per_sample: int,
    attrs: list[str],
) -> tuple[np.ndarray, _MadProvenance]:
    """Build one cell mask from per-sample MAD bounds.

    Inactive cells are left ``True`` in the returned mask so callers can
    intersect with the current selection via ``update_key``.
    """
    n_cells = active.shape[0]
    sample_labels = _validated_sample_labels(sample_labels, active)
    keep = np.ones(n_cells, dtype=bool)
    policies = {attr: _metric_policy(attr) for attr in attrs}
    raw_values_by_attr: dict[str, np.ndarray] = {}
    work_values_by_attr: dict[str, np.ndarray] = {}
    for attr in attrs:
        raw = np.asarray(values_by_attr[attr], dtype=float)
        raw_values_by_attr[attr] = raw
        work = np.empty(n_cells, dtype=float)
        work[active] = _validated_work_scale(
            raw[active],
            attr=attr,
            transform=policies[attr]["transform"],
        )
        work_values_by_attr[attr] = work
    sample_sizes: dict[str, int] = {}
    skip_reasons: dict[str, str] = {}
    resolved_bounds: dict[str, dict[str, dict[str, object]]] = {}
    warnings: list[str] = []

    # Preserve first-seen label order for deterministic provenance.
    ordered_samples: list[object] = []
    seen: set[object] = set()
    for label in sample_labels[active]:
        key = label if not isinstance(label, np.generic) else label.item()
        if key not in seen:
            seen.add(key)
            ordered_samples.append(key)

    for sample in ordered_samples:
        sample_key = (
            sample.decode("utf-8") if isinstance(sample, bytes) else str(sample)
        )
        if sample_key in sample_sizes:
            raise ValueError(
                "Sample labels collide after deterministic provenance encoding"
            )
        sample_mask = np.zeros(n_cells, dtype=bool)
        sample_mask[active] = sample_labels[active] == sample
        sample_idx = np.flatnonzero(sample_mask)
        sample_sizes[sample_key] = int(sample_idx.shape[0])
        resolved_bounds[sample_key] = {}

        if sample_idx.shape[0] < min_cells_per_sample:
            skip_reasons[sample_key] = "insufficient_cells"
            warnings.append(
                f"Sample '{sample_key}' has fewer than {min_cells_per_sample} "
                "active cells; retaining all of its cells without MAD filtering"
            )
            continue

        sample_keep = np.ones(sample_idx.shape[0], dtype=bool)
        for attr in attrs:
            policy = policies[attr]
            transform = policy["transform"]
            direction = policy["bound_direction"]
            is_percent = direction == "upper" and transform == "identity"
            raw = raw_values_by_attr[attr][sample_idx]
            work = work_values_by_attr[attr][sample_idx]
            low_t, high_t, scaled_mad = _mad_bounds(work, n_mads)
            if scaled_mad == 0.0:
                resolved_bounds[sample_key][attr] = {
                    "low": None,
                    "high": None,
                    "skip_reason": "zero_mad",
                    "transform": transform,
                    "bound_direction": direction,
                    "scaled_mad": 0.0,
                }
                warnings.append(
                    f"Sample '{sample_key}' has zero MAD for '{attr}'; "
                    "retaining cells for this metric"
                )
                continue

            low: float | None
            high: float | None
            if direction == "upper":
                low = None
                high = _clamp_metric_bound(
                    _from_work_scale(high_t, transform),
                    transform=transform,
                    is_percent=is_percent,
                )
            else:
                low = _clamp_metric_bound(
                    _from_work_scale(low_t, transform),
                    transform=transform,
                    is_percent=is_percent,
                )
                high = _clamp_metric_bound(
                    _from_work_scale(high_t, transform),
                    transform=transform,
                    is_percent=is_percent,
                )

            resolved_bounds[sample_key][attr] = {
                "low": low,
                "high": high,
                "skip_reason": None,
                "transform": transform,
                "bound_direction": direction,
                "scaled_mad": scaled_mad,
            }
            sample_keep &= _apply_bounds(raw, low, high)

        keep[sample_idx] = sample_keep

    provenance: _MadProvenance = {
        "mad_scale": _MAD_SCALE,
        "metric_policies": policies,
        "sample_sizes": sample_sizes,
        "skip_reasons": skip_reasons,
        "resolved_bounds": resolved_bounds,
        "warnings": warnings,
    }
    return keep, provenance
