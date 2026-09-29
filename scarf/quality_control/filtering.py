from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..metadata.selection import NamedCellArtifact
from ..utils.arrays import within_bounds

__all__ = ["gaussian_quantile_bounds"]

FilterMethod = Literal["manual", "gaussian", "mad"]

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
    if not 0 < min_p < max_p < 1:
        raise ValueError("Gaussian filtering requires 0 < min_p < max_p < 1")
    median, deviation = float(np.median(values)), float(np.std(values))
    if deviation == 0:
        return median, median
    dist = norm(median, deviation)
    return float(dist.ppf(min_p)), float(dist.ppf(max_p))


def mad_bounds(values: np.ndarray, n_mads: float) -> tuple[float, float, float]:
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


def validated_work_scale(
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


def from_work_scale(bound: float, transform: str) -> float:
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


def clamp_metric_bound(
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


def validated_sample_labels(
    sample_labels: np.ndarray,
    active: np.ndarray,
    *,
    label_name: str = "sample labels",
) -> np.ndarray:
    """Check the sample labels of active rows and return an object copy.

    Active labels must be present, non-blank, finite, and of one kind: text,
    UTF-8 bytes, booleans, integers, or floats. They are Python scalars in the
    returned copy.
    """
    labels = np.asarray(sample_labels)
    active_mask = np.asarray(active)
    if labels.ndim != 1 or active_mask.ndim != 1 or labels.shape != active_mask.shape:
        raise ValueError("Sample labels and active selection must be aligned vectors")
    normalized = labels.astype(object, copy=True)
    # A typed array holds one kind of label, so each distinct value needs one
    # check; object arrays can mix kinds and are checked cell by cell.
    if labels.dtype != object:
        kinds = {
            _sample_label_kind(value, label_name)
            for value in pd.unique(labels[active_mask])
        }
    else:
        kinds = set()
        for index in np.flatnonzero(active_mask):
            value = labels[index]
            if isinstance(value, np.generic):
                value = value.item()
            kinds.add(_sample_label_kind(value, label_name))
            normalized[index] = value
    if len(kinds) > 1:
        raise ValueError(
            f"{label_name} must use one consistent label type among active cells"
        )
    return normalized


def unique_label_keys(labels: Iterable[object], *, label_name: str) -> list[str]:
    """Return the text that provenance records for each distinct label.

    Byte strings decode as UTF-8 and other labels use ``str``, so sample,
    capture, and group labels share one key encoding.

    Raises:
        ValueError: If two distinct labels have the same key.
    """
    keys = []
    for label in labels:
        value = label.item() if isinstance(label, np.generic) else label
        keys.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label_name} collide after provenance encoding")
    return keys


def _sample_label_kind(value: object, label_name: str) -> str:
    """Return the kind of one sample label, rejecting missing labels."""
    if isinstance(value, np.generic):
        value = value.item()
    missing = pd.isna(value)
    if isinstance(missing, bool | np.bool_) and bool(missing):
        raise ValueError(f"{label_name} contains missing labels among active cells")
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label_name} contains a non-UTF-8 bytes label") from exc
        if decoded.strip() == "":
            raise ValueError(f"{label_name} contains missing labels among active cells")
        return "bytes"
    if isinstance(value, str):
        if value.strip() == "":
            raise ValueError(f"{label_name} contains missing labels among active cells")
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{label_name} must contain finite labels")
        return "float"
    raise TypeError(
        f"{label_name} contains unsupported label type {type(value).__name__!r}"
    )


def _apply_bounds(
    values: np.ndarray,
    low: float | str | None,
    high: float | str | None,
    *,
    keep_bounds: bool = False,
) -> np.ndarray:
    """Return a boolean mask for one-dimensional values within bounds."""
    resolved = np.asarray(values)
    if resolved.ndim != 1:
        raise ValueError("Filter values must be a one-dimensional array")
    return within_bounds(resolved, low, high, keep_bounds=keep_bounds)


def sample_aware_mad_mask(
    *,
    values_by_attr: dict[str, np.ndarray],
    sample_labels: np.ndarray | None,
    active: np.ndarray,
    n_mads: float,
    min_cells_per_sample: int,
    attrs: list[str],
) -> tuple[np.ndarray, _MadProvenance]:
    """Build one cell mask from pooled or per-sample MAD bounds.

    Inactive cells are left ``True`` in the returned mask so callers can
    intersect with the current selection via ``update_key``.
    """
    n_cells = active.shape[0]
    pooled = sample_labels is None
    sample_labels = (
        np.full(n_cells, "all")
        if sample_labels is None
        else validated_sample_labels(sample_labels, active)
    )
    keep = np.ones(n_cells, dtype=bool)
    policies = {attr: _metric_policy(attr) for attr in attrs}
    raw_values_by_attr: dict[str, np.ndarray] = {}
    work_values_by_attr: dict[str, np.ndarray] = {}
    for attr in attrs:
        raw = np.asarray(values_by_attr[attr], dtype=float)
        raw_values_by_attr[attr] = raw
        work = np.empty(n_cells, dtype=float)
        work[active] = validated_work_scale(
            raw[active],
            attr=attr,
            transform=policies[attr]["transform"],
        )
        work_values_by_attr[attr] = work
    sample_sizes: dict[str, int] = {}
    skip_reasons: dict[str, str] = {}
    resolved_bounds: dict[str, dict[str, dict[str, object]]] = {}
    warnings: list[str] = []

    # Factorizing keeps first-seen order, which keeps provenance deterministic,
    # and turns per-sample label comparisons into integer comparisons.
    active_idx = np.flatnonzero(active)
    sample_codes, sample_uniques = pd.factorize(sample_labels[active])
    sample_keys = unique_label_keys(sample_uniques, label_name="Sample labels")

    for code, sample_key in enumerate(sample_keys):
        sample_idx = active_idx[sample_codes == code]
        sample_sizes[sample_key] = int(sample_idx.shape[0])
        resolved_bounds[sample_key] = {}
        group_label = "Selected cells" if pooled else f"Sample '{sample_key}'"

        if sample_idx.shape[0] < min_cells_per_sample:
            skip_reasons[sample_key] = "insufficient_cells"
            warnings.append(
                f"{group_label}: fewer than {min_cells_per_sample} "
                "active cells; retaining them without MAD filtering"
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
            low_t, high_t, scaled_mad = mad_bounds(work, n_mads)
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
                    f"{group_label}: zero MAD for '{attr}'; "
                    "retaining cells for this metric"
                )
                continue

            low: float | None
            high: float | None
            if direction == "upper":
                low = None
                high = clamp_metric_bound(
                    from_work_scale(high_t, transform),
                    transform=transform,
                    is_percent=is_percent,
                )
            else:
                low = clamp_metric_bound(
                    from_work_scale(low_t, transform),
                    transform=transform,
                    is_percent=is_percent,
                )
                high = clamp_metric_bound(
                    from_work_scale(high_t, transform),
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


@dataclass(frozen=True, slots=True)
class CellFilterResult:
    """Rows retained by one QC filtering rule and the bounds it resolved.

    ``retained`` aligns with the rows given to ``filter_cell_metrics``.
    ``gaussian_bounds`` and ``mad_provenance`` hold the resolved Gaussian
    bounds or the MAD provenance of those methods. ``sample_labels`` holds the
    validated MAD sample labels, normalized to Python scalars.
    """

    retained: np.ndarray
    gaussian_bounds: dict[str, dict[str, float]] | None
    mad_provenance: _MadProvenance | None
    sample_labels: np.ndarray | None


def _check_filter_bound(value: object, name: str) -> None:
    if value is None or isinstance(value, str):
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} values must be finite numbers, text, or None")
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} values must be finite; use None for no bound")


def validate_named_cell_artifacts(
    values: Iterable[NamedCellArtifact] | None,
    *,
    expected_kind: str,
    label: str,
) -> list[NamedCellArtifact]:
    """Check that named cell artifacts reference one kind under unique names.

    Args:
        values: Named cell-aligned artifacts, or None for none.
        expected_kind: The artifact kind every value must reference.
        label: The argument name that error messages use.

    Returns:
        The values as a list.

    Raises:
        TypeError: If a value is not a ``NamedCellArtifact``.
        ValueError: If a value references another kind or repeats a name.
    """
    sources = list(values or ())
    names: set[str] = set()
    for source in sources:
        if not isinstance(source, NamedCellArtifact):
            raise TypeError(f"{label} must contain NamedCellArtifact values")
        if source.artifact.kind != expected_kind:
            raise ValueError(f"{label} must reference {expected_kind!r} artifacts")
        if source.name in names:
            raise ValueError(f"{label} must use unique semantic names")
        names.add(source.name)
    return sources


def validate_cell_filter_sources(
    attrs: Iterable[str],
    artifact_metrics: Iterable[NamedCellArtifact] | None = None,
    *,
    sample_column: str | None = None,
    sample_artifact: NamedCellArtifact | None = None,
) -> tuple[list[str], list[NamedCellArtifact], NamedCellArtifact | None]:
    """Check the metric and sample sources of a cell filter before any read.

    Metadata metrics are distinct column names. Artifact metrics reference
    ``quality_metric`` artifacts under unique names that no metadata metric
    uses. Sample labels come from a metadata column or from a named
    ``hto_identity`` artifact, not both, and a sample artifact name differs
    from every artifact metric name.

    Returns:
        The metadata metrics, the artifact metrics, and the sample artifact.

    Raises:
        TypeError: If a metric or sample source has an invalid type.
        ValueError: If names repeat or collide, an artifact has another kind,
            or both sample sources are given.
    """
    attrs_list = list(attrs)
    if any(not isinstance(attr, str) for attr in attrs_list):
        raise TypeError("attrs must contain only column names")
    if len(set(attrs_list)) != len(attrs_list):
        raise ValueError("attrs must not contain duplicate columns")
    metrics = validate_named_cell_artifacts(
        artifact_metrics,
        expected_kind="quality_metric",
        label="artifact_metrics",
    )
    sample = (
        None
        if sample_artifact is None
        else validate_named_cell_artifacts(
            [sample_artifact],
            expected_kind="hto_identity",
            label="sample_artifact",
        )[0]
    )
    if sample_column is not None and sample is not None:
        raise ValueError("sample_column and sample_artifact are mutually exclusive")
    metric_names = {source.name for source in metrics}
    if sample is not None and sample.name in metric_names:
        raise ValueError("Sample and metric artifact names must be distinct")
    collisions = sorted(metric_names.intersection(attrs_list))
    if collisions:
        raise ValueError(
            f"Metadata and artifact QC metrics must use distinct names: {collisions}"
        )
    return attrs_list, metrics, sample


def validate_filter_bounds(
    lows: Sequence[object],
    highs: Sequence[object],
    *,
    keep_bounds: object,
) -> None:
    """Check manual filtering bounds before any metadata is read.

    Numeric bounds must be finite and cannot be booleans. Text bounds compare
    text columns lexically. A lower bound cannot exceed its upper bound.
    """
    if not isinstance(keep_bounds, bool):
        raise TypeError("keep_bounds must be a boolean")
    for low, high in zip(lows, highs, strict=True):
        _check_filter_bound(low, "lows")
        _check_filter_bound(high, "highs")
        if low is None or high is None:
            continue
        if isinstance(low, str) != isinstance(high, str):
            raise TypeError("Paired lows and highs must both be numbers or both text")
        if low > high:  # type: ignore[operator]
            raise ValueError("A lower bound cannot exceed its upper bound")


def _require_finite_metrics(
    values_by_attr: Mapping[str, np.ndarray],
    complete: np.ndarray,
) -> None:
    """Reject non-finite values among rows that inform automatic bounds."""
    n_complete = int(np.count_nonzero(complete))
    for attr, values in values_by_attr.items():
        array = np.asarray(values)
        if array.dtype.kind not in "fc":
            continue
        n_nonfinite = int(np.count_nonzero(~np.isfinite(array[complete])))
        if n_nonfinite:
            raise ValueError(
                f"QC metric {attr!r} has {n_nonfinite} non-finite value(s) among "
                f"{n_complete} selected cells with recorded values. Percentages "
                "are undefined for cells without counts, so exclude zero-count "
                "cells first: pass a cell selection that requires nCounts > 0, or "
                "open the DataStore with min_features_per_cell of 0 or more."
            )


def filter_cell_metrics(
    values_by_attr: Mapping[str, np.ndarray],
    missing_by_attr: Mapping[str, np.ndarray],
    active: np.ndarray,
    *,
    method: FilterMethod,
    lows: Sequence[float | None] = (),
    highs: Sequence[float | None] = (),
    keep_bounds: bool = False,
    min_p: float = 0.01,
    max_p: float = 0.99,
    sample_labels: np.ndarray | None = None,
    sample_missing: np.ndarray | None = None,
    sample_label_name: str = "sample labels",
    n_mads: float = 3.0,
    min_cells_per_sample: int = 20,
) -> CellFilterResult:
    """Apply one QC filtering rule to metric rows aligned with ``active``.

    Every metric, mask, and label vector must align with ``active``. A row
    flagged in ``missing_by_attr`` has no recorded value for that metric. It
    never passes a filter and does not inform Gaussian or MAD bounds.
    Automatic bounds require finite metrics on the remaining active rows. MAD
    filtering rejects an active row flagged in ``sample_missing``. The result
    must retain at least one active row.
    """
    active_mask = np.asarray(active, dtype=bool)
    attrs = list(values_by_attr)
    metric_missing = np.zeros(active_mask.shape[0], dtype=bool)
    for missing in missing_by_attr.values():
        metric_missing |= np.asarray(missing, dtype=bool)
    complete = active_mask & ~metric_missing
    keep = ~metric_missing
    gaussian_bounds: dict[str, dict[str, float]] | None = None
    mad_provenance: _MadProvenance | None = None
    validated_labels: np.ndarray | None = None
    if method == "manual":
        for attr, low, high in zip(attrs, lows, highs, strict=True):
            keep &= _apply_bounds(
                values_by_attr[attr],
                low,
                high,
                keep_bounds=keep_bounds,
            )
    elif method in ("gaussian", "mad"):
        if not complete.any():
            raise ValueError(
                "Cell filtering has no selected cells with complete metrics"
            )
        if method == "mad" and sample_labels is not None:
            if sample_missing is not None and np.any(
                active_mask & np.asarray(sample_missing, dtype=bool)
            ):
                raise ValueError(
                    f"{sample_label_name} contains missing labels among active cells"
                )
            validated_labels = validated_sample_labels(
                sample_labels,
                active_mask,
                label_name=sample_label_name,
            )
        _require_finite_metrics(values_by_attr, complete)
        if method == "gaussian":
            gaussian_bounds = {}
            for attr in attrs:
                values = np.asarray(values_by_attr[attr])
                low, high = gaussian_quantile_bounds(values[complete], min_p, max_p)
                if not np.isfinite([low, high]).all():
                    raise ValueError(
                        f"QC metric {attr!r} produced non-finite Gaussian bounds"
                    )
                gaussian_bounds[attr] = {"low": low, "high": high}
                keep &= _apply_bounds(values, low, high, keep_bounds=low == high)
        elif attrs:
            # Typed labels keep their dtype so the second validation checks
            # each distinct label once instead of every cell.
            mad_keep, mad_provenance = sample_aware_mad_mask(
                values_by_attr=dict(values_by_attr),
                sample_labels=sample_labels,
                active=complete,
                n_mads=n_mads,
                min_cells_per_sample=min_cells_per_sample,
                attrs=attrs,
            )
            keep &= mad_keep
    else:
        raise ValueError("method must be 'manual', 'gaussian', or 'mad'")
    retained = np.asarray(active_mask & keep, dtype=bool)
    if not retained.any():
        raise ValueError("Cell filtering removed every selected cell")
    return CellFilterResult(
        retained=retained,
        gaussian_bounds=gaussian_bounds,
        mad_provenance=mad_provenance,
        sample_labels=validated_labels,
    )
