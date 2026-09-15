"""Registered and core-parity cell-quality profile projections."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

from ...quality_control.filtering import (
    _apply_bounds,
    _clamp_metric_bound,
    _from_work_scale,
    _mad_bounds,
    _sample_aware_mad_mask,
    _validated_sample_labels,
    _validated_work_scale,
    gaussian_quantile_bounds,
)

type RegisteredCellQcProfile = Literal[
    "retainWithFlags",
    "globalMad5",
    "captureMad5",
    "captureMad3Sensitivity",
    "pooledReferenceMad5",
]
type CellQualityProfile = (
    RegisteredCellQcProfile | Literal["coreGlobalGaussian", "coreSampleMad3"]
)


def cell_qc_policy(
    action: str, registered_profile: RegisteredCellQcProfile | None
) -> CellQualityProfile | None:
    """Name the exact filtering route shared by selection, execution, and resume."""
    if registered_profile is not None:
        return registered_profile
    if action == "globalGaussian":
        return "coreGlobalGaussian"
    if action == "sampleMad":
        return "coreSampleMad3"
    return None


type AutoFilterAction = Literal["globalGaussian", "sampleMad"]
type QcMetricRole = Literal[
    "count",
    "feature",
    "mitochondrial",
    "ribosomal",
    "diagnostic",
]
type QcRemovalDirection = Literal["lower", "upper", "none"]

REGISTERED_CELL_QC_PROFILES: tuple[RegisteredCellQcProfile, ...] = (
    "retainWithFlags",
    "globalMad5",
    "captureMad5",
    "captureMad3Sensitivity",
    "pooledReferenceMad5",
)


def qc_metric_execution_name(
    name: str,
    *,
    artifact_id: str | None = None,
    collides_with_metadata: bool = False,
) -> str:
    """Return the deterministic metric name passed to core filtering helpers."""
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ValueError("QC metric names must be non-empty trimmed strings")
    if not collides_with_metadata:
        return name
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("A colliding artifact metric requires an artifact id")
    return f"artifact_{artifact_id[:16]}_{name}"


@dataclass(frozen=True, slots=True)
class RegisteredQcThreshold:
    """One data-derived threshold for one metric and reference population."""

    metric: str
    role: QcMetricRole
    group: str
    nMads: float
    transform: Literal["identity", "log1p"]
    removalDirection: QcRemovalDirection
    median: float
    scaledMad: float
    lowerRemoval: float | None
    upperRemoval: float | None
    upperFlag: float | None
    skipReason: Literal["zeroMad"] | None = None

    def to_dict(self) -> dict[str, str | float | None]:
        """Return JSON-safe threshold evidence."""
        return {
            "metric": self.metric,
            "role": self.role,
            "group": self.group,
            "nMads": self.nMads,
            "transform": self.transform,
            "removalDirection": self.removalDirection,
            "median": self.median,
            "scaledMad": self.scaledMad,
            "lowerRemoval": self.lowerRemoval,
            "upperRemoval": self.upperRemoval,
            "upperFlag": self.upperFlag,
            "skipReason": self.skipReason,
        }


@dataclass(frozen=True, slots=True)
class CaptureQcComparison:
    """Comparison of one capture median with the complete active population."""

    capture: str
    cells: int
    adverseGlobalOutlier: bool
    adverseAxes: tuple[QcMetricRole, ...] = ()
    independentAdverseAxes: int = 0
    wholeCaptureFailure: bool = False
    retainedCells: int | None = None
    retainedFraction: float | None = None
    reasons: tuple[str, ...] = ()
    metricComparisons: dict[str, dict[str, float | str | None]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe failed-capture evidence."""
        return {
            "capture": self.capture,
            "cells": self.cells,
            "adverseGlobalOutlier": self.adverseGlobalOutlier,
            "adverseAxes": list(self.adverseAxes),
            "independentAdverseAxes": self.independentAdverseAxes,
            "wholeCaptureFailure": self.wholeCaptureFailure,
            "retainedCells": self.retainedCells,
            "retainedFraction": self.retainedFraction,
            "reasons": list(self.reasons),
            "metricComparisons": self.metricComparisons,
        }


@dataclass(frozen=True, slots=True)
class RegisteredQcProjection:
    """Exact masks and evidence for one registered cell-QC profile."""

    profile: RegisteredCellQcProfile
    keep: np.ndarray
    flags: dict[str, np.ndarray]
    thresholds: tuple[RegisteredQcThreshold, ...]
    captureSizes: dict[str, int] = field(default_factory=dict)
    retainedByCapture: dict[str, int] = field(default_factory=dict)
    captureComparisons: tuple[CaptureQcComparison, ...] = ()
    failedCaptureCandidates: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def retainedCells(self) -> int:
        """Number of active cells retained by the projection."""
        return int(self.keep.sum())

    @property
    def flagCounts(self) -> dict[str, int]:
        """Counts for each non-removal diagnostic flag."""
        return {name: int(mask.sum()) for name, mask in self.flags.items()}

    @property
    def metricFlagCounts(self) -> dict[str, dict[str, int]]:
        """Counts grouped by exact metric and flag name."""
        grouped: dict[str, dict[str, int]] = {}
        for name, mask in self.flags.items():
            metric, flag = name.rsplit(":", 1)
            grouped.setdefault(metric, {})[flag] = int(mask.sum())
        return grouped


@dataclass(frozen=True, slots=True)
class AutoFilterProjection:
    """Exact in-memory projection of one core ``auto_filter_cells`` path."""

    action: AutoFilterAction
    keep: np.ndarray
    flags: dict[str, np.ndarray]
    parameters: dict[str, object]
    captureSizes: dict[str, int] = field(default_factory=dict)
    retainedByCapture: dict[str, int] = field(default_factory=dict)
    captureComparisons: tuple[CaptureQcComparison, ...] = ()
    failedCaptureCandidates: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def retainedCells(self) -> int:
        """Number of active cells retained by the projection."""
        return int(self.keep.sum())

    @property
    def flagCounts(self) -> dict[str, int]:
        """Counts for each exact metric-bound flag."""
        return {name: int(mask.sum()) for name, mask in self.flags.items()}

    @property
    def metricFlagCounts(self) -> dict[str, dict[str, int]]:
        """Counts grouped by exact metric and bound side."""
        grouped: dict[str, dict[str, int]] = {}
        for name, mask in self.flags.items():
            metric, flag = name.rsplit(":", 1)
            grouped.setdefault(metric, {})[flag] = int(mask.sum())
        return grouped


def registered_qc_metric_role(metric: str) -> QcMetricRole:
    """Classify one conventional cell-quality metric without gene inspection."""
    normalized = metric.replace("_", "").lower()
    if normalized.endswith("ncounts") or normalized.endswith("totalcounts"):
        return "count"
    if normalized.endswith("nfeatures") or normalized.endswith("ngenesbycounts"):
        return "feature"
    if (
        normalized.endswith("percentmito")
        or normalized.endswith("pctcountsmt")
        or normalized.endswith("mitochondrialpercent")
    ):
        return "mitochondrial"
    if (
        normalized.endswith("percentribo")
        or normalized.endswith("pctcountsribo")
        or normalized.endswith("ribosomalpercent")
    ):
        return "ribosomal"
    return "diagnostic"


def _metric_policy(
    role: QcMetricRole,
) -> tuple[Literal["identity", "log1p"], QcRemovalDirection]:
    if role in {"count", "feature"}:
        return "log1p", "lower"
    if role in {"mitochondrial", "ribosomal"}:
        return "identity", "upper"
    return "identity", "none"


def _validated_inputs(
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    active_mask = np.asarray(active, dtype=bool)
    if active_mask.ndim != 1:
        raise ValueError("active must be a one-dimensional boolean vector")
    if not active_mask.any():
        raise ValueError("active must select at least one cell")
    values: dict[str, np.ndarray] = {}
    for metric, raw in values_by_metric.items():
        if (
            not isinstance(metric, str)
            or not metric.strip()
            or metric != metric.strip()
        ):
            raise ValueError("QC metric names must be non-empty trimmed strings")
        array = np.asarray(raw, dtype=float)
        if array.ndim != 1 or array.shape != active_mask.shape:
            raise ValueError(f"QC metric {metric!r} must align with active")
        if not np.isfinite(array[active_mask]).all():
            raise ValueError(f"QC metric {metric!r} contains non-finite active values")
        values[metric] = array
    return values, active_mask


def _threshold(
    metric: str,
    values: np.ndarray,
    reference: np.ndarray,
    *,
    group: str,
    n_mads: float,
) -> RegisteredQcThreshold:
    role = registered_qc_metric_role(metric)
    transform, removal_direction = _metric_policy(role)
    work = _validated_work_scale(values[reference], attr=metric, transform=transform)
    median_work = float(np.median(work))
    low_work, high_work, scaled_mad = _mad_bounds(work, n_mads)
    if scaled_mad == 0.0:
        median = _clamp_metric_bound(
            _from_work_scale(median_work, transform),
            transform=transform,
            is_percent=role in {"mitochondrial", "ribosomal"},
        )
        return RegisteredQcThreshold(
            metric=metric,
            role=role,
            group=group,
            nMads=n_mads,
            transform=transform,
            removalDirection=removal_direction,
            median=median,
            scaledMad=0.0,
            lowerRemoval=None,
            upperRemoval=None,
            upperFlag=None,
            skipReason="zeroMad",
        )

    median = _clamp_metric_bound(
        _from_work_scale(median_work, transform),
        transform=transform,
        is_percent=role in {"mitochondrial", "ribosomal"},
    )
    low = _clamp_metric_bound(
        _from_work_scale(low_work, transform),
        transform=transform,
        is_percent=role in {"mitochondrial", "ribosomal"},
    )
    high = _clamp_metric_bound(
        _from_work_scale(high_work, transform),
        transform=transform,
        is_percent=role in {"mitochondrial", "ribosomal"},
    )
    return RegisteredQcThreshold(
        metric=metric,
        role=role,
        group=group,
        nMads=n_mads,
        transform=transform,
        removalDirection=removal_direction,
        median=median,
        scaledMad=scaled_mad,
        lowerRemoval=low if removal_direction == "lower" else None,
        upperRemoval=high if removal_direction == "upper" else None,
        upperFlag=high if role in {"count", "feature"} else None,
    )


def _apply_threshold(
    values: np.ndarray,
    target: np.ndarray,
    threshold: RegisteredQcThreshold,
    keep: np.ndarray,
    flags: dict[str, np.ndarray],
    *,
    apply_removal: bool,
) -> None:
    if threshold.lowerRemoval is not None:
        removed = target & (values < threshold.lowerRemoval)
        low_flag = flags.setdefault(
            f"{threshold.metric}:lowQuality",
            np.zeros(target.shape[0], dtype=bool),
        )
        low_flag[removed] = True
        if apply_removal:
            keep[removed] = False
    if threshold.upperRemoval is not None:
        removed = target & (values > threshold.upperRemoval)
        mito_flag = flags.setdefault(
            f"{threshold.metric}:highMito",
            np.zeros(target.shape[0], dtype=bool),
        )
        mito_flag[removed] = True
        if apply_removal:
            keep[removed] = False
    if threshold.upperFlag is not None:
        flag_name = f"{threshold.metric}:high"
        flag = flags.setdefault(flag_name, np.zeros(target.shape[0], dtype=bool))
        flag[target & (values > threshold.upperFlag)] = True


def _ordered_capture_masks(
    capture_labels: np.ndarray,
    active: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    labels = _validated_sample_labels(
        capture_labels,
        active,
        label_name="physical capture labels",
    )
    captures: list[tuple[str, np.ndarray]] = []
    seen_values: list[object] = []
    seen_keys: set[str] = set()
    for raw in labels[active]:
        value = raw.item() if isinstance(raw, np.generic) else raw
        if any(value == seen for seen in seen_values):
            continue
        key = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if key in seen_keys:
            raise ValueError(
                "Physical capture labels collide after provenance encoding"
            )
        seen_values.append(value)
        seen_keys.add(key)
        captures.append((key, active & (labels == raw)))
    return captures


def _global_capture_comparisons(
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
    captures: list[tuple[str, np.ndarray]],
) -> tuple[CaptureQcComparison, ...]:
    global_thresholds = {
        metric: _threshold(
            metric,
            values,
            active,
            group="globalComparison",
            n_mads=5.0,
        )
        for metric, values in values_by_metric.items()
        if registered_qc_metric_role(metric) != "diagnostic"
    }
    comparisons: list[CaptureQcComparison] = []
    for capture, mask in captures:
        metric_comparisons: dict[str, dict[str, float | str | None]] = {}
        reasons: list[str] = []
        adverse_axes: set[QcMetricRole] = set()
        for metric, threshold in global_thresholds.items():
            role = threshold.role
            transform, _ = _metric_policy(role)
            capture_work = _validated_work_scale(
                values_by_metric[metric][mask],
                attr=metric,
                transform=transform,
            )
            capture_median_work = float(np.median(capture_work))
            capture_median = _clamp_metric_bound(
                _from_work_scale(capture_median_work, transform),
                transform=transform,
                is_percent=role in {"mitochondrial", "ribosomal"},
            )
            global_mad = threshold.scaledMad
            standardized_shift = (
                None
                if global_mad == 0.0
                else (capture_median_work - np.log1p(threshold.median)) / global_mad
                if transform == "log1p"
                else (capture_median - threshold.median) / global_mad
            )
            adverse = (
                threshold.lowerRemoval is not None
                and capture_median < threshold.lowerRemoval
            ) or (
                threshold.upperRemoval is not None
                and capture_median > threshold.upperRemoval
            )
            if adverse:
                reasons.append(f"{metric}:{role}:adverseGlobalMedian")
                adverse_axes.add(role)
            metric_comparisons[metric] = {
                "role": role,
                "captureMedian": capture_median,
                "globalMedian": threshold.median,
                "globalLower": threshold.lowerRemoval,
                "globalUpper": threshold.upperRemoval,
                "standardizedMedianShift": standardized_shift,
            }
        comparisons.append(
            CaptureQcComparison(
                capture=capture,
                cells=int(mask.sum()),
                adverseGlobalOutlier=bool(reasons),
                adverseAxes=tuple(sorted(adverse_axes)),
                independentAdverseAxes=len(adverse_axes),
                wholeCaptureFailure=len(adverse_axes) >= 2,
                reasons=tuple(reasons),
                metricComparisons=metric_comparisons,
            )
        )
    return tuple(comparisons)


def _with_capture_retention(
    comparisons: tuple[CaptureQcComparison, ...],
    captures: list[tuple[str, np.ndarray]],
    keep: np.ndarray,
) -> tuple[CaptureQcComparison, ...]:
    masks = dict(captures)
    output: list[CaptureQcComparison] = []
    for comparison in comparisons:
        mask = masks[comparison.capture]
        retained = int((mask & keep).sum())
        fraction = retained / comparison.cells if comparison.cells else 0.0
        output.append(
            CaptureQcComparison(
                capture=comparison.capture,
                cells=comparison.cells,
                adverseGlobalOutlier=comparison.adverseGlobalOutlier,
                adverseAxes=comparison.adverseAxes,
                independentAdverseAxes=comparison.independentAdverseAxes,
                wholeCaptureFailure=comparison.wholeCaptureFailure,
                retainedCells=retained,
                retainedFraction=fraction,
                reasons=comparison.reasons,
                metricComparisons=comparison.metricComparisons,
            )
        )
    return tuple(output)


def project_registered_qc_profile(
    profile: RegisteredCellQcProfile,
    *,
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
    capture_labels: np.ndarray | None = None,
    grouping_proven: bool = False,
    min_cells_per_capture: int = 20,
    pooled_reference_captures: tuple[str, ...] | None = None,
) -> RegisteredQcProjection:
    """Project one registered QC profile without reading or writing a matrix."""
    if profile not in REGISTERED_CELL_QC_PROFILES:
        raise ValueError(f"Unknown registered cell-QC profile {profile!r}")
    if min_cells_per_capture < 2:
        raise ValueError("min_cells_per_capture must be at least 2")
    values, active_mask = _validated_inputs(values_by_metric, active)
    filtering_values = {
        metric: metric_values
        for metric, metric_values in values.items()
        if registered_qc_metric_role(metric) in {"count", "feature", "mitochondrial"}
    }
    if not filtering_values:
        if profile != "retainWithFlags":
            raise ValueError("MAD profiles require count, feature, or mito metrics")
        return RegisteredQcProjection(
            profile=profile,
            keep=active_mask.copy(),
            flags={},
            thresholds=(),
            warnings=("No registered count, feature, or mito metrics were available",),
        )

    capture_profile = profile in {
        "captureMad5",
        "captureMad3Sensitivity",
        "pooledReferenceMad5",
    }
    if capture_profile and (capture_labels is None or not grouping_proven):
        raise ValueError(
            f"{profile} requires an explicitly proven physical capture grouping"
        )
    captures = (
        _ordered_capture_masks(np.asarray(capture_labels), active_mask)
        if capture_labels is not None and grouping_proven
        else []
    )
    capture_sizes = {name: int(mask.sum()) for name, mask in captures}
    if profile in {"captureMad5", "captureMad3Sensitivity"}:
        undersized = [
            name for name, size in capture_sizes.items() if size < min_cells_per_capture
        ]
        if undersized:
            raise ValueError(
                f"{profile} requires at least {min_cells_per_capture} cells in every "
                f"capture; undersized={undersized}"
            )
    if profile == "pooledReferenceMad5":
        references = tuple(pooled_reference_captures or ())
        if len(references) < 2 or len(references) != len(set(references)):
            raise ValueError(
                "pooledReferenceMad5 requires at least two unique reference captures"
            )
        unknown = sorted(set(references).difference(capture_sizes))
        if unknown:
            raise ValueError(f"Unknown pooled reference captures: {unknown}")
        pooled_cells = sum(capture_sizes[name] for name in references)
        if pooled_cells < min_cells_per_capture:
            raise ValueError(
                "Pooled reference captures do not contain enough active cells"
            )

    n_mads = 3.0 if profile == "captureMad3Sensitivity" else 5.0
    keep = active_mask.copy()
    flags: dict[str, np.ndarray] = {}
    thresholds: list[RegisteredQcThreshold] = []
    warnings: list[str] = []

    reference_groups: list[tuple[str, np.ndarray, np.ndarray]]
    if profile in {"retainWithFlags", "globalMad5"}:
        reference_groups = [("global", active_mask, active_mask)]
    elif profile == "pooledReferenceMad5":
        reference_names = set(pooled_reference_captures or ())
        reference = np.zeros(active_mask.shape[0], dtype=bool)
        for name, mask in captures:
            if name in reference_names:
                reference |= mask
        reference_groups = [("pooledReference", active_mask, reference)]
    else:
        reference_groups = [(name, mask, mask) for name, mask in captures]

    for group, target, reference in reference_groups:
        for metric, metric_values in filtering_values.items():
            threshold = _threshold(
                metric,
                metric_values,
                reference,
                group=group,
                n_mads=n_mads,
            )
            thresholds.append(threshold)
            if threshold.skipReason is not None:
                warnings.append(
                    f"Ignored {metric!r} for {group!r} because its MAD was zero"
                )
                continue
            _apply_threshold(
                metric_values,
                target,
                threshold,
                keep,
                flags,
                apply_removal=profile != "retainWithFlags",
            )

    comparison_values = {
        metric: metric_values
        for metric, metric_values in values.items()
        if registered_qc_metric_role(metric) != "diagnostic"
    }
    comparisons = (
        _with_capture_retention(
            _global_capture_comparisons(
                comparison_values,
                active_mask,
                captures,
            ),
            captures,
            keep,
        )
        if captures
        else ()
    )
    failed = tuple(
        comparison.capture
        for comparison in comparisons
        if comparison.wholeCaptureFailure
    )
    if failed:
        warnings.append(
            "Captures failed at least two independent global QC axes: "
            + ", ".join(failed)
        )
    retained_by_capture = {name: int((mask & keep).sum()) for name, mask in captures}
    return RegisteredQcProjection(
        profile=profile,
        keep=keep,
        flags=flags,
        thresholds=tuple(thresholds),
        captureSizes=capture_sizes,
        retainedByCapture=retained_by_capture,
        captureComparisons=comparisons,
        failedCaptureCandidates=failed,
        warnings=tuple(warnings),
    )


def _auto_bound_flags(
    *,
    metric: str,
    values: np.ndarray,
    target: np.ndarray,
    low: float | None,
    high: float | None,
    flags: dict[str, np.ndarray],
) -> None:
    if low is not None:
        flags.setdefault(
            f"{metric}:low",
            np.zeros(target.shape[0], dtype=bool),
        )[target & (values <= low)] = True
    if high is not None:
        flags.setdefault(
            f"{metric}:high",
            np.zeros(target.shape[0], dtype=bool),
        )[target & (values >= high)] = True


def project_auto_filter_profile(
    action: AutoFilterAction,
    *,
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
    sample_labels: np.ndarray | None = None,
    grouping_proven: bool = False,
    min_p: float = 0.01,
    max_p: float = 0.99,
    n_mads: float = 3.0,
    min_cells_per_sample: int = 20,
) -> AutoFilterProjection:
    """Project one existing core auto-filter path without writing data.

    The projection calls the same filtering helpers as
    :meth:`DataStore.auto_filter_cells`. It rejects any input for which the
    core operation would not produce finite bounds.
    """
    if action not in {"globalGaussian", "sampleMad"}:
        raise ValueError(f"Unknown automatic cell-QC action {action!r}")
    values, active_mask = _validated_inputs(values_by_metric, active)
    if not values:
        raise ValueError("Automatic cell-QC profiles require at least one metric")

    flags: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    captures: list[tuple[str, np.ndarray]] = []
    parameters: dict[str, object]
    if action == "globalGaussian":
        keep = active_mask.copy()
        resolved_bounds: dict[str, dict[str, float]] = {}
        for metric, metric_values in values.items():
            low, high = gaussian_quantile_bounds(
                metric_values[active_mask],
                min_p,
                max_p,
            )
            if not np.isfinite([low, high]).all():
                raise ValueError(
                    f"QC metric {metric!r} produced non-finite Gaussian bounds"
                )
            resolved_bounds[metric] = {"low": low, "high": high}
            keep &= _apply_bounds(metric_values, low, high)
            _auto_bound_flags(
                metric=metric,
                values=metric_values,
                target=active_mask,
                low=low,
                high=high,
                flags=flags,
            )
        parameters = {
            "minP": float(min_p),
            "maxP": float(max_p),
            "resolvedBounds": resolved_bounds,
        }
        if sample_labels is not None and grouping_proven:
            captures = _ordered_capture_masks(
                np.asarray(sample_labels),
                active_mask,
            )
    else:
        if sample_labels is None or not grouping_proven:
            raise ValueError(
                "sampleMad requires an explicitly proven physical capture grouping"
            )
        if min_p != 0.01 or max_p != 0.99:
            raise ValueError(
                "sampleMad requires the core Gaussian probabilities to remain "
                "at 0.01 and 0.99"
            )
        keep_from_core, provenance = _sample_aware_mad_mask(
            values_by_attr=values,
            sample_labels=np.asarray(sample_labels),
            active=active_mask,
            n_mads=n_mads,
            min_cells_per_sample=min_cells_per_sample,
            attrs=list(values),
        )
        keep = active_mask & keep_from_core
        captures = _ordered_capture_masks(
            np.asarray(sample_labels),
            active_mask,
        )
        capture_masks = dict(captures)
        raw_bounds = provenance["resolved_bounds"]
        for capture, bounds_by_metric in raw_bounds.items():
            target = capture_masks[capture]
            for metric, raw_bound in bounds_by_metric.items():
                bound = cast(Mapping[str, object], raw_bound)
                low_value = bound.get("low")
                high_value = bound.get("high")
                sample_low = (
                    float(cast(float, low_value)) if low_value is not None else None
                )
                sample_high = (
                    float(cast(float, high_value)) if high_value is not None else None
                )
                _auto_bound_flags(
                    metric=metric,
                    values=values[metric],
                    target=target,
                    low=sample_low,
                    high=sample_high,
                    flags=flags,
                )
        warnings.extend(provenance["warnings"])
        parameters = {
            "minP": 0.01,
            "maxP": 0.99,
            "nMads": float(n_mads),
            "minCellsPerSample": int(min_cells_per_sample),
            "madScale": float(provenance["mad_scale"]),
            "metricPolicies": provenance["metric_policies"],
            "sampleSizes": provenance["sample_sizes"],
            "skipReasons": provenance["skip_reasons"],
            "resolvedBounds": provenance["resolved_bounds"],
        }

    capture_sizes = {name: int(mask.sum()) for name, mask in captures}
    retained_by_capture = {name: int((mask & keep).sum()) for name, mask in captures}
    comparisons = (
        _with_capture_retention(
            _global_capture_comparisons(values, active_mask, captures),
            captures,
            keep,
        )
        if captures
        else ()
    )
    failed = tuple(
        comparison.capture
        for comparison in comparisons
        if comparison.wholeCaptureFailure
    )
    if failed:
        warnings.append(
            "Captures failed at least two independent global QC axes: "
            + ", ".join(failed)
        )
    return AutoFilterProjection(
        action=action,
        keep=keep,
        flags=flags,
        parameters=parameters,
        captureSizes=capture_sizes,
        retainedByCapture=retained_by_capture,
        captureComparisons=comparisons,
        failedCaptureCandidates=failed,
        warnings=tuple(warnings),
    )


def offered_registered_qc_profiles(
    *,
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
    capture_labels: np.ndarray | None = None,
    grouping_proven: bool = False,
    min_cells_per_capture: int = 20,
    pooled_reference_captures: tuple[str, ...] | None = None,
) -> list[RegisteredQcProjection]:
    """Build only profiles whose deterministic eligibility gates pass."""
    projections = [
        project_registered_qc_profile(
            "retainWithFlags",
            values_by_metric=values_by_metric,
            active=active,
            capture_labels=capture_labels,
            grouping_proven=grouping_proven,
        )
    ]
    try:
        global_projection = project_registered_qc_profile(
            "globalMad5",
            values_by_metric=values_by_metric,
            active=active,
            capture_labels=capture_labels,
            grouping_proven=grouping_proven,
        )
    except ValueError:
        return projections
    projections.append(global_projection)
    if capture_labels is None or not grouping_proven:
        return projections
    for profile in ("captureMad5", "captureMad3Sensitivity"):
        try:
            projection = project_registered_qc_profile(
                profile,
                values_by_metric=values_by_metric,
                active=active,
                capture_labels=capture_labels,
                grouping_proven=True,
                min_cells_per_capture=min_cells_per_capture,
            )
        except ValueError:
            continue
        projections.append(projection)
    if pooled_reference_captures:
        try:
            projection = project_registered_qc_profile(
                "pooledReferenceMad5",
                values_by_metric=values_by_metric,
                active=active,
                capture_labels=capture_labels,
                grouping_proven=True,
                min_cells_per_capture=min_cells_per_capture,
                pooled_reference_captures=pooled_reference_captures,
            )
        except ValueError:
            pass
        else:
            projections.append(projection)
    return projections


__all__ = [
    "REGISTERED_CELL_QC_PROFILES",
    "AutoFilterAction",
    "AutoFilterProjection",
    "CaptureQcComparison",
    "QcMetricRole",
    "RegisteredCellQcProfile",
    "RegisteredQcProjection",
    "RegisteredQcThreshold",
    "offered_registered_qc_profiles",
    "project_auto_filter_profile",
    "project_registered_qc_profile",
    "qc_metric_execution_name",
    "registered_qc_metric_role",
]
