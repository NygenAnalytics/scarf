import hashlib
import json
import math
import re
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import zarr
from zarr.abc.store import Store
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.sync import collect_aiterator, sync
from zarr.storage import (
    LocalStore,
    MemoryStore,
    ObjectStore,
    StorePath,
    WrapperStore,
)

from .artifacts import require_complete_artifact
from .refs import ArtifactRef
from .types import as_zarr_group


PIPELINE_RUNS_PATH = "pipeline/runs"
_PIPELINE_LABEL_CLAIMS_PATH = f"{PIPELINE_RUNS_PATH}/.label-claims"
_PIPELINE_LABEL_CLAIMS_NAME = ".label-claims"

type PipelineRunStatus = Literal["running", "completed", "failed", "interrupted"]
type PipelineStageStatus = Literal[
    "running", "completed", "skipped", "failed", "interrupted"
]
type PipelineAxis = Literal["cells", "features"]
type PipelineReportFormat = Literal["dict", "markdown"]
type ArtifactPlanDisposition = Literal["created", "reused"]

_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_FIELDS = frozenset(
    {
        "runId",
        "recipe",
        "requestedLabel",
        "label",
        "assay",
        "startedAtNs",
        "finishedAtNs",
        "status",
        "complete",
        "scarfVersion",
        "config",
        "stageOrder",
        "outputs",
        "fields",
        "error",
        "interruption",
    }
)
_STAGE_FIELDS = frozenset(
    {
        "stage",
        "ordinal",
        "startedAtNs",
        "finishedAtNs",
        "status",
        "complete",
        "outputs",
        "plans",
        "metrics",
        "error",
        "interruption",
    }
)
_OUTPUT_FIELDS = frozenset({"key", "artifact"})
_STAGE_OUTPUT_FIELDS = frozenset({"outputKey", "artifact", "reused"})
_PLAN_FIELDS = frozenset({"operation", "ref", "disposition"})
_FIELD_DESCRIPTOR_FIELDS = frozenset(
    {
        "key",
        "axis",
        "artifact",
        "sourceValue",
        "valueIndex",
        "dtype",
        "fill",
        "missingMask",
        "display",
    }
)
_ERROR_FIELDS = frozenset({"type", "message"})
_INTERRUPTION_FIELDS = frozenset(
    {"kind", "message", "requestedAtNs", "signalNumber", "signalName"}
)
_METRIC_FIELDS = frozenset(
    {
        "wallSeconds",
        "rssBaselineBytes",
        "rssPeakBytes",
        "rssIncrementalPeakBytes",
        "sampleIntervalSeconds",
        "sampleCount",
        "samplingErrorCount",
        "rssUnavailableReason",
    }
)
_LABEL_CLAIM_FIELDS = frozenset({"label", "runId"})
_LABEL_CLAIM_KEY_PATTERN = re.compile(
    r"^(?P<digest>[0-9a-f]{64})/(?P<predecessor>head|[0-9a-f]{64})\.json$"
)


def new_pipeline_run_id() -> str:
    """Return a random storage identity for a pipeline invocation."""

    return secrets.token_hex(32)


def pipeline_run_path(run_id: str) -> str:
    _validate_run_id(run_id)
    return f"{PIPELINE_RUNS_PATH}/{run_id}"


def pipeline_stage_path(run_id: str, ordinal: int) -> str:
    _validate_run_id(run_id)
    _validate_non_negative_int(ordinal, "ordinal")
    return f"{pipeline_run_path(run_id)}/stages/{ordinal}"


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must be a 64-character lowercase hex token")


def _validate_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _validate_nullable_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _validate_non_empty_string(value, name)


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _validate_nullable_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _validate_non_negative_int(value, name)


def _validate_nullable_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(value, name)


def _validate_non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a non-negative number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return resolved


def _validate_positive_float(value: Any, name: str) -> float:
    resolved = _validate_non_negative_float(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_mapping(
    value: Any,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    result = _mapping(value, name)
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"extra {extra}")
        raise ValueError(
            f"{name} fields do not match the contract: {', '.join(details)}"
        )
    return result


def _json_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} mapping keys must be strings")
            result[key] = _json_value(item, f"{name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [
            _json_value(item, f"{name}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    result = _json_value(_mapping(value, name), name)
    assert isinstance(result, dict)
    return result


def _artifact_ref(value: Any, name: str) -> ArtifactRef:
    try:
        return ArtifactRef.from_dict(_mapping(value, name))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid ArtifactRef") from exc


def _raise_type(message: str) -> Any:
    raise TypeError(message)


@dataclass(frozen=True, slots=True)
class PipelineErrorRecord:
    type: str
    message: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.type, "error type")
        if not isinstance(self.message, str):
            raise TypeError("error message must be a string")
        if len(self.type) > 128:
            raise ValueError("error type must be at most 128 characters")
        if len(self.message) > 512:
            raise ValueError("error message must be at most 512 characters")

    @classmethod
    def from_exception(cls, error: BaseException) -> "PipelineErrorRecord":
        error_type = type(error).__name__[:128] or "Exception"
        message = str(error)
        if len(message) > 512:
            message = f"{message[:509]}..."
        return cls(type=error_type, message=message)

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "message": self.message}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineErrorRecord":
        raw = _exact_mapping(value, _ERROR_FIELDS, "pipeline error")
        return cls(
            type=_validate_non_empty_string(raw["type"], "error type"),
            message=(
                raw["message"]
                if isinstance(raw["message"], str)
                else _raise_type("error message must be a string")
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineInterruptionRecord:
    kind: str
    message: str
    requested_at_ns: int
    signal_number: int | None = None
    signal_name: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.kind, "interruption kind")
        _validate_non_empty_string(self.message, "interruption message")
        _validate_positive_int(self.requested_at_ns, "interruption requested_at_ns")
        _validate_nullable_non_negative_int(
            self.signal_number,
            "interruption signal_number",
        )
        _validate_nullable_string(self.signal_name, "interruption signal_name")
        if (self.signal_number is None) != (self.signal_name is None):
            raise ValueError("interruption signal number and name must appear together")
        if len(self.kind) > 128 or len(self.message) > 512:
            raise ValueError("interruption kind or message is too long")

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "kind": self.kind,
            "message": self.message,
            "requestedAtNs": self.requested_at_ns,
            "signalNumber": self.signal_number,
            "signalName": self.signal_name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineInterruptionRecord":
        raw = _exact_mapping(value, _INTERRUPTION_FIELDS, "pipeline interruption")
        return cls(
            kind=_validate_non_empty_string(raw["kind"], "interruption kind"),
            message=_validate_non_empty_string(
                raw["message"],
                "interruption message",
            ),
            requested_at_ns=_validate_positive_int(
                raw["requestedAtNs"],
                "interruption requestedAtNs",
            ),
            signal_number=_validate_nullable_non_negative_int(
                raw["signalNumber"],
                "interruption signalNumber",
            ),
            signal_name=_validate_nullable_string(
                raw["signalName"],
                "interruption signalName",
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineOutputRecord:
    key: str
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.key, "output key")
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("output artifact must be an ArtifactRef")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "artifact": self.artifact.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineOutputRecord":
        raw = _exact_mapping(value, _OUTPUT_FIELDS, "pipeline output")
        return cls(
            key=_validate_non_empty_string(raw["key"], "output key"),
            artifact=_artifact_ref(raw["artifact"], "output artifact"),
        )


@dataclass(frozen=True, slots=True)
class PipelineStageOutputRecord:
    output_key: str
    artifact: ArtifactRef
    reused: bool

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.output_key, "stage output key")
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("stage output artifact must be an ArtifactRef")
        _validate_bool(self.reused, "stage output reused")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputKey": self.output_key,
            "artifact": self.artifact.to_dict(),
            "reused": self.reused,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineStageOutputRecord":
        raw = _exact_mapping(value, _STAGE_OUTPUT_FIELDS, "pipeline stage output")
        return cls(
            output_key=_validate_non_empty_string(raw["outputKey"], "output key"),
            artifact=_artifact_ref(raw["artifact"], "stage output artifact"),
            reused=_validate_bool(raw["reused"], "stage output reused"),
        )


@dataclass(frozen=True, slots=True)
class PipelinePlanRecord:
    operation: str
    ref: ArtifactRef
    disposition: ArtifactPlanDisposition

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.operation, "plan operation")
        if not isinstance(self.ref, ArtifactRef):
            raise TypeError("plan ref must be an ArtifactRef")
        if self.disposition not in {"created", "reused"}:
            raise ValueError(f"Invalid artifact plan disposition: {self.disposition!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "ref": self.ref.to_dict(),
            "disposition": self.disposition,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelinePlanRecord":
        raw = _exact_mapping(value, _PLAN_FIELDS, "pipeline artifact plan")
        disposition = raw["disposition"]
        if disposition not in {"created", "reused"}:
            raise ValueError(f"Invalid artifact plan disposition: {disposition!r}")
        return cls(
            operation=_validate_non_empty_string(raw["operation"], "plan operation"),
            ref=_artifact_ref(raw["ref"], "plan ref"),
            disposition=disposition,
        )


@dataclass(frozen=True, slots=True)
class PipelineStageMetrics:
    wall_seconds: float
    rss_baseline_bytes: int | None
    rss_peak_bytes: int | None
    rss_incremental_peak_bytes: int | None
    sample_interval_seconds: float
    sample_count: int
    sampling_error_count: int
    rss_unavailable_reason: str | None

    def __post_init__(self) -> None:
        _validate_non_negative_float(self.wall_seconds, "metrics wall_seconds")
        _validate_nullable_non_negative_int(
            self.rss_baseline_bytes,
            "metrics rss_baseline_bytes",
        )
        _validate_nullable_non_negative_int(
            self.rss_peak_bytes,
            "metrics rss_peak_bytes",
        )
        _validate_nullable_non_negative_int(
            self.rss_incremental_peak_bytes,
            "metrics rss_incremental_peak_bytes",
        )
        _validate_positive_float(
            self.sample_interval_seconds,
            "metrics sample_interval_seconds",
        )
        _validate_non_negative_int(self.sample_count, "metrics sample_count")
        _validate_non_negative_int(
            self.sampling_error_count,
            "metrics sampling_error_count",
        )
        _validate_nullable_string(
            self.rss_unavailable_reason,
            "metrics rss_unavailable_reason",
        )
        if self.rss_peak_bytes is None:
            if self.rss_baseline_bytes is not None:
                raise ValueError("RSS baseline cannot exist without an RSS peak")
            if self.rss_unavailable_reason is None:
                raise ValueError("Unavailable RSS requires an explicit reason")
        elif self.rss_unavailable_reason is not None:
            raise ValueError("Available RSS cannot have an unavailable reason")

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "wallSeconds": self.wall_seconds,
            "rssBaselineBytes": self.rss_baseline_bytes,
            "rssPeakBytes": self.rss_peak_bytes,
            "rssIncrementalPeakBytes": self.rss_incremental_peak_bytes,
            "sampleIntervalSeconds": self.sample_interval_seconds,
            "sampleCount": self.sample_count,
            "samplingErrorCount": self.sampling_error_count,
            "rssUnavailableReason": self.rss_unavailable_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineStageMetrics":
        raw = _exact_mapping(value, _METRIC_FIELDS, "pipeline stage metrics")
        return cls(
            wall_seconds=_validate_non_negative_float(
                raw["wallSeconds"],
                "metrics wallSeconds",
            ),
            rss_baseline_bytes=_validate_nullable_non_negative_int(
                raw["rssBaselineBytes"],
                "metrics rssBaselineBytes",
            ),
            rss_peak_bytes=_validate_nullable_non_negative_int(
                raw["rssPeakBytes"],
                "metrics rssPeakBytes",
            ),
            rss_incremental_peak_bytes=_validate_nullable_non_negative_int(
                raw["rssIncrementalPeakBytes"],
                "metrics rssIncrementalPeakBytes",
            ),
            sample_interval_seconds=_validate_positive_float(
                raw["sampleIntervalSeconds"],
                "metrics sampleIntervalSeconds",
            ),
            sample_count=_validate_non_negative_int(
                raw["sampleCount"],
                "metrics sampleCount",
            ),
            sampling_error_count=_validate_non_negative_int(
                raw["samplingErrorCount"],
                "metrics samplingErrorCount",
            ),
            rss_unavailable_reason=_validate_nullable_string(
                raw["rssUnavailableReason"],
                "metrics rssUnavailableReason",
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineFieldDescriptor:
    key: str
    axis: PipelineAxis
    artifact: ArtifactRef
    source_value: str
    value_index: int | None
    dtype: str
    fill: str | int | float | bool | None
    missing_mask: str | None
    display: dict[str, Any] | None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.key, "field key")
        if self.axis not in {"cells", "features"}:
            raise ValueError(f"Invalid pipeline field axis: {self.axis!r}")
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("field artifact must be an ArtifactRef")
        _validate_non_empty_string(self.source_value, "field source_value")
        if self.value_index is not None:
            _validate_non_negative_int(self.value_index, "field value_index")
        _validate_non_empty_string(self.dtype, "field dtype")
        if isinstance(self.fill, float) and not math.isfinite(self.fill):
            raise ValueError("field fill uses the string 'nan' for NaN")
        if not isinstance(self.fill, str | int | float | bool | type(None)):
            raise TypeError("field fill must be a JSON scalar")
        _validate_nullable_string(self.missing_mask, "field missing_mask")
        if self.display is not None:
            object.__setattr__(
                self,
                "display",
                _json_mapping(self.display, "field display"),
            )

    @property
    def fill_value(self) -> str | int | float | bool | None:
        return float("nan") if self.fill == "nan" else self.fill

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "axis": self.axis,
            "artifact": self.artifact.to_dict(),
            "sourceValue": self.source_value,
            "valueIndex": self.value_index,
            "dtype": self.dtype,
            "fill": self.fill,
            "missingMask": self.missing_mask,
            "display": None if self.display is None else dict(self.display),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineFieldDescriptor":
        raw = _exact_mapping(
            value,
            _FIELD_DESCRIPTOR_FIELDS,
            "pipeline field descriptor",
        )
        axis = raw["axis"]
        if axis not in {"cells", "features"}:
            raise ValueError(f"Invalid pipeline field axis: {axis!r}")
        value_index = raw["valueIndex"]
        if value_index is not None:
            value_index = _validate_non_negative_int(value_index, "field valueIndex")
        fill = raw["fill"]
        if not isinstance(fill, str | int | float | bool | type(None)):
            raise TypeError("field fill must be a JSON scalar")
        display = raw["display"]
        if display is not None:
            display = _json_mapping(display, "field display")
        return cls(
            key=_validate_non_empty_string(raw["key"], "field key"),
            axis=axis,
            artifact=_artifact_ref(raw["artifact"], "field artifact"),
            source_value=_validate_non_empty_string(
                raw["sourceValue"],
                "field sourceValue",
            ),
            value_index=value_index,
            dtype=_validate_non_empty_string(raw["dtype"], "field dtype"),
            fill=fill,
            missing_mask=_validate_nullable_string(
                raw["missingMask"],
                "field missingMask",
            ),
            display=display,
        )


@dataclass(frozen=True, slots=True)
class PipelineStageRecord:
    stage: str
    ordinal: int
    started_at_ns: int
    finished_at_ns: int | None
    status: PipelineStageStatus
    complete: bool
    outputs: tuple[PipelineStageOutputRecord, ...]
    plans: tuple[PipelinePlanRecord, ...]
    metrics: PipelineStageMetrics | None
    error: PipelineErrorRecord | None
    interruption: PipelineInterruptionRecord | None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.stage, "stage")
        _validate_non_negative_int(self.ordinal, "stage ordinal")
        _validate_positive_int(self.started_at_ns, "stage started_at_ns")
        if self.finished_at_ns is not None:
            _validate_positive_int(self.finished_at_ns, "stage finished_at_ns")
            if self.finished_at_ns < self.started_at_ns:
                raise ValueError("stage finished_at_ns cannot precede started_at_ns")
        if self.status not in {
            "running",
            "completed",
            "skipped",
            "failed",
            "interrupted",
        }:
            raise ValueError(f"Invalid pipeline stage status: {self.status!r}")
        _validate_bool(self.complete, "stage complete")
        if not isinstance(self.outputs, tuple) or any(
            not isinstance(output, PipelineStageOutputRecord) for output in self.outputs
        ):
            raise TypeError("stage outputs must be PipelineStageOutputRecord values")
        if not isinstance(self.plans, tuple) or any(
            not isinstance(plan, PipelinePlanRecord) for plan in self.plans
        ):
            raise TypeError("stage plans must be PipelinePlanRecord values")
        output_keys = [output.output_key for output in self.outputs]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError("stage output keys must be unique")
        if self.status == "running":
            if self.complete or self.finished_at_ns is not None:
                raise ValueError("running stages cannot be complete or finished")
            if self.outputs or self.plans or self.metrics is not None:
                raise ValueError("running stages cannot contain receipts")
            if self.error is not None or self.interruption is not None:
                raise ValueError("running stages cannot contain terminal details")
            return
        if self.finished_at_ns is None or self.metrics is None:
            raise ValueError("terminal stages require finish time and metrics")
        if self.status == "skipped" and self.outputs:
            raise ValueError("skipped stages cannot claim outputs")
        if self.status == "failed":
            if self.error is None or self.interruption is not None:
                raise ValueError("failed stages require only an error")
        elif self.status == "interrupted":
            if self.interruption is None or self.error is not None:
                raise ValueError("interrupted stages require only interruption details")
            if self.outputs:
                raise ValueError("interrupted stages cannot claim complete outputs")
        elif self.error is not None or self.interruption is not None:
            raise ValueError("successful stages cannot contain terminal details")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ordinal": self.ordinal,
            "startedAtNs": self.started_at_ns,
            "finishedAtNs": self.finished_at_ns,
            "status": self.status,
            "complete": self.complete,
            "outputs": [output.to_dict() for output in self.outputs],
            "plans": [plan.to_dict() for plan in self.plans],
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
            "interruption": (
                None if self.interruption is None else self.interruption.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineStageRecord":
        raw = _exact_mapping(value, _STAGE_FIELDS, "pipeline stage")
        status = raw["status"]
        if status not in {
            "running",
            "completed",
            "skipped",
            "failed",
            "interrupted",
        }:
            raise ValueError(f"Invalid pipeline stage status: {status!r}")
        outputs = raw["outputs"]
        plans = raw["plans"]
        if not isinstance(outputs, Sequence) or isinstance(outputs, str | bytes):
            raise TypeError("stage outputs must be a sequence")
        if not isinstance(plans, Sequence) or isinstance(plans, str | bytes):
            raise TypeError("stage plans must be a sequence")
        metrics = raw["metrics"]
        error = raw["error"]
        interruption = raw["interruption"]
        return cls(
            stage=_validate_non_empty_string(raw["stage"], "stage"),
            ordinal=_validate_non_negative_int(raw["ordinal"], "stage ordinal"),
            started_at_ns=_validate_positive_int(
                raw["startedAtNs"],
                "stage startedAtNs",
            ),
            finished_at_ns=_validate_nullable_positive_int(
                raw["finishedAtNs"],
                "stage finishedAtNs",
            ),
            status=status,
            complete=_validate_bool(raw["complete"], "stage complete"),
            outputs=tuple(
                PipelineStageOutputRecord.from_dict(_mapping(item, "stage output"))
                for item in outputs
            ),
            plans=tuple(
                PipelinePlanRecord.from_dict(_mapping(item, "stage plan"))
                for item in plans
            ),
            metrics=(
                None
                if metrics is None
                else PipelineStageMetrics.from_dict(_mapping(metrics, "stage metrics"))
            ),
            error=(
                None
                if error is None
                else PipelineErrorRecord.from_dict(_mapping(error, "stage error"))
            ),
            interruption=(
                None
                if interruption is None
                else PipelineInterruptionRecord.from_dict(
                    _mapping(interruption, "stage interruption")
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineRunRecord:
    run_id: str
    recipe: str
    requested_label: str | None
    label: str | None
    assay: str
    started_at_ns: int
    finished_at_ns: int | None
    status: PipelineRunStatus
    complete: bool
    scarf_version: str
    config: dict[str, Any]
    stage_order: tuple[str, ...]
    outputs: tuple[PipelineOutputRecord, ...]
    fields: tuple[PipelineFieldDescriptor, ...]
    error: PipelineErrorRecord | None
    interruption: PipelineInterruptionRecord | None

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_non_empty_string(self.recipe, "recipe")
        _validate_nullable_string(self.requested_label, "requested_label")
        _validate_nullable_string(self.label, "label")
        _validate_non_empty_string(self.assay, "assay")
        _validate_positive_int(self.started_at_ns, "started_at_ns")
        if self.finished_at_ns is not None:
            _validate_positive_int(self.finished_at_ns, "finished_at_ns")
            if self.finished_at_ns < self.started_at_ns:
                raise ValueError("finished_at_ns cannot precede started_at_ns")
        if self.status not in {"running", "completed", "failed", "interrupted"}:
            raise ValueError(f"Invalid pipeline run status: {self.status!r}")
        _validate_bool(self.complete, "run complete")
        _validate_non_empty_string(self.scarf_version, "scarf_version")
        object.__setattr__(
            self, "config", _json_mapping(self.config, "pipeline config")
        )
        if not isinstance(self.stage_order, tuple) or not self.stage_order:
            raise TypeError("stage_order must be a non-empty tuple")
        for stage in self.stage_order:
            _validate_non_empty_string(stage, "stage_order item")
        if len(self.stage_order) != len(set(self.stage_order)):
            raise ValueError("stage_order values must be unique")
        if not isinstance(self.outputs, tuple) or any(
            not isinstance(output, PipelineOutputRecord) for output in self.outputs
        ):
            raise TypeError("outputs must be PipelineOutputRecord values")
        if not isinstance(self.fields, tuple) or any(
            not isinstance(field, PipelineFieldDescriptor) for field in self.fields
        ):
            raise TypeError("fields must be PipelineFieldDescriptor values")
        output_keys = [output.key for output in self.outputs]
        field_keys = [(field.axis, field.key) for field in self.fields]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError("run output keys must be unique")
        if len(field_keys) != len(set(field_keys)):
            raise ValueError("run field keys must be unique within each axis")
        if self.status == "running":
            if self.complete or self.finished_at_ns is not None:
                raise ValueError("running runs cannot be complete or finished")
            if self.label is not None or self.outputs or self.fields:
                raise ValueError("running runs cannot contain terminal results")
            if self.error is not None or self.interruption is not None:
                raise ValueError("running runs cannot contain terminal details")
            return
        if self.finished_at_ns is None:
            raise ValueError("terminal runs require finished_at_ns")
        if self.status == "completed":
            if self.error is not None or self.interruption is not None:
                raise ValueError("completed runs cannot contain terminal details")
            if self.label != self.requested_label:
                raise ValueError("completed run label must equal requested_label")
        elif self.status == "failed":
            if self.label is not None or self.outputs or self.fields:
                raise ValueError("failed runs cannot expose results or a label")
            if self.error is None or self.interruption is not None:
                raise ValueError("failed runs require only an error")
        else:
            if self.label is not None or self.outputs or self.fields:
                raise ValueError("interrupted runs cannot expose results or a label")
            if self.interruption is None or self.error is not None:
                raise ValueError("interrupted runs require only interruption details")

    @property
    def successfully_completed(self) -> bool:
        return self.status == "completed" and self.complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "recipe": self.recipe,
            "requestedLabel": self.requested_label,
            "label": self.label,
            "assay": self.assay,
            "startedAtNs": self.started_at_ns,
            "finishedAtNs": self.finished_at_ns,
            "status": self.status,
            "complete": self.complete,
            "scarfVersion": self.scarf_version,
            "config": _json_mapping(self.config, "pipeline config"),
            "stageOrder": list(self.stage_order),
            "outputs": [output.to_dict() for output in self.outputs],
            "fields": [field.to_dict() for field in self.fields],
            "error": None if self.error is None else self.error.to_dict(),
            "interruption": (
                None if self.interruption is None else self.interruption.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineRunRecord":
        raw = _exact_mapping(value, _RUN_FIELDS, "pipeline run")
        status = raw["status"]
        if status not in {"running", "completed", "failed", "interrupted"}:
            raise ValueError(f"Invalid pipeline run status: {status!r}")
        stage_order = raw["stageOrder"]
        outputs = raw["outputs"]
        fields = raw["fields"]
        if not isinstance(stage_order, Sequence) or isinstance(
            stage_order,
            str | bytes,
        ):
            raise TypeError("run stageOrder must be a sequence")
        if not isinstance(outputs, Sequence) or isinstance(outputs, str | bytes):
            raise TypeError("run outputs must be a sequence")
        if not isinstance(fields, Sequence) or isinstance(fields, str | bytes):
            raise TypeError("run fields must be a sequence")
        error = raw["error"]
        interruption = raw["interruption"]
        return cls(
            run_id=(
                raw["runId"]
                if isinstance(raw["runId"], str)
                else _raise_type("runId must be a string")
            ),
            recipe=_validate_non_empty_string(raw["recipe"], "recipe"),
            requested_label=_validate_nullable_string(
                raw["requestedLabel"],
                "requestedLabel",
            ),
            label=_validate_nullable_string(raw["label"], "label"),
            assay=_validate_non_empty_string(raw["assay"], "assay"),
            started_at_ns=_validate_positive_int(
                raw["startedAtNs"],
                "startedAtNs",
            ),
            finished_at_ns=_validate_nullable_positive_int(
                raw["finishedAtNs"],
                "finishedAtNs",
            ),
            status=status,
            complete=_validate_bool(raw["complete"], "run complete"),
            scarf_version=_validate_non_empty_string(
                raw["scarfVersion"],
                "scarfVersion",
            ),
            config=_json_mapping(raw["config"], "pipeline config"),
            stage_order=tuple(
                _validate_non_empty_string(stage, "stageOrder item")
                for stage in stage_order
            ),
            outputs=tuple(
                PipelineOutputRecord.from_dict(_mapping(item, "run output"))
                for item in outputs
            ),
            fields=tuple(
                PipelineFieldDescriptor.from_dict(_mapping(item, "run field"))
                for item in fields
            ),
            error=(
                None
                if error is None
                else PipelineErrorRecord.from_dict(_mapping(error, "run error"))
            ),
            interruption=(
                None
                if interruption is None
                else PipelineInterruptionRecord.from_dict(
                    _mapping(interruption, "run interruption")
                )
            ),
        )


def _get_group(root: zarr.Group, path: str, name: str) -> zarr.Group:
    if path not in root:
        raise KeyError(f"{name} does not exist: {path}")
    return as_zarr_group(root[path], name=path)


def _ensure_group(root: zarr.Group, path: str) -> zarr.Group:
    current = root
    current_path = ""
    for part in path.split("/"):
        current_path = f"{current_path}/{part}".strip("/")
        if part in current:
            current = as_zarr_group(current[part], name=current_path)
        else:
            current = current.create_group(part)
    return current


def _read_attrs(group: zarr.Group) -> dict[str, Any]:
    return {str(key): value for key, value in group.attrs.items()}


def _write_terminal_attrs(group: zarr.Group, value: Mapping[str, Any]) -> None:
    """Commit a terminal record with complete as the final write."""

    payload = dict(value)
    final_complete = payload.pop("complete")
    if final_complete is not True:
        raise ValueError("terminal writes require complete=True")
    group.attrs["complete"] = False
    group.attrs.update(payload)
    group.attrs["complete"] = True


def create_pipeline_run_record(
    root: zarr.Group,
    *,
    recipe: str,
    requested_label: str | None,
    assay: str,
    config: Mapping[str, Any],
    stage_order: Sequence[str],
    scarf_version: str,
    run_id: str | None = None,
    started_at_ns: int | None = None,
) -> PipelineRunRecord:
    """Create a durable running record before pipeline computation starts."""

    if requested_label is not None:
        ensure_pipeline_label_claimable(root, requested_label)
    record = PipelineRunRecord(
        run_id=new_pipeline_run_id() if run_id is None else run_id,
        recipe=recipe,
        requested_label=requested_label,
        label=None,
        assay=assay,
        started_at_ns=time.time_ns() if started_at_ns is None else started_at_ns,
        finished_at_ns=None,
        status="running",
        complete=False,
        scarf_version=scarf_version,
        config=_json_mapping(config, "pipeline config"),
        stage_order=tuple(stage_order),
        outputs=(),
        fields=(),
        error=None,
        interruption=None,
    )
    path = pipeline_run_path(record.run_id)
    if path in root:
        raise FileExistsError(f"Pipeline run already exists: {record.run_id}")
    group = _ensure_group(root, PIPELINE_RUNS_PATH).create_group(record.run_id)
    group.attrs.update(record.to_dict())
    group.create_group("stages")
    return record


def load_pipeline_run_record(root: zarr.Group, run_id: str) -> PipelineRunRecord:
    path = pipeline_run_path(run_id)
    group = _get_group(root, path, "Pipeline run")
    if "stages" not in group:
        raise ValueError(f"Pipeline run {run_id} has no stages group")
    as_zarr_group(group["stages"], name=f"{path}/stages")
    record = PipelineRunRecord.from_dict(_read_attrs(group))
    if record.run_id != run_id:
        raise ValueError(f"Pipeline run path {run_id!r} contains {record.run_id!r}")
    return record


def _pipeline_label_claim_path(
    root: zarr.Group,
    label: str,
    predecessor: str,
) -> StorePath:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return root.store_path / (
        f"{_PIPELINE_LABEL_CLAIMS_PATH}/{digest}/{predecessor}.json"
    )


def _group_store_prefix(root: zarr.Group) -> str:
    path = root.store_path.path.rstrip("/")
    return f"{path}/" if path else ""


def _pipeline_label_claim_namespaces(root: zarr.Group) -> tuple[str, ...]:
    namespaces: list[str] = []

    def visit(group: zarr.Group, relative_path: str) -> None:
        if "pipeline" in group:
            pipeline = group["pipeline"]
            if isinstance(pipeline, zarr.Group) and "runs" in pipeline:
                runs = pipeline["runs"]
                if isinstance(runs, zarr.Group) and _PIPELINE_LABEL_CLAIMS_NAME in runs:
                    container = runs[_PIPELINE_LABEL_CLAIMS_NAME]
                    if (
                        not isinstance(container, zarr.Array)
                        or tuple(container.shape) != (0,)
                        or container.dtype != "uint8"
                    ):
                        raise ValueError(
                            "Pipeline label claim container is incompatible"
                        )
                    namespace = f"{relative_path}/{_PIPELINE_LABEL_CLAIMS_PATH}"
                    namespaces.append(namespace.lstrip("/"))
        for name in group.group_keys():
            if name == "pipeline":
                continue
            child_path = f"{relative_path}/{name}" if relative_path else name
            child = group[name]
            assert isinstance(child, zarr.Group)
            visit(child, child_path)

    visit(root, "")
    return tuple(namespaces)


def _store_supports_atomic_label_claims(store: Store) -> bool:
    visited: set[int] = set()
    while isinstance(store, WrapperStore):
        identity = id(store)
        if identity in visited:
            return False
        visited.add(identity)
        store = store._store
    return isinstance(store, LocalStore | MemoryStore | ObjectStore)


def _pipeline_label_claim_bytes(label: str, run_id: str) -> Buffer:
    payload = json.dumps(
        {"label": label, "runId": run_id},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return default_buffer_prototype().buffer.from_bytes(payload)


def _ensure_pipeline_label_claim_container(root: zarr.Group) -> None:
    # Mark the raw claim namespace as a Zarr child so hierarchy walks remain
    # warning-free. Its zero-length payload is never read or written.
    runs = _get_group(root, PIPELINE_RUNS_PATH, "Pipeline runs")
    if _PIPELINE_LABEL_CLAIMS_NAME not in runs:
        try:
            runs.create_array(
                _PIPELINE_LABEL_CLAIMS_NAME,
                shape=(0,),
                dtype="uint8",
            )
        except zarr.errors.ContainsArrayError:
            pass
    _validate_pipeline_label_claim_container(root)


def _validate_pipeline_label_claim_container(root: zarr.Group) -> None:
    """Validate an existing raw claim namespace without creating it."""

    try:
        runs = _get_group(root, PIPELINE_RUNS_PATH, "Pipeline runs")
    except KeyError:
        return
    if _PIPELINE_LABEL_CLAIMS_NAME not in runs:
        return
    container = runs[_PIPELINE_LABEL_CLAIMS_NAME]
    if (
        not isinstance(container, zarr.Array)
        or tuple(container.shape) != (0,)
        or container.dtype != "uint8"
    ):
        raise ValueError("Pipeline label claim container is incompatible")


def _decode_pipeline_label_claim(stored: Buffer, label: str) -> str:
    try:
        value = json.loads(stored.to_bytes().decode("utf-8"))
        raw = _exact_mapping(value, _LABEL_CLAIM_FIELDS, "pipeline label claim")
        stored_label = _validate_non_empty_string(raw["label"], "claim label")
        run_id = _validate_non_empty_string(raw["runId"], "claim runId")
        _validate_run_id(run_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Pipeline label {label!r} has an invalid durable claim"
        ) from exc
    if stored_label != label:
        raise ValueError(
            f"Pipeline label {label!r} collides with another durable claim"
        )
    return run_id


def _read_pipeline_label_claim(path: StorePath, label: str) -> str:
    stored = sync(path.get())
    if stored is None:
        raise ValueError(f"Pipeline label {label!r} has an incomplete durable claim")
    return _decode_pipeline_label_claim(stored, label)


def _pipeline_label_claim_owner(root: zarr.Group, label: str) -> str | None:
    """Return the tail owner of an immutable label-claim chain."""

    predecessor = "head"
    visited: set[str] = set()
    while True:
        path = _pipeline_label_claim_path(root, label, predecessor)
        stored = sync(path.get())
        if stored is None:
            return None if predecessor == "head" else predecessor
        owner_id = _decode_pipeline_label_claim(stored, label)
        if owner_id in visited:
            raise ValueError(f"Pipeline label {label!r} has a cyclic durable claim")
        visited.add(owner_id)
        predecessor = owner_id


def _claim_pipeline_label(root: zarr.Group, label: str, run_id: str) -> None:
    """Atomically elect one finalizer while retaining recoverable predecessors."""

    _validate_non_empty_string(label, "label")
    _validate_run_id(run_id)
    if not _store_supports_atomic_label_claims(root.store):
        raise RuntimeError(
            "Pipeline label uniqueness requires a Zarr store with atomic "
            "set_if_not_exists support"
        )
    _ensure_pipeline_label_claim_container(root)

    claim_bytes = _pipeline_label_claim_bytes(label, run_id)
    predecessor = "head"
    visited: set[str] = set()
    while True:
        claim_path = _pipeline_label_claim_path(root, label, predecessor)
        sync(claim_path.set_if_not_exists(claim_bytes))
        owner_id = _read_pipeline_label_claim(claim_path, label)
        if owner_id == run_id:
            return
        if owner_id in visited:
            raise ValueError(f"Pipeline label {label!r} has a cyclic durable claim")
        visited.add(owner_id)
        try:
            owner = load_pipeline_run_record(root, owner_id)
        except KeyError:
            # A claim whose run was removed cannot own a public label. Keep the
            # immutable predecessor and let conditional creation elect its successor.
            predecessor = owner_id
            continue
        if owner.requested_label != label:
            raise ValueError(
                f"Pipeline label {label!r} has a claim from an incompatible run"
            )
        if owner.successfully_completed:
            raise ValueError(
                f"Pipeline label {label!r} is already committed by run {owner_id}"
            )
        if owner.complete and owner.status in {"failed", "interrupted"}:
            predecessor = owner_id
            continue
        raise ValueError(
            f"Pipeline label {label!r} is currently being finalized by run {owner_id}"
        )


def _copy_pipeline_label_claims(
    source: zarr.Group,
    destination: zarr.Group,
) -> None:
    """Copy raw append-only claims in every nested datastore workspace."""

    namespaces = _pipeline_label_claim_namespaces(source)
    source_prefix = _group_store_prefix(source)
    destination_prefix = _group_store_prefix(destination)
    for namespace in namespaces:
        try:
            destination_container = destination[namespace]
        except KeyError as error:
            raise ValueError(
                f"Pipeline label claim container is missing: {namespace}"
            ) from error
        if (
            not isinstance(destination_container, zarr.Array)
            or tuple(destination_container.shape) != (0,)
            or destination_container.dtype != "uint8"
        ):
            raise ValueError(
                f"Pipeline label claim container is incompatible: {namespace}"
            )
        claim_prefix = f"{namespace}/"
        source_claim_prefix = f"{source_prefix}{claim_prefix}"
        claim_keys = sorted(
            collect_aiterator(source.store.list_prefix(source_claim_prefix))
        )
        for source_key in claim_keys:
            if not source_key.startswith(source_claim_prefix):
                continue
            relative_claim = source_key[len(source_claim_prefix) :]
            match = _LABEL_CLAIM_KEY_PATTERN.fullmatch(relative_claim)
            if match is None:
                continue
            relative = f"{claim_prefix}{relative_claim}"
            stored = sync(
                source.store.get(
                    source_key,
                    prototype=default_buffer_prototype(),
                )
            )
            if stored is None:
                raise ValueError(
                    f"Pipeline label claim disappeared during copy: {relative}"
                )
            try:
                value = json.loads(stored.to_bytes().decode("utf-8"))
                raw = _exact_mapping(value, _LABEL_CLAIM_FIELDS, "pipeline label claim")
                label = _validate_non_empty_string(raw["label"], "claim label")
                run_id = _validate_non_empty_string(raw["runId"], "claim runId")
                _validate_run_id(run_id)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Pipeline label claim is invalid: {relative}"
                ) from error
            if hashlib.sha256(label.encode("utf-8")).hexdigest() != match["digest"]:
                raise ValueError(f"Pipeline label claim digest is invalid: {relative}")
            destination_key = f"{destination_prefix}{relative}"
            sync(destination.store.set(destination_key, stored))


def _load_pipeline_stage_record_for_run(
    root: zarr.Group,
    run: PipelineRunRecord,
    ordinal: int,
) -> PipelineStageRecord:
    path = pipeline_stage_path(run.run_id, ordinal)
    record = PipelineStageRecord.from_dict(
        _read_attrs(_get_group(root, path, "Pipeline stage"))
    )
    if record.ordinal != ordinal:
        raise ValueError(f"Pipeline stage path {ordinal} contains {record.ordinal}")
    if ordinal >= len(run.stage_order) or run.stage_order[ordinal] != record.stage:
        raise ValueError("Pipeline stage does not match its run stage order")
    return record


def _load_pipeline_stage_records_for_run(
    root: zarr.Group,
    run: PipelineRunRecord,
) -> tuple[PipelineStageRecord, ...]:
    stages = _get_group(
        root,
        f"{pipeline_run_path(run.run_id)}/stages",
        "Pipeline stages",
    )
    ordinals: list[int] = []
    for name in stages.group_keys():
        if not name.isdigit() or str(int(name)) != name:
            raise ValueError(f"Invalid pipeline stage child name: {name!r}")
        ordinal = int(name)
        if ordinal >= len(run.stage_order):
            raise ValueError(f"Pipeline stage ordinal is out of range: {ordinal}")
        ordinals.append(ordinal)
    return tuple(
        _load_pipeline_stage_record_for_run(root, run, ordinal)
        for ordinal in sorted(ordinals)
    )


def start_pipeline_stage_record(
    root: zarr.Group,
    *,
    run_id: str,
    ordinal: int,
    stage: str,
    started_at_ns: int | None = None,
) -> PipelineStageRecord:
    run = load_pipeline_run_record(root, run_id)
    if run.complete or run.status != "running":
        raise ValueError("Cannot start a stage on a terminal pipeline run")
    _validate_non_negative_int(ordinal, "stage ordinal")
    if ordinal >= len(run.stage_order) or run.stage_order[ordinal] != stage:
        raise ValueError("Stage and ordinal do not match the persisted stage order")
    prior = _load_pipeline_stage_records_for_run(root, run)
    if any(item.ordinal >= ordinal for item in prior):
        raise FileExistsError(f"Pipeline stage {ordinal} already exists")
    if len(prior) != ordinal or any(
        not item.complete or item.status not in {"completed", "skipped"}
        for item in prior
    ):
        raise ValueError("Pipeline stages must start sequentially")
    record = PipelineStageRecord(
        stage=stage,
        ordinal=ordinal,
        started_at_ns=time.time_ns() if started_at_ns is None else started_at_ns,
        finished_at_ns=None,
        status="running",
        complete=False,
        outputs=(),
        plans=(),
        metrics=None,
        error=None,
        interruption=None,
    )
    stages = _get_group(
        root,
        f"{pipeline_run_path(run_id)}/stages",
        "Pipeline stages",
    )
    stages.create_group(str(ordinal)).attrs.update(record.to_dict())
    return record


def load_pipeline_stage_record(
    root: zarr.Group,
    run_id: str,
    ordinal: int,
) -> PipelineStageRecord:
    run = load_pipeline_run_record(root, run_id)
    return _load_pipeline_stage_record_for_run(root, run, ordinal)


def load_pipeline_stage_records(
    root: zarr.Group,
    run_id: str,
) -> tuple[PipelineStageRecord, ...]:
    run = load_pipeline_run_record(root, run_id)
    return _load_pipeline_stage_records_for_run(root, run)


def finish_pipeline_stage_record(
    root: zarr.Group,
    *,
    run_id: str,
    ordinal: int,
    status: Literal["completed", "skipped", "failed", "interrupted"],
    outputs: Sequence[PipelineStageOutputRecord] = (),
    plans: Sequence[PipelinePlanRecord] = (),
    metrics: PipelineStageMetrics,
    error: PipelineErrorRecord | BaseException | None = None,
    interruption: PipelineInterruptionRecord | None = None,
    finished_at_ns: int | None = None,
) -> PipelineStageRecord:
    run = load_pipeline_run_record(root, run_id)
    if run.complete or run.status != "running":
        raise ValueError("Cannot finish a stage on a terminal pipeline run")
    current = _load_pipeline_stage_record_for_run(root, run, ordinal)
    if current.complete or current.status != "running":
        raise ValueError("Pipeline stage is already terminal")
    resolved_outputs = tuple(outputs)
    resolved_plans = tuple(plans)
    for output in resolved_outputs:
        if not isinstance(output, PipelineStageOutputRecord):
            raise TypeError("outputs must contain PipelineStageOutputRecord values")
        require_complete_artifact(root, output.artifact)
    resolved_error = (
        PipelineErrorRecord.from_exception(error)
        if isinstance(error, BaseException)
        else error
    )
    record = PipelineStageRecord(
        stage=current.stage,
        ordinal=current.ordinal,
        started_at_ns=current.started_at_ns,
        finished_at_ns=time.time_ns() if finished_at_ns is None else finished_at_ns,
        status=status,
        complete=True,
        outputs=resolved_outputs,
        plans=resolved_plans,
        metrics=metrics,
        error=resolved_error,
        interruption=interruption,
    )
    _write_terminal_attrs(
        _get_group(root, pipeline_stage_path(run_id, ordinal), "Pipeline stage"),
        record.to_dict(),
    )
    return record


def _validate_successful_stages(root: zarr.Group, run: PipelineRunRecord) -> None:
    stages = _load_pipeline_stage_records_for_run(root, run)
    if len(stages) != len(run.stage_order):
        raise ValueError("A completed run requires every stage record")
    for ordinal, stage in enumerate(stages):
        if (
            stage.ordinal != ordinal
            or stage.stage != run.stage_order[ordinal]
            or not stage.complete
            or stage.status not in {"completed", "skipped"}
        ):
            raise ValueError("A completed run requires terminal successful stages")


def complete_pipeline_run_record(
    root: zarr.Group,
    *,
    run_id: str,
    outputs: Sequence[PipelineOutputRecord],
    fields: Sequence[PipelineFieldDescriptor],
    finished_at_ns: int | None = None,
) -> PipelineRunRecord:
    run = load_pipeline_run_record(root, run_id)
    if run.complete or run.status != "running":
        raise ValueError("Pipeline run is already terminal")
    _validate_successful_stages(root, run)
    resolved_outputs = tuple(outputs)
    resolved_fields = tuple(fields)
    for output in resolved_outputs:
        if not isinstance(output, PipelineOutputRecord):
            raise TypeError("outputs must contain PipelineOutputRecord values")
        require_complete_artifact(root, output.artifact)
    for field in resolved_fields:
        if not isinstance(field, PipelineFieldDescriptor):
            raise TypeError("fields must contain PipelineFieldDescriptor values")
        require_complete_artifact(root, field.artifact)
    if run.requested_label is not None:
        try:
            ensure_pipeline_label_available(
                root,
                run.requested_label,
                exclude_run_id=run.run_id,
            )
            _claim_pipeline_label(root, run.requested_label, run.run_id)
        except (ValueError, RuntimeError) as exc:
            error_type = (
                "PipelineLabelConflict"
                if isinstance(exc, ValueError)
                else "PipelineLabelClaimUnavailable"
            )
            fail_pipeline_run_record(
                root,
                run_id=run.run_id,
                error=PipelineErrorRecord(
                    type=error_type,
                    message=str(exc)[:512],
                ),
                finished_at_ns=finished_at_ns,
            )
            raise
    record = replace(
        run,
        label=run.requested_label,
        finished_at_ns=time.time_ns() if finished_at_ns is None else finished_at_ns,
        status="completed",
        complete=True,
        outputs=resolved_outputs,
        fields=resolved_fields,
        error=None,
        interruption=None,
    )
    _write_terminal_attrs(
        _get_group(root, pipeline_run_path(run_id), "Pipeline run"),
        record.to_dict(),
    )
    return record


def fail_pipeline_run_record(
    root: zarr.Group,
    *,
    run_id: str,
    error: PipelineErrorRecord | BaseException,
    finished_at_ns: int | None = None,
) -> PipelineRunRecord:
    run = load_pipeline_run_record(root, run_id)
    if run.complete:
        raise ValueError("Pipeline run is already terminal")
    resolved_error = (
        PipelineErrorRecord.from_exception(error)
        if isinstance(error, BaseException)
        else error
    )
    record = replace(
        run,
        label=None,
        finished_at_ns=time.time_ns() if finished_at_ns is None else finished_at_ns,
        status="failed",
        complete=True,
        outputs=(),
        fields=(),
        error=resolved_error,
        interruption=None,
    )
    _write_terminal_attrs(
        _get_group(root, pipeline_run_path(run_id), "Pipeline run"),
        record.to_dict(),
    )
    return record


def interrupt_pipeline_run_record(
    root: zarr.Group,
    *,
    run_id: str,
    interruption: PipelineInterruptionRecord,
    finished_at_ns: int | None = None,
) -> PipelineRunRecord:
    run = load_pipeline_run_record(root, run_id)
    if run.complete:
        raise ValueError("Pipeline run is already terminal")
    record = replace(
        run,
        label=None,
        finished_at_ns=time.time_ns() if finished_at_ns is None else finished_at_ns,
        status="interrupted",
        complete=True,
        outputs=(),
        fields=(),
        error=None,
        interruption=interruption,
    )
    _write_terminal_attrs(
        _get_group(root, pipeline_run_path(run_id), "Pipeline run"),
        record.to_dict(),
    )
    return record


def _runs_group(root: zarr.Group) -> zarr.Group | None:
    if "pipeline" not in root:
        return None
    pipeline = as_zarr_group(root["pipeline"], name="pipeline")
    if "runs" not in pipeline:
        return None
    return as_zarr_group(pipeline["runs"], name=PIPELINE_RUNS_PATH)


def _valid_pipeline_run_records(root: zarr.Group) -> list[PipelineRunRecord]:
    group = _runs_group(root)
    if group is None:
        return []
    records = []
    for run_id in group.group_keys():
        try:
            records.append(load_pipeline_run_record(root, run_id))
        except (KeyError, TypeError, ValueError):
            continue
    return records


def list_pipeline_run_records(
    root: zarr.Group,
    *,
    status: str | Sequence[str] | None = None,
    limit: int = 20,
) -> tuple[PipelineRunRecord, ...]:
    """Scan strict run records and return newest records first."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    allowed = {"running", "completed", "failed", "interrupted"}
    if status is None:
        statuses = allowed
    elif isinstance(status, str):
        statuses = {status}
    elif isinstance(status, Sequence) and not isinstance(status, bytes):
        statuses = set(status)
    else:
        raise TypeError("status must be a string, sequence of strings, or None")
    if not statuses or any(not isinstance(item, str) for item in statuses):
        raise ValueError("status must contain at least one pipeline status")
    unknown = statuses - allowed
    if unknown:
        raise ValueError(f"Unknown pipeline run status: {sorted(unknown)!r}")
    records = _valid_pipeline_run_records(root)
    records = [record for record in records if record.status in statuses]
    records.sort(key=lambda item: (item.started_at_ns, item.run_id), reverse=True)
    return tuple(records[:limit])


def open_pipeline_run_record(
    root: zarr.Group,
    *,
    run_id: str | None = None,
    label: str | None = None,
) -> PipelineRunRecord:
    """Open exactly one run identity or scan for one completed label."""

    if (run_id is None) == (label is None):
        raise ValueError("Provide exactly one of run_id or label")
    if run_id is not None:
        return load_pipeline_run_record(root, run_id)
    assert label is not None
    _validate_non_empty_string(label, "label")
    matches = [
        record
        for record in list_pipeline_run_records(
            root,
            status="completed",
            limit=2**31 - 1,
        )
        if record.successfully_completed and record.label == label
    ]
    if not matches:
        raise KeyError(f"No completed pipeline run has label {label!r}")
    if len(matches) != 1:
        ids = ", ".join(record.run_id for record in matches)
        raise ValueError(
            f"Completed pipeline label {label!r} is duplicated by runs: {ids}"
        )
    return matches[0]


def ensure_pipeline_label_available(
    root: zarr.Group,
    label: str,
    *,
    exclude_run_id: str | None = None,
) -> None:
    """Reject a label already committed by a completed run."""

    _validate_non_empty_string(label, "label")
    if exclude_run_id is not None:
        _validate_run_id(exclude_run_id)
    matches = []
    for record in _valid_pipeline_run_records(root):
        if (
            record.successfully_completed
            and record.label == label
            and record.run_id != exclude_run_id
        ):
            matches.append(record.run_id)
    if matches:
        raise ValueError(
            f"Pipeline label {label!r} is already committed by run {matches[0]}"
        )


def ensure_pipeline_label_claimable(root: zarr.Group, label: str) -> None:
    """Fail before computation when a requested label cannot be claimed safely."""

    _validate_non_empty_string(label, "label")
    if not _store_supports_atomic_label_claims(root.store):
        raise RuntimeError(
            "Pipeline labels require a Zarr store with atomic set_if_not_exists support"
        )
    _validate_pipeline_label_claim_container(root)
    ensure_pipeline_label_available(root, label)
    owner_id = _pipeline_label_claim_owner(root, label)
    if owner_id is None:
        return
    try:
        owner = load_pipeline_run_record(root, owner_id)
    except KeyError:
        return
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Pipeline label {label!r} has an invalid claim-owner record"
        ) from exc
    if owner.requested_label != label:
        raise ValueError(
            f"Pipeline label {label!r} has a claim from an incompatible run"
        )
    if owner.successfully_completed:
        raise ValueError(
            f"Pipeline label {label!r} is already committed by run {owner_id}"
        )
    if owner.complete and owner.status in {"failed", "interrupted"}:
        return
    raise RuntimeError(
        f"Pipeline label {label!r} is held by unfinished run {owner_id}; "
        "after confirming that process has stopped, call "
        "pipeline.abandon_label_claim with this label and run_id"
    )


def abandon_pipeline_label_claim(
    root: zarr.Group,
    *,
    label: str,
    run_id: str,
    reason: str,
) -> PipelineRunRecord:
    """Interrupt an explicitly identified claim owner after its process has stopped."""

    _validate_non_empty_string(label, "label")
    _validate_run_id(run_id)
    _validate_non_empty_string(reason, "reason")
    if not _store_supports_atomic_label_claims(root.store):
        raise RuntimeError(
            "Pipeline labels require a Zarr store with atomic set_if_not_exists support"
        )
    owner_id = _pipeline_label_claim_owner(root, label)
    if owner_id is None:
        raise KeyError(f"Pipeline label {label!r} has no durable claim")
    if owner_id != run_id:
        raise ValueError(
            f"Pipeline label {label!r} is owned by run {owner_id}, not {run_id}"
        )
    owner = load_pipeline_run_record(root, owner_id)
    if owner.requested_label != label:
        raise ValueError(
            f"Pipeline label {label!r} has a claim from an incompatible run"
        )
    if owner.complete:
        raise ValueError(f"Pipeline label {label!r} is not held by an unfinished run")
    return interrupt_pipeline_run_record(
        root,
        run_id=run_id,
        interruption=PipelineInterruptionRecord(
            kind="abandoned_label_claim",
            message=reason,
            requested_at_ns=time.time_ns(),
        ),
    )
