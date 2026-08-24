"""Shared helpers for format-specific ingest handlers."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...storage.profiles import is_local_zarr_path
from ...storage.stores import zarr_location_has_content
from ..decide import DecisionValidationError, decide
from ..types import Decision, EvidenceItem
from .result import (
    IngestResult,
    done,
    failed,
    failed_from_exception,
    failure_note,
    needs_input,
)

# `hto(?![a-z])` matches HTO, HTO1, HTO-1; avoids requiring a word boundary
# after HTO (digits are word chars, so `hto\b` misses HTO1).
_HTO_NAME_RE = re.compile(
    r"(hashtag|hto(?![a-z])|totalseq[^a-z0-9]*hash|hash[^a-z0-9]*tag)",
    re.IGNORECASE,
)

CONVERT_FORMATS = frozenset({"h5ad", "10x_h5", "10x_dir", "mtx", "loom", "seurat"})

# Inspection and summary boundaries expose data, layout, and I/O failures.
# Some readers use RuntimeError for data-dependent conversion failures, so that
# broader set is restricted to reader and writer execution.
DATA_LAYOUT_ERRORS = (OSError, ValueError, KeyError)
CONVERSION_DATA_ERRORS = (*DATA_LAYOUT_ERRORS, RuntimeError)


def _local_path(location: str) -> Path:
    if location.startswith("file://"):
        return Path(location[7:])
    return Path(location)


def _paths_overlap(source: Path, destination: Path) -> bool:
    try:
        source.relative_to(destination)
        return True
    except ValueError:
        pass
    try:
        destination.relative_to(source)
        return True
    except ValueError:
        return False


def ensure_convert_destination(
    source: Path,
    zarrPath: str | Path | None,
    directions: Mapping[str, Any],
    *,
    format_name: str,
) -> str | IngestResult:
    """Validate conversion destination before inspect, reader, or model work."""
    if zarrPath is None:
        return failed(
            format_name=format_name,
            notes=[f"zarrPath is required when converting {format_name} inputs"],
        )

    destination = str(zarrPath)
    overwrite = directions.get("overwrite")
    if overwrite is not None and type(overwrite) is not bool:
        return failed(
            format_name=format_name,
            zarr_path=destination,
            notes=[
                "overwrite must be boolean true; "
                f"got {type(overwrite).__name__}: {overwrite!r}"
            ],
        )

    if is_local_zarr_path(destination):
        try:
            source_resolved = source.resolve()
            dest_resolved = _local_path(destination).resolve()
        except OSError as exc:
            return failed_from_exception(
                format_name=format_name,
                operation="resolve destination paths",
                exc=exc,
                zarr_path=destination,
                notes=[],
            )
        if source_resolved == dest_resolved or _paths_overlap(
            source_resolved, dest_resolved
        ):
            return failed(
                format_name=format_name,
                zarr_path=destination,
                notes=[
                    "destination must not equal or nest with the source path; "
                    f"source={source_resolved} destination={dest_resolved}"
                ],
            )

    try:
        exists = zarr_location_has_content(destination)
    except Exception as exc:
        return failed_from_exception(
            format_name=format_name,
            operation="probe destination",
            exc=exc,
            zarr_path=destination,
            notes=[],
            extra_notes=[
                "Destination existence could not be verified; refusing to write",
            ],
        )

    if exists and overwrite is not True:
        return failed(
            format_name=format_name,
            zarr_path=destination,
            notes=[
                f"Destination already exists: {destination}. "
                'Pass directions={"overwrite": true} to replace it.'
            ],
        )
    return destination


def open_summary(
    zarr_path: str,
    *,
    default_assay: str | None = None,
) -> tuple[list[str], str | None, dict[str, Any]]:
    """Open a converted store for first-time QC initialization and summary."""
    import zarr

    from ...datastore.datastore import DataStore
    from ...storage.stores import load_zarr

    resolved_default = default_assay
    if resolved_default is None:
        root = load_zarr(zarr_loc=zarr_path, mode="r")
        assay_names = [
            name
            for name in sorted(dict.fromkeys(root.group_keys()))
            if isinstance(root[name], zarr.Group) and "is_assay" in root[name].attrs
        ]
        if "RNA" in assay_names:
            resolved_default = "RNA"
        elif assay_names:
            resolved_default = assay_names[0]

    ds = DataStore(zarr_path, default_assay=resolved_default)
    summary = ds.summary().to_dict()
    return list(ds.assay_names), ds._defaultAssay, summary


def open_readonly_summary(
    zarr_path: str,
    *,
    default_assay: str | None = None,
) -> tuple[list[str], str | None, dict[str, Any]]:
    """Summarize an existing store without constructing a mutable DataStore."""
    from ...datastore.summary import summarize_zarr_readonly

    summary = summarize_zarr_readonly(zarr_path, default_assay=default_assay)
    return (
        [assay.name for assay in summary.assays],
        summary.default_assay,
        summary.to_dict(),
    )


def datastore_action(
    zarr_path: str,
    default_assay: str | None,
    *,
    zarr_mode: str = "r+",
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "op": "DataStore",
        "zarrPath": zarr_path,
        "zarrMode": zarr_mode,
    }
    if default_assay is not None:
        action["defaultAssay"] = default_assay
    return action


def finish(
    *,
    format_name: str,
    zarr_path: str,
    notes: list[str],
    convert_actions: list[dict[str, Any]],
    action_labels: list[str],
    default_assay: str | None = None,
    decision: Decision | None = None,
    summary_mode: str = "r+",
) -> IngestResult:
    """Open the store, append DataStore replay action, and return a done result."""
    try:
        if summary_mode == "r":
            assay_names, resolved_default, summary = open_readonly_summary(
                zarr_path,
                default_assay=default_assay,
            )
            action = {
                "op": "summarizeZarr",
                "zarrPath": zarr_path,
                "zarrMode": "r",
            }
            if resolved_default is not None:
                action["defaultAssay"] = resolved_default
        else:
            assay_names, resolved_default, summary = open_summary(
                zarr_path,
                default_assay=default_assay,
            )
            action = datastore_action(zarr_path, resolved_default, zarr_mode="r+")
    except DATA_LAYOUT_ERRORS as exc:
        read_only = summary_mode == "r"
        return failed_from_exception(
            format_name=format_name,
            operation=(
                "summarize existing Zarr" if read_only else "open converted store"
            ),
            exc=exc,
            zarr_path=zarr_path,
            notes=notes,
            extra_notes=(
                ()
                if read_only
                else (f"Destination may contain a converted store at {zarr_path}",)
            ),
        )
    return done(
        format_name=format_name,
        zarr_path=zarr_path,
        assay_names=assay_names,
        summary=summary,
        accepted_actions=[*convert_actions, action],
        action_labels=action_labels,
        notes=notes,
        decision=decision,
    )


def antibody_names_look_like_hto(names: Sequence[str]) -> bool:
    if not names:
        return False
    hits = sum(1 for name in names if _HTO_NAME_RE.search(str(name)))
    return hits >= max(1, (len(names) + 1) // 2)


def _modality_needs_input(
    *,
    format_name: str,
    evidence: Sequence[EvidenceItem],
    notes: list[str],
) -> IngestResult:
    return needs_input(
        format_name=format_name,
        question=(
            "Antibody Capture features look like hashtags. "
            "Should they be imported as ADT or HTO?"
        ),
        options=["ADT", "HTO"],
        evidence_ids=[item.id for item in evidence],
        notes=notes,
    )


def resolve_modality_choice(
    *,
    model: Any | None,
    directions: Mapping[str, Any],
    feature_names: Sequence[str],
    format_name: str,
) -> tuple[str | None, Decision | None, IngestResult | None]:
    forced = directions.get("modalityChoice")
    if forced in {"ADT", "HTO"}:
        return str(forced), None, None
    if not antibody_names_look_like_hto(feature_names):
        return "ADT", None, None

    evidence = [
        EvidenceItem(
            id="modality:ADT",
            label="ADT",
            summary="Treat Antibody Capture features as surface protein ADT assay",
        ),
        EvidenceItem(
            id="modality:HTO",
            label="HTO",
            summary=(
                "Treat Antibody Capture features as multiplexing HTO assay; "
                f"names include {[str(name) for name in feature_names[:8]]}"
            ),
        ),
    ]
    if model is None:
        return (
            None,
            None,
            _modality_needs_input(
                format_name=format_name,
                evidence=evidence,
                notes=[
                    "Hashtag-like Antibody Capture features require an explicit choice"
                ],
            ),
        )
    try:
        decision = decide(
            model=model,
            question=(
                "Antibody Capture features look like hashtags. "
                "Choose modality:ADT or modality:HTO."
            ),
            evidence=evidence,
        )
    except DecisionValidationError as exc:
        return (
            None,
            None,
            _modality_needs_input(
                format_name=format_name,
                evidence=evidence,
                notes=[
                    "Hashtag-like Antibody Capture features require an explicit choice",
                    failure_note("modalityChoice", exc),
                ],
            ),
        )
    choice = "HTO" if decision.selectedId.endswith("HTO") else "ADT"
    return choice, decision, None
