"""Shared helpers for format-specific ingest handlers."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..decide import decide
from ..types import Decision, EvidenceItem
from .result import IngestResult, done, needs_input

# `hto(?![a-z])` matches HTO, HTO1, HTO-1; avoids requiring a word boundary
# after HTO (digits are word chars, so `hto\b` misses HTO1).
_HTO_NAME_RE = re.compile(
    r"(hashtag|hto(?![a-z])|totalseq[^a-z0-9]*hash|hash[^a-z0-9]*tag)",
    re.IGNORECASE,
)


def require_zarr_path(zarrPath: str | Path | None, *, format_name: str) -> str:
    if zarrPath is None:
        raise ValueError(f"zarrPath is required when converting {format_name} inputs")
    return str(zarrPath)


def open_summary(
    zarr_path: str,
    *,
    default_assay: str | None = None,
) -> tuple[list[str], str | None, dict[str, Any]]:
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
    return list(ds.assay_names), resolved_default, summary


def datastore_action(zarr_path: str, default_assay: str | None) -> dict[str, Any]:
    action: dict[str, Any] = {"op": "DataStore", "zarrPath": zarr_path}
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
) -> IngestResult:
    """Open the store, append DataStore replay action, and return a done result."""
    assay_names, resolved_default, summary = open_summary(
        zarr_path,
        default_assay=default_assay,
    )
    return done(
        format_name=format_name,
        zarr_path=zarr_path,
        assay_names=assay_names,
        summary=summary,
        accepted_actions=[
            *convert_actions,
            datastore_action(zarr_path, resolved_default),
        ],
        action_labels=action_labels,
        notes=notes,
        decision=decision,
    )


def antibody_names_look_like_hto(names: Sequence[str]) -> bool:
    if not names:
        return False
    hits = sum(1 for name in names if _HTO_NAME_RE.search(str(name)))
    return hits >= max(1, (len(names) + 1) // 2)


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
            needs_input(
                format_name=format_name,
                question=(
                    "Antibody Capture features look like hashtags. "
                    "Should they be imported as ADT or HTO?"
                ),
                options=["ADT", "HTO"],
                evidence_ids=[item.id for item in evidence],
                notes=[
                    "Hashtag-like Antibody Capture features require an explicit choice"
                ],
            ),
        )
    decision = decide(
        model=model,
        question=(
            "Antibody Capture features look like hashtags. "
            "Choose modality:ADT or modality:HTO."
        ),
        evidence=evidence,
    )
    choice = "HTO" if decision.selectedId.endswith("HTO") else "ADT"
    return choice, decision, None
