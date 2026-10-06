"""Operation revisions, the one explicit way to stop reusing stored results.

An artifact's identity is its scope, assay, kind, and canonical provenance:
the producing operation, its parameters, and its inputs. Reuse requires an
exact provenance match, so a fix that changes what an operation computes for
unchanged provenance must also change provenance, or stored results of the
old code would keep being reused. This module records each such fix as an
:class:`OperationRevision`.

Planning computes the effective revision of a new artifact with
:func:`effective_revision` and records it in provenance only when it is 2 or
more. An operation without revisions therefore records exactly the provenance
it recorded before this registry existed, and a revision changes only the
identities of the artifacts it applies to. Artifacts that record no revision
are revision 1.

The registry is append-only. A released :class:`OperationRevision` is never
edited, renumbered, or removed, and its predicate never changes: stored
artifacts are judged against it by every later release. A new release that
changes the same results again adds the next revision.

See ``docs/source/developers/operation_revisions.md`` for when a change needs
a revision and how to add one.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

type RevisionPredicate = Callable[[str, Mapping[str, Any], Mapping[str, Any]], bool]
"""Predicate over an artifact's kind and its recorded parameters and inputs."""


@dataclass(frozen=True, slots=True)
class OperationRevision:
    """One released change to what an operation computes.

    Attributes:
        revision: The revision number; an operation's first change is 2.
        release: The first Scarf release that records it, such as ``"1.0.0"``.
        change: One line that says what changed.
        applies: A pure predicate that selects the affected artifacts, or None for all.
    """

    revision: int
    release: str
    change: str
    applies: RevisionPredicate | None


def _doublets_with_forced_heterotypic_pairs(
    _kind: str, parameters: Mapping[str, Any], _inputs: Mapping[str, Any]
) -> bool:
    """Revision 2 of ``run_doublet_detection``, released in 1.0.0.

    Only forced heterotypic pairs changed. Scores that record a
    ``heterotypic_fraction`` of exactly 0 drew both parents uniformly, and
    they still do with the same draws; every other record is affected,
    including one without the parameter.
    """
    fraction = parameters.get("heterotypic_fraction")
    return not (
        isinstance(fraction, int | float)
        and not isinstance(fraction, bool)
        and fraction == 0
    )


def _records_library_size(method: object) -> bool:
    """Whether a recorded normalizer identity names ``norm_lib_size``.

    Every release records ``norm_lib_size`` under the module ``scarf.assay``;
    ``run_normalization`` adds ``external_hook``. Released predicates use
    this helper, so it never changes.
    """
    return (
        isinstance(method, Mapping)
        and method.get("module") == "scarf.assay"
        and method.get("qualname") == "norm_lib_size"
    )


def _flags_of_another_normalizer(
    _kind: str, parameters: Mapping[str, Any], _inputs: Mapping[str, Any]
) -> bool:
    """Revision 2 of ``run_normalization``, released in 1.0.0.

    For a normalizer other than ``norm_lib_size``, saved normalization did
    not store what its recorded flags mean. With ``log_transform`` True, RNA
    assays saved library-size log values and every other assay saved
    unlogged values. With ``renormalize_subset`` True, RNA assays, which
    alone record a size factor, saved library-size values of the subset.
    """
    if _records_library_size(parameters.get("normalization_method")):
        return False
    if parameters.get("log_transform") is True:
        return True
    return (
        parameters.get("size_factor") is not None
        and parameters.get("renormalize_subset") is True
    )


def _logged_values_of_another_normalizer(
    _kind: str, parameters: Mapping[str, Any], _inputs: Mapping[str, Any]
) -> bool:
    """Revision 2 of the marker and pseudotime feature searches, released in 1.0.0.

    With ``log_transform`` True, a normalizer other than ``norm_lib_size``
    was ranked or correlated as library-size log values or as its unlogged
    values, not as ``log1p`` of its own values.
    """
    normalization = parameters.get("normalization")
    return (
        isinstance(normalization, Mapping)
        and normalization.get("log_transform") is True
        and not _records_library_size(parameters.get("normalization_method"))
    )


# Released revisions by operation. An operation that is not listed has no
# revisions, so every artifact of it is revision 1. Append a revision to the
# tuple of the operation it changes, adding the operation if needed; never
# edit a released entry.
_OPERATION_REVISIONS: dict[str, tuple[OperationRevision, ...]] = {
    "build_connectivity_map": (
        OperationRevision(
            2,
            "1.0.0",
            "Edge weights are fitted with each cell itself at distance zero, as "
            "umap-learn does, so a cell's k weights sum to bandwidth * log2(k + 1) "
            "instead of bandwidth * log2(k) + 1",
            None,
        ),
    ),
    "calc_membership_strength": (
        OperationRevision(
            2,
            "1.0.0",
            "Membership strength is the share of a cell's neighbors that carry "
            "its own cluster label, not the share of the most common neighbor "
            "label",
            None,
        ),
    ),
    "run_doublet_detection": (
        OperationRevision(
            2,
            "1.0.0",
            "With heterotypic_fraction above 0, simulated doublets meet the "
            "fraction by drawing forced partners from other clusters directly, "
            "which changes every simulated pair",
            _doublets_with_forced_heterotypic_pairs,
        ),
    ),
    "run_harmony": (
        OperationRevision(
            2,
            "1.0.0",
            "Zero-norm centroids stay zero instead of NaN, and batch_levels follow "
            "the design order of per-level theta and lamb",
            None,
        ),
    ),
    "run_marker_search": (
        OperationRevision(
            2,
            "1.0.0",
            "fold_change is +inf for a feature that no other cell expresses and "
            "NaN when both means are 0 or either is negative, instead of the "
            "100.1 and 0 sentinels",
            None,
        ),
        OperationRevision(
            3,
            "1.0.0",
            "Marker search with log_transform ranks log1p of the configured "
            "normalizer's values; with a normalizer other than norm_lib_size it "
            "ranked library-size log values or unlogged values",
            _logged_values_of_another_normalizer,
        ),
    ),
    "run_normalization": (
        OperationRevision(
            2,
            "1.0.0",
            "Saved normalization applies the configured normalizer and its "
            "flags; for a normalizer other than norm_lib_size, log_transform "
            "saved library-size log values on RNA assays and unlogged values on "
            "other assays, and renormalize_subset saved RNA library-size values",
            _flags_of_another_normalizer,
        ),
    ),
    "run_pseudotime_aggregation": (
        OperationRevision(
            2,
            "1.0.0",
            "Pseudotime aggregation with log_transform aggregates log1p of the "
            "configured normalizer's values; with a normalizer other than "
            "norm_lib_size it aggregated library-size log values or unlogged values",
            _logged_values_of_another_normalizer,
        ),
    ),
    "run_pseudotime_marker_search": (
        OperationRevision(
            2,
            "1.0.0",
            "Pseudotime markers with log_transform correlate log1p of the "
            "configured normalizer's values; with a normalizer other than "
            "norm_lib_size they correlated library-size log values or unlogged values",
            _logged_values_of_another_normalizer,
        ),
    ),
}


def _validate_registry(
    registry: Mapping[str, tuple[OperationRevision, ...]],
) -> None:
    for operation, revisions in registry.items():
        numbers = [entry.revision for entry in revisions]
        if numbers != list(range(2, len(numbers) + 2)):
            raise ValueError(
                f"Revisions of {operation!r} must be numbered 2, 3, ... in order, "
                f"got {numbers}"
            )


_validate_registry(_OPERATION_REVISIONS)
OPERATION_REVISIONS: Mapping[str, tuple[OperationRevision, ...]] = MappingProxyType(
    _OPERATION_REVISIONS
)
"""Released revisions by operation; an operation that is not listed has none."""


def applicable_revisions(
    operation: str,
    kind: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> tuple[OperationRevision, ...]:
    """Return the released revisions that apply to one artifact, oldest first."""
    return tuple(
        entry
        for entry in OPERATION_REVISIONS.get(operation, ())
        if entry.applies is None or entry.applies(kind, parameters, inputs)
    )


def effective_revision(
    operation: str,
    kind: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> int:
    """Return the highest applicable revision of ``operation`` for an artifact, or 1."""
    applicable = applicable_revisions(operation, kind, parameters, inputs)
    return applicable[-1].revision if applicable else 1
