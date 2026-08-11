"""Select a neighbor graph with one optional design-guarded Harmony branch."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from ..graph.state import validate_normalized_artifact_selection
from ..metadata.queries import column_constant_within, reduce_observation_units
from ..metrics.association import coefficient_estimability
from ..metrics.connectivity import graph_connectivity
from ..metrics.lisi import clisi_knn, ilisi_knn
from ..metrics.pca_diagnostics import (
    branch_nuisance_summary,
    per_pc_covariate_associations,
)
from ..storage.artifacts import (
    ArtifactRef,
    artifact_group,
    artifact_path,
    inspect_artifact,
)
from ..storage.types import as_zarr_array
from ._deps import AGENT_INSTALL_HINT
from .types import Decision, NeedsInput, StageStatus

try:
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "GraphSelectionResult",
    "select_graph",
]

_ASSOCIATION_FLOOR = 0.1
_DEFAULT_K = 11
_DEFAULT_ANN = {"ann_metric": "l2", "ann_efc": 50, "ann_ef": 50, "ann_m": 16}
_CHEAP_TECH_WORSEN = 1.05
_ASSOCIATION_NONREGRESSION_TOLERANCE = 0.01
_GRAPH_ILISI_GAIN = 0.02
_GRAPH_PROTECTED_FLOOR = 0.95
_FORBIDDEN_BATCH_NAMES = frozenset(
    {
        "cell_type",
        "celltype",
        "cellType",
        "annotation",
        "cluster",
        "leiden",
        "paris",
        "labels",
    }
)


class GraphSelectionResult(BaseModel):
    status: StageStatus
    auditLog: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    needsInput: NeedsInput | None = None
    acceptedActions: list[dict[str, Any]] = Field(default_factory=list)
    assay: str | None = None
    branches: list[dict[str, Any]] = Field(default_factory=list)
    selectedBranch: str | None = None
    selectedCoordinates: dict[str, Any] | None = None
    selectedNeighbors: dict[str, Any] | None = None
    selectedGraph: dict[str, Any] | None = None
    harmonyBatchColumns: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _Run:
    store: Any
    assay_name: str
    pca: Mapping[str, Any]
    covariates: Mapping[str, Any] | None
    directions: dict[str, Any]
    study_context: str | None
    cell_key: str = "I"
    audit: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_input: NeedsInput | None = None

    def note(self, *, kind: str, detail: str, **fields: Any) -> None:
        self.audit.append({"kind": kind, "detail": detail, **fields})


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if hasattr(value, "model_dump"):
        return cast(Mapping[str, Any], value.model_dump())
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{name} must be a mapping or pydantic model")


def _artifact_dict(ref: ArtifactRef) -> dict[str, Any]:
    return ref.to_dict()


def _pca_ref(pca: Mapping[str, Any]) -> ArtifactRef:
    selected = pca.get("selectedPca")
    if not isinstance(selected, Mapping):
        raise ValueError("pca result has no selectedPca artifact")
    return ArtifactRef.from_dict(dict(selected))


def _isna(values: np.ndarray) -> np.ndarray:
    import pandas as pd

    return np.asarray(pd.isna(values), dtype=bool)


def _covariate_catalog(
    covariates: Mapping[str, Any] | None,
) -> tuple[
    list[str],
    list[str],
    dict[str, Literal["categorical", "continuous"]],
    dict[str, str],
    set[str],
]:
    technical: list[str] = []
    protected: list[str] = []
    kinds: dict[str, Literal["categorical", "continuous"]] = {}
    domains: dict[str, str] = {}
    aliases: set[str] = set()
    if covariates is None:
        return technical, protected, kinds, domains, aliases
    for column in covariates.get("columns", []):
        if not isinstance(column, Mapping):
            continue
        name = column.get("name")
        domain = column.get("domain")
        kind = column.get("kind")
        if name is None or kind not in {"categorical", "continuous"}:
            continue
        column_name = str(name)
        kinds[column_name] = cast(Literal["categorical", "continuous"], kind)
        if isinstance(domain, str):
            domains[column_name] = domain
        for alias in column.get("aliases") or []:
            aliases.add(str(alias))
        if domain == "technical":
            technical.append(column_name)
        elif domain == "biological":
            protected.append(column_name)
    coefficients = {
        str(item.get("name") or item.get("column"))
        for item in covariates.get("coefficients", [])
        if isinstance(item, Mapping)
        and (item.get("name") is not None or item.get("column") is not None)
    }
    if coefficients:
        protected = [name for name in protected if name in coefficients] or protected
    return technical, protected, kinds, domains, aliases


def _diagnostic_roster(
    pca: Mapping[str, Any],
    *,
    technical: Sequence[str],
    protected: Sequence[str],
    kinds: Mapping[str, Literal["categorical", "continuous"]],
) -> list[dict[str, str]]:
    diagnostics = pca.get("diagnostics") or {}
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    raw_roster = diagnostics.get("diagnosticCovariates")
    if isinstance(raw_roster, Sequence) and not isinstance(
        raw_roster,
        str | bytes,
    ):
        roster: dict[str, dict[str, str]] = {}
        for raw in raw_roster:
            if not isinstance(raw, Mapping):
                raise ValueError(
                    "Stage 3 diagnostic roster contains a non-mapping entry"
                )
            name = raw.get("name")
            kind = raw.get("kind")
            role = raw.get("role")
            source = raw.get("source")
            if (
                not isinstance(name, str)
                or kind not in {"categorical", "continuous"}
                or role not in {"technical", "nuisance", "protected"}
                or not isinstance(source, str)
            ):
                raise ValueError(
                    "Stage 3 diagnostic roster entries require name, kind, role, "
                    "and source"
                )
            entry = {
                "name": name,
                "kind": kind,
                "role": role,
                "source": source,
            }
            if name in roster and roster[name] != entry:
                raise ValueError(
                    f"Stage 3 diagnostic roster gives conflicting roles for {name!r}"
                )
            roster[name] = entry
        for role, key in (
            ("technical", "technicalCovariates"),
            ("nuisance", "nuisanceCovariates"),
            ("protected", "protectedCovariates"),
        ):
            raw_names = diagnostics.get(key)
            if not isinstance(raw_names, list):
                continue
            expected = {name for name, entry in roster.items() if entry["role"] == role}
            if {str(name) for name in raw_names} != expected:
                raise ValueError(f"Stage 3 {key} does not match diagnosticCovariates")
        return [roster[name] for name in sorted(roster)]

    roster = {}
    for name, role in [
        *((name, "technical") for name in technical),
        *((name, "protected") for name in protected),
    ]:
        kind = kinds.get(name)
        if kind is not None:
            roster[name] = {
                "name": name,
                "kind": kind,
                "role": role,
                "source": "legacyStage3Diagnostics",
            }
    protect_proliferation = bool(diagnostics.get("protectProliferation", True))
    cell_cycle = pca.get("cellCycle") or {}
    if isinstance(cell_cycle, Mapping):
        for key, kind in (
            ("sScoreColumn", "continuous"),
            ("g2mScoreColumn", "continuous"),
            ("phaseColumn", "categorical"),
        ):
            name = cell_cycle.get(key)
            if isinstance(name, str):
                roster[name] = {
                    "name": name,
                    "kind": kind,
                    "role": ("protected" if protect_proliferation else "nuisance"),
                    "source": "legacyCellCycleDiagnostics",
                }
    return [roster[name] for name in sorted(roster)]


def _drop_nested_duplicates(
    candidates: Sequence[str],
    nesting: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Keep coarser technical factors when one nests inside another."""
    drop: set[str] = set()
    notes: list[dict[str, Any]] = []
    present = set(candidates)
    for report in nesting:
        left = report.get("left")
        right = report.get("right")
        relation = report.get("nesting")
        if left not in present or right not in present:
            continue
        if relation == "leftInRight":
            drop.add(str(left))
            notes.append(
                {
                    "kind": "droppedNestedTechnical",
                    "detail": f"Dropped nested technical {left} inside {right}",
                    "kept": right,
                    "dropped": left,
                }
            )
        elif relation == "rightInLeft":
            drop.add(str(right))
            notes.append(
                {
                    "kind": "droppedNestedTechnical",
                    "detail": f"Dropped nested technical {right} inside {left}",
                    "kept": left,
                    "dropped": right,
                }
            )
        elif relation == "equivalent":
            ordered = sorted([str(left), str(right)])
            drop.add(ordered[1])
            notes.append(
                {
                    "kind": "droppedEquivalentTechnical",
                    "detail": f"Dropped equivalent technical {ordered[1]}",
                    "kept": ordered[0],
                    "dropped": ordered[1],
                }
            )
    return [name for name in candidates if name not in drop], notes


def _stage3_unwanted_signal_remains(pca: Mapping[str, Any]) -> bool:
    diagnostics = pca.get("diagnostics") or {}
    if not isinstance(diagnostics, Mapping):
        return False
    summary = diagnostics.get("selectedSummary") or {}
    if not isinstance(summary, Mapping):
        return False
    for role in ("technical", "nuisance"):
        bucket = summary.get(role) or {}
        if not isinstance(bucket, Mapping):
            continue
        if (
            int(bucket.get("nFlaggedPcs") or 0) > 0
            or float(bucket.get("meanAssociation") or 0.0) >= _ASSOCIATION_FLOOR
        ):
            return True
    return False


def _observation_unit(covariates: Mapping[str, Any] | None) -> str | None:
    if covariates is None:
        return None
    units: list[str] = []
    for report in covariates.get("confounding", []):
        if not isinstance(report, Mapping):
            continue
        unit = report.get("observationUnit")
        if isinstance(unit, str) and unit:
            units.append(unit)
    unique = sorted(set(units))
    return unique[0] if len(unique) == 1 else None


def _resolve_cell_key(store: Any, pca_ref: ArtifactRef) -> str:
    """Validate and return the selection key that produced the Stage 3 PCA."""
    status = inspect_artifact(store.zw, pca_ref)
    if not status.exists or not status.complete:
        raise ValueError("Selected PCA artifact is incomplete")
    inputs = status.inputs or {}
    raw_normalized = inputs.get("normalized")
    if not isinstance(raw_normalized, dict):
        raise ValueError("Selected PCA artifact has no normalized input")
    normalized = ArtifactRef.from_dict(raw_normalized)
    normalized_status = inspect_artifact(store.zw, normalized)
    if not normalized_status.exists or not normalized_status.complete:
        raise ValueError("Normalized artifact for selected PCA is incomplete")
    execution = normalized_status.execution_options or {}
    cell_key = execution.get("cell_key")
    feat_key = execution.get("feat_key")
    if not isinstance(cell_key, str) or not cell_key:
        raise ValueError("Normalized artifact has no cell_key")
    if not isinstance(feat_key, str) or not feat_key:
        raise ValueError("Normalized artifact has no feat_key")
    if cell_key not in store.cells.columns:
        raise ValueError(
            f"Artifact cell_key {cell_key!r} is missing from cell metadata"
        )
    validate_normalized_artifact_selection(
        store.zw,
        normalized,
        cell_key,
        feat_key,
    )
    return cell_key


def _selection_values(run: _Run, column: str) -> np.ndarray:
    return np.asarray(run.store.cells.fetch(column, key=run.cell_key))


def _estimability_blocks_batch(
    run: _Run,
    *,
    batch_columns: Sequence[str],
    protected: Sequence[str],
    kinds: Mapping[str, Literal["categorical", "continuous"]],
    observation_unit: str | None,
) -> str | None:
    """Return a blocking reason when batch terms jointly confound protected biology."""
    if not batch_columns:
        return "No batch columns were provided"
    for batch_column in batch_columns:
        if batch_column not in run.store.cells.columns:
            return f"Batch column {batch_column!r} is missing"
        batch_values = _selection_values(run, batch_column)
        if len(np.unique(batch_values[~_isna(batch_values)])) < 2:
            return f"Batch column {batch_column!r} has fewer than two levels"

    if not protected:
        return None

    for coefficient in protected:
        if coefficient not in run.store.cells.columns:
            return (
                f"Protected coefficient {coefficient!r} is missing from cell metadata"
            )
        coeff_kind = kinds.get(coefficient)
        if coeff_kind is None:
            return (
                f"Protected coefficient {coefficient!r} has no typed kind; "
                "provide covariates before Harmony"
            )
        use_design = (
            observation_unit is not None
            and observation_unit in run.store.cells.columns
            and column_constant_within(
                run.store.cells,
                coefficient,
                observation_unit,
                cell_key=run.cell_key,
            )
            and all(
                column_constant_within(
                    run.store.cells,
                    name,
                    observation_unit,
                    cell_key=run.cell_key,
                )
                for name in batch_columns
            )
        )
        if use_design:
            assert observation_unit is not None
            columns = list(
                dict.fromkeys([observation_unit, coefficient, *batch_columns])
            )
            design = reduce_observation_units(
                run.store.cells,
                observation_unit,
                columns,
                cell_key=run.cell_key,
            )
            report = coefficient_estimability(
                design[coefficient].to_numpy(),
                coefficientKind=coeff_kind,
                technicals={name: design[name].to_numpy() for name in batch_columns},
                technicalKinds={name: "categorical" for name in batch_columns},
            )
        else:
            report = coefficient_estimability(
                _selection_values(run, coefficient),
                coefficientKind=coeff_kind,
                technicals={
                    name: _selection_values(run, name) for name in batch_columns
                },
                technicalKinds={name: "categorical" for name in batch_columns},
            )
        if report.get("status") != "ok":
            return (
                f"Estimability for {coefficient!r} vs {list(batch_columns)} "
                f"not computed ({report.get('reason')})"
            )
        if report.get("rankDeficient") or not report.get("coefficientEstimable"):
            return (
                f"Batch columns {list(batch_columns)} saturate or alias protected "
                f"coefficient {coefficient!r}"
            )
        if int(report.get("residualDf") or 0) <= 0:
            return (
                f"Batch columns {list(batch_columns)} leave no residual df for "
                f"coefficient {coefficient!r}"
            )
    return None


def _phase_within_unit_support(
    run: _Run,
    *,
    phase_column: str,
    observation_unit: str | None,
    protected: Sequence[str],
    design_columns: Sequence[str],
) -> tuple[bool, str]:
    """Require phase to cross observation units and protected/design groups."""
    if phase_column not in run.store.cells.columns:
        return False, f"Phase column {phase_column!r} is missing"
    phase = _selection_values(run, phase_column)
    levels = {str(value) for value in phase[~_isna(phase)]}
    if len(levels) < 2:
        return False, "Cell-cycle phase has fewer than two observed levels"

    def _crosses(group_column: str) -> bool:
        groups = _selection_values(run, group_column)
        frame_ok = ~_isna(phase) & ~_isna(groups)
        if int(frame_ok.sum()) < 4:
            return False
        supported = 0
        for group in np.unique(groups[frame_ok]):
            subset = phase[frame_ok & (groups == group)]
            if len({str(value) for value in subset}) >= 2:
                supported += 1
        return supported >= 2

    if observation_unit is None:
        return False, "Phase Harmony requires a unique observation unit"
    if observation_unit not in run.store.cells.columns:
        return False, f"Observation unit {observation_unit!r} is missing"
    if not _crosses(observation_unit):
        return (
            False,
            f"Phase does not cross within observation unit {observation_unit!r}",
        )
    for name in [*protected, *design_columns]:
        if name not in run.store.cells.columns or name == observation_unit:
            continue
        if not _crosses(name):
            return False, f"Phase does not cross within group {name!r}"
    return True, "Phase has within-unit support"


def _stage3_phase_persists(pca: Mapping[str, Any], phase_column: str) -> bool:
    diagnostics = pca.get("diagnostics") or {}
    if not isinstance(diagnostics, Mapping):
        return False
    for key in ("cellCycleSummary",):
        summary = diagnostics.get(key) or {}
        if not isinstance(summary, Mapping):
            continue
        by_covariate = summary.get("byCovariate") or {}
        if not isinstance(by_covariate, Mapping):
            continue
        entry = by_covariate.get(phase_column) or {}
        if isinstance(entry, Mapping) and float(
            entry.get("meanAssociation") or 0.0
        ) >= (_ASSOCIATION_FLOOR):
            return True
    selected = diagnostics.get("selectedSummary") or {}
    if isinstance(selected, Mapping):
        for bucket in ("protected", "technical", "nuisance"):
            summary = selected.get(bucket) or {}
            if not isinstance(summary, Mapping):
                continue
            by_covariate = summary.get("byCovariate") or {}
            if not isinstance(by_covariate, Mapping):
                continue
            entry = by_covariate.get(phase_column) or {}
            if (
                isinstance(entry, Mapping)
                and float(entry.get("meanAssociation") or 0.0) >= _ASSOCIATION_FLOOR
            ):
                return True
    # Fall back to selected-branch association records.
    selected_id = pca.get("selectedBranch")
    for branch in pca.get("branches") or []:
        if not isinstance(branch, Mapping):
            continue
        if selected_id is not None and branch.get("id") != selected_id:
            continue
        for record in branch.get("diagnostics", {}).get("associations") or []:
            if not isinstance(record, Mapping):
                continue
            if str(record.get("covariate")) != phase_column:
                continue
            strength = record.get("strength")
            if strength is None:
                association = record.get("association") or {}
                value = association.get("value")
                if value is None:
                    continue
                strength = abs(float(value))
            if float(strength) >= _ASSOCIATION_FLOOR:
                return True
    return False


def _forbidden_batch_name(name: str) -> bool:
    lowered = name.lower().replace("-", "_")
    if name in _FORBIDDEN_BATCH_NAMES or lowered in {
        item.lower() for item in _FORBIDDEN_BATCH_NAMES
    }:
        return True
    return any(
        token in lowered
        for token in ("cell_type", "celltype", "leiden", "cluster_label")
    )


def _resolve_batch_columns(
    run: _Run,
    *,
    technical: Sequence[str],
    protected: Sequence[str],
    kinds: Mapping[str, Literal["categorical", "continuous"]],
    domains: Mapping[str, str],
    aliases: set[str],
) -> list[str]:
    directed = run.directions.get("harmonyBatchColumns")
    is_directed = directed is not None
    if directed is not None:
        if not isinstance(directed, Sequence) or isinstance(directed, str | bytes):
            raise TypeError("harmonyBatchColumns must be a sequence of column names")
        candidates = [str(name) for name in directed]
    else:
        candidates = [
            name
            for name in technical
            if kinds.get(name) == "categorical" and name not in aliases
        ]

    protect_proliferation = bool(run.directions.get("protectProliferation", True))
    phase_column = None
    cell_cycle = run.pca.get("cellCycle") or {}
    if isinstance(cell_cycle, Mapping):
        phase_column = cell_cycle.get("phaseColumn")
    allow_phase = bool(run.directions.get("allowPhaseHarmony", False))
    design_columns = [
        name
        for name, domain in domains.items()
        if domain == "design" and kinds.get(name) == "categorical"
    ]
    phase_allowed = False
    phase_detail = (
        "Phase Harmony requires protectProliferation=False, explicit permission, "
        "and persistent Stage 3 phase association"
    )
    if (
        not protect_proliferation
        and allow_phase
        and isinstance(phase_column, str)
        and phase_column
        and _stage3_phase_persists(run.pca, phase_column)
    ):
        ok, detail = _phase_within_unit_support(
            run,
            phase_column=phase_column,
            observation_unit=_observation_unit(run.covariates),
            protected=protected,
            design_columns=design_columns,
        )
        phase_allowed = ok
        phase_detail = detail
        if ok:
            if not is_directed and phase_column not in candidates:
                candidates.append(phase_column)
            run.note(
                kind="phaseHarmonyCandidate",
                detail=detail,
                phaseColumn=phase_column,
            )
        elif not ok:
            run.note(
                kind="phaseHarmonyBlocked",
                detail=detail,
                phaseColumn=phase_column,
            )
    elif allow_phase and isinstance(phase_column, str) and phase_column:
        run.note(
            kind="phaseHarmonyBlocked",
            detail=(
                "Phase Harmony requires protectProliferation=False and a "
                "persistent Stage 3 phase association"
            ),
            phaseColumn=phase_column,
            protectProliferation=protect_proliferation,
        )

    if is_directed:
        for name in candidates:
            if name == phase_column:
                if not phase_allowed:
                    raise ValueError(
                        f"Directed phase Harmony is not allowed: {phase_detail}"
                    )
                continue
            if name not in domains:
                raise ValueError(
                    f"Directed Harmony column {name!r} was not characterized"
                )
            if domains[name] != "technical":
                raise ValueError(
                    f"Directed Harmony column {name!r} has domain "
                    f"{domains[name]!r}; only technical columns are allowed"
                )
            if name not in technical:
                raise ValueError(
                    f"Directed Harmony column {name!r} was not a Stage 3 "
                    "technical diagnostic"
                )
            if kinds.get(name) != "categorical":
                raise ValueError(f"Directed Harmony column {name!r} is not categorical")
            if name in aliases:
                raise ValueError(
                    f"Directed Harmony column {name!r} is an alias, not a "
                    "canonical characterized covariate"
                )

    candidates = list(dict.fromkeys(candidates))
    candidates = [name for name in candidates if name not in aliases]
    nesting = []
    if run.covariates is not None:
        nesting = [
            item
            for item in run.covariates.get("technicalNesting", [])
            if isinstance(item, Mapping)
        ]
    candidates, nest_notes = _drop_nested_duplicates(candidates, nesting)
    for note in nest_notes:
        run.note(**note)

    blocked: list[str] = []
    allowed: list[str] = []
    for name in candidates:
        if name not in run.store.cells.columns:
            if is_directed:
                raise ValueError(f"Directed Harmony column {name!r} is missing")
            blocked.append(name)
            run.note(
                kind="batchColumnMissing",
                detail=f"Batch column {name!r} is not present",
            )
            continue
        domain = domains.get(name)
        if name != phase_column:
            if domain != "technical":
                blocked.append(name)
                run.note(
                    kind="batchColumnNotTechnical",
                    detail=(
                        f"Batch column {name!r} has domain {domain!r}; "
                        "only technical columns or guarded phase are allowed"
                    ),
                    batchColumn=name,
                    domain=domain,
                )
                continue
            if domain is None and _forbidden_batch_name(name):
                blocked.append(name)
                run.note(
                    kind="batchColumnForbidden",
                    detail=f"Batch column {name!r} looks like a cell identity label",
                    batchColumn=name,
                )
                continue
            if kinds.get(name) != "categorical":
                blocked.append(name)
                run.note(
                    kind="batchColumnNotCategorical",
                    detail=f"Batch column {name!r} is not categorical",
                )
                continue
        allowed.append(name)

    if not allowed:
        if blocked:
            run.notes.append(
                "All candidate Harmony batch columns were blocked by design guards"
            )
        return []

    reason = _estimability_blocks_batch(
        run,
        batch_columns=allowed,
        protected=protected,
        kinds=kinds,
        observation_unit=_observation_unit(run.covariates),
    )
    if reason is not None:
        for name in allowed:
            run.note(
                kind="batchColumnBlocked",
                detail=reason,
                batchColumn=name,
            )
        run.notes.append(reason)
        return []
    return allowed


def _build_graph(
    run: _Run,
    coordinates: ArtifactRef,
    *,
    k: int,
    ann_params: Mapping[str, Any],
    connectivity_params: Mapping[str, Any],
    update_state: bool,
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    ann = run.store.build_ann_index(
        coordinates,
        from_assay=run.assay_name,
        update_state=False,
        **dict(ann_params),
    )
    neighbors = run.store.query_neighbors(
        ann,
        coordinates=coordinates,
        k=k,
        update_state=False,
    )
    graph = run.store.build_connectivity_map(
        neighbors,
        update_state=update_state,
        **dict(connectivity_params),
    )
    return ann, neighbors, graph


def _coordinate_summary(
    run: _Run,
    coordinates: ArtifactRef,
    *,
    diagnostic_roster: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    group = artifact_group(run.store.zw, coordinates)
    values = as_zarr_array(group["data"], name="data")
    covariates: dict[str, Any] = {}
    column_kinds: dict[str, Literal["categorical", "continuous"]] = {}
    roles: dict[str, str] = {}
    for record in diagnostic_roster:
        name = record.get("name")
        kind = record.get("kind")
        role = record.get("role")
        if (
            not isinstance(name, str)
            or kind not in {"categorical", "continuous"}
            or role not in {"technical", "nuisance", "protected"}
        ):
            raise ValueError("Invalid Stage 3 diagnostic roster entry")
        if name not in run.store.cells.columns:
            raise ValueError(
                f"Diagnostic covariate {name!r} is missing from cell metadata"
            )
        values_for_column = _selection_values(run, name)
        if len(values_for_column) != values.shape[0]:
            raise ValueError(
                f"Covariate {name!r} length {len(values_for_column)} does not "
                f"match artifact cells {values.shape[0]} for cell_key {run.cell_key!r}"
            )
        covariates[name] = values_for_column
        column_kinds[name] = cast(
            Literal["categorical", "continuous"],
            kind,
        )
        roles[name] = role
    associations = per_pc_covariate_associations(
        values,
        covariates,
        columnKinds=column_kinds,
        associationFloor=_ASSOCIATION_FLOOR,
    )
    summary = branch_nuisance_summary(
        associations,
        technicalCovariates=[
            name for name, role in roles.items() if role == "technical"
        ],
        nuisanceCovariates=[name for name, role in roles.items() if role == "nuisance"],
        protectedCovariates=[
            name for name, role in roles.items() if role == "protected"
        ],
        associationFloor=_ASSOCIATION_FLOOR,
    )
    return {
        "summary": summary,
        "nFlaggedAssociations": sum(1 for item in associations if item.get("flagged")),
        "nCells": int(values.shape[0]),
        "nDims": int(values.shape[1]),
    }


def _association_means(
    coordinate_summary: Mapping[str, Any],
    roles: Sequence[str],
) -> dict[str, float]:
    summary = coordinate_summary.get("summary") or {}
    if not isinstance(summary, Mapping):
        return {}
    values: dict[str, float] = {}
    for role in roles:
        bucket = summary.get(role) or {}
        if not isinstance(bucket, Mapping):
            continue
        by_covariate = bucket.get("byCovariate") or {}
        if not isinstance(by_covariate, Mapping):
            continue
        for name, record in by_covariate.items():
            if isinstance(record, Mapping):
                values[str(name)] = float(record.get("meanAssociation") or 0.0)
        if not by_covariate and "meanAssociation" in bucket:
            values[f"__{role}__"] = float(bucket.get("meanAssociation") or 0.0)
    return values


def _protected_regressions(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[str]:
    base = _association_means(baseline, ("protected",))
    other = _association_means(candidate, ("protected",))
    return [
        name
        for name, value in base.items()
        if name not in other
        or other[name] + _ASSOCIATION_NONREGRESSION_TOLERANCE < value
    ]


def _cheap_gate(
    baseline: Mapping[str, Any],
    harmony: Mapping[str, Any],
) -> tuple[Literal["reject", "continue"], str]:
    base_unwanted = _association_means(baseline, ("technical", "nuisance"))
    harm_unwanted = _association_means(harmony, ("technical", "nuisance"))
    mismatched = set(base_unwanted) != set(harm_unwanted)
    worsened = [
        name
        for name, value in base_unwanted.items()
        if name in harm_unwanted
        and harm_unwanted[name] > max(value * _CHEAP_TECH_WORSEN, value + 0.02)
    ]
    protected_regressed = _protected_regressions(baseline, harmony)
    if mismatched or worsened or protected_regressed:
        details: list[str] = []
        if mismatched:
            details.append("diagnostic covariate sets differ")
        if worsened:
            details.append("unwanted association worsened for " + ", ".join(worsened))
        if protected_regressed:
            details.append(
                "protected association regressed for " + ", ".join(protected_regressed)
            )
        return (
            "reject",
            "Harmony coordinates failed the cheap gate: " + "; ".join(details),
        )
    if base_unwanted:
        base_mean = float(np.mean(list(base_unwanted.values())))
        harm_mean = float(np.mean(list(harm_unwanted.values())))
    else:
        base_mean = harm_mean = 0.0
    if harm_mean < base_mean or not protected_regressed:
        return (
            "continue",
            "Harmony coordinates look non-regressive; continue to graph evaluation",
        )
    return (
        "continue",
        "Harmony coordinate evidence is inconclusive but non-regressive",
    )


def _labels_for_metric(run: _Run, column: str, n_cells: int) -> np.ndarray | None:
    if column not in run.store.cells.columns:
        return None
    labels = _selection_values(run, column)
    if len(labels) != n_cells:
        raise ValueError(
            f"Labels for {column!r} length {len(labels)} do not match "
            f"graph cells {n_cells} for cell_key {run.cell_key!r}"
        )
    if bool(_isna(labels).any()):
        return None
    if len(np.unique(labels)) < 2:
        return None
    return labels


def _sample_support(
    run: _Run,
    sample_column: str | None,
    n_cells: int,
) -> dict[str, Any] | None:
    """Report branch-invariant cell counts for the observation unit."""
    if sample_column is None or sample_column not in run.store.cells.columns:
        return None
    labels = _selection_values(run, sample_column)
    if len(labels) != n_cells:
        raise ValueError(
            f"Sample column {sample_column!r} length {len(labels)} does not "
            f"match artifact cells {n_cells}"
        )
    missing = _isna(labels)
    counts: dict[str, int] = {}
    for value in labels[~missing]:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {
        "definition": "selectedCellCountsByObservationUnit",
        "column": sample_column,
        "nLevels": len(counts),
        "minCells": min(counts.values()) if counts else 0,
        "maxCells": max(counts.values()) if counts else 0,
        "missingCells": int(missing.sum()),
        "counts": counts,
        "branchInvariant": True,
    }


def _harmony_memory_estimate(
    run: _Run,
    *,
    batch_columns: Sequence[str],
    n_cells: int,
    n_dims: int,
    nclust: int,
) -> int:
    level_count = 0
    for name in batch_columns:
        values = _selection_values(run, name)
        level_count += len(np.unique(values[~_isna(values)]))
    float_values = (
        6 * n_cells * n_dims + 6 * n_cells * nclust + 3 * n_cells * (level_count + 1)
    )
    return int(float_values * np.dtype(np.float64).itemsize)


def _graph_metrics(
    run: _Run,
    *,
    neighbors: ArtifactRef,
    graph: ArtifactRef,
    batch_columns: Sequence[str],
    label_column: str | None,
) -> dict[str, Any]:
    neighbor_group = artifact_group(run.store.zw, neighbors)
    indices = as_zarr_array(neighbor_group["indices"], name="indices")
    distances = as_zarr_array(neighbor_group["distances"], name="distances")
    graph_group = artifact_group(run.store.zw, graph)
    edges = as_zarr_array(graph_group["edges"], name="edges")
    n_cells = int(indices.shape[0])

    metrics: dict[str, Any] = {
        "knnLoc": artifact_path(neighbors),
        "graphLoc": artifact_path(graph),
        "nNeighbors": int(indices.shape[1]),
    }
    omitted: list[dict[str, str]] = []
    ilisi_scores: list[float] = []
    for column in batch_columns:
        labels = _labels_for_metric(run, column, n_cells)
        if labels is None:
            omitted.append(
                {
                    "metric": "ilisi",
                    "column": column,
                    "reason": "missingLevelsOrMissingValues",
                }
            )
            continue
        if int(indices.shape[1]) < 3:
            omitted.append(
                {
                    "metric": "ilisi",
                    "column": column,
                    "reason": "neighborKBelowThree",
                }
            )
            continue
        score = float(ilisi_knn(distances, indices, labels))
        ilisi_scores.append(score)
        metrics.setdefault("ilisiByBatch", {})[column] = score
    if ilisi_scores:
        metrics["ilisi"] = float(np.mean(ilisi_scores))

    if label_column:
        labels = _labels_for_metric(run, label_column, n_cells)
        if labels is None:
            omitted.append(
                {
                    "metric": "labelPreservation",
                    "column": label_column,
                    "reason": "missingLevelsOrMissingValues",
                }
            )
        elif int(indices.shape[1]) < 3:
            omitted.append(
                {
                    "metric": "clisi",
                    "column": label_column,
                    "reason": "neighborKBelowThree",
                }
            )
        else:
            metrics["clisi"] = float(clisi_knn(distances, indices, labels))
            metrics["graphConnectivity"] = float(graph_connectivity(edges, labels))
            metrics["labelColumn"] = label_column

    if omitted:
        metrics["omitted"] = omitted
    return metrics


def _select_graph_branch(
    baseline: Mapping[str, Any],
    harmony: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if harmony is None:
        return "uncorrected", "No Harmony branch survived design or cheap gates"
    base_metrics = baseline.get("graphMetrics") or {}
    harm_metrics = harmony.get("graphMetrics") or {}
    base_coords = cast(Mapping[str, Any], baseline["coordinateSummary"])
    harm_coords = cast(Mapping[str, Any], harmony["coordinateSummary"])
    base_unwanted = _association_means(base_coords, ("technical", "nuisance"))
    harm_unwanted = _association_means(harm_coords, ("technical", "nuisance"))
    coordinate_sets_match = set(base_unwanted) == set(harm_unwanted)
    coordinate_non_regression = coordinate_sets_match and all(
        harm_unwanted[name] <= value + _ASSOCIATION_NONREGRESSION_TOLERANCE
        for name, value in base_unwanted.items()
    )
    tech_improved = (
        coordinate_non_regression
        and bool(base_unwanted)
        and float(np.mean(list(harm_unwanted.values())))
        < float(np.mean(list(base_unwanted.values())))
    )

    base_ilisi = base_metrics.get("ilisiByBatch") or {}
    harm_ilisi = harm_metrics.get("ilisiByBatch") or {}
    if isinstance(base_ilisi, Mapping) and isinstance(harm_ilisi, Mapping):
        if base_ilisi or harm_ilisi:
            ilisi_sets_match = set(base_ilisi) == set(harm_ilisi)
            ilisi_non_regression = ilisi_sets_match and all(
                float(harm_ilisi[name]) >= float(value)
                for name, value in base_ilisi.items()
            )
            ilisi_gain = (
                float(np.mean([float(value) for value in harm_ilisi.values()]))
                - float(np.mean([float(value) for value in base_ilisi.values()]))
                if ilisi_sets_match and base_ilisi
                else 0.0
            )
            tech_improved = (
                tech_improved
                and ilisi_non_regression
                and ilisi_gain >= _GRAPH_ILISI_GAIN
            )

    biology_ok = not _protected_regressions(base_coords, harm_coords)
    for key in ("clisi", "graphConnectivity"):
        if key not in base_metrics or key not in harm_metrics:
            continue
        if float(harm_metrics[key]) < float(base_metrics[key]) * _GRAPH_PROTECTED_FLOOR:
            biology_ok = False
            break

    if tech_improved and biology_ok:
        return (
            "harmony",
            "Harmony reduced technical structure without regressing biology",
        )
    return (
        "uncorrected",
        "Harmony evidence conflicted or did not improve technical mixing",
    )


def _preflight(run: _Run, pca_ref: ArtifactRef) -> GraphSelectionResult | None:
    status = run.pca.get("status")
    if status is not None and status != "done":
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[f"pca status must be 'done', got {status!r}"],
            auditLog=[
                {
                    "kind": "invalidPcaStatus",
                    "detail": f"Upstream PCA status is {status!r}",
                }
            ],
        )
    if run.pca.get("selectedPca") is None:
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=["pca result has no selectedPca"],
            auditLog=[
                {
                    "kind": "missingSelectedPca",
                    "detail": "Stage 4 requires a selected PCA artifact",
                }
            ],
        )
    if pca_ref.assay is not None and pca_ref.assay != run.assay_name:
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[
                f"selectedPca assay {pca_ref.assay!r} does not match "
                f"Stage 4 assay {run.assay_name!r}"
            ],
            auditLog=[
                {
                    "kind": "assayMismatch",
                    "detail": "PCA artifact assay does not match Stage 4 assay",
                    "pcaAssay": pca_ref.assay,
                    "stageAssay": run.assay_name,
                }
            ],
        )
    if pca_ref.kind != "reduction" or pca_ref.scope != "assay":
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=["selectedPca must be an assay-scoped reduction artifact"],
            auditLog=[
                {
                    "kind": "invalidSelectedPcaKind",
                    "detail": "selectedPca is not an assay-scoped reduction",
                }
            ],
        )
    selected_id = run.pca.get("selectedBranch")
    if isinstance(selected_id, str):
        selected_records = [
            branch
            for branch in run.pca.get("branches") or []
            if isinstance(branch, Mapping) and branch.get("id") == selected_id
        ]
        if len(selected_records) != 1:
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[
                    f"selectedBranch {selected_id!r} does not identify one PCA branch"
                ],
                auditLog=[
                    {
                        "kind": "invalidSelectedBranch",
                        "detail": "Selected PCA branch is missing or duplicated",
                    }
                ],
            )
        branch_pca = selected_records[0].get("pca")
        if not isinstance(branch_pca, Mapping) or dict(branch_pca) != pca_ref.to_dict():
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=["selectedPca does not match the selected Stage 3 branch"],
                auditLog=[
                    {
                        "kind": "selectedPcaBranchMismatch",
                        "detail": "Selected PCA artifact and branch disagree",
                    }
                ],
            )
    if run.covariates is not None:
        cov_status = run.covariates.get("status")
        if cov_status is not None and cov_status != "done":
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"covariates status must be 'done', got {cov_status!r}"],
                auditLog=[
                    {
                        "kind": "invalidCovariatesStatus",
                        "detail": f"Upstream covariates status is {cov_status!r}",
                    }
                ],
            )
    assay_names = list(getattr(run.store, "assay_names", []))
    if assay_names and run.assay_name not in assay_names:
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[f"Assay {run.assay_name!r} is not present in the store"],
            auditLog=[
                {
                    "kind": "missingAssay",
                    "detail": f"Assay {run.assay_name!r} not in store assays",
                }
            ],
        )
    for key, minimum in (("neighborK", 3),):
        if key not in run.directions:
            continue
        try:
            parsed = int(run.directions[key])
        except (TypeError, ValueError):
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"{key} must be an integer >= {minimum}"],
            )
        if parsed < minimum:
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"{key} must be an integer >= {minimum}"],
            )
    directed = run.directions.get("harmonyBatchColumns")
    if directed is not None and (
        not isinstance(directed, Sequence) or isinstance(directed, str | bytes)
    ):
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=["harmonyBatchColumns must be a sequence of column names"],
            auditLog=[
                {
                    "kind": "invalidHarmonyBatchColumns",
                    "detail": "harmonyBatchColumns must be a sequence",
                }
            ],
        )
    for key in ("annParams", "harmonyParams", "connectivityParams"):
        value = run.directions.get(key)
        if value is not None and not isinstance(value, Mapping):
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"{key} must be a mapping"],
            )
    for key in ("protectSex", "protectProliferation", "allowPhaseHarmony"):
        if key in run.directions and not isinstance(run.directions[key], bool):
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"{key} must be a boolean"],
            )
    try:
        _resolve_cell_key(run.store, pca_ref)
    except ValueError as exc:
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[str(exc)],
            auditLog=[{"kind": "invalidPcaSelection", "detail": str(exc)}],
        )
    return None


def select_graph(
    store: Any,
    *,
    pca: Any,
    covariates: Any | None = None,
    studyContext: str | None = None,
    directions: Mapping[str, Any] | None = None,
    fromAssay: str | None = None,
) -> GraphSelectionResult:
    """Build the uncorrected graph and optionally compare one Harmony branch."""
    pca_map = _as_mapping(pca, name="pca")
    covariate_map = (
        None if covariates is None else _as_mapping(covariates, name="covariates")
    )
    assay_name = (
        fromAssay
        or pca_map.get("assay")
        or getattr(store, "_defaultAssay", None)
        or "RNA"
    )
    run = _Run(
        store=store,
        assay_name=str(assay_name),
        pca=pca_map,
        covariates=covariate_map,
        directions=dict(directions or {}),
        study_context=studyContext,
    )
    pca_diagnostics = pca_map.get("diagnostics") or {}
    if isinstance(pca_diagnostics, Mapping):
        for key in ("protectSex", "protectProliferation"):
            if key in run.directions:
                continue
            value = pca_diagnostics.get(key)
            if isinstance(value, bool):
                run.directions[key] = value
    for key in ("protectSex", "protectProliferation"):
        run.directions.setdefault(key, True)

    pca_ref = _pca_ref(pca_map)
    preflight = _preflight(run, pca_ref)
    if preflight is not None:
        return preflight
    run.cell_key = _resolve_cell_key(store, pca_ref)
    run.note(
        kind="resolvedArtifactCellKey",
        detail=f"Using artifact cell selection {run.cell_key}",
        cellKey=run.cell_key,
    )

    catalog_technical, catalog_protected, kinds, domains, aliases = _covariate_catalog(
        covariate_map
    )
    try:
        diagnostic_roster = _diagnostic_roster(
            pca_map,
            technical=catalog_technical,
            protected=catalog_protected,
            kinds=kinds,
        )
    except ValueError as exc:
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[str(exc)],
            auditLog=[{"kind": "invalidDiagnosticRoster", "detail": str(exc)}],
        )
    for record in diagnostic_roster:
        name = record["name"]
        kind = cast(
            Literal["categorical", "continuous"],
            record["kind"],
        )
        kinds[name] = kind
        if name not in run.store.cells.columns:
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"Diagnostic covariate {name!r} is missing"],
                auditLog=[
                    {
                        "kind": "missingDiagnosticCovariate",
                        "detail": f"Stage 3 diagnostic covariate {name!r} is missing",
                    }
                ],
            )
    technical = [
        record["name"] for record in diagnostic_roster if record["role"] == "technical"
    ]
    nuisance = [
        record["name"] for record in diagnostic_roster if record["role"] == "nuisance"
    ]
    protected = [
        record["name"] for record in diagnostic_roster if record["role"] == "protected"
    ]

    k = int(run.directions.get("neighborK", _DEFAULT_K))
    if k < 3:
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=["neighborK must be an integer >= 3"],
        )
    ann_params = dict(_DEFAULT_ANN)
    directed_ann = run.directions.get("annParams")
    if isinstance(directed_ann, Mapping):
        ann_params.update(dict(directed_ann))
    connectivity_params: dict[str, Any] = {}
    directed_connectivity = run.directions.get("connectivityParams")
    if isinstance(directed_connectivity, Mapping):
        connectivity_params.update(dict(directed_connectivity))

    label_column = run.directions.get("labelColumn")
    if label_column is not None:
        label_column = str(label_column)
        label_domain = domains.get(label_column)
        protected_labels = {
            record["name"]
            for record in diagnostic_roster
            if record["role"] == "protected" and record["kind"] == "categorical"
        }
        if label_domain != "biological" and label_column not in protected_labels:
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[
                    f"labelColumn {label_column!r} must be a characterized "
                    f"biological covariate, got domain {label_domain!r}"
                ],
                auditLog=[
                    {
                        "kind": "invalidLabelColumn",
                        "detail": "labelColumn must have biological domain",
                        "column": label_column,
                        "domain": label_domain,
                    }
                ],
            )
    sample_column = run.directions.get("sampleColumn") or _observation_unit(
        covariate_map
    )
    if isinstance(sample_column, str) and sample_column not in run.store.cells.columns:
        return GraphSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[f"sampleColumn {sample_column!r} is missing"],
            auditLog=[
                {
                    "kind": "missingSampleColumn",
                    "detail": f"Sample column {sample_column!r} is missing",
                }
            ],
        )

    residual_unwanted = _stage3_unwanted_signal_remains(pca_map)
    directed_batches = run.directions.get("harmonyBatchColumns")
    batch_columns: list[str] = []
    if residual_unwanted or directed_batches is not None:
        if covariate_map is None:
            if directed_batches is not None:
                return GraphSelectionResult(
                    status="failed",
                    assay=run.assay_name,
                    notes=[
                        "Characterized covariates are required for directed Harmony"
                    ],
                    auditLog=[
                        {
                            "kind": "missingCovariatesForHarmony",
                            "detail": (
                                "Directed Harmony cannot run without characterized "
                                "domains and kinds"
                            ),
                        }
                    ],
                )
            run.note(
                kind="harmonySkipped",
                detail=(
                    "Residual unwanted association remains, but characterized "
                    "covariates were not supplied"
                ),
            )
            run.notes.append("Harmony skipped: characterized covariates are required")
        else:
            try:
                resolved_batches = _resolve_batch_columns(
                    run,
                    technical=technical,
                    protected=protected,
                    kinds=kinds,
                    domains=domains,
                    aliases=aliases,
                )
            except (TypeError, ValueError) as exc:
                return GraphSelectionResult(
                    status="failed",
                    assay=run.assay_name,
                    notes=[str(exc)],
                    auditLog=run.audit
                    + [{"kind": "invalidHarmonyDesign", "detail": str(exc)}],
                    actions=run.actions,
                    acceptedActions=run.accepted,
                )
            if residual_unwanted:
                batch_columns = resolved_batches
            if residual_unwanted and not batch_columns:
                run.note(
                    kind="harmonySkipped",
                    detail=(
                        "No eligible categorical batch columns after design guards"
                    ),
                )
                run.notes.append("Harmony skipped: no eligible batch columns")
    if not residual_unwanted:
        run.note(
            kind="harmonySkipped",
            detail="No supported Stage 3 unwanted association remains",
        )
        run.notes.append("Harmony skipped: no residual unwanted PCA association")

    # Graph mixing metrics are categorical. Coordinate diagnostics retain the
    # full Stage 3 roster, including continuous technical and biological terms.
    comparison_technical = [
        name
        for name in technical
        if kinds.get(name) == "categorical" and name not in aliases
    ]
    nesting = []
    if covariate_map is not None:
        nesting = [
            item
            for item in covariate_map.get("technicalNesting", [])
            if isinstance(item, Mapping)
        ]
    comparison_technical, _ = _drop_nested_duplicates(comparison_technical, nesting)
    if not comparison_technical and technical:
        comparison_technical = [
            name for name in technical if kinds.get(name) == "categorical"
        ]
    comparison_technical = list(dict.fromkeys([*comparison_technical, *batch_columns]))

    baseline_coords = _coordinate_summary(
        run,
        pca_ref,
        diagnostic_roster=diagnostic_roster,
    )
    resolved_harmony_params: dict[str, Any] = {}
    if batch_columns:
        directed_harmony_params = run.directions.get("harmonyParams")
        if isinstance(directed_harmony_params, Mapping):
            resolved_harmony_params = dict(directed_harmony_params)
        else:
            resolved_harmony_params = {
                "nclust": min(20, max(2, int(baseline_coords["nCells"] // 5)))
            }
        raw_nclust = resolved_harmony_params.get("nclust")
        if raw_nclust is None:
            raw_nclust = min(
                20,
                max(2, int(baseline_coords["nCells"] // 5)),
            )
            resolved_harmony_params["nclust"] = raw_nclust
        if isinstance(raw_nclust, bool) or not isinstance(raw_nclust, int):
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=["harmonyParams.nclust must be an integer"],
                auditLog=run.audit
                + [
                    {
                        "kind": "invalidHarmonyParameters",
                        "detail": "harmonyParams.nclust must be an integer",
                    }
                ],
            )
        nclust = int(raw_nclust)
        if nclust < 1 or nclust > int(baseline_coords["nCells"]):
            return GraphSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=["harmonyParams.nclust is outside the selected cell range"],
                auditLog=run.audit
                + [
                    {
                        "kind": "invalidHarmonyParameters",
                        "detail": "Harmony cluster count is outside the cell range",
                    }
                ],
            )
        memory_estimate = _harmony_memory_estimate(
            run,
            batch_columns=batch_columns,
            n_cells=int(baseline_coords["nCells"]),
            n_dims=int(baseline_coords["nDims"]),
            nclust=nclust,
        )
        memory_limit = int(run.store.memoryBytes)
        run.note(
            kind="harmonyMemoryEstimate",
            detail="Estimated in-memory Harmony working set",
            estimatedBytes=memory_estimate,
            memoryLimitBytes=memory_limit,
        )
        if memory_estimate > memory_limit:
            run.note(
                kind="harmonySkipped",
                detail="Harmony estimate exceeds the datastore memory budget",
                estimatedBytes=memory_estimate,
                memoryLimitBytes=memory_limit,
            )
            run.notes.append("Harmony skipped: estimated memory exceeds budget")
            batch_columns = []
    sample_support = _sample_support(
        run,
        sample_column if isinstance(sample_column, str) else None,
        int(baseline_coords["nCells"]),
    )
    ann, neighbors, graph = _build_graph(
        run,
        pca_ref,
        k=k,
        ann_params=ann_params,
        connectivity_params=connectivity_params,
        update_state=False,
    )
    run.actions.append("buildUncorrectedGraph")
    baseline_metrics = _graph_metrics(
        run,
        neighbors=neighbors,
        graph=graph,
        batch_columns=comparison_technical,
        label_column=label_column,
    )
    baseline_branch: dict[str, Any] = {
        "id": "uncorrected",
        "coordinates": _artifact_dict(pca_ref),
        "annIndex": _artifact_dict(ann),
        "neighbors": _artifact_dict(neighbors),
        "graph": _artifact_dict(graph),
        "coordinateSummary": baseline_coords,
        "graphMetrics": baseline_metrics,
        "sampleSupport": sample_support,
    }
    run.accepted.append(
        {
            "action": "buildUncorrectedGraph",
            "neighbors": baseline_branch["neighbors"],
            "graph": baseline_branch["graph"],
            "knnLoc": baseline_metrics.get("knnLoc"),
            "graphLoc": baseline_metrics.get("graphLoc"),
        }
    )

    harmony_branch: dict[str, Any] | None = None
    if batch_columns:
        try:
            correction = run.store.run_harmony(
                list(batch_columns),
                pca_ref,
                from_assay=run.assay_name,
                harmony_params=resolved_harmony_params,
                update_state=False,
            )
        except (MemoryError, TypeError, ValueError) as exc:
            run.note(
                kind="harmonyFailed",
                detail=str(exc),
                batchColumns=batch_columns,
            )
            run.notes.append(f"Harmony branch failed: {exc}")
            batch_columns = []
        else:
            run.actions.append("runHarmony")
            harmony_coords = _coordinate_summary(
                run,
                correction,
                diagnostic_roster=diagnostic_roster,
            )
            gate, gate_detail = _cheap_gate(baseline_coords, harmony_coords)
            run.note(
                kind="harmonyCheapGate",
                detail=gate_detail,
                decision=gate,
                batchColumns=batch_columns,
                comparisonTechnical=comparison_technical,
            )
            if gate == "reject":
                run.notes.append(gate_detail)
                harmony_branch = {
                    "id": "harmony",
                    "coordinates": _artifact_dict(correction),
                    "batchColumns": batch_columns,
                    "coordinateSummary": harmony_coords,
                    "sampleSupport": sample_support,
                    "cheapGate": {"decision": gate, "detail": gate_detail},
                    "rejectedBeforeGraph": True,
                }
            else:
                harm_ann, harm_neighbors, harm_graph = _build_graph(
                    run,
                    correction,
                    k=k,
                    ann_params=ann_params,
                    connectivity_params=connectivity_params,
                    update_state=False,
                )
                run.actions.append("buildHarmonyGraph")
                harm_metrics = _graph_metrics(
                    run,
                    neighbors=harm_neighbors,
                    graph=harm_graph,
                    batch_columns=comparison_technical,
                    label_column=label_column,
                )
                harmony_branch = {
                    "id": "harmony",
                    "coordinates": _artifact_dict(correction),
                    "annIndex": _artifact_dict(harm_ann),
                    "neighbors": _artifact_dict(harm_neighbors),
                    "graph": _artifact_dict(harm_graph),
                    "batchColumns": batch_columns,
                    "coordinateSummary": harmony_coords,
                    "graphMetrics": harm_metrics,
                    "sampleSupport": sample_support,
                    "cheapGate": {"decision": gate, "detail": gate_detail},
                    "rejectedBeforeGraph": False,
                }
                run.accepted.append(
                    {
                        "action": "buildHarmonyGraph",
                        "batchColumns": batch_columns,
                        "neighbors": harmony_branch["neighbors"],
                        "graph": harmony_branch["graph"],
                        "knnLoc": harm_metrics.get("knnLoc"),
                        "graphLoc": harm_metrics.get("graphLoc"),
                    }
                )

    selected_id, rationale = _select_graph_branch(
        baseline_branch,
        None
        if harmony_branch is None or harmony_branch.get("rejectedBeforeGraph")
        else harmony_branch,
    )
    if selected_id == "harmony" and harmony_branch is not None:
        selected_branch = harmony_branch
    else:
        selected_id = "uncorrected"
        selected_branch = baseline_branch
        if harmony_branch is not None and harmony_branch.get("rejectedBeforeGraph"):
            rationale = (
                cast(dict[str, Any], harmony_branch.get("cheapGate") or {}).get(
                    "detail"
                )
                or "Harmony rejected by cheap coordinate gate"
            )

    selected_neighbors = ArtifactRef.from_dict(selected_branch["neighbors"])
    run.store.build_connectivity_map(
        selected_neighbors,
        update_state=True,
        **connectivity_params,
    )
    run.actions.append(f"selectGraph:{selected_id}")
    run.accepted.append(
        {
            "action": "selectGraph",
            "branch": selected_id,
            "graph": selected_branch["graph"],
            "neighbors": selected_branch["neighbors"],
            "coordinates": selected_branch["coordinates"],
            "rationale": rationale,
        }
    )
    run.decisions.append(
        Decision(
            selectedId=selected_id,
            rationale=rationale,
            evidenceIds=[selected_id],
        ).model_dump()
    )

    branches = [baseline_branch]
    if harmony_branch is not None:
        branches.append(harmony_branch)

    return GraphSelectionResult(
        status="done",
        assay=run.assay_name,
        auditLog=run.audit,
        actions=run.actions,
        notes=run.notes,
        decisions=run.decisions,
        acceptedActions=run.accepted,
        branches=branches,
        selectedBranch=selected_id,
        selectedCoordinates=cast(dict[str, Any], selected_branch["coordinates"]),
        selectedNeighbors=cast(dict[str, Any], selected_branch["neighbors"]),
        selectedGraph=cast(dict[str, Any], selected_branch["graph"]),
        harmonyBatchColumns=batch_columns,
        diagnostics={
            "associationFloor": _ASSOCIATION_FLOOR,
            "diagnosticCovariates": diagnostic_roster,
            "technicalCovariates": technical,
            "nuisanceCovariates": nuisance,
            "protectedCovariates": protected,
            "comparisonTechnicalCovariates": comparison_technical,
            "neighborK": k,
            "annParams": ann_params,
            "connectivityParams": connectivity_params,
            "labelColumn": label_column,
            "cellKey": run.cell_key,
            "protectSex": bool(run.directions.get("protectSex", True)),
            "protectProliferation": bool(
                run.directions.get("protectProliferation", True)
            ),
            "sampleSupport": sample_support,
            "selectedGraphMetrics": selected_branch.get("graphMetrics"),
            "selectedCoordinateSummary": selected_branch.get("coordinateSummary"),
        },
    )
