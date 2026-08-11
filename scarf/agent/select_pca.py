"""Select an HVG and PCA branch with one optional nuisance-feature backtrack."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Concatenate, Literal, ParamSpec, cast

import numpy as np

from ..metadata.artifacts import artifact_values
from ..metrics.pca_diagnostics import (
    branch_nuisance_summary,
    family_loading_concentration,
    per_pc_covariate_associations,
)
from ..quality_control.cell_cycle_genes import (
    g2m_phase_genes,
    g2m_phase_genes_mouse,
    s_phase_genes,
    s_phase_genes_mouse,
)
from ..storage.artifacts import artifact_group, fingerprint_array
from ..storage.refs import ArtifactRef
from ..storage.types import as_zarr_array
from ._deps import AGENT_INSTALL_HINT
from .decide import DecisionValidationError, decide
from .types import Decision, EvidenceItem, NeedsInput, StageStatus

try:
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "PcaSelectionResult",
    "select_pca",
]

_ASSOCIATION_FLOOR = 0.1
_FAMILY_SHARE_FLOOR = 0.25
_DEFAULT_PCA_DIMS = 25
_DEFAULT_TOP_N = 2000
_DEFAULT_HVG_KEY = "hvgs"
_BASE_EXCLUDE_FAMILIES = ("mitochondrial", "ribosomal", "histone")
_ASSOCIATION_NONREGRESSION_TOLERANCE = 0.01
_CELL_CYCLE = {
    "homo_sapiens": {"s": s_phase_genes, "g2m": g2m_phase_genes},
    "mus_musculus": {"s": s_phase_genes_mouse, "g2m": g2m_phase_genes_mouse},
}
_P = ParamSpec("_P")


class PcaSelectionResult(BaseModel):
    status: StageStatus
    auditLog: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    needsInput: NeedsInput | None = None
    acceptedActions: list[dict[str, Any]] = Field(default_factory=list)
    assay: str | None = None
    species: str | None = None
    qcRetention: dict[str, Any] = Field(default_factory=dict)
    cellCycle: dict[str, Any] = Field(default_factory=dict)
    branches: list[dict[str, Any]] = Field(default_factory=list)
    selectedBranch: str | None = None
    selectedPca: dict[str, Any] | None = None
    selectedHvgs: dict[str, Any] | None = None
    blacklistIndexes: list[int] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _Run:
    store: Any
    assay_name: str
    features: Mapping[str, Any]
    covariates: Mapping[str, Any] | None
    directions: dict[str, Any]
    model: Any | None
    study_context: str | None
    audit: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_input: NeedsInput | None = None

    def note(self, *, kind: str, detail: str, **fields: Any) -> None:
        self.audit.append({"kind": kind, "detail": detail, **fields})


def _restore_selection_on_failure(
    function: Callable[Concatenate[Any, _P], PcaSelectionResult],
) -> Callable[Concatenate[Any, _P], PcaSelectionResult]:
    @wraps(function)
    def wrapper(
        store: Any,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> PcaSelectionResult:
        initial = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()

        def _restore() -> bool:
            current = np.asarray(store.cells.fetch_all("I"), dtype=bool)
            if np.array_equal(current, initial):
                return False
            store.cells.reset_key(key="I")
            store.cells.update_key(initial, key="I")
            return True

        try:
            result = function(store, *args, **kwargs)
        except BaseException:
            _restore()
            raise
        if result.status != "done" and _restore():
            result.auditLog.append(
                {
                    "kind": "selectionRolledBack",
                    "detail": (
                        "Restored the caller's cell selection because Stage 3 "
                        "did not complete"
                    ),
                }
            )
        return result

    return wrapper


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if hasattr(value, "model_dump"):
        return cast(Mapping[str, Any], value.model_dump())
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{name} must be a mapping or pydantic model")


def _assay_record(
    features: Mapping[str, Any],
    assay_name: str,
) -> Mapping[str, Any] | None:
    for assay in features.get("assays", []):
        if not isinstance(assay, Mapping):
            continue
        if assay.get("assay") == assay_name:
            return assay
    return None


def _family_indexes(assay: Mapping[str, Any]) -> dict[str, list[int]]:
    indexes: dict[str, list[int]] = {}
    for family in assay.get("families", []):
        if not isinstance(family, Mapping):
            continue
        name = family.get("family")
        raw = family.get("featureIndexes")
        if name is None or raw is None:
            continue
        indexes[str(name)] = [int(index) for index in raw]
    return indexes


def _merge_indexes(*groups: Sequence[int]) -> list[int]:
    merged: set[int] = set()
    for group in groups:
        merged.update(int(index) for index in group)
    return sorted(merged)


def _covariate_columns(
    covariates: Mapping[str, Any] | None,
) -> tuple[list[str], list[str], dict[str, Literal["categorical", "continuous"]]]:
    technical: list[str] = []
    protected: list[str] = []
    kinds: dict[str, Literal["categorical", "continuous"]] = {}
    if covariates is None:
        return technical, protected, kinds
    for column in covariates.get("columns", []):
        if not isinstance(column, Mapping):
            continue
        name = column.get("name")
        domain = column.get("domain")
        kind = column.get("kind")
        if name is None or kind not in {"categorical", "continuous"}:
            continue
        kinds[str(name)] = cast(Literal["categorical", "continuous"], kind)
        if domain == "technical":
            technical.append(str(name))
        elif domain == "biological":
            protected.append(str(name))
    coefficients = {
        str(item.get("name") or item.get("column"))
        for item in covariates.get("coefficients", [])
        if isinstance(item, Mapping)
        and (item.get("name") is not None or item.get("column") is not None)
    }
    if coefficients:
        protected = [name for name in protected if name in coefficients] or protected
    return technical, protected, kinds


def _coefficient_names(covariates: Mapping[str, Any] | None) -> set[str]:
    if covariates is None:
        return set()
    return {
        str(item.get("name") or item.get("column"))
        for item in covariates.get("coefficients", [])
        if isinstance(item, Mapping)
        and (item.get("name") is not None or item.get("column") is not None)
    }


def _is_sex_column(name: str) -> bool:
    parts = [
        part
        for part in re.split(
            r"[^a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name).lower()
        )
        if part
    ]
    return "sex" in parts or "gender" in parts


def _sex_columns(
    covariates: Mapping[str, Any] | None,
    kinds: Mapping[str, Literal["categorical", "continuous"]],
) -> list[str]:
    if covariates is None:
        return []
    return sorted(
        {
            str(column["name"])
            for column in covariates.get("columns", [])
            if isinstance(column, Mapping)
            and column.get("name") is not None
            and str(column["name"]) in kinds
            and _is_sex_column(str(column["name"]))
        }
    )


def _diagnostic_roster(
    *,
    technical: Sequence[str],
    protected: Sequence[str],
    kinds: Mapping[str, Literal["categorical", "continuous"]],
    sex_columns: Sequence[str],
    cell_cycle: Mapping[str, Any],
    protect_sex: bool,
    protect_proliferation: bool,
) -> list[dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for name in technical:
        kind = kinds.get(name)
        if kind is not None:
            entries[name] = {
                "name": name,
                "kind": kind,
                "role": "technical",
                "source": "covariateCharacterization",
            }
    for name in protected:
        kind = kinds.get(name)
        if kind is not None:
            entries[name] = {
                "name": name,
                "kind": kind,
                "role": "protected",
                "source": "covariateCharacterization",
            }
    for name in sex_columns:
        kind = kinds.get(name)
        if kind is not None:
            entries[name] = {
                "name": name,
                "kind": kind,
                "role": "protected" if protect_sex else "nuisance",
                "source": "sexMetadata",
            }
    for key, kind in (
        ("sScoreColumn", "continuous"),
        ("g2mScoreColumn", "continuous"),
        ("phaseColumn", "categorical"),
    ):
        column = cell_cycle.get(key)
        if isinstance(column, str):
            entries[column] = {
                "name": column,
                "kind": cast(
                    Literal["categorical", "continuous"],
                    kind,
                ),
                "role": "protected" if protect_proliferation else "nuisance",
                "source": "cellCycleScoring",
            }
    return [entries[name] for name in sorted(entries)]


def _sample_column(
    run: _Run,
    technical: Sequence[str],
) -> str | None:
    directed = run.directions.get("sampleColumn")
    if isinstance(directed, str) and directed:
        return directed
    if run.covariates is None:
        return None
    units = []
    for report in run.covariates.get("confounding", []):
        if not isinstance(report, Mapping):
            continue
        unit = report.get("observationUnit")
        if isinstance(unit, str) and unit:
            units.append(unit)
    unique_units = sorted(set(units))
    if len(unique_units) == 1:
        return unique_units[0]
    if len(unique_units) > 1:
        run.needs_input = NeedsInput(
            question=(
                "Multiple observation units are available for sample-aware QC. "
                "Which column should define samples?"
            ),
            options=unique_units,
        )
        run.note(
            kind="ambiguousSampleColumn",
            detail="Multiple observation units available for QC",
            options=unique_units,
        )
        return None
    # Do not guess among technical columns; require an explicit sample unit.
    if technical:
        run.needs_input = NeedsInput(
            question=(
                "No observation unit was inferred for sample-aware QC. "
                "Which column should define samples?"
            ),
            options=list(technical),
        )
        run.note(
            kind="missingSampleColumn",
            detail="Technical columns present but no observation unit resolved",
            options=list(technical),
        )
    return None


def _stage3_input_column(assay_name: str) -> str:
    return f"{assay_name}_stage3Input"


def _resolve_input_selection(run: _Run) -> np.ndarray:
    """Return the Stage 3 input mask, persisting it for idempotent reruns."""
    column = _stage3_input_column(run.assay_name)
    refresh = bool(run.directions.get("refreshStage3Input", False))
    if column in run.store.cells.columns and not refresh:
        mask = np.asarray(run.store.cells.fetch_all(column), dtype=bool)
        run.note(
            kind="reusedStage3Input",
            detail=f"Reused persisted Stage 3 input selection {column}",
            column=column,
            nCells=int(mask.sum()),
        )
        return mask
    mask = np.asarray(run.store.cells.fetch_all("I"), dtype=bool).copy()
    run.store.cells.insert(column, mask, overwrite=True)
    run.note(
        kind="recordedStage3Input",
        detail=f"Persisted Stage 3 input selection to {column}",
        column=column,
        nCells=int(mask.sum()),
    )
    return mask


def _restore_input_selection(run: _Run, input_mask: np.ndarray) -> None:
    run.store.cells.reset_key(key="I")
    run.store.cells.update_key(input_mask, key="I")


def _preflight(
    run: _Run,
    *,
    assay_info: Mapping[str, Any],
) -> PcaSelectionResult | None:
    """Validate inputs before any store mutation. Returns a result on failure."""
    status = run.features.get("status")
    if status is not None and status != "done":
        return PcaSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[f"features status must be 'done', got {status!r}"],
            auditLog=[
                {
                    "kind": "invalidFeaturesStatus",
                    "detail": f"Upstream features status is {status!r}",
                }
            ],
        )
    if run.covariates is not None:
        cov_status = run.covariates.get("status")
        if cov_status is not None and cov_status != "done":
            return PcaSelectionResult(
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
        return PcaSelectionResult(
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
    try:
        assay = run.store._get_assay(run.assay_name)
    except Exception as exc:
        return PcaSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[str(exc)],
            auditLog=[{"kind": "assayLookupFailed", "detail": str(exc)}],
        )
    from ..assay import RNAassay

    if not isinstance(assay, RNAassay):
        return PcaSelectionResult(
            status="failed",
            assay=run.assay_name,
            notes=[f"Stage 3 requires an RNAassay, got {type(assay).__name__}"],
            auditLog=[
                {
                    "kind": "invalidAssayType",
                    "detail": f"Assay type {type(assay).__name__} is not RNA",
                }
            ],
        )
    sample_column = run.directions.get("sampleColumn")
    if isinstance(sample_column, str) and sample_column:
        if sample_column not in run.store.cells.columns:
            return PcaSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[
                    f"sampleColumn {sample_column!r} is not present in cell metadata"
                ],
                auditLog=[
                    {
                        "kind": "missingSampleColumn",
                        "detail": f"Directed sampleColumn {sample_column!r} missing",
                    }
                ],
            )
    for key, minimum in (("pcaDims", 2), ("topN", 1)):
        if key not in run.directions:
            continue
        value = run.directions[key]
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return PcaSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"{key} must be an integer >= {minimum}"],
            )
        if parsed < minimum:
            return PcaSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"{key} must be an integer >= {minimum}"],
            )
    for key in ("protectSex", "protectProliferation", "refreshStage3Input"):
        if key in run.directions and not isinstance(run.directions[key], bool):
            return PcaSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=[f"{key} must be a boolean"],
            )
    nuisance = run.directions.get("nuisanceGeneIndexes")
    if nuisance is not None:
        try:
            indexes = [int(index) for index in nuisance]
        except (TypeError, ValueError):
            return PcaSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=["nuisanceGeneIndexes must be a sequence of integers"],
            )
        n_features = int(assay.feats.N)
        if any(index < 0 or index >= n_features for index in indexes):
            return PcaSelectionResult(
                status="failed",
                assay=run.assay_name,
                notes=["nuisanceGeneIndexes contains an out-of-range index"],
            )
    _ = assay_info
    return None


def _recompute_percentages(
    run: _Run,
    family_indexes: Mapping[str, Sequence[int]],
) -> dict[str, bool]:
    assay = run.store._get_assay(run.assay_name)
    feature_sets: dict[str, list[int]] = {}
    mito = list(family_indexes.get("mitochondrial", []))
    ribo = list(family_indexes.get("ribosomal", []))
    if mito:
        feature_sets[f"{run.assay_name}_percentMito"] = mito
    if ribo:
        feature_sets[f"{run.assay_name}_percentRibo"] = ribo
    if not feature_sets:
        run.note(
            kind="percentFeaturesSkipped",
            detail="No mitochondrial or ribosomal indexes available",
        )
        return {}
    recomputed = cast(
        dict[str, bool],
        assay.add_percent_features_by_indexes(feature_sets),
    )
    run.actions.append("recomputeExactPercentages")
    run.accepted.append(
        {
            "action": "recomputeExactPercentages",
            "featureSets": {
                name: list(indexes) for name, indexes in feature_sets.items()
            },
            "recomputed": recomputed,
        }
    )
    run.note(
        kind="percentFeatures",
        detail="Recomputed exact mitochondrial and ribosomal percentages",
        recomputed=recomputed,
    )
    return recomputed


def _run_qc(run: _Run, sample_column: str | None) -> dict[str, Any]:
    before = int(np.asarray(run.store.cells.fetch_all("I"), dtype=bool).sum())
    attrs = []
    for suffix in ("nCounts", "nFeatures", "percentMito", "percentRibo"):
        column = f"{run.assay_name}_{suffix}"
        if column in run.store.cells.columns:
            attrs.append(column)
    run.store.auto_filter_cells(
        attrs=attrs or None,
        show_qc_plots=False,
        sample_column=sample_column,
    )
    after = int(np.asarray(run.store.cells.fetch_all("I"), dtype=bool).sum())
    retention = {
        "inputCells": before,
        "retainedCells": after,
        "sampleColumn": sample_column,
        "attrs": attrs,
    }
    run.actions.append("autoFilterCells")
    run.accepted.append({"action": "autoFilterCells", **retention})
    run.note(
        kind="qcRetention",
        detail=f"Retained {after}/{before} cells after sample-aware QC",
        **retention,
    )
    return retention


def _cell_cycle_genes(
    species: str,
    reference_symbols: set[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    catalog = _CELL_CYCLE.get(species)
    if catalog is None:
        return [], [], list(_CELL_CYCLE)
    s_genes = list(catalog["s"])
    g2m_genes = list(catalog["g2m"])
    dropped: list[str] = []
    if reference_symbols is not None:
        filtered_s = [gene for gene in s_genes if gene in reference_symbols]
        filtered_g2m = [gene for gene in g2m_genes if gene in reference_symbols]
        dropped = sorted(
            set(s_genes)
            .union(g2m_genes)
            .difference(filtered_s)
            .difference(filtered_g2m)
        )
        s_genes = filtered_s
        g2m_genes = filtered_g2m
    return s_genes, g2m_genes, dropped


def _score_cell_cycle(run: _Run, species: str) -> dict[str, Any]:
    assay = run.store._get_assay(run.assay_name)
    present_symbols = {
        str(name) for name in np.asarray(assay.feats.fetch_all("names")) if name
    }
    s_genes, g2m_genes, dropped = _cell_cycle_genes(species, present_symbols)
    result: dict[str, Any] = {
        "species": species,
        "nSGenes": len(s_genes),
        "nG2mGenes": len(g2m_genes),
        "droppedCatalogSymbols": dropped[:25],
        "nDroppedCatalogSymbols": len(dropped),
    }
    if not s_genes or not g2m_genes:
        run.note(
            kind="cellCycleSkipped",
            detail="Insufficient species-matched cell-cycle genes",
            **result,
        )
        return result
    n_features = int(assay.feats.N)
    n_bins = min(50, max(5, n_features // 4))
    try:
        run.store.run_cell_cycle_scoring(
            from_assay=run.assay_name,
            cell_key="I",
            s_genes=s_genes,
            g2m_genes=g2m_genes,
            n_bins=n_bins,
        )
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        result["error"] = str(exc)
        run.note(
            kind="cellCycleFailed",
            detail="Cell-cycle scoring failed; continuing without phase covariates",
            error=str(exc),
            **{key: value for key, value in result.items() if key != "error"},
        )
        return result
    s_label = run.store._col_renamer(run.assay_name, "I", "S_score")
    g2m_label = run.store._col_renamer(run.assay_name, "I", "G2M_score")
    phase_label = run.store._col_renamer(run.assay_name, "I", "cell_cycle_phase")
    result.update(
        {
            "sScoreColumn": s_label,
            "g2mScoreColumn": g2m_label,
            "phaseColumn": phase_label,
            "nBins": n_bins,
        }
    )
    run.actions.append("runCellCycleScoring")
    run.accepted.append({"action": "runCellCycleScoring", **result})
    run.note(
        kind="cellCycleScored",
        detail="Scored S and G2M activity on retained cells",
        **result,
    )
    return result


def _mark_hvgs(
    run: _Run,
    *,
    blacklist_indexes: Sequence[int],
    hvg_key_name: str,
) -> str:
    top_n = int(run.directions.get("topN", _DEFAULT_TOP_N))
    max_cells = run.directions.get("maxCells")
    mark_kwargs: dict[str, Any] = {
        "from_assay": run.assay_name,
        "cell_key": "I",
        "top_n": top_n,
        "blacklist": "",
        "blacklist_indexes": list(blacklist_indexes),
        "hvg_key_name": hvg_key_name,
        "show_plot": False,
    }
    if max_cells is not None:
        mark_kwargs["max_cells"] = max_cells
    run.store.mark_hvgs(**mark_kwargs)
    stored_key = f"I__{hvg_key_name}"
    assay = run.store._get_assay(run.assay_name)
    n_hvgs = int(np.asarray(assay.feats.fetch_all(stored_key), dtype=bool).sum())
    run.note(
        kind="hvgSelection",
        detail=f"Marked {n_hvgs} HVGs under {stored_key}",
        hvgKey=stored_key,
        nHvgs=n_hvgs,
        blacklistIndexes=list(blacklist_indexes),
    )
    return stored_key


def _run_pca_branch(
    run: _Run,
    *,
    feat_key: str,
    dims: int,
) -> tuple[ArtifactRef, ArtifactRef]:
    normalized = run.store.run_normalization(
        from_assay=run.assay_name,
        cell_key="I",
        feat_key=feat_key,
        update_state=False,
    )
    pca = run.store.run_pca(
        normalized,
        from_assay=run.assay_name,
        dims=dims,
        show_elbow_plot=False,
        update_state=False,
    )
    return normalized, pca


def _artifact_dict(ref: ArtifactRef) -> dict[str, Any]:
    return ref.to_dict()


def _diagnose_branch(
    run: _Run,
    *,
    pca_ref: ArtifactRef,
    feat_key: str,
    family_indexes: Mapping[str, Sequence[int]],
    diagnostic_roster: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    group = artifact_group(run.store.zw, pca_ref)
    coordinates = as_zarr_array(group["data"], name="data")
    loadings = artifact_values(group, "loadings")
    assay = run.store._get_assay(run.assay_name)
    hvg_mask = np.asarray(assay.feats.fetch_all(feat_key), dtype=bool)
    feature_indexes = np.where(hvg_mask)[0].astype(np.int64)
    feature_names = np.asarray(assay.feats.fetch_all("names"))[feature_indexes]

    covariates: dict[str, Any] = {}
    column_kinds: dict[str, Literal["categorical", "continuous"]] = {}
    roles: dict[str, str] = {}
    for record in diagnostic_roster:
        name = record.get("name")
        kind = record.get("kind")
        role = record.get("role")
        if (
            not isinstance(name, str)
            or name not in run.store.cells.columns
            or kind not in {"categorical", "continuous"}
            or role not in {"technical", "nuisance", "protected"}
        ):
            continue
        values = np.asarray(run.store.cells.fetch(name, key="I"))
        if values.shape[0] != coordinates.shape[0]:
            raise ValueError(
                f"Covariate {name!r} has {values.shape[0]} selected values, "
                f"but PCA artifact has {coordinates.shape[0]} cells"
            )
        covariates[name] = values
        column_kinds[name] = cast(
            Literal["categorical", "continuous"],
            kind,
        )
        roles[name] = role

    associations = per_pc_covariate_associations(
        coordinates,
        covariates,
        columnKinds=column_kinds,
        associationFloor=_ASSOCIATION_FLOOR,
    )
    loading_reports = family_loading_concentration(
        loadings,
        featureIndexes=feature_indexes.tolist(),
        familyIndexes=family_indexes,
        featureNames=[str(name) for name in feature_names],
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
        "associations": associations,
        "loadings": loading_reports,
        "summary": summary,
        "nCells": int(coordinates.shape[0]),
        "nDims": int(coordinates.shape[1]),
        "nFeatures": int(feature_indexes.size),
    }


def _candidate_nuisance_indexes(
    diagnostics: Mapping[str, Any],
    family_indexes: Mapping[str, Sequence[int]],
    *,
    protect_sex: bool,
    protect_proliferation: bool,
    nuisance_covariates: Sequence[str] = (),
    directed_indexes: Sequence[int] = (),
) -> tuple[list[int], list[str]]:
    blocked_families: list[str] = []
    if not protect_sex:
        blocked_families.append("sex")
    if not protect_proliferation:
        blocked_families.append("cellCycle")

    selected: set[int] = {int(index) for index in directed_indexes}
    used_families: list[str] = []
    if directed_indexes:
        used_families.append("directed")

    if not blocked_families:
        return sorted(selected), used_families

    # Only blacklist family genes when they concentrate on a PC that is also
    # flagged for an explicitly unwanted technical or biological association.
    nuisance = set(nuisance_covariates)
    flagged_pcs = {
        int(record["pc"])
        for record in diagnostics.get("associations", [])
        if (
            isinstance(record, Mapping)
            and record.get("flagged")
            and str(record.get("covariate")) in nuisance
        )
    }
    for report in diagnostics.get("loadings", []):
        if not isinstance(report, Mapping):
            continue
        pc = int(report.get("pc", -1))
        if pc not in flagged_pcs:
            continue
        shares = report.get("familyShares") or {}
        for family in blocked_families:
            share = float(shares.get(family, 0.0))
            if share < _FAMILY_SHARE_FLOOR:
                continue
            indexes = family_indexes.get(family, [])
            if not indexes:
                continue
            selected.update(int(index) for index in indexes)
            if family not in used_families:
                used_families.append(family)
    return sorted(selected), used_families


def _bounded_context(study_context: str | None) -> str:
    text = (study_context or "").strip()
    if len(text) <= 1200:
        return text
    return text[:1197] + "..."


def _decide_protection(
    run: _Run,
    *,
    family: str,
    evidence_summary: str,
) -> bool:
    """Return True when the family should remain protected."""
    directed = run.directions.get(
        "protectSex" if family == "sex" else "protectProliferation"
    )
    if isinstance(directed, bool):
        return directed
    if run.model is None:
        # Conservative default: keep sex and proliferation.
        return True
    context = _bounded_context(run.study_context)
    context_note = f" Study context: {context}" if context else ""
    options = [
        EvidenceItem(
            id=f"{family}:protect",
            label="protect",
            summary=f"Keep {family} genes; {evidence_summary}.{context_note}",
        ),
        EvidenceItem(
            id=f"{family}:exclude",
            label="exclude",
            summary=(
                f"Allow excluding {family} genes for one nuisance-filtered PCA "
                f"branch; {evidence_summary}.{context_note}"
            ),
        ),
    ]
    try:
        decision = decide(
            model=run.model,
            question=(
                "Should this gene family stay protected for the study objective, "
                "or may it be excluded once to test nuisance reduction?"
            ),
            evidence=options,
        )
    except DecisionValidationError as exc:
        run.note(
            kind="protectionDecisionInvalid",
            detail=str(exc),
            family=family,
        )
        return True
    run.decisions.append(decision.model_dump())
    return decision.selectedId.endswith(":protect")


def _select_branch(
    original: Mapping[str, Any],
    alternate: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if alternate is None:
        return "original", "No nuisance-filtered branch was created"
    original_summary = original["diagnostics"]["summary"]
    alternate_summary = alternate["diagnostics"]["summary"]

    def _association_means(
        summary: Mapping[str, Any],
        roles: Sequence[str],
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for role in roles:
            bucket = summary.get(role, {})
            if not isinstance(bucket, Mapping):
                continue
            by_covariate = bucket.get("byCovariate", {})
            if not isinstance(by_covariate, Mapping):
                continue
            for name, record in by_covariate.items():
                if isinstance(record, Mapping):
                    values[str(name)] = float(record.get("meanAssociation", 0.0))
        return values

    original_unwanted = _association_means(
        original_summary,
        ("technical", "nuisance"),
    )
    alternate_unwanted = _association_means(
        alternate_summary,
        ("technical", "nuisance"),
    )
    unwanted_improved = (
        bool(original_unwanted)
        and set(original_unwanted) == set(alternate_unwanted)
        and all(
            alternate_unwanted[name]
            <= original_value + _ASSOCIATION_NONREGRESSION_TOLERANCE
            for name, original_value in original_unwanted.items()
        )
        and (
            float(np.mean(list(alternate_unwanted.values())))
            < float(np.mean(list(original_unwanted.values())))
        )
    )
    original_protected = _association_means(original_summary, ("protected",))
    alternate_protected = _association_means(alternate_summary, ("protected",))
    regressed = [
        name
        for name, original_value in original_protected.items()
        if name not in alternate_protected
        or alternate_protected[name] + _ASSOCIATION_NONREGRESSION_TOLERANCE
        < original_value
    ]
    if unwanted_improved and not regressed:
        return (
            "nuisanceFiltered",
            "Nuisance association decreased without weakening protected biology",
        )
    if regressed:
        return (
            "original",
            "Nuisance-filtered branch weakened protected covariates: "
            + ", ".join(regressed),
        )
    return (
        "original",
        "Nuisance-filtered branch did not improve the branch-level tradeoff",
    )


@_restore_selection_on_failure
def select_pca(
    store: Any,
    *,
    features: Any,
    covariates: Any | None = None,
    studyContext: str | None = None,
    model: Any | None = None,
    directions: Mapping[str, Any] | None = None,
    fromAssay: str | None = None,
) -> PcaSelectionResult:
    """Run QC, HVG selection, PCA, and one optional nuisance-feature backtrack."""
    feature_map = _as_mapping(features, name="features")
    covariate_map = (
        None if covariates is None else _as_mapping(covariates, name="covariates")
    )
    assay_name = fromAssay or getattr(store, "_defaultAssay", None) or "RNA"
    assay_info = _assay_record(feature_map, assay_name)
    run = _Run(
        store=store,
        assay_name=assay_name,
        features=feature_map,
        covariates=covariate_map,
        directions=dict(directions or {}),
        model=model,
        study_context=studyContext,
    )

    if assay_info is None:
        return PcaSelectionResult(
            status="failed",
            assay=assay_name,
            notes=[f"No feature characterization found for assay {assay_name!r}"],
            auditLog=[
                {
                    "kind": "missingFeatureAssay",
                    "detail": f"Assay {assay_name!r} missing from features result",
                }
            ],
        )

    species = str(assay_info.get("species") or "unknown")
    if species == "unknown":
        return PcaSelectionResult(
            status="needsInput",
            assay=assay_name,
            species=species,
            needsInput=NeedsInput(
                question="Species is unresolved. Provide a species direction before PCA.",
                options=sorted(_CELL_CYCLE),
            ),
            auditLog=[
                {
                    "kind": "speciesUnresolved",
                    "detail": "Cannot score cell-cycle or apply species catalogs",
                }
            ],
        )

    preflight = _preflight(run, assay_info=assay_info)
    if preflight is not None:
        return preflight

    family_indexes = _family_indexes(assay_info)
    if not any(family_indexes.get(name) for name in _BASE_EXCLUDE_FAMILIES):
        run.notes.append(
            "Exact mitochondrial, ribosomal, and histone indexes were empty; "
            "HVG selection will rely on active features only for those families."
        )

    technical, protected, kinds = _covariate_columns(covariate_map)
    sex_columns = _sex_columns(covariate_map, kinds)
    sample_column = _sample_column(run, technical)
    if run.needs_input is not None:
        return PcaSelectionResult(
            status="needsInput",
            assay=assay_name,
            species=species,
            needsInput=run.needs_input,
            auditLog=run.audit,
            actions=run.actions,
            notes=run.notes,
            decisions=run.decisions,
            acceptedActions=run.accepted,
        )

    input_mask = _resolve_input_selection(run)
    input_fingerprint = fingerprint_array(input_mask)
    run.accepted.append(
        {
            "action": "recordInputSelection",
            "column": "I",
            "persistedColumn": _stage3_input_column(assay_name),
            "fingerprint": input_fingerprint,
            "nCells": int(input_mask.sum()),
        }
    )

    # Always restore the Stage 3 input selection before QC so reruns do not
    # tighten against already-filtered cells.
    _restore_input_selection(run, input_mask)
    _recompute_percentages(run, family_indexes)
    _restore_input_selection(run, input_mask)
    qc_retention = _run_qc(run, sample_column)
    cell_cycle = _score_cell_cycle(run, species)
    protect_sex = _decide_protection(
        run,
        family="sex",
        evidence_summary=(
            f"{len(family_indexes.get('sex', []))} annotated sex-linked genes "
            f"and metadata columns {sex_columns or 'none'} are available"
        ),
    )
    sex_coefficients = _coefficient_names(covariate_map).intersection(sex_columns)
    if sex_coefficients and not protect_sex:
        protect_sex = True
        run.note(
            kind="sexProtectionRequired",
            detail="Sex is a coefficient of interest and cannot be treated as nuisance",
            columns=sorted(sex_coefficients),
        )
    protect_proliferation = _decide_protection(
        run,
        family="cellCycle",
        evidence_summary=(
            f"{len(family_indexes.get('cellCycle', []))} annotated cell-cycle "
            f"genes and score columns "
            f"{[cell_cycle.get(key) for key in ('sScoreColumn', 'g2mScoreColumn', 'phaseColumn') if cell_cycle.get(key)] or 'none'} "
            "are available"
        ),
    )
    run.directions["protectSex"] = protect_sex
    run.directions["protectProliferation"] = protect_proliferation
    diagnostic_roster = _diagnostic_roster(
        technical=technical,
        protected=protected,
        kinds=kinds,
        sex_columns=sex_columns,
        cell_cycle=cell_cycle,
        protect_sex=protect_sex,
        protect_proliferation=protect_proliferation,
    )

    base_blacklist = _merge_indexes(
        *(family_indexes.get(name, []) for name in _BASE_EXCLUDE_FAMILIES)
    )
    dims = int(run.directions.get("pcaDims", _DEFAULT_PCA_DIMS))
    n_cells = int(np.asarray(store.cells.fetch_all("I"), dtype=bool).sum())
    if n_cells < 3:
        return PcaSelectionResult(
            status="failed",
            assay=assay_name,
            species=species,
            notes=[f"Too few cells ({n_cells}) remain after QC for PCA"],
            auditLog=run.audit,
            actions=run.actions,
            acceptedActions=run.accepted,
            qcRetention=qc_retention,
            cellCycle=cell_cycle,
        )
    dims = max(2, min(dims, max(2, n_cells - 1)))

    original_feat_key = _DEFAULT_HVG_KEY
    _mark_hvgs(run, blacklist_indexes=base_blacklist, hvg_key_name=original_feat_key)
    assay = run.store._get_assay(run.assay_name)
    n_hvgs = int(
        np.asarray(assay.feats.fetch_all(f"I__{original_feat_key}"), dtype=bool).sum()
    )
    if n_hvgs < 3:
        return PcaSelectionResult(
            status="failed",
            assay=assay_name,
            species=species,
            notes=[f"Too few HVGs ({n_hvgs}) remain for PCA"],
            auditLog=run.audit,
            actions=run.actions,
            acceptedActions=run.accepted,
            qcRetention=qc_retention,
            cellCycle=cell_cycle,
        )
    dims = max(2, min(dims, n_hvgs - 1, max(2, n_cells - 1)))
    original_norm, original_pca = _run_pca_branch(
        run,
        feat_key=original_feat_key,
        dims=dims,
    )
    original_diagnostics = _diagnose_branch(
        run,
        pca_ref=original_pca,
        feat_key=f"I__{original_feat_key}",
        family_indexes=family_indexes,
        diagnostic_roster=diagnostic_roster,
    )
    original_branch = {
        "id": "original",
        "hvgKey": f"I__{original_feat_key}",
        "blacklistIndexes": base_blacklist,
        "normalized": _artifact_dict(original_norm),
        "pca": _artifact_dict(original_pca),
        "diagnostics": {
            "summary": original_diagnostics["summary"],
            "nCells": original_diagnostics["nCells"],
            "nDims": original_diagnostics["nDims"],
            "nFeatures": original_diagnostics["nFeatures"],
            "nFlaggedAssociations": sum(
                1 for item in original_diagnostics["associations"] if item["flagged"]
            ),
            "loadings": original_diagnostics["loadings"],
            "associations": original_diagnostics["associations"],
        },
    }

    alternate_branch: dict[str, Any] | None = None
    rejected_alternate: dict[str, Any] | None = None
    directed_indexes = [
        int(index) for index in (run.directions.get("nuisanceGeneIndexes") or [])
    ]
    extra_indexes, used_families = _candidate_nuisance_indexes(
        original_diagnostics,
        family_indexes,
        protect_sex=protect_sex,
        protect_proliferation=protect_proliferation,
        nuisance_covariates=[
            str(record["name"])
            for record in diagnostic_roster
            if record["role"] in {"technical", "nuisance"}
        ],
        directed_indexes=directed_indexes,
    )
    if extra_indexes:
        alternate_blacklist = _merge_indexes(base_blacklist, extra_indexes)
        alt_key = "hvgs_nuisance"
        _mark_hvgs(
            run,
            blacklist_indexes=alternate_blacklist,
            hvg_key_name=alt_key,
        )
        alt_mask = np.asarray(
            assay.feats.fetch_all(f"I__{alt_key}"),
            dtype=bool,
        )
        original_mask = np.asarray(
            assay.feats.fetch_all(f"I__{original_feat_key}"),
            dtype=bool,
        )
        alt_n_hvgs = int(alt_mask.sum())
        rejection_reason = None
        if alt_n_hvgs <= dims:
            rejection_reason = (
                f"Alternate branch has {alt_n_hvgs} HVGs and cannot support "
                f"the matched {dims} PCA dimensions"
            )
        elif np.array_equal(alt_mask, original_mask):
            rejection_reason = "Alternate blacklist did not change the selected HVG set"
        if rejection_reason is not None:
            rejected_alternate = {
                "id": "nuisanceFiltered",
                "status": "rejected",
                "hvgKey": f"I__{alt_key}",
                "blacklistIndexes": alternate_blacklist,
                "nuisanceFamilies": used_families,
                "reason": rejection_reason,
                "nFeatures": alt_n_hvgs,
                "requiredDims": dims,
            }
            run.note(
                kind="nuisanceBranchRejected",
                detail=rejection_reason,
                nFeatures=alt_n_hvgs,
                requiredDims=dims,
            )
        else:
            alt_norm, alt_pca = _run_pca_branch(
                run,
                feat_key=alt_key,
                dims=dims,
            )
            alt_diagnostics = _diagnose_branch(
                run,
                pca_ref=alt_pca,
                feat_key=f"I__{alt_key}",
                family_indexes=family_indexes,
                diagnostic_roster=diagnostic_roster,
            )
            alternate_branch = {
                "id": "nuisanceFiltered",
                "status": "complete",
                "hvgKey": f"I__{alt_key}",
                "blacklistIndexes": alternate_blacklist,
                "nuisanceFamilies": used_families,
                "normalized": _artifact_dict(alt_norm),
                "pca": _artifact_dict(alt_pca),
                "diagnostics": {
                    "summary": alt_diagnostics["summary"],
                    "nCells": alt_diagnostics["nCells"],
                    "nDims": alt_diagnostics["nDims"],
                    "nFeatures": alt_diagnostics["nFeatures"],
                    "nFlaggedAssociations": sum(
                        1 for item in alt_diagnostics["associations"] if item["flagged"]
                    ),
                    "loadings": alt_diagnostics["loadings"],
                    "associations": alt_diagnostics["associations"],
                },
            }
            run.actions.append("nuisanceFilteredPca")
            run.accepted.append(
                {
                    "action": "nuisanceFilteredPca",
                    "families": used_families,
                    "blacklistIndexes": alternate_blacklist,
                }
            )
    else:
        run.note(
            kind="nuisanceBranchSkipped",
            detail="No concentrated unprotected nuisance drivers on the original PCA",
            protectSex=protect_sex,
            protectProliferation=protect_proliferation,
        )

    selected_id, rationale = _select_branch(original_branch, alternate_branch)
    selected_branch = (
        alternate_branch
        if selected_id == "nuisanceFiltered" and alternate_branch is not None
        else original_branch
    )
    selected_normalized = cast(dict[str, Any], selected_branch["normalized"])
    selected_pca = cast(dict[str, Any], selected_branch["pca"])
    selected_hvg_key = str(selected_branch["hvgKey"])
    selected_blacklist = [
        int(cast(Any, index)) for index in selected_branch["blacklistIndexes"]
    ]
    selected_diagnostics = cast(dict[str, Any], selected_branch["diagnostics"])
    run.store.run_pca(
        ArtifactRef.from_dict(selected_normalized),
        from_assay=assay_name,
        dims=dims,
        show_elbow_plot=False,
        update_state=True,
    )
    run.actions.append(f"selectBranch:{selected_id}")
    run.accepted.append(
        {
            "action": "selectPcaBranch",
            "branch": selected_id,
            "pca": selected_pca,
            "hvgKey": selected_hvg_key,
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

    branches = [original_branch]
    if alternate_branch is not None:
        branches.append(alternate_branch)
    elif rejected_alternate is not None:
        branches.append(rejected_alternate)

    cell_cycle_columns = [
        str(cell_cycle[key])
        for key in ("sScoreColumn", "g2mScoreColumn", "phaseColumn")
        if isinstance(cell_cycle.get(key), str)
    ]
    selected_associations = cast(
        list[dict[str, Any]],
        selected_diagnostics.get("associations") or [],
    )
    cell_cycle_summary = branch_nuisance_summary(
        selected_associations,
        technicalCovariates=[],
        nuisanceCovariates=(cell_cycle_columns if not protect_proliferation else []),
        protectedCovariates=(cell_cycle_columns if protect_proliferation else []),
        associationFloor=_ASSOCIATION_FLOOR,
    )
    technical_diagnostics = [
        record["name"] for record in diagnostic_roster if record["role"] == "technical"
    ]
    nuisance_diagnostics = [
        record["name"] for record in diagnostic_roster if record["role"] == "nuisance"
    ]
    protected_diagnostics = [
        record["name"] for record in diagnostic_roster if record["role"] == "protected"
    ]

    return PcaSelectionResult(
        status="done",
        assay=assay_name,
        species=species,
        auditLog=run.audit,
        actions=run.actions,
        notes=run.notes,
        decisions=run.decisions,
        acceptedActions=run.accepted,
        qcRetention=qc_retention,
        cellCycle=cell_cycle,
        branches=branches,
        selectedBranch=selected_id,
        selectedPca=selected_pca,
        selectedHvgs={
            "key": selected_hvg_key,
            "blacklistIndexes": selected_blacklist,
        },
        blacklistIndexes=selected_blacklist,
        diagnostics={
            "associationFloor": _ASSOCIATION_FLOOR,
            "familyShareFloor": _FAMILY_SHARE_FLOOR,
            "diagnosticCovariates": diagnostic_roster,
            "technicalCovariates": technical_diagnostics,
            "nuisanceCovariates": nuisance_diagnostics,
            "protectedCovariates": protected_diagnostics,
            "protectSex": protect_sex,
            "protectProliferation": protect_proliferation,
            "cellCycleCovariates": cell_cycle_columns,
            "cellCycleSummary": cell_cycle_summary[
                "protected" if protect_proliferation else "nuisance"
            ],
            "selectedSummary": selected_diagnostics["summary"],
        },
    )
