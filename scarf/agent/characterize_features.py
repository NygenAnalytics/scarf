"""Characterize feature identity, species, families, and exogenous candidates."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..assay import RNAassay
from ..features.gene_reference import (
    GeneReference,
    default_cache_dir,
    ensure_reference,
    load_reference,
    species_registry,
)
from ..features.identity import (
    audit_feature_identity,
    backfill_symbols,
    exogenous_candidates,
    observe_families,
    reference_misses,
    resolve_species,
)
from ..quality_control.cell_cycle_genes import (
    g2m_phase_genes,
    g2m_phase_genes_mouse,
    s_phase_genes,
    s_phase_genes_mouse,
)
from ._deps import AGENT_INSTALL_HINT
from .decide import DecisionValidationError, decide
from .types import Decision, EvidenceItem, StageStatus

try:
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "FeatureCharacterization",
    "characterize_features",
]

_DEFAULT_MAX_EXOGENOUS = 25
_CONTEXT_LIMIT = 1200
_AUTO_DOWNLOAD_SPECIES = frozenset({"homo_sapiens", "mus_musculus"})
_CELL_CYCLE = {
    "homo_sapiens": {"s": s_phase_genes, "g2m": g2m_phase_genes},
    "mus_musculus": {"s": s_phase_genes_mouse, "g2m": g2m_phase_genes_mouse},
}
_SEX_COEFFICIENT_TOKENS = frozenset({"sex", "gender", "Sex", "Gender"})


class FeatureCharacterization(BaseModel):
    status: StageStatus
    auditLog: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    assays: list[dict[str, Any]] = Field(default_factory=list)


def _bounded_context(study_context: str | None) -> str:
    text = (study_context or "").strip()
    return text if len(text) <= _CONTEXT_LIMIT else text[: _CONTEXT_LIMIT - 3] + "..."


def _audit(
    audit_log: list[dict[str, Any]],
    *,
    kind: str,
    detail: str,
    **fields: Any,
) -> None:
    audit_log.append({"kind": kind, "detail": detail, **fields})


def _ask(
    *,
    model: Any | None,
    question: str,
    evidence: Sequence[EvidenceItem],
    decisions: list[dict[str, Any]],
    audit_log: list[dict[str, Any]],
    task: str,
    assay: str | None = None,
) -> Decision | None:
    if model is None or len(evidence) < 2:
        return None
    try:
        decision = decide(model=model, question=question, evidence=evidence)
    except DecisionValidationError as exc:
        _audit(
            audit_log,
            kind="decisionInvalid",
            detail=str(exc),
            task=task,
            assay=assay,
        )
        return None
    record: dict[str, Any] = {
        "task": task,
        "selectedId": decision.selectedId,
        "rationale": decision.rationale,
        "evidenceIds": list(decision.evidenceIds),
    }
    if assay is not None:
        record["assay"] = assay
    decisions.append(record)
    return decision


def _sex_coefficient_note(
    covariates: Any | None,
) -> str | None:
    if covariates is None:
        return None
    coefficients = getattr(covariates, "coefficients", None) or []
    for item in coefficients:
        name = item.get("name") if isinstance(item, Mapping) else None
        if name in _SEX_COEFFICIENT_TOKENS:
            return (
                f"Prior covariate characterization marked {name!r} as a coefficient "
                "of interest; tracked sex-chromosome genes must not be excluded later"
            )
    return None


def _load_or_fetch_reference(
    species: str,
    *,
    cache_dir: Path,
    allow_download: bool,
    audit_log: list[dict[str, Any]],
    assay: str,
) -> GeneReference | None:
    if species == "unknown":
        return None
    cached = load_reference(species, cacheDir=cache_dir)
    if cached is not None:
        return cached
    if not allow_download:
        _audit(
            audit_log,
            kind="referenceUnavailable",
            detail=f"No cached reference for {species}; download disabled",
            assay=assay,
            species=species,
        )
        return None
    try:
        reference = ensure_reference(species, cacheDir=cache_dir)
    except Exception as exc:
        _audit(
            audit_log,
            kind="referenceDownloadFailed",
            detail=f"Failed to download reference for {species}: {exc}",
            assay=assay,
            species=species,
        )
        return None
    _audit(
        audit_log,
        kind="referenceDownloaded",
        detail=f"Cached gene reference for {species} release {reference.release}",
        assay=assay,
        species=species,
        release=reference.release,
    )
    return reference


def _assist_species(
    *,
    model: Any | None,
    unresolved: Mapping[str, Any],
    context: str,
    decisions: list[dict[str, Any]],
    audit_log: list[dict[str, Any]],
    assay: str,
) -> str | None:
    # Only ask among species that already have overlap evidence. Expanding to the
    # full registry would let a guess trigger a non-human/mouse download.
    candidates = [
        key for key in (unresolved.get("candidates") or []) if key in species_registry()
    ]
    if len(candidates) < 2:
        return None
    evidence = [
        EvidenceItem(
            id=f"species:{key}",
            label=species_registry()[key].label,
            summary=(
                f"overlap hits="
                f"{(unresolved.get('overlap') or {}).get('scores', {}).get(key, {}).get('hits', 0)}; "
                f"prefix count="
                f"{(unresolved.get('prefixCounts') or {}).get(key, 0)}"
            ),
        )
        for key in candidates
    ]
    evidence.append(
        EvidenceItem(
            id="species:unknown",
            label="unknown",
            summary="Cannot settle species from available evidence",
        )
    )
    decision = _ask(
        model=model,
        question=(
            f"Choose the species for assay {assay}. "
            f"Identity notes: {unresolved.get('reason', '')}. "
            f"Study context: {context or 'none provided'}."
        ),
        evidence=evidence,
        decisions=decisions,
        audit_log=audit_log,
        task="species",
        assay=assay,
    )
    if decision is None:
        return None
    selected = decision.selectedId.removeprefix("species:")
    return selected if selected in species_registry() or selected == "unknown" else None


def _classify_exogenous(
    *,
    model: Any | None,
    candidates: Sequence[Mapping[str, Any]],
    context: str,
    decisions: list[dict[str, Any]],
    audit_log: list[dict[str, Any]],
    assay: str,
    species: str,
) -> list[dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    for item in candidates:
        label = item.get("name") or item.get("id") or "feature"
        evidence = [
            EvidenceItem(
                id="exogenous:potentialExogenous",
                label="potentialExogenous",
                summary="Spike-in, transgene, guide, antibody tag, or other non-endogenous feature",
            ),
            EvidenceItem(
                id="exogenous:unresolved",
                label="unresolved",
                summary="Not enough evidence to treat as exogenous",
            ),
        ]
        decision = _ask(
            model=model,
            question=(
                f"Classify feature {label!r} (id={item.get('id')!r}) for assay {assay} "
                f"under species {species}. Study context: {context or 'none provided'}."
            ),
            evidence=evidence,
            decisions=decisions,
            audit_log=audit_log,
            task="exogenous",
            assay=assay,
        )
        record = dict(item)
        if decision is None:
            record["class"] = "unresolved"
        else:
            record["class"] = decision.selectedId.removeprefix("exogenous:")
        classified.append(record)
    return classified


def _characterize_assay(
    store: Any,
    assay_name: str,
    *,
    model: Any | None,
    context: str,
    directions: Mapping[str, Any],
    cache_dir: Path,
    allow_download: bool,
    audit_log: list[dict[str, Any]],
    actions: list[str],
    decisions: list[dict[str, Any]],
    sex_note: str | None,
) -> dict[str, Any]:
    assay = store.get_assay(assay_name)
    ids = [str(value) for value in assay.feats.fetch_all("ids")]
    names = [str(value) for value in assay.feats.fetch_all("names")]
    identity = audit_feature_identity(ids, names)
    record: dict[str, Any] = {
        "assay": assay_name,
        "assayKind": type(assay).__name__,
        "identity": identity,
        "species": "unknown",
        "speciesMethod": None,
        "families": [],
        "exogenous": [],
        "symbolBackfill": None,
    }

    if not isinstance(assay, RNAassay):
        record["skipped"] = "familyPlanningNotApplicable"
        _audit(
            audit_log,
            kind="nonRnaAssay",
            detail=f"Assay {assay_name} is not RNA; stopped after identity audit",
            assay=assay_name,
        )
        return record

    species_by_assay = dict(directions.get("speciesByAssay") or {})
    directed_species = species_by_assay.get(assay_name)
    resolution = resolve_species(
        ids,
        names,
        directed=directed_species,
        cacheDir=cache_dir,
        allowDownload=allow_download,
    )
    species = resolution["species"]
    if species == "unknown" and resolution.get("method") == "inconclusive":
        assisted = _assist_species(
            model=model,
            unresolved=resolution,
            context=context,
            decisions=decisions,
            audit_log=audit_log,
            assay=assay_name,
        )
        if assisted is not None:
            species = assisted
            resolution = {
                **resolution,
                "species": species,
                "method": "llmAssist",
                "reason": "model choice among inconclusive candidates",
            }
            actions.append(f"species:{assay_name}->{species}")

    record["species"] = species
    record["speciesMethod"] = resolution.get("method")
    record["speciesResolution"] = {
        key: value
        for key, value in resolution.items()
        if key not in {"overlap"} or value is not None
    }
    _audit(
        audit_log,
        kind="speciesResolved",
        detail=resolution.get("reason", f"species={species}"),
        assay=assay_name,
        species=species,
        method=resolution.get("method"),
    )

    if species == "unknown":
        record["families"] = observe_families(
            species="unknown",
            ids=ids,
            symbols=names,
            reference=None,
        )
        _audit(
            audit_log,
            kind="speciesUnknown",
            detail=f"Skipped species-dependent steps for assay {assay_name}",
            assay=assay_name,
        )
        return record

    # Only human/mouse auto-download; any other species needs an explicit direction.
    may_download = allow_download and (
        species in _AUTO_DOWNLOAD_SPECIES or directed_species == species
    )
    reference = _load_or_fetch_reference(
        species,
        cache_dir=cache_dir,
        allow_download=may_download,
        audit_log=audit_log,
        assay=assay_name,
    )

    symbols = list(names)
    if reference is not None:
        backfill = backfill_symbols(ids, names, reference)
        if backfill["nRecovered"]:
            symbols = backfill["symbols"]
            record["symbolBackfill"] = {
                "nRecovered": backfill["nRecovered"],
                "joinRate": backfill["joinRate"],
            }
            actions.append(
                f"symbolBackfill:{assay_name}:{backfill['nRecovered']}/{backfill['nFeatures']}"
            )
    elif identity.get("idsEqualNames") or identity.get("nEmptyNames", 0) > 0:
        _audit(
            audit_log,
            kind="familiesNotAssessable",
            detail=(
                f"Names are empty or ID-shaped on {assay_name} and no reference "
                "is available to recover symbols"
            ),
            assay=assay_name,
        )

    record["families"] = observe_families(
        species=species,
        ids=ids,
        symbols=symbols,
        reference=reference,
        cellCycleGenes=_CELL_CYCLE,
    )
    if sex_note is not None:
        record["notes"] = [sex_note]
        _audit(
            audit_log,
            kind="sexCoefficientNote",
            detail=sex_note,
            assay=assay_name,
        )
    elif species != "unknown":
        _audit(
            audit_log,
            kind="sexChromosomeTracked",
            detail=(
                "Sex-chromosome genes are tracked with defaultExclude=false; "
                "Phase 3 may exclude them when sex is not a coefficient of interest"
            ),
            assay=assay_name,
        )

    raw_max = directions.get("maxExogenousCandidates", _DEFAULT_MAX_EXOGENOUS)
    try:
        max_exogenous = int(raw_max)
    except (TypeError, ValueError):
        _audit(
            audit_log,
            kind="invalidDirection",
            detail=f"maxExogenousCandidates={raw_max!r}; using {_DEFAULT_MAX_EXOGENOUS}",
            assay=assay_name,
        )
        max_exogenous = _DEFAULT_MAX_EXOGENOUS
    if reference is not None:
        misses = reference_misses(ids, symbols, reference)
        if misses["count"]:
            record["referenceMisses"] = misses
            _audit(
                audit_log,
                kind="referenceMiss",
                detail=(
                    f"{misses['count']} Ensembl-shaped id(s) absent from the "
                    f"{species} reference (release drift, not exogenous)"
                ),
                assay=assay_name,
                count=misses["count"],
                examples=misses["examples"],
            )
    candidates = exogenous_candidates(
        ids,
        symbols,
        reference=reference,
        maxCandidates=max_exogenous,
    )
    if reference is None and not candidates:
        _audit(
            audit_log,
            kind="exogenousUnresolved",
            detail=(
                f"No reference and no structural exogenous candidates for {assay_name}"
            ),
            assay=assay_name,
        )
    record["exogenous"] = _classify_exogenous(
        model=model,
        candidates=candidates,
        context=context,
        decisions=decisions,
        audit_log=audit_log,
        assay=assay_name,
        species=species,
    )
    actions.append(f"families:{assay_name}")
    return record


def characterize_features(
    store: Any,
    *,
    studyContext: str | None = None,
    model: Any | None = None,
    assays: Sequence[str] | None = None,
    directions: Mapping[str, Any] | None = None,
    covariates: Any | None = None,
    cacheDir: Path | str | None = None,
    allowDownload: bool = False,
) -> FeatureCharacterization:
    """Label feature identity, species, families, and exogenous candidates."""
    direction_map = dict(directions or {})
    audit_log: list[dict[str, Any]] = []
    actions: list[str] = []
    decisions: list[dict[str, Any]] = []
    notes: list[str] = []
    context = _bounded_context(studyContext)
    cache_dir = Path(cacheDir) if cacheDir is not None else default_cache_dir()

    available = list(store.assay_names)
    selected = list(assays) if assays is not None else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        return FeatureCharacterization(
            status="failed",
            notes=[f"unknown assays: {unknown}"],
        )
    species_by_assay = direction_map.get("speciesByAssay")
    if species_by_assay is not None:
        if not isinstance(species_by_assay, Mapping):
            return FeatureCharacterization(
                status="failed",
                notes=["speciesByAssay must be a mapping"],
            )
        bad = sorted(set(species_by_assay) - set(available))
        if bad:
            return FeatureCharacterization(
                status="failed",
                notes=[f"speciesByAssay cites unknown assays: {bad}"],
            )

    sex_note = _sex_coefficient_note(covariates)
    assay_records = [
        _characterize_assay(
            store,
            assay_name,
            model=model,
            context=context,
            directions=direction_map,
            cache_dir=cache_dir,
            allow_download=allowDownload,
            audit_log=audit_log,
            actions=actions,
            decisions=decisions,
            sex_note=sex_note,
        )
        for assay_name in selected
    ]
    notes.append(f"Characterized {len(assay_records)} assay(s)")
    return FeatureCharacterization(
        status="done",
        auditLog=audit_log,
        actions=actions,
        notes=notes,
        decisions=decisions,
        assays=assay_records,
    )
