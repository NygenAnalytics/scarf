"""Characterize feature identity, species, families, and exogenous candidates."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...assay import RNAassay
from ...features.gene_families import GENE_FAMILY_PATTERNS, gene_family_mask
from ...features.gene_reference import (
    GeneReference,
    default_cache_dir,
    ensure_reference,
    load_reference,
)
from ...features.identity import (
    audit_feature_identity,
    backfill_symbols,
    exogenous_candidates,
    observe_families,
    reference_misses,
    resolve_species,
)
from ...features.variability import DEFAULT_HVG_BLACKLIST
from ...quality_control.cell_cycle_genes import (
    g2m_phase_genes,
    g2m_phase_genes_mouse,
    s_phase_genes,
    s_phase_genes_mouse,
)
from ...utils.arrays import regex_match_mask
from .._deps import AGENT_INSTALL_HINT
from ..types import AgentDataModel, StageStatus

try:
    from pydantic import Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "FeatureCharacterization",
    "characterize_features",
]


_CELL_CYCLE = {
    "homo_sapiens": {"s": s_phase_genes, "g2m": g2m_phase_genes},
    "mus_musculus": {"s": s_phase_genes_mouse, "g2m": g2m_phase_genes_mouse},
}
_FEATURE_INVENTORY_EXAMPLE_LIMIT = 8
_AUTO_DOWNLOAD_SPECIES = frozenset({"homo_sapiens", "mus_musculus"})
_MAX_EXOGENOUS = 25


class FeatureCharacterization(AgentDataModel):
    status: StageStatus
    auditLog: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    assays: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureCharacterization":
        return cls(status="failed")


def _bounded_feature_examples(matches: Sequence[str]) -> list[str]:
    return sorted(set(matches), key=lambda value: (value.casefold(), value))[
        :_FEATURE_INVENTORY_EXAMPLE_LIMIT
    ]


def _scarf_default_feature_inventory(
    assay_name: str,
    names: Sequence[str],
) -> dict[str, Any]:
    evidence_prefix = f"assay:{assay_name}:scarfDefaultHvg"
    values = np.asarray(names, dtype=object)
    combined_matches = values[regex_match_mask(values, DEFAULT_HVG_BLACKLIST)].tolist()
    families: list[dict[str, Any]] = []
    family_evidence_ids: list[str] = []
    for family, pattern in GENE_FAMILY_PATTERNS.items():
        matches = values[gene_family_mask(values, family)].tolist()
        evidence_id = f"{evidence_prefix}:family:{family}"
        families.append(
            {
                "family": family,
                "pattern": pattern,
                "caseInsensitive": True,
                "count": len(matches),
                "examples": _bounded_feature_examples(matches),
                "evidenceId": evidence_id,
            }
        )
        family_evidence_ids.append(evidence_id)
    evidence_id = f"{evidence_prefix}:combined"
    return {
        "source": "scarfDefaultHvgBlacklist",
        "policyEffect": "evidenceOnly",
        "featureColumn": "names",
        "totalFeatures": len(names),
        "blacklist": DEFAULT_HVG_BLACKLIST,
        "matchCount": len(combined_matches),
        "examples": _bounded_feature_examples(combined_matches),
        "families": families,
        "evidenceId": evidence_id,
        "evidenceIds": [evidence_id, *family_evidence_ids],
    }


def _audit(
    audit_log: list[dict[str, Any]],
    *,
    kind: str,
    detail: str,
    **fields: Any,
) -> None:
    audit_log.append({"kind": kind, "detail": detail, **fields})


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


def _characterize_assay(
    store: Any,
    assay_name: str,
    *,
    cache_dir: Path,
    allow_download: bool,
    audit_log: list[dict[str, Any]],
    actions: list[str],
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

    default_inventory = _scarf_default_feature_inventory(assay_name, names)
    record["defaultFeatureInventory"] = default_inventory
    _audit(
        audit_log,
        kind="scarfDefaultFeatureInventory",
        detail=(
            f"Scarf's default HVG blacklist matched "
            f"{default_inventory['matchCount']} of {len(names)} RNA features"
        ),
        assay=assay_name,
        evidenceIds=default_inventory["evidenceIds"],
    )

    resolution = resolve_species(
        ids,
        names,
        cacheDir=cache_dir,
        allowDownload=allow_download,
    )
    species = resolution["species"]
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

    # Only human and mouse references are downloaded automatically.
    reference = _load_or_fetch_reference(
        species,
        cache_dir=cache_dir,
        allow_download=allow_download and species in _AUTO_DOWNLOAD_SPECIES,
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
    _audit(
        audit_log,
        kind="sexChromosomeTracked",
        detail=(
            "Sex-chromosome genes are tracked with defaultExclude=false; "
            "Phase 3 may exclude them when sex is not a coefficient of interest"
        ),
        assay=assay_name,
    )

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
        maxCandidates=_MAX_EXOGENOUS,
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
    # Structural candidates stay unresolved; policies cite them with evidence.
    record["exogenous"] = [{**item, "class": "unresolved"} for item in candidates]
    actions.append(f"families:{assay_name}")
    return record


def characterize_features(
    store: Any,
    *,
    assays: Sequence[str] | None = None,
    cacheDir: Path | str | None = None,
    allowDownload: bool = False,
) -> FeatureCharacterization:
    """Label feature identity, species, families, and exogenous candidates."""
    audit_log: list[dict[str, Any]] = []
    actions: list[str] = []
    notes: list[str] = []
    cache_dir = Path(cacheDir) if cacheDir is not None else default_cache_dir()

    available = list(store.assay_names)
    selected = list(assays) if assays is not None else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        return FeatureCharacterization(
            status="failed",
            notes=[f"unknown assays: {unknown}"],
        )
    assay_records = [
        _characterize_assay(
            store,
            assay_name,
            cache_dir=cache_dir,
            allow_download=allowDownload,
            audit_log=audit_log,
            actions=actions,
        )
        for assay_name in selected
    ]
    notes.append(f"Characterized {len(assay_records)} assay(s)")
    return FeatureCharacterization(
        status="done",
        auditLog=audit_log,
        actions=actions,
        notes=notes,
        assays=assay_records,
    )
