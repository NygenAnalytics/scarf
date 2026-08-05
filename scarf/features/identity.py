"""Feature identity audit, species resolution, and family observations."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .gene_reference import (
    GeneReference,
    cached_species,
    ensure_reference,
    load_reference,
    prefix_species,
    species_registry,
)

__all__ = [
    "audit_feature_identity",
    "backfill_symbols",
    "exogenous_candidates",
    "observe_families",
    "reference_misses",
    "resolve_species",
    "score_symbol_overlap",
    "strip_ensembl_version",
]

_RIBO_PREFIXES = ("RPS", "RPL", "MRPS", "MRPL")
_SAMPLE = 8
_OVERLAP_MIN_HITS = 5
_OVERLAP_MIN_SHARE = 0.05
_OVERLAP_MARGIN = 1.5
_SEX_CHROMOSOMES = frozenset({"X", "Y"})

_HISTONE_PREFIXES: dict[str, tuple[str, ...]] = {
    "homo_sapiens": ("HIST", "H3F3", "H2AF", "H2BF", "H1F"),
    "mus_musculus": ("Hist", "H3f3", "H2af", "H2bf", "H1f"),
}


def strip_ensembl_version(gene_id: str) -> str:
    if "." in gene_id and gene_id.rsplit(".", 1)[-1].isdigit():
        return gene_id.rsplit(".", 1)[0]
    return gene_id


def _as_str_list(values: Sequence[Any]) -> list[str]:
    return ["" if value is None else str(value) for value in values]


def _bounded(values: Sequence[str], limit: int = _SAMPLE) -> list[str]:
    return list(values[:limit])


def audit_feature_identity(
    ids: Sequence[Any],
    names: Sequence[Any],
) -> dict[str, Any]:
    """Local audit of feature ids and names. No network."""
    id_list = _as_str_list(ids)
    name_list = _as_str_list(names)
    if len(id_list) != len(name_list):
        raise ValueError("ids and names must have the same length")
    n = len(id_list)
    empty_ids = sum(1 for value in id_list if not value)
    empty_names = sum(1 for value in name_list if not value)
    id_counts = Counter(value for value in id_list if value)
    name_counts = Counter(value for value in name_list if value)
    duplicate_ids = sorted(value for value, count in id_counts.items() if count > 1)
    duplicate_names = sorted(value for value, count in name_counts.items() if count > 1)
    ids_equal_names = bool(n) and all(
        left == right for left, right in zip(id_list, name_list, strict=True)
    )
    versioned = [
        value for value in id_list if value and value != strip_ensembl_version(value)
    ]
    prefixes = prefix_species(id_list)
    return {
        "nFeatures": n,
        "nEmptyIds": empty_ids,
        "nEmptyNames": empty_names,
        "nDuplicateIds": len(duplicate_ids),
        "nDuplicateNames": len(duplicate_names),
        "duplicateIdExamples": _bounded(duplicate_ids),
        "duplicateNameExamples": _bounded(duplicate_names),
        "idsEqualNames": ids_equal_names,
        "nVersionSuffixes": len(versioned),
        "versionSuffixExamples": _bounded(versioned),
        "prefixCounts": prefixes,
    }


def score_symbol_overlap(
    names: Sequence[Any],
    references: Mapping[str, GeneReference],
) -> dict[str, Any]:
    """Case-sensitive symbol overlap against available species references."""
    name_list = [name for name in _as_str_list(names) if name]
    if not name_list or not references:
        return {"scores": {}, "best": None, "second": None, "clear": False}
    n = len(name_list)
    scores: dict[str, dict[str, Any]] = {}
    for species, reference in references.items():
        symbols = reference.symbols()
        hits = sum(1 for name in name_list if name in symbols)
        scores[species] = {
            "hits": hits,
            "share": hits / n,
            "nQuery": n,
            "nReferenceSymbols": len(symbols),
        }
    ranked = sorted(
        scores.items(),
        key=lambda item: (item[1]["hits"], item[1]["share"]),
        reverse=True,
    )
    best_key, best = ranked[0]
    second_key, second = ranked[1] if len(ranked) > 1 else (None, None)
    clear = bool(
        best["hits"] >= _OVERLAP_MIN_HITS
        and best["share"] >= _OVERLAP_MIN_SHARE
        and (
            second is None
            or best["hits"] >= max(second["hits"] * _OVERLAP_MARGIN, second["hits"] + 3)
        )
    )
    return {
        "scores": scores,
        "best": best_key,
        "second": second_key,
        "clear": clear,
    }


def resolve_species(
    ids: Sequence[Any],
    names: Sequence[Any],
    *,
    directed: str | None = None,
    cacheDir: Any = None,
    allowDownload: bool = False,
    references: Mapping[str, GeneReference] | None = None,
) -> dict[str, Any]:
    """Resolve species without applying family rules.

    Order: directions override, Ensembl ID prefix, symbol overlap against
    available references (human/mouse may download when allowDownload), else
    inconclusive for an optional LLM assist by the caller.
    """
    if directed is not None:
        if directed not in species_registry() and directed != "unknown":
            return {
                "species": "unknown",
                "method": "directions",
                "reason": f"unsupported species {directed!r}",
            }
        return {
            "species": directed,
            "method": "directions",
            "reason": "caller override",
        }

    prefixes = prefix_species(_as_str_list(ids))
    if prefixes:
        ranked = sorted(prefixes.items(), key=lambda item: item[1], reverse=True)
        best_key, best_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        if best_count >= max(3, second_count * 2):
            return {
                "species": best_key,
                "method": "ensemblPrefix",
                "prefixCounts": dict(ranked),
                "reason": f"{best_count} ids carry {species_registry()[best_key].idPrefix}",
            }

    available: dict[str, GeneReference] = dict(references or {})
    for species in cached_species(cacheDir):
        if species not in available:
            loaded = load_reference(species, cacheDir=cacheDir)
            if loaded is not None:
                available[species] = loaded
    download_errors: list[str] = []
    if allowDownload:
        for species in ("homo_sapiens", "mus_musculus"):
            if species in available:
                continue
            try:
                available[species] = ensure_reference(species, cacheDir=cacheDir)
            except Exception as exc:  # keep cached refs; fall through to overlap
                download_errors.append(f"{species}: {exc}")

    overlap = score_symbol_overlap(names, available)
    if overlap["clear"]:
        result = {
            "species": overlap["best"],
            "method": "symbolOverlap",
            "overlap": overlap,
            "reason": (
                f"clear symbol overlap for {overlap['best']} "
                f"({overlap['scores'][overlap['best']]['hits']} hits)"
            ),
        }
        if download_errors:
            result["downloadErrors"] = download_errors
        return result
    candidates = [
        species
        for species, score in sorted(
            overlap["scores"].items(),
            key=lambda item: item[1]["hits"],
            reverse=True,
        )
        if score["hits"] > 0
    ]
    result = {
        "species": "unknown",
        "method": "inconclusive",
        "overlap": overlap,
        "candidates": candidates,
        "prefixCounts": prefixes,
        "reason": "prefix and symbol overlap did not settle a species",
    }
    if download_errors:
        result["downloadErrors"] = download_errors
    return result


def backfill_symbols(
    ids: Sequence[Any],
    names: Sequence[Any],
    reference: GeneReference,
) -> dict[str, Any]:
    """Recover symbols from the reference when a name is empty or ID-shaped."""
    id_list = _as_str_list(ids)
    name_list = _as_str_list(names)
    filled: list[str] = []
    recovered = 0
    for gene_id, name in zip(id_list, name_list, strict=True):
        if name and name != gene_id:
            filled.append(name)
            continue
        symbol = reference.symbol_for(gene_id)
        if symbol:
            filled.append(symbol)
            recovered += 1
        else:
            filled.append(name)
    return {
        "symbols": filled,
        "nRecovered": recovered,
        "nFeatures": len(filled),
        "joinRate": recovered / len(filled) if filled else 0.0,
        "idsEqualNames": bool(id_list)
        and all(left == right for left, right in zip(id_list, name_list, strict=True)),
    }


def _chromosomes_for(
    id_list: Sequence[str],
    symbol_list: Sequence[str],
    reference: GeneReference,
    chromosomes: frozenset[str],
) -> list[str]:
    hits: list[str] = []
    wanted = {chrom.upper() for chrom in chromosomes}
    for gene_id, symbol in zip(id_list, symbol_list, strict=True):
        chrom = reference.chromosome_for(gene_id)
        if chrom is not None and chrom.upper() in wanted:
            hits.append(symbol or gene_id)
    return hits


def _match_prefix(symbols: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    hits: list[str] = []
    upper_prefixes = tuple(prefix.upper() for prefix in prefixes)
    for symbol in symbols:
        if not symbol:
            continue
        upper = symbol.upper()
        if any(upper.startswith(prefix) for prefix in upper_prefixes):
            hits.append(symbol)
    return hits


def _match_set(symbols: Sequence[str], catalog: frozenset[str]) -> list[str]:
    return [symbol for symbol in symbols if symbol in catalog]


def _family_record(
    *,
    family: str,
    method: str,
    matches: Sequence[str],
    species: str,
    skipped: str | None = None,
    defaultExclude: bool | None = None,
    catalogSuspect: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "family": family,
        "species": species,
        "method": method,
        "count": len(matches),
        "examples": _bounded(sorted(set(matches))),
    }
    if skipped is not None:
        record["skipped"] = skipped
    if defaultExclude is not None:
        record["defaultExclude"] = defaultExclude
    if catalogSuspect is not None:
        record["catalogSuspect"] = catalogSuspect
    record.update(extra)
    return record


def _suspect_if_empty(
    *,
    matches: Sequence[str],
    skipped: str | None,
    reference: GeneReference | None,
    symbols: Sequence[str],
) -> str | None:
    """Flag empty family hits when the lookup surface was non-empty."""
    if skipped is not None or matches:
        return None
    if reference is not None or any(symbols):
        return "zeroMatches"
    return None


def observe_families(
    *,
    species: str,
    ids: Sequence[Any],
    symbols: Sequence[Any],
    reference: GeneReference | None,
    cellCycleGenes: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> list[dict[str, Any]]:
    """Species-aware family observations. Does not decide exclusions.

    ``defaultExclude`` is a Phase 3 hint only. Mitochondrial, ribosomal, and
    histone families are typical QC exclusion candidates. Sex tracks every gene
    on chromosomes X and Y for later use when sex is not a coefficient of
    interest; it is never a default blacklist. Cell-cycle is for scoring, not
    bulk exclusion.
    """
    id_list = _as_str_list(ids)
    symbol_list = _as_str_list(symbols)
    if species == "unknown":
        return [
            _family_record(
                family=name,
                method="skipped",
                matches=[],
                species=species,
                skipped="speciesUnknown",
                defaultExclude=False if name in {"sex", "cellCycle"} else True,
            )
            for name in ("mitochondrial", "ribosomal", "cellCycle", "sex", "histone")
        ]

    families: list[dict[str, Any]] = []

    if reference is None:
        for family, method, default_exclude in (
            ("mitochondrial", "chromosome", True),
            ("sex", "chromosome", False),
        ):
            families.append(
                _family_record(
                    family=family,
                    method=method,
                    matches=[],
                    species=species,
                    skipped="referenceUnavailable",
                    defaultExclude=default_exclude,
                )
            )
    else:
        mito = _chromosomes_for(id_list, symbol_list, reference, frozenset({"MT"}))
        families.append(
            _family_record(
                family="mitochondrial",
                method="chromosome",
                matches=mito,
                species=species,
                defaultExclude=True,
                catalogSuspect=_suspect_if_empty(
                    matches=mito,
                    skipped=None,
                    reference=reference,
                    symbols=symbol_list,
                ),
            )
        )
        sex = _chromosomes_for(id_list, symbol_list, reference, _SEX_CHROMOSOMES)
        families.append(
            _family_record(
                family="sex",
                method="chromosome",
                matches=sex,
                species=species,
                defaultExclude=False,
                catalogSuspect=_suspect_if_empty(
                    matches=sex,
                    skipped=None,
                    reference=reference,
                    symbols=symbol_list,
                ),
            )
        )

    ribo = _match_prefix(symbol_list, _RIBO_PREFIXES)
    families.append(
        _family_record(
            family="ribosomal",
            method="symbolPrefix",
            matches=ribo,
            species=species,
            defaultExclude=True,
            catalogSuspect=_suspect_if_empty(
                matches=ribo,
                skipped=None,
                reference=reference,
                symbols=symbol_list,
            ),
        )
    )

    if cellCycleGenes is not None and species in cellCycleGenes:
        catalog = cellCycleGenes[species]
        catalog_symbols = frozenset([*catalog.get("s", []), *catalog.get("g2m", [])])
        matches = sorted(_match_set(symbol_list, catalog_symbols))
        extra: dict[str, Any] = {"catalogSize": len(catalog_symbols)}
        if reference is not None and catalog_symbols:
            present = sum(
                1 for symbol in catalog_symbols if reference.has_symbol(symbol)
            )
            extra["catalogJoinRate"] = present / len(catalog_symbols)
            extra["catalogJoined"] = present
        families.append(
            _family_record(
                family="cellCycle",
                method="staticList",
                matches=matches,
                species=species,
                defaultExclude=False,
                catalogSuspect=_suspect_if_empty(
                    matches=matches,
                    skipped=None,
                    reference=reference,
                    symbols=symbol_list,
                ),
                **extra,
            )
        )
    else:
        families.append(
            _family_record(
                family="cellCycle",
                method="staticList",
                matches=[],
                species=species,
                skipped="catalogUnavailable",
                defaultExclude=False,
            )
        )

    if species in _HISTONE_PREFIXES:
        histone = _match_prefix(symbol_list, _HISTONE_PREFIXES[species])
        families.append(
            _family_record(
                family="histone",
                method="symbolPrefix",
                matches=histone,
                species=species,
                defaultExclude=True,
                catalogSuspect=_suspect_if_empty(
                    matches=histone,
                    skipped=None,
                    reference=reference,
                    symbols=symbol_list,
                ),
            )
        )
    else:
        families.append(
            _family_record(
                family="histone",
                method="symbolPrefix",
                matches=[],
                species=species,
                skipped="catalogUnavailable",
                defaultExclude=True,
            )
        )
    return families


def reference_misses(
    ids: Sequence[Any],
    names: Sequence[Any],
    reference: GeneReference,
    *,
    maxExamples: int = _SAMPLE,
) -> dict[str, Any]:
    """Ensembl-shaped ids absent from the reference (release drift, not exogenous)."""
    id_list = _as_str_list(ids)
    name_list = _as_str_list(names)
    prefixes = tuple(spec.idPrefix for spec in species_registry().values())
    misses: list[str] = []
    for gene_id, name in zip(id_list, name_list, strict=True):
        stripped = strip_ensembl_version(gene_id)
        if not any(stripped.startswith(prefix) for prefix in prefixes):
            continue
        if reference.has_gene_id(gene_id) or (name and reference.has_symbol(name)):
            continue
        misses.append(name or gene_id)
    return {
        "count": len(misses),
        "examples": _bounded(sorted(set(misses)), maxExamples),
    }


def exogenous_candidates(
    ids: Sequence[Any],
    names: Sequence[Any],
    *,
    reference: GeneReference | None,
    maxCandidates: int = 25,
) -> list[dict[str, Any]]:
    """Rank features that look non-endogenous under the available reference.

    Ensembl-shaped ids missing from a loaded reference are release mismatches,
    not exogenous evidence; use ``reference_misses`` for those. Without a
    reference, Ensembl-shaped ids are treated as endogenous by construction.
    """
    id_list = _as_str_list(ids)
    name_list = _as_str_list(names)
    prefixes = tuple(spec.idPrefix for spec in species_registry().values())
    candidates: list[dict[str, Any]] = []
    for gene_id, name in zip(id_list, name_list, strict=True):
        stripped = strip_ensembl_version(gene_id)
        in_reference = reference is not None and (
            reference.has_gene_id(gene_id) or (name and reference.has_symbol(name))
        )
        if in_reference:
            continue
        ensembl_shaped = any(stripped.startswith(prefix) for prefix in prefixes)
        # Ensembl-shaped ids are never exogenous cues; see reference_misses.
        if ensembl_shaped:
            continue
        label = name or gene_id
        if not label:
            continue
        score = 0
        upper = label.upper()
        if any(
            token in upper
            for token in ("ERCC", "GFP", "MCHERRY", "TDTOMATO", "CAS9", "GUIDE", "HTO")
        ):
            score += 3
        if "-" in label or "_" in label:
            score += 1
        if label == gene_id or not name:
            score += 1
        candidates.append(
            {
                "id": gene_id,
                "name": name,
                "score": score,
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["name"], item["id"]))
    return candidates[: max(0, maxCandidates)]
