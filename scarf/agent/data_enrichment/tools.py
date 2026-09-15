"""Read-only feature inspection and lookup tools."""

from typing import Any, Literal

from ...features.variability import DEFAULT_HVG_BLACKLIST
from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..tools import bounded_list
from .characterization import characterize_features
from .contracts import (
    AdtControlEvidence,
    AssayFeatureInspection,
    AssayFeatureInspectionBatch,
    AssayModalityEvidence,
    AtacCoordinateEvidence,
    DataEnrichmentDependencies,
    DataEnrichmentToolCall,
    ExogenousFeatureEvidence,
    FeatureFamilyEvidence,
    FeatureLookupBatch,
    FeatureLookupResult,
    FeatureMatch,
    FeatureReference,
    HtoTagEvidence,
    RnaFeatureInventoryEvidence,
)

try:
    from pydantic_ai import ModelRetry, RunContext
    from pydantic_ai.tools import ToolDefinition
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

_MAX_FEATURE_QUERIES = 50


def _prepare_data_enrichment_tool(
    ctx: RunContext[DataEnrichmentDependencies],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    """Expose each batched enrichment tool only while its work is pending."""
    deps = ctx.deps
    completed_calls = {call.name for call in deps.toolCalls}
    inspection_complete = "inspect_assay_features_batch" in completed_calls

    if tool_definition.name == "inspect_assay_features_batch":
        return None if inspection_complete else tool_definition
    if tool_definition.name == "find_present_features_batch":
        if not inspection_complete or tool_definition.name in completed_calls:
            return None
        return tool_definition
    return tool_definition


def _assay_modality(
    assay_type: str | None,
    assay_kind: str,
) -> tuple[
    Literal["RNA", "ATAC", "ADT", "HTO", "unsupported"],
    str,
    Literal["persisted", "assayClass", "unknown"],
]:
    """Map persisted types to supported routes, with a mock-store class fallback."""
    if assay_type is not None:
        if assay_type == "RNA":
            return "RNA", assay_type, "persisted"
        if assay_type == "ATAC":
            return "ATAC", assay_type, "persisted"
        if assay_type == "ADT":
            return "ADT", assay_type, "persisted"
        if assay_type == "HTO":
            return "HTO", assay_type, "persisted"
        return "unsupported", assay_type, "persisted"
    if assay_kind == "RNAassay":
        return "RNA", assay_kind, "assayClass"
    if assay_kind == "ATACassay":
        return "ATAC", assay_kind, "assayClass"
    if assay_kind:
        return "unsupported", assay_kind, "assayClass"
    return "unsupported", "Assay", "unknown"


def _feature_tokens(*values: str) -> set[str]:
    """Return literal alphanumeric tokens without accepting generated patterns."""
    text = " ".join(values).casefold()
    normalized = "".join(
        character if character.isalnum() else " " for character in text
    )
    return set(normalized.split())


def _valid_peak_coordinate(value: str) -> bool:
    """Validate the documented ``chrom:start-end`` representation exactly."""
    chromosome, separator, interval = value.partition(":")
    if not separator or not chromosome:
        return False
    start_text, separator, end_text = interval.partition("-")
    if not separator or not start_text or not end_text:
        return False
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError:
        return False
    return start >= 0 and end > start


def _inspect_adt_features(
    assay_name: str,
    feature_rows: list[tuple[str, str]],
) -> list[AdtControlEvidence]:
    """Return control candidates from exact observed ADT features."""
    candidates: list[AdtControlEvidence] = []
    for feature_id, feature_name in feature_rows:
        tokens = _feature_tokens(feature_id, feature_name)
        matched_token: Literal["control", "isotype"] | None = None
        if "isotype" in tokens:
            matched_token = "isotype"
        elif "control" in tokens:
            matched_token = "control"
        if matched_token is None:
            continue
        candidates.append(
            AdtControlEvidence(
                featureId=feature_id,
                featureName=feature_name,
                matchedToken=matched_token,
                evidenceId=f"assay:{assay_name}:adtControl:{feature_id}",
            )
        )
    return candidates


def _inspect_hto_features(
    assay_name: str,
    feature_rows: list[tuple[str, str]],
) -> list[HtoTagEvidence]:
    """Return HTO tag evidence in exact observed order."""
    return [
        HtoTagEvidence(
            featureId=feature_id,
            featureName=feature_name,
            evidenceId=f"assay:{assay_name}:htoTag:{feature_id}",
        )
        for feature_id, feature_name in feature_rows
    ]


def _inspect_atac_features(
    assay_name: str,
    feature_ids: list[str],
) -> AtacCoordinateEvidence:
    """Validate exact observed ATAC coordinates without inferring a build."""
    valid_ids = [value for value in feature_ids if _valid_peak_coordinate(value)]
    invalid_ids = [value for value in feature_ids if not _valid_peak_coordinate(value)]
    if not feature_ids or not valid_ids:
        coordinate_status: Literal["valid", "partial", "invalid"] = "invalid"
    elif invalid_ids:
        coordinate_status = "partial"
    else:
        coordinate_status = "valid"
    return AtacCoordinateEvidence(
        status=coordinate_status,
        totalFeatures=len(feature_ids),
        validFeatures=len(valid_ids),
        validExamples=bounded_list(valid_ids, limit=5),
        invalidExamples=bounded_list(invalid_ids, limit=5),
        evidenceId=f"assay:{assay_name}:atacCoordinates",
    )


def _inspect_modality_features(
    *,
    assay_name: str,
    assay: Any,
    assay_type: str | None,
    assay_kind: str,
    identity: dict[str, Any],
) -> AssayModalityEvidence:
    """Build bounded modality evidence from exact observed feature metadata."""
    modality, resolved_type, type_source = _assay_modality(assay_type, assay_kind)
    modality_evidence_id = f"assay:{assay_name}:modality"
    total_features = int(identity.get("nFeatures", 0))
    evidence_ids = [modality_evidence_id]
    graph_eligible = modality in {"RNA", "ATAC", "ADT"}
    marker_eligible = graph_eligible

    adt_controls: list[AdtControlEvidence] = []
    hto_tags: list[HtoTagEvidence] = []
    atac_coordinates = AtacCoordinateEvidence.get_blank()
    reported_features = 0
    truncated = False

    if modality in {"ADT", "HTO", "ATAC"}:
        feature_ids = [str(value) for value in assay.feats.fetch_all("ids")]
        total_features = len(feature_ids)
    else:
        feature_ids = []

    if modality in {"ADT", "HTO"}:
        feature_names = [str(value) for value in assay.feats.fetch_all("names")]
        feature_rows = list(zip(feature_ids, feature_names, strict=True))
    else:
        feature_rows = []

    if modality == "ADT":
        control_candidates = _inspect_adt_features(assay_name, feature_rows)
        adt_controls = bounded_list(
            control_candidates,
            limit=_MAX_FEATURE_QUERIES,
        )
        evidence_ids.extend(item.evidenceId for item in adt_controls)
        reported_features = len(adt_controls)
        truncated = len(control_candidates) > len(adt_controls)

    if modality == "HTO":
        limited_rows = bounded_list(feature_rows, limit=_MAX_FEATURE_QUERIES)
        hto_tags = _inspect_hto_features(assay_name, limited_rows)
        evidence_ids.extend(item.evidenceId for item in hto_tags)
        reported_features = len(hto_tags)
        truncated = len(feature_rows) > len(limited_rows)

    if modality == "ATAC":
        atac_coordinates = _inspect_atac_features(
            assay_name,
            feature_ids,
        )
        evidence_ids.append(atac_coordinates.evidenceId)
        reported_features = len(atac_coordinates.validExamples) + len(
            atac_coordinates.invalidExamples
        )
        truncated = len(feature_ids) > reported_features

    return AssayModalityEvidence(
        assayType=resolved_type,
        modality=modality,
        typeSource=type_source,
        graphEligible=graph_eligible,
        markerEligible=marker_eligible,
        demultiplexEligible=modality == "HTO",
        adtControls=adt_controls,
        htoTags=hto_tags,
        atacCoordinates=atac_coordinates,
        totalObservedFeatures=total_features,
        reportedFeatures=reported_features,
        truncated=truncated,
        evidenceIds=evidence_ids,
    )


async def inspect_assay_features(
    ctx: RunContext[DataEnrichmentDependencies],
    assay_name: str,
) -> AssayFeatureInspection:
    """Inspect feature identity, species evidence, families, and exogenous cues."""
    deps = ctx.deps
    if deps.store is None:
        raise ModelRetry("The datastore is unavailable")
    if assay_name not in deps.assays:
        raise ModelRetry(
            f"assay_name must be one of the requested assays: {deps.assays}"
        )
    cached = deps.inspections.get(assay_name)
    if cached is not None:
        logger.debug(
            f"Data Enrichment reused cached inspection for assay {assay_name!r}"
        )
        return cached

    characterization = characterize_features(
        deps.store,
        studyContext=deps.context.studyContext,
        model=None,
        assays=[assay_name],
        cacheDir=deps.cacheDir,
        allowDownload=deps.allowDownload,
    )
    if characterization.status != "done" or not characterization.assays:
        logger.warning(f"Data Enrichment could not characterize assay {assay_name!r}")
        detail = "; ".join(characterization.notes) or "feature inspection failed"
        raise ModelRetry(detail)

    record = characterization.assays[0]
    family_evidence: list[FeatureFamilyEvidence] = []
    evidence_ids = [f"assay:{assay_name}:identity", f"assay:{assay_name}:species"]
    raw_default_inventory = record.get("defaultFeatureInventory")
    default_inventory: RnaFeatureInventoryEvidence | None = None
    if raw_default_inventory is not None:
        default_inventory = RnaFeatureInventoryEvidence.model_validate(
            raw_default_inventory
        )
        if default_inventory.blacklist != DEFAULT_HVG_BLACKLIST:
            raise ModelRetry(
                "RNA feature inventory does not use Scarf's exact default HVG blacklist"
            )
        evidence_prefix = f"assay:{assay_name}:scarfDefaultHvg"
        default_inventory.evidenceId = f"{evidence_prefix}:combined"
        for family in default_inventory.families:
            family.evidenceId = f"{evidence_prefix}:family:{family.family}"
        default_inventory.evidenceIds = [
            default_inventory.evidenceId,
            *(family.evidenceId for family in default_inventory.families),
        ]
        evidence_ids.extend(default_inventory.evidenceIds)
    for family in record.get("families", []):
        family_name = str(family.get("family", ""))
        evidence_id = f"assay:{assay_name}:family:{family_name}"
        family_evidence.append(
            FeatureFamilyEvidence(
                family=family_name,
                species=str(family.get("species", record.get("species", "unknown"))),
                method=str(family.get("method", "")),
                count=int(family.get("count", 0)),
                examples=[str(value) for value in family.get("examples", [])],
                defaultExclude=family.get("defaultExclude"),
                skipped=family.get("skipped"),
                catalogSuspect=family.get("catalogSuspect"),
                catalogSize=family.get("catalogSize"),
                catalogJoinRate=family.get("catalogJoinRate"),
                catalogJoined=family.get("catalogJoined"),
                evidenceId=evidence_id,
            )
        )
        evidence_ids.append(evidence_id)

    exogenous_evidence: list[ExogenousFeatureEvidence] = []
    for item in record.get("exogenous", []):
        feature_id = str(item.get("id", ""))
        evidence_id = f"assay:{assay_name}:exogenous:{feature_id}"
        exogenous_evidence.append(
            ExogenousFeatureEvidence(
                featureId=feature_id,
                featureName=str(item.get("name", "")),
                score=int(item.get("score", 0)),
                classification=str(item.get("class", "unresolved")),
                evidenceId=evidence_id,
            )
        )
        evidence_ids.append(evidence_id)

    resolution = record.get("speciesResolution") or {}
    assay = deps.store.get_assay(assay_name)
    identity = dict(record.get("identity") or {})
    modality_evidence = _inspect_modality_features(
        assay_name=assay_name,
        assay=assay,
        assay_type=deps.assayTypes.get(assay_name),
        assay_kind=str(record.get("assayKind", "")),
        identity=identity,
    )
    evidence_ids.extend(modality_evidence.evidenceIds)
    inspection = AssayFeatureInspection(
        assay=assay_name,
        assayKind=str(record.get("assayKind", "")),
        identity=identity,
        species=str(record.get("species", "unknown")),
        speciesMethod=record.get("speciesMethod"),
        speciesReason=str(resolution.get("reason", "")),
        families=family_evidence,
        defaultFeatureInventory=default_inventory,
        exogenous=exogenous_evidence,
        modalityEvidence=modality_evidence,
        notes=[str(value) for value in record.get("notes", [])],
        evidenceIds=evidence_ids,
    )
    deps.inspections[assay_name] = inspection
    deps.evidenceIds.update(evidence_ids)
    deps.toolCalls.append(
        DataEnrichmentToolCall(
            name="inspect_assay_features",
            assay=assay_name,
            evidenceIds=evidence_ids,
        )
    )
    logger.debug(
        "Data Enrichment inspected "
        f"assay={assay_name!r}, modality={modality_evidence.modality}, "
        f"species={inspection.species}, families={len(family_evidence)}, "
        f"defaultHvgMatches="
        f"{default_inventory.matchCount if default_inventory is not None else 0}, "
        f"exogenous={len(exogenous_evidence)}, evidence={len(evidence_ids)}"
    )
    return inspection


async def inspect_assay_features_batch(
    ctx: RunContext[DataEnrichmentDependencies],
) -> AssayFeatureInspectionBatch:
    """Inspect every requested assay and return one bounded tool result."""
    deps = ctx.deps
    if not deps.assays:
        raise ModelRetry("No assays were requested")
    if any(
        call.name == "inspect_assay_features_batch" for call in deps.toolCalls
    ) and all(assay_name in deps.inspections for assay_name in deps.assays):
        inspections = [deps.inspections[assay_name] for assay_name in deps.assays]
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for inspection in inspections
                for evidence_id in inspection.evidenceIds
            )
        )
        logger.info("Data Enrichment reused the completed feature inspection batch")
        return AssayFeatureInspectionBatch(
            inspections=inspections,
            evidenceIds=evidence_ids,
        )
    logger.info(
        f"Data Enrichment feature inspection started for {len(deps.assays)} assays"
    )
    start = len(deps.toolCalls)
    try:
        inspections = [
            await inspect_assay_features(ctx, assay_name=assay_name)
            for assay_name in deps.assays
        ]
    except Exception:
        del deps.toolCalls[start:]
        raise
    del deps.toolCalls[start:]
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for inspection in inspections
            for evidence_id in inspection.evidenceIds
        )
    )
    deps.toolCalls.append(
        DataEnrichmentToolCall(
            name="inspect_assay_features_batch",
            assay=",".join(deps.assays),
            evidenceIds=evidence_ids,
        )
    )
    supported_routes = sum(
        inspection.modalityEvidence.modality != "unsupported"
        for inspection in inspections
    )
    logger.info(
        "Data Enrichment feature inspection completed: "
        f"assays={len(inspections)}, supportedRoutes={supported_routes}, "
        f"evidence={len(evidence_ids)}"
    )
    return AssayFeatureInspectionBatch(
        inspections=inspections,
        evidenceIds=evidence_ids,
    )


async def find_present_features(
    ctx: RunContext[DataEnrichmentDependencies],
    assay_name: str,
    queries: list[str],
) -> FeatureLookupResult:
    """Resolve a bounded list of gene IDs or names against one exact assay."""
    deps = ctx.deps
    if deps.store is None:
        raise ModelRetry("The datastore is unavailable")
    if assay_name not in deps.assays:
        raise ModelRetry(
            f"assay_name must be one of the requested assays: {deps.assays}"
        )
    clean_queries = list(
        dict.fromkeys(value.strip() for value in queries if value.strip())
    )
    if not clean_queries or len(clean_queries) > _MAX_FEATURE_QUERIES:
        raise ModelRetry(
            f"queries must contain between 1 and {_MAX_FEATURE_QUERIES} values"
        )

    assay = deps.store.get_assay(assay_name)
    feature_ids = [str(value) for value in assay.feats.fetch_all("ids")]
    feature_names = [str(value) for value in assay.feats.fetch_all("names")]
    rows = list(zip(feature_ids, feature_names, strict=True))
    results: list[FeatureMatch] = []
    result_evidence_ids: list[str] = []
    confirmed = deps.confirmedFeatures.setdefault(assay_name, set())

    for query in clean_queries:
        exact = [row for row in rows if query in row]
        candidates = exact
        if not candidates:
            folded = query.casefold()
            candidates = [
                row
                for row in rows
                if folded == row[0].casefold() or folded == row[1].casefold()
            ]
        unique_candidates = bounded_list(
            dict.fromkeys(candidates),
            limit=10,
        )
        references = [
            FeatureReference(featureId=feature_id, featureName=feature_name)
            for feature_id, feature_name in unique_candidates
        ]
        evidence_ids = [
            f"assay:{assay_name}:feature:{reference.featureId}"
            for reference in references
        ]
        if len(references) == 1:
            status: Literal["present", "ambiguous", "absent"] = "present"
            confirmed.update({references[0].featureId, references[0].featureName})
            deps.evidenceIds.update(evidence_ids)
            result_evidence_ids.extend(evidence_ids)
        elif references:
            status = "ambiguous"
        else:
            status = "absent"
        results.append(
            FeatureMatch(
                query=query,
                status=status,
                matches=references,
                evidenceIds=evidence_ids if status == "present" else [],
            )
        )

    result = FeatureLookupResult(
        assay=assay_name,
        results=results,
        evidenceIds=list(dict.fromkeys(result_evidence_ids)),
    )
    deps.toolCalls.append(
        DataEnrichmentToolCall(
            name="find_present_features",
            assay=assay_name,
            evidenceIds=result.evidenceIds,
        )
    )
    return result


async def find_present_features_batch(
    ctx: RunContext[DataEnrichmentDependencies],
    queries_by_assay: dict[str, list[str]],
) -> FeatureLookupBatch:
    """Resolve all proposed individual features through one model tool call."""
    deps = ctx.deps
    unknown_assays = sorted(set(queries_by_assay) - set(deps.assays))
    if unknown_assays:
        raise ModelRetry(f"Unknown requested assays: {unknown_assays}")
    if not queries_by_assay:
        raise ModelRetry("queries_by_assay must contain at least one assay")
    clean_queries_by_assay = {
        assay_name: list(
            dict.fromkeys(value.strip() for value in queries if value.strip())
        )
        for assay_name, queries in queries_by_assay.items()
    }
    empty_assays = sorted(
        assay_name
        for assay_name, queries in clean_queries_by_assay.items()
        if not queries
    )
    if empty_assays:
        raise ModelRetry(f"Feature-query batches cannot be empty: {empty_assays}")
    query_count = sum(len(queries) for queries in clean_queries_by_assay.values())
    if query_count > _MAX_FEATURE_QUERIES:
        raise ModelRetry(
            "The batch may contain at most "
            f"{_MAX_FEATURE_QUERIES} feature queries in total"
        )
    if deps.lookupBatch is not None:
        if clean_queries_by_assay != deps.lookupQueries:
            raise ModelRetry(
                "Feature lookup already completed. Use only the returned lookup "
                "evidence and do not request a different batch."
            )
        logger.info("Data Enrichment reused the completed feature lookup batch")
        return deps.lookupBatch

    logger.info(
        "Data Enrichment feature lookup started: "
        f"assays={len(clean_queries_by_assay)}, queries={query_count}"
    )

    start = len(deps.toolCalls)
    lookups = [
        await find_present_features(
            ctx,
            assay_name=assay_name,
            queries=clean_queries_by_assay[assay_name],
        )
        for assay_name in deps.assays
        if assay_name in clean_queries_by_assay
    ]
    del deps.toolCalls[start:]
    evidence_ids = list(
        dict.fromkeys(
            evidence_id for lookup in lookups for evidence_id in lookup.evidenceIds
        )
    )
    deps.toolCalls.append(
        DataEnrichmentToolCall(
            name="find_present_features_batch",
            assay=",".join(queries_by_assay),
            evidenceIds=evidence_ids,
        )
    )
    result_counts = {"present": 0, "ambiguous": 0, "absent": 0}
    for lookup in lookups:
        for result in lookup.results:
            result_counts[result.status] += 1
    logger.info(
        "Data Enrichment feature lookup completed: "
        f"present={result_counts['present']}, "
        f"ambiguous={result_counts['ambiguous']}, "
        f"absent={result_counts['absent']}, evidence={len(evidence_ids)}"
    )
    batch = FeatureLookupBatch(lookups=lookups, evidenceIds=evidence_ids)
    deps.lookupBatch = batch
    deps.lookupQueries = clean_queries_by_assay
    return batch
