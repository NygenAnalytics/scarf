"""Characterize cell covariates and study-design confounding."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ..metadata.queries import (
    PartitionDigest,
    column_constant_within,
    column_partition_digest,
    columns_same_partition,
    reduce_observation_units,
)
from ..metrics.association import directional_mapping, report_confounding
from ..storage.types import as_zarr_array, as_zarr_group
from .config import CONFIG
from .config._deps import AGENT_INSTALL_HINT
from .decide import DecisionValidationError, decide
from .types import AgentDataModel, Decision, EvidenceItem, StageStatus

try:
    from pydantic import Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "CovariateCharacterization",
    "characterize_covariates",
]

Domain = Literal["biological", "technical", "design", "ignore", "unknown"]
ColumnKind = Literal["categorical", "continuous"]


_DOMAIN_EVIDENCE = [
    EvidenceItem(
        id="domain:biological",
        label="biological",
        summary="Biology of interest such as disease, sex, genotype, treatment",
    ),
    EvidenceItem(
        id="domain:technical",
        label="technical",
        summary="Technical handling such as batch, chemistry, sequencing run",
    ),
    EvidenceItem(
        id="domain:design",
        label="design",
        summary="Sampling design such as donor, sample, replicate, subject",
    ),
    EvidenceItem(
        id="domain:ignore",
        label="ignore",
        summary="Identifiers, QC metrics, clusters, or other non-design labels",
    ),
    EvidenceItem(
        id="domain:unknown",
        label="unknown",
        summary="Cannot classify from available evidence",
    ),
]
_COEFFICIENT_EVIDENCE = [
    EvidenceItem(
        id="coefficient:yes",
        label="yes",
        summary="Treat this biological column as a coefficient of interest",
    ),
    EvidenceItem(
        id="coefficient:no",
        label="no",
        summary="Do not treat this biological column as a coefficient of interest",
    ),
]


class CovariateCharacterization(AgentDataModel):
    status: StageStatus
    auditLog: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[dict[str, Any]] = Field(default_factory=list)
    coefficients: list[dict[str, Any]] = Field(default_factory=list)
    technicalNesting: list[dict[str, Any]] = Field(default_factory=list)
    confounding: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "CovariateCharacterization":
        return cls(status="failed")

    @classmethod
    def get_example(cls) -> "CovariateCharacterization":
        return cls(
            status="done",
            notes=["Cell covariates and confounding were characterized."],
            columns=[{"name": "batch", "domain": "technical"}],
        )


@dataclass(frozen=True, slots=True)
class _ColumnProfile:
    kind: ColumnKind
    summary: str
    digest: PartitionDigest


@dataclass
class _Run:
    """Mutable state shared by the stage steps."""

    store: Any
    cell_key: str
    n_rows: int
    context: str
    model: Any | None
    profiles: dict[str, _ColumnProfile] = field(default_factory=dict)
    domains: dict[str, Domain] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def note(self, *, kind: str, detail: str, **fields: Any) -> None:
        self.audit.append({"kind": kind, "detail": detail, **fields})

    def summary(self, name: str) -> str:
        return self.profiles[name].summary

    def kind(self, name: str) -> ColumnKind:
        return self.profiles[name].kind

    def digest(self, name: str) -> PartitionDigest:
        return self.profiles[name].digest

    def ask(
        self,
        *,
        task: str,
        question: str,
        evidence: Sequence[EvidenceItem],
        column: str | None = None,
    ) -> Decision | None:
        """Run one grounded decision, or return None when it cannot be asked."""
        if self.model is None or len(evidence) < 2:
            return None
        try:
            decision = decide(model=self.model, question=question, evidence=evidence)
        except DecisionValidationError as exc:
            self.note(
                kind="decisionInvalid",
                detail=str(exc),
                task=task,
                column=column,
            )
            return None
        record: dict[str, Any] = {
            "task": task,
            "selectedId": decision.selectedId,
            "rationale": decision.rationale,
            "evidenceIds": list(decision.evidenceIds),
        }
        if column is not None:
            record["column"] = column
        self.decisions.append(record)
        return decision


def _is_embedding_column(name: str) -> bool:
    match = CONFIG._INDEXED_NAME.fullmatch(name)
    if match is None:
        return False
    parts = [part for part in re.split(r"[-_]+", match.group("stem").lower()) if part]
    compact = "".join(parts)
    if any(token in compact for token in CONFIG._EMBEDDING_TOKENS):
        return True
    return any(part in CONFIG._SHORT_EMBEDDING_PARTS for part in parts)


def _has_source_artifact(store: Any, column: str) -> bool:
    try:
        cell_data = as_zarr_group(store.zw["cellData"], name="cellData")
        if column not in cell_data:
            return False
        attrs = as_zarr_array(cell_data[column], name=column).attrs
    except (KeyError, TypeError, ValueError):
        return False
    return isinstance(attrs.get("source_artifact"), dict)


def _infer_kind(values: np.ndarray) -> ColumnKind:
    if (
        values.dtype == object
        or np.issubdtype(values.dtype, np.str_)
        or np.issubdtype(values.dtype, np.bool_)
    ):
        return "categorical"
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return "categorical"
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0 or not bool(np.all(np.mod(finite, 1) == 0)):
        return "continuous"
    limit = min(CONFIG._CATEGORICAL_MAX_LEVELS, max(2, len(values) // 20))
    return "categorical" if int(np.unique(finite).size) <= limit else "continuous"


def _summarize(values: np.ndarray, kind: ColumnKind) -> str:
    series = pd.Series(values)
    missing = int(series.isna().sum())
    if kind == "categorical":
        levels = series.dropna().astype(str).value_counts()
        top = ", ".join(
            f"{level}={int(count)}"
            for level, count in levels.head(CONFIG._SAMPLE_LEVELS).items()
        )
        return f"categorical levels={levels.shape[0]} missing={missing} top=[{top}]"
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return f"continuous missing={missing} finite=0"
    return (
        f"continuous missing={missing} min={float(finite.min()):.4g} "
        f"max={float(finite.max()):.4g} mean={float(finite.mean()):.4g}"
    )


def _digest_key(digest: PartitionDigest) -> tuple[bytes, int, int]:
    return (digest.digest, digest.nLevels, digest.nMissing)


def _profile_column(
    store: Any,
    name: str,
    *,
    cell_key: str,
    kind: ColumnKind | None = None,
) -> _ColumnProfile:
    values = store.cells.fetch(name, key=cell_key)
    resolved_kind = kind or _infer_kind(values)
    summary = _summarize(values, resolved_kind)
    digest = column_partition_digest(store.cells, name, cell_key=cell_key)
    return _ColumnProfile(kind=resolved_kind, summary=summary, digest=digest)


def _triage_columns(
    store: Any,
    *,
    cell_key: str,
    exclude: set[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split cell columns into model candidates and deterministic drops."""
    assay_prefixes = tuple(f"{name}_" for name in store.assay_names)
    candidates: list[str] = []
    dropped: list[tuple[str, str]] = []
    for name in store.cells.columns:
        if name in CONFIG._RESERVED_COLUMNS or name == cell_key or name in exclude:
            continue
        if name.startswith(assay_prefixes):
            dropped.append((name, "dropAssayStat"))
        elif _has_source_artifact(store, name):
            dropped.append((name, "dropProvenance"))
        elif _is_embedding_column(name):
            dropped.append((name, "dropEmbedding"))
        else:
            candidates.append(name)
    return candidates, dropped


def _collapse_ontology_aliases(
    store: Any,
    columns: Sequence[str],
    profiles: Mapping[str, _ColumnProfile],
    *,
    cell_key: str,
) -> tuple[list[str], dict[str, list[str]], list[dict[str, Any]]]:
    """Collapse ``x`` with ``x_ontology_term_id`` when partitions match.

    Arbitrary identical partitions are kept apart: perfect confounding between
    biology and batch is a finding to report, not an alias to drop.
    """
    present = set(columns)
    aliases: dict[str, list[str]] = {}
    dropped: set[str] = set()
    notes: list[dict[str, Any]] = []
    for name in columns:
        if not name.endswith(CONFIG._ONTOLOGY_SUFFIX):
            continue
        base = name[: -len(CONFIG._ONTOLOGY_SUFFIX)]
        if base not in present or {name, base} & dropped:
            continue
        if _digest_key(profiles[name].digest) != _digest_key(profiles[base].digest):
            continue
        same, correspondence = columns_same_partition(
            store.cells,
            base,
            name,
            cell_key=cell_key,
        )
        if not same:
            continue
        aliases.setdefault(base, []).append(name)
        dropped.add(name)
        note: dict[str, Any] = {
            "kind": "ontologyAlias",
            "detail": f"Collapsed ontology alias {name} onto {base}",
            "representative": base,
            "aliases": [name],
        }
        if correspondence:
            note["levels"] = correspondence
        notes.append(note)
    return [name for name in columns if name not in dropped], aliases, notes


def _bounded_context(study_context: str | None) -> str:
    text = (study_context or "").strip()
    return (
        text
        if len(text) <= CONFIG._CONTEXT_LIMIT
        else text[: CONFIG._CONTEXT_LIMIT - 3] + "..."
    )


def _validate_directions(
    directions: Mapping[str, Any],
    available: set[str],
) -> list[str]:
    errors: list[str] = []

    def check_names(key: str, names: Any) -> list[str] | None:
        if not isinstance(names, Sequence) or isinstance(names, str | bytes):
            errors.append(f"{key} must be a sequence of column names")
            return None
        unknown = sorted(set(names) - available)
        if unknown:
            errors.append(f"{key} cites unknown columns: {unknown}")
        return list(names)

    for key, allowed in (
        ("columnKinds", CONFIG._KINDS),
        ("columnDomains", CONFIG._DOMAINS),
    ):
        mapping = directions.get(key)
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            errors.append(f"{key} must be a mapping")
            continue
        check_names(key, list(mapping))
        invalid = sorted({str(value) for value in mapping.values()} - allowed)
        if invalid:
            errors.append(f"{key} has unsupported values: {invalid}")

    for key in ("coefficientsOfInterest", "excludeColumns"):
        names = directions.get(key)
        if names is not None:
            check_names(key, names)

    units = directions.get("unitsOfInference")
    if units is None:
        return errors
    if not isinstance(units, Mapping):
        errors.append("unitsOfInference must be a mapping")
        return errors
    for coefficient, unit_map in units.items():
        if coefficient not in available:
            errors.append(f"unitsOfInference cites unknown coefficient {coefficient!r}")
            continue
        if not isinstance(unit_map, Mapping):
            errors.append(f"unitsOfInference[{coefficient!r}] must be a mapping")
            continue
        for unit_key in ("observationUnit", "independentUnit"):
            unit_name = unit_map.get(unit_key)
            if unit_name is not None and unit_name not in available:
                errors.append(
                    f"unitsOfInference[{coefficient!r}].{unit_key} "
                    f"cites unknown column {unit_name!r}"
                )
    return errors


def _choose_representative(
    run: _Run,
    members: Sequence[str],
    correspondence: str,
) -> str | None:
    decision = run.ask(
        task="equivalentColumns",
        question=(
            f"Columns {', '.join(members)} assign every cell to the same groups. "
            f"Their labels line up as {correspondence}. Decide whether they record "
            "one variable under different labels, and if so whose labels to keep."
        ),
        evidence=[
            EvidenceItem(
                id="equivalent:distinct",
                label="distinct variables",
                summary="Different variables that happen to coincide in this dataset",
            ),
            *(
                EvidenceItem(
                    id=f"equivalent:{name}",
                    label=name,
                    summary=run.summary(name),
                )
                for name in members
            ),
        ],
    )
    selected = (
        None if decision is None else decision.selectedId.removeprefix("equivalent:")
    )
    if selected in set(members):
        run.actions.append(f"equivalentColumns:{selected}")
        return selected
    run.note(
        kind="equivalentKeptApart",
        detail=(
            f"{', '.join(members)} share one partition but were not collapsed "
            + ("(no model decision)" if decision is None else "(judged distinct)")
        ),
        columns=list(members),
        levels=correspondence,
    )
    return None


def _collapse_equivalent_columns(
    run: _Run,
    candidates: Sequence[str],
    aliases: dict[str, list[str]],
) -> list[str]:
    """Collapse categorical columns that cut the cells into identical groups.

    An identical partition is also what perfect confounding looks like, so a
    class is only eligible when every member carries the same analysis domain,
    and the representative is a model judgement rather than a name rule.
    Extends ``aliases`` in place.
    """
    classes: dict[tuple[bytes, int, int], list[str]] = {}
    for name in candidates:
        if run.kind(name) != "categorical" or run.domains[name] not in CONFIG._ANALYSED:
            continue
        classes.setdefault(_digest_key(run.digest(name)), []).append(name)

    # Store column order is not stable, and members of a class are
    # interchangeable, so order them here to keep prompts and notes reproducible.
    dropped: set[str] = set()
    for members in sorted(sorted(group) for group in classes.values()):
        if len(members) < 2:
            continue
        representative_name = members[0]
        verified: list[str] = [representative_name]
        correspondence = ""
        for other in members[1:]:
            same, corr = columns_same_partition(
                run.store.cells,
                representative_name,
                other,
                cell_key=run.cell_key,
            )
            if not same:
                continue
            verified.append(other)
            correspondence = corr
        if len(verified) < 2:
            continue
        members = verified
        domains = sorted({run.domains[name] for name in members})
        if len(domains) > 1:
            run.note(
                kind="equivalentAcrossDomains",
                detail=(
                    f"{', '.join(members)} share one partition across domains "
                    f"{domains}; kept apart as perfect confounding"
                ),
                columns=list(members),
                domains=domains,
                levels=correspondence,
            )
            continue
        representative = _choose_representative(run, members, correspondence)
        if representative is None:
            continue
        others = [name for name in members if name != representative]
        aliases.setdefault(representative, []).extend(others)
        dropped.update(others)
        run.note(
            kind="equivalentColumns",
            detail=f"Collapsed {', '.join(others)} onto {representative}",
            representative=representative,
            aliases=others,
            levels=correspondence,
        )
    return [name for name in candidates if name not in dropped]


def _assign_domain(run: _Run, name: str, directed: Mapping[str, Domain]) -> Domain:
    if name in directed:
        domain = directed[name]
        run.actions.append(f"domain:{name}->{domain} (directions)")
        return domain
    decision = run.ask(
        task="columnDomain",
        column=name,
        question=(
            f"Assign a domain for cell metadata column {name}. "
            f"Column summary: {run.summary(name)}. "
            f"Study context: {run.context or 'none provided'}."
        ),
        evidence=_DOMAIN_EVIDENCE,
    )
    if decision is None:
        run.note(
            kind="domainUnknown",
            detail=f"Left {name} as unknown domain",
            column=name,
        )
        return "unknown"
    selected = decision.selectedId.removeprefix("domain:")
    if selected not in CONFIG._DOMAINS:
        run.note(
            kind="domainUnknown",
            detail=f"Unsupported domain {selected!r} returned for {name}",
            column=name,
        )
        return "unknown"
    run.actions.append(f"domain:{name}->{selected}")
    return cast(Domain, selected)


def _select_coefficients(
    run: _Run,
    *,
    biological: Sequence[str],
    directed: set[str],
) -> list[str]:
    selected: list[str] = []
    for name in biological:
        if name in directed:
            selected.append(name)
            run.actions.append(f"coefficient:{name} (directions)")
            continue
        decision = run.ask(
            task="coefficientOfInterest",
            column=name,
            question=(
                f"Should biological column {name} be a coefficient of interest? "
                f"Column summary: {run.summary(name)}. "
                f"Study context: {run.context or 'none provided'}."
            ),
            evidence=_COEFFICIENT_EVIDENCE,
        )
        if decision is None:
            run.note(
                kind="coefficientSkipped",
                detail=f"No model or direction for biological column {name}",
                column=name,
            )
        elif decision.selectedId == "coefficient:yes":
            selected.append(name)
            run.actions.append(f"coefficient:{name}")
    return selected


def _unit_evidence(run: _Run, names: Sequence[str], prefix: str) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id=f"{prefix}:{name}",
            label=name,
            summary=(f"domain={run.domains.get(name, 'unknown')}; {run.summary(name)}"),
        )
        for name in names
    ]


def _observation_unit_candidates(
    run: _Run,
    coefficient: str,
    pool: Sequence[str],
) -> list[str]:
    """Design/technical columns that can host a between-unit coefficient.

    A valid observation unit is a metadata fact, not a judgement: the coefficient
    must be constant within each level, and the unit must leave more than one
    design row while still being coarser than cell-level variation.
    """
    return [
        name
        for name in pool
        if name != coefficient
        and name in run.profiles
        and column_constant_within(
            run.store.cells,
            coefficient,
            name,
            cell_key=run.cell_key,
        )
        and 2 <= run.digest(name).nLevels < run.n_rows
    ]


def _independent_is_coarser(
    store: Any,
    *,
    cell_key: str,
    observation: str,
    independent: str,
) -> bool:
    """True when each observation level maps to one independent level.

    The independent unit must be coarser (or equal), never finer. A finer
    independent unit inflates the design table with pseudo-replicated rows.
    """
    observation_values = store.cells.fetch(observation, key=cell_key)
    independent_values = store.cells.fetch(independent, key=cell_key)
    nesting = directional_mapping(observation_values, independent_values).get("nesting")
    return nesting in {"leftInRight", "equivalent"}


def _resolve_units(
    run: _Run,
    coefficient: str,
    *,
    directed: Mapping[str, Any],
    design_columns: Sequence[str],
    unit_candidates: Sequence[str],
) -> tuple[str | None, str | None]:
    unit_map = dict(directed.get(coefficient) or {})
    observation = unit_map.get("observationUnit")
    independent = unit_map.get("independentUnit")
    valid_observation = _observation_unit_candidates(run, coefficient, unit_candidates)

    if observation is not None:
        if observation not in run.profiles:
            run.note(
                kind="invalidObservationUnit",
                detail=f"Directed observation unit {observation!r} is missing",
                column=coefficient,
                observationUnit=observation,
            )
            observation = None
        elif not column_constant_within(
            run.store.cells,
            coefficient,
            observation,
            cell_key=run.cell_key,
        ):
            # Keep the directed unit; _characterize_coefficient records withinUnit.
            pass
        elif observation not in valid_observation:
            run.note(
                kind="invalidObservationUnit",
                detail=(
                    f"Directed observation unit {observation!r} is vacuous or "
                    f"otherwise invalid for {coefficient}"
                ),
                column=coefficient,
                observationUnit=observation,
            )
            observation = None
    elif len(valid_observation) == 1:
        observation = valid_observation[0]
        run.actions.append(f"observationUnit:{coefficient}->{observation}")
    elif len(valid_observation) >= 2:
        decision = run.ask(
            task="observationUnit",
            column=coefficient,
            question=(
                f"Choose the observation unit for coefficient {coefficient}. "
                "Only columns where this coefficient is constant within each "
                "level are listed. Each distinct value is one design-table row. "
                f"Study context: {run.context or 'none provided'}."
            ),
            evidence=_unit_evidence(run, valid_observation, "unit"),
        )
        if decision is not None:
            observation = decision.selectedId.removeprefix("unit:")
            if observation not in valid_observation:
                run.note(
                    kind="invalidObservationUnit",
                    detail=(
                        f"Model chose {observation!r}, which is not a valid "
                        f"observation unit for {coefficient}"
                    ),
                    column=coefficient,
                    observationUnit=observation,
                )
                observation = None
            else:
                run.actions.append(f"observationUnit:{coefficient}->{observation}")
    else:
        run.note(
            kind="noValidObservationUnit",
            detail=(
                f"No design/technical column keeps {coefficient} constant; "
                "cannot build a between-unit design table from available metadata"
            ),
            column=coefficient,
        )

    if observation is None:
        return None, None

    valid_independent = [
        name
        for name in design_columns
        if name not in {coefficient, observation}
        and name in run.profiles
        and _independent_is_coarser(
            run.store,
            cell_key=run.cell_key,
            observation=observation,
            independent=name,
        )
    ]

    if independent is not None:
        if independent not in run.profiles:
            run.note(
                kind="invalidIndependentUnit",
                detail=f"Directed independent unit {independent!r} is missing",
                column=coefficient,
                independentUnit=independent,
            )
            independent = None
        elif not _independent_is_coarser(
            run.store,
            cell_key=run.cell_key,
            observation=observation,
            independent=independent,
        ):
            run.note(
                kind="independentUnitFiner",
                detail=(
                    f"Independent unit {independent!r} is finer than observation "
                    f"unit {observation!r}; dropped to avoid pseudo-replicated "
                    "design rows"
                ),
                column=coefficient,
                observationUnit=observation,
                independentUnit=independent,
            )
            independent = None
    elif valid_independent:
        decision = run.ask(
            task="independentUnit",
            column=coefficient,
            question=(
                f"Optional independent unit for coefficient {coefficient} "
                f"with observation unit {observation}. Listed columns are "
                "coarser than the observation unit (each observation level "
                "maps to one independent level)."
            ),
            evidence=[
                EvidenceItem(
                    id="independentUnit:none",
                    label="none",
                    summary="No separate independent unit or subject column",
                ),
                *_unit_evidence(run, valid_independent, "independentUnit"),
            ],
        )
        if decision is not None and decision.selectedId != "independentUnit:none":
            chosen = decision.selectedId.removeprefix("independentUnit:")
            if chosen not in valid_independent:
                run.note(
                    kind="invalidIndependentUnit",
                    detail=(
                        f"Model chose {chosen!r}, which is not coarser than "
                        f"observation unit {observation!r}"
                    ),
                    column=coefficient,
                    independentUnit=chosen,
                )
            else:
                independent = chosen
                run.actions.append(f"independentUnit:{coefficient}->{independent}")
    return observation, independent


def _characterize_coefficient(
    run: _Run,
    coefficient: str,
    *,
    observation_unit: str | None,
    independent_unit: str | None,
    technical: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record: dict[str, Any] = {
        "name": coefficient,
        "kind": run.kind(coefficient),
        "observationUnit": observation_unit,
        "independentUnit": independent_unit,
        "scope": "unresolvedUnit",
    }
    if observation_unit is None or observation_unit not in run.profiles:
        run.note(
            kind="unresolvedUnit",
            detail=f"No usable observation unit for coefficient {coefficient}",
            column=coefficient,
        )
        return record, None

    if not column_constant_within(
        run.store.cells,
        coefficient,
        observation_unit,
        cell_key=run.cell_key,
    ):
        record["scope"] = "withinUnit"
        run.note(
            kind="withinUnit",
            detail=(
                f"{coefficient} varies within {observation_unit}; recorded as "
                "composition and skipped for design-table association"
            ),
            column=coefficient,
            observationUnit=observation_unit,
        )
        return record, None

    # Technically only columns constant inside the observation unit have a
    # well-defined design-table value; drop_duplicates would otherwise keep an
    # arbitrary cell row.
    unit_constant: list[str] = []
    for name in technical:
        if name not in run.profiles:
            continue
        if column_constant_within(
            run.store.cells,
            name,
            observation_unit,
            cell_key=run.cell_key,
        ):
            unit_constant.append(name)
            continue
        run.note(
            kind="technicalVariesWithinUnit",
            detail=(
                f"{name} varies within {observation_unit}; "
                f"excluded from design-table association for {coefficient}"
            ),
            column=name,
            coefficient=coefficient,
            observationUnit=observation_unit,
        )

    group_cols = [observation_unit]
    if independent_unit is not None and independent_unit in run.profiles:
        if _independent_is_coarser(
            run.store,
            cell_key=run.cell_key,
            observation=observation_unit,
            independent=independent_unit,
        ):
            group_cols.append(independent_unit)
        else:
            run.note(
                kind="independentUnitFiner",
                detail=(
                    f"Independent unit {independent_unit!r} is finer than "
                    f"observation unit {observation_unit!r}; omitted from "
                    "design table"
                ),
                column=coefficient,
                observationUnit=observation_unit,
                independentUnit=independent_unit,
            )
            independent_unit = None
            record["independentUnit"] = None
    columns = list(dict.fromkeys([*group_cols, coefficient, *unit_constant]))
    design = reduce_observation_units(
        run.store.cells,
        observation_unit,
        columns,
        cell_key=run.cell_key,
    )
    design_rows = int(len(design))
    if not (2 <= design_rows < run.n_rows):
        run.note(
            kind="invalidObservationUnit",
            detail=(
                f"Observation unit {observation_unit!r} yields {design_rows} "
                f"design rows for {coefficient}; need at least 2 and fewer "
                f"than {run.n_rows} active cells"
            ),
            column=coefficient,
            observationUnit=observation_unit,
            designRows=design_rows,
        )
        record["scope"] = "unresolvedUnit"
        return record, None

    record["scope"] = "betweenUnit"
    record["designRows"] = design_rows

    report = report_confounding(
        design,
        coefficient=coefficient,
        technicalColumns=unit_constant,
        columnKinds={
            coefficient: run.kind(coefficient),
            **{name: run.kind(name) for name in unit_constant},
        },
        associationFloor=CONFIG._ASSOCIATION_FLOOR,
    )
    report["observationUnit"] = observation_unit
    report["independentUnit"] = independent_unit
    run.actions.append(f"confounding:{coefficient}")
    return record, report


def _characterize_coefficients(
    run: _Run,
    coefficients: Sequence[str],
    *,
    unit_directions: Mapping[str, Any],
    design_columns: Sequence[str],
    technical: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for coefficient in coefficients:
        observation, independent = _resolve_units(
            run,
            coefficient,
            directed=unit_directions,
            design_columns=design_columns,
            unit_candidates=[*design_columns, *technical],
        )
        record, report = _characterize_coefficient(
            run,
            coefficient,
            observation_unit=observation,
            independent_unit=independent,
            technical=technical,
        )
        records.append(record)
        if report is not None:
            reports.append(report)
    return records, reports


def _technical_nesting_reports(
    store: Any,
    names: Sequence[str],
    *,
    cell_key: str,
) -> list[dict[str, Any]]:
    """Directional nesting among categorical technical columns without bulk fetch."""
    name_list = list(names)
    reports: list[dict[str, Any]] = []
    for index, left_name in enumerate(name_list):
        left_values = store.cells.fetch(left_name, key=cell_key)
        for right_name in name_list[index + 1 :]:
            right_values = store.cells.fetch(right_name, key=cell_key)
            mapping = directional_mapping(left_values, right_values)
            if mapping["nesting"] == "none":
                continue
            reports.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "nesting": mapping["nesting"],
                    "directionalMapping": mapping,
                }
            )
    return reports


def _column_records(
    run: _Run,
    candidates: Sequence[str],
    *,
    aliases: Mapping[str, list[str]],
    dropped: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    records = [
        {
            "name": name,
            "kind": run.kind(name),
            "domain": run.domains[name],
            "summary": run.summary(name),
            "aliases": list(aliases.get(name, [])),
        }
        for name in candidates
    ]
    records.extend(
        {
            "name": name,
            "kind": "continuous",
            "domain": "ignore",
            "summary": f"dropped before triage ({CONFIG._DROP_REASONS[reason]})",
            "aliases": [],
        }
        for name, reason in dropped
    )
    return records


def characterize_covariates(
    store: Any,
    *,
    studyContext: str | None = None,
    model: Any | None = None,
    cellKey: str = "I",
    directions: Mapping[str, Any] | None = None,
) -> CovariateCharacterization:
    """Label cell covariates and record design-level confounding."""
    direction_map = dict(directions or {})
    available = set(store.cells.columns)
    if cellKey not in available:
        return CovariateCharacterization(
            status="failed",
            notes=[f"cellKey {cellKey!r} is not present in cell metadata"],
        )
    errors = _validate_directions(direction_map, available)
    if errors:
        return CovariateCharacterization(status="failed", notes=errors)

    candidates, dropped = _triage_columns(
        store,
        cell_key=cellKey,
        exclude=set(direction_map.get("excludeColumns") or []),
    )
    reviewed = len(candidates) + len(dropped)
    candidates = [name for name in candidates if name in store.cells.columns]
    kind_directions = dict(direction_map.get("columnKinds") or {})
    directed_coefficients = set(direction_map.get("coefficientsOfInterest") or [])

    profiles: dict[str, _ColumnProfile] = {}
    n_rows = 0
    varying: list[str] = []
    for name in candidates:
        directed_kind = kind_directions.get(name)
        profile = _profile_column(
            store,
            name,
            cell_key=cellKey,
            kind=cast(ColumnKind, directed_kind)
            if directed_kind in CONFIG._KINDS
            else None,
        )
        profiles[name] = profile
        n_rows = profile.digest.nRows
        if profile.digest.nLevels <= 1:
            dropped.append((name, "dropConstant"))
            continue
        varying.append(name)
    candidates = varying
    candidates, aliases, alias_notes = _collapse_ontology_aliases(
        store,
        candidates,
        profiles,
        cell_key=cellKey,
    )

    run = _Run(
        store=store,
        cell_key=cellKey,
        n_rows=n_rows,
        context=_bounded_context(studyContext),
        model=model,
        profiles=profiles,
    )
    for name, reason in dropped:
        detail = f"Dropped {CONFIG._DROP_REASONS[reason]} {name}"
        if reason == "dropConstant" and name in directed_coefficients:
            detail = (
                f"{detail}; also listed in coefficientsOfInterest but has no variation"
            )
        run.note(
            kind=reason,
            detail=detail,
            column=name,
        )
    run.audit.extend(alias_notes)

    domain_directions = dict(direction_map.get("columnDomains") or {})
    for name in candidates:
        run.domains[name] = _assign_domain(run, name, domain_directions)
    candidates = _collapse_equivalent_columns(run, candidates, aliases)

    technical = [name for name in candidates if run.domains[name] == "technical"]
    design_columns = [name for name in candidates if run.domains[name] == "design"]
    records, reports = _characterize_coefficients(
        run,
        _select_coefficients(
            run,
            biological=[
                name for name in candidates if run.domains[name] == "biological"
            ],
            directed=set(direction_map.get("coefficientsOfInterest") or []),
        ),
        unit_directions=dict(direction_map.get("unitsOfInference") or {}),
        design_columns=design_columns,
        technical=technical,
    )

    categorical_technical = [
        name for name in technical if run.kind(name) == "categorical"
    ]
    return CovariateCharacterization(
        status="done",
        auditLog=run.audit,
        actions=run.actions,
        notes=[
            f"Reviewed {reviewed} columns; {len(candidates)} triaged after "
            "deterministic drops and ontology alias collapse"
        ],
        decisions=run.decisions,
        columns=_column_records(run, candidates, aliases=aliases, dropped=dropped),
        coefficients=records,
        technicalNesting=(
            _technical_nesting_reports(
                store,
                categorical_technical,
                cell_key=cellKey,
            )
            if len(categorical_technical) >= 2
            else []
        ),
        confounding=reports,
    )
