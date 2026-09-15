"""Characterize cell covariates and study-design confounding."""

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ...metadata.queries import (
    PartitionDigest,
    column_constant_within,
    column_partition_digest,
    columns_same_partition,
    reduce_observation_units,
)
from ...metadata.rows import (
    MetaDataRowBlock,
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import resolve_cell_aligned_artifact
from ...metrics.association import directional_mapping, report_confounding
from ...storage.artifacts import fingerprint_array
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices
from ..decisions.selection import DecisionValidationError, decide
from ..tools import artifact_reference
from ..types import Decision, EvidenceItem
from .contracts import CovariateCharacterization

__all__ = [
    "CovariateCharacterization",
    "characterize_covariates",
]

Domain = Literal["biological", "technical", "design", "ignore", "unknown"]
ColumnKind = Literal["categorical", "continuous"]

_CATEGORICAL_MAX_LEVELS = 50
_EMBEDDING_TOKENS = (
    "umap",
    "pca",
    "tsne",
    "scvi",
    "latent",
    "phate",
    "forceatlas",
    "diffmap",
    "diffusionmap",
    "diffusion",
)
_DOMAINS = frozenset({"biological", "technical", "design", "ignore", "unknown"})
_ANALYSED = frozenset({"biological", "technical", "design"})
_KINDS = frozenset({"categorical", "continuous"})
_RESERVED_COLUMNS = frozenset({"I", "ids", "names"})
_SHORT_EMBEDDING_PARTS = frozenset({"fa", "dm", "pc"})
_INDEXED_NAME = re.compile(r"(?P<stem>.+?)[-_]?(?P<index>\d+)")
_ONTOLOGY_SUFFIX = "_ontology_term_id"
_SAMPLE_LEVELS = 8
_ASSOCIATION_FLOOR = 0.1
_DROP_REASONS = {
    "dropAssayStat": "Scarf assay statistic column",
    "dropProvenance": "analysis-linked column",
    "dropEmbedding": "embedding-style column",
    "dropConstant": "single-level column",
}


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


@dataclass(frozen=True, slots=True)
class _ColumnProfile:
    kind: ColumnKind
    summary: str
    digest: PartitionDigest
    artifact: ArtifactRef | None = None
    levelCounts: tuple[dict[str, Any], ...] = ()
    levelCountsTruncated: bool = False


_MAX_LEVEL_COUNT_ITEMS = 32
_MAX_STRUCTURE_ITEMS = 32


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    missing = pd.isna(value)
    if isinstance(missing, bool | np.bool_) and bool(missing):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _bounded_level_counts(
    values: Sequence[Any] | np.ndarray | pd.Series,
    *,
    limit: int = _MAX_LEVEL_COUNT_ITEMS,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    series = pd.Series(values)
    counts = series.value_counts(dropna=False, sort=False)
    output = tuple(
        {
            "value": _json_scalar(value),
            "count": int(count),
        }
        for value, count in counts.iloc[:limit].items()
    )
    return output, len(counts) > limit


class _SelectionBoundCells:
    """Read metadata through one validated immutable cell selection."""

    __slots__ = (
        "_artifact_sources",
        "_artifact_values",
        "_indices",
        "_source",
        "_metadata_values",
        "_cache_values",
    )

    def __init__(
        self,
        root: Any,
        source: Any,
        selection: ArtifactRef,
        *,
        artifacts: Mapping[str, ArtifactRef] | None = None,
        cache_values: bool = False,
    ) -> None:
        self._source = source
        self._cache_values = cache_values
        self._metadata_values: dict[str, np.ndarray] = {}
        self._indices = read_stored_selection_indices(
            root,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if len(self._indices) == 0:
            raise ValueError("cellSelection selects no cells")
        self._artifact_sources = dict(artifacts or {})
        collisions = sorted(set(self._artifact_sources).intersection(source.columns))
        if collisions:
            raise ValueError(
                f"Artifact covariate names collide with metadata columns: {collisions}"
            )
        self._artifact_values: dict[str, np.ndarray] = {}
        for name, artifact in self._artifact_sources.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Artifact covariates require non-empty names")
            resolved = resolve_cell_aligned_artifact(
                root,
                artifact,
                cell_selection=selection,
            )
            self._artifact_values[name] = resolved.values

    @property
    def columns(self) -> list[str]:
        return [*self._source.columns, *self._artifact_sources]

    @property
    def artifact_columns(self) -> list[str]:
        return list(self._artifact_sources)

    def artifact_source(self, column: str) -> ArtifactRef | None:
        return self._artifact_sources.get(column)

    def fetch(self, column: str, key: str = "I") -> np.ndarray:
        if key != "I":
            raise ValueError("A bound metadata view accepts only its stored selection")
        if not self._cache_values:
            return self._read(column, self._indices)
        if column not in self._metadata_values:
            self._metadata_values[column] = self._read(column, self._indices)
        return self._metadata_values[column]

    def _read(self, column: str, indices: np.ndarray) -> np.ndarray:
        artifact_values = self._metadata_values.get(column)
        if artifact_values is None:
            artifact_values = self._artifact_values.get(column)
        if artifact_values is not None:
            positions = np.searchsorted(self._indices, indices)
            if np.any(positions >= len(self._indices)) or not np.array_equal(
                self._indices[positions], indices
            ):
                raise ValueError(
                    "Artifact covariate rows do not align with the stored selection"
                )
            return artifact_values[positions]
        values = read_metadata_rows_chunkwise(self._source, column, indices)
        missing = read_metadata_missing_rows_chunkwise(
            self._source,
            column,
            indices,
        )
        if missing is None or not np.any(missing):
            return values
        if values.dtype.kind == "f":
            output = values.astype(np.float64, copy=True)
            output[missing] = np.nan
            return output
        output = values.astype(object, copy=True)
        output[missing] = None
        return output

    def iter_row_blocks(
        self,
        *,
        cell_key: str = "I",
        columns: Sequence[str] | None = None,
        block_rows: int | None = None,
    ) -> Iterator[MetaDataRowBlock]:
        if cell_key != "I":
            raise ValueError("A bound metadata view accepts only its stored selection")
        requested = list(columns or ())
        unknown = [column for column in requested if column not in self.columns]
        if unknown:
            raise KeyError(f"Metadata columns were not found: {unknown!r}")
        rows_per_block = (
            int(self._source.default_block_rows("I"))
            if block_rows is None
            else int(block_rows)
        )
        if rows_per_block < 1:
            raise ValueError("block_rows must be at least one")
        for start in range(0, len(self._indices), rows_per_block):
            stop = min(start + rows_per_block, len(self._indices))
            indices = self._indices[start:stop]
            yield MetaDataRowBlock(
                start=int(indices[0]),
                stop=int(indices[-1]) + 1,
                active_global_indices=indices,
                values={column: self._read(column, indices) for column in requested},
            )


class _SelectionBoundStore:
    __slots__ = ("assay_names", "cells")

    def __init__(
        self,
        store: Any,
        selection: ArtifactRef,
        *,
        artifacts: Mapping[str, ArtifactRef] | None = None,
    ) -> None:
        self.cells = _SelectionBoundCells(
            store.zw,
            store.cells,
            selection,
            artifacts=artifacts,
            cache_values=True,
        )
        self.assay_names = store.assay_names


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
    match = _INDEXED_NAME.fullmatch(name)
    if match is None:
        return False
    parts = [part for part in re.split(r"[-_]+", match.group("stem").lower()) if part]
    compact = "".join(parts)
    if any(token in compact for token in _EMBEDDING_TOKENS):
        return True
    return any(part in _SHORT_EMBEDDING_PARTS for part in parts)


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
    limit = min(_CATEGORICAL_MAX_LEVELS, max(2, len(values) // 20))
    return "categorical" if int(np.unique(finite).size) <= limit else "continuous"


def _summarize(values: np.ndarray, kind: ColumnKind) -> str:
    series = pd.Series(values)
    missing = int(series.isna().sum())
    if kind == "categorical":
        levels = series.dropna().astype(str).value_counts()
        top = ", ".join(
            f"{level}={int(count)}"
            for level, count in levels.head(_SAMPLE_LEVELS).items()
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
    inventory: dict[str, Any] | None = None,
) -> _ColumnProfile:
    values = store.cells.fetch(name, key=cell_key)
    resolved_kind = kind or _infer_kind(values)
    artifact_source = getattr(store.cells, "artifact_source", lambda _name: None)(name)
    identity = None
    if inventory is not None:
        # Metadata are mutable even when the selected cell identities are frozen.
        value_identity = (
            hashlib.sha256(
                "\n".join(
                    repr((type(value).__name__, value)) for value in values.tolist()
                ).encode()
            ).hexdigest()
            if values.dtype.hasobject
            else fingerprint_array(values)
        )
        identity = (str(values.dtype), value_identity, resolved_kind, artifact_source)
        saved = inventory.get(name)
        if saved is not None and saved[0] == identity:
            return cast(_ColumnProfile, saved[1])
    summary = _summarize(values, resolved_kind)
    digest = column_partition_digest(store.cells, name, cell_key=cell_key)
    level_counts: tuple[dict[str, Any], ...] = ()
    level_counts_truncated = False
    if resolved_kind == "categorical":
        level_counts, level_counts_truncated = _bounded_level_counts(values)
    profile = _ColumnProfile(
        kind=resolved_kind,
        summary=summary,
        digest=digest,
        artifact=artifact_source,
        levelCounts=level_counts,
        levelCountsTruncated=level_counts_truncated,
    )
    if inventory is not None:
        inventory[name] = (identity, profile)
    return profile


def _triage_columns(
    store: Any,
    *,
    cell_key: str,
    exclude: set[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split cell columns into model candidates and deterministic drops."""
    assay_prefixes = tuple(f"{name}_" for name in store.assay_names)
    artifact_columns = set(getattr(store.cells, "artifact_columns", ()))
    candidates: list[str] = []
    dropped: list[tuple[str, str]] = []
    for name in store.cells.columns:
        if name in _RESERVED_COLUMNS or name == cell_key or name in exclude:
            continue
        if name in artifact_columns:
            candidates.append(name)
        elif name.startswith(assay_prefixes):
            dropped.append((name, "dropAssayStat"))
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
        if not name.endswith(_ONTOLOGY_SUFFIX):
            continue
        base = name[: -len(_ONTOLOGY_SUFFIX)]
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
        ("columnKinds", _KINDS),
        ("columnDomains", _DOMAINS),
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
        if run.kind(name) != "categorical" or run.domains[name] not in _ANALYSED:
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
    if selected not in _DOMAINS:
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


def _valid_value_mask(values: Sequence[Any] | np.ndarray | pd.Series) -> np.ndarray:
    missing = pd.isna(np.asarray(values, dtype=object))
    return np.asarray(~missing, dtype=bool)


def _ordered_group_values(values: pd.Series) -> list[str | int | float | bool]:
    raw = values.to_numpy(dtype=object, copy=False)
    valid = _valid_value_mask(raw)
    return [
        cast(str | int | float | bool, _json_scalar(value))
        for value in pd.unique(raw[valid])
    ]


def _group_counts(
    values: pd.Series,
) -> tuple[list[dict[str, Any]], bool]:
    raw = values.to_numpy(dtype=object, copy=False)
    valid = _valid_value_mask(raw)
    counts, truncated = _bounded_level_counts(raw[valid])
    return list(counts), truncated


def _imbalance_from_counts(counts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    positive = [int(item["count"]) for item in counts if int(item["count"]) > 0]
    if not positive:
        return {
            "minimum": 0,
            "maximum": 0,
            "maxToMinRatio": None,
            "balanced": False,
        }
    minimum = min(positive)
    maximum = max(positive)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "maxToMinRatio": float(maximum / minimum),
        "balanced": minimum == maximum,
    }


def _distinct_units_by_group(
    design: pd.DataFrame,
    *,
    coefficient: str,
    unit: str,
    group_order: Sequence[str | int | float | bool],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    coefficient_values = design[coefficient].to_numpy(dtype=object, copy=False)
    unit_values = design[unit].to_numpy(dtype=object, copy=False)
    for group in group_order:
        mask = coefficient_values == group
        valid_units = unit_values[mask & _valid_value_mask(unit_values)]
        output.append(
            {
                "group": group,
                "count": int(pd.Series(valid_units).nunique(dropna=True)),
            }
        )
    return output


def _design_missingness(
    design: pd.DataFrame,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rows = int(len(design))
    for column in dict.fromkeys(columns):
        values = design[column].to_numpy(dtype=object, copy=False)
        missing = int((~_valid_value_mask(values)).sum())
        output.append(
            {
                "column": column,
                "rows": rows,
                "missing": missing,
                "missingFraction": missing / rows if rows else 0.0,
            }
        )
    return output


def _categorical_structure(
    design: pd.DataFrame,
    left: str,
    right: str,
) -> dict[str, Any]:
    left_values = design[left].to_numpy(dtype=object, copy=False)
    right_values = design[right].to_numpy(dtype=object, copy=False)
    mapping = directional_mapping(left_values, right_values)
    valid = _valid_value_mask(left_values) & _valid_value_mask(right_values)
    fully_crossed = False
    if valid.any():
        table = pd.crosstab(left_values[valid], right_values[valid])
        fully_crossed = bool(table.size and np.all(table.to_numpy() > 0))
    nesting = str(mapping.get("nesting", "none"))
    relationship = (
        nesting
        if nesting != "none"
        else "crossed"
        if fully_crossed
        else "partiallyCrossed"
    )
    return {
        "left": left,
        "right": right,
        "relationship": relationship,
        "fullyCrossed": fully_crossed,
        "directionalMapping": mapping,
    }


def _paired_coverage(
    design: pd.DataFrame,
    *,
    coefficient: str,
    pair_by: str,
    group_order: Sequence[str | int | float | bool],
) -> dict[str, Any]:
    groups = design[coefficient].to_numpy(dtype=object, copy=False)
    pairs = design[pair_by].to_numpy(dtype=object, copy=False)
    valid = _valid_value_mask(groups) & _valid_value_mask(pairs)
    frame = pd.DataFrame({"group": groups[valid], "pair": pairs[valid]})
    required = set(group_order)
    complete = 0
    incomplete = 0
    duplicate_pair_groups = 0
    single_group_pairs = 0
    incomplete_examples: list[dict[str, Any]] = []
    for pair, pair_frame in frame.groupby("pair", sort=False, observed=False):
        counts = pair_frame["group"].value_counts(dropna=False)
        present = {_json_scalar(value) for value in counts.index}
        has_duplicate = bool((counts > 1).any())
        if len(present) == 1:
            single_group_pairs += 1
        if present == required and not has_duplicate:
            complete += 1
            continue
        incomplete += 1
        duplicate_pair_groups += int((counts > 1).sum())
        if len(incomplete_examples) < _MAX_LEVEL_COUNT_ITEMS:
            incomplete_examples.append(
                {
                    "pair": _json_scalar(pair),
                    "presentGroups": [
                        value for value in group_order if value in present
                    ],
                    "duplicateGroups": [
                        _json_scalar(value)
                        for value, count in counts.items()
                        if int(count) > 1
                    ],
                }
            )
    total_pairs = int(frame["pair"].nunique(dropna=True))
    return {
        "pairBy": pair_by,
        "requiredGroups": list(group_order),
        "pairs": total_pairs,
        "completePairs": complete,
        "incompletePairs": incomplete,
        "duplicatePairGroups": duplicate_pair_groups,
        "betweenIndependentUnits": (
            total_pairs >= 2 and single_group_pairs == total_pairs
        ),
        "design": (
            "paired"
            if total_pairs >= 2
            and complete == total_pairs
            and incomplete == 0
            and duplicate_pair_groups == 0
            else "betweenIndependentUnits"
            if total_pairs >= 2 and single_group_pairs == total_pairs
            else "mixedOrIncomplete"
        ),
        "incompleteExamples": incomplete_examples,
        "examplesTruncated": incomplete > len(incomplete_examples),
        "complete": (
            total_pairs >= 2
            and complete == total_pairs
            and incomplete == 0
            and duplicate_pair_groups == 0
        ),
    }


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
    design_columns: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record: dict[str, Any] = {
        "name": coefficient,
        "kind": run.kind(coefficient),
        "observationUnit": observation_unit,
        "independentUnit": independent_unit,
        "scope": "unresolvedUnit",
        "groupOrder": [],
        "groupCounts": [],
        "groupCountsTruncated": False,
        "groupImbalance": {},
        "unitLevelCounts": {},
        "replication": {},
        "missingness": [],
        "designStructures": [],
        "pairedCoverage": {},
        "estimability": {},
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
    design_constant = [
        name
        for name in design_columns
        if name not in {coefficient, observation_unit, independent_unit}
        and name in run.profiles
        and column_constant_within(
            run.store.cells,
            name,
            observation_unit,
            cell_key=run.cell_key,
        )
    ]
    columns = list(
        dict.fromkeys(
            [
                *group_cols,
                coefficient,
                *unit_constant,
                *design_constant,
            ]
        )
    )
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
    group_order = _ordered_group_values(design[coefficient])
    group_counts, group_counts_truncated = _group_counts(design[coefficient])
    record["groupOrder"] = group_order
    record["groupCounts"] = group_counts
    record["groupCountsTruncated"] = group_counts_truncated
    record["groupImbalance"] = _imbalance_from_counts(group_counts)

    unit_level_counts: dict[str, Any] = {
        "observationUnit": {
            "column": observation_unit,
            "levels": design_rows,
        },
        "coefficient": {
            "column": coefficient,
            "levels": len(group_order),
            "counts": group_counts,
            "truncated": group_counts_truncated,
        },
    }
    observation_by_group = [
        {
            "group": item["value"],
            "count": item["count"],
        }
        for item in group_counts
    ]
    independent_by_group: list[dict[str, Any]] = []
    if independent_unit is not None:
        independent_counts, independent_counts_truncated = _bounded_level_counts(
            design[independent_unit]
        )
        independent_by_group = _distinct_units_by_group(
            design,
            coefficient=coefficient,
            unit=independent_unit,
            group_order=group_order,
        )
        unit_level_counts["independentUnit"] = {
            "column": independent_unit,
            "levels": int(design[independent_unit].nunique(dropna=True)),
            "counts": list(independent_counts),
            "truncated": independent_counts_truncated,
            "byGroup": independent_by_group,
        }
    record["unitLevelCounts"] = unit_level_counts
    replication_counts = (
        independent_by_group if independent_unit is not None else observation_by_group
    )
    minimum_replication = min(
        (int(item["count"]) for item in replication_counts),
        default=0,
    )
    record["replication"] = {
        "unit": independent_unit or observation_unit,
        "observationUnitsByGroup": observation_by_group,
        "independentUnitsByGroup": independent_by_group,
        "minimumPerGroup": minimum_replication,
        "sufficient": (
            len(group_order) >= 2
            and not group_counts_truncated
            and minimum_replication >= 2
        ),
    }
    record["missingness"] = _design_missingness(design, columns)

    structural_columns = list(
        dict.fromkeys(
            [
                *([independent_unit] if independent_unit is not None else []),
                *unit_constant,
                *design_constant,
            ]
        )
    )
    structures: list[dict[str, Any]] = []
    if run.kind(coefficient) == "categorical":
        for name in structural_columns[:_MAX_STRUCTURE_ITEMS]:
            if run.kind(name) != "categorical":
                continue
            structures.append(_categorical_structure(design, coefficient, name))
    record["designStructures"] = structures
    if independent_unit is not None and len(group_order) >= 2:
        record["pairedCoverage"] = _paired_coverage(
            design,
            coefficient=coefficient,
            pair_by=independent_unit,
            group_order=group_order,
        )

    report = report_confounding(
        design,
        coefficient=coefficient,
        technicalColumns=unit_constant,
        columnKinds={
            coefficient: run.kind(coefficient),
            **{name: run.kind(name) for name in unit_constant},
        },
        associationFloor=_ASSOCIATION_FLOOR,
    )
    report["observationUnit"] = observation_unit
    report["independentUnit"] = independent_unit
    report["groupOrder"] = group_order
    report["groupCounts"] = group_counts
    report["groupCountsTruncated"] = group_counts_truncated
    report["groupImbalance"] = record["groupImbalance"]
    report["unitLevelCounts"] = unit_level_counts
    report["replication"] = record["replication"]
    report["missingness"] = record["missingness"]
    report["designStructures"] = structures
    report["pairedCoverage"] = record["pairedCoverage"]
    record["estimability"] = dict(report.get("estimability") or {})
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
            design_columns=design_columns,
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
    records: list[dict[str, Any]] = []
    for name in candidates:
        profile = run.profiles[name]
        record: dict[str, Any] = {
            "name": name,
            "kind": profile.kind,
            "domain": run.domains[name],
            "summary": profile.summary,
            "aliases": list(aliases.get(name, [])),
            "nRows": profile.digest.nRows,
            "nLevels": profile.digest.nLevels,
            "nMissing": profile.digest.nMissing,
            "missingFraction": (
                profile.digest.nMissing / profile.digest.nRows
                if profile.digest.nRows
                else 0.0
            ),
            "levelCounts": list(profile.levelCounts),
            "levelCountsTruncated": profile.levelCountsTruncated,
        }
        if profile.artifact is None:
            record.update(
                {
                    "sourceType": "metadataColumn",
                    "metadataColumn": name,
                }
            )
        else:
            record.update(
                {
                    "sourceType": "artifact",
                    "artifact": profile.artifact.to_dict(),
                }
            )
        records.append(record)
    for name, reason in dropped:
        dropped_profile = run.profiles.get(name)
        record = {
            "name": name,
            "kind": dropped_profile.kind
            if dropped_profile is not None
            else "continuous",
            "domain": "ignore",
            "summary": (
                f"excluded from design roles ({_DROP_REASONS[reason]})"
                + (
                    f"; {dropped_profile.summary}"
                    if dropped_profile is not None
                    else ""
                )
            ),
            "aliases": [],
            "nRows": (
                dropped_profile.digest.nRows if dropped_profile is not None else 0
            ),
            "nLevels": (
                dropped_profile.digest.nLevels if dropped_profile is not None else 0
            ),
            "nMissing": (
                dropped_profile.digest.nMissing if dropped_profile is not None else 0
            ),
            "missingFraction": (
                dropped_profile.digest.nMissing / dropped_profile.digest.nRows
                if dropped_profile is not None and dropped_profile.digest.nRows
                else 0.0
            ),
            "levelCounts": (
                list(dropped_profile.levelCounts) if dropped_profile is not None else []
            ),
            "levelCountsTruncated": (
                dropped_profile.levelCountsTruncated
                if dropped_profile is not None
                else False
            ),
        }
        if dropped_profile is not None and dropped_profile.artifact is not None:
            record.update(
                {
                    "sourceType": "artifact",
                    "artifact": dropped_profile.artifact.to_dict(),
                }
            )
        else:
            record.update(
                {
                    "sourceType": "metadataColumn",
                    "metadataColumn": name,
                }
            )
        records.append(record)
    return records


def characterize_covariates(
    store: Any,
    *,
    cellSelection: ArtifactRef,
    studyContext: str | None = None,
    model: Any | None = None,
    directions: Mapping[str, Any] | None = None,
    groupingArtifacts: Mapping[str, ArtifactRef] | None = None,
    inventory: dict[str, Any] | None = None,
) -> CovariateCharacterization:
    """Label cell covariates and record design-level confounding."""
    if (
        not isinstance(cellSelection, ArtifactRef)
        or cellSelection.kind != "cell_selection"
    ):
        raise TypeError("cellSelection must be a cell_selection ArtifactRef")
    grouping_artifacts = dict(groupingArtifacts or {})
    bound_store = _SelectionBoundStore(
        store,
        cellSelection,
        artifacts=grouping_artifacts,
    )
    if inventory is not None and inventory.get("cellSelection") != cellSelection:
        inventory.clear()
        inventory["cellSelection"] = cellSelection
    column_inventory = (
        inventory.setdefault("columns", {}) if inventory is not None else None
    )
    cell_key = "I"
    direction_map = dict(directions or {})
    available = set(bound_store.cells.columns)
    errors = _validate_directions(direction_map, available)
    if errors:
        return CovariateCharacterization(
            status="failed",
            cellSelection=artifact_reference(cellSelection),
            notes=errors,
        )

    candidates, dropped = _triage_columns(
        bound_store,
        cell_key=cell_key,
        exclude=set(direction_map.get("excludeColumns") or []),
    )
    reviewed = len(candidates) + len(dropped)
    candidates = [name for name in candidates if name in bound_store.cells.columns]
    kind_directions = dict(direction_map.get("columnKinds") or {})
    directed_coefficients = set(direction_map.get("coefficientsOfInterest") or [])

    profiles: dict[str, _ColumnProfile] = {}
    n_rows = 0
    varying: list[str] = []
    for name in candidates:
        directed_kind = kind_directions.get(name)
        profile = _profile_column(
            bound_store,
            name,
            cell_key=cell_key,
            kind=cast(ColumnKind, directed_kind) if directed_kind in _KINDS else None,
            inventory=column_inventory,
        )
        profiles[name] = profile
        n_rows = profile.digest.nRows
        if profile.digest.nLevels <= 1:
            dropped.append((name, "dropConstant"))
            continue
        varying.append(name)
    candidates = varying
    for name, reason in dropped:
        if reason == "dropAssayStat" and name not in profiles:
            profiles[name] = _profile_column(
                bound_store,
                name,
                cell_key=cell_key,
                kind="continuous",
                inventory=column_inventory,
            )
    candidates, aliases, alias_notes = _collapse_ontology_aliases(
        bound_store,
        candidates,
        profiles,
        cell_key=cell_key,
    )

    run = _Run(
        store=bound_store,
        cell_key=cell_key,
        n_rows=n_rows,
        context=(studyContext or "").strip(),
        model=model,
        profiles=profiles,
    )
    for name, reason in dropped:
        detail = f"Dropped {_DROP_REASONS[reason]} {name}"
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
    for name in grouping_artifacts:
        if domain_directions.get(name) != "ignore":
            domain_directions[name] = "design"
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
    column_records = _column_records(
        run,
        candidates,
        aliases=aliases,
        dropped=dropped,
    )
    return CovariateCharacterization(
        status="done",
        cellSelection=artifact_reference(cellSelection),
        auditLog=run.audit,
        actions=run.actions,
        notes=[
            f"Reviewed {reviewed} columns; {len(candidates)} triaged after "
            "deterministic drops and ontology alias collapse"
        ],
        decisions=run.decisions,
        columns=column_records,
        coefficients=records,
        technicalNesting=(
            _technical_nesting_reports(
                bound_store,
                categorical_technical,
                cell_key=cell_key,
            )
            if len(categorical_technical) >= 2
            else []
        ),
        confounding=reports,
        unitLevelCounts=[
            {
                "coefficient": record["name"],
                **dict(record["unitLevelCounts"]),
            }
            for record in records
            if record.get("unitLevelCounts")
        ],
        groupImbalance=[
            {
                "coefficient": record["name"],
                **dict(record["groupImbalance"]),
            }
            for record in records
            if record.get("groupImbalance")
        ],
        missingness=[
            {
                "column": record["name"],
                "rows": record.get("nRows", 0),
                "missing": record.get("nMissing", 0),
                "missingFraction": record.get("missingFraction", 0.0),
            }
            for record in column_records
        ],
        designStructures=[
            {
                "coefficient": record["name"],
                **structure,
            }
            for record in records
            for structure in record.get("designStructures", [])
        ],
        pairedCoverage=[
            {
                "coefficient": record["name"],
                **dict(record["pairedCoverage"]),
            }
            for record in records
            if record.get("pairedCoverage")
        ],
        coefficientEstimability=[
            {
                "coefficient": record["name"],
                **dict(record["estimability"]),
            }
            for record in records
            if record.get("estimability")
        ],
    )
