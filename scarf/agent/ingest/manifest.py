"""Read-only H5AD inventory for decision-driven RNA workflows."""

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np

from ...readers._h5ad_inspect import (
    _as_text,
    _column_names,
    _matrix_candidates,
    _MatrixCandidate,
    _node_length,
    _select_matrix,
    inspect_h5ad,
)
from .._deps import AGENT_INSTALL_HINT
from ..types import AgentDataModel

try:
    from pydantic import Field
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


type AuthorLabelPolicy = Literal["holdout", "preservation"]
type ManifestStatus = Literal["supported", "needsInput", "abstained"]
type MatrixCountSemantics = Literal["integerLikeCounts", "nonIntegerValues"]

_AUTHOR_LABEL_EXACT = frozenset(
    {
        "annotation",
        "annotations",
        "author_annotation",
        "author_cell_type",
        "author_cell_type_ontology_term_id",
        "cell_type",
        "cell_type_ontology_term_id",
        "cluster",
        "clusters",
        "clustering",
        "leiden",
        "louvain",
    }
)
_AUTHOR_LABEL_FRAGMENTS = (
    "annotation",
    "cell_label",
    "cell_type",
    "celltype",
    "cluster",
    "leiden",
    "louvain",
)
_OBS_SCHEMA_FIELDS = frozenset(
    {
        "assay",
        "assay_ontology_term_id",
        "development_stage",
        "development_stage_ontology_term_id",
        "disease",
        "disease_ontology_term_id",
        "donor_id",
        "is_primary_data",
        "organism",
        "organism_ontology_term_id",
        "self_reported_ethnicity",
        "self_reported_ethnicity_ontology_term_id",
        "sex",
        "suspension_type",
        "tissue",
        "tissue_ontology_term_id",
    }
)
_VAR_SCHEMA_FIELDS = frozenset(
    {
        "feature_biotype",
        "feature_is_filtered",
        "feature_length",
        "feature_name",
        "feature_reference",
        "feature_type",
        "feature_types",
        "gene_id",
        "gene_ids",
        "gene_name",
        "gene_symbol",
    }
)
_ANALYSIS_KEY_FRAGMENTS = (
    "cluster",
    "leiden",
    "louvain",
    "marker",
    "neighbors",
    "pca",
    "rank_gene",
    "tsne",
    "umap",
)
_DEFAULT_CHUNK_VALUES = 8_192
_DEFAULT_MAX_COLUMNS = 256
_DEFAULT_MAX_DOMAIN_VALUES = 16
_DEFAULT_MAX_INVENTORY_ITEMS = 256
_DIGEST_CHUNK_BYTES = 8 * 1024 * 1024


class MatrixCandidateManifest(AgentDataModel):
    """One dimension-compatible matrix observed in the source file."""

    key: str
    encoding: Literal["csr", "csc", "dense"]
    nCells: int
    nFeatures: int
    integerLike: bool
    countSemantics: MatrixCountSemantics
    dimensionCompatible: bool
    featureMetadataKey: str | None = None
    selected: bool = False


class MetadataColumnSummary(AgentDataModel):
    """Bounded summary of one H5AD dataframe column."""

    name: str
    dtype: str
    storageKind: Literal["categorical", "boolean", "numeric", "string", "other"]
    valueCount: int
    missingCount: int
    missingFraction: float
    domainSize: int | None = None
    domainValues: list[str] = Field(default_factory=list)
    domainTruncated: bool = False
    valueCounts: dict[str, int] = Field(default_factory=dict)
    minimum: float | None = None
    maximum: float | None = None
    schemaField: bool = False


class MetadataTableSummary(AgentDataModel):
    """Column inventory for an H5AD dataframe-like node."""

    key: str
    rowCount: int
    columns: list[MetadataColumnSummary] = Field(default_factory=list)
    schemaFields: list[str] = Field(default_factory=list)
    identifierColumns: list[str] = Field(default_factory=list)
    omittedColumnCount: int = 0
    heldOutAuthorColumnCount: int = 0


class H5adInventory(AgentDataModel):
    """Bounded top-level inventory of reusable and author-produced assets."""

    layers: list[str] = Field(default_factory=list)
    obsm: list[str] = Field(default_factory=list)
    uns: list[str] = Field(default_factory=list)
    priorEmbeddings: list[str] = Field(default_factory=list)
    authorAnalysisArtifacts: list[str] = Field(default_factory=list)
    omittedLayerCount: int = 0
    omittedObsmCount: int = 0
    omittedUnsCount: int = 0
    heldOutUnsItemCount: int = 0


class PriorFilteringFacts(AgentDataModel):
    """Facts and limitations imposed by the published H5AD cell universe."""

    cellXGeneSchemaDetected: bool
    cellXGeneSchemaVersion: str | None = None
    cellXGeneSchemaReference: str | None = None
    cellSetStatus: Literal["publishedCellsOnly", "unknown"]
    rawCountsAvailable: bool
    originalDropletsAvailable: bool = False
    originalCellCallingSupported: bool = False
    ambientCorrectionSupported: bool = False
    featureFilteringFlagAvailable: bool = False
    filteredFeatureCount: int | None = None
    primaryDataFlagAvailable: bool = False
    nonPrimaryCellCount: int | None = None
    limitations: list[str] = Field(default_factory=list)


class DatasetManifestDecision(AgentDataModel):
    """Eligibility result for the count-dependent RNA decision workflow."""

    status: ManifestStatus
    reasonCode: Literal[
        "rawRnaCountsAvailable",
        "ambiguousCountMatrices",
        "normalizedOnly",
        "rnaModalityUnavailable",
    ]
    summary: str
    selectedMatrixKey: str | None = None
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)


class DatasetManifest(AgentDataModel):
    """Read-only evidence collected before materializing an H5AD source."""

    formatVersion: Literal[1] = 1
    sourcePath: str
    sourceUri: str | None = None
    sourceSha256: str
    sourceSizeBytes: int
    authorLabelPolicy: AuthorLabelPolicy
    labelBenchmarkEligible: bool
    nCells: int
    nFeatures: int
    selectedFeatureMetadataKey: str | None = None
    matrixCandidates: list[MatrixCandidateManifest] = Field(default_factory=list)
    obs: MetadataTableSummary
    var: MetadataTableSummary
    rawVar: MetadataTableSummary | None = None
    assayMetadata: MetadataColumnSummary | None = None
    suspensionMetadata: MetadataColumnSummary | None = None
    organismMetadata: MetadataColumnSummary | None = None
    declaredBatchColumns: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )
    inventory: H5adInventory
    priorFiltering: PriorFilteringFacts
    decision: DatasetManifestDecision


def _source_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(_DIGEST_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _is_author_label(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in _AUTHOR_LABEL_EXACT or any(
        fragment in normalized for fragment in _AUTHOR_LABEL_FRAGMENTS
    )


def is_author_label_column(name: str) -> bool:
    """Return whether a metadata column is quarantined as an author label."""
    return _is_author_label(name)


def _is_missing(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"f", "c"}:
        return cast(np.ndarray, ~np.isfinite(values))
    if values.dtype.kind in {"S", "U"}:
        return np.asarray([not _as_text(value) for value in values], dtype=bool)
    if values.dtype.kind != "O":
        return np.zeros(values.shape, dtype=bool)
    return np.asarray(
        [
            value is None
            or (isinstance(value, float | np.floating) and not np.isfinite(value))
            or (
                isinstance(value, str | bytes | np.str_ | np.bytes_)
                and not _as_text(value)
            )
            for value in values
        ],
        dtype=bool,
    )


def _value_text(value: Any) -> str:
    if isinstance(value, bool | np.bool_):
        return "true" if bool(value) else "false"
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, np.floating):
        return repr(float(value))
    return str(value)


def _storage_kind(
    dtype: np.dtype[Any], *, categorical: bool = False
) -> Literal["categorical", "boolean", "numeric", "string", "other"]:
    if categorical:
        return "categorical"
    if np.issubdtype(dtype, np.bool_):
        return "boolean"
    if np.issubdtype(dtype, np.number):
        return "numeric"
    if dtype.kind in {"O", "S", "U"}:
        return "string"
    return "other"


def _dataset_value_chunks(
    dataset: h5py.Dataset,
    *,
    field: str | None,
    row_count: int,
    chunk_values: int,
) -> Iterator[np.ndarray]:
    for start in range(0, row_count, chunk_values):
        stop = min(row_count, start + chunk_values)
        raw = np.asarray(dataset[start:stop])
        yield np.asarray(raw[field] if field is not None else raw).reshape(-1)


def _column_source(
    table: h5py.Group | h5py.Dataset,
    name: str,
    *,
    row_count: int,
    chunk_values: int,
) -> tuple[Iterator[np.ndarray], np.dtype[Any], h5py.Dataset | None]:
    if isinstance(table, h5py.Dataset):
        if table.dtype.names is None or name not in table.dtype.names:
            raise ValueError(f"Column {name!r} is unavailable in {table.name}")
        dtype = table.dtype.fields[name][0]
        return (
            _dataset_value_chunks(
                table,
                field=name,
                row_count=row_count,
                chunk_values=chunk_values,
            ),
            dtype,
            None,
        )

    node = table.get(name)
    if isinstance(node, h5py.Dataset):
        categories: h5py.Dataset | None = None
        for category_group_name in ("__categories", "categories"):
            category_group = table.get(category_group_name)
            if isinstance(category_group, h5py.Group):
                category_node = category_group.get(name)
                if isinstance(category_node, h5py.Dataset):
                    categories = category_node
                    break
        return (
            _dataset_value_chunks(
                node,
                field=None,
                row_count=row_count,
                chunk_values=chunk_values,
            ),
            node.dtype,
            categories,
        )
    if isinstance(node, h5py.Group):
        codes = node.get("codes")
        categories = node.get("categories")
        if isinstance(codes, h5py.Dataset) and isinstance(categories, h5py.Dataset):
            return (
                _dataset_value_chunks(
                    codes,
                    field=None,
                    row_count=row_count,
                    chunk_values=chunk_values,
                ),
                codes.dtype,
                categories,
            )
    raise ValueError(f"Unsupported H5AD column encoding for {table.name}/{name}")


def _categorical_summary(
    *,
    name: str,
    chunks: Iterator[np.ndarray],
    dtype: np.dtype[Any],
    categories: h5py.Dataset,
    row_count: int,
    max_domain_values: int,
    schema_field: bool,
) -> MetadataColumnSummary:
    category_count = int(categories.shape[0])
    shown_count = min(category_count, max_domain_values)
    shown = [_value_text(value) for value in np.asarray(categories[:shown_count])]
    counts = np.zeros(shown_count, dtype=np.int64)
    missing_count = 0
    for values in chunks:
        codes = np.asarray(values, dtype=np.int64)
        valid = (codes >= 0) & (codes < category_count)
        missing_count += int((~valid).sum())
        shown_codes = codes[valid & (codes < shown_count)]
        if shown_codes.size:
            counts += np.bincount(shown_codes, minlength=shown_count)
    value_counts = (
        {value: int(count) for value, count in zip(shown, counts, strict=True)}
        if category_count <= max_domain_values
        else {}
    )
    return MetadataColumnSummary(
        name=name,
        dtype=str(dtype),
        storageKind="categorical",
        valueCount=row_count,
        missingCount=missing_count,
        missingFraction=missing_count / row_count if row_count else 0.0,
        domainSize=category_count,
        domainValues=shown,
        domainTruncated=category_count > max_domain_values,
        valueCounts=value_counts,
        schemaField=schema_field,
    )


def _plain_summary(
    *,
    name: str,
    chunks: Iterator[np.ndarray],
    dtype: np.dtype[Any],
    row_count: int,
    max_domain_values: int,
    schema_field: bool,
) -> MetadataColumnSummary:
    missing_count = 0
    observed_counts: dict[str, int] = {}
    domain_truncated = False
    minimum: float | None = None
    maximum: float | None = None
    numeric = np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.bool_)
    for values in chunks:
        missing = _is_missing(values)
        missing_count += int(missing.sum())
        observed = values[~missing]
        if numeric and observed.size:
            observed_float = np.asarray(observed, dtype=np.float64)
            block_minimum = float(observed_float.min())
            block_maximum = float(observed_float.max())
            minimum = block_minimum if minimum is None else min(minimum, block_minimum)
            maximum = block_maximum if maximum is None else max(maximum, block_maximum)
        if domain_truncated:
            continue
        for value in observed:
            normalized = _value_text(value)
            observed_counts[normalized] = observed_counts.get(normalized, 0) + 1
            if len(observed_counts) > max_domain_values:
                observed_counts.clear()
                domain_truncated = True
                break
    domain_values = sorted(observed_counts)
    return MetadataColumnSummary(
        name=name,
        dtype=str(dtype),
        storageKind=_storage_kind(dtype),
        valueCount=row_count,
        missingCount=missing_count,
        missingFraction=missing_count / row_count if row_count else 0.0,
        domainSize=None if domain_truncated else len(domain_values),
        domainValues=domain_values,
        domainTruncated=domain_truncated,
        valueCounts={} if domain_truncated else observed_counts,
        minimum=minimum,
        maximum=maximum,
        schemaField=schema_field,
    )


def _summarize_column(
    table: h5py.Group | h5py.Dataset,
    name: str,
    *,
    row_count: int,
    chunk_values: int,
    max_domain_values: int,
    schema_fields: frozenset[str],
) -> MetadataColumnSummary:
    chunks, dtype, categories = _column_source(
        table,
        name,
        row_count=row_count,
        chunk_values=chunk_values,
    )
    if categories is not None:
        return _categorical_summary(
            name=name,
            chunks=chunks,
            dtype=dtype,
            categories=categories,
            row_count=row_count,
            max_domain_values=max_domain_values,
            schema_field=name.lower() in schema_fields,
        )
    return _plain_summary(
        name=name,
        chunks=chunks,
        dtype=dtype,
        row_count=row_count,
        max_domain_values=max_domain_values,
        schema_field=name.lower() in schema_fields,
    )


def _index_columns(table: h5py.Group | h5py.Dataset) -> set[str]:
    if isinstance(table, h5py.Dataset):
        return {"_index", "index"} & set(table.dtype.names or ())
    names = {"_index", "index"} & set(table.keys())
    index_attribute = table.attrs.get("_index")
    if index_attribute is not None:
        names.add(_as_text(index_attribute))
    return names


def _summarize_table(
    h5: h5py.File,
    key: str,
    *,
    author_label_policy: AuthorLabelPolicy,
    schema_fields: frozenset[str],
    chunk_values: int,
    max_columns: int,
    max_domain_values: int,
) -> MetadataTableSummary:
    node = h5.get(key)
    if not isinstance(node, h5py.Group | h5py.Dataset):
        return MetadataTableSummary(key=key, rowCount=0)
    row_count = _node_length(node) or 0
    names = sorted(_column_names(node))
    identifiers = _index_columns(node)
    held_out = {
        name
        for name in names
        if author_label_policy == "holdout" and _is_author_label(name)
    }
    inspectable = [
        name for name in names if name not in identifiers and name not in held_out
    ]
    inspectable.sort(key=lambda name: (name.lower() not in schema_fields, name))
    selected = inspectable[:max_columns]
    columns = [
        _summarize_column(
            node,
            name,
            row_count=row_count,
            chunk_values=chunk_values,
            max_domain_values=max_domain_values,
            schema_fields=schema_fields,
        )
        for name in selected
    ]
    return MetadataTableSummary(
        key=key,
        rowCount=row_count,
        columns=columns,
        schemaFields=sorted(summary.name for summary in columns if summary.schemaField),
        identifierColumns=sorted(identifiers),
        omittedColumnCount=max(0, len(inspectable) - len(selected)),
        heldOutAuthorColumnCount=len(held_out),
    )


def _inventory_keys(
    h5: h5py.File,
    key: str,
    *,
    max_items: int,
    hide_author_labels: bool = False,
) -> tuple[list[str], int, int]:
    node = h5.get(key)
    if not isinstance(node, h5py.Group):
        return [], 0, 0
    names = sorted(str(name) for name in node.keys())
    held_out = [name for name in names if hide_author_labels and _is_author_label(name)]
    visible = [name for name in names if name not in held_out]
    return visible[:max_items], max(0, len(visible) - max_items), len(held_out)


def _read_text_scalar(
    h5: h5py.File,
    paths: tuple[str, ...],
    *,
    max_length: int = 500,
) -> str | None:
    for path in paths:
        node = h5.get(path)
        if not isinstance(node, h5py.Dataset) or node.shape not in {(), (1,)}:
            continue
        value = node[()] if node.shape == () else node[0]
        return _as_text(value)[:max_length]
    return None


def _read_text_vector(
    h5: h5py.File,
    paths: tuple[str, ...],
    *,
    max_items: int = 32,
    max_length: int = 256,
) -> list[str]:
    for path in paths:
        node = h5.get(path)
        if not isinstance(node, h5py.Dataset):
            continue
        if node.shape == ():
            values = [node[()]]
        elif len(node.shape) == 1:
            if node.shape[0] > max_items:
                raise ValueError(f"{path} contains too many values")
            values = list(np.asarray(node[:]).reshape(-1))
        else:
            continue
        resolved = list(
            dict.fromkeys(
                text
                for value in values
                if (text := _as_text(value).strip()[:max_length])
            )
        )
        return resolved
    return []


def _column_by_name(
    *tables: MetadataTableSummary | None,
    names: tuple[str, ...],
) -> MetadataColumnSummary | None:
    wanted = {name.lower() for name in names}
    for table in tables:
        if table is None:
            continue
        for column in table.columns:
            if column.name.lower() in wanted:
                return column
    return None


def _count_value(
    column: MetadataColumnSummary | None,
    value: str,
) -> int | None:
    if column is None or not column.valueCounts:
        return None
    return column.valueCounts.get(value, 0)


def _matrix_manifests(
    h5: h5py.File,
    *,
    selected_key: str | None,
) -> tuple[list[MatrixCandidateManifest], list[_MatrixCandidate]]:
    manifests: list[MatrixCandidateManifest] = []
    compatible: list[_MatrixCandidate] = []
    for candidate in _matrix_candidates(h5):
        if candidate.encoding not in {"csr", "csc", "dense"}:
            raise ValueError(
                f"Unsupported matrix encoding for {candidate.key}: {candidate.encoding}"
            )
        encoding = cast(Literal["csr", "csc", "dense"], candidate.encoding)
        feature_key: str | None = None
        dimension_compatible = False
        try:
            _, feature_key = _select_matrix(h5, [candidate])
            dimension_compatible = True
            compatible.append(candidate)
        except ValueError:
            pass
        manifests.append(
            MatrixCandidateManifest(
                key=candidate.key,
                encoding=encoding,
                nCells=candidate.shape[0],
                nFeatures=candidate.shape[1],
                integerLike=candidate.integerLike,
                countSemantics=(
                    "integerLikeCounts" if candidate.integerLike else "nonIntegerValues"
                ),
                dimensionCompatible=dimension_compatible,
                featureMetadataKey=feature_key,
                selected=candidate.key == selected_key,
            )
        )
    return manifests, compatible


def _matrix_decision(
    compatible: list[_MatrixCandidate],
    *,
    matrix_key: str | None,
) -> tuple[DatasetManifestDecision, str | None]:
    by_key = {candidate.key: candidate for candidate in compatible}
    if matrix_key is not None:
        candidate = by_key.get(matrix_key)
        if candidate is None:
            available = ", ".join(sorted(by_key))
            raise ValueError(
                f"matrix_key {matrix_key!r} is not dimension-compatible. "
                f"Available: {available}"
            )
        if not candidate.integerLike:
            return (
                DatasetManifestDecision(
                    status="abstained",
                    reasonCode="normalizedOnly",
                    summary=(
                        f"Selected matrix {matrix_key} is not integer-like and "
                        "cannot authorize count-dependent RNA analysis"
                    ),
                    options=[matrix_key],
                    evidenceIds=[f"matrix:{matrix_key}:nonIntegerValues"],
                ),
                None,
            )
        return (
            DatasetManifestDecision(
                status="supported",
                reasonCode="rawRnaCountsAvailable",
                summary=f"Selected integer-like count candidate {matrix_key}",
                selectedMatrixKey=matrix_key,
                options=[matrix_key],
                evidenceIds=[f"matrix:{matrix_key}:integerLikeCounts"],
            ),
            matrix_key,
        )

    integer_keys = sorted(
        candidate.key for candidate in compatible if candidate.integerLike
    )
    if not integer_keys:
        non_integer_keys = sorted(candidate.key for candidate in compatible)
        return (
            DatasetManifestDecision(
                status="abstained",
                reasonCode="normalizedOnly",
                summary=(
                    "No dimension-compatible integer-like count matrix is "
                    "available in this H5AD"
                ),
                options=non_integer_keys,
                evidenceIds=[
                    f"matrix:{key}:nonIntegerValues" for key in non_integer_keys
                ],
            ),
            None,
        )
    if "raw/X" in integer_keys:
        conflicting_keys = [key for key in integer_keys if key not in {"raw/X", "X"}]
        if not conflicting_keys:
            return (
                DatasetManifestDecision(
                    status="supported",
                    reasonCode="rawRnaCountsAvailable",
                    summary=(
                        "Selected authoritative integer-like raw/X count matrix "
                        "over the visualization matrix X"
                    ),
                    selectedMatrixKey="raw/X",
                    options=["raw/X"],
                    evidenceIds=["matrix:raw/X:integerLikeCounts"],
                ),
                "raw/X",
            )
    if len(integer_keys) > 1:
        return (
            DatasetManifestDecision(
                status="needsInput",
                reasonCode="ambiguousCountMatrices",
                summary=(
                    "Multiple dimension-compatible integer-like matrices are "
                    "available; select the intended raw count matrix"
                ),
                options=integer_keys,
                evidenceIds=[f"matrix:{key}:integerLikeCounts" for key in integer_keys],
            ),
            None,
        )
    selected_key = integer_keys[0]
    return (
        DatasetManifestDecision(
            status="supported",
            reasonCode="rawRnaCountsAvailable",
            summary=f"One integer-like count candidate is available: {selected_key}",
            selectedMatrixKey=selected_key,
            options=[selected_key],
            evidenceIds=[f"matrix:{selected_key}:integerLikeCounts"],
        ),
        selected_key,
    )


def _inventory(
    h5: h5py.File,
    *,
    author_label_policy: AuthorLabelPolicy,
    max_items: int,
) -> H5adInventory:
    layers, omitted_layers, _ = _inventory_keys(h5, "layers", max_items=max_items)
    obsm, omitted_obsm, _ = _inventory_keys(h5, "obsm", max_items=max_items)
    uns, omitted_uns, held_out_uns = _inventory_keys(
        h5,
        "uns",
        max_items=max_items,
        hide_author_labels=author_label_policy == "holdout",
    )
    prior_embeddings = [
        f"obsm/{name}"
        for name in obsm
        if name.lower().startswith("x_")
        or any(fragment in name.lower() for fragment in ("pca", "tsne", "umap"))
    ]
    analysis_artifacts = [
        f"{group}/{name}"
        for group, names in (("obsm", obsm), ("uns", uns))
        for name in names
        if any(fragment in name.lower() for fragment in _ANALYSIS_KEY_FRAGMENTS)
    ]
    if held_out_uns:
        analysis_artifacts.append(f"uns/heldOutAuthorItems:{held_out_uns}")
    return H5adInventory(
        layers=layers,
        obsm=obsm,
        uns=uns,
        priorEmbeddings=prior_embeddings,
        authorAnalysisArtifacts=analysis_artifacts,
        omittedLayerCount=omitted_layers,
        omittedObsmCount=omitted_obsm,
        omittedUnsCount=omitted_uns,
        heldOutUnsItemCount=held_out_uns,
    )


def inspect_h5ad_manifest(
    path: str | Path,
    *,
    source_uri: str | None = None,
    author_label_policy: AuthorLabelPolicy = "holdout",
    matrix_key: str | None = None,
    chunk_values: int = _DEFAULT_CHUNK_VALUES,
    max_columns: int = _DEFAULT_MAX_COLUMNS,
    max_domain_values: int = _DEFAULT_MAX_DOMAIN_VALUES,
    max_inventory_items: int = _DEFAULT_MAX_INVENTORY_ITEMS,
) -> DatasetManifest:
    """Inspect an H5AD without converting or modifying it.

    Matrix values are sampled only through :func:`inspect_h5ad`; metadata is
    summarized in bounded chunks. Author cell-type and clustering columns are
    not read when ``author_label_policy`` is ``"holdout"``.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"H5AD source is not a file: {source}")
    if author_label_policy not in {"holdout", "preservation"}:
        raise ValueError(
            "author_label_policy must be either 'holdout' or 'preservation'"
        )
    if chunk_values < 1:
        raise ValueError("chunk_values must be at least 1")
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1")
    if max_domain_values < 1:
        raise ValueError("max_domain_values must be at least 1")
    if max_inventory_items < 1:
        raise ValueError("max_inventory_items must be at least 1")

    with h5py.File(source, mode="r") as h5:
        matrix_candidates, compatible = _matrix_manifests(h5, selected_key=None)
        if not compatible:
            raise ValueError("No matrix candidate matches the obs and var dimensions")
        decision, selected_key = _matrix_decision(compatible, matrix_key=matrix_key)
        if selected_key is not None:
            inspection = inspect_h5ad(str(source), matrix_key=selected_key)
        else:
            inspection = inspect_h5ad(str(source))
        matrix_candidates, _ = _matrix_manifests(h5, selected_key=selected_key)

        obs = _summarize_table(
            h5,
            "obs",
            author_label_policy=author_label_policy,
            schema_fields=_OBS_SCHEMA_FIELDS,
            chunk_values=chunk_values,
            max_columns=max_columns,
            max_domain_values=max_domain_values,
        )
        var = _summarize_table(
            h5,
            "var",
            author_label_policy="preservation",
            schema_fields=_VAR_SCHEMA_FIELDS,
            chunk_values=chunk_values,
            max_columns=max_columns,
            max_domain_values=max_domain_values,
        )
        raw_var = (
            _summarize_table(
                h5,
                "raw/var",
                author_label_policy="preservation",
                schema_fields=_VAR_SCHEMA_FIELDS,
                chunk_values=chunk_values,
                max_columns=max_columns,
                max_domain_values=max_domain_values,
            )
            if isinstance(h5.get("raw/var"), h5py.Group | h5py.Dataset)
            else None
        )
        inventory = _inventory(
            h5,
            author_label_policy=author_label_policy,
            max_items=max_inventory_items,
        )
        schema_version = _read_text_scalar(
            h5,
            ("uns/schema_version", "uns/cellxgene_schema_version"),
        )
        schema_reference = _read_text_scalar(
            h5,
            ("uns/schema_reference", "uns/cellxgene_schema_reference"),
        )
        declared_batch_columns = _read_text_vector(
            h5,
            ("uns/batch_condition",),
        )
        obs_node = h5.get("obs")
        obs_columns = (
            set(_column_names(obs_node))
            if isinstance(obs_node, h5py.Group | h5py.Dataset)
            else set()
        )
        unknown_batch_columns = sorted(set(declared_batch_columns) - obs_columns)
        if unknown_batch_columns:
            raise ValueError(
                "uns/batch_condition references unknown obs columns: "
                f"{unknown_batch_columns}"
            )

    selected_table = raw_var if inspection.featureAttrsKey == "raw/var" else var
    assay = _column_by_name(
        obs,
        names=("assay_ontology_term_id", "assay"),
    )
    suspension = _column_by_name(obs, names=("suspension_type",))
    organism = _column_by_name(
        obs,
        names=("organism_ontology_term_id", "organism"),
    )
    feature_filtered = _column_by_name(
        selected_table,
        var,
        raw_var,
        names=("feature_is_filtered",),
    )
    is_primary = _column_by_name(obs, names=("is_primary_data",))
    cxg_detected = schema_version is not None or (
        assay is not None
        and suspension is not None
        and organism is not None
        and feature_filtered is not None
    )
    limitations = [
        "The H5AD observation table contains only the published cell universe; "
        "discarded barcodes and empty droplets are unavailable.",
        "Original cell calling and empty-droplet ambient correction cannot be "
        "reconstructed from this file.",
    ]
    if decision.status == "abstained" and decision.reasonCode == "normalizedOnly":
        limitations.append(
            "Count-dependent quality control and feature selection require an "
            "unambiguous integer-like source matrix."
        )
    prior_filtering = PriorFilteringFacts(
        cellXGeneSchemaDetected=cxg_detected,
        cellXGeneSchemaVersion=schema_version,
        cellXGeneSchemaReference=schema_reference,
        cellSetStatus="publishedCellsOnly" if cxg_detected else "unknown",
        rawCountsAvailable=any(
            candidate.integerLike and candidate.dimensionCompatible
            for candidate in matrix_candidates
        ),
        featureFilteringFlagAvailable=feature_filtered is not None,
        filteredFeatureCount=_count_value(feature_filtered, "true"),
        primaryDataFlagAvailable=is_primary is not None,
        nonPrimaryCellCount=_count_value(is_primary, "false"),
        limitations=limitations,
    )

    if decision.status == "supported" and inspection.suggestedAssays:
        if "RNA" not in inspection.suggestedAssays:
            decision = DatasetManifestDecision(
                status="abstained",
                reasonCode="rnaModalityUnavailable",
                summary="The selected count matrix contains no RNA feature span",
                options=sorted(inspection.suggestedAssays),
                evidenceIds=[
                    f"assay:{name}:{count}"
                    for name, count in sorted(inspection.suggestedAssays.items())
                ],
            )
            selected_key = None
            for candidate in matrix_candidates:
                candidate.selected = False

    return DatasetManifest(
        sourcePath=str(source.resolve()),
        sourceUri=source_uri,
        sourceSha256=_source_sha256(source),
        sourceSizeBytes=source.stat().st_size,
        authorLabelPolicy=author_label_policy,
        labelBenchmarkEligible=author_label_policy == "holdout",
        nCells=inspection.nCells,
        nFeatures=inspection.nFeatures,
        selectedFeatureMetadataKey=(
            inspection.featureAttrsKey if selected_key is not None else None
        ),
        matrixCandidates=matrix_candidates,
        obs=obs,
        var=var,
        rawVar=raw_var,
        assayMetadata=assay,
        suspensionMetadata=suspension,
        organismMetadata=organism,
        declaredBatchColumns=declared_batch_columns,
        inventory=inventory,
        priorFiltering=prior_filtering,
        decision=decision,
    )


__all__ = [
    "AuthorLabelPolicy",
    "DatasetManifest",
    "DatasetManifestDecision",
    "H5adInventory",
    "ManifestStatus",
    "MatrixCandidateManifest",
    "MetadataColumnSummary",
    "MetadataTableSummary",
    "PriorFilteringFacts",
    "inspect_h5ad_manifest",
    "is_author_label_column",
]
