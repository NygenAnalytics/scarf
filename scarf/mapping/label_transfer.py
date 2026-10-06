"""Frozen reference labels and immutable query label transfers.

A label transfer gives each projected query cell the reference label with the
largest share of its neighbors' inverse-distance weight. The reference labels
are first frozen into the query datastore as a ``reference_labels`` artifact,
so a saved transfer never depends on a live reference column. The transfer is
a ``label_transfer`` artifact aligned to the projection's query cells. Its
``labels`` array records abstentions in the canonical linked missing mask,
next to the neighbor votes and the evidence that decided each cell.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..metadata.selection import (
    CELL_VALUE_NAMES,
    resolve_cell_aligned_artifact,
    valid_category_mask,
)
from ..storage.arrays import (
    MISSING_MASK_PREFIX,
    _decode_metadata_values,
    create_metadata_column,
    create_zarr_dataset,
    linked_missing_mask,
    text_value,
)
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    artifact_transaction,
    plan_artifact,
)
from ..storage.artifacts import (
    ArtifactStatus,
    ValueFingerprintBuilder,
    artifact_group,
    fingerprint_stored_arrays,
    inspect_artifact,
)
from ..storage.errors import ArtifactResolutionError
from ..storage.profiles import StorageProfile
from ..storage.refs import ArtifactRef, ExternalArtifactRef
from ..storage.selections import (
    read_stored_selection_indices,
    validate_cell_selection,
)
from ..storage.types import as_zarr_array
from ..utils.arrays import read_only_copy
from .confidence import _label_vote_block, _LabelVotes, distance_weights
from .models import LabelTransferResult
from .reference import MappingReference, contract_error

REFERENCE_LABELS_OPERATION = "freeze_reference_labels"
LABEL_TRANSFER_OPERATION = "transfer_labels"
LABEL_TRANSFER_ALGORITHM_VERSION = 1
LABEL_TRANSFER_RERUN_MESSAGE = (
    "Re-run run_label_transfer to create a new label transfer."
)

ASSIGNED = 0
UNINFORMATIVE_CELL = 1
NO_LABELED_NEIGHBORS = 2
TIED_VOTE = 3
BELOW_THRESHOLD = 4
BEYOND_MAX_DISTANCE = 5
ABSTENTION_REASONS: tuple[str, ...] = (
    "uninformative_cell",
    "no_labeled_neighbors",
    "tied_vote",
    "below_threshold",
    "beyond_max_distance",
)
"""Abstention reasons in precedence order; reason code ``n`` is entry ``n - 1``."""

_SOURCE_MESSAGE = (
    "reference_labels must be a reference cell-metadata column name or a "
    "cell-label ArtifactRef of the reference datastore"
)
_REFERENCE_LABEL_ARRAYS = ("categories", "codes")
_LABELS = "labels"
_LABELS_MISSING = f"{MISSING_MASK_PREFIX}{_LABELS}"
_EVIDENCE_ARRAYS = (
    "vote_fraction",
    "top_two_margin",
    "vote_entropy",
    "nearest_distance",
    "reference_distance_percentile",
)
_VOTE_ARRAYS = ("vote_class_codes", "vote_class_fractions")
_TRANSFER_ARRAYS = tuple(
    sorted(
        (
            "abstention_reason",
            "candidate_codes",
            "categories",
            _LABELS,
            _LABELS_MISSING,
            *_VOTE_ARRAYS,
            *_EVIDENCE_ARRAYS,
        )
    )
)
_TRANSFER_ATTRIBUTES = frozenset(
    {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "scarf_version",
        "complete",
        "array_digests",
        "payload_fingerprint",
    }
)
_TRANSFER_PARAMETERS = frozenset(
    {"threshold_fraction", "max_distance", "algorithm_version"}
)
_TRANSFER_INPUTS = frozenset({"projection", "reference_labels", "cell_selection"})


def validate_reference_label_source(source: Any) -> str | ArtifactRef:
    """Return ``source`` when it can name reference labels.

    Reference labels are a non-empty cell-metadata column name, or a
    cell-label ``ArtifactRef``, of the reference datastore.

    Raises:
        TypeError: If ``source`` is neither.
    """
    if isinstance(source, ArtifactRef) or (isinstance(source, str) and source):
        return source
    raise TypeError(_SOURCE_MESSAGE)


def read_reference_labels(
    reference: MappingReference,
    source: str | ArtifactRef,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one label per selected reference cell and mark the usable ones.

    ``source`` is a cell-metadata column of the reference datastore, or a
    cell-label artifact in it, such as ``cluster_labels``, ``cluster_cut``,
    or ``smart_label``. Byte strings are decoded. Missing, blank, and masked
    labels are not usable. Callers validate the reference binding first.
    """
    source = validate_reference_label_source(source)
    if isinstance(source, str):
        values, valid = reference._fetch_cell_labels(source)
    else:
        spec = CELL_VALUE_NAMES.get(source.kind)
        if spec is None or not spec.categorical:
            raise ValueError(
                "reference_labels artifacts must hold cell labels, such as "
                f"cluster_labels, cluster_cut, or smart_label, not {source.kind!r}"
            )
        resolved = resolve_cell_aligned_artifact(
            reference.datastore.zw,
            source,
            cell_selection=reference.cell_selection,
            value_name=spec.name,
            expected_kind=source.kind,
        )
        values = resolved.values
        valid = valid_category_mask(values, missing_mask=resolved.missing_mask)
    labels = np.asarray(values)
    usable = np.asarray(valid, dtype=bool)
    expected_shape = (reference.selected_cell_count,)
    if labels.shape != expected_shape or usable.shape != expected_shape:
        raise ValueError(
            "Reference labels must have one value per selected reference cell"
        )
    return _decode_metadata_values(labels), usable


def encode_reference_labels(
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference classes and one class code per reference cell.

    Classes keep the order in which they first occur. Signed and unsigned
    integer, floating, and boolean labels keep their value type; other labels
    are stored as text. Unusable labels get code -1.
    """
    labels = np.asarray(values)
    usable = np.asarray(valid, dtype=bool)
    if not usable.any():
        raise ValueError("Reference labels contain no usable label")
    codes = np.full(len(labels), -1, dtype=np.int64)
    usable_codes, classes = pd.factorize(labels[usable], sort=False)
    codes[usable] = usable_codes
    return _stored_categories(np.asarray(classes), labels.dtype), codes


def _stored_categories(classes: np.ndarray, source_dtype: np.dtype[Any]) -> np.ndarray:
    kind = np.dtype(source_dtype).kind
    if kind == "i":
        return np.asarray(classes, dtype=np.int64)
    if kind == "u":
        return np.asarray(classes, dtype=np.uint64)
    if kind == "f":
        return np.asarray(classes, dtype=np.float64)
    if kind == "b":
        return np.asarray(classes, dtype=bool)
    texts = [text_value(value) for value in classes.tolist()]
    if len(set(texts)) != len(texts):
        raise ValueError(
            "Reference labels contain distinct values with the same text, such "
            "as 1 and '1'; use one value type"
        )
    width = max([1, *(len(text) for text in texts)])
    return np.asarray(texts, dtype=f"<U{width}")


def reference_labels_fingerprint(categories: np.ndarray, codes: np.ndarray) -> str:
    """Fingerprint frozen reference classes and codes as they are stored."""
    builder = ValueFingerprintBuilder()
    builder.update_array("categories", categories)
    builder.update_array("codes", codes)
    return builder.hexdigest()


def external_reference_artifact(
    reference: MappingReference,
    ref: ArtifactRef,
) -> ExternalArtifactRef:
    """Identify an artifact of the reference datastore from a query datastore."""
    return ExternalArtifactRef(
        dataset_fingerprint=reference.dataset_fingerprint,
        ref=ref,
        anchor_assay=None
        if ref.assay == reference.assay_name
        else reference.assay_name,
    )


@dataclass(frozen=True, slots=True)
class ReferenceLabelsPlan:
    """Frozen reference labels planned in a query datastore, with their values."""

    artifact: PlannedArtifact
    categories: np.ndarray
    codes: np.ndarray

    @property
    def ref(self) -> ArtifactRef:
        return self.artifact.ref

    @property
    def reused(self) -> bool:
        return self.artifact.reused


def plan_reference_labels(
    root: zarr.Group,
    reference: MappingReference,
    source: str | ArtifactRef,
    *,
    invalidate_cache: bool = False,
) -> ReferenceLabelsPlan:
    """Plan the frozen copy of reference labels in a query datastore.

    The copy's identity holds the mapping reference, the label source, and a
    fingerprint of the labels, so unchanged labels reuse one copy and changed
    labels create another. Nothing is written.
    """
    values, valid = read_reference_labels(reference, source)
    categories, codes = encode_reference_labels(values, valid)
    fingerprint = reference_labels_fingerprint(categories, codes)
    parameters: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "mapping_reference": reference.external_ref,
        "labels_fingerprint": fingerprint,
    }
    if isinstance(source, str):
        parameters["source_column"] = source
    else:
        inputs["source_labels"] = external_reference_artifact(reference, source)

    def payload_matches(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            return (
                not set(group.group_keys())
                and set(group.array_keys()) == set(_REFERENCE_LABEL_ARRAYS)
                and fingerprint_stored_arrays(group, _REFERENCE_LABEL_ARRAYS)
                == fingerprint
            )
        except (KeyError, TypeError, ValueError):
            return False

    planned = plan_artifact(
        root,
        scope="datastore",
        kind="reference_labels",
        operation=REFERENCE_LABELS_OPERATION,
        parameters=parameters,
        inputs=inputs,
        execution_options={},
        invalidate_cache=invalidate_cache,
        required_arrays=(
            ArrayRequirement(
                "categories", shape=categories.shape, dtype=categories.dtype
            ),
            ArrayRequirement("codes", shape=codes.shape, dtype=codes.dtype),
        ),
        reuse_validator=payload_matches,
    )
    return ReferenceLabelsPlan(
        artifact=planned,
        categories=read_only_copy(categories),
        codes=read_only_copy(codes),
    )


def write_reference_labels(
    root: zarr.Group,
    plan: ReferenceLabelsPlan,
    *,
    profile: StorageProfile | None = None,
) -> ArtifactRef:
    """Write planned frozen reference labels unless a matching copy exists."""
    if plan.reused:
        return plan.ref
    with artifact_transaction(root, plan.artifact) as group:
        _write_vector(group, "categories", plan.categories, profile)
        _write_vector(group, "codes", plan.codes, profile)
    return plan.ref


def _write_vector(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
    profile: StorageProfile | None = None,
) -> None:
    chunk_rows = min(max(len(values), 1), 100_000)
    if values.dtype.kind == "U":
        create_metadata_column(
            group,
            name,
            data=values,
            dtype=values.dtype,
            chunkSize=chunk_rows,
            profile=profile,
        )
        return
    output = create_zarr_dataset(
        group,
        name,
        (chunk_rows,),
        values.dtype,
        values.shape,
        profile=profile,
    )
    output[...] = values


@dataclass(frozen=True, slots=True)
class ReferenceDistancePercentiles:
    """Percentiles of reference nearest-neighbor distances, for interpolation."""

    distances: np.ndarray
    percentiles: np.ndarray

    @classmethod
    def from_reference(
        cls, reference: MappingReference
    ) -> "ReferenceDistancePercentiles":
        values = reference.reference_distance_values
        quantiles = reference.reference_distance_quantiles
        distances = np.unique(values)
        # Repeated distances take the highest quantile at which they occur.
        last = np.searchsorted(values, distances, side="right") - 1
        return cls(distances=distances, percentiles=quantiles[last])

    def percentile_of(self, distances: np.ndarray) -> np.ndarray:
        """Return the reference percentile of each query distance."""
        return np.asarray(
            np.interp(
                distances,
                self.distances,
                self.percentiles,
                left=0.0,
                right=1.0,
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class LabelTransferBlock:
    """Decisions and evidence for one contiguous block of projected query cells.

    ``label_codes`` holds the transferred class code of each cell, or -1 where
    the cell abstained, and ``abstention_reason`` the reason code.
    ``candidate_codes`` holds the class that the vote favored before the
    threshold and distance rules, or -1 where the vote favored no single class.
    """

    label_codes: np.ndarray
    abstention_reason: np.ndarray
    candidate_codes: np.ndarray
    vote_class_codes: np.ndarray
    vote_class_fractions: np.ndarray
    vote_fraction: np.ndarray
    top_two_margin: np.ndarray
    vote_entropy: np.ndarray
    nearest_distance: np.ndarray
    reference_distance_percentile: np.ndarray


def transfer_label_block(
    neighbor_codes: np.ndarray,
    distances: np.ndarray,
    uninformative: np.ndarray,
    *,
    threshold_fraction: float,
    max_distance: float | None,
    distance_percentiles: ReferenceDistancePercentiles,
) -> LabelTransferBlock:
    """Vote and decide the labels of one block of projected query cells.

    ``neighbor_codes`` holds the reference class code of each neighbor, or -1
    for a neighbor without a usable label, and ``distances`` the matching
    distances, nearest first. An uninformative cell carries no query evidence,
    so it abstains and its vote metrics are NaN.
    """
    codes = np.asarray(neighbor_codes, dtype=np.int64)
    neighbor_distances = np.asarray(distances, dtype=np.float64)
    if codes.ndim != 2 or neighbor_distances.shape != codes.shape:
        raise ValueError("Neighbor codes and distances must be matching matrices")
    n_rows, n_neighbors = codes.shape
    informative = ~np.asarray(uninformative, dtype=bool)
    if informative.shape != (n_rows,):
        raise ValueError("Uninformative flags must have one value per query cell")
    reason = np.full(n_rows, UNINFORMATIVE_CELL, dtype=np.uint8)
    label_codes = np.full(n_rows, -1, dtype=np.int64)
    candidate_codes = np.full(n_rows, -1, dtype=np.int64)
    vote_class_codes = np.full((n_rows, n_neighbors), -1, dtype=np.int64)
    vote_class_fractions = np.zeros((n_rows, n_neighbors), dtype=np.float64)
    evidence = {name: np.full(n_rows, np.nan) for name in _EVIDENCE_ARRAYS}
    if informative.any():
        informative_distances = neighbor_distances[informative]
        votes = _label_vote_block(
            codes[informative],
            distance_weights(informative_distances),
        )
        nearest = informative_distances[:, 0]
        decided = _abstention_reasons(
            votes,
            nearest,
            threshold_fraction=threshold_fraction,
            max_distance=max_distance,
        )
        reason[informative] = decided
        label_codes[informative] = np.where(
            decided == ASSIGNED,
            votes.winner_codes,
            -1,
        )
        candidate_codes[informative] = np.where(
            votes.has_labeled_votes & ~votes.is_tied,
            votes.winner_codes,
            -1,
        )
        vote_class_codes[informative] = votes.class_codes
        vote_class_fractions[informative] = votes.fractions
        evidence["vote_fraction"][informative] = votes.vote_fraction
        evidence["top_two_margin"][informative] = votes.top_two_margin
        evidence["vote_entropy"][informative] = votes.vote_entropy
        evidence["nearest_distance"][informative] = nearest
        evidence["reference_distance_percentile"][informative] = (
            distance_percentiles.percentile_of(nearest)
        )
    return LabelTransferBlock(
        label_codes=label_codes,
        abstention_reason=reason,
        candidate_codes=candidate_codes,
        vote_class_codes=vote_class_codes,
        vote_class_fractions=vote_class_fractions,
        **evidence,
    )


def _abstention_reasons(
    votes: _LabelVotes,
    nearest_distance: np.ndarray,
    *,
    threshold_fraction: float,
    max_distance: float | None,
) -> np.ndarray:
    reason = np.full(len(nearest_distance), ASSIGNED, dtype=np.uint8)
    # Later assignments win, so a cell keeps its most basic reason.
    if max_distance is not None:
        reason[nearest_distance > max_distance] = BEYOND_MAX_DISTANCE
    reason[votes.vote_fraction < threshold_fraction] = BELOW_THRESHOLD
    reason[votes.is_tied] = TIED_VOTE
    reason[~votes.has_labeled_votes] = NO_LABELED_NEIGHBORS
    return reason


@dataclass(frozen=True, slots=True)
class LabelTransferPlan:
    """A planned label transfer and the reference classes its labels use."""

    artifact: PlannedArtifact
    n_cells: int
    n_neighbors: int
    categories: np.ndarray

    @property
    def ref(self) -> ArtifactRef:
        return self.artifact.ref

    @property
    def reused(self) -> bool:
        return self.artifact.reused


def plan_label_transfer(
    root: zarr.Group,
    *,
    projection: ArtifactRef,
    cell_selection: ArtifactRef,
    reference_labels: ArtifactRef,
    categories: np.ndarray,
    n_cells: int,
    n_neighbors: int,
    threshold_fraction: float,
    max_distance: float | None,
    invalidate_cache: bool = False,
) -> LabelTransferPlan:
    """Plan one label transfer over a projection's query cells.

    Nothing is written; a complete transfer with the same projection, frozen
    reference labels, and decision rule is reused.
    """
    if validate_cell_selection(root, cell_selection).selected_count != n_cells:
        raise ValueError("The projection's query cell selection has changed size")
    classes = read_only_copy(categories)

    def payload_matches(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            _validate_transfer_payload(
                group,
                n_cells=n_cells,
                n_neighbors=n_neighbors,
                categories=classes,
            )
        except (KeyError, TypeError, ValueError):
            return False
        return True

    planned = plan_artifact(
        root,
        scope="assay",
        assay=projection.assay,
        kind="label_transfer",
        operation=LABEL_TRANSFER_OPERATION,
        parameters={
            "threshold_fraction": threshold_fraction,
            "max_distance": max_distance,
            "algorithm_version": LABEL_TRANSFER_ALGORITHM_VERSION,
        },
        inputs={
            "projection": projection,
            "reference_labels": reference_labels,
            "cell_selection": cell_selection,
        },
        execution_options={},
        invalidate_cache=invalidate_cache,
        required_arrays=_transfer_array_requirements(
            n_cells,
            n_neighbors,
            n_classes=len(classes),
            class_dtype=classes.dtype,
        ),
        required_attributes=(
            AttributeRequirement("array_digests", expected_types=(dict,)),
            AttributeRequirement("payload_fingerprint", expected_types=(str,)),
        ),
        reuse_validator=payload_matches,
    )
    return LabelTransferPlan(
        artifact=planned,
        n_cells=n_cells,
        n_neighbors=n_neighbors,
        categories=classes,
    )


def _transfer_array_requirements(
    n_cells: int,
    n_neighbors: int,
    *,
    n_classes: int,
    class_dtype: np.dtype[Any],
) -> tuple[ArrayRequirement, ...]:
    return (
        ArrayRequirement("categories", shape=(n_classes,), dtype=class_dtype),
        ArrayRequirement(_LABELS, shape=(n_cells,), dtype=class_dtype),
        ArrayRequirement(_LABELS_MISSING, shape=(n_cells,), dtype=bool),
        ArrayRequirement("abstention_reason", shape=(n_cells,), dtype=np.uint8),
        ArrayRequirement("candidate_codes", shape=(n_cells,), dtype=np.int64),
        ArrayRequirement(
            "vote_class_codes", shape=(n_cells, n_neighbors), dtype=np.int64
        ),
        ArrayRequirement(
            "vote_class_fractions", shape=(n_cells, n_neighbors), dtype=np.float64
        ),
        *(
            ArrayRequirement(name, shape=(n_cells,), dtype=np.float64)
            for name in _EVIDENCE_ARRAYS
        ),
    )


def write_label_transfer(
    root: zarr.Group,
    plan: LabelTransferPlan,
    blocks: Iterable[tuple[int, LabelTransferBlock]],
    *,
    chunk_rows: int,
    profile: StorageProfile | None = None,
) -> ArtifactRef:
    """Write a planned label transfer from contiguous blocks and complete it.

    If writing fails before publication, the incomplete artifact is removed.
    """
    if plan.reused:
        raise ValueError("A reused label transfer is loaded, not written")
    rows_per_chunk = max(1, min(int(chunk_rows), plan.n_cells))
    categories = plan.categories
    with artifact_transaction(root, plan.artifact) as group:
        arrays = _create_transfer_arrays(group, plan, rows_per_chunk, profile)
        # Digest the blocks as they are written, so the payload is not read
        # back to fingerprint it.
        digests = _BlockDigests(arrays)
        digests.add("categories", 0, categories)
        next_row = 0
        for start, block in blocks:
            if start != next_row:
                raise RuntimeError(
                    f"Label transfer blocks must be contiguous; expected row "
                    f"{next_row}, received {start}"
                )
            stop = start + len(block.label_codes)
            if stop == start or stop > plan.n_cells:
                raise RuntimeError("Label transfer blocks must fit the query cells")
            abstained = block.label_codes < 0
            labels = np.zeros(len(abstained), dtype=categories.dtype)
            labels[~abstained] = categories[block.label_codes[~abstained]]
            rows = {
                _LABELS: labels,
                _LABELS_MISSING: abstained,
                "abstention_reason": block.abstention_reason,
                "candidate_codes": block.candidate_codes,
                "vote_class_codes": block.vote_class_codes,
                "vote_class_fractions": block.vote_class_fractions,
                **{name: getattr(block, name) for name in _EVIDENCE_ARRAYS},
            }
            for name, values in rows.items():
                arrays[name][start:stop] = values
                digests.add(name, start, values)
            next_row = stop
        if next_row != plan.n_cells:
            raise RuntimeError(
                "Label transfer did not cover every projected query cell"
            )
        array_digests = digests.finish()
        # Each array's digest is recorded, so a loader verifies exactly the
        # arrays it reads; the fingerprint binds the digests together.
        group.attrs["array_digests"] = array_digests
        group.attrs["payload_fingerprint"] = _payload_fingerprint(array_digests)
    return plan.ref


class _BlockDigests:
    """Digest each payload array from the contiguous blocks written to it."""

    def __init__(self, arrays: Mapping[str, zarr.Array]) -> None:
        self._builders: dict[str, ValueFingerprintBuilder] = {}
        for name, array in arrays.items():
            builder = ValueFingerprintBuilder()
            builder.begin_array(name, tuple(array.shape), np.dtype(array.dtype))
            self._builders[name] = builder

    def add(self, name: str, start: int, values: np.ndarray) -> None:
        block = np.asarray(values)
        offset = (start,) + (0,) * (block.ndim - 1)
        self._builders[name].update_array_block(name, offset, block)

    def finish(self) -> dict[str, str]:
        digests = {}
        for name, builder in self._builders.items():
            builder.end_array(name)
            digests[name] = builder.hexdigest()
        return digests


def _array_digest(name: str, values: np.ndarray) -> str:
    """Digest one payload array held in memory."""
    builder = ValueFingerprintBuilder()
    builder.update_array(name, values)
    return builder.hexdigest()


def _payload_fingerprint(array_digests: Mapping[str, str]) -> str:
    """Combine one digest per payload array into the payload fingerprint.

    Each array is digested on its own, so a writer can fingerprint the blocks
    it writes, and a loader can verify only the arrays it reads.
    """
    if set(array_digests) != set(_TRANSFER_ARRAYS):
        raise ValueError("Label transfer digests do not cover its payload arrays")
    builder = ValueFingerprintBuilder()
    for name in _TRANSFER_ARRAYS:
        builder.update_bytes(name, array_digests[name].encode())
    return builder.hexdigest()


def _recorded_array_digests(group: zarr.Group) -> dict[str, str]:
    """Return the recorded array digests once the payload fingerprint binds them."""
    recorded = group.attrs["array_digests"]
    if not isinstance(recorded, Mapping) or set(recorded) != set(_TRANSFER_ARRAYS):
        raise ValueError("Label transfer array digests are malformed")
    digests: dict[str, str] = {}
    for name, digest in recorded.items():
        if not isinstance(digest, str):
            raise ValueError("Label transfer array digests are malformed")
        digests[name] = digest
    fingerprint = group.attrs["payload_fingerprint"]
    if not isinstance(fingerprint, str) or fingerprint != _payload_fingerprint(digests):
        raise ValueError(
            "Label transfer payload fingerprint does not match its array digests"
        )
    return digests


def _require_array_digests(
    recorded: Mapping[str, str],
    measured: Mapping[str, str],
) -> None:
    changed = sorted(name for name in measured if measured[name] != recorded[name])
    if changed:
        raise ValueError(
            "Label transfer arrays differ from their recorded digests: "
            + ", ".join(changed)
        )


def _create_transfer_arrays(
    group: zarr.Group,
    plan: LabelTransferPlan,
    chunk_rows: int,
    profile: StorageProfile | None,
) -> dict[str, zarr.Array]:
    n_cells = plan.n_cells
    categories = plan.categories
    _write_vector(group, "categories", categories, profile)
    arrays: dict[str, zarr.Array] = {
        "categories": as_zarr_array(group["categories"], name="categories")
    }
    if categories.dtype.kind == "U":
        arrays[_LABELS] = create_metadata_column(
            group,
            _LABELS,
            dtype=categories.dtype,
            shape=n_cells,
            chunkSize=chunk_rows,
            profile=profile,
        )
    else:
        arrays[_LABELS] = create_zarr_dataset(
            group,
            _LABELS,
            (chunk_rows,),
            categories.dtype,
            (n_cells,),
            profile=profile,
        )
    arrays[_LABELS_MISSING] = create_metadata_column(
        group,
        _LABELS_MISSING,
        dtype=bool,
        shape=n_cells,
        chunkSize=chunk_rows,
        profile=profile,
    )
    arrays[_LABELS].attrs["missing_mask"] = _LABELS_MISSING
    arrays["abstention_reason"] = create_zarr_dataset(
        group,
        "abstention_reason",
        (chunk_rows,),
        np.uint8,
        (n_cells,),
        profile=profile,
    )
    arrays["candidate_codes"] = create_zarr_dataset(
        group,
        "candidate_codes",
        (chunk_rows,),
        np.int64,
        (n_cells,),
        profile=profile,
    )
    for name, dtype in (
        ("vote_class_codes", np.int64),
        ("vote_class_fractions", np.float64),
    ):
        arrays[name] = create_zarr_dataset(
            group,
            name,
            (chunk_rows, plan.n_neighbors),
            dtype,
            (n_cells, plan.n_neighbors),
            profile=profile,
        )
    for name in _EVIDENCE_ARRAYS:
        arrays[name] = create_zarr_dataset(
            group,
            name,
            (chunk_rows,),
            np.float64,
            (n_cells,),
            profile=profile,
        )
    return arrays


def _transfer_payload_arrays(
    group: zarr.Group,
    *,
    n_cells: int | None = None,
    n_neighbors: int | None = None,
) -> tuple[dict[str, zarr.Array], dict[str, str]]:
    """Open the payload arrays and their recorded digests without reading data."""
    if set(group.group_keys()):
        raise ValueError("Label transfer payload contains unexpected groups")
    if set(group.array_keys()) != set(_TRANSFER_ARRAYS):
        raise ValueError(
            "Label transfer arrays do not match the transfer_labels contract"
        )
    if set(group.attrs) != _TRANSFER_ATTRIBUTES:
        raise ValueError(
            "Label transfer attributes do not match the transfer_labels contract"
        )
    arrays = {name: as_zarr_array(group[name], name=name) for name in _TRANSFER_ARRAYS}
    stored_categories = arrays["categories"]
    labels = arrays[_LABELS]
    votes = arrays["vote_class_codes"]
    if stored_categories.ndim != 1 or stored_categories.shape[0] < 1:
        raise ValueError("Label transfer classes must be a non-empty vector")
    if labels.ndim != 1 or votes.ndim != 2 or votes.shape[0] != labels.shape[0]:
        raise ValueError("Label transfer arrays must have one row per query cell")
    rows = int(labels.shape[0])
    width = int(votes.shape[1])
    if n_cells is not None and rows != n_cells:
        raise ValueError("Label transfer rows do not match the projected query cells")
    if n_neighbors is not None and width != n_neighbors:
        raise ValueError("Label transfer votes do not match the projection neighbors")
    requirements = _transfer_array_requirements(
        rows,
        width,
        n_classes=int(stored_categories.shape[0]),
        class_dtype=np.dtype(stored_categories.dtype),
    )
    if not all(requirement.matches(group) for requirement in requirements):
        raise ValueError("Label transfer arrays have the wrong shape or value type")
    if linked_missing_mask(group, _LABELS, values=labels) is None:
        raise ValueError("Label transfer labels have no linked missing-label mask")
    if any(set(array.attrs) for name, array in arrays.items() if name != _LABELS):
        raise ValueError("Label transfer arrays carry unexpected attributes")
    return arrays, _recorded_array_digests(group)


def _validate_transfer_payload(
    group: zarr.Group,
    *,
    n_cells: int,
    n_neighbors: int,
    categories: np.ndarray,
) -> dict[str, zarr.Array]:
    """Validate the payload layout, classes, and every array in one read."""
    arrays, recorded = _transfer_payload_arrays(
        group,
        n_cells=n_cells,
        n_neighbors=n_neighbors,
    )
    _require_array_digests(
        recorded,
        {
            name: fingerprint_stored_arrays(group, (name,), arrays=arrays)
            for name in _TRANSFER_ARRAYS
        },
    )
    # Verified digests stand for the stored classes, which are not read again.
    if recorded["categories"] != _array_digest("categories", categories):
        raise ValueError("Label transfer classes differ from its reference labels")
    return arrays


def load_label_transfer(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    load_votes: bool = False,
) -> LabelTransferResult:
    """Load one complete label transfer after validating its contract.

    Loading reads only the query datastore, never the reference. The neighbor
    vote matrices are read only with ``load_votes``. Every array that is read
    is checked against the digest recorded when it was written.
    """
    if (
        not isinstance(ref, ArtifactRef)
        or ref.scope != "assay"
        or ref.assay is None
        or ref.kind != "label_transfer"
    ):
        raise ValueError(
            "transfer must identify an assay-scoped label_transfer artifact"
        )
    if not isinstance(load_votes, bool):
        raise TypeError("load_votes must be a boolean")
    try:
        return _load_label_transfer(root, ref, load_votes=load_votes)
    except ArtifactResolutionError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise contract_error(str(exc), LABEL_TRANSFER_RERUN_MESSAGE) from exc


def _load_label_transfer(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    load_votes: bool,
) -> LabelTransferResult:
    status = inspect_artifact(root, ref)
    if not status.exists or not status.complete:
        raise ValueError("Label transfer artifact is missing or incomplete")
    if status.operation != LABEL_TRANSFER_OPERATION:
        raise ValueError("Label transfer artifact has another operation")
    threshold_fraction, max_distance = _transfer_parameters(status.parameters or {})
    if set(status.inputs or {}) != _TRANSFER_INPUTS:
        raise ValueError(
            "Label transfer inputs do not match the transfer_labels contract"
        )
    projection = _input_ref(status, "projection", kind="projection", assay=ref.assay)
    reference_labels = _input_ref(status, "reference_labels", kind="reference_labels")
    cell_selection = _input_ref(status, "cell_selection", kind="cell_selection")
    group = artifact_group(root, ref)
    arrays, recorded = _transfer_payload_arrays(group)
    n_cells, n_neighbors = (int(size) for size in arrays["vote_class_codes"].shape)
    _validate_projection_input(root, projection, cell_selection, n_neighbors)
    source = _stored_reference_label_source(root, reference_labels)
    if validate_cell_selection(root, cell_selection).selected_count != n_cells:
        raise ValueError("Label transfer rows do not match its query cell selection")
    cell_idx = read_stored_selection_indices(
        root,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    cell_idx.setflags(write=False)

    # Read each needed array once and check it against its recorded digest.
    # The arrays are frozen in place, so the result keeps them without copies.
    values: dict[str, np.ndarray] = {}
    for name in _TRANSFER_ARRAYS:
        if name in _VOTE_ARRAYS and not load_votes:
            continue
        array = np.asarray(arrays[name][:])
        array.setflags(write=False)
        values[name] = array
    _require_array_digests(
        recorded,
        {name: _array_digest(name, array) for name, array in values.items()},
    )
    categories = values["categories"]
    reason = values["abstention_reason"]
    abstained = values[_LABELS_MISSING]
    candidates = values["candidate_codes"]
    if reason.size and int(reason.max()) > len(ABSTENTION_REASONS):
        raise ValueError("Label transfer has an unknown abstention reason")
    if not np.array_equal(abstained, reason != ASSIGNED):
        raise ValueError("Label transfer abstentions do not match their reasons")
    if candidates.size and (
        int(candidates.min()) < -1 or int(candidates.max()) >= len(categories)
    ):
        raise ValueError("Label transfer candidates name an unknown reference class")
    decisive = ~np.isin(reason, (UNINFORMATIVE_CELL, NO_LABELED_NEIGHBORS, TIED_VOTE))
    if not np.array_equal(decisive, candidates >= 0):
        raise ValueError("Label transfer candidates do not match their reasons")
    assigned = reason == ASSIGNED
    if not np.array_equal(values[_LABELS][assigned], categories[candidates[assigned]]):
        raise ValueError("Label transfer labels do not match their candidates")
    if load_votes:
        _check_votes(
            values["vote_class_codes"],
            values["vote_class_fractions"],
            candidates,
            decisive,
            n_classes=len(categories),
        )

    classes = np.asarray(categories.tolist(), dtype=object)
    labels = np.full(n_cells, None, dtype=object)
    labels[assigned] = classes[candidates[assigned]]
    candidate_labels = np.full(n_cells, None, dtype=object)
    candidate_labels[decisive] = classes[candidates[decisive]]
    reason_names = np.asarray((None, *ABSTENTION_REASONS), dtype=object)[reason]
    evidence = pd.DataFrame(
        {
            # Labels keep their value type, so these columns stay object-typed
            # rather than being inferred as text, and missing values are None.
            "label": pd.Series(labels, dtype=object),
            "candidateLabel": pd.Series(candidate_labels, dtype=object),
            "voteFraction": values["vote_fraction"],
            "topTwoMargin": values["top_two_margin"],
            "voteEntropy": values["vote_entropy"],
            "nearestDistance": values["nearest_distance"],
            "referenceDistancePercentile": values["reference_distance_percentile"],
            "abstained": abstained,
            "abstentionReason": pd.Series(reason_names, dtype=object),
        }
    )
    return LabelTransferResult(
        ref=ref,
        projection=projection,
        reference_labels=reference_labels,
        reference_label_source=source,
        cell_selection=cell_selection,
        cell_idx=cell_idx,
        threshold_fraction=threshold_fraction,
        max_distance=max_distance,
        categories=categories,
        evidence=evidence,
        vote_class_codes=values.get("vote_class_codes"),
        vote_class_fractions=values.get("vote_class_fractions"),
    )


def _check_votes(
    vote_class_codes: np.ndarray,
    vote_class_fractions: np.ndarray,
    candidates: np.ndarray,
    decisive: np.ndarray,
    *,
    n_classes: int,
) -> None:
    """Check that the saved votes name known classes and favor the candidates."""
    if vote_class_codes.size and (
        int(vote_class_codes.min()) < -1 or int(vote_class_codes.max()) >= n_classes
    ):
        raise ValueError("Label transfer votes name an unknown reference class")
    rows = np.arange(len(candidates))
    winners = vote_class_codes[rows, np.argmax(vote_class_fractions, axis=1)]
    if not np.array_equal(winners[decisive], candidates[decisive]):
        raise ValueError("Label transfer candidates do not match their votes")


def _transfer_parameters(parameters: Mapping[str, Any]) -> tuple[float, float | None]:
    if set(parameters) != _TRANSFER_PARAMETERS:
        raise ValueError(
            "Label transfer parameters do not match the transfer_labels contract"
        )
    if parameters["algorithm_version"] != LABEL_TRANSFER_ALGORITHM_VERSION:
        raise ValueError("Label transfer was made by another algorithm version")
    threshold = parameters["threshold_fraction"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, float)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("Label transfer threshold_fraction is malformed")
    max_distance = parameters["max_distance"]
    if max_distance is not None and (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, float)
        or not np.isfinite(max_distance)
        or max_distance < 0.0
    ):
        raise ValueError("Label transfer max_distance is malformed")
    return threshold, max_distance


def _input_ref(
    status: ArtifactStatus,
    name: str,
    *,
    kind: str,
    assay: str | None = None,
) -> ArtifactRef:
    ref = status.input_ref(name)
    scope = "datastore" if assay is None else "assay"
    if ref.kind != kind or ref.scope != scope or ref.assay != assay:
        raise ValueError(f"Label transfer input {name!r} has the wrong kind or scope")
    return ref


def _validate_projection_input(
    root: zarr.Group,
    projection: ArtifactRef,
    cell_selection: ArtifactRef,
    n_neighbors: int,
) -> None:
    status = inspect_artifact(root, projection)
    if not status.exists or not status.complete:
        raise ValueError("The label transfer's projection is missing or incomplete")
    if status.operation != "map_query":
        raise ValueError("The label transfer's projection has another operation")
    if (status.inputs or {}).get("cell_selection") != cell_selection.to_dict():
        raise ValueError("The label transfer and its projection use different cells")
    if (status.parameters or {}).get("save_k") != n_neighbors:
        raise ValueError("Label transfer votes do not match the projection neighbors")


def _stored_reference_label_source(
    root: zarr.Group,
    ref: ArtifactRef,
) -> str | ExternalArtifactRef:
    """Return the label source that a frozen reference_labels artifact records."""
    status = inspect_artifact(root, ref)
    if not status.exists or not status.complete:
        raise ValueError(
            "The label transfer's frozen reference labels are missing or incomplete"
        )
    if status.operation != REFERENCE_LABELS_OPERATION:
        raise ValueError("Frozen reference labels have another operation")
    parameters = status.parameters or {}
    inputs = status.inputs or {}
    frozen_inputs = {"mapping_reference", "labels_fingerprint"}
    if set(parameters) == {"source_column"} and set(inputs) == frozen_inputs:
        column = parameters["source_column"]
        if isinstance(column, str) and column:
            return column
    elif not parameters and set(inputs) == frozen_inputs | {"source_labels"}:
        raw_source = inputs["source_labels"]
        if isinstance(raw_source, Mapping):
            return ExternalArtifactRef.from_dict(raw_source)
    raise ValueError(
        "Frozen reference labels do not match the freeze_reference_labels contract"
    )
