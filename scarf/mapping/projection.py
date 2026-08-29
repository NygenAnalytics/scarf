"""Persistence for query-owned mapping projection artifacts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zarr

from ..storage.arrays import create_zarr_dataset
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.artifacts import (
    ArtifactRef,
    ExternalArtifactRef,
    ValueFingerprintBuilder,
    artifact_group,
    canonical_bytes,
    fingerprint_array,
    fingerprint_stored_arrays,
    inspect_artifact,
)
from ..storage.errors import ArtifactResolutionError
from ..storage.feature_selection import resolve_feature_selection
from ..storage.geometry import array_geometry
from ..storage.partition import row_band
from ..storage.profiles import StorageProfile
from ..storage.selections import validate_stored_selection_integrity
from ..storage.types import as_zarr_array
from .models import MappingResult, _MappingResultAxes
from .reference import MappingReference

PROJECTION_RERUN_MESSAGE = "Re-run run_mapping to create a new query projection."
NO_QUERY_BATCH_FINGERPRINT = fingerprint_array(np.empty(0, dtype=np.int64))

_PARAMETER_NAMES = frozenset(
    {
        "save_k",
        "missing_feature_policy",
        "correction_method",
    }
)
_INPUT_NAMES = frozenset(
    {
        "cell_selection",
        "feature_selection",
        "selected_expression_fingerprint",
        "query_batch_fingerprint",
        "query_batch_count",
        "mapping_reference",
    }
)
_ARRAY_NAMES = frozenset({"indices", "distances", "uninformative"})
_DIAGNOSTIC_NAMES = frozenset(
    {
        "featureCoverage",
        "queryBatchCount",
        "algorithmVariant",
        "zeroNormCellCount",
        "queryScaledDispersion",
    }
)
_ATTRIBUTE_NAMES = frozenset(
    {
        "artifact_id",
        "kind",
        "provenance",
        "execution_options",
        "created_at_ns",
        "scarf_version",
        "complete",
        "diagnostics",
        "payload_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class ProjectionPlan:
    """A projection artifact plan that has not opened a writer."""

    artifact: PlannedArtifact = field(repr=False)
    n_cells: int
    save_k: int
    reference_cell_count: int
    query_batch_count: int
    feature_coverage: float
    algorithm_variant: str

    @property
    def ref(self) -> ArtifactRef:
        return self.artifact.ref

    @property
    def reused(self) -> bool:
        return self.artifact.reused


class ProjectionWriter:
    """Write one query projection in contiguous bounded row blocks."""

    def __init__(
        self,
        root: zarr.Group,
        plan: ProjectionPlan,
        *,
        chunk_rows: int,
        profile: StorageProfile | None = None,
    ) -> None:
        if not isinstance(plan, ProjectionPlan):
            raise TypeError("plan must be a ProjectionPlan")
        if plan.reused:
            raise ValueError("A reused projection plan must be loaded without a writer")
        resolved_chunk_rows = _positive_int(chunk_rows, "chunk_rows")
        self._root = root
        self._plan = plan
        self._next_row = 0
        self._uninformative_count = 0
        self._finished = False
        self._aborted = False
        self._group = start_artifact(root, plan.artifact)
        row_chunk = min(resolved_chunk_rows, plan.n_cells)
        try:
            self._indices = create_zarr_dataset(
                self._group,
                "indices",
                (row_chunk, plan.save_k),
                np.uint64,
                (plan.n_cells, plan.save_k),
                profile=profile,
            )
            self._distances = create_zarr_dataset(
                self._group,
                "distances",
                (row_chunk, plan.save_k),
                np.float64,
                (plan.n_cells, plan.save_k),
                profile=profile,
            )
            self._uninformative = create_zarr_dataset(
                self._group,
                "uninformative",
                (row_chunk,),
                bool,
                (plan.n_cells,),
                profile=profile,
            )
        except BaseException:
            self._group.attrs["complete"] = False
            self._aborted = True
            raise

    @property
    def ref(self) -> ArtifactRef:
        return self._plan.ref

    @property
    def next_row(self) -> int:
        return self._next_row

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def aborted(self) -> bool:
        return self._aborted

    def write_block(
        self,
        start: int,
        indices: np.ndarray,
        distances: np.ndarray,
        uninformative: np.ndarray,
    ) -> None:
        """Write the next contiguous row block."""
        self._require_open()
        try:
            if isinstance(start, bool) or not isinstance(start, int | np.integer):
                raise TypeError("Projection block start must be an integer")
            resolved_start = int(start)
            if resolved_start != self._next_row:
                raise ValueError(
                    "Projection blocks must be contiguous; "
                    f"expected {self._next_row}, received {resolved_start}"
                )
            index_values = np.asarray(indices)
            distance_values = np.asarray(distances)
            uninformative_values = np.asarray(uninformative)
            if index_values.dtype.kind != "u":
                raise TypeError("Projection indices must use an unsigned integer dtype")
            if distance_values.dtype.kind != "f":
                raise TypeError("Projection distances must use a floating dtype")
            if uninformative_values.dtype != np.dtype(bool):
                raise TypeError("Projection uninformative values must be boolean")
            if index_values.ndim != 2 or index_values.shape[1] != self._plan.save_k:
                raise ValueError(
                    "Projection index blocks must have shape "
                    f"(rows, {self._plan.save_k})"
                )
            if distance_values.shape != index_values.shape:
                raise ValueError(
                    "Projection distance blocks must match the index block shape"
                )
            if uninformative_values.shape != (index_values.shape[0],):
                raise ValueError(
                    "Projection uninformative blocks must have one value per row"
                )
            if index_values.shape[0] < 1:
                raise ValueError("Projection blocks cannot be empty")
            if not np.all(np.isfinite(distance_values)):
                raise ValueError("Projection distances must be finite")
            if np.any(distance_values < 0):
                raise ValueError("Projection distances must be non-negative")
            if np.any(index_values >= self._plan.reference_cell_count):
                raise ValueError(
                    "Projection indices must identify selected reference cells"
                )
            stop = resolved_start + index_values.shape[0]
            if stop > self._plan.n_cells:
                raise ValueError("Projection block exceeds the declared cell count")
            self._indices[resolved_start:stop] = index_values
            self._distances[resolved_start:stop] = distance_values
            self._uninformative[resolved_start:stop] = uninformative_values
            self._next_row = stop
            self._uninformative_count += int(np.count_nonzero(uninformative_values))
        except BaseException:
            self._group.attrs["complete"] = False
            self._aborted = True
            raise

    def finish(self, diagnostics: Mapping[str, Any]) -> ArtifactRef:
        """Validate and complete the projection artifact."""
        self._require_open()
        try:
            if self._next_row != self._plan.n_cells:
                raise ValueError(
                    "Projection rows are incomplete: "
                    f"wrote {self._next_row} of {self._plan.n_cells}"
                )
            validated = _validated_diagnostics(
                diagnostics,
                n_cells=self._plan.n_cells,
                uninformative_count=self._uninformative_count,
                expected_feature_coverage=self._plan.feature_coverage,
                expected_algorithm_variant=self._plan.algorithm_variant,
                expected_query_batch_count=self._plan.query_batch_count,
            )
            self._group.attrs["diagnostics"] = validated
            self._group.attrs["payload_fingerprint"] = _payload_fingerprint(
                self._group,
                validated,
            )
            finish_artifact(self._group, self._plan.artifact)
        except BaseException:
            self._group.attrs["complete"] = False
            self._aborted = True
            raise
        self._finished = True
        return self._plan.ref

    def abort(self) -> None:
        """Leave an unfinished projection explicitly incomplete."""
        if self._finished:
            raise RuntimeError("A completed projection artifact cannot be aborted")
        self._group.attrs["complete"] = False
        self._aborted = True

    def _require_open(self) -> None:
        if self._finished:
            raise RuntimeError("Projection writer is already finished")
        if self._aborted:
            raise RuntimeError("Projection writer is aborted")


def plan_projection(
    root: zarr.Group,
    *,
    query_assay: str,
    n_cells: int,
    save_k: int,
    missing_feature_policy: str,
    correction_method: str,
    cell_selection: ArtifactRef,
    feature_selection: ArtifactRef,
    selected_expression_fingerprint: str,
    query_batch_fingerprint: str,
    query_batch_count: int,
    mapping_reference: ExternalArtifactRef,
    reference: MappingReference,
    reference_cell_count: int,
    invalidate_cache: bool = False,
) -> ProjectionPlan:
    """Plan one immutable query-owned projection."""
    assay = _nonempty_string(query_assay, "query_assay")
    resolved_n_cells = _positive_int(n_cells, "n_cells")
    resolved_save_k = _positive_int(save_k, "save_k")
    resolved_reference_cell_count = _positive_int(
        reference_cell_count,
        "reference_cell_count",
    )
    policy = _nonempty_string(missing_feature_policy, "missing_feature_policy")
    if policy not in {"reference_mean", "zero", "error"}:
        raise ValueError(
            "missing_feature_policy must be 'reference_mean', 'zero', or 'error'"
        )
    correction = _nonempty_string(correction_method, "correction_method")
    if correction not in {"none", "symphony"}:
        raise ValueError("correction_method must be 'none' or 'symphony'")
    expression_fingerprint = _nonempty_string(
        selected_expression_fingerprint,
        "selected_expression_fingerprint",
    )
    batch_fingerprint = _nonempty_string(
        query_batch_fingerprint,
        "query_batch_fingerprint",
    )
    resolved_query_batch_count = _positive_int(
        query_batch_count,
        "query_batch_count",
    )
    if resolved_query_batch_count > resolved_n_cells:
        raise ValueError("query_batch_count cannot exceed n_cells")
    external = _validate_external_mapping_reference(mapping_reference)
    if not isinstance(reference, MappingReference):
        raise TypeError("reference must be a MappingReference")
    reference.validate_dataset_fingerprint()
    if external != reference.external_ref:
        raise ValueError("mapping_reference does not match reference")
    if resolved_reference_cell_count != reference.selected_cell_count:
        raise ValueError("reference_cell_count does not match reference")
    expected_correction = "symphony" if reference.method == "symphony" else "none"
    if correction != expected_correction:
        raise ValueError("correction_method does not match reference")
    validated_cells = _validate_cell_selection(
        root,
        cell_selection,
    )
    if validated_cells.selected_count != resolved_n_cells:
        raise ValueError("n_cells must equal the selected row count in cell_selection")
    feature_coverage = _validate_mapping_overlap_selection(
        root,
        assay,
        feature_selection,
        mapping_reference=external,
        reference_feature_ids=reference.feature_ids,
    )
    algorithm_variant = "symphony" if correction == "symphony" else "scaled_pca"

    def valid_projection(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            _validate_payload(
                group,
                expected_n_cells=resolved_n_cells,
                expected_save_k=resolved_save_k,
                reference_cell_count=resolved_reference_cell_count,
                expected_feature_coverage=feature_coverage,
                expected_algorithm_variant=algorithm_variant,
                expected_query_batch_count=resolved_query_batch_count,
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False
        return True

    planned = plan_artifact(
        root,
        scope="assay",
        assay=assay,
        kind="projection",
        operation="map_query",
        parameters={
            "save_k": resolved_save_k,
            "missing_feature_policy": policy,
            "correction_method": correction,
        },
        inputs={
            "cell_selection": cell_selection,
            "feature_selection": feature_selection,
            "selected_expression_fingerprint": expression_fingerprint,
            "query_batch_fingerprint": batch_fingerprint,
            "query_batch_count": resolved_query_batch_count,
            "mapping_reference": external,
        },
        execution_options={},
        invalidate_cache=invalidate_cache,
        required_arrays=(
            ArrayRequirement(
                "indices",
                shape=(resolved_n_cells, resolved_save_k),
                dtype_kind="u",
            ),
            ArrayRequirement(
                "distances",
                shape=(resolved_n_cells, resolved_save_k),
                dtype_kind="f",
            ),
            ArrayRequirement(
                "uninformative",
                shape=(resolved_n_cells,),
                dtype=bool,
            ),
        ),
        required_attributes=(
            AttributeRequirement("diagnostics", expected_types=(dict,)),
            AttributeRequirement("payload_fingerprint", expected_types=(str,)),
        ),
        reuse_validator=valid_projection,
    )
    return ProjectionPlan(
        artifact=planned,
        n_cells=resolved_n_cells,
        save_k=resolved_save_k,
        reference_cell_count=resolved_reference_cell_count,
        query_batch_count=resolved_query_batch_count,
        feature_coverage=feature_coverage,
        algorithm_variant=algorithm_variant,
    )


def load_projection(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    reference: MappingReference,
    load_arrays: bool = False,
) -> MappingResult:
    """Load a query projection after validating the complete contract."""
    if not isinstance(load_arrays, bool):
        raise TypeError("load_arrays must be a boolean")
    if not isinstance(reference, MappingReference):
        raise TypeError("reference must be a MappingReference")
    try:
        return _load_projection(
            root,
            ref,
            load_arrays=load_arrays,
            reference=reference,
        )
    except ArtifactResolutionError:
        raise
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise _contract_error(str(exc)) from None


def _load_projection(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    load_arrays: bool,
    reference: MappingReference,
) -> MappingResult:
    assay = _validate_projection_ref(ref)
    status = inspect_artifact(root, ref)
    if not status.exists or not status.complete:
        raise ValueError("Projection artifact is missing or incomplete")
    if status.operation != "map_query":
        raise ValueError("Projection artifact has an old operation")

    parameters = status.parameters or {}
    if set(parameters) != _PARAMETER_NAMES:
        raise ValueError("Projection parameters do not match the map_query contract")
    save_k = _positive_int(parameters["save_k"], "save_k")
    policy = _nonempty_string(
        parameters["missing_feature_policy"],
        "missing_feature_policy",
    )
    if policy not in {"reference_mean", "zero", "error"}:
        raise ValueError("Projection missing_feature_policy is unsupported")
    correction_method = _nonempty_string(
        parameters["correction_method"],
        "correction_method",
    )
    if correction_method not in {"none", "symphony"}:
        raise ValueError("Projection correction_method is unsupported")

    inputs = status.inputs or {}
    if set(inputs) != _INPUT_NAMES:
        raise ValueError("Projection inputs do not match the map_query contract")
    cell_selection = _local_ref_from_input(
        inputs,
        "cell_selection",
        kind="cell_selection",
        scope="datastore",
        assay=None,
    )
    feature_selection = _local_ref_from_input(
        inputs,
        "feature_selection",
        kind="feature_selection",
        scope="assay",
        assay=assay,
    )
    _nonempty_string(
        inputs["selected_expression_fingerprint"],
        "selected_expression_fingerprint",
    )
    _nonempty_string(
        inputs["query_batch_fingerprint"],
        "query_batch_fingerprint",
    )
    query_batch_count = _positive_int(
        inputs["query_batch_count"],
        "query_batch_count",
    )
    raw_external = inputs["mapping_reference"]
    if not isinstance(raw_external, Mapping):
        raise TypeError("Projection mapping_reference input is malformed")
    external = _validate_external_mapping_reference(
        ExternalArtifactRef.from_dict(raw_external)
    )
    reference.validate_dataset_fingerprint()
    provided_external = reference.external_ref
    if provided_external != external:
        raise ValueError(
            "Provided mapping reference does not match the projection input; "
            f"expected {external!r}, received {provided_external!r}"
        )
    expected_correction = "symphony" if reference.method == "symphony" else "none"
    if correction_method != expected_correction:
        raise ValueError(
            "Projection correction method does not match its mapping reference"
        )
    reference_cell_count = _positive_int(
        reference.selected_cell_count,
        "Mapping reference selected_cell_count",
    )
    validated_cells = _validate_cell_selection(
        root,
        cell_selection,
    )
    feature_coverage = _validate_mapping_overlap_selection(
        root,
        assay,
        feature_selection,
        mapping_reference=external,
        reference_feature_ids=reference.feature_ids,
    )
    algorithm_variant = "symphony" if correction_method == "symphony" else "scaled_pca"

    group = artifact_group(root, ref)
    n_cells, diagnostics = _validate_payload(
        group,
        expected_save_k=save_k,
        reference_cell_count=reference_cell_count,
        expected_feature_coverage=feature_coverage,
        expected_algorithm_variant=algorithm_variant,
        expected_query_batch_count=query_batch_count,
    )
    if validated_cells.selected_count != n_cells:
        raise ValueError("Projection rows do not match the stored query cell selection")

    indices = distances = uninformative = None
    if load_arrays:
        indices = np.array(
            as_zarr_array(group["indices"], name="indices")[:],
            copy=True,
        )
        distances = np.array(
            as_zarr_array(group["distances"], name="distances")[:],
            copy=True,
        )
        uninformative = np.array(
            as_zarr_array(group["uninformative"], name="uninformative")[:],
            copy=True,
        )
    result = MappingResult(
        ref=ref,
        n_cells=n_cells,
        correction_method=correction_method,
        diagnostics=diagnostics,
        indices=indices,
        distances=distances,
        uninformative=uninformative,
        reference=reference,
    )
    object.__setattr__(
        result,
        "_axes",
        _MappingResultAxes(
            cell_selection=cell_selection,
            feature_selection=feature_selection,
        ),
    )
    return result


def _validate_payload(
    group: zarr.Group,
    *,
    expected_n_cells: int | None = None,
    expected_save_k: int | None = None,
    reference_cell_count: int | None = None,
    expected_feature_coverage: float | None = None,
    expected_algorithm_variant: str | None = None,
    expected_query_batch_count: int | None = None,
) -> tuple[int, dict[str, float | int | str]]:
    if set(group.group_keys()):
        raise ValueError("Projection payload contains unexpected groups")
    arrays = set(group.array_keys())
    if arrays != _ARRAY_NAMES:
        missing = _ARRAY_NAMES - arrays
        extra = arrays - _ARRAY_NAMES
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise ValueError("Projection payload arrays are invalid: " + "; ".join(details))
    if set(group.attrs) != _ATTRIBUTE_NAMES:
        raise ValueError("Projection attributes do not match the map_query contract")
    _created_at_ns(group)

    indices = as_zarr_array(group["indices"], name="indices")
    distances = as_zarr_array(group["distances"], name="distances")
    uninformative = as_zarr_array(
        group["uninformative"],
        name="uninformative",
    )
    if any(set(array.attrs) for array in (indices, distances, uninformative)):
        raise ValueError("Projection array attributes do not match the contract")
    if indices.ndim != 2 or indices.shape[0] < 1 or indices.shape[1] < 1:
        raise ValueError("Projection indices must be a non-empty matrix")
    if np.dtype(indices.dtype).kind != "u":
        raise TypeError("Projection indices must use an unsigned integer dtype")
    if distances.shape != indices.shape or np.dtype(distances.dtype).kind != "f":
        raise TypeError(
            "Projection distances must be a floating matrix matching indices"
        )
    n_cells = int(indices.shape[0])
    save_k = int(indices.shape[1])
    if expected_n_cells is not None and n_cells != expected_n_cells:
        raise ValueError("Projection cell count does not match its plan")
    if expected_save_k is not None and save_k != expected_save_k:
        raise ValueError("Projection neighbor count does not match save_k")
    if uninformative.shape != (n_cells,) or np.dtype(uninformative.dtype) != np.dtype(
        bool
    ):
        raise TypeError("Projection uninformative must be a boolean row vector")

    block_rows = min(
        row_band(array_geometry(indices), unit="chunk", fallback=1),
        row_band(array_geometry(distances), unit="chunk", fallback=1),
        row_band(array_geometry(uninformative), unit="chunk", fallback=1),
    )
    uninformative_count = 0
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        if reference_cell_count is not None:
            index_block = np.asarray(indices[start:stop])
            if np.any(index_block >= reference_cell_count):
                raise ValueError(
                    "Projection indices contain a neighbor outside the selected "
                    "reference cell range"
                )
        distance_block = np.asarray(distances[start:stop])
        if not np.all(np.isfinite(distance_block)):
            raise ValueError("Projection distances must be finite")
        if np.any(distance_block < 0):
            raise ValueError("Projection distances must be non-negative")
        uninformative_count += int(
            np.count_nonzero(np.asarray(uninformative[start:stop], dtype=bool))
        )
    raw_diagnostics = group.attrs["diagnostics"]
    if not isinstance(raw_diagnostics, Mapping):
        raise TypeError("Projection diagnostics must be a mapping")
    diagnostics = _validated_diagnostics(
        raw_diagnostics,
        n_cells=n_cells,
        uninformative_count=uninformative_count,
        expected_feature_coverage=expected_feature_coverage,
        expected_algorithm_variant=expected_algorithm_variant,
        expected_query_batch_count=expected_query_batch_count,
    )
    stored_fingerprint = group.attrs["payload_fingerprint"]
    if (
        not isinstance(stored_fingerprint, str)
        or not stored_fingerprint
        or stored_fingerprint != _payload_fingerprint(group, diagnostics)
    ):
        raise ValueError("Projection payload fingerprint does not match stored output")
    return n_cells, diagnostics


def _payload_fingerprint(
    group: zarr.Group,
    diagnostics: Mapping[str, Any],
) -> str:
    builder = ValueFingerprintBuilder()
    builder.update_bytes(
        "arrays",
        fingerprint_stored_arrays(group, tuple(sorted(_ARRAY_NAMES))).encode(),
    )
    builder.update_bytes("diagnostics", canonical_bytes(diagnostics))
    return builder.hexdigest()


def _validated_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    n_cells: int,
    uninformative_count: int,
    expected_feature_coverage: float | None = None,
    expected_algorithm_variant: str | None = None,
    expected_query_batch_count: int | None = None,
) -> dict[str, float | int | str]:
    if set(diagnostics) != _DIAGNOSTIC_NAMES:
        missing = sorted(_DIAGNOSTIC_NAMES - set(diagnostics))
        detail = "Projection diagnostics must contain exactly " + ", ".join(
            sorted(_DIAGNOSTIC_NAMES)
        )
        if missing:
            detail += ", and is missing " + ", ".join(missing)
        raise ValueError(detail)
    feature_coverage = diagnostics["featureCoverage"]
    if (
        isinstance(feature_coverage, bool | np.bool_)
        or not isinstance(feature_coverage, float | np.floating)
        or not np.isfinite(feature_coverage)
        or not 0 < float(feature_coverage) <= 1
    ):
        raise ValueError("featureCoverage must be a finite float in (0, 1]")
    if expected_feature_coverage is not None and not np.isclose(
        float(feature_coverage),
        expected_feature_coverage,
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    ):
        raise ValueError("featureCoverage does not match the reference overlap")
    query_batch_count = _positive_int(
        diagnostics["queryBatchCount"],
        "queryBatchCount",
    )
    if query_batch_count > n_cells:
        raise ValueError("queryBatchCount cannot exceed the projection cell count")
    if (
        expected_query_batch_count is not None
        and query_batch_count != expected_query_batch_count
    ):
        raise ValueError("queryBatchCount does not match the query-batch input")
    algorithm_variant = _nonempty_string(
        diagnostics["algorithmVariant"],
        "algorithmVariant",
    )
    if (
        expected_algorithm_variant is not None
        and algorithm_variant != expected_algorithm_variant
    ):
        raise ValueError("algorithmVariant does not match the correction method")
    zero_norm_count = _nonnegative_int(
        diagnostics["zeroNormCellCount"],
        "zeroNormCellCount",
    )
    if zero_norm_count > n_cells:
        raise ValueError("zeroNormCellCount cannot exceed the projection cell count")
    if zero_norm_count != uninformative_count:
        raise ValueError(
            "zeroNormCellCount must equal the number of uninformative rows"
        )
    dispersion = diagnostics["queryScaledDispersion"]
    if (
        isinstance(dispersion, bool | np.bool_)
        or not isinstance(dispersion, float | np.floating)
        or not np.isfinite(dispersion)
        or float(dispersion) < 0
    ):
        raise ValueError("queryScaledDispersion must be a finite non-negative float")
    return {
        "featureCoverage": float(feature_coverage),
        "queryBatchCount": query_batch_count,
        "algorithmVariant": algorithm_variant,
        "zeroNormCellCount": zero_norm_count,
        "queryScaledDispersion": float(dispersion),
    }


def _validate_projection_ref(ref: ArtifactRef) -> str:
    if (
        not isinstance(ref, ArtifactRef)
        or ref.scope != "assay"
        or ref.assay is None
        or ref.kind != "projection"
    ):
        raise ValueError("Expected an assay-scoped projection ArtifactRef")
    return ref.assay


def _validate_external_mapping_reference(
    reference: ExternalArtifactRef,
) -> ExternalArtifactRef:
    if not isinstance(reference, ExternalArtifactRef):
        raise TypeError("mapping_reference must be an ExternalArtifactRef")
    if reference.ref.kind != "mapping_reference":
        raise ValueError("mapping_reference must identify a mapping_reference artifact")
    return reference


def _validate_mapping_overlap_selection(
    root: zarr.Group,
    assay: str,
    ref: ArtifactRef,
    *,
    mapping_reference: ExternalArtifactRef,
    reference_feature_ids: np.ndarray,
) -> float:
    """Validate the mapping-specific feature-selection lineage and values."""
    resolve_feature_selection(root, assay, ref)
    status = inspect_artifact(root, ref)
    if status.operation != "select_mapping_overlap":
        raise ValueError(
            "Projection feature selection was not produced by select_mapping_overlap"
        )
    raw_reference = (status.inputs or {}).get("mapping_reference")
    if not isinstance(raw_reference, Mapping):
        raise ValueError("Projection feature-selection reference is malformed")
    try:
        selection_reference = ExternalArtifactRef.from_dict(raw_reference)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Projection feature-selection reference is malformed") from exc
    if selection_reference != mapping_reference:
        raise ValueError(
            "Projection feature selection belongs to a different mapping reference"
        )
    raw_reference_ids = np.asarray(reference_feature_ids)
    if raw_reference_ids.ndim != 1 or raw_reference_ids.size == 0:
        raise ValueError("Mapping reference feature identifiers are malformed")
    reference_ids = raw_reference_ids.astype(str)
    reference_id_set = set(reference_ids.tolist())
    feature_ids = as_zarr_array(
        root[f"{assay}/featureData/ids"],
        name=f"{assay}/featureData/ids",
    )
    values = as_zarr_array(
        artifact_group(root, ref)["values"],
        name="feature_selection.values",
    )
    block_rows = min(
        row_band(array_geometry(feature_ids), unit="chunk", fallback=1),
        row_band(array_geometry(values), unit="chunk", fallback=1),
    )
    overlap_count = 0
    for start in range(0, int(feature_ids.shape[0]), block_rows):
        stop = min(start + block_rows, int(feature_ids.shape[0]))
        query_ids = np.asarray(feature_ids[start:stop]).astype(str)
        expected = np.fromiter(
            (identifier in reference_id_set for identifier in query_ids),
            dtype=bool,
            count=len(query_ids),
        )
        overlap_count += int(np.count_nonzero(expected))
        if not np.array_equal(
            np.asarray(values[start:stop], dtype=bool),
            expected,
        ):
            raise ValueError(
                "Projection feature selection does not match the reference overlap"
            )
    if overlap_count == 0:
        raise ValueError("Projection feature selection has no reference overlap")
    return float(overlap_count / len(reference_ids))


def _local_ref_from_input(
    inputs: Mapping[str, Any],
    name: str,
    *,
    kind: str,
    scope: str,
    assay: str | None,
) -> ArtifactRef:
    raw_ref = inputs[name]
    if not isinstance(raw_ref, Mapping):
        raise TypeError(f"Projection input {name!r} must be an ArtifactRef")
    expected_keys = {"type", "scope", "kind", "artifact_id"}
    if scope == "assay":
        expected_keys.add("assay")
    if set(raw_ref) != expected_keys:
        raise ValueError(f"Projection input {name!r} is malformed")
    ref = ArtifactRef.from_dict(raw_ref)
    if ref.kind != kind or ref.scope != scope or ref.assay != assay:
        raise ValueError(f"Projection input {name!r} has the wrong kind or scope")
    return ref


def _validate_local_selection(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    kind: str,
    scope: str,
    assay: str | None,
) -> None:
    if not isinstance(ref, ArtifactRef):
        raise TypeError(f"{kind} must be an ArtifactRef")
    if ref.kind != kind or ref.scope != scope or ref.assay != assay:
        raise ValueError(f"{kind} has the wrong artifact kind or scope")
    status = inspect_artifact(root, ref)
    if not status.exists or not status.complete:
        raise ValueError(f"{kind} must be a complete local artifact")
    values = as_zarr_array(
        artifact_group(root, ref)["values"],
        name=f"{kind}.values",
    )
    if values.ndim != 1 or np.dtype(values.dtype) != np.dtype(bool):
        raise TypeError(f"{kind} values must be a boolean row vector")


def _validate_cell_selection(root: zarr.Group, ref: ArtifactRef) -> Any:
    """Validate a query selection against its immutable payload and row axis."""
    _validate_local_selection(
        root,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
    )
    return validate_stored_selection_integrity(
        root,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _created_at_ns(group: zarr.Group) -> int:
    value = group.attrs.get("created_at_ns")
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or int(value) < 1
    ):
        raise ValueError("Projection created_at_ns must be a positive integer")
    return int(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    resolved = _nonnegative_int(value, name)
    if resolved < 1:
        raise ValueError(f"{name} must be positive")
    return resolved


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _contract_error(detail: str) -> ValueError:
    return ValueError(f"{detail}. {PROJECTION_RERUN_MESSAGE}")
