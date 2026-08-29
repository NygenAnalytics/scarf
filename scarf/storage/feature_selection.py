from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from .arrays import create_zarr_dataset
from .artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from .artifacts import (
    ArtifactRef,
    ArtifactStatus,
    artifact_group,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
)
from .errors import ArtifactResolutionError
from .geometry import array_geometry
from .partition import row_band
from .types import as_zarr_array, as_zarr_group
from .refs import ExternalArtifactRef
from .selections import (
    validate_run_metadata_snapshot,
    validate_stored_selection_integrity,
)


@dataclass(frozen=True, slots=True)
class _ValidatedFeatureSelection:
    ref: ArtifactRef
    values: zarr.Array
    operation: str


def _ordered_feature_ids_fingerprint(assay: Any) -> str:
    feature_data = as_zarr_group(assay.z["featureData"], name="featureData")
    return fingerprint_stored_strings(
        as_zarr_array(feature_data["ids"], name="featureData/ids")
    )


def _feature_selection_plan(
    root: zarr.Group,
    *,
    assay: str,
    n_features: int,
    ordered_feature_ids_fingerprint: str,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    execution_options: dict[str, Any],
    payload_names: tuple[str, ...] = ("values",),
    expected_payload_fingerprint: str | None = None,
    invalidate_cache: bool = False,
) -> PlannedArtifact:
    requirements = tuple(
        ArrayRequirement(
            name,
            shape=(n_features,),
            dtype=(bool if name == "values" else np.float64),
        )
        for name in payload_names
    )
    attributes = (
        AttributeRequirement(
            "ordered_feature_ids_fingerprint",
            expected_types=(str,),
            predicate=lambda value: value == ordered_feature_ids_fingerprint,
        ),
        AttributeRequirement("payload_fingerprint", expected_types=(str,)),
    )

    def reuse_validator(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            if set(group.array_keys()) != set(payload_names):
                return False
            payload_fingerprint = fingerprint_stored_arrays(group, payload_names)
            return (
                group.attrs.get("ordered_feature_ids_fingerprint")
                == ordered_feature_ids_fingerprint
                and group.attrs.get("payload_fingerprint") == payload_fingerprint
                and (
                    expected_payload_fingerprint is None
                    or payload_fingerprint == expected_payload_fingerprint
                )
            )
        except (KeyError, TypeError, ValueError):
            return False

    return plan_artifact(
        root,
        scope="assay",
        assay=assay,
        kind="feature_selection",
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        execution_options=execution_options,
        invalidate_cache=invalidate_cache,
        required_arrays=requirements,
        required_attributes=attributes,
        reuse_validator=reuse_validator,
    )


def _write_feature_selection(
    root: zarr.Group,
    planned: PlannedArtifact,
    *,
    ordered_feature_ids_fingerprint: str,
    payload: dict[str, np.ndarray],
    payload_names: tuple[str, ...] = ("values",),
) -> None:
    if planned.reused:
        return
    group = start_artifact(root, planned)
    n_features = int(np.asarray(payload["values"]).shape[0])
    chunks = (min(max(n_features, 1), 100_000),)
    for name in payload_names:
        values = np.asarray(
            payload[name],
            dtype=(bool if name == "values" else np.float64),
        )
        if values.shape != (n_features,):
            raise ValueError(
                f"Feature-selection array {name!r} has shape {values.shape}; "
                f"expected ({n_features},)"
            )
        output = create_zarr_dataset(
            group,
            name,
            chunks,
            values.dtype,
            values.shape,
        )
        output[:] = values
    group.attrs["ordered_feature_ids_fingerprint"] = ordered_feature_ids_fingerprint
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        group,
        payload_names,
    )
    finish_artifact(group, planned)


def _feature_selection_values(
    root: zarr.Group,
    ref: ArtifactRef,
    name: str = "values",
) -> np.ndarray:
    group = artifact_group(root, ref)
    return np.asarray(as_zarr_array(group[name], name=name)[:])


def _ref_context(ref: ArtifactRef, *, assay: str) -> dict[str, str | None]:
    return {
        "assay": assay,
        "scope": ref.scope,
        "actual_assay": ref.assay,
        "kind": ref.kind,
        "artifact_id": ref.artifact_id,
    }


def _feature_data(root: zarr.Group, assay: str) -> zarr.Group:
    path = f"{assay}/featureData"
    if path not in root:
        raise ArtifactResolutionError(
            f"Assay {assay!r} has no feature metadata table",
            code="wrong_assay",
            context={"assay": assay},
        )
    try:
        feature_data = as_zarr_group(root[path], name=path)
    except TypeError as exc:
        raise ArtifactResolutionError(
            f"Assay {assay!r} has an invalid feature metadata table",
            code="corrupt_payload",
            context={"assay": assay},
        ) from exc
    if "ids" not in feature_data:
        raise ArtifactResolutionError(
            f"Assay {assay!r} has no feature row identifiers",
            code="row_mismatch",
            context={"assay": assay},
        )
    return feature_data


def _validate_ref_scope(ref: ArtifactRef, assay: str) -> None:
    context = _ref_context(ref, assay=assay)
    if ref.kind != "feature_selection":
        raise ArtifactResolutionError(
            "Expected a feature_selection artifact",
            code="wrong_kind",
            context={**context, "expected_kind": "feature_selection"},
        )
    if ref.scope != "assay":
        raise ArtifactResolutionError(
            "Feature selections must be assay-scoped",
            code="wrong_scope",
            context={**context, "expected_scope": "assay"},
        )
    if ref.assay != assay:
        raise ArtifactResolutionError(
            f"Feature selection belongs to assay {ref.assay!r}, not {assay!r}",
            code="wrong_assay",
            context=context,
        )


def _payload_names(group: zarr.Group) -> tuple[str, ...]:
    names = tuple(group.array_keys())
    allowed = {"values", "corrected_variance"}
    unexpected = set(names) - allowed
    if unexpected:
        raise ArtifactResolutionError(
            "Feature selection contains unexpected payload arrays",
            code="corrupt_payload",
            context={"arrays": ",".join(sorted(unexpected))},
        )
    return (
        ("values", "corrected_variance")
        if "corrected_variance" in names
        else ("values",)
    )


_FEATURE_SELECTION_CONTRACTS = {
    "create_all_features": (
        frozenset(),
        frozenset({"dataset_fingerprint", "ordered_feature_ids_fingerprint"}),
        ("values",),
    ),
    "set_feature_selection": (
        frozenset({"all_features"}),
        frozenset({"values_fingerprint"}),
        ("values",),
    ),
    "select_detected_features": (
        frozenset({"feature_summary"}),
        frozenset({"min_cells"}),
        ("values",),
    ),
    "select_hvgs": (
        frozenset({"feature_snapshot", "feature_summary"}),
        frozenset(
            {
                "min_cells",
                "max_cells",
                "top_n",
                "min_var",
                "max_var",
                "min_mean",
                "max_mean",
                "n_bins",
                "lowess_frac",
                "blacklist",
                "keep_bounds",
                "bin_strategy",
            }
        ),
        ("values", "corrected_variance"),
    ),
    "select_prevalent_peaks": (
        frozenset({"feature_summary"}),
        frozenset({"top_n"}),
        ("values",),
    ),
    "select_mapping_overlap": (
        frozenset({"mapping_reference", "all_features"}),
        frozenset(),
        ("values",),
    ),
}


def _local_input_ref(raw: Any) -> ArtifactRef | None:
    if not isinstance(raw, Mapping):
        return None
    if raw.get("type") == "external_artifact":
        return None
    expected_keys = {"type", "scope", "kind", "artifact_id"}
    if raw.get("scope") == "assay":
        expected_keys.add("assay")
    if set(raw) != expected_keys:
        return None
    try:
        return ArtifactRef.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return None


def _validate_feature_summary_parent(
    root: zarr.Group,
    assay: str,
    ref: ArtifactRef,
) -> None:
    context = _ref_context(ref, assay=assay)
    if ref.kind != "feature_summary":
        raise ArtifactResolutionError(
            "Feature-selection summary input has the wrong kind",
            code="wrong_kind",
            context=context,
        )
    if ref.scope != "assay" or ref.assay != assay:
        raise ArtifactResolutionError(
            "Feature-selection summary input has the wrong assay scope",
            code="wrong_assay" if ref.assay != assay else "wrong_scope",
            context=context,
        )
    try:
        status = inspect_artifact(root, ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Feature-summary artifact record is malformed",
            code="corrupt_payload",
            context=context,
        ) from exc
    if not status.exists:
        raise ArtifactResolutionError(
            "Feature-summary artifact is missing",
            code="missing_artifact",
            context=context,
        )
    if not status.complete:
        raise ArtifactResolutionError(
            "Feature-summary artifact is incomplete",
            code="incomplete_artifact",
            context=context,
        )
    expected = {
        "summarize_rna_features": (
            frozenset({"normalization_method", "size_factor"}),
            ("normed_tot", "normed_n", "sigmas"),
        ),
        "summarize_atac_features": (
            frozenset({"normalization_method"}),
            ("prevalence", "document_frequency"),
        ),
    }.get(status.operation or "")
    if expected is None:
        raise ArtifactResolutionError(
            "Feature-summary operation is incompatible",
            code="corrupt_payload",
            context=context,
        )
    parameter_names, payload_names = expected
    if set(status.parameters or {}) != parameter_names or set(status.inputs or {}) != {
        "cell_selection"
    }:
        raise ArtifactResolutionError(
            "Feature-summary provenance does not match its operation",
            code="corrupt_payload",
            context=context,
        )
    raw_cell_selection = (status.inputs or {}).get("cell_selection")
    cell_selection = _local_input_ref(raw_cell_selection)
    if (
        cell_selection is None
        or cell_selection.kind != "cell_selection"
        or cell_selection.scope != "datastore"
    ):
        raise ArtifactResolutionError(
            "Feature-summary cell-selection input is malformed",
            code="corrupt_payload",
            context=context,
        )
    try:
        cell_status = inspect_artifact(root, cell_selection)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Feature-summary cell-selection record is malformed",
            code="corrupt_payload",
            context=context,
        ) from exc
    if not cell_status.exists:
        raise ArtifactResolutionError(
            "Feature-summary cell-selection input is missing",
            code="missing_artifact",
            context=context,
        )
    if not cell_status.complete:
        raise ArtifactResolutionError(
            "Feature-summary cell-selection input is incomplete",
            code="incomplete_artifact",
            context=context,
        )
    validate_stored_selection_integrity(
        root,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    feature_data = _feature_data(root, assay)
    ids = as_zarr_array(feature_data["ids"], name=f"{assay}/featureData/ids")
    group = artifact_group(root, ref)
    if set(group.array_keys()) != set(payload_names):
        raise ArtifactResolutionError(
            "Feature-summary payload arrays do not match its operation",
            code="corrupt_payload",
            context=context,
        )
    try:
        arrays = [as_zarr_array(group[name], name=name) for name in payload_names]
    except (KeyError, TypeError) as exc:
        raise ArtifactResolutionError(
            "Feature-summary payload is malformed",
            code="corrupt_payload",
            context=context,
        ) from exc
    if any(
        array.ndim != 1
        or array.shape != ids.shape
        or np.dtype(array.dtype) != np.dtype(np.float64)
        for array in arrays
    ):
        raise ArtifactResolutionError(
            "Feature-summary arrays do not align with assay features",
            code="corrupt_payload",
            context=context,
        )
    if group.attrs.get("ordered_feature_ids_fingerprint") != (
        fingerprint_stored_strings(ids)
    ):
        raise ArtifactResolutionError(
            "Feature-summary row identity does not match the assay",
            code="row_mismatch",
            context=context,
        )
    try:
        payload_fingerprint = fingerprint_stored_arrays(group, payload_names)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Feature-summary payload is malformed",
            code="corrupt_payload",
            context=context,
        ) from exc
    if group.attrs.get("payload_fingerprint") != payload_fingerprint:
        raise ArtifactResolutionError(
            "Feature-summary payload fingerprint does not match",
            code="corrupt_payload",
            context=context,
        )


def _validate_feature_selection_provenance(
    root: zarr.Group,
    assay: str,
    ref: ArtifactRef,
    status: ArtifactStatus,
    group: zarr.Group,
    *,
    seen: set[ArtifactRef],
) -> tuple[str, ...]:
    context = _ref_context(ref, assay=assay)
    contract = _FEATURE_SELECTION_CONTRACTS.get(status.operation or "")
    if contract is None:
        raise ArtifactResolutionError(
            "Feature-selection operation is incompatible",
            code="corrupt_payload",
            context=context,
        )
    input_names, parameter_names, payload_names = contract
    inputs = status.inputs or {}
    parameters = status.parameters or {}
    received_inputs = set(inputs)
    if received_inputs != input_names or set(parameters) != parameter_names:
        raise ArtifactResolutionError(
            "Feature-selection provenance does not match its operation",
            code="corrupt_payload",
            context=context,
        )
    if set(group.array_keys()) != set(payload_names):
        raise ArtifactResolutionError(
            "Feature-selection payload arrays do not match its operation",
            code="corrupt_payload",
            context=context,
        )
    if "all_features" in inputs:
        all_features = _local_input_ref(inputs["all_features"])
        if all_features is None:
            raise ArtifactResolutionError(
                "Feature-selection universe input is malformed",
                code="corrupt_payload",
                context=context,
            )
        validated = _validate_feature_selection(
            root,
            assay,
            all_features,
            seen=seen,
        )
        if validated.operation != "create_all_features":
            raise ArtifactResolutionError(
                "Feature-selection universe input is not all_features",
                code="corrupt_payload",
                context=context,
            )
    if "feature_summary" in inputs:
        summary = _local_input_ref(inputs["feature_summary"])
        if summary is None:
            raise ArtifactResolutionError(
                "Feature-selection summary input is malformed",
                code="corrupt_payload",
                context=context,
            )
        _validate_feature_summary_parent(root, assay, summary)
    if "feature_snapshot" in inputs:
        snapshot = _local_input_ref(inputs["feature_snapshot"])
        if (
            snapshot is None
            or snapshot.kind != "metadata_snapshot"
            or snapshot.scope != "assay"
            or snapshot.assay != assay
        ):
            raise ArtifactResolutionError(
                "Feature-selection metadata snapshot input is malformed",
                code="corrupt_payload",
                context=context,
            )
        validate_run_metadata_snapshot(
            root,
            snapshot,
            axis="feature",
            assay=assay,
            table_path=f"{assay}/featureData",
            ordered_columns=("names",),
        )
    if "mapping_reference" in inputs:
        raw_mapping = inputs["mapping_reference"]
        if not isinstance(raw_mapping, Mapping):
            raise ArtifactResolutionError(
                "Mapping-reference input is malformed",
                code="corrupt_payload",
                context=context,
            )
        try:
            mapping_ref = ExternalArtifactRef.from_dict(raw_mapping)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                "Mapping-reference input is malformed",
                code="corrupt_payload",
                context=context,
            ) from exc
        if mapping_ref.ref.kind != "mapping_reference":
            raise ArtifactResolutionError(
                "Mapping-reference input has the wrong kind",
                code="wrong_kind",
                context=context,
            )
    return payload_names


def _validate_feature_selection(
    root: zarr.Group,
    assay: str,
    ref: ArtifactRef,
    *,
    seen: set[ArtifactRef] | None = None,
) -> _ValidatedFeatureSelection:
    _validate_ref_scope(ref, assay)
    context = _ref_context(ref, assay=assay)
    if seen is None:
        seen = set()
    if ref in seen:
        raise ArtifactResolutionError(
            "Feature-selection provenance contains a cycle",
            code="corrupt_payload",
            context=context,
        )
    seen.add(ref)
    try:
        status = inspect_artifact(root, ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Feature selection artifact record is malformed",
            code="corrupt_payload",
            context=context,
        ) from exc
    if not status.exists:
        raise ArtifactResolutionError(
            "Feature selection artifact does not exist",
            code="missing_artifact",
            context=context,
        )
    if not status.complete:
        raise ArtifactResolutionError(
            "Feature selection artifact is incomplete",
            code="incomplete_artifact",
            context=context,
        )

    feature_data = _feature_data(root, assay)
    try:
        ids = as_zarr_array(feature_data["ids"], name=f"{assay}/featureData/ids")
        group = artifact_group(root, ref)
        values = as_zarr_array(group["values"], name="values")
    except (KeyError, TypeError) as exc:
        raise ArtifactResolutionError(
            "Feature selection payload is missing or malformed",
            code="corrupt_payload",
            context=context,
        ) from exc
    if (
        values.ndim != 1
        or np.dtype(values.dtype) != np.dtype(bool)
        or ids.ndim != 1
        or values.shape != ids.shape
    ):
        raise ArtifactResolutionError(
            "Feature selection values do not align with assay features",
            code="corrupt_payload",
            context=context,
        )

    expected_rows = group.attrs.get("ordered_feature_ids_fingerprint")
    try:
        current_rows = fingerprint_stored_strings(ids)
    except (TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Assay feature row identifiers are malformed",
            code="row_mismatch",
            context=context,
        ) from exc
    if not isinstance(expected_rows, str) or expected_rows != current_rows:
        raise ArtifactResolutionError(
            "Feature selection row identity does not match the assay",
            code="row_mismatch",
            context=context,
        )

    try:
        payload_names = _validate_feature_selection_provenance(
            root,
            assay,
            ref,
            status,
            group,
            seen=seen,
        )
        actual_payload = fingerprint_stored_arrays(group, payload_names)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ArtifactResolutionError):
            raise
        raise ArtifactResolutionError(
            "Feature selection payload cannot be fingerprinted",
            code="corrupt_payload",
            context=context,
        ) from exc
    expected_payload = group.attrs.get("payload_fingerprint")
    if not isinstance(expected_payload, str) or expected_payload != actual_payload:
        raise ArtifactResolutionError(
            "Feature selection payload fingerprint does not match",
            code="corrupt_payload",
            context=context,
        )

    operation = status.operation
    if operation is None:
        raise ArtifactResolutionError(
            "Feature selection has no producing operation",
            code="corrupt_payload",
            context=context,
        )
    parameters = status.parameters or {}
    if (
        operation == "set_feature_selection"
        and parameters.get("values_fingerprint") != actual_payload
    ):
        raise ArtifactResolutionError(
            "Feature-selection value identity does not match its payload",
            code="corrupt_payload",
            context=context,
        )
    if operation == "create_all_features":
        if parameters.get("ordered_feature_ids_fingerprint") != expected_rows:
            raise ArtifactResolutionError(
                "Feature-universe row identity is inconsistent",
                code="row_mismatch",
                context=context,
            )
        block_rows = row_band(array_geometry(values), unit="chunk", fallback=1)
        for start in range(0, int(values.shape[0]), block_rows):
            stop = min(start + block_rows, int(values.shape[0]))
            if not bool(np.asarray(values[start:stop], dtype=bool).all()):
                raise ArtifactResolutionError(
                    "Feature universe must select every feature",
                    code="corrupt_payload",
                    context=context,
                )
    return _ValidatedFeatureSelection(ref=ref, values=values, operation=operation)


def resolve_feature_selection(
    root: zarr.Group,
    assay: str,
    features: ArtifactRef,
) -> ArtifactRef:
    """Validate and return one explicit feature-selection artifact."""
    if not isinstance(features, ArtifactRef):
        raise TypeError("features must be an ArtifactRef")
    return _validate_feature_selection(root, assay, features).ref


def read_feature_selection_indices(
    root: zarr.Group,
    assay: str,
    features: ArtifactRef,
) -> np.ndarray:
    """Read selected feature indices without materializing the full mask."""
    validated = _validate_feature_selection(root, assay, features)
    values = validated.values
    block_rows = row_band(array_geometry(values), unit="chunk", fallback=1)
    selected: list[np.ndarray] = []
    for start in range(0, int(values.shape[0]), block_rows):
        stop = min(start + block_rows, int(values.shape[0]))
        indices = np.flatnonzero(np.asarray(values[start:stop], dtype=bool))
        if len(indices):
            selected.append(indices.astype(np.intp, copy=False) + start)
    if not selected:
        return np.empty(0, dtype=np.intp)
    return np.concatenate(selected)
