from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from ..storage.arrays import create_zarr_dataset
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.artifacts import (
    ArtifactRef,
    artifact_path,
    callable_identity,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
)
from ..storage.errors import ArtifactResolutionError
from ..storage.selections import validate_stored_selection_artifact
from ..storage.types import as_zarr_array, as_zarr_group

if TYPE_CHECKING:
    from .base import Assay


_RNA_ARRAYS = ("normed_tot", "normed_n", "sigmas")
_ATAC_ARRAYS = ("prevalence", "document_frequency")


def _selection_mask(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    n_cells: int,
) -> np.ndarray:
    if ref.kind != "cell_selection" or ref.scope != "datastore":
        raise ArtifactResolutionError(
            "cell_selection must be a datastore cell-selection artifact",
            code="artifact_reference_mismatch",
            context={
                "scope": ref.scope,
                "assay": ref.assay,
                "kind": ref.kind,
                "artifact_id": ref.artifact_id,
                "expected_scope": "datastore",
                "expected_assay": None,
                "expected_kind": "cell_selection",
            },
        )
    try:
        status = inspect_artifact(root, ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Cell-selection artifact record is malformed",
            code="artifact_missing",
            context={"artifact_id": ref.artifact_id},
        ) from exc
    if not status.exists:
        raise ArtifactResolutionError(
            "Cell-selection artifact does not exist",
            code="artifact_missing",
            context={"artifact_id": ref.artifact_id},
        )
    if not status.complete:
        raise ArtifactResolutionError(
            "Cell-selection artifact is incomplete",
            code="artifact_incomplete",
            context={"artifact_id": ref.artifact_id},
        )
    source_column = (status.execution_options or {}).get("source_column")
    if not isinstance(source_column, str) or not source_column:
        raise ArtifactResolutionError(
            "Cell-selection artifact has no source column",
            code="selection_column_missing",
            context={"artifact_id": ref.artifact_id},
        )
    validate_stored_selection_artifact(
        root,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
        column=source_column,
    )
    group = as_zarr_group(root[status.path], name=status.path)
    stored_values = as_zarr_array(group["values"], name="values")
    values = np.asarray(stored_values[:])
    if values.shape != (n_cells,) or values.dtype != np.dtype(bool):
        raise ArtifactResolutionError(
            "Cell-selection artifact values do not align with the assay cells",
            code="selection_values_changed",
            context={"artifact_id": ref.artifact_id},
        )
    return values


def _summary_contract(
    names: Sequence[str],
    *,
    n_features: int,
    ordered_feature_ids_fingerprint: str,
) -> tuple[
    tuple[ArrayRequirement, ...],
    tuple[AttributeRequirement, ...],
    Any,
]:
    arrays = tuple(
        ArrayRequirement(name, shape=(n_features,), dtype=np.float64) for name in names
    )
    attributes = (
        AttributeRequirement(
            "ordered_feature_ids_fingerprint",
            expected_types=(str,),
            predicate=lambda value: value == ordered_feature_ids_fingerprint,
        ),
        AttributeRequirement(
            "payload_fingerprint",
            expected_types=(str,),
        ),
    )

    def reuse_validator(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            if set(group.array_keys()) != set(names):
                return False
            stored = group.attrs["payload_fingerprint"]
            return isinstance(stored, str) and stored == fingerprint_stored_arrays(
                group,
                names,
            )
        except (KeyError, TypeError, ValueError):
            return False

    return arrays, attributes, reuse_validator


def ensure_feature_summary(
    root: zarr.Group,
    assay: "Assay",
    cell_selection: ArtifactRef,
    *,
    invalidate_cache: bool = False,
) -> ArtifactRef:
    """Plan or compute the sufficient-statistics artifact for an assay.

    Counts and feature row order are immutable ambient inputs. The only direct
    provenance input is the selected-cell artifact. All computations cover the
    complete physical feature axis and never mount arrays on feature metadata.
    """
    cell_mask = _selection_mask(root, cell_selection, n_cells=assay.cells.N)
    cell_idx = np.flatnonzero(cell_mask).astype(np.int64, copy=False)
    n_features = int(assay.feats.N)
    feature_idx = np.arange(n_features, dtype=np.int64)
    feature_data = as_zarr_group(assay.z["featureData"], name="featureData")
    feature_ids_fingerprint = fingerprint_stored_strings(
        as_zarr_array(feature_data["ids"], name="featureData/ids")
    )

    operation = getattr(assay, "_feature_summary_operation", None)
    if operation == "summarize_rna_features":
        names: tuple[str, ...] = _RNA_ARRAYS
        parameters: dict[str, Any] = {
            "normalization_method": callable_identity(assay.normMethod),
            "size_factor": assay.sf,
        }
    elif operation == "summarize_atac_features":
        names = _ATAC_ARRAYS
        parameters = {
            "normalization_method": callable_identity(assay.normMethod),
        }
    else:
        raise TypeError(
            "Feature summaries are supported only for RNAassay and ATACassay; "
            f"received {type(assay).__name__}"
        )

    arrays, attributes, reuse_validator = _summary_contract(
        names,
        n_features=n_features,
        ordered_feature_ids_fingerprint=feature_ids_fingerprint,
    )
    planned = plan_artifact(
        root,
        scope="assay",
        assay=assay.name,
        kind="feature_summary",
        operation=operation,
        parameters=parameters,
        inputs={"cell_selection": cell_selection},
        execution_options={"nthreads": assay.nthreads},
        invalidate_cache=invalidate_cache,
        required_arrays=arrays,
        required_attributes=attributes,
        reuse_validator=reuse_validator,
    )
    if planned.reused:
        return planned.ref

    raw_payload = assay._compute_feature_summary(cell_idx, feature_idx)
    payload = {name: np.asarray(raw_payload[name], dtype=np.float64) for name in names}
    group = start_artifact(root, planned)
    chunks = (min(max(n_features, 1), 100_000),)
    for name in names:
        values = payload[name]
        if values.shape != (n_features,):
            raise ValueError(
                f"Feature summary array {name!r} has shape {values.shape}; "
                f"expected ({n_features},)"
            )
        output = create_zarr_dataset(
            group,
            name,
            chunks,
            np.float64,
            (n_features,),
        )
        output[:] = values
    group.attrs["ordered_feature_ids_fingerprint"] = feature_ids_fingerprint
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, names)
    finish_artifact(group, planned)
    return planned.ref


def feature_summary_values(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    n_selected: int,
) -> dict[str, np.ndarray]:
    """Load a validated summary and derive its non-persisted statistics."""
    if ref.kind != "feature_summary" or ref.scope != "assay":
        raise ValueError("ref must be an assay feature-summary artifact")
    status = inspect_artifact(root, ref)
    if not status.exists:
        raise KeyError(f"Feature-summary artifact does not exist: {status.path}")
    if not status.complete:
        raise RuntimeError(f"Feature-summary artifact is incomplete: {status.path}")
    group = as_zarr_group(root[artifact_path(ref)], name=artifact_path(ref))
    operation = status.operation
    if operation == "summarize_rna_features":
        names: tuple[str, ...] = _RNA_ARRAYS
    elif operation == "summarize_atac_features":
        names = _ATAC_ARRAYS
    else:
        raise ValueError(f"Unsupported feature-summary operation: {operation!r}")
    stored_fingerprint = group.attrs.get("payload_fingerprint")
    if not isinstance(stored_fingerprint, str) or stored_fingerprint != (
        fingerprint_stored_arrays(group, names)
    ):
        raise ValueError("Feature-summary payload fingerprint is invalid")

    values = {
        name: np.asarray(as_zarr_array(group[name], name=name)[:], dtype=np.float64)
        for name in names
    }
    if names == _RNA_ARRAYS:
        normed_tot = values["normed_tot"]
        normed_n = values["normed_n"]
        values["avg"] = (
            normed_tot / n_selected
            if n_selected > 0
            else np.zeros_like(normed_tot, dtype=np.float64)
        )
        values["nz_mean"] = np.divide(
            normed_tot,
            normed_n,
            out=np.zeros_like(normed_tot, dtype=np.float64),
            where=normed_n != 0,
        )
    return values


def feature_summary_selected_count(
    root: zarr.Group,
    cell_selection: ArtifactRef,
    *,
    n_cells: int,
) -> int:
    """Return the selected-cell count from the canonical selection payload."""
    return int(_selection_mask(root, cell_selection, n_cells=n_cells).sum())
