from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from ..storage.arrays import create_metadata_column, create_numeric_array
from ..storage.artifact_writer import (
    ArrayRequirement,
    PlannedArtifact,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.artifacts import (
    ValueFingerprintBuilder,
    artifact_group,
    fingerprint_stored_strings,
    fingerprint_strings,
    inspect_artifact,
    require_complete_artifact,
)
from ..storage.errors import ArtifactResolutionError
from ..storage.geometry import array_geometry
from ..storage.layout import _group_zarr_format, row_sharded_array_spec
from ..storage.partition import row_band
from ..storage.profiles import resolve_storage_profile
from ..storage.refs import ArtifactRef
from ..storage.selections import (
    ValidatedStoredSelection,
    fingerprint_selected_stored_strings,
    validate_stored_selection_integrity,
)
from ..storage.types import as_zarr_array, as_zarr_group


@dataclass(frozen=True, slots=True)
class ImportedArtifactStorage:
    """Narrow storage adapter for imported embedding artifacts."""

    root: zarr.Group

    @staticmethod
    def resolution_error(
        message: str,
        *,
        code: str,
        context: Mapping[str, Any],
    ) -> ArtifactResolutionError:
        return ArtifactResolutionError(message, code=code, context=context)

    @staticmethod
    def requirement(
        name: str,
        *,
        shape: tuple[int, ...],
        dtype_kind: str | None,
    ) -> ArrayRequirement:
        return ArrayRequirement(name, shape=shape, dtype_kind=dtype_kind)

    def plan(
        self,
        *,
        assay: str,
        kind: str,
        parameters: dict[str, Any],
        inputs: dict[str, Any],
        execution_options: dict[str, Any],
        invalidate_cache: bool,
        required_arrays: tuple[ArrayRequirement, ...],
        reuse_validator: Callable[[ArtifactRef, zarr.Group], bool],
    ) -> PlannedArtifact:
        return plan_artifact(
            self.root,
            scope="assay",
            assay=assay,
            kind=kind,
            operation="import_dimreduc",
            parameters=parameters,
            inputs=inputs,
            execution_options=execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=required_arrays,
            reuse_validator=reuse_validator,
        )

    def start(self, planned: PlannedArtifact) -> zarr.Group:
        return start_artifact(self.root, planned)

    @staticmethod
    def finish(group: zarr.Group, planned: PlannedArtifact) -> None:
        finish_artifact(group, planned)

    def artifact_group(self, ref: ArtifactRef) -> zarr.Group:
        return artifact_group(self.root, ref)

    def require_complete(self, ref: ArtifactRef) -> Any:
        return require_complete_artifact(self.root, ref)

    def validate_cell_selection(
        self,
        ref: ArtifactRef,
    ) -> ValidatedStoredSelection:
        return validate_stored_selection_integrity(
            self.root,
            ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

    @staticmethod
    def as_array(node: zarr.Array | zarr.Group, name: str) -> zarr.Array:
        return as_zarr_array(node, name=name)

    @staticmethod
    def as_group(node: zarr.Array | zarr.Group, name: str) -> zarr.Group:
        return as_zarr_group(node, name=name)

    @staticmethod
    def fingerprint_builder() -> ValueFingerprintBuilder:
        return ValueFingerprintBuilder()

    @staticmethod
    def fingerprint_strings(values: np.ndarray) -> str:
        return fingerprint_strings(values)

    @staticmethod
    def fingerprint_stored_strings(array: zarr.Array) -> str:
        return fingerprint_stored_strings(array)

    @staticmethod
    def fingerprint_selected_strings(
        ids: zarr.Array,
        selection: zarr.Array,
    ) -> tuple[str, int]:
        return fingerprint_selected_stored_strings(ids, selection)

    @staticmethod
    def block_rows(array: zarr.Array) -> int:
        return row_band(array_geometry(array), unit="chunk", fallback=1)

    @staticmethod
    def create_numeric(
        group: zarr.Group,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        block_rows: int,
    ) -> zarr.Array:
        return create_numeric_array(
            group,
            name,
            row_sharded_array_spec(
                shape,
                dtype,
                profile=resolve_storage_profile(group.store),
                band_rows=min(max(shape[0], 1), block_rows),
                zarr_format=_group_zarr_format(group),
                fill_value=0.0,
            ),
        )

    @staticmethod
    def create_metadata(
        group: zarr.Group,
        name: str,
        *,
        dtype: Any,
        shape: int,
        block_rows: int,
    ) -> zarr.Array:
        return create_metadata_column(
            group,
            name,
            dtype=dtype,
            shape=shape,
            chunkSize=min(max(shape, 1), block_rows),
            overwrite=True,
        )


def _require_artifact_ref(
    ref: ArtifactRef,
    *,
    kind: str,
    scope: str,
    assay: str | None,
) -> None:
    if ref.kind != kind or ref.scope != scope or ref.assay != assay:
        raise ArtifactResolutionError(
            f"Expected {scope}-scoped {kind} artifact",
            code="artifact_reference_mismatch",
            context={
                "expected_scope": scope,
                "expected_assay": assay,
                "expected_kind": kind,
                "actual_scope": ref.scope,
                "actual_assay": ref.assay,
                "actual_kind": ref.kind,
                "artifact_id": ref.artifact_id,
            },
        )


def _stored_value_fingerprint(array: zarr.Array) -> str:
    builder = ValueFingerprintBuilder()
    builder.begin_array("values", array.shape, array.dtype)
    block_rows = row_band(array_geometry(array), unit="chunk", fallback=1)
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        block = np.asarray(array[start:stop])
        builder.update_array_block(
            "values",
            (start,) + (0,) * (array.ndim - 1),
            block,
        )
    builder.end_array("values")
    return builder.hexdigest()


def _payload_fingerprint(group: zarr.Group, name: str) -> str:
    array = as_zarr_array(group[name], name=name)
    if name == "feature_ids":
        return fingerprint_stored_strings(array)
    return _stored_value_fingerprint(array)


def validate_imported_coordinates_artifact(
    root: zarr.Group,
    coordinates: ArtifactRef,
) -> None:
    _require_artifact_ref(
        coordinates,
        kind="imported_coordinates",
        scope="assay",
        assay=coordinates.assay,
    )
    if coordinates.assay is None:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has no assay",
            code="wrong_scope",
            context={
                "artifact_id": coordinates.artifact_id,
                "actual_scope": coordinates.scope,
                "actual_assay": coordinates.assay,
                "actual_kind": coordinates.kind,
            },
        )
    error_context = {
        "assay": coordinates.assay,
        "artifact_id": coordinates.artifact_id,
        "actual_kind": coordinates.kind,
    }
    try:
        status = inspect_artifact(root, coordinates)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact record is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    if not status.exists:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact does not exist",
            code="missing_artifact",
            context=error_context,
        )
    if not status.complete:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact is incomplete",
            code="incomplete_artifact",
            context=error_context,
        )
    if status.operation != "import_dimreduc":
        raise ArtifactResolutionError(
            "Imported-coordinate artifact operation must be 'import_dimreduc'",
            code="corrupt_payload",
            context=error_context,
        )
    execution = status.execution_options or {}
    block_rows = execution.get("block_rows")
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int | np.integer)
        or int(block_rows) < 1
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate block_rows is invalid",
            code="corrupt_payload",
            context=error_context,
        )

    inputs = status.inputs or {}
    raw_selection = inputs.get("cell_selection")
    if not isinstance(raw_selection, Mapping):
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has no 'cell_selection' artifact input",
            code="corrupt_payload",
            context=error_context,
        )
    try:
        selection = ArtifactRef.from_dict(raw_selection)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has a malformed cell_selection input",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    validated_selection = validate_stored_selection_integrity(
        root,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )

    source_digest = inputs.get("source_digest")
    if (
        not isinstance(source_digest, Mapping)
        or set(source_digest) != {"bytes_hex"}
        or not isinstance(source_digest.get("bytes_hex"), str)
        or len(source_digest["bytes_hex"]) != 64
        or source_digest["bytes_hex"].lower() != source_digest["bytes_hex"]
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate source digest is missing",
            code="corrupt_payload",
            context=error_context,
        )
    try:
        bytes.fromhex(source_digest["bytes_hex"])
    except ValueError as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate source digest is not hexadecimal",
            code="corrupt_payload",
            context=error_context,
        ) from exc

    payload_fingerprints = inputs.get("payload_fingerprints")
    if not isinstance(payload_fingerprints, Mapping) or not all(
        isinstance(name, str)
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        for name, fingerprint in payload_fingerprints.items()
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate payload fingerprints are malformed",
            code="corrupt_payload",
            context=error_context,
        )

    try:
        group = artifact_group(root, coordinates)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact payload is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    if "data" not in group:
        raise ArtifactResolutionError(
            "Imported-coordinate artifact has no data array",
            code="corrupt_payload",
            context=error_context,
        )
    try:
        data = as_zarr_array(group["data"], name="data")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate data payload is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    if data.ndim != 2 or int(data.shape[0]) < 1 or np.dtype(data.dtype).kind != "f":
        raise ArtifactResolutionError(
            "Imported-coordinate data must be a floating-point matrix",
            code="corrupt_payload",
            context=error_context,
        )

    parameters = status.parameters or {}
    dimreduc_key = parameters.get("dimreduc_key")
    if not isinstance(dimreduc_key, str) or not dimreduc_key:
        raise ArtifactResolutionError(
            "Imported-coordinate source key is missing",
            code="corrupt_payload",
            context=error_context,
        )
    dims = parameters.get("dims")
    if isinstance(dims, bool) or not isinstance(dims, int | np.integer):
        raise ArtifactResolutionError(
            "Imported-coordinate dimensions are missing",
            code="corrupt_payload",
            context=error_context,
        )
    if int(dims) < 1 or int(data.shape[1]) != int(dims):
        raise ArtifactResolutionError(
            "Imported-coordinate dimensions do not match data",
            code="corrupt_payload",
            context=error_context,
        )
    role = parameters.get("role")
    if (
        not isinstance(role, str)
        or not role
        or role.lower() != role
        or role in {"umap", "tsne"}
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate role is invalid",
            code="corrupt_payload",
            context=error_context,
        )

    try:
        mask = validated_selection.values
        row_ids = validated_selection.row_ids
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate cell alignment payload is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    try:
        selected_fingerprint, selected_count = fingerprint_selected_stored_strings(
            row_ids,
            mask,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactResolutionError(
            "Imported-coordinate cell alignment payload is malformed",
            code="corrupt_payload",
            context=error_context,
        ) from exc
    if int(data.shape[0]) != selected_count:
        raise ArtifactResolutionError(
            "Imported-coordinate rows do not match the exact cell selection",
            code="dimreduc_row_count_mismatch",
            context={
                **error_context,
                "coordinate_rows": int(data.shape[0]),
                "selected_count": selected_count,
            },
        )
    ordered_fingerprint = inputs.get("ordered_cell_ids_fingerprint")
    if (
        not isinstance(ordered_fingerprint, str)
        or ordered_fingerprint != selected_fingerprint
    ):
        raise ArtifactResolutionError(
            "Imported-coordinate cell IDs do not match the selected cell order",
            code="dimreduc_cell_identity_mismatch",
            context={
                **error_context,
                "expected_fingerprint": ordered_fingerprint
                if isinstance(ordered_fingerprint, str)
                else None,
                "actual_fingerprint": selected_fingerprint,
            },
        )

    optional = {
        "loadings": parameters.get("loadings_stored"),
        "feature_ids": parameters.get("feature_ids_stored"),
        "stdev": parameters.get("stdev_stored"),
    }
    for name, stored in optional.items():
        if not isinstance(stored, bool) or stored != (name in group):
            raise ArtifactResolutionError(
                f"Imported-coordinate {name!r} storage flag does not match payload",
                code="corrupt_payload",
                context={**error_context, "payload": name},
            )
    if optional["loadings"] != optional["feature_ids"]:
        raise ArtifactResolutionError(
            "Imported-coordinate loadings and feature IDs must be stored together",
            code="corrupt_payload",
            context=error_context,
        )
    if "loadings" in group:
        try:
            loadings = as_zarr_array(group["loadings"], name="loadings")
            feature_ids = as_zarr_array(group["feature_ids"], name="feature_ids")
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                "Imported-coordinate loadings or feature IDs are malformed",
                code="corrupt_payload",
                context=error_context,
            ) from exc
        if (
            loadings.ndim != 2
            or np.dtype(loadings.dtype).kind != "f"
            or int(loadings.shape[0]) < 1
            or int(loadings.shape[1]) != int(dims)
            or feature_ids.ndim != 1
            or np.dtype(feature_ids.dtype).kind not in {"O", "S", "U"}
            or int(feature_ids.shape[0]) != int(loadings.shape[0])
        ):
            raise ArtifactResolutionError(
                "Imported-coordinate loadings and feature IDs are misaligned",
                code="corrupt_payload",
                context=error_context,
            )
    if "stdev" in group:
        try:
            stdev = as_zarr_array(group["stdev"], name="stdev")
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                "Imported-coordinate stdev payload is malformed",
                code="corrupt_payload",
                context=error_context,
            ) from exc
        if (
            stdev.ndim != 1
            or np.dtype(stdev.dtype).kind != "f"
            or tuple(stdev.shape) != (int(dims),)
        ):
            raise ArtifactResolutionError(
                "Imported-coordinate stdev does not match dimensions",
                code="corrupt_payload",
                context=error_context,
            )

    payload_names = ("data", "loadings", "feature_ids", "stdev")
    stored_payloads = {name for name in payload_names if name in group}
    if set(payload_fingerprints) != stored_payloads:
        raise ArtifactResolutionError(
            "Imported-coordinate payload fingerprints do not match stored payloads",
            code="corrupt_payload",
            context=error_context,
        )
    for name in payload_names:
        if name not in group:
            continue
        expected = payload_fingerprints.get(name)
        if not isinstance(expected, str):
            raise ArtifactResolutionError(
                f"Imported-coordinate payload fingerprint for {name!r} is missing",
                code="corrupt_payload",
                context={**error_context, "payload": name},
            )
        try:
            actual = _payload_fingerprint(group, name)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                f"Imported-coordinate payload fingerprint for {name!r} is unreadable",
                code="corrupt_payload",
                context={**error_context, "payload": name},
            ) from exc
        if actual != expected:
            raise ArtifactResolutionError(
                f"Imported-coordinate payload fingerprint for {name!r} does not match",
                code="corrupt_payload",
                context={**error_context, "payload": name},
            )
