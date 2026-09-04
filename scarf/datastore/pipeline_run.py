import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, Protocol, overload

import numpy as np
import pandas as pd
import zarr

from ..metadata.rows import MetaDataRowBlock, read_array_rows_chunkwise
from ..storage.artifacts import (
    artifact_group,
    fingerprint_stored_strings,
    inspect_artifact,
)
from ..storage.errors import ArtifactResolutionError
from ..storage.feature_selection import resolve_feature_selection
from ..storage.geometry import array_geometry
from ..storage.partition import partition_indices, row_band
from ..storage.pipeline_runs import (
    PipelineFieldDescriptor,
    PipelineRunRecord,
    PipelineStageRecord,
    list_pipeline_run_records,
    load_pipeline_stage_records,
    open_pipeline_run_record,
)
from ..storage.refs import ArtifactRef
from ..storage.selections import (
    StoredSelectionBlock,
    iter_stored_selection_blocks,
    validate_run_metadata_snapshot,
    validate_stored_selection_integrity,
)
from ..storage.types import as_zarr_array


class _MetadataTable(Protocol):
    N: int

    def fetch_all(self, column: str) -> np.ndarray: ...

    def _get_array(self, column: str) -> zarr.Array: ...


class _AssayOwner(Protocol):
    feats: _MetadataTable


class PipelineRunOwner(Protocol):
    zw: zarr.Group
    cells: _MetadataTable

    def get_assay(self, assay_name: str) -> _AssayOwner: ...


class PipelineExecutionError(RuntimeError):
    """A handled pipeline failure with a durable run identifier."""

    def __init__(
        self,
        run_id: str,
        stage: str,
        cause: BaseException,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("Pipeline execution error run_id must be non-empty")
        if not isinstance(stage, str) or not stage:
            raise TypeError("Pipeline execution error stage must be non-empty")
        if not isinstance(cause, BaseException):
            raise TypeError("Pipeline execution error cause must be an exception")
        super().__init__(
            f"Pipeline run {run_id} failed during stage {stage!r}: {cause}"
        )
        self.run_id = run_id
        self.stage = stage


def _artifact_context(
    ref: ArtifactRef,
    *,
    run_id: str,
    field: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "field": field,
        "scope": ref.scope,
        "assay": ref.assay or "",
        "kind": ref.kind,
        "artifact_id": ref.artifact_id,
    }


class PipelineAxisView:
    """A narrow, immutable view over one completed pipeline-run axis."""

    __slots__ = (
        "_axis",
        "_assay",
        "_descriptors",
        "_descriptor_by_key",
        "_live_table",
        "_root",
        "_run_id",
        "_selection_descriptor",
        "_selection_ref",
    )

    def __init__(
        self,
        owner: PipelineRunOwner,
        record: PipelineRunRecord,
        *,
        axis: Literal["cells", "features"],
    ) -> None:
        if not record.successfully_completed:
            raise RuntimeError(
                f"Pipeline run {record.run_id} is not completed successfully"
            )
        if axis not in {"cells", "features"}:
            raise ValueError(f"Invalid pipeline view axis: {axis!r}")
        self._root = owner.zw
        self._run_id = record.run_id
        self._axis = axis
        self._assay = record.assay
        self._live_table = (
            owner.cells if axis == "cells" else owner.get_assay(record.assay).feats
        )
        self._descriptors = tuple(
            field for field in record.fields if field.axis == axis
        )
        self._descriptor_by_key = {field.key: field for field in self._descriptors}
        missing = tuple(
            key for key in ("I", "ids", "names") if key not in self._descriptor_by_key
        )
        if missing:
            raise ArtifactResolutionError(
                f"Pipeline run {self._run_id} has no persisted {self._axis} "
                f"descriptor(s): {', '.join(missing)}",
                code="pipeline_view_required_fields_missing",
                context={"run_id": self._run_id, "axis": self._axis},
            )
        self._selection_descriptor = self._descriptor_by_key["I"]
        self._selection_ref = self._selection_descriptor.artifact
        self._validate_contract()

    @property
    def columns(self) -> tuple[str, ...]:
        ordered = ["I", "ids", "names"]
        ordered.extend(
            descriptor.key
            for descriptor in self._descriptors
            if descriptor.key not in {"I", "ids", "names"}
        )
        return tuple(ordered)

    def _complete_group(self, descriptor: PipelineFieldDescriptor) -> zarr.Group:
        ref = descriptor.artifact
        if self._axis == "features" and (
            ref.scope != "assay" or ref.assay != self._assay
        ):
            raise ArtifactResolutionError(
                f"Pipeline feature field {descriptor.key!r} belongs to "
                f"assay {ref.assay!r}, not {self._assay!r}",
                code="pipeline_field_axis_mismatch",
                context=_artifact_context(
                    ref,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        if self._axis == "cells" and ref.scope == "assay" and ref.assay != self._assay:
            raise ArtifactResolutionError(
                f"Pipeline cell field {descriptor.key!r} belongs to "
                f"assay {ref.assay!r}, not {self._assay!r}",
                code="pipeline_field_axis_mismatch",
                context=_artifact_context(
                    ref,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        try:
            status = inspect_artifact(self._root, ref)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} has a malformed artifact",
                code="pipeline_field_artifact_malformed",
                context=_artifact_context(
                    ref,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            ) from exc
        if not status.exists:
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} artifact is missing",
                code="artifact_missing",
                context=_artifact_context(
                    ref,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        if not status.complete:
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} artifact is incomplete",
                code="artifact_incomplete",
                context=_artifact_context(
                    ref,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        return artifact_group(self._root, ref)

    def _source_array(
        self,
        descriptor: PipelineFieldDescriptor,
        *,
        missing: bool = False,
    ) -> zarr.Array:
        group = self._complete_group(descriptor)
        name = descriptor.missing_mask if missing else descriptor.source_value
        if name is None or name not in group:
            kind = "missing mask" if missing else "source value"
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} has no {kind} {name!r}",
                code="pipeline_field_payload_missing",
                context=_artifact_context(
                    descriptor.artifact,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        try:
            return as_zarr_array(group[name], name=name)
        except TypeError as exc:
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} payload is not an array",
                code="pipeline_field_payload_malformed",
                context=_artifact_context(
                    descriptor.artifact,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            ) from exc

    def _resolved_array_shape(
        self,
        descriptor: PipelineFieldDescriptor,
        array: zarr.Array,
        *,
        missing: bool = False,
    ) -> tuple[int]:
        value_index = descriptor.value_index
        if value_index is None:
            if array.ndim != 1:
                label = "missing-mask" if missing else "source"
                raise ArtifactResolutionError(
                    f"Pipeline field {descriptor.key!r} {label} must be one-dimensional",
                    code="pipeline_field_shape_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                )
        elif array.ndim == 1 and missing:
            # One mask may describe all components of a multi-value payload.
            pass
        elif array.ndim != 2 or value_index >= int(array.shape[1]):
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} valueIndex is out of bounds",
                code="pipeline_field_shape_mismatch",
                context=_artifact_context(
                    descriptor.artifact,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        return (int(array.shape[0]),)

    def _expected_dtype(self, descriptor: PipelineFieldDescriptor) -> np.dtype[Any]:
        try:
            return np.dtype(descriptor.dtype)
        except TypeError as exc:
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} has invalid dtype metadata",
                code="pipeline_field_dtype_mismatch",
                context=_artifact_context(
                    descriptor.artifact,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            ) from exc

    @staticmethod
    def _dtype_matches(actual: np.dtype[Any], expected: np.dtype[Any]) -> bool:
        if actual.kind in {"O", "S", "U"} and expected.kind in {"O", "S", "U"}:
            return True
        return actual == expected

    def _selection_array(self) -> zarr.Array:
        descriptor = self._selection_descriptor
        array = self._source_array(descriptor)
        if array.ndim != 1 or np.dtype(array.dtype) != np.dtype(bool):
            raise ArtifactResolutionError(
                f"Pipeline {self._axis} I field must be a boolean vector",
                code="pipeline_view_selection_malformed",
                context=_artifact_context(
                    self._selection_ref,
                    run_id=self._run_id,
                    field="I",
                ),
            )
        if int(array.shape[0]) != int(self._live_table.N):
            raise ArtifactResolutionError(
                f"Pipeline {self._axis} I field does not match the live axis length",
                code="pipeline_field_shape_mismatch",
                context=_artifact_context(
                    self._selection_ref,
                    run_id=self._run_id,
                    field="I",
                ),
            )
        return array

    def _selected_count(self) -> int:
        selection = self._selection_array()
        block_rows = row_band(array_geometry(selection), unit="chunk", fallback=1)
        total = 0
        for start in range(0, int(selection.shape[0]), block_rows):
            stop = min(start + block_rows, int(selection.shape[0]))
            total += int(np.count_nonzero(np.asarray(selection[start:stop])))
        return total

    def _expected_row_fingerprint(self) -> str:
        ref = self._descriptor_by_key["ids"].artifact
        try:
            status = inspect_artifact(self._root, ref)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                f"Pipeline {self._axis} selection artifact is malformed",
                code="pipeline_view_selection_malformed",
                context=_artifact_context(
                    ref,
                    run_id=self._run_id,
                    field="I",
                ),
            ) from exc
        inputs = status.inputs or {}
        value = inputs.get("ordered_row_ids_fingerprint")
        if isinstance(value, str) and value:
            return value
        raise ArtifactResolutionError(
            f"Pipeline {self._axis} selection has no ordered-ID fingerprint",
            code="row_identity_fingerprint_missing",
            context=_artifact_context(
                ref,
                run_id=self._run_id,
                field="I",
            ),
        )

    def _validate_row_identity(self) -> None:
        try:
            ids = self._live_table._get_array("ids")
            received = fingerprint_stored_strings(ids)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactResolutionError(
                f"Pipeline {self._axis} live IDs are unavailable or malformed",
                code="row_identity_mismatch",
                context={"run_id": self._run_id, "axis": self._axis},
            ) from exc
        expected = self._expected_row_fingerprint()
        if received != expected:
            raise ArtifactResolutionError(
                f"Pipeline {self._axis} ordered row identities have changed",
                code="row_identity_mismatch",
                context={"run_id": self._run_id, "axis": self._axis},
            )

    def _validate_descriptor(
        self,
        descriptor: PipelineFieldDescriptor,
        *,
        selected_count: int,
    ) -> None:
        if descriptor.axis != self._axis:
            raise AssertionError("Descriptor axis was filtered incorrectly")
        if descriptor.key == "ids":
            # IDs remain live so exact row replacement can be detected. The
            # snapshot artifact is an immutable identity anchor and need not
            # duplicate the physical ids array.
            self._complete_group(descriptor)
            if descriptor.source_value != "ids" or descriptor.value_index is not None:
                raise ArtifactResolutionError(
                    "Pipeline ids descriptors require sourceValue='ids' and no index",
                    code="pipeline_field_shape_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                )
            expected_dtype = self._expected_dtype(descriptor)
            live_ids = self._live_table._get_array("ids")
            if not self._dtype_matches(np.dtype(live_ids.dtype), expected_dtype):
                raise ArtifactResolutionError(
                    "Pipeline ids descriptor dtype does not match the live axis",
                    code="pipeline_field_dtype_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                )
            if descriptor.missing_mask is not None:
                raise ArtifactResolutionError(
                    "Pipeline ids descriptors cannot define a missing mask",
                    code="pipeline_field_missing_mask_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                )
            return
        array = self._source_array(descriptor)
        if self._axis == "features" and descriptor.artifact.kind == "feature_selection":
            resolve_feature_selection(
                self._root,
                self._assay,
                descriptor.artifact,
            )
        length = self._resolved_array_shape(descriptor, array)[0]
        expected_dtype = self._expected_dtype(descriptor)
        if not self._dtype_matches(np.dtype(array.dtype), expected_dtype):
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} dtype does not match its descriptor",
                code="pipeline_field_dtype_mismatch",
                context=_artifact_context(
                    descriptor.artifact,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        if length not in {int(self._live_table.N), selected_count}:
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} has {length} rows; expected "
                f"{self._live_table.N} or {selected_count}",
                code="pipeline_field_shape_mismatch",
                context=_artifact_context(
                    descriptor.artifact,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        if descriptor.artifact.kind == "metadata_snapshot" and length != int(
            self._live_table.N
        ):
            raise ArtifactResolutionError(
                f"Pipeline snapshot field {descriptor.key!r} is not full-axis",
                code="pipeline_field_shape_mismatch",
                context=_artifact_context(
                    descriptor.artifact,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        if (
            self._axis == "cells"
            and descriptor.key not in {"I", "ids", "names"}
            and descriptor.artifact.kind != "metadata_snapshot"
        ):
            try:
                status = inspect_artifact(self._root, descriptor.artifact)
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactResolutionError(
                    f"Pipeline field {descriptor.key!r} has malformed lineage",
                    code="pipeline_field_selection_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                ) from exc
            if (status.inputs or {}).get(
                "cell_selection"
            ) != self._selection_ref.to_dict():
                raise ArtifactResolutionError(
                    f"Pipeline field {descriptor.key!r} does not use the run "
                    "cell selection",
                    code="pipeline_field_selection_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                )
        if length != int(self._live_table.N):
            try:
                np.asarray([descriptor.fill_value], dtype=expected_dtype)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ArtifactResolutionError(
                    f"Pipeline field {descriptor.key!r} fill is incompatible with dtype",
                    code="pipeline_field_fill_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                ) from exc
        if descriptor.missing_mask is not None:
            missing = self._source_array(descriptor, missing=True)
            missing_length = self._resolved_array_shape(
                descriptor,
                missing,
                missing=True,
            )[0]
            if np.dtype(missing.dtype) != np.dtype(bool):
                raise ArtifactResolutionError(
                    f"Pipeline field {descriptor.key!r} missing mask is not boolean",
                    code="pipeline_field_missing_mask_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                )
            if missing_length != length:
                raise ArtifactResolutionError(
                    f"Pipeline field {descriptor.key!r} missing mask is misaligned",
                    code="pipeline_field_missing_mask_mismatch",
                    context=_artifact_context(
                        descriptor.artifact,
                        run_id=self._run_id,
                        field=descriptor.key,
                    ),
                )

    def _validate_contract(self) -> None:
        snapshot_keys = {"ids", "names"}
        snapshot_refs = {
            descriptor.artifact
            for descriptor in self._descriptors
            if descriptor.artifact.kind == "metadata_snapshot"
        }
        snapshot_refs.update(
            self._descriptor_by_key[key].artifact for key in snapshot_keys
        )
        snapshot_axis = "cell" if self._axis == "cells" else "feature"
        snapshot_assay = None if self._axis == "cells" else self._assay
        table_path = (
            "cellData" if self._axis == "cells" else f"{self._assay}/featureData"
        )
        for ref in snapshot_refs:
            validate_run_metadata_snapshot(
                self._root,
                ref,
                axis=snapshot_axis,
                assay=snapshot_assay,
                table_path=table_path,
            )
        if (
            self._selection_descriptor.source_value != "values"
            or self._selection_descriptor.value_index is not None
            or self._selection_descriptor.missing_mask is not None
        ):
            raise ArtifactResolutionError(
                f"Pipeline {self._axis} I descriptor does not identify selection values",
                code="pipeline_view_selection_malformed",
                context=_artifact_context(
                    self._selection_ref,
                    run_id=self._run_id,
                    field="I",
                ),
            )
        if self._axis == "cells":
            validate_stored_selection_integrity(
                self._root,
                self._selection_ref,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path=table_path,
            )
        else:
            resolve_feature_selection(
                self._root,
                self._assay,
                self._selection_ref,
            )
        self._validate_row_identity()
        selection = self._selection_array()
        selected_count = self._selected_count()
        if int(selection.shape[0]) != int(self._live_table.N):
            raise AssertionError("Selection shape was not validated")
        for descriptor in self._descriptors:
            self._validate_descriptor(descriptor, selected_count=selected_count)

    @staticmethod
    def _read_component(
        array: zarr.Array,
        value_index: int | None,
    ) -> np.ndarray:
        if value_index is None or array.ndim == 1:
            values: np.ndarray = np.asarray(array[:])
            return values
        values = np.asarray(array[:, value_index])
        return values

    @staticmethod
    def _read_component_slice(
        array: zarr.Array,
        start: int,
        stop: int,
        value_index: int | None,
    ) -> np.ndarray:
        if value_index is None or array.ndim == 1:
            values: np.ndarray = np.asarray(array[start:stop])
            return values
        values = np.asarray(array[start:stop, value_index])
        return values

    @classmethod
    def _read_component_rows(
        cls,
        array: zarr.Array,
        rows: np.ndarray,
        value_index: int | None,
    ) -> np.ndarray:
        indices = np.asarray(rows, dtype=np.int64)
        if value_index is None or array.ndim == 1:
            return read_array_rows_chunkwise(array, indices)
        output = np.empty(len(indices), dtype=np.dtype(array.dtype))
        if not len(indices):
            return output
        geometry = array_geometry(array)
        if geometry is None:
            for destination, index in enumerate(indices):
                output[destination] = cls._read_component_slice(
                    array,
                    int(index),
                    int(index) + 1,
                    value_index,
                )[0]
            return output
        for block in partition_indices(geometry, 0, indices):
            start = int(block.indices.min())
            stop = int(block.indices.max()) + 1
            values = cls._read_component_slice(array, start, stop, value_index)
            output[block.destinations] = values[block.indices - start]
        return output

    def _raw_values(self, descriptor: PipelineFieldDescriptor) -> np.ndarray:
        return self._read_component(
            self._source_array(descriptor),
            descriptor.value_index,
        )

    def _raw_missing(self, descriptor: PipelineFieldDescriptor) -> np.ndarray | None:
        if descriptor.missing_mask is None:
            return None
        missing: np.ndarray = np.asarray(
            self._read_component(
                self._source_array(descriptor, missing=True),
                descriptor.value_index,
            ),
            dtype=bool,
        )
        return missing

    def _full_axis_values(self, descriptor: PipelineFieldDescriptor) -> np.ndarray:
        values = self._raw_values(descriptor)
        if len(values) == int(self._live_table.N):
            return values
        selection = self.fetch_all("I")
        if len(values) != int(np.count_nonzero(selection)):
            raise ArtifactResolutionError(
                f"Pipeline field {descriptor.key!r} no longer aligns to view I",
                code="pipeline_field_shape_mismatch",
                context=_artifact_context(
                    descriptor.artifact,
                    run_id=self._run_id,
                    field=descriptor.key,
                ),
            )
        expected_dtype = self._expected_dtype(descriptor)
        full: np.ndarray = np.full(
            int(self._live_table.N),
            descriptor.fill_value,
            dtype=expected_dtype,
        )
        full[selection] = values
        return full

    def _selected_values(self, descriptor: PipelineFieldDescriptor) -> np.ndarray:
        values = self._raw_values(descriptor)
        if len(values) == int(self._live_table.N):
            selected: np.ndarray = np.asarray(values[self.fetch_all("I")])
            return selected
        return values

    def _iter_selection_blocks(
        self,
        *,
        block_rows: int | None,
    ) -> Iterator[StoredSelectionBlock]:
        if self._axis == "cells":
            yield from iter_stored_selection_blocks(
                self._root,
                self._selection_ref,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
                block_rows=block_rows,
            )
            return
        resolve_feature_selection(self._root, self._assay, self._selection_ref)
        selection = self._selection_array()
        chunk_rows = row_band(array_geometry(selection), unit="chunk", fallback=1)
        if block_rows is None:
            resolved_rows = chunk_rows
        else:
            requested_rows = int(block_rows)
            if requested_rows < 1:
                raise ValueError("block_rows must be >= 1")
            resolved_rows = min(requested_rows, chunk_rows)
        compact_start = 0
        for start in range(0, int(selection.shape[0]), resolved_rows):
            stop = min(start + resolved_rows, int(selection.shape[0]))
            mask = np.asarray(selection[start:stop], dtype=bool)
            indices = np.flatnonzero(mask).astype(np.intp, copy=False) + start
            compact_stop = compact_start + len(indices)
            yield StoredSelectionBlock(
                start=start,
                stop=stop,
                mask=mask,
                selected_indices=indices,
                compact_start=compact_start,
                compact_stop=compact_stop,
            )
            compact_start = compact_stop

    def _iter_selected_blocks(
        self,
        columns: Sequence[str],
        block_rows: int | None = None,
    ) -> Iterator[MetaDataRowBlock]:
        """Yield bounded full-axis blocks containing only run-selected values."""
        if isinstance(columns, str | bytes) or not isinstance(columns, Sequence):
            raise TypeError("columns must be a sequence of field names")
        requested = tuple(columns)
        if any(not isinstance(column, str) or not column for column in requested):
            raise TypeError("columns must contain non-empty strings")
        if len(requested) != len(set(requested)):
            raise ValueError("columns must not contain duplicates")
        unknown = [column for column in requested if column not in self.columns]
        if unknown:
            raise KeyError(f"Pipeline run fields were not captured: {unknown!r}")
        self._validate_row_identity()
        sources: dict[str, tuple[zarr.Array, int | None]] = {}
        missing_sources: dict[str, tuple[zarr.Array, int | None]] = {}
        for column in requested:
            if column == "I":
                continue
            if column == "ids":
                sources[column] = (self._live_table._get_array("ids"), None)
                continue
            descriptor = self._descriptor_by_key[column]
            sources[column] = (
                self._source_array(descriptor),
                descriptor.value_index,
            )
            if descriptor.missing_mask is not None:
                missing_sources[column] = (
                    self._source_array(descriptor, missing=True),
                    descriptor.value_index,
                )
        full_length = int(self._live_table.N)
        for block in self._iter_selection_blocks(block_rows=block_rows):
            values: dict[str, np.ndarray] = {}
            for column in requested:
                if column == "I":
                    values[column] = np.ones(len(block.selected_indices), dtype=bool)
                    continue
                source, value_index = sources[column]
                if int(source.shape[0]) == full_length:
                    raw = self._read_component_slice(
                        source,
                        block.start,
                        block.stop,
                        value_index,
                    )
                    values[column] = np.asarray(raw[block.mask])
                else:
                    values[column] = self._read_component_slice(
                        source,
                        block.compact_start,
                        block.compact_stop,
                        value_index,
                    )
                missing_source = missing_sources.get(column)
                if missing_source is not None:
                    missing_array, missing_index = missing_source
                    if int(missing_array.shape[0]) == full_length:
                        raw_missing = self._read_component_slice(
                            missing_array,
                            block.start,
                            block.stop,
                            missing_index,
                        )
                        missing = np.asarray(raw_missing[block.mask], dtype=bool)
                    else:
                        missing = np.asarray(
                            self._read_component_slice(
                                missing_array,
                                block.compact_start,
                                block.compact_stop,
                                missing_index,
                            ),
                            dtype=bool,
                        )
                    if np.any(missing):
                        values[column] = self._apply_plot_missing(
                            values[column],
                            missing,
                        )
            yield MetaDataRowBlock(
                start=block.start,
                stop=block.stop,
                active_global_indices=block.selected_indices.astype(
                    np.int64,
                    copy=False,
                ),
                values=values,
            )

    def _field_dtype(self, column: str) -> np.dtype[Any]:
        """Return one validated field dtype for private plotting adapters."""
        if column == "I":
            return np.dtype(bool)
        if column == "ids":
            return np.dtype(self._live_table._get_array("ids").dtype)
        descriptor = self._descriptor_by_key.get(column)
        if descriptor is None:
            raise KeyError(f"Pipeline run field {column!r} was not captured")
        return self._expected_dtype(descriptor)

    def _field_display(self, column: str) -> dict[str, Any] | None:
        """Return one frozen display contract for private plotting adapters."""
        descriptor = self._descriptor_by_key.get(column)
        if descriptor is None:
            if column in {"I", "ids"}:
                return None
            raise KeyError(f"Pipeline run field {column!r} was not captured")
        return None if descriptor.display is None else dict(descriptor.display)

    @staticmethod
    def _apply_plot_missing(
        values: np.ndarray,
        missing: np.ndarray,
    ) -> np.ndarray:
        """Represent nullable values without turning them into real plot values."""
        array = np.asarray(values)
        mask = np.asarray(missing, dtype=bool)
        if not np.any(mask):
            return array
        if array.dtype.kind == "b":
            output = array.copy()
            output[mask] = False
            return output
        if array.dtype.kind in {"f", "i", "u"}:
            output = array.astype(np.float64, copy=True)
            output[mask] = np.nan
            return output
        output = array.astype(object, copy=True)
        output[mask] = None
        return output

    def _plot_fetch_all(self, column: str) -> np.ndarray:
        """Return one full-axis field with missing values safe for plotting."""
        values = self.fetch_all(column)
        descriptor = self._descriptor_by_key.get(column)
        if descriptor is None:
            return values
        missing = self._raw_missing(descriptor)
        if missing is None:
            return values
        if len(missing) == int(self._live_table.N):
            full_missing = missing
        else:
            selection = self.fetch_all("I")
            full_missing = np.zeros(int(self._live_table.N), dtype=bool)
            full_missing[selection] = missing
        return self._apply_plot_missing(values, full_missing)

    def _plot_fetch_selected(self, column: str) -> np.ndarray:
        """Return one selected-row field with missing values safe for plotting."""
        values = self.fetch(column)
        descriptor = self._descriptor_by_key.get(column)
        if descriptor is None:
            return values
        missing = self._selected_missing(descriptor)
        if missing is None:
            return values
        return self._apply_plot_missing(values, missing)

    def _selected_prefix_indices(self, n: int) -> np.ndarray:
        if n == 0:
            return np.empty(0, dtype=np.int64)
        parts: list[np.ndarray] = []
        remaining = n
        for block in self._iter_selection_blocks(block_rows=None):
            if len(block.selected_indices):
                selected = block.selected_indices[:remaining]
                parts.append(selected.astype(np.int64, copy=False))
                remaining -= len(selected)
                if remaining == 0:
                    break
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts)

    def fetch_all(self, column: str) -> np.ndarray:
        """Return one run field aligned to the complete stored axis."""
        if not isinstance(column, str) or not column:
            raise TypeError("column must be a non-empty string")
        self._validate_row_identity()
        if column == "ids":
            ids: np.ndarray = np.asarray(self._live_table.fetch_all("ids"))
            return ids
        if column == "I":
            selected: np.ndarray = np.asarray(self._selection_array()[:], dtype=bool)
            return selected
        descriptor = self._descriptor_by_key.get(column)
        if descriptor is None:
            raise KeyError(
                f"Pipeline run field {column!r} was not captured on the {self._axis} axis"
            )
        return self._full_axis_values(descriptor)

    def fetch(self, column: str) -> np.ndarray:
        """Return one run field for rows selected by the stored run I."""
        if not isinstance(column, str) or not column:
            raise TypeError("column must be a non-empty string")
        self._validate_row_identity()
        if column == "ids":
            values = np.asarray(self._live_table.fetch_all("ids"))
            selected: np.ndarray = np.asarray(values[self.fetch_all("I")])
            return selected
        if column == "I":
            return np.ones(self._selected_count(), dtype=bool)
        descriptor = self._descriptor_by_key.get(column)
        if descriptor is None:
            raise KeyError(
                f"Pipeline run field {column!r} was not captured on the {self._axis} axis"
            )
        return self._selected_values(descriptor)

    def _selected_missing(
        self,
        descriptor: PipelineFieldDescriptor,
    ) -> np.ndarray | None:
        missing = self._raw_missing(descriptor)
        if missing is None:
            return None
        if len(missing) == int(self._live_table.N):
            selected: np.ndarray = np.asarray(
                missing[self.fetch_all("I")],
                dtype=bool,
            )
            return selected
        return missing

    def to_pandas_dataframe(self, columns: Sequence[str]) -> pd.DataFrame:
        """Return selected rows for the requested frozen run fields."""
        if isinstance(columns, str | bytes) or not isinstance(columns, Sequence):
            raise TypeError("columns must be a sequence of field names")
        requested = tuple(columns)
        if any(not isinstance(column, str) or not column for column in requested):
            raise TypeError("columns must contain non-empty strings")
        if len(requested) != len(set(requested)):
            raise ValueError("columns must not contain duplicates")
        unknown = [column for column in requested if column not in self.columns]
        if unknown:
            raise KeyError(f"Pipeline run fields were not captured: {unknown!r}")
        data: dict[str, Any] = {}
        for column in requested:
            values = self.fetch(column)
            descriptor = self._descriptor_by_key.get(column)
            missing = None if descriptor is None else self._selected_missing(descriptor)
            if missing is not None and np.any(missing):
                series = pd.Series(values)
                series[missing] = pd.NA
                data[column] = series
            else:
                data[column] = values
        return pd.DataFrame(data)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first selected rows of every frozen run field."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        self._validate_row_identity()
        rows = self._selected_prefix_indices(n)
        data: dict[str, Any] = {}
        full_length = int(self._live_table.N)
        for column in self.columns:
            if column == "I":
                data[column] = np.ones(len(rows), dtype=bool)
                continue
            if column == "ids":
                data[column] = self._read_component_rows(
                    self._live_table._get_array("ids"),
                    rows,
                    None,
                )
                continue
            descriptor = self._descriptor_by_key[column]
            source = self._source_array(descriptor)
            values = (
                self._read_component_rows(source, rows, descriptor.value_index)
                if int(source.shape[0]) == full_length
                else self._read_component_slice(
                    source,
                    0,
                    len(rows),
                    descriptor.value_index,
                )
            )
            missing = None
            if descriptor.missing_mask is not None:
                missing_source = self._source_array(descriptor, missing=True)
                missing = (
                    self._read_component_rows(
                        missing_source,
                        rows,
                        descriptor.value_index,
                    )
                    if int(missing_source.shape[0]) == full_length
                    else self._read_component_slice(
                        missing_source,
                        0,
                        len(rows),
                        descriptor.value_index,
                    )
                )
            if missing is not None and np.any(missing):
                series = pd.Series(values)
                series[np.asarray(missing, dtype=bool)] = pd.NA
                data[column] = series
            else:
                data[column] = values
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        return (
            f"PipelineAxisView(axis={self._axis!r}, "
            f"columns={len(self.columns)}, run_id={self._run_id[:12]!r})"
        )


class PipelineRun(Mapping[str, ArtifactRef]):
    """A datastore-bound handle for one durable pipeline invocation."""

    __slots__ = ("_cells_view", "_features_view", "_owner", "_outputs", "_record")

    def __init__(
        self,
        owner: PipelineRunOwner,
        record: PipelineRunRecord,
    ) -> None:
        if not hasattr(owner, "zw") or not hasattr(owner, "cells"):
            raise TypeError("PipelineRun owner must provide zw and cells")
        if not isinstance(record, PipelineRunRecord):
            raise TypeError("PipelineRun record must be a PipelineRunRecord")
        self._owner = owner
        self._record = record
        self._outputs = {output.key: output.artifact for output in record.outputs}
        self._cells_view: PipelineAxisView | None = None
        self._features_view: PipelineAxisView | None = None

    @property
    def run_id(self) -> str:
        return self._record.run_id

    @property
    def label(self) -> str | None:
        return self._record.label

    @property
    def assay(self) -> str:
        return self._record.assay

    @property
    def status(self) -> str:
        return self._record.status

    @property
    def recipe(self) -> str:
        return self._record.recipe

    @property
    def started_at_ns(self) -> int:
        return self._record.started_at_ns

    @property
    def finished_at_ns(self) -> int | None:
        return self._record.finished_at_ns

    def _require_completed(self, operation: str) -> None:
        if not self._record.successfully_completed:
            raise RuntimeError(
                f"Pipeline run {self.run_id} is {self.status!r}; "
                f"{operation} requires a completed run"
            )

    def __getitem__(self, key: str) -> ArtifactRef:
        self._require_completed("output access")
        return self._outputs[key]

    def __iter__(self) -> Iterator[str]:
        self._require_completed("output iteration")
        return iter(self._outputs)

    def __len__(self) -> int:
        self._require_completed("output inspection")
        return len(self._outputs)

    @property
    def cells(self) -> PipelineAxisView:
        self._require_completed("cell views")
        if self._cells_view is None:
            self._cells_view = PipelineAxisView(
                self._owner,
                self._record,
                axis="cells",
            )
        return self._cells_view

    @property
    def features(self) -> PipelineAxisView:
        self._require_completed("feature views")
        if self._features_view is None:
            self._features_view = PipelineAxisView(
                self._owner,
                self._record,
                axis="features",
            )
        return self._features_view

    def _report_dict(
        self,
        stages: Sequence[PipelineStageRecord] | None = None,
    ) -> dict[str, Any]:
        if stages is None:
            stages = load_pipeline_stage_records(self._owner.zw, self.run_id)
        created: list[dict[str, Any]] = []
        reused: list[dict[str, Any]] = []
        for stage in stages:
            for plan in stage.plans:
                target = created if plan.disposition == "created" else reused
                target.append(plan.ref.to_dict())
        return {
            "run": self._record.to_dict(),
            "stages": [stage.to_dict() for stage in stages],
            "summary": {
                "createdArtifacts": created,
                "reusedArtifacts": reused,
                "uncleanIncomplete": not self._record.complete,
                "signalProtection": self._record.config.get("shutdown"),
            },
        }

    @staticmethod
    def _stage_markdown(stage: PipelineStageRecord) -> str:
        duration = "" if stage.metrics is None else f"{stage.metrics.wall_seconds:.3f}"
        peak = (
            "unavailable"
            if stage.metrics is None or stage.metrics.rss_peak_bytes is None
            else str(stage.metrics.rss_peak_bytes)
        )
        created = sum(plan.disposition == "created" for plan in stage.plans)
        reused = sum(plan.disposition == "reused" for plan in stage.plans)
        return (
            f"| {stage.ordinal} | {stage.stage} | {stage.status} | {duration} | "
            f"{peak} | {created} | {reused} |"
        )

    def _report_markdown(self) -> str:
        stages = load_pipeline_stage_records(self._owner.zw, self.run_id)
        report = self._report_dict(stages)
        run = report["run"]
        lines = [
            f"# Pipeline run `{self.run_id}`",
            "",
            f"- Status: `{self.status}`",
            f"- Assay: `{self.assay}`",
            f"- Label: `{self.label}`" if self.label is not None else "- Label: none",
            f"- Recipe: `{run['recipe']}`",
            f"- Scarf version: `{run['scarfVersion']}`",
            f"- Started at ns: `{run['startedAtNs']}`",
            (
                f"- Finished at ns: `{run['finishedAtNs']}`"
                if run["finishedAtNs"] is not None
                else "- Finished at ns: not finished"
            ),
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(run["config"], indent=2, sort_keys=True),
            "```",
            "",
            "## Stages",
            "",
            "| # | Stage | Status | Wall seconds | Peak RSS bytes | Created | Reused |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
        lines.extend(self._stage_markdown(stage) for stage in stages)
        outputs = run["outputs"]
        lines.extend(["", "## Outputs", ""])
        if outputs:
            lines.extend(
                f"- `{output['key']}`: `{output['artifact']['artifact_id']}`"
                for output in outputs
            )
        else:
            lines.append("No completed outputs.")
        if run["error"] is not None:
            lines.extend(
                [
                    "",
                    "## Failure",
                    "",
                    f"- Type: `{run['error']['type']}`",
                    f"- Message: {run['error']['message']}",
                ]
            )
        if run["interruption"] is not None:
            lines.extend(
                [
                    "",
                    "## Interruption",
                    "",
                    f"- Kind: `{run['interruption']['kind']}`",
                    f"- Message: {run['interruption']['message']}",
                ]
            )
        return "\n".join(lines) + "\n"

    @overload
    def report(self, *, format: Literal["dict"] = "dict") -> dict[str, Any]: ...

    @overload
    def report(self, *, format: Literal["markdown"]) -> str: ...

    def report(
        self,
        *,
        format: Literal["dict", "markdown"] = "dict",
    ) -> dict[str, Any] | str:
        """Return deterministic persisted facts for any run status."""
        if format == "dict":
            return self._report_dict()
        if format == "markdown":
            return self._report_markdown()
        raise ValueError("format must be 'dict' or 'markdown'")

    def __repr__(self) -> str:
        label = f", label={self.label!r}" if self.label is not None else ""
        return (
            f"PipelineRun(run_id='{self.run_id[:12]}...', "
            f"status={self.status!r}, assay={self.assay!r}{label})"
        )


def open_pipeline_run(
    owner: PipelineRunOwner,
    *,
    run_id: str | None = None,
    label: str | None = None,
) -> PipelineRun:
    """Open a datastore-bound run by exact identity or completed label."""
    record = open_pipeline_run_record(owner.zw, run_id=run_id, label=label)
    return PipelineRun(owner, record)


def list_pipeline_runs(
    owner: PipelineRunOwner,
    *,
    status: str | Sequence[str] | None = None,
    limit: int = 20,
) -> tuple[PipelineRun, ...]:
    """Return lightweight datastore-bound handles newest first."""
    return tuple(
        PipelineRun(owner, record)
        for record in list_pipeline_run_records(
            owner.zw,
            status=status,
            limit=limit,
        )
    )
