from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
import time
from typing import Any, Literal

import numpy as np
import zarr
from zarr.errors import ContainsArrayError, ContainsGroupError

from .artifacts import (
    ArtifactRef,
    ArtifactScope,
    artifact_path,
    canonical_bytes,
    make_provenance,
    new_artifact_id,
    reusable_artifact_groups,
    serialize_artifact_value,
)
from .stores import metadata_workers, run_concurrently


type ArtifactPlanDisposition = Literal["created", "reused"]


@dataclass(frozen=True, slots=True)
class ArtifactPlanReceipt:
    """One artifact planning decision observed by a pipeline stage."""

    operation: str
    ref: ArtifactRef
    disposition: ArtifactPlanDisposition


_PLAN_COLLECTORS: ContextVar[tuple[list[ArtifactPlanReceipt], ...]] = ContextVar(
    "scarf_artifact_plan_collectors",
    default=(),
)


@contextmanager
def artifact_plan_scope() -> Any:
    """Collect nested artifact planning decisions without changing producers."""

    receipts: list[ArtifactPlanReceipt] = []
    collectors = _PLAN_COLLECTORS.get()
    token = _PLAN_COLLECTORS.set((*collectors, receipts))
    try:
        yield receipts
    finally:
        _PLAN_COLLECTORS.reset(token)


def _record_plan(planned: "PlannedArtifact") -> "PlannedArtifact":
    operation = planned.provenance.get("operation")
    if not isinstance(operation, str) or not operation:
        raise TypeError("Planned artifact operation must be a non-empty string")
    receipt = ArtifactPlanReceipt(
        operation=operation,
        ref=planned.ref,
        disposition="reused" if planned.reused else "created",
    )
    for collector in _PLAN_COLLECTORS.get():
        collector.append(receipt)
    return planned


@dataclass(frozen=True, slots=True)
class PlannedArtifact:
    ref: ArtifactRef
    provenance: dict[str, Any]
    execution_options: dict[str, Any]
    reused: bool
    required_arrays: tuple[Any, ...]
    required_attributes: tuple[Any, ...]
    reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None

    def invalidated(
        self,
        root: zarr.Group,
        *,
        required_arrays: tuple[Any, ...] | None = None,
        required_attributes: tuple[Any, ...] | None = None,
        reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None = None,
    ) -> "PlannedArtifact":
        if not isinstance(self.provenance, dict):
            raise TypeError("PlannedArtifact provenance must be a mapping")
        operation = self.provenance.get("operation")
        parameters = self.provenance.get("parameters")
        inputs = self.provenance.get("inputs")
        if (
            not isinstance(operation, str)
            or not isinstance(parameters, dict)
            or not isinstance(inputs, dict)
        ):
            raise TypeError("PlannedArtifact provenance is incomplete for invalidation")
        return plan_artifact(
            root,
            scope=self.ref.scope,
            assay=self.ref.assay,
            kind=self.ref.kind,
            operation=operation,
            parameters=parameters,
            inputs=inputs,
            execution_options=dict(self.execution_options),
            invalidate_cache=True,
            required_arrays=(
                self.required_arrays if required_arrays is None else required_arrays
            ),
            required_attributes=(
                self.required_attributes
                if required_attributes is None
                else required_attributes
            ),
            reuse_validator=(
                self.reuse_validator if reuse_validator is None else reuse_validator
            ),
        )


@dataclass(frozen=True, slots=True)
class ArrayRequirement:
    name: str
    shape: tuple[int | None, ...] | None = None
    dtype_kind: str | None = None
    dtype: Any | None = None

    def matches(self, group: zarr.Group) -> bool:
        from .types import as_zarr_array

        try:
            array = as_zarr_array(group[self.name], name=self.name)
        except (KeyError, TypeError):
            return False
        if self.shape is not None:
            if len(array.shape) != len(self.shape):
                return False
            if any(
                expected is not None and int(actual) != expected
                for actual, expected in zip(array.shape, self.shape, strict=True)
            ):
                return False
        if self.dtype_kind is not None:
            if np.dtype(array.dtype).kind != self.dtype_kind:
                return False
        if self.dtype is not None and np.dtype(array.dtype) != np.dtype(self.dtype):
            return False
        return True


@dataclass(frozen=True, slots=True)
class AttributeRequirement:
    name: str
    expected_types: tuple[type[Any], ...] | None = None
    predicate: Callable[[Any], bool] | None = None

    def matches(self, group: zarr.Group) -> bool:
        if self.name not in group.attrs:
            return False
        value = group.attrs[self.name]
        if self.expected_types is not None and not isinstance(
            value,
            self.expected_types,
        ):
            return False
        return self.predicate(value) if self.predicate is not None else True


def _array_requirements(
    required: tuple[str | ArrayRequirement, ...],
) -> tuple[ArrayRequirement, ...]:
    return tuple(
        requirement
        if isinstance(requirement, ArrayRequirement)
        else ArrayRequirement(str(requirement))
        for requirement in required
    )


def _attribute_requirements(
    required: tuple[str | AttributeRequirement, ...],
) -> tuple[AttributeRequirement, ...]:
    return tuple(
        requirement
        if isinstance(requirement, AttributeRequirement)
        else AttributeRequirement(str(requirement))
        for requirement in required
    )


def _requirements_met(
    group: zarr.Group, requirements: tuple[ArrayRequirement, ...]
) -> list[bool]:
    # Each check opens one array; object stores overlap them.
    return run_concurrently(
        [partial(requirement.matches, group) for requirement in requirements],
        workers=metadata_workers(group),
    )


def plan_artifact(
    root: zarr.Group,
    *,
    scope: ArtifactScope,
    kind: str,
    operation: str,
    parameters: dict[str, Any],
    inputs: dict[str, Any],
    execution_options: dict[str, Any],
    assay: str | None = None,
    invalidate_cache: bool = False,
    required_arrays: tuple[str | ArrayRequirement, ...] = (),
    required_attributes: tuple[str | AttributeRequirement, ...] = (),
    reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None = None,
) -> PlannedArtifact:
    provenance = make_provenance(
        operation=operation,
        parameters=parameters,
        inputs=inputs,
    )
    stored_execution_options = serialize_artifact_value(execution_options)
    if not isinstance(stored_execution_options, dict):
        raise TypeError("execution_options must serialize to a mapping")
    canonical_bytes(stored_execution_options)
    candidates = reusable_artifact_groups(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        provenance=provenance,
        invalidate_cache=invalidate_cache,
    )
    requirements = _array_requirements(required_arrays)
    attribute_requirements = _attribute_requirements(required_attributes)
    reused = None
    for candidate, group in candidates:
        if not all(_requirements_met(group, requirements)):
            continue
        if any(
            not requirement.matches(group) for requirement in attribute_requirements
        ):
            continue
        if reuse_validator is not None and not reuse_validator(
            candidate,
            group,
        ):
            continue
        reused = candidate
        break
    if reused is not None:
        return _record_plan(
            PlannedArtifact(
                ref=reused,
                provenance=provenance,
                execution_options=stored_execution_options,
                reused=True,
                required_arrays=required_arrays,
                required_attributes=required_attributes,
                reuse_validator=reuse_validator,
            )
        )
    # A random 256-bit ID does not collide; start_artifact refuses an existing path.
    ref = ArtifactRef(
        scope=scope,
        assay=assay,
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    return _record_plan(
        PlannedArtifact(
            ref=ref,
            provenance=provenance,
            execution_options=stored_execution_options,
            reused=False,
            required_arrays=required_arrays,
            required_attributes=required_attributes,
            reuse_validator=reuse_validator,
        )
    )


def start_artifact(root: zarr.Group, planned: PlannedArtifact) -> zarr.Group:
    if planned.reused:
        raise ValueError("Cannot start a reused artifact")
    path = artifact_path(planned.ref)
    from .. import __version__

    try:
        # One metadata write creates the group with its record.
        return root.create_group(
            path,
            attributes={
                "artifact_id": planned.ref.artifact_id,
                "kind": planned.ref.kind,
                "provenance": planned.provenance,
                "execution_options": planned.execution_options,
                "created_at_ns": time.time_ns(),
                "scarf_version": __version__,
                "complete": False,
            },
        )
    except (ContainsArrayError, ContainsGroupError) as exc:
        raise FileExistsError(f"Artifact path already exists: {path}") from exc


def finish_artifact(
    group: zarr.Group,
    planned: PlannedArtifact,
) -> None:
    if planned.reused:
        raise ValueError("Cannot finish a reused artifact")
    if group.attrs.get("complete") is not False:
        group.attrs["complete"] = False
    if (
        group.attrs.get("artifact_id") != planned.ref.artifact_id
        or group.attrs.get("kind") != planned.ref.kind
    ):
        raise ValueError("Artifact group does not match its creation plan")
    requirements = _array_requirements(planned.required_arrays)
    for requirement, met in zip(
        requirements, _requirements_met(group, requirements), strict=True
    ):
        if not met:
            raise ValueError(
                f"Artifact array {requirement.name!r} does not satisfy its contract"
            )
    for attribute_requirement in _attribute_requirements(planned.required_attributes):
        if not attribute_requirement.matches(group):
            raise ValueError(
                f"Artifact attribute {attribute_requirement.name!r} "
                "does not satisfy its contract"
            )
    if planned.reuse_validator is not None and not planned.reuse_validator(
        planned.ref,
        group,
    ):
        raise ValueError("Artifact payload does not satisfy its reuse contract")
    group.attrs["complete"] = True


def reused_artifact_group(
    root: zarr.Group,
    planned: PlannedArtifact,
) -> zarr.Group:
    from .artifacts import artifact_group

    if not planned.reused:
        raise ValueError("Artifact is not reused")
    return artifact_group(root, planned.ref)
