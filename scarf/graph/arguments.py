from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field, fields
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import numpy as np
import zarr

from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    PlannedArtifact,
    plan_artifact,
)
from ..storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    make_provenance,
    provenance_hash,
    serialize_artifact_value,
)

type ArgumentRole = Literal["input", "parameter", "execution"]


def parameter(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
) -> Any:
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both default and default_factory")
    if default_factory is not MISSING:
        return field(
            default_factory=default_factory,
            metadata={"argument_role": "parameter"},
        )
    if default is not MISSING:
        return field(default=default, metadata={"argument_role": "parameter"})
    return field(metadata={"argument_role": "parameter"})


def execution(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
) -> Any:
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both default and default_factory")
    if default_factory is not MISSING:
        return field(
            default_factory=default_factory,
            metadata={"argument_role": "execution"},
        )
    if default is not MISSING:
        return field(default=default, metadata={"argument_role": "execution"})
    return field(metadata={"argument_role": "execution"})


def artifact_input(
    default: Any = MISSING,
    *,
    default_factory: Any = MISSING,
) -> Any:
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both default and default_factory")
    if default_factory is not MISSING:
        return field(
            default_factory=default_factory,
            metadata={"argument_role": "input"},
        )
    if default is not MISSING:
        return field(default=default, metadata={"argument_role": "input"})
    return field(metadata={"argument_role": "input"})


@dataclass(frozen=True, slots=True)
class ArgumentRecord:
    parameters: dict[str, Any]
    execution_options: dict[str, Any]
    inputs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OperationArguments:
    operation: ClassVar[str]
    artifact_kind: ClassVar[str]

    def to_record(self) -> ArgumentRecord:
        partitions: dict[ArgumentRole, dict[str, Any]] = {
            "input": {},
            "parameter": {},
            "execution": {},
        }
        for model_field in fields(self):
            role = model_field.metadata.get("argument_role")
            if role not in partitions:
                raise TypeError(
                    f"{type(self).__name__}.{model_field.name} has no argument role"
                )
            partitions[role][model_field.name] = serialize_artifact_value(
                getattr(self, model_field.name)
            )
        return ArgumentRecord(
            parameters=partitions["parameter"],
            execution_options=partitions["execution"],
            inputs=partitions["input"],
        )

    def provenance(self) -> dict[str, Any]:
        record = self.to_record()
        return make_provenance(
            operation=self.operation,
            parameters=record.parameters,
            inputs=record.inputs,
        )

    def provenance_hash(self) -> str:
        return provenance_hash(self.provenance())

    def plan(
        self,
        root: zarr.Group,
        *,
        scope: ArtifactScope,
        assay: str | None = None,
        invalidate_cache: bool = False,
        required_arrays: tuple[str | ArrayRequirement, ...] = (),
        required_attributes: tuple[str | AttributeRequirement, ...] = (),
        reuse_validator: Callable[[ArtifactRef, zarr.Group], bool] | None = None,
    ) -> PlannedArtifact:
        record = self.to_record()
        return plan_artifact(
            root,
            scope=scope,
            assay=assay,
            kind=self.artifact_kind,
            operation=self.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=required_arrays,
            required_attributes=required_attributes,
            reuse_validator=reuse_validator,
        )


@dataclass(frozen=True, slots=True)
class NormalizationArguments(OperationArguments):
    operation: ClassVar[str] = "run_normalization"
    artifact_kind: ClassVar[str] = "normalized"

    from_assay: str = execution()
    cell_key: str = execution()
    feat_key: str = execution()
    cell_selection: ArtifactRef = artifact_input()
    feature_selection: ArtifactRef = artifact_input()
    normalization_method: Callable[..., Any] | str = parameter()
    size_factor: float | None = parameter()
    log_transform: bool = parameter()
    renormalize_subset: bool = parameter()
    batch_size: int = execution()
    update_state: bool = execution()
    local_cache: bool | str = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class FeatureScalingArguments(OperationArguments):
    operation: ClassVar[str] = "calculate_feature_scaling"
    artifact_kind: ClassVar[str] = "feature_scaling"

    normalized: ArtifactRef = artifact_input()
    enabled: bool = parameter()
    calculation_batch_size: int | None = parameter()
    batch_size: int = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class PcaArguments(OperationArguments):
    operation: ClassVar[str] = "run_pca"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = artifact_input()
    feature_scaling: ArtifactRef = artifact_input()
    pca_cell_selection: ArtifactRef = artifact_input()
    pca_cell_key: str = execution()
    dims: int = parameter()
    feat_scaling: bool = parameter()
    batch_size: int = parameter()
    show_elbow_plot: bool = execution()
    update_state: bool = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class LsiArguments(OperationArguments):
    operation: ClassVar[str] = "run_lsi"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = artifact_input()
    feature_scaling: ArtifactRef = artifact_input()
    dims: int = parameter()
    skip_first: bool = parameter()
    rand_state: int = parameter()
    batch_size: int = execution()
    update_state: bool = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class CustomReductionArguments(OperationArguments):
    operation: ClassVar[str] = "run_custom_reduction"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = artifact_input()
    feature_scaling: ArtifactRef = artifact_input()
    loadings: np.ndarray = artifact_input()
    update_state: bool = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class HarmonyArguments(OperationArguments):
    operation: ClassVar[str] = "run_harmony"
    artifact_kind: ClassVar[str] = "batch_correction"

    reduction: ArtifactRef = artifact_input()
    batch_values: ArtifactRef = artifact_input()
    batch_columns: tuple[str, ...] = parameter()
    harmony_parameters: Mapping[str, Any] = parameter()
    batch_size: int = execution()
    force_refit: bool = execution()
    invalidate_cache: bool = execution(False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "harmony_parameters",
            MappingProxyType(dict(self.harmony_parameters)),
        )


@dataclass(frozen=True, slots=True)
class AnnIndexArguments(OperationArguments):
    operation: ClassVar[str] = "build_ann_index"
    artifact_kind: ClassVar[str] = "ann_index"

    coordinates: ArtifactRef = artifact_input()
    ann_metric: str = parameter()
    ann_efc: int = parameter()
    ann_ef: int = parameter()
    ann_m: int = parameter()
    rand_state: int = parameter()
    ann_parallel: bool = parameter()
    parallel_threads: int | None = parameter()
    batch_size: int = execution()
    ann_index_fetcher: Callable[..., Any] | None = artifact_input()
    ann_index_saver: Callable[..., Any] | None = execution()
    local_cache: bool | str = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class NeighborQueryArguments(OperationArguments):
    operation: ClassVar[str] = "query_neighbors"
    artifact_kind: ClassVar[str] = "neighbors"

    ann_index: ArtifactRef = artifact_input()
    coordinates: ArtifactRef = artifact_input()
    k: int = parameter()
    batch_size: int = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class ConnectivityMapArguments(OperationArguments):
    operation: ClassVar[str] = "build_connectivity_map"
    artifact_kind: ClassVar[str] = "connectivity_map"

    neighbors: ArtifactRef = artifact_input()
    local_connectivity: float = parameter()
    bandwidth: float = parameter()
    batch_size: int = execution()
    invalidate_cache: bool = execution(False)


@dataclass(frozen=True, slots=True)
class EmbeddingInitializationArguments(OperationArguments):
    operation: ClassVar[str] = "build_embedding_initialization"
    artifact_kind: ClassVar[str] = "embedding_initialization"

    reduction: ArtifactRef = artifact_input()
    n_centroids: int = parameter()
    rand_state: int = parameter()
    batch_size: int = parameter()
    invalidate_cache: bool = execution(False)


MAKE_GRAPH_ARGUMENT_OWNERS = {
    "from_assay": "normalization",
    "cell_key": "normalization",
    "feat_key": "normalization",
    "pca_cell_key": "reduction",
    "reduction_method": "reduction",
    "dims": "reduction",
    "k": "neighbors",
    "ann_metric": "ann_index",
    "ann_efc": "ann_index",
    "ann_ef": "ann_index",
    "ann_m": "ann_index",
    "ann_parallel": "ann_index",
    "rand_state": "reduction_ann_index_and_embedding_initialization",
    "n_centroids": "embedding_initialization",
    "batch_size": "stage_specific_parameter_or_execution_option",
    "log_transform": "normalization",
    "renormalize_subset": "normalization",
    "local_connectivity": "connectivity_map",
    "bandwidth": "connectivity_map",
    "update_keys": "normalization",
    "return_ann_object": "facade",
    "custom_loadings": "reduction",
    "feat_scaling": "reduction",
    "lsi_skip_first": "reduction",
    "harmonize": "facade",
    "batch_columns": "harmony",
    "show_elbow_plot": "reduction",
    "ann_index_fetcher": "ann_index",
    "ann_index_saver": "ann_index",
    "local_cache": "stage_execution",
    "harmony_params": "harmony",
    "_force_harmony_refit": "harmony",
}
