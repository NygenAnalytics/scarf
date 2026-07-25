from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import numpy as np

from ..storage.artifacts import (
    ArtifactRef,
    callable_identity,
    fingerprint_array,
    make_provenance,
    provenance_hash,
)

type ArgumentRole = Literal["input", "parameter", "execution"]


@dataclass(frozen=True, slots=True)
class ArgumentRecord:
    parameters: dict[str, Any]
    execution_options: dict[str, Any]
    inputs: dict[str, Any]


def _serialize_argument(name: str, value: Any) -> Any:
    if isinstance(value, ArtifactRef):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return {"value_fingerprint": fingerprint_array(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {"external_hook": True, **callable_identity(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_argument(str(key), item) for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_serialize_argument(name, item) for item in value]
    return value


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
            partitions[role][model_field.name] = _serialize_argument(
                model_field.name,
                getattr(self, model_field.name),
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


@dataclass(frozen=True, slots=True)
class NormalizationArguments(OperationArguments):
    operation: ClassVar[str] = "run_normalization"
    artifact_kind: ClassVar[str] = "normalized"

    from_assay: str = field(metadata={"argument_role": "execution"})
    cell_key: str = field(metadata={"argument_role": "execution"})
    feat_key: str = field(metadata={"argument_role": "execution"})
    cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    normalization_method: Callable[..., Any] | str = field(
        metadata={"argument_role": "parameter"}
    )
    size_factor: float | None = field(metadata={"argument_role": "parameter"})
    log_transform: bool = field(metadata={"argument_role": "parameter"})
    renormalize_subset: bool = field(metadata={"argument_role": "parameter"})
    batch_size: int = field(metadata={"argument_role": "execution"})
    update_state: bool = field(metadata={"argument_role": "execution"})
    local_cache: bool | str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


@dataclass(frozen=True, slots=True)
class FeatureScalingArguments(OperationArguments):
    operation: ClassVar[str] = "calculate_feature_scaling"
    artifact_kind: ClassVar[str] = "feature_scaling"

    normalized: ArtifactRef = field(metadata={"argument_role": "input"})
    enabled: bool = field(metadata={"argument_role": "parameter"})
    calculation_batch_size: int | None = field(metadata={"argument_role": "parameter"})
    batch_size: int = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


@dataclass(frozen=True, slots=True)
class PcaArguments(OperationArguments):
    operation: ClassVar[str] = "run_pca"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_scaling: ArtifactRef = field(metadata={"argument_role": "input"})
    pca_cell_selection: ArtifactRef = field(metadata={"argument_role": "input"})
    pca_cell_key: str = field(metadata={"argument_role": "execution"})
    dims: int = field(metadata={"argument_role": "parameter"})
    feat_scaling: bool = field(metadata={"argument_role": "parameter"})
    batch_size: int = field(metadata={"argument_role": "parameter"})
    show_elbow_plot: bool = field(metadata={"argument_role": "execution"})
    update_state: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


@dataclass(frozen=True, slots=True)
class LsiArguments(OperationArguments):
    operation: ClassVar[str] = "run_lsi"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_scaling: ArtifactRef = field(metadata={"argument_role": "input"})
    dims: int = field(metadata={"argument_role": "parameter"})
    skip_first: bool = field(metadata={"argument_role": "parameter"})
    rand_state: int = field(metadata={"argument_role": "parameter"})
    batch_size: int = field(metadata={"argument_role": "execution"})
    update_state: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


@dataclass(frozen=True, slots=True)
class CustomReductionArguments(OperationArguments):
    operation: ClassVar[str] = "run_custom_reduction"
    artifact_kind: ClassVar[str] = "reduction"

    normalized: ArtifactRef = field(metadata={"argument_role": "input"})
    feature_scaling: ArtifactRef = field(metadata={"argument_role": "input"})
    loadings: np.ndarray = field(metadata={"argument_role": "input"})
    update_state: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


@dataclass(frozen=True, slots=True)
class HarmonyArguments(OperationArguments):
    operation: ClassVar[str] = "run_harmony"
    artifact_kind: ClassVar[str] = "batch_correction"

    reduction: ArtifactRef = field(metadata={"argument_role": "input"})
    batch_values: ArtifactRef = field(metadata={"argument_role": "input"})
    batch_columns: tuple[str, ...] = field(metadata={"argument_role": "parameter"})
    harmony_parameters: Mapping[str, Any] = field(
        metadata={"argument_role": "parameter"}
    )
    batch_size: int = field(metadata={"argument_role": "execution"})
    force_refit: bool = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )

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

    coordinates: ArtifactRef = field(metadata={"argument_role": "input"})
    ann_metric: str = field(metadata={"argument_role": "parameter"})
    ann_efc: int = field(metadata={"argument_role": "parameter"})
    ann_ef: int = field(metadata={"argument_role": "parameter"})
    ann_m: int = field(metadata={"argument_role": "parameter"})
    rand_state: int = field(metadata={"argument_role": "parameter"})
    ann_parallel: bool = field(metadata={"argument_role": "parameter"})
    parallel_threads: int | None = field(metadata={"argument_role": "parameter"})
    batch_size: int = field(metadata={"argument_role": "execution"})
    ann_index_fetcher: Callable[..., Any] | None = field(
        metadata={"argument_role": "input"}
    )
    ann_index_saver: Callable[..., Any] | None = field(
        metadata={"argument_role": "execution"}
    )
    local_cache: bool | str = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


@dataclass(frozen=True, slots=True)
class NeighborQueryArguments(OperationArguments):
    operation: ClassVar[str] = "query_neighbors"
    artifact_kind: ClassVar[str] = "neighbors"

    ann_index: ArtifactRef = field(metadata={"argument_role": "input"})
    coordinates: ArtifactRef = field(metadata={"argument_role": "input"})
    k: int = field(metadata={"argument_role": "parameter"})
    batch_size: int = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


@dataclass(frozen=True, slots=True)
class ConnectivityMapArguments(OperationArguments):
    operation: ClassVar[str] = "build_connectivity_map"
    artifact_kind: ClassVar[str] = "connectivity_map"

    neighbors: ArtifactRef = field(metadata={"argument_role": "input"})
    local_connectivity: float = field(metadata={"argument_role": "parameter"})
    bandwidth: float = field(metadata={"argument_role": "parameter"})
    batch_size: int = field(metadata={"argument_role": "execution"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


@dataclass(frozen=True, slots=True)
class EmbeddingInitializationArguments(OperationArguments):
    operation: ClassVar[str] = "build_embedding_initialization"
    artifact_kind: ClassVar[str] = "embedding_initialization"

    reduction: ArtifactRef = field(metadata={"argument_role": "input"})
    n_centroids: int = field(metadata={"argument_role": "parameter"})
    rand_state: int = field(metadata={"argument_role": "parameter"})
    batch_size: int = field(metadata={"argument_role": "parameter"})
    invalidate_cache: bool = field(
        default=False,
        metadata={"argument_role": "execution"},
    )


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
