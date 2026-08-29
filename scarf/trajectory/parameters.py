"""Shared parameter resolution for trajectory algorithms and artifacts."""

from collections.abc import Mapping
from typing import Any


AGGREGATION_ANN_DEFAULTS: tuple[tuple[str, str | int], ...] = (
    ("space", "l2"),
    ("ef_construction", 80),
    ("M", 50),
    ("random_seed", 444),
    ("ef", 80),
    ("num_threads", 1),
)
AGGREGATION_ANN_STATIC_PARAMETER_NAMES = frozenset(
    name for name, _value in AGGREGATION_ANN_DEFAULTS
)
AGGREGATION_ANN_DERIVED_PARAMETER_NAMES = frozenset({"dim", "max_elements"})
AGGREGATION_ANN_PARAMETER_NAMES = (
    AGGREGATION_ANN_STATIC_PARAMETER_NAMES | AGGREGATION_ANN_DERIVED_PARAMETER_NAMES
)


def resolve_aggregation_ann_params(
    ann_params: Mapping[str, Any] | None,
    *,
    dim: int,
) -> dict[str, Any]:
    """Resolve result-affecting static HNSW defaults before artifact planning."""
    if ann_params is not None and not isinstance(ann_params, Mapping):
        raise TypeError("ann_params must be a mapping or None")
    resolved = dict(AGGREGATION_ANN_DEFAULTS)
    if ann_params is not None:
        resolved.update(ann_params)
    if "dim" in resolved and resolved["dim"] != dim:
        raise ValueError("ann_params.dim must match the effective bin count")
    resolved["dim"] = dim
    return resolved
