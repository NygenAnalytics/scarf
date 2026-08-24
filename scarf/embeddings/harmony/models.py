from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


type ClusterFn = str | Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class HarmonyResult:
    """Converged Harmony state required for portable reference mapping."""

    original: np.ndarray
    corrected: np.ndarray
    assignments: np.ndarray
    centroids: np.ndarray
    sigma: np.ndarray
    ridge: np.ndarray
    batch_columns: tuple[str, ...]
    batch_levels: tuple[tuple[str, ...], ...]
    parameters: dict[str, object]
