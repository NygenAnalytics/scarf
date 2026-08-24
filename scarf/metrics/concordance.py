from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd


def label_concordance_score(
    label_sets: Sequence[np.ndarray],
    metric: Literal["ari", "nmi"] = "ari",
) -> float:
    """Compare two label partitions using ARI or NMI.

    Args:
        label_sets: Two arrays of labels to compare.
        metric: Either ``"ari"`` or ``"nmi"``.

    Returns:
        Label agreement. ARI ranges from -1 to 1 and NMI from 0 to 1.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if len(label_sets) != 2:
        raise ValueError("Exactly two label arrays are required")

    first = np.asarray(label_sets[0])
    second = np.asarray(label_sets[1])
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Label arrays must be one-dimensional")
    if len(first) != len(second):
        raise ValueError("Label arrays must have matching lengths")
    if pd.isna(first).any() or pd.isna(second).any():
        raise ValueError("Label arrays must not contain missing values")

    if metric == "ari":
        return float(adjusted_rand_score(first, second))
    if metric == "nmi":
        return float(normalized_mutual_info_score(first, second))
    raise ValueError(f"Metric {metric!r} is not one of 'ari' or 'nmi'")
