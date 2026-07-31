import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

__all__ = ["hto_demux"]

_NORMALIZATION_METHOD = "clr_per_hto"
_CLUSTERING_METHOD = "kmeans"
_KMEANS_INIT = "random"
_KMEANS_N_STARTS = 100
_CLUSTER_COUNT_RULE = "n_htos_plus_one"
_BACKGROUND_STATISTIC = "raw_mean"
_CUTOFF_DISTRIBUTION = "negative_binomial_nb2"
_POSITIVE_QUANTILE = 0.99
_CUTOFF_LOCATION = 0
_CUTOFF_COMPARISON = "strictly_greater"
_SINGLET_ASSIGNMENT = "clr_argmax"
_RESERVED_IDENTITIES = frozenset({"Negative", "Singlet", "Doublet"})


def _hto_demux_method() -> dict[str, object]:
    return {
        "normalization": _NORMALIZATION_METHOD,
        "clustering": {
            "method": _CLUSTERING_METHOD,
            "init": _KMEANS_INIT,
            "n_starts": _KMEANS_N_STARTS,
            "cluster_count": _CLUSTER_COUNT_RULE,
        },
        "background": _BACKGROUND_STATISTIC,
        "cutoff": {
            "distribution": _CUTOFF_DISTRIBUTION,
            "quantile": _POSITIVE_QUANTILE,
            "location": _CUTOFF_LOCATION,
            "comparison": _CUTOFF_COMPARISON,
        },
        "singlet_assignment": _SINGLET_ASSIGNMENT,
    }


def _validated_hto_counts(hto_counts: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(hto_counts, pd.DataFrame):
        raise TypeError("hto_counts must be a pandas DataFrame")
    if hto_counts.shape[1] == 0:
        raise ValueError("hto_counts must contain at least one HTO")
    if not hto_counts.index.is_unique:
        raise ValueError("hto_counts cell index must be unique")
    if not hto_counts.columns.is_unique:
        raise ValueError("HTO IDs must be unique")

    hto_names = hto_counts.columns.tolist()
    if any(not isinstance(name, str) or not name.strip() for name in hto_names):
        raise ValueError("HTO IDs must be non-empty strings")
    reserved = sorted(_RESERVED_IDENTITIES.intersection(hto_names))
    if reserved:
        raise ValueError(
            "HTO IDs conflict with reserved identity labels: "
            + ", ".join(repr(name) for name in reserved)
        )
    if not all(is_numeric_dtype(dtype) for dtype in hto_counts.dtypes):
        raise TypeError("hto_counts must contain only numeric raw counts")

    raw_counts = hto_counts.to_numpy()
    if np.iscomplexobj(raw_counts):
        raise TypeError("hto_counts must contain only real numeric raw counts")
    counts = raw_counts.astype(float)
    if not np.all(np.isfinite(counts)):
        raise ValueError("hto_counts must contain only finite raw counts")
    if np.any(counts < 0):
        raise ValueError("hto_counts must contain only nonnegative raw counts")
    if np.any(counts != np.floor(counts)):
        raise ValueError("hto_counts must contain integer-valued raw counts")

    required_cells = hto_counts.shape[1] + 1
    if hto_counts.shape[0] < required_cells:
        raise ValueError(
            f"HTO demultiplexing requires at least {required_cells} selected cells"
        )

    empty_htos = [
        hto_names[index]
        for index in np.flatnonzero(np.all(counts == 0, axis=0)).tolist()
    ]
    if empty_htos:
        raise ValueError(
            "HTOs with no positive counts cannot be demultiplexed: "
            + ", ".join(repr(name) for name in empty_htos)
        )
    return pd.DataFrame(counts, index=hto_counts.index, columns=hto_counts.columns)


def _clr_normalize(hto_counts: pd.DataFrame) -> pd.DataFrame:
    scale = np.exp(np.log1p(hto_counts).sum(axis=0) / len(hto_counts))
    normalized = np.log1p(hto_counts / scale)
    if not np.all(np.isfinite(normalized.to_numpy())):
        raise ValueError("CLR normalization produced non-finite HTO values")
    return normalized


def _cluster_labels(normalized: pd.DataFrame, random_seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    n_centers = normalized.shape[1] + 1
    unique_profiles = np.unique(normalized.to_numpy(), axis=0).shape[0]
    if unique_profiles < n_centers:
        raise ValueError(
            "HTO demultiplexing requires at least "
            f"{n_centers} distinct normalized cell profiles; found {unique_profiles}"
        )
    labels = np.asarray(
        KMeans(
            n_clusters=n_centers,
            init=_KMEANS_INIT,
            n_init=_KMEANS_N_STARTS,
            random_state=random_seed,
        ).fit_predict(normalized)
    )
    occupied = np.unique(labels).size
    if occupied != n_centers:
        raise ValueError(
            f"HTO clustering produced {occupied} occupied clusters; "
            f"expected {n_centers}"
        )
    return labels


def _background_clusters(
    hto_counts: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.Series:
    cluster_means = hto_counts.groupby(cluster_labels, sort=True).mean()
    return cluster_means.idxmin(axis=0)


def _fit_negative_binomial_parameters(
    values: np.ndarray,
    hto_name: str,
) -> tuple[float, float]:
    from statsmodels.discrete.discrete_model import NegativeBinomial

    exog = np.ones((len(values), 1), dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = NegativeBinomial(
                values,
                exog,
                loglike_method="nb2",
            ).fit(
                start_params=np.asarray([1.0, 1.0]),
                disp=0,
                full_output=True,
            )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Negative-binomial background fit failed for HTO {hto_name!r}"
        ) from exc

    fit_details = getattr(fit, "mle_retvals", None)
    convergence = (
        fit_details.get("converged") if isinstance(fit_details, Mapping) else None
    )
    if not isinstance(convergence, bool | np.bool_) or not bool(convergence):
        raise ValueError(
            f"Negative-binomial background fit did not converge for HTO {hto_name!r}"
        )
    try:
        parameters = np.asarray(fit.params, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Negative-binomial background fit returned invalid parameters "
            f"for HTO {hto_name!r}"
        ) from exc
    if parameters.size != 2:
        raise ValueError(
            f"Negative-binomial background fit returned invalid parameters "
            f"for HTO {hto_name!r}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        mu = float(np.exp(parameters[0]))
    alpha = float(parameters[-1])
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError(
            f"Negative-binomial background fit returned invalid mean "
            f"for HTO {hto_name!r}"
        )
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            f"Negative-binomial background fit returned invalid dispersion "
            f"for HTO {hto_name!r}"
        )
    return mu, alpha


def _negative_binomial_cutoff(
    mu: float,
    alpha: float,
    quantile: float = _POSITIVE_QUANTILE,
) -> int:
    from scipy.stats import nbinom

    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("Negative-binomial mean must be finite and greater than 0")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            "Negative-binomial dispersion must be finite and greater than 0"
        )
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("Negative-binomial quantile must be between 0 and 1")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        size = 1 / alpha
        probability = 1 / (1 + alpha * mu)
    if (
        not np.isfinite(size)
        or size <= 0
        or not np.isfinite(probability)
        or not 0 < probability <= 1
    ):
        raise ValueError(
            "Negative-binomial parameters produced an invalid distribution"
        )

    cutoff = float(nbinom.ppf(quantile, n=size, p=probability))
    if not np.isfinite(cutoff) or not cutoff.is_integer():
        raise ValueError("Negative-binomial cutoff must be a finite integer")
    return int(cutoff)


def _positive_hto_calls(
    hto_counts: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    background_clusters = _background_clusters(hto_counts, cluster_labels)
    cutoffs: dict[str, int] = {}
    for hto_name in hto_counts.columns:
        background = hto_counts.loc[
            cluster_labels == background_clusters[hto_name],
            hto_name,
        ].to_numpy()
        if len(background) < 2:
            raise ValueError(
                f"Background cluster for HTO {hto_name!r} must contain "
                "at least two cells"
            )
        if not np.any(background > 0):
            raise ValueError(
                f"Background cluster for HTO {hto_name!r} contains only zero counts"
            )
        mu, alpha = _fit_negative_binomial_parameters(background, hto_name)
        try:
            cutoffs[hto_name] = _negative_binomial_cutoff(mu, alpha)
        except ValueError as exc:
            raise ValueError(
                f"Negative-binomial cutoff is invalid for HTO {hto_name!r}"
            ) from exc
    return hto_counts > pd.Series(cutoffs)


def _classify_hto_identities(
    normalized: pd.DataFrame,
    positive_calls: pd.DataFrame,
) -> pd.Series:
    positive_count = positive_calls.sum(axis=1)
    identities = positive_count.map(
        lambda count: (
            "Negative" if count == 0 else "Singlet" if count == 1 else "Doublet"
        )
    )
    singlets = positive_count == 1
    identities.loc[singlets] = normalized.loc[singlets].idxmax(axis=1)
    return identities


def hto_demux(
    hto_counts: pd.DataFrame,
    *,
    random_seed: int = 0,
) -> pd.Series:
    """Assigns HTO identity to each cell based on the HTO count distribution.
    The algorithm is adapted from the Seurat package's HTOdemux function [Satija15]_.

    Args:
        hto_counts: A dataframe containing the raw HTO counts for each cell.

    Returns:
        A series containing the HTO identity for each cell.
    """
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer")
    counts = _validated_hto_counts(hto_counts)
    normalized = _clr_normalize(counts)
    cluster_labels = _cluster_labels(normalized, random_seed)
    positive_calls = _positive_hto_calls(counts, cluster_labels)
    return _classify_hto_identities(normalized, positive_calls)
