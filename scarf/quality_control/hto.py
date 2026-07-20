from typing import Any

import numpy as np
import pandas as pd

__all__ = ["hto_demux"]


def hto_demux(hto_counts: pd.DataFrame) -> pd.Series:
    """Assigns HTO identity to each cell based on the HTO count distribution.
    The algorithm is adapted from the Seurat package's HTOdemux function [Satija15]_.

    Args:
        hto_counts: A dataframe containing the raw HTO counts for each cell.

    Returns:
        A series containing the HTO identity for each cell.
    """
    from scipy.stats import nbinom
    from sklearn.cluster import KMeans
    from statsmodels.discrete.discrete_model import NegativeBinomial

    def clr_normalize(df: pd.DataFrame) -> pd.DataFrame:
        f = np.exp(np.log1p(df).sum(axis=0) / len(df))
        return np.log1p(df / f)

    def calc_cluster_labels(
        df: pd.DataFrame, n_centers: int | None = None, n_starts: int = 100
    ) -> np.ndarray:
        if n_centers is None:
            n_centers = df.shape[1] + 1
        kmeans = KMeans(n_clusters=n_centers, init="random", n_init=n_starts)
        kmeans.fit(df)
        return np.asarray(kmeans.labels_)

    def calc_cluster_avg_exp(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        df["cluster"] = calc_cluster_labels(df)
        return df["cluster"], df.groupby("cluster").mean()

    def get_background_cutoff(vals: np.ndarray, quantile: float = 0.99) -> int:
        fit = NegativeBinomial(vals, np.ones_like(vals)).fit(
            start_params=[1, 1], disp=0
        )
        mu = np.exp(fit.params[0])
        p = 1 / (1 + np.exp(fit.params[0]) * fit.params[1])
        n = np.exp(fit.params[0]) * p / (1 - p)
        dist = nbinom(n=n, p=p, loc=mu)
        return int(round(dist.ppf(quantile)))

    def discretize_counts(
        df: pd.DataFrame, clust_labels: pd.Series, clust_exp: pd.DataFrame
    ) -> pd.DataFrame:
        min_clust = clust_exp.idxmin()
        cutoffs: dict[Any, int] = {}
        for hto in df:
            bg_values = df[hto][clust_labels == min_clust[hto]].values
            cutoffs[hto] = get_background_cutoff(bg_values)
        cutoff_series = pd.Series(cutoffs)
        return df > cutoff_series

    def identity_renamer(x: int) -> str:
        if x == 0:
            return "Negative"
        elif x == 1:
            return "Singlet"
        else:
            return "Doublet"

    cluster_labels, avg_exp = calc_cluster_avg_exp(clr_normalize(hto_counts))
    # Seurat does the following check and hard stops the process if the assertion fails
    assert any(avg_exp.sum(axis=1) == 0) is False
    hto_discrete = discretize_counts(hto_counts, cluster_labels, avg_exp)
    g_class = hto_discrete.sum(axis=1).apply(identity_renamer)
    singlet_ident = hto_counts[g_class == "Singlet"].idxmax(axis=1)
    g_class[singlet_ident.index] = singlet_ident
    return g_class
