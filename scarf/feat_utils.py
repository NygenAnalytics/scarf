"""Utility functions for features."""


import numpy as np
import pandas as pd

__all__ = ["fit_lowess", "binned_sampling", "hto_demux"]


def fit_lowess(a, b, n_bins: int, lowess_frac: float) -> np.ndarray:
    """Fits a LOWESS (Locally Weighted Scatterplot Smoothing) curve.

    Args:
        a:
        b:
        n_bins:
        lowess_frac:

    Returns:
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    stats = pd.DataFrame({"a": a, "b": b}).apply(np.log)
    bin_edges = np.histogram(stats.a, bins=n_bins)[1]
    bin_edges[-1] += 0.1  # For including last gene
    bin_idx = []
    for i in range(n_bins):
        idx = pd.Series((stats.a >= bin_edges[i]) & (stats.a < bin_edges[i + 1]))
        if sum(idx) > 0:
            bin_idx.append(list(idx[idx].index))
    bin_vals = []
    for idx in bin_idx:
        temp_stat = stats.reindex(idx)
        temp_gene = temp_stat.idxmin().b
        bin_vals.append([temp_stat.b[temp_gene], temp_stat.a[temp_gene]])
    bin_vals = np.array(bin_vals).T
    bin_cor_fac = lowess(
        bin_vals[0], bin_vals[1], return_sorted=False, frac=lowess_frac, it=100
    ).T
    fixed_var = {}
    for bcf, indices in zip(bin_cor_fac, bin_idx):
        for idx in indices:
            fixed_var[idx] = np.e ** (stats.b[idx] - bcf)
    return np.array([fixed_var[x] for x in range(len(a))])


def binned_sampling(
    values: pd.Series,
    feature_list: list[str],
    ctrl_size: int,
    n_bins: int,
    rand_seed: int,
) -> list[str]:
    """Score a set of genes [Satija15]_. The score is the average expression of
    a set of genes subtracted with the average expression of a reference set of
    genes. The reference set is randomly sampled from the `gene_pool` for each
    binned expression value.

    This reproduces the approach in Seurat [Satija15]_ and has been implemented
    for Scanpy by Davide Cittaro.

    This function is adapted from Scanpy's `score_genes`.

    Args:
        values: The values for the features.
        feature_list: The list of features to use for score calculation.
        ctrl_size: Number of reference features to be sampled from each bin.
        n_bins: Number of bins for sampling.
        rand_seed: The seed to use for the random number generation.

    Returns:
        A list of sampled features.
    """
    n_items = int(np.round(len(values) / (n_bins - 1)))
    feature_list = set(feature_list)
    # Made following more linter friendly
    # obs_cut = obs_avg.rank(method='min') // n_items
    obs_cut: pd.Series = values.fillna(0).rank(method="min").divide(n_items).astype(int)

    control_genes = set()
    for cut in np.unique(obs_cut[list(feature_list)]):
        sub_obs = obs_cut[obs_cut == cut]
        if len(sub_obs) == 0:
            continue
        if len(sub_obs) < ctrl_size:
            sample_size = len(sub_obs)
        else:
            sample_size = ctrl_size
        r_genes = sub_obs.sample(n=sample_size, random_state=rand_seed).index
        control_genes.update(set(r_genes))
    return list(control_genes - feature_list)


def hto_demux(hto_counts: pd.DataFrame) -> pd.Series:
    """Assigns HTO identity to each cell based on the HTO count distribution.
    The algorithm is adapted from the Seurat package's HTOdemux function [Satija15]_.

    Args:
        hto_counts: A dataframe containing the raw HTO counts for each cell.

    Returns:
        A series containing the HTO identity for each cell.
    """
    from sklearn.cluster import KMeans
    from statsmodels.discrete.discrete_model import NegativeBinomial
    from scipy.stats import nbinom

    def clr_normalize(df: pd.DataFrame) -> pd.DataFrame:
        f = np.exp(np.log1p(df).sum(axis=0) / len(df))
        return np.log1p(df / f)

    def calc_cluster_labels(
        df: pd.DataFrame, n_centers: int | None = None, n_starts: int = 100
    ):
        if n_centers is None:
            n_centers = df.shape[1] + 1
        kmeans = KMeans(n_clusters=n_centers, init="random", n_init=n_starts)
        kmeans.fit(df)
        return kmeans.labels_

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
        return round(dist.ppf(quantile))

    def discretize_counts(
        df: pd.DataFrame, clust_labels: pd.Series, clust_exp: pd.DataFrame
    ) -> pd.DataFrame:
        min_clust = clust_exp.idxmin()
        cutoffs = {}
        for hto in df:
            bg_values = df[hto][clust_labels == min_clust[hto]].values
            cutoffs[hto] = get_background_cutoff(bg_values)
        cutoffs = pd.Series(cutoffs)
        return df > cutoffs

    def identity_renamer(x: int):
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
