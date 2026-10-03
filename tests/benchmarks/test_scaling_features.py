"""Benchmarks of feature statistics, marker writes, HTO calls, and HVG trends.

Each benchmark times the function a DataStore operation calls on inputs shaped
like production data, and checks its value against an independent oracle, so
it doubles as a test of that function at the smoke size.
"""

import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from . import inputs
from .harness import Ladder

pytestmark = pytest.mark.benchmark

# Group labels read from a categorical cell column arrive as Python strings.
GROUP_NAMES = np.array(["T cell", "B cell", "NK cell", "Monocyte"], dtype=object)
# Hashtags of one multiplexed run.
HTO_NAMES = [f"HTO_{index}" for index in range(8)]
# Log-normal noise around the variance trend of the HVG benchmark.
TREND_NOISE = 0.3


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment, written out."""
    order = np.argsort(p_values, kind="stable")
    scaled = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    return adjusted


@lru_cache(maxsize=4)
def _labels_with_missing(n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the label of each cell, one in a hundred missing, and the valid mask."""
    rng = np.random.default_rng(inputs.SEED)
    labels = GROUP_NAMES[rng.integers(0, len(GROUP_NAMES), n_cells)]
    missing = rng.random(n_cells) < 0.01
    placeholders = np.array([None, np.nan, "", "   "], dtype=object)
    labels[missing] = placeholders[rng.integers(0, 4, int(missing.sum()))]
    return labels, ~missing


def test_valid_category_mask(bench) -> None:
    from scarf.metadata.selection import valid_category_mask

    def make(n_cells: int):
        labels, _valid = _labels_with_missing(n_cells)
        return lambda: valid_category_mask(labels)

    def check(n_cells: int, mask) -> None:
        # Missing and blank labels were planted at known cells.
        _labels, valid = _labels_with_missing(n_cells)
        np.testing.assert_array_equal(mask, valid)

    # compare_group_distributions runs this mask three times for every key.
    ladder = Ladder(sizes=(250_000, 500_000, 1_000_000, 2_000_000), smoke=2_000)
    bench("statistics.valid_category_mask", make, ladder, check=check)


@lru_cache(maxsize=4)
def _grouped_values(n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-inflated values of one gene and the group label of each cell."""
    rng = np.random.default_rng(inputs.SEED)
    codes = rng.integers(0, len(GROUP_NAMES), n_cells)
    detected = rng.random(n_cells) < 0.6
    values = rng.gamma(2.0, 1.0 + 0.2 * codes) * detected
    return values, GROUP_NAMES[codes]


@pytest.mark.filterwarnings("ignore:Cell-level statistical testing")
def test_group_distribution_comparison(bench) -> None:
    from scarf.features.statistical import compare_group_distributions

    def make(n_cells: int):
        values, labels = _grouped_values(n_cells)
        # run_statistical_testing passes the resolved group order and adjusts
        # p-values across keys itself.
        return lambda: compare_group_distributions(
            values,
            labels,
            test="kruskal_wallis",
            posthoc="dunn",
            adjustment="none",
            group_order=list(GROUP_NAMES),
        )

    def check(n_cells: int, result) -> None:
        values, labels = _grouped_values(n_cells)
        members = [labels == name for name in GROUP_NAMES]
        expected = stats.kruskal(*(values[member] for member in members))
        assert result.table.loc[0, "kruskal_statistic"] == pytest.approx(
            expected.statistic, rel=1e-9
        )
        assert result.table.loc[0, "p_value"] == pytest.approx(
            expected.pvalue, rel=1e-6, abs=1e-300
        )
        # Dunn's z of every pair from tie-averaged ranks and the tie-corrected
        # rank variance.
        ranks = stats.rankdata(values)
        _, ties = np.unique(values, return_counts=True)
        tied = ties.astype(np.float64)
        n = float(n_cells)
        variance = n * (n + 1) / 12 - np.sum(tied**3 - tied) / (12 * (n - 1))
        pairs = result.posthoc_table
        assert len(pairs) == 6
        for row in pairs.itertuples():
            first, second = labels == row.group_1, labels == row.group_2
            error = np.sqrt(variance * (1 / first.sum() + 1 / second.sum()))
            z = (ranks[first].mean() - ranks[second].mean()) / error
            assert row.z == pytest.approx(z, rel=1e-9)

    # One call tests one gene; a run_statistical_testing call of ten genes
    # makes ten.
    ladder = Ladder(sizes=(40_000, 80_000, 160_000, 320_000), smoke=2_000, work=10.0)
    bench("statistics.group_comparison", make, ladder, check=check)


@lru_cache(maxsize=4)
def _marker_result(n_groups: int):
    """Return rank statistics of every gene in ``n_groups`` clusters."""
    from scarf.features.markers import RankMarkerResult

    rng = np.random.default_rng(inputs.SEED)
    shape = (inputs.N_GENES, n_groups)
    statistics = np.empty((*shape, 8))
    statistics[..., 0] = rng.random(shape)
    statistics[..., 1] = rng.lognormal(0.0, 1.0, shape)
    statistics[..., 2] = rng.lognormal(0.0, 1.0, shape)
    statistics[..., 3] = rng.random(shape)
    statistics[..., 4] = rng.random(shape)
    statistics[..., 5] = statistics[..., 1] / statistics[..., 2]
    statistics[..., 6] = rng.normal(0.0, 3.0, shape)
    statistics[..., 7] = rng.random(shape)
    # Leiden numbers its clusters from one.
    return RankMarkerResult(
        group_ids=np.arange(1, n_groups + 1),
        group_sizes=rng.integers(50, 5_000, n_groups),
        feature_index=np.arange(inputs.N_GENES),
        statistics=statistics,
    )


def test_marker_table_writes(bench, tmp_path) -> None:
    import zarr

    from scarf.datastore.datastore import DataStore

    names = np.array([f"GENE{index:05d}" for index in range(inputs.N_GENES)])
    ids = np.array([f"ENSG{index:011d}" for index in range(inputs.N_GENES)])
    slots = itertools.count()

    def make(n_groups: int):
        result = _marker_result(n_groups)
        # Every write needs a new artifact group.
        slot = zarr.open_group(str(tmp_path / f"markers_{next(slots)}.zarr"), mode="w")

        def call():
            DataStore._write_marker_slot(
                slot, result, workers=1, feature_names=names, feature_ids=ids
            )
            return slot

        return call

    def check(n_groups: int, slot) -> None:
        result = _marker_result(n_groups)
        total = int(result.group_sizes.sum())
        np.testing.assert_array_equal(slot["feature_index"][:], np.arange(len(names)))
        np.testing.assert_array_equal(slot["feature_names"][:], names)
        assert sorted(slot.group_keys(), key=int) == [
            str(group) for group in result.group_ids
        ]
        for position, (group, size) in enumerate(
            zip(result.group_ids, result.group_sizes, strict=True)
        ):
            cluster = slot[str(group)]
            assert cluster.attrs["n_group"] == size
            assert cluster.attrs["n_reference"] == total - size
            # Statistics round to five decimals; the z statistic becomes its
            # two-sided p-value, which is then adjusted within the group.
            rank = result.statistics[:, position]
            p_values = 2.0 * stats.norm.sf(np.abs(rank[:, 6]))
            expected = np.column_stack(
                [
                    np.round(rank[:, :6], 5),
                    p_values,
                    np.round(rank[:, 7], 5),
                    _benjamini_hochberg(p_values),
                ]
            )
            np.testing.assert_allclose(cluster["stats"][:], expected, rtol=1e-12)

    # run_marker_search writes one table per cluster; large clusterings hold
    # hundreds of clusters. Writing the three feature columns is a fixed cost.
    ladder = Ladder(
        sizes=(10, 20, 40, 80),
        smoke=4,
        unit="groups",
        targets=(200, 1_000),
        model="linear",
    )
    bench("markers.table_write", make, ladder, check=check, fresh=True)


@lru_cache(maxsize=4)
def _planted_hashtags(n_cells: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return hashtag counts of singlets, doublets, and negatives, and their calls."""
    rng = np.random.default_rng(inputs.SEED)
    n_tags = len(HTO_NAMES)
    # An overdispersed background of at most two counts. Once each background
    # cluster holds a few hundred cells, its fitted negative-binomial 99th
    # percentile is three or more, so no background count passes the cutoff
    # and every planted call is recoverable.
    counts = rng.choice(3, p=[0.8, 0.1, 0.1], size=(n_cells, n_tags))
    kind = rng.random(n_cells)
    doublet = (kind >= 0.90) & (kind < 0.96)
    negative = kind >= 0.96
    first = rng.integers(0, n_tags, n_cells)
    second = (first + rng.integers(1, n_tags, n_cells)) % n_tags
    tagged = np.flatnonzero(~negative)
    counts[tagged, first[tagged]] += rng.integers(100, 300, tagged.size)
    paired = np.flatnonzero(doublet)
    counts[paired, second[paired]] += rng.integers(100, 300, paired.size)
    calls = np.where(
        negative,
        "Negative",
        np.where(doublet, "Doublet", np.asarray(HTO_NAMES)[first]),
    )
    return pd.DataFrame(counts.astype(np.uint32), columns=HTO_NAMES), calls


def test_hto_demultiplexing(bench) -> None:
    from threadpoolctl import threadpool_limits

    from scarf.quality_control.hto import hto_demux

    def make(n_cells: int):
        counts, _calls = _planted_hashtags(n_cells)

        def call():
            # Projections assume one thread. Many-threaded K-means restarts on
            # a busy machine spend most of their time waiting at barriers.
            with threadpool_limits(limits=1):
                return hto_demux(counts)

        return call

    def check(n_cells: int, identities) -> None:
        counts, calls = _planted_hashtags(n_cells)
        assert identities.index.equals(counts.index)
        np.testing.assert_array_equal(identities.to_numpy(), calls)

    # Each of the one hundred K-means restarts scans every cell; validation
    # and the background fits of small clusters add a fixed cost.
    ladder = Ladder(sizes=(2_000, 4_000, 8_000, 16_000), smoke=2_000, model="linear")
    bench("quality_control.hto_demux", make, ladder, check=check)


@lru_cache(maxsize=4)
def _variance_trend(n_genes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gene means, variances on a power-law trend, and their noise."""
    rng = np.random.default_rng(inputs.SEED)
    means = np.exp(rng.normal(0.0, 2.0, n_genes))
    noise = np.exp(rng.normal(0.0, TREND_NOISE, n_genes))
    return means, means**1.4 * noise, noise


def test_hvg_variance_trend(bench) -> None:
    from scarf.features.variability import fit_lowess

    def make(n_genes: int):
        means, variances, _noise = _variance_trend(n_genes)
        # select_hvgs defaults: 200 bins and a span of one tenth.
        return lambda: fit_lowess(means, variances, n_bins=200, lowess_frac=0.1)

    def check(n_genes: int, corrected) -> None:
        _means, _variances, noise = _variance_trend(n_genes)
        # The adaptive trend follows the lower quartile of the log variances,
        # so each corrected variance is the gene's noise over the quartile of
        # the log-normal noise.
        quartile = np.exp(stats.norm.ppf(0.25) * TREND_NOISE)
        ratio = corrected * quartile / noise
        assert np.median(ratio) == pytest.approx(1.0, abs=0.02)
        low, high = np.quantile(ratio, [0.01, 0.99])
        assert 0.85 < low and high < 1.15

    # select_hvgs fits every detected gene; the 200 local fits are a fixed cost.
    ladder = Ladder(
        sizes=(3_000, 6_000, 12_000, 24_000),
        smoke=2_000,
        unit="features",
        targets=(30_000, 60_000),
        model="linear",
    )
    bench("hvg.variance_trend", make, ladder, check=check)
