"""HTO demultiplexing against planted identities and an exact NB2 quantile."""

import re
from math import exp, lgamma, log
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scarf import DataStore
from scarf.quality_control import hto
from scarf.quality_control.hto import (
    _background_clusters,
    _classify_hto_identities,
    _cluster_labels,
    _clr_normalize,
    _fit_negative_binomial_parameters,
    _negative_binomial_cutoff,
    _positive_hto_calls,
    hto_demux,
)
from scarf.storage.selections import read_stored_selection_indices
from tests.storage_helpers import write_count_store

HTOS = ["HTO_A", "HTO_B", "HTO_C"]


def planted_hto_counts(
    *,
    names: list[str] = HTOS,
    n_singlets: int = 16,
    n_doublets: int = 6,
    n_negatives: int = 6,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return HTO counts and the identity planted in each cell.

    Background counts cycle through 0 to 6 with mean 3 and variance 4, so the
    negative-binomial 99% cutoff of every background lies above all of them,
    and each tagged HTO adds 100 to 199 counts.
    """
    truth = [name for name in names for _ in range(n_singlets)]
    truth += ["Doublet"] * n_doublets + ["Negative"] * n_negatives
    rows = np.arange(len(truth))[:, None]
    counts = (rows * 5 + np.arange(len(names))[None, :] * 3) % 7
    rng = np.random.default_rng(seed)
    for row, label in enumerate(truth):
        if label in names:
            tagged = [names.index(label)]
        elif label == "Doublet":
            tagged = [row % 3, (row + 1) % 3]
        else:
            tagged = []
        for column in tagged:
            counts[row, column] += 100 + rng.integers(0, 100)
    frame = pd.DataFrame(
        counts, columns=names, index=[f"cell{index}" for index in range(len(truth))]
    )
    return frame, np.asarray(truth, dtype=object)


def nb2_quantile(mu: float, alpha: float, quantile: float = 0.99) -> int:
    """Smallest count whose NB2 cumulative probability reaches ``quantile``."""
    size = 1 / alpha
    success = 1 / (1 + alpha * mu)
    cumulative = 0.0
    count = 0
    while True:
        cumulative += exp(
            lgamma(count + size)
            - lgamma(count + 1)
            - lgamma(size)
            + size * log(success)
            + count * log(1 - success)
        )
        if cumulative >= quantile:
            return count
        count += 1


def _replace_nb2_model(
    monkeypatch,
    *,
    params: object = (0.0, 1.0),
    retvals: object = None,
    error: Exception | None = None,
) -> None:
    """Replace the statsmodels NB2 model with one that returns a fixed fit."""

    class FitResult:
        pass

    if retvals is not None:
        FitResult.mle_retvals = retvals
    FitResult.params = params

    class Model:
        def __init__(self, *args, **kwargs):
            if error is not None:
                raise error

        def fit(self, **kwargs):
            return FitResult()

    monkeypatch.setattr("statsmodels.discrete.discrete_model.NegativeBinomial", Model)


@pytest.mark.parametrize(
    ("seed", "names"),
    [(0, HTOS), (1, HTOS), (2, ["cluster", "HTO_B", "HTO_C"])],
    ids=["seed0", "seed1", "hto_named_cluster"],
)
def test_hto_demux_recovers_planted_singlets_doublets_and_negatives(seed, names):
    counts, truth = planted_hto_counts(names=names, seed=seed)
    original = counts.copy(deep=True)

    identities = hto_demux(counts, random_seed=seed)

    np.testing.assert_array_equal(identities.to_numpy(), truth)
    assert identities.index.equals(counts.index)
    pd.testing.assert_frame_equal(counts, original)
    pd.testing.assert_series_equal(hto_demux(counts, random_seed=seed), identities)
    # Permuting cells permutes their identities.
    order = np.random.default_rng(seed).permutation(len(counts))
    permuted = hto_demux(counts.iloc[order], random_seed=seed)
    np.testing.assert_array_equal(permuted.to_numpy(), truth[order])


@pytest.mark.parametrize(
    ("mu", "alpha"),
    [(1.0, 1.0), (3.0, 0.1), (0.2, 4.0), (7.5, 0.05), (3.0, 1 / 9)],
)
def test_negative_binomial_cutoff_is_the_nb2_99th_percentile(mu, alpha):
    assert _negative_binomial_cutoff(mu, alpha) == nb2_quantile(mu, alpha)
    assert _negative_binomial_cutoff(mu, alpha, quantile=0.5) == nb2_quantile(
        mu, alpha, 0.5
    )


def test_positive_calls_exceed_the_unshifted_cutoff_strictly(monkeypatch):
    cutoff = nb2_quantile(1.0, 1.0)
    counts = pd.DataFrame({"HTO_A": [1, 2, cutoff, cutoff + 1]})
    fitted: list[tuple[list[int], str]] = []

    def fit(values, name):
        fitted.append((np.asarray(values).tolist(), name))
        return 1.0, 1.0

    monkeypatch.setattr(hto, "_fit_negative_binomial_parameters", fit)

    positive = _positive_hto_calls(counts, np.asarray([0, 0, 1, 1]))

    # The fit sees the cluster with the lower raw mean, and a count equal to
    # the 99th percentile itself is not positive.
    assert fitted == [([1, 2], "HTO_A")]
    assert positive["HTO_A"].tolist() == [False, False, False, True]


def test_background_cluster_uses_raw_means():
    counts = pd.DataFrame({"HTO_A": [0, 100, 40, 40]})
    cluster_labels = np.asarray([0, 0, 1, 1])

    background = _background_clusters(counts, cluster_labels)
    normalized_means = _clr_normalize(counts).groupby(cluster_labels).mean()

    # Cluster 1 has the lower raw mean although cluster 0 has the lower CLR mean.
    assert background["HTO_A"] == 1
    assert normalized_means["HTO_A"].idxmin() == 0


def test_classification_uses_clr_argmax_for_singlets():
    index = ["negative", "singlet", "doublet", "tie"]
    normalized = pd.DataFrame(
        {
            "HTO_A": [0.2, 0.2, 1.0, 0.5],
            "HTO_B": [0.1, 1.5, 0.9, 0.5],
        },
        index=index,
    )
    positive = pd.DataFrame(
        {
            "HTO_A": [False, True, True, True],
            "HTO_B": [False, False, True, False],
        },
        index=index,
    )

    identities = _classify_hto_identities(normalized, positive)

    assert identities.to_dict() == {
        "negative": "Negative",
        "singlet": "HTO_B",
        "doublet": "Doublet",
        "tie": "HTO_A",
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0.0, 1.0), "mean must be finite and greater than 0"),
        ((np.inf, 1.0), "mean must be finite and greater than 0"),
        ((1.0, 0.0), "dispersion must be finite and greater than 0"),
        ((1.0, np.nan), "dispersion must be finite and greater than 0"),
        ((1.0, 1.0, 1.0), "quantile must be between 0 and 1"),
        ((1.0, 1.0, 0.0), "quantile must be between 0 and 1"),
        ((1.0, 1.0, np.nan), "quantile must be between 0 and 1"),
        # The success probability underflows to zero.
        ((1e300, 1e300), "parameters produced an invalid distribution"),
        # The size 1 / alpha overflows.
        ((1.0, 1e-320), "parameters produced an invalid distribution"),
    ],
)
def test_negative_binomial_cutoff_rejects_invalid_parameters(arguments, message):
    with pytest.raises(ValueError, match=f"^Negative-binomial {message}$"):
        _negative_binomial_cutoff(*arguments)


@pytest.mark.parametrize("quantile", [np.inf, np.nan, 2.5])
def test_negative_binomial_cutoff_requires_a_finite_integer_quantile(
    monkeypatch, quantile
):
    monkeypatch.setattr("scipy.stats.nbinom.ppf", lambda *args, **kwargs: quantile)

    with pytest.raises(
        ValueError, match="^Negative-binomial cutoff must be a finite integer$"
    ):
        _negative_binomial_cutoff(mu=1, alpha=1)


def test_positive_calls_report_which_hto_has_an_invalid_cutoff(monkeypatch):
    counts = pd.DataFrame({"HTO_A": [1, 2, 60, 70], "HTO_B": [50, 60, 1, 2]})
    monkeypatch.setattr(
        hto,
        "_fit_negative_binomial_parameters",
        lambda values, name: (1e300, 1e300) if name == "HTO_B" else (1.0, 1.0),
    )

    with pytest.raises(
        ValueError, match="^Negative-binomial cutoff is invalid for HTO 'HTO_B'$"
    ) as caught:
        _positive_hto_calls(counts, np.asarray([0, 0, 1, 1]))
    assert "invalid distribution" in str(caught.value.__cause__)


@pytest.mark.parametrize(
    ("counts", "labels", "message"),
    [
        (
            pd.DataFrame({"HTO_A": [1, 10, 20]}),
            np.asarray([0, 1, 2]),
            "must contain at least two cells",
        ),
        (
            pd.DataFrame({"HTO_A": [0, 0, 10, 20]}),
            np.asarray([0, 0, 1, 1]),
            "contains only zero counts",
        ),
    ],
    ids=["single-cell", "all-zero"],
)
def test_positive_calls_reject_invalid_backgrounds(counts, labels, message):
    with pytest.raises(
        ValueError, match=f"^Background cluster for HTO 'HTO_A' {message}$"
    ):
        _positive_hto_calls(counts, labels)


@pytest.mark.parametrize(
    "background",
    [
        np.tile(np.arange(7), 40),
        np.random.default_rng(4).negative_binomial(2, 0.3, size=300),
    ],
    ids=["cycled", "negative_binomial"],
)
def test_negative_binomial_fit_is_the_nb2_maximum_likelihood(background):
    from scipy.optimize import minimize_scalar
    from scipy.stats import nbinom

    mu, alpha = _fit_negative_binomial_parameters(background, "HTO_A")

    # The NB2 likelihood of an intercept-only model peaks at the sample mean,
    # and its dispersion maximizes the likelihood profiled at that mean.
    mean = float(np.mean(background))

    def negative_log_likelihood(log_alpha: float) -> float:
        dispersion = np.exp(log_alpha)
        return -float(
            nbinom.logpmf(background, 1 / dispersion, 1 / (1 + dispersion * mean)).sum()
        )

    profile = minimize_scalar(
        negative_log_likelihood,
        bounds=(-12, 5),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert mu == pytest.approx(mean, rel=1e-5)
    assert alpha == pytest.approx(np.exp(profile.x), rel=1e-4)


@pytest.mark.parametrize(
    ("fit", "message"),
    [
        ({"retvals": {"converged": False}}, "did not converge"),
        ({}, "did not converge"),
        ({"retvals": {"converged": "yes"}}, "did not converge"),
        (
            {"retvals": {"converged": True}, "params": {"intercept": 1.0}},
            "returned invalid parameters",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([0.5])},
            "returned invalid parameters",
        ),
        (
            {"retvals": {"converged": True}, "params": np.zeros(3)},
            "returned invalid parameters",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([np.inf, 1.0])},
            "returned invalid mean",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([np.nan, 1.0])},
            "returned invalid mean",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([0.0, 0.0])},
            "returned invalid dispersion",
        ),
        (
            {"retvals": {"converged": True}, "params": np.asarray([0.0, -0.5])},
            "returned invalid dispersion",
        ),
    ],
    ids=[
        "not_converged",
        "no_convergence_record",
        "convergence_not_boolean",
        "mapping",
        "one_parameter",
        "three_parameters",
        "infinite_mean",
        "nan_mean",
        "zero_dispersion",
        "negative_dispersion",
    ],
)
def test_negative_binomial_fit_rejects_unusable_results(monkeypatch, fit, message):
    _replace_nb2_model(monkeypatch, **fit)

    with pytest.raises(
        ValueError,
        match=f"^Negative-binomial background fit {message} for HTO 'HTO_A'$",
    ):
        _fit_negative_binomial_parameters(np.asarray([1, 2]), "HTO_A")


def test_negative_binomial_fit_wraps_optimizer_errors(monkeypatch):
    _replace_nb2_model(monkeypatch, error=RuntimeError("optimizer failed"))

    with pytest.raises(
        ValueError, match="^Negative-binomial background fit failed for HTO 'HTO_A'$"
    ) as caught:
        _fit_negative_binomial_parameters(np.asarray([1, 2]), "HTO_A")
    assert str(caught.value.__cause__) == "optimizer failed"


def test_cluster_labels_reject_collapsed_kmeans(monkeypatch):
    class CollapsedKMeans:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, values):
            return np.asarray([0, 0, 1])

    monkeypatch.setattr("sklearn.cluster.KMeans", CollapsedKMeans)
    normalized = pd.DataFrame(
        {
            "HTO_A": [0.0, 1.0, 2.0],
            "HTO_B": [2.0, 1.0, 0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="^HTO clustering produced 2 occupied clusters; expected 3$",
    ):
        _cluster_labels(normalized, random_seed=0)


@pytest.mark.parametrize(
    ("hto_counts", "error", "message"),
    [
        (np.asarray([[1], [2]]), TypeError, "hto_counts must be a pandas DataFrame"),
        (
            pd.DataFrame(index=range(2)),
            ValueError,
            "hto_counts must contain at least one HTO",
        ),
        (
            pd.DataFrame(np.ones((3, 2)), columns=["HTO_A", "HTO_A"]),
            ValueError,
            "HTO IDs must be unique",
        ),
        (pd.DataFrame({" ": [1, 2]}), ValueError, "HTO IDs must be non-empty strings"),
        (
            pd.DataFrame({"Negative": [1, 2]}),
            ValueError,
            "HTO IDs conflict with reserved identity labels: 'Negative'",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 2]}, index=["cell", "cell"]),
            ValueError,
            "hto_counts cell index must be unique",
        ),
        (
            pd.DataFrame({"HTO_A": ["1", "2"]}),
            TypeError,
            "hto_counts must contain only numeric raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1 + 0j, 2 + 0j, 3 + 0j]}),
            TypeError,
            "hto_counts must contain only real numeric raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, np.nan]}),
            ValueError,
            "hto_counts must contain only finite raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, -1]}),
            ValueError,
            "hto_counts must contain only nonnegative raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 1.5]}),
            ValueError,
            "hto_counts must contain integer-valued raw counts",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 2], "HTO_B": [2, 1]}),
            ValueError,
            "HTO demultiplexing requires at least 3 selected cells",
        ),
        (
            pd.DataFrame({"HTO_A": [0, 0], "HTO_B": [1, 2], "HTO_C": [0, 0]}),
            ValueError,
            "HTO demultiplexing requires at least 4 selected cells",
        ),
        (
            pd.DataFrame({"HTO_A": [0, 0, 0], "HTO_B": [1, 2, 3]}),
            ValueError,
            "HTOs with no positive counts cannot be demultiplexed: 'HTO_A'",
        ),
        (
            pd.DataFrame({"HTO_A": [1, 1, 1], "HTO_B": [2, 2, 2]}),
            ValueError,
            "HTO demultiplexing requires at least 3 distinct normalized cell "
            "profiles; found 1",
        ),
    ],
    ids=[
        "not-dataframe",
        "no-htos",
        "duplicate-htos",
        "empty-hto",
        "reserved-hto",
        "duplicate-cells",
        "nonnumeric",
        "complex",
        "nonfinite",
        "negative",
        "fractional",
        "too-few-cells",
        "too-few-cells-for-three-htos",
        "all-zero-hto",
        "identical-profiles",
    ],
)
def test_hto_demux_rejects_invalid_input(hto_counts, error, message):
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        hto_demux(hto_counts)


@pytest.mark.parametrize("random_seed", [True, 1.5, "0", None])
def test_hto_demux_requires_an_integer_seed(random_seed):
    counts, _ = planted_hto_counts()

    with pytest.raises(TypeError, match="^random_seed must be an integer$"):
        hto_demux(counts, random_seed=random_seed)


def test_hto_identity_artifact_holds_the_planted_identities_of_selected_cells(
    tmp_path: Path,
):
    counts, truth = planted_hto_counts()
    n_cells = len(counts)
    rna = np.random.default_rng(3).poisson(2.0, size=(n_cells, 12))
    path = tmp_path / "store.zarr"
    write_count_store(str(path), {"RNA": rna, "HTO": counts.to_numpy()}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    keep = np.ones(n_cells, dtype=bool)
    keep[[2, 17, 40, 55]] = False
    store.cells.insert("tagged", keep)
    selection = store.snapshot_cell_selection("tagged")

    ref = store.run_hto_demultiplexing(selection)

    status = store.inspect_artifact(ref)
    assert ref.kind == "hto_identity"
    assert status.operation == "run_hto_demultiplexing"
    assert status.parameters["method"] == {
        "normalization": "clr_per_hto",
        "clustering": {
            "method": "kmeans",
            "init": "random",
            "n_starts": 100,
            "cluster_count": "n_htos_plus_one",
        },
        "background": "raw_mean",
        "cutoff": {
            "distribution": "negative_binomial_nb2",
            "quantile": 0.99,
            "location": 0,
            "comparison": "strictly_greater",
        },
        "singlet_assignment": "clr_argmax",
    }
    cells = read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(cells, np.flatnonzero(keep))
    # The HTO assay names its features HTO0, HTO1, and HTO2.
    renamed = {name: f"HTO{index}" for index, name in enumerate(HTOS)}
    expected = [renamed.get(label, label) for label in truth[keep]]
    np.testing.assert_array_equal(store.load_artifact(ref)["values"][:], expected)
    assert store.run_hto_demultiplexing(selection) == ref
    assert "HTO0" not in store.cells.columns
