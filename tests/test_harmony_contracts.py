import numpy as np
import pandas as pd
import pytest

import scarf.embeddings as embeddings
import scarf.embeddings.harmony as harmony
from scarf.embeddings.harmony.api import fit_harmony as implementation_fit_harmony
from scarf.embeddings.harmony.api import validate_harmony_parameters
from scarf.embeddings.harmony.models import HarmonyResult as implementation_result
from scarf.embeddings.harmony.optimizer import Harmony as implementation_optimizer
from tests.signature_contracts import signature_digest


def test_harmony_facade_exports_canonical_objects():
    assert harmony.__all__ == [
        "ClusterFn",
        "Harmony",
        "HarmonyResult",
        "fit_harmony",
        "moe_correct_ridge",
        "safe_entropy",
    ]
    assert harmony.fit_harmony is implementation_fit_harmony
    assert harmony.HarmonyResult is implementation_result
    assert harmony.Harmony is implementation_optimizer
    assert embeddings.Harmony is harmony.Harmony
    assert embeddings.HarmonyResult is harmony.HarmonyResult
    assert embeddings.fit_harmony is harmony.fit_harmony
    assert "run_harmony" not in embeddings.__all__
    assert not hasattr(harmony, "run_harmony")


def test_harmony_public_metadata_and_signatures_remain_stable():
    public_objects = (
        harmony.Harmony,
        harmony.HarmonyResult,
        harmony.fit_harmony,
        harmony.moe_correct_ridge,
        harmony.safe_entropy,
    )
    assert {obj.__module__ for obj in public_objects} == {"scarf.embeddings.harmony"}
    assert signature_digest({"fit_harmony": harmony.fit_harmony}) == (
        "8db928c01bc083bbf16b1a1603459353ae7de20d72f4e1a7b7f55ecd2d7e70a6"
    )


@pytest.mark.parametrize(
    ("values", "metadata", "kwargs", "message"),
    [
        (np.zeros(4), pd.DataFrame({"batch": ["a"] * 4}), {}, "two-dimensional"),
        (
            np.zeros((2, 3)),
            pd.DataFrame({"batch": ["a"] * 4}),
            {},
            "metadata rows",
        ),
        (
            np.zeros((2, 1)),
            pd.DataFrame({"batch": ["a"]}),
            {},
            "at least two cells",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame(index=range(4)),
            {},
            "at least one batch metadata column",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame(
                np.array([["a", "x"], ["b", "y"], ["a", "x"], ["b", "y"]]),
                columns=["batch", "batch"],
            ),
            {},
            "column names must be unique",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", None, "a", "b"]}),
            {},
            "cannot contain missing",
        ),
        (
            np.array([[0.0, np.inf, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {},
            "contains non-finite",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 0},
            "nclust must be between",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 2, "sigma": np.ones(3)},
            "sigma must be scalar",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 2, "sigma": 0.0},
            "sigma values must be finite and positive",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 2, "theta": [1.0, 2.0, 3.0]},
            "Each Harmony batch level must have a theta",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 2, "theta": -1.0},
            "theta values must be finite and non-negative",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 2, "lamb": np.nan},
            "lamb values must be finite and non-negative",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 2, "cluster_fn": "unknown"},
            "cluster_fn must be 'kmeans'",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 2, "max_iter_kmeans": 0},
            "max_iter_kmeans must be at least 1",
        ),
    ],
)
def test_fit_harmony_rejects_invalid_contracts(values, metadata, kwargs, message):
    with pytest.raises(ValueError, match=message):
        harmony.fit_harmony(values, metadata, **kwargs)


def test_harmony_parameters_are_validated_without_data():
    def cluster(values, count):
        return values[:count]

    parameters = {
        "nclust": np.int64(3),
        "sigma": [0.1, 0.2, 0.3],
        "theta": None,
        "lamb": 1,
        "tau": 0,
        "block_size": 0.5,
        "max_iter_harmony": 0,
        "max_iter_kmeans": 5,
        "epsilon_cluster": 0.0,
        "epsilon_harmony": 1e-3,
        "random_state": 4,
        "cluster_fn": cluster,
    }

    resolved = validate_harmony_parameters(parameters)

    assert resolved == parameters
    assert resolved is not parameters
    assert validate_harmony_parameters(None) == {}


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ([("nclust", 2)], TypeError, "mapping"),
        ({"data_mat": np.zeros((2, 2))}, ValueError, "Unsupported .*data_mat"),
        ({"n_clusters": 2}, ValueError, "Unsupported .*n_clusters"),
        ({"nclust": 0}, ValueError, "nclust must be at least 1"),
        ({"nclust": True}, TypeError, "nclust must be an integer"),
        ({"max_iter_harmony": -1}, ValueError, "max_iter_harmony"),
        ({"max_iter_kmeans": 2.0}, TypeError, "max_iter_kmeans"),
        ({"max_iter_kmeans": 0}, ValueError, "max_iter_kmeans must be at least 1"),
        ({"random_state": None}, TypeError, "random_state"),
        ({"random_state": -3}, ValueError, "random_state"),
        ({"tau": -1.0}, ValueError, "tau must be finite"),
        ({"block_size": 0}, ValueError, "block_size must be finite and positive"),
        ({"epsilon_harmony": np.nan}, ValueError, "epsilon_harmony"),
        ({"epsilon_cluster": "small"}, TypeError, "epsilon_cluster"),
        ({"sigma": 0.0}, ValueError, "sigma values must be finite and positive"),
        ({"sigma": None}, TypeError, "sigma must contain real numbers"),
        ({"sigma": "wide"}, TypeError, "sigma must contain real numbers"),
        ({"theta": [1.0, "high"]}, TypeError, "theta must contain real numbers"),
        ({"lamb": {"batch": 1.0}}, TypeError, "lamb must contain real numbers"),
        ({"theta": [1.0, -1.0]}, ValueError, "theta values"),
        ({"lamb": np.inf}, ValueError, "lamb values"),
        ({"cluster_fn": "unknown"}, ValueError, "cluster_fn"),
    ],
)
def test_harmony_parameter_validation_rejects_invalid_values(
    parameters,
    error,
    message,
):
    with pytest.raises(error, match=message):
        validate_harmony_parameters(parameters)


def test_fit_harmony_expands_per_column_parameters_and_records_callable(monkeypatch):
    captured = {}

    class FakeHarmony:
        def __init__(self, *args):
            captured["theta"] = args[5].copy()
            captured["ridge"] = args[12].copy()
            data_mat = args[0]
            nclust = args[10]
            self.Z_orig = data_mat.copy()
            self.R = np.zeros((nclust, data_mat.shape[1]))
            self.Y = np.zeros((data_mat.shape[0], nclust))

        def result(self):
            return self.Z_orig.copy()

    def cluster_backend(*_args, **_kwargs):
        return None

    monkeypatch.setattr(embeddings, "Harmony", FakeHarmony)
    metadata = pd.DataFrame(
        {
            "batch": ["a", "b", "a", "b"],
            "donor": ["x", "x", "y", "y"],
        }
    )

    result = harmony.fit_harmony(
        np.zeros((2, 4)),
        metadata,
        theta=[2.0, 3.0],
        lamb=[1.0, 2.0, 3.0, 4.0],
        sigma=np.array([0.2, 0.3]),
        nclust=2,
        tau=1.0,
        cluster_fn=cluster_backend,
    )

    # One theta per column expands to its two levels; with tau, each level's
    # theta shrinks by 1 - exp(-(cells / (nclust * tau)) ** 2), here 1 - 1/e.
    np.testing.assert_allclose(
        captured["theta"], np.array([2.0, 2.0, 3.0, 3.0]) * (1 - np.exp(-1.0))
    )
    np.testing.assert_array_equal(np.diag(captured["ridge"]), [0, 1, 2, 3, 4])
    assert result.parameters["clusterBackend"].endswith(".cluster_backend")


def test_harmony_keeps_an_independent_original_coordinate_snapshot():
    values = np.random.default_rng(4).normal(size=(3, 12))
    metadata = pd.DataFrame({"batch": ["a", "b"] * 6})

    result = harmony.fit_harmony(
        values,
        metadata,
        nclust=2,
        max_iter_harmony=1,
        max_iter_kmeans=1,
    )

    assert result.original is not values
    np.testing.assert_array_equal(result.original, values)
    assert result.corrected.dtype == np.dtype(np.float64)


def test_harmony_progress_completes_when_optimization_converges(monkeypatch):
    closed: list[tuple[int, int]] = []

    class Progress:
        def __init__(self, total: int) -> None:
            self.n = 0
            self.total = total

        def update(self) -> None:
            self.n += 1

        def refresh(self) -> None:
            pass

        def close(self) -> None:
            closed.append((self.n, self.total))

    monkeypatch.setattr(
        "scarf.embeddings.harmony.optimizer.tqdmbar",
        lambda *_, total, **__: Progress(total),
    )
    values = np.random.default_rng(5).normal(size=(3, 12))
    metadata = pd.DataFrame({"batch": ["a", "b"] * 6})

    harmony.fit_harmony(
        values,
        metadata,
        nclust=2,
        max_iter_harmony=50,
        max_iter_kmeans=1,
        epsilon_harmony=1e9,
    )

    assert len(closed) == 1
    completed, total = closed[0]
    assert completed == total
    assert completed < 50


def test_harmony_supports_numeric_batch_columns_without_global_rng_changes():
    values = np.random.default_rng(9).normal(size=(3, 12))
    metadata = pd.DataFrame({"batch": [0, 1] * 6})
    np.random.seed(17)
    expected_next = np.random.random()
    np.random.seed(17)

    result = harmony.fit_harmony(
        values,
        metadata,
        nclust=2,
        max_iter_harmony=1,
        max_iter_kmeans=1,
    )

    assert result.parameters["clusterBackend"] == "sklearn.cluster.KMeans"
    assert np.random.random() == expected_next


def test_harmony_centroids_match_final_assignments_and_coordinates():
    values = np.random.default_rng(11).normal(size=(4, 16))
    metadata = pd.DataFrame({"batch": ["a", "b"] * 8})

    result = harmony.fit_harmony(
        values,
        metadata,
        nclust=3,
        max_iter_harmony=2,
        max_iter_kmeans=2,
    )

    normalized = result.corrected / np.linalg.norm(
        result.corrected,
        axis=0,
        keepdims=True,
    )
    expected = normalized @ result.assignments.T
    expected /= np.linalg.norm(expected, axis=0, keepdims=True)
    np.testing.assert_allclose(result.centroids, expected)


def test_harmony_uses_a_callable_cluster_backend():
    from sklearn.cluster import KMeans

    values = np.random.default_rng(11).normal(size=(4, 16))
    metadata = pd.DataFrame({"batch": ["a", "b"] * 8})
    calls = []

    def kmeans_backend(data, n_clusters):
        calls.append((data.copy(), n_clusters))
        return (
            KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=10,
                max_iter=25,
                random_state=0,
            )
            .fit(data)
            .cluster_centers_
        )

    options = {"nclust": 3, "max_iter_harmony": 2, "max_iter_kmeans": 2}
    custom = harmony.fit_harmony(values, metadata, cluster_fn=kmeans_backend, **options)
    default = harmony.fit_harmony(values, metadata, **options)

    ((data, n_clusters),) = calls
    assert n_clusters == 3
    # The backend clusters cells by direction: unit-length rows of the input.
    np.testing.assert_allclose(data, (values / np.linalg.norm(values, axis=0)).T)
    # A backend that reproduces the default k-means gives the default fit.
    np.testing.assert_allclose(custom.corrected, default.corrected)
    assert custom.parameters["clusterBackend"].endswith(".kmeans_backend")


def test_moe_correct_ridge_matches_a_hand_solved_ridge_regression():
    # One cluster holding four cells in two batches, with ridge penalty 1 on
    # each batch level and none on the intercept.
    values = np.array([[1.0, 3.0, 10.0, 14.0]])
    design = np.array(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )

    unit, corrected = harmony.moe_correct_ridge(
        values, np.ones((1, 4)), 1, design, np.diag([0.0, 1.0, 1.0])
    )

    # (X X^T + L) W = X Z^T gives intercept 7 and batch effects -10/3 and
    # 10/3; removing the batch terms moves each batch 10/3 toward the other.
    np.testing.assert_allclose(
        corrected, [[1 + 10 / 3, 3 + 10 / 3, 10 - 10 / 3, 14 - 10 / 3]]
    )
    np.testing.assert_allclose(unit, np.ones((1, 4)))
    np.testing.assert_array_equal(values, [[1.0, 3.0, 10.0, 14.0]])


def test_moe_correct_ridge_rejects_a_singular_system():
    # A batch level without cells and without a ridge penalty leaves its
    # coefficient undetermined.
    design = np.array(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    )

    with pytest.raises(ValueError, match="Harmony ridge system is singular"):
        harmony.moe_correct_ridge(
            np.ones((2, 4)), np.ones((1, 4)), 1, design, np.zeros((3, 3))
        )
