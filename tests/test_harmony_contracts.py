import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scarf.embeddings as embeddings
import scarf.embeddings.harmony as harmony
from scarf.embeddings.harmony.api import fit_harmony as implementation_fit_harmony
from scarf.embeddings.harmony.api import run_harmony as implementation_run_harmony
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
        "run_harmony",
        "safe_entropy",
    ]
    assert harmony.fit_harmony is implementation_fit_harmony
    assert harmony.run_harmony is implementation_run_harmony
    assert harmony.HarmonyResult is implementation_result
    assert harmony.Harmony is implementation_optimizer
    assert embeddings.Harmony is harmony.Harmony
    assert embeddings.HarmonyResult is harmony.HarmonyResult
    assert embeddings.fit_harmony is harmony.fit_harmony
    assert embeddings.run_harmony is harmony.run_harmony


def test_harmony_public_metadata_and_signatures_remain_stable():
    public_objects = (
        harmony.Harmony,
        harmony.HarmonyResult,
        harmony.fit_harmony,
        harmony.moe_correct_ridge,
        harmony.run_harmony,
        harmony.safe_entropy,
    )
    assert {obj.__module__ for obj in public_objects} == {"scarf.embeddings.harmony"}
    assert signature_digest(
        {
            "fit_harmony": harmony.fit_harmony,
            "run_harmony": harmony.run_harmony,
        }
    ) == ("7b192e50655559a92f78d67ba963db2f4bb1df3195f5f7305effc37f913dbf9c")
    assert (
        inspect.signature(harmony.fit_harmony).parameters
        == inspect.signature(harmony.run_harmony).parameters
    )


def test_run_harmony_resolves_fit_through_public_facade(monkeypatch):
    corrected = np.array([[1.0, 2.0]])
    calls = []

    def fake_fit_harmony(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(corrected=corrected)

    monkeypatch.setattr(embeddings, "fit_harmony", fake_fit_harmony)
    actual = embeddings.run_harmony(
        np.zeros((1, 2)),
        pd.DataFrame({"batch": ["a", "b"]}),
    )

    assert actual is corrected
    assert len(calls) == 1


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
            "lambda values must be finite and non-negative",
        ),
        (
            np.zeros((2, 4)),
            pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
            {"nclust": 2, "cluster_fn": "unknown"},
            "cluster_fn must be 'kmeans'",
        ),
    ],
)
def test_fit_harmony_rejects_invalid_contracts(values, metadata, kwargs, message):
    with pytest.raises(ValueError, match=message):
        harmony.fit_harmony(values, metadata, **kwargs)


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

    assert captured["theta"].shape == (4,)
    np.testing.assert_array_equal(np.diag(captured["ridge"])[1:], [1, 2, 3, 4])
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
