import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scarf.embeddings as embeddings
import scarf.embeddings.harmony as harmony
from scarf.embeddings.harmony.api import fit_harmony as implementation_fit_harmony
from scarf.embeddings.harmony.api import run_harmony as implementation_run_harmony
from scarf.embeddings.harmony.models import HarmonyResult as implementation_result
from scarf.embeddings.harmony.optimizer import Harmony as implementation_optimizer


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
    assert str(inspect.signature(harmony.run_harmony)) == (
        "(data_mat: numpy.ndarray, meta_data: pandas.core.frame.DataFrame, "
        "theta: float | int | numpy.ndarray | list[float] | None = None, "
        "lamb: float | int | numpy.ndarray | list[float] | None = None, "
        "sigma: float | numpy.ndarray = 0.1, nclust: int | None = None, "
        "tau: float = 0, block_size: float = 0.05, "
        "max_iter_harmony: int = 50, max_iter_kmeans: int = 20, "
        "epsilon_cluster: float = 1e-05, epsilon_harmony: float = 0.0001, "
        "random_state: int = 0, cluster_fn: ClusterFn = 'kmeans') -> numpy.ndarray"
    )
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
