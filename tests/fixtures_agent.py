"""Offline agent decisions and temporary RNA fixtures for the global suite."""

from collections.abc import Iterator
from typing import Any

import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True)
def offline_agent_models(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep the agent guard and thread limits scoped to agent test modules."""
    if not request.node.path.name.startswith("test_agent_"):
        yield
        return
    from pydantic_ai import models

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def agent_rna_source(tmp_path: Any) -> Any:
    """A small count store with crossed technical batches and three populations."""
    import numpy as np
    from scipy.sparse import csr_matrix

    from scarf import DataStore
    from scarf.writers import SparseToZarr

    source = tmp_path / "source.zarr"
    rng = np.random.default_rng(38)
    values = rng.poisson(1.0, size=(120, 90)).astype(np.uint32)
    for group in range(3):
        values[group * 40 : (group + 1) * 40, 2 + group * 15 : 17 + group * 15] += (
            rng.poisson(5.0, size=(40, 15)).astype(np.uint32)
        )
    names = ["MT-CO1", "RPL3", *[f"GENE{i}" for i in range(88)]]
    writer = SparseToZarr(
        csr_matrix(values),
        str(source),
        [f"cell{i}" for i in range(120)],
        names,
        mem_budget="512M",
        nthreads=2,
    )
    writer.dump()
    store = DataStore(
        str(source),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=2,
        mem_budget="512M",
    )
    store.cells.insert("batch", np.tile(["a", "b"], 60))
    store.cells.insert(
        "condition", np.tile(["control", "treated", "control", "treated"], 30)
    )
    store.cells.insert(
        "cell_type",
        np.repeat(["HELD_OUT_ALPHA", "HELD_OUT_BETA", "HELD_OUT_GAMMA"], 40),
    )
    return source
