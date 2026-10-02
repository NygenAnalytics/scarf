"""Core pipeline delegation and measured correction acceptance boundaries."""

from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scarf.agent.evidence import inspect_source, open_store
from scarf.agent.execution import (
    execute_pipeline,
    finalist_evidence,
    summarize_candidate,
    validate_harmony,
    _doublet_concentration,
)
from scarf.agent.models import AnalysisConfig, Candidate
from tests.test_agent_evidence import runtime, study


def test_pipeline_adapter_uses_explicit_recipe_without_optional_work() -> None:
    received: dict[str, Any] = {}
    store = SimpleNamespace(
        pipeline=SimpleNamespace(run=lambda **kwargs: received.update(kwargs))
    )
    prepared = {
        "assay": "RNA",
        "cellKey": "I",
        "filtering": False,
        "blacklist": "(?i)^mt-",
        "retainedCells": 100,
        "availableFeatures": 80,
        "technicalBatchColumns": ["batch"],
        "snapshotColumns": ["condition", "batch"],
        "correctionEligible": True,
    }
    candidate = Candidate(candidateId="c0", hvgCount=30, pcaDims=5, neighborsK=5)
    execute_pipeline(store, prepared, candidate, AnalysisConfig(), label="screen")
    assert received["snapshot_columns"] == ["batch", "condition"]
    assert all(
        received[name] is False
        for name in ("umap", "cell_cycle", "paris", "doublets", "markers")
    )
    assert received["params"]["hvg"]["blacklist"] == "(?i)^mt-"
    assert received["params"]["leiden"]["partitions"] == [0.5, 0.75, 1.0, 1.25]
    execute_pipeline(
        store,
        prepared,
        candidate.model_copy(update={"useHarmony": True}),
        AnalysisConfig(scoreDoublets=True),
        label="finalist",
        resolution=0.5,
        markers=True,
    )
    assert received["doublets"] and received["markers"]
    assert received["params"]["harmony"]["batch_columns"] == ["batch"]
    assert "harmony_batch_columns" not in received
    assert received["params"]["leiden"]["selected"] == 0.5


def _matched() -> tuple[dict[str, Any], dict[str, Any]]:
    native: dict[str, Any] = {
        "parameters": {
            "hvgCount": 1000,
            "pcaDims": 21,
            "neighborsK": 11,
            "resolution": 1.0,
            "useHarmony": False,
        },
        "selection": {"artifact_id": "same_cells"},
        "features": {"artifact_id": "same_features"},
        "requiredSampleSupport": True,
        "diagnosticScope": {
            "method": "sampledCoordinatesKnn",
            "cellIdsSha256": "same_sample",
            "sampleCells": 10000,
        },
        "metrics": {
            "mixing": {"batch": 0.3},
            "protection": {"condition": {"cLISI": 0.9, "graphConnectivity": 0.9}},
            "markerSupportFraction": 0.9,
            "markerSpecificityMedian": 0.8,
            "doubletHighScoreConcentration": 2.0,
            "crossUnitSupport": 0.9,
        },
    }
    corrected = deepcopy(native)
    corrected["parameters"]["useHarmony"] = True
    corrected["metrics"]["mixing"]["batch"] = 0.5
    return native, corrected


def test_harmony_accepts_only_complete_matched_improvement() -> None:
    native, corrected = _matched()
    assert validate_harmony(native, corrected) == []
    corrected["metrics"]["mixing"]["batch"] = 0.34
    assert any("No batch" in value for value in validate_harmony(native, corrected))


@pytest.mark.parametrize(
    "damage",
    [
        "cells",
        "features",
        "resolution",
        "missing",
        "doublet",
        "biology",
        "marker",
        "sample",
        "other_batch",
    ],
)
def test_harmony_rejects_mismatch_missing_evidence_and_harm(damage: str) -> None:
    native, corrected = _matched()
    if damage == "cells":
        corrected["selection"] = {"artifact_id": "different"}
    elif damage == "features":
        corrected["features"] = {"artifact_id": "different"}
    elif damage == "resolution":
        corrected["parameters"]["resolution"] = 0.5
    elif damage == "missing":
        corrected["metrics"]["doubletHighScoreConcentration"] = None
    elif damage == "doublet":
        corrected["metrics"]["doubletHighScoreConcentration"] = 2.2
    elif damage == "biology":
        corrected["metrics"]["protection"]["condition"]["cLISI"] = 0.7
    elif damage == "marker":
        corrected["metrics"]["markerSupportFraction"] = 0.7
    elif damage == "sample":
        corrected["metrics"]["crossUnitSupport"] = 0.7
    else:
        native["metrics"]["mixing"]["other"] = 0.8
        corrected["metrics"]["mixing"]["other"] = 0.7
    assert validate_harmony(native, corrected)


def test_doublet_metric_uses_cohort_top_decile_without_a_probability_threshold() -> (
    None
):
    scores = np.arange(100, dtype=float) / 1000
    labels = np.repeat(["a", "b"], 50)
    assert _doublet_concentration(scores, labels) == 2
    assert _doublet_concentration(scores * 10, labels) == 2
    assert _doublet_concentration(np.full(100, np.nan), labels) is None


def test_diagnostic_sampling_is_bounded_stable_and_bound_to_the_gate() -> None:
    from scarf.agent.execution import _diagnostic_graph, _diagnostic_indices

    first = _diagnostic_indices(50000, 4444)
    assert len(first) == 10000
    assert np.array_equal(first, _diagnostic_indices(50000, 4444))
    assert len(np.unique(first)) == len(first)
    with pytest.raises(ValueError, match="limited to 10000"):
        _diagnostic_graph(
            None,
            None,
            Candidate(candidateId="c0", hvgCount=20, pcaDims=3, neighborsK=5),
            np.arange(10001),
            4444,
        )
    native, corrected = _matched()
    corrected["diagnosticScope"]["cellIdsSha256"] = "another_sample"
    assert any("sample" in reason for reason in validate_harmony(native, corrected))


@pytest.mark.parametrize("unscoreable", [True, False])
def test_candidate_summary_caps_silhouette_and_preserves_unscoreable_partitions(
    monkeypatch: Any, unscoreable: bool
) -> None:
    from scarf.metrics import cluster_selection

    class Run(dict[str, Any]):
        status = "completed"
        run_id = "synthetic"
        cells = SimpleNamespace(fetch=lambda key: np.zeros(4000, dtype=int))

    reference = SimpleNamespace(to_dict=lambda: {"artifact_id": "cells"})
    run = Run(pca="coordinates", leiden_0_5="labels", analysis_cell_selection=reference)
    run["leiden_0.5"] = run.pop("leiden_0_5")
    store = SimpleNamespace(
        memoryBytes=256 * 1024**2,
        load_artifact=lambda ref: {
            "data": np.zeros((4000, 3)),
            "values": np.zeros(4000),
        },
    )

    def selector(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["max_sample_size"] == 2000
        if unscoreable:
            raise ValueError(
                "No clustering candidate is silhouette-scoreable: one cluster"
            )
        return SimpleNamespace(
            candidate_keys=["leiden_0.5"], scores=[0.3], invalid_reasons=[None]
        )

    monkeypatch.setattr(cluster_selection, "select_clusters_by_silhouette", selector)
    monkeypatch.setattr(
        "scarf.agent.execution.representation_diagnostics", lambda *args: {}
    )
    evidence = summarize_candidate(
        store,
        run,
        Candidate(candidateId="c0", hvgCount=20, pcaDims=3, neighborsK=5),
        {},
        AnalysisConfig(),
    )
    assert evidence["silhouetteSampleCells"] == 2000
    assert len(evidence["partitions"]) == 1
    assert evidence["partitions"][0]["score"] == (None if unscoreable else 0.3)
    assert bool(evidence["limitations"]) is unscoreable


@pytest.mark.slow
def test_real_pipeline_screen_finalist_and_final_share_exact_scientific_artifacts(
    agent_rna_source: Path,
) -> None:
    source = agent_rna_source
    config = AnalysisConfig(
        hvgCount=20, pcaDims=3, neighborsK=5, resolutions=(0.5, 1.0)
    )
    supplied = study(protectedColumns=["condition"])
    prepared = inspect_source(source, supplied, config, runtime())
    store = open_store(source, config, runtime(), writable=True)
    candidate = Candidate(candidateId="c0", hvgCount=20, pcaDims=3, neighborsK=5)
    screen = execute_pipeline(store, prepared, candidate, config, label="screen")
    evidence = summarize_candidate(store, screen, candidate, prepared, config)
    assert {item["optionId"] for item in evidence["partitions"]} == {"c0:r0.5", "c0:r1"}
    assert all(row["count"] == 120 for row in evidence["partitions"])
    finalist = execute_pipeline(
        store,
        prepared,
        candidate,
        config,
        label="finalist",
        resolution=0.5,
        markers=True,
    )
    markers = finalist_evidence(store, finalist, candidate, prepared, config)
    assert sum(group["count"] for group in markers["clusters"]) == 120
    assert markers["diagnosticScope"]["sampleCells"] == 120
    assert markers["metrics"]["protection"]["condition"]["cLISI"] is not None
    assert set(group["clusterId"] for group in markers["clusters"]) == set(
        finalist.cells.fetch("clusters").astype(str)
    )
    assert markers["selection"] == evidence["selection"]
    # Reopening a completed computation uses core artifact reuse, not an
    # agent-owned cache or private Zarr orchestration hierarchy.
    repeated = execute_pipeline(
        store,
        prepared,
        candidate,
        config,
        label="repeated",
        resolution=0.5,
        markers=True,
    )
    for field in ("analysis_cell_selection", "clusters", "markers"):
        assert repeated[field] == finalist[field]
