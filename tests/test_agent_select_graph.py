"""Tests for Stage 4 select_graph."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from scarf.agent import (
    characterize_features,
    select_graph,
    select_pca,
)
from scarf.datastore.datastore import DataStore
from scarf.features.gene_reference import write_reference_fixture
from scarf.writers import SparseToZarr
import importlib

select_graph_module = importlib.import_module("scarf.agent.select_graph")


def _human_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    write_reference_fixture(
        cache / "homo_sapiens.tsv",
        species="homo_sapiens",
        release="test",
        rows=[
            ("ENSG00000075624", "ACTB", "7"),
            ("ENSG00000111640", "GAPDH", "12"),
            ("ENSG00000198727", "MT-CYB", "MT"),
            ("ENSG00000198712", "MT-CO2", "MT"),
            ("ENSG00000071082", "RPL31", "2"),
            ("ENSG00000198034", "RPS4X", "X"),
            ("ENSG00000229807", "XIST", "X"),
            ("ENSG00000129824", "RPS4Y1", "Y"),
            ("ENSG00000184640", "HIST1H4A", "6"),
            ("ENSG00000012048", "BRCA1", "17"),
            ("ENSG00000141510", "TP53", "17"),
            ("ENSG00000146648", "EGFR", "7"),
            ("ENSG00000157764", "BRAF", "7"),
            ("ENSG00000133703", "KRAS", "12"),
            ("ENSG00000171862", "PTEN", "10"),
            ("ENSG00000136997", "MYC", "8"),
            ("ENSG00000186081", "PCNA", "20"),
            ("ENSG00000132646", "MCM2", "3"),
            ("ENSG00000148773", "MKI67", "10"),
            ("ENSG00000094804", "CDC6", "17"),
        ],
    )
    return cache


def _store(tmp_path: Path, *, batch_effect: bool = False) -> DataStore:
    feature_ids = [
        "ENSG00000075624",
        "ENSG00000111640",
        "ENSG00000198727",
        "ENSG00000198712",
        "ENSG00000071082",
        "ENSG00000198034",
        "ENSG00000229807",
        "ENSG00000129824",
        "ENSG00000184640",
        "ENSG00000012048",
        "ENSG00000141510",
        "ENSG00000146648",
        "ENSG00000157764",
        "ENSG00000133703",
        "ENSG00000171862",
        "ENSG00000136997",
        "ENSG00000186081",
        "ENSG00000132646",
        "ENSG00000148773",
        "ENSG00000094804",
    ]
    feature_names = [
        "ACTB",
        "GAPDH",
        "MT-CYB",
        "MT-CO2",
        "RPL31",
        "RPS4X",
        "XIST",
        "RPS4Y1",
        "HIST1H4A",
        "BRCA1",
        "TP53",
        "EGFR",
        "BRAF",
        "KRAS",
        "PTEN",
        "MYC",
        "PCNA",
        "MCM2",
        "MKI67",
        "CDC6",
    ]
    n_cells = 80
    n_feats = len(feature_ids)
    rng = np.random.default_rng(0)
    dense = rng.poisson(0.3, size=(n_cells, n_feats)).astype(np.uint16)
    dense[:, 9:] = dense[:, 9:] + rng.poisson(1.5, size=(n_cells, n_feats - 9)).astype(
        np.uint16
    )
    if batch_effect:
        dense[np.r_[0:20, 40:60], 9:12] += 20
        dense[np.r_[20:40, 60:80], 12:15] += 20
        dense[:40, 15:19] += 8
    dense[:20, 2:4] = dense[:20, 2:4] + 30
    matrix = csr_matrix(dense)
    location = tmp_path / "rna.zarr"
    writer = SparseToZarr(
        matrix,
        str(location),
        cell_ids=[f"cell-{i}" for i in range(n_cells)],
        feature_ids=feature_ids,
        feature_names=feature_names,
        assay_name="RNA",
        mem_budget=64 * 1024 * 1024,
        nthreads=1,
    )
    writer.dump()
    store = DataStore(
        str(location),
        default_assay="RNA",
        min_features_per_cell=0,
        min_cells_per_feature=0,
        nthreads=1,
        mem_budget=64 * 1024 * 1024,
    )
    store.cells.insert(
        "sample_id",
        np.array(["s1"] * 40 + ["s2"] * 40),
        overwrite=True,
    )
    return store


def _pca_result(tmp_path: Path, store: DataStore):
    features = characterize_features(
        store,
        cacheDir=_human_cache(tmp_path),
        allowDownload=False,
    )
    return select_pca(
        store,
        features=features,
        directions={
            "sampleColumn": "sample_id",
            "pcaDims": 5,
            "topN": 10,
            "maxCells": float("inf"),
            "protectSex": True,
            "protectProliferation": True,
        },
    )


def _force_technical_signal(pca_dump: dict[str, Any]) -> dict[str, Any]:
    pca_dump["diagnostics"]["selectedSummary"] = {
        "technical": {
            "nFlaggedPcs": 2,
            "meanAssociation": 0.4,
            "maxAssociation": 0.5,
            "flaggedPcs": [1, 2],
            "byCovariate": {},
        },
        "protected": {
            "nFlaggedPcs": 0,
            "meanAssociation": 0.2,
            "maxAssociation": 0.2,
            "flaggedPcs": [],
            "byCovariate": {},
        },
        "associationFloor": 0.1,
    }
    _set_diagnostic_role(
        pca_dump,
        name="sample_id",
        kind="categorical",
        role="technical",
    )
    pca_dump["diagnostics"]["protectSex"] = True
    pca_dump["diagnostics"]["protectProliferation"] = True
    return pca_dump


def _set_diagnostic_role(
    pca_dump: dict[str, Any],
    *,
    name: str,
    kind: str,
    role: str,
) -> None:
    diagnostics = pca_dump["diagnostics"]
    roster = [
        item
        for item in diagnostics.get("diagnosticCovariates", [])
        if item["name"] != name
    ]
    roster.append(
        {
            "name": name,
            "kind": kind,
            "role": role,
            "source": "testFixture",
        }
    )
    diagnostics["diagnosticCovariates"] = roster
    for roster_role, key in (
        ("technical", "technicalCovariates"),
        ("nuisance", "nuisanceCovariates"),
        ("protected", "protectedCovariates"),
    ):
        diagnostics[key] = sorted(
            item["name"] for item in roster if item["role"] == roster_role
        )


def _assert_published(store: DataStore, result) -> None:
    state = store.get_assay_state("RNA")
    assert state is not None
    assert state.connectivity_map is not None
    assert state.connectivity_map.to_dict() == result.selectedGraph


def test_select_graph_skips_harmony_without_technical_signal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    pca = _pca_result(tmp_path, store)
    assert pca.status == "done"
    assert "protectSex" in pca.diagnostics
    assert "cellCycleSummary" in pca.diagnostics
    result = select_graph(
        store,
        pca=pca,
        directions={
            "neighborK": 5,
            "annParams": {"ann_m": 8, "ann_efc": 20, "ann_ef": 20},
            "harmonyParams": {"nclust": 4},
        },
    )
    assert result.status == "done"
    assert result.selectedBranch == "uncorrected"
    assert result.selectedGraph is not None
    assert any(entry.get("kind") == "harmonySkipped" for entry in result.auditLog)
    assert result.diagnostics["cellKey"] == "I"
    _assert_published(store, result)


def test_select_graph_blocks_confounded_batch_columns(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.cells.insert(
        "condition",
        np.array(["ctrl"] * 40 + ["treat"] * 40),
        overwrite=True,
    )
    pca = _pca_result(tmp_path, store)
    pca_dump = _force_technical_signal(pca.model_dump())
    _set_diagnostic_role(
        pca_dump,
        name="condition",
        kind="categorical",
        role="protected",
    )
    pca_dump["diagnostics"]["selectedSummary"]["protected"] = {
        "nFlaggedPcs": 1,
        "meanAssociation": 0.3,
        "maxAssociation": 0.3,
        "flaggedPcs": [1],
        "byCovariate": {},
    }
    covariates = {
        "status": "done",
        "columns": [
            {"name": "sample_id", "domain": "technical", "kind": "categorical"},
            {"name": "condition", "domain": "biological", "kind": "categorical"},
        ],
        "coefficients": [{"name": "condition", "kind": "categorical"}],
        "confounding": [{"observationUnit": "sample_id"}],
        "technicalNesting": [],
    }
    result = select_graph(
        store,
        pca=pca_dump,
        covariates=covariates,
        directions={
            "neighborK": 5,
            "annParams": {"ann_m": 8, "ann_efc": 20, "ann_ef": 20},
            "harmonyBatchColumns": ["sample_id"],
            "harmonyParams": {"nclust": 4},
        },
    )
    assert result.status == "done"
    assert result.selectedBranch == "uncorrected"
    assert result.harmonyBatchColumns == []
    assert any(entry.get("kind") == "batchColumnBlocked" for entry in result.auditLog)
    _assert_published(store, result)


def test_select_graph_checks_joint_multi_batch_confounding(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.cells.insert(
        "lane_id",
        np.array(["L1", "L2"] * 40),
        overwrite=True,
    )
    store.cells.insert(
        "condition",
        np.array(["ctrl"] * 40 + ["treat"] * 40),
        overwrite=True,
    )
    pca_dump = _force_technical_signal(_pca_result(tmp_path, store).model_dump())
    _set_diagnostic_role(
        pca_dump,
        name="lane_id",
        kind="categorical",
        role="technical",
    )
    _set_diagnostic_role(
        pca_dump,
        name="condition",
        kind="categorical",
        role="protected",
    )
    result = select_graph(
        store,
        pca=pca_dump,
        covariates={
            "status": "done",
            "columns": [
                {"name": "sample_id", "domain": "technical", "kind": "categorical"},
                {"name": "lane_id", "domain": "technical", "kind": "categorical"},
                {"name": "condition", "domain": "biological", "kind": "categorical"},
            ],
            "coefficients": [{"name": "condition", "kind": "categorical"}],
            "confounding": [{"observationUnit": "sample_id"}],
            "technicalNesting": [],
        },
        directions={
            "neighborK": 5,
            "harmonyBatchColumns": ["sample_id", "lane_id"],
            "harmonyParams": {"nclust": 4},
        },
    )

    assert result.status == "done"
    assert result.harmonyBatchColumns == []
    blocked = [
        item["detail"]
        for item in result.auditLog
        if item["kind"] == "batchColumnBlocked"
    ]
    assert blocked
    assert all("sample_id" in detail and "lane_id" in detail for detail in blocked)


def test_select_graph_rejects_biological_batch_columns(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.cells.insert(
        "cell_type",
        np.array(["T"] * 40 + ["B"] * 40),
        overwrite=True,
    )
    pca = _pca_result(tmp_path, store)
    pca_dump = _force_technical_signal(pca.model_dump())
    result = select_graph(
        store,
        pca=pca_dump,
        covariates={
            "status": "done",
            "columns": [
                {"name": "sample_id", "domain": "technical", "kind": "categorical"},
                {"name": "cell_type", "domain": "biological", "kind": "categorical"},
            ],
            "coefficients": [],
            "confounding": [{"observationUnit": "sample_id"}],
            "technicalNesting": [],
        },
        directions={
            "neighborK": 5,
            "annParams": {"ann_m": 8, "ann_efc": 20, "ann_ef": 20},
            "harmonyBatchColumns": ["cell_type"],
            "harmonyParams": {"nclust": 4},
        },
    )
    assert result.status == "failed"
    assert result.harmonyBatchColumns == []
    assert any(entry.get("kind") == "invalidHarmonyDesign" for entry in result.auditLog)


def test_select_graph_preflight_rejects_invalid_directions_before_graph(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    pca = _pca_result(tmp_path, store)
    pca_dump = _force_technical_signal(pca.model_dump())
    before = store.get_assay_state("RNA")
    before_graph = None if before is None else before.connectivity_map
    result = select_graph(
        store,
        pca=pca_dump,
        directions={
            "neighborK": 5,
            "harmonyBatchColumns": "sample_id",
        },
    )
    assert result.status == "failed"
    assert "harmonyBatchColumns" in result.notes[0]
    after = store.get_assay_state("RNA")
    after_graph = None if after is None else after.connectivity_map
    assert after_graph == before_graph


def test_select_graph_preflight_rejects_neighbor_k_below_three(tmp_path: Path) -> None:
    store = _store(tmp_path)
    pca = _pca_result(tmp_path, store)
    result = select_graph(
        store,
        pca=pca,
        directions={"neighborK": 2},
    )
    assert result.status == "failed"
    assert "neighborK" in result.notes[0]


def test_cheap_gate_rejects_clear_technical_worsening() -> None:
    baseline = {
        "summary": {
            "technical": {"meanAssociation": 0.2},
            "protected": {"meanAssociation": 0.3},
        }
    }
    harmony = {
        "summary": {
            "technical": {"meanAssociation": 0.5},
            "protected": {"meanAssociation": 0.3},
        }
    }
    decision, detail = select_graph_module._cheap_gate(baseline, harmony)
    assert decision == "reject"
    assert "worsened" in detail


def test_cheap_gate_continues_when_inconclusive() -> None:
    baseline = {
        "summary": {
            "technical": {"meanAssociation": 0.4},
            "protected": {"meanAssociation": 0.3},
        }
    }
    harmony = {
        "summary": {
            "technical": {"meanAssociation": 0.41},
            "protected": {"meanAssociation": 0.29},
        }
    }
    decision, detail = select_graph_module._cheap_gate(baseline, harmony)
    assert decision == "continue"
    assert "inconclusive" in detail or "non-regressive" in detail


def test_graph_branch_gate_rejects_one_protected_covariate_regression() -> None:
    def branch(technical: float, first: float, second: float) -> dict[str, Any]:
        return {
            "coordinateSummary": {
                "summary": {
                    "technical": {
                        "byCovariate": {"batch": {"meanAssociation": technical}}
                    },
                    "nuisance": {"byCovariate": {}},
                    "protected": {
                        "byCovariate": {
                            "condition": {"meanAssociation": first},
                            "sex": {"meanAssociation": second},
                        }
                    },
                }
            },
            "graphMetrics": {},
        }

    selected, _ = select_graph_module._select_graph_branch(
        branch(0.4, 0.3, 0.3),
        branch(0.2, 0.4, 0.2),
    )

    assert selected == "uncorrected"


def test_select_graph_rejects_harmony_on_clear_cheap_gate_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    pca = _pca_result(tmp_path, store)
    pca_dump = _force_technical_signal(pca.model_dump())

    def reject_gate(baseline: Any, harmony: Any) -> tuple[str, str]:
        _ = baseline, harmony
        return "reject", "forced cheap-gate rejection"

    monkeypatch.setattr(select_graph_module, "_cheap_gate", reject_gate)
    result = select_graph(
        store,
        pca=pca_dump,
        covariates={
            "status": "done",
            "columns": [
                {"name": "sample_id", "domain": "technical", "kind": "categorical"},
            ],
            "coefficients": [],
            "confounding": [{"observationUnit": "sample_id"}],
            "technicalNesting": [],
        },
        directions={
            "neighborK": 5,
            "annParams": {"ann_m": 8, "ann_efc": 20, "ann_ef": 20},
            "harmonyBatchColumns": ["sample_id"],
            "harmonyParams": {"nclust": 4},
        },
    )
    assert result.status == "done"
    assert result.selectedBranch == "uncorrected"
    assert len(result.branches) == 2
    assert result.branches[1]["rejectedBeforeGraph"] is True
    assert "annIndex" not in result.branches[1]
    _assert_published(store, result)


def test_select_graph_continues_after_inconclusive_cheap_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    pca = _pca_result(tmp_path, store)
    pca_dump = _force_technical_signal(pca.model_dump())

    def continue_gate(baseline: Any, harmony: Any) -> tuple[str, str]:
        _ = baseline, harmony
        return "continue", "inconclusive but non-regressive"

    monkeypatch.setattr(select_graph_module, "_cheap_gate", continue_gate)
    result = select_graph(
        store,
        pca=pca_dump,
        covariates={
            "status": "done",
            "columns": [
                {"name": "sample_id", "domain": "technical", "kind": "categorical"},
            ],
            "coefficients": [],
            "confounding": [{"observationUnit": "sample_id"}],
            "technicalNesting": [],
        },
        directions={
            "neighborK": 5,
            "annParams": {"ann_m": 8, "ann_efc": 20, "ann_ef": 20},
            "harmonyBatchColumns": ["sample_id"],
            "harmonyParams": {"nclust": 4},
        },
    )
    assert result.status == "done"
    assert result.selectedGraph is not None
    assert any(branch["id"] == "harmony" for branch in result.branches)
    harmony = next(branch for branch in result.branches if branch["id"] == "harmony")
    assert harmony.get("rejectedBeforeGraph") is False
    assert "graph" in harmony
    assert harmony["graphMetrics"]["knnLoc"]
    assert harmony["graphMetrics"]["graphLoc"]
    assert harmony["graphMetrics"]["nNeighbors"] == 5
    assert result.branches[0]["graphMetrics"]["nNeighbors"] == 5
    assert result.diagnostics["neighborK"] == 5
    assert result.diagnostics["comparisonTechnicalCovariates"] == ["sample_id"]
    _assert_published(store, result)


def test_select_graph_uses_matched_technical_covariates_for_comparisons(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path)
    store.cells.insert(
        "lane_id",
        np.array(["L1"] * 20 + ["L2"] * 20 + ["L3"] * 20 + ["L4"] * 20),
        overwrite=True,
    )
    pca = _pca_result(tmp_path, store)
    pca_dump = _force_technical_signal(pca.model_dump())
    _set_diagnostic_role(
        pca_dump,
        name="lane_id",
        kind="categorical",
        role="technical",
    )
    seen: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def continue_gate(baseline: Any, harmony: Any) -> tuple[str, str]:
        base_tech = set(
            (baseline["summary"]["technical"].get("byCovariate") or {}).keys()
        )
        harm_tech = set(
            (harmony["summary"]["technical"].get("byCovariate") or {}).keys()
        )
        seen.append((tuple(sorted(base_tech)), tuple(sorted(harm_tech))))
        return "continue", "inconclusive but non-regressive"

    monkeypatch.setattr(select_graph_module, "_cheap_gate", continue_gate)
    result = select_graph(
        store,
        pca=pca_dump,
        covariates={
            "status": "done",
            "columns": [
                {"name": "sample_id", "domain": "technical", "kind": "categorical"},
                {"name": "lane_id", "domain": "technical", "kind": "categorical"},
            ],
            "coefficients": [],
            "confounding": [{"observationUnit": "sample_id"}],
            "technicalNesting": [],
        },
        directions={
            "neighborK": 5,
            "annParams": {"ann_m": 8, "ann_efc": 20, "ann_ef": 20},
            "harmonyBatchColumns": ["sample_id"],
            "harmonyParams": {"nclust": 4},
        },
    )
    assert result.status == "done"
    assert seen
    assert seen[0][0] == seen[0][1]
    assert set(seen[0][0]) == {"lane_id", "sample_id"}
    base_ilisi = set(
        (result.branches[0]["graphMetrics"].get("ilisiByBatch") or {}).keys()
    )
    harm = next(branch for branch in result.branches if branch["id"] == "harmony")
    harm_ilisi = set((harm["graphMetrics"].get("ilisiByBatch") or {}).keys())
    assert base_ilisi == harm_ilisi == {"lane_id", "sample_id"}


def test_select_graph_rejects_uncharacterized_directed_batch_without_signal(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    pca = _pca_result(tmp_path, store)
    before = store.get_assay_state("RNA")
    before_graph = None if before is None else before.connectivity_map

    result = select_graph(
        store,
        pca=pca,
        covariates={
            "status": "done",
            "columns": [
                {"name": "sample_id", "domain": "technical", "kind": "categorical"}
            ],
            "coefficients": [],
            "confounding": [{"observationUnit": "sample_id"}],
            "technicalNesting": [],
        },
        directions={"harmonyBatchColumns": ["unknown_batch"]},
    )

    assert result.status == "failed"
    assert "not characterized" in result.notes[0]
    after = store.get_assay_state("RNA")
    assert (None if after is None else after.connectivity_map) == before_graph


def test_select_graph_rejects_directed_harmony_without_covariates(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    pca = _force_technical_signal(_pca_result(tmp_path, store).model_dump())

    result = select_graph(
        store,
        pca=pca,
        directions={"harmonyBatchColumns": ["sample_id"]},
    )

    assert result.status == "failed"
    assert any(
        item["kind"] == "missingCovariatesForHarmony" for item in result.auditLog
    )


def test_select_graph_skips_harmony_when_memory_estimate_exceeds_budget(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    pca = _force_technical_signal(_pca_result(tmp_path, store).model_dump())
    store.memoryBytes = 1

    result = select_graph(
        store,
        pca=pca,
        covariates={
            "status": "done",
            "columns": [
                {"name": "sample_id", "domain": "technical", "kind": "categorical"}
            ],
            "coefficients": [],
            "confounding": [{"observationUnit": "sample_id"}],
            "technicalNesting": [],
        },
        directions={
            "neighborK": 5,
            "harmonyBatchColumns": ["sample_id"],
            "harmonyParams": {"nclust": 4},
        },
    )

    assert result.status == "done"
    assert result.selectedBranch == "uncorrected"
    assert result.harmonyBatchColumns == []
    assert any(
        item["kind"] == "harmonySkipped" and "memory budget" in item["detail"]
        for item in result.auditLog
    )


def test_select_graph_rejects_drifted_pca_selection_before_graph(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    pca = _pca_result(tmp_path, store)
    mask = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()
    selected = int(np.flatnonzero(mask)[0])
    mask[selected] = False
    store.cells.reset_key(key="I")
    store.cells.update_key(mask, key="I")

    result = select_graph(store, pca=pca)

    assert result.status == "failed"
    assert any(item["kind"] == "invalidPcaSelection" for item in result.auditLog)
    state = store.get_assay_state("RNA")
    assert state is not None
    assert state.connectivity_map is None


def test_stage3_stage4_preserve_complete_diagnostic_roster(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.cells.insert(
        "sex",
        np.array(["F", "M"] * 40),
        overwrite=True,
    )
    store.cells.insert(
        "library_metric",
        np.linspace(0.0, 1.0, 80),
        overwrite=True,
    )
    features = characterize_features(
        store,
        cacheDir=_human_cache(tmp_path),
        allowDownload=False,
    )
    covariates = {
        "status": "done",
        "columns": [
            {"name": "sample_id", "domain": "technical", "kind": "categorical"},
            {
                "name": "library_metric",
                "domain": "technical",
                "kind": "continuous",
            },
            {"name": "sex", "domain": "biological", "kind": "categorical"},
        ],
        "coefficients": [],
        "confounding": [{"observationUnit": "sample_id"}],
        "technicalNesting": [],
    }
    pca = select_pca(
        store,
        features=features,
        covariates=covariates,
        directions={
            "sampleColumn": "sample_id",
            "pcaDims": 5,
            "topN": 10,
            "maxCells": float("inf"),
            "protectSex": True,
            "protectProliferation": True,
        },
    )
    pca_dump = _force_technical_signal(pca.model_dump())

    result = select_graph(
        store,
        pca=pca_dump,
        covariates=covariates,
        directions={
            "neighborK": 5,
            "annParams": {"ann_m": 8, "ann_efc": 20, "ann_ef": 20},
            "harmonyBatchColumns": ["sample_id"],
            "harmonyParams": {"nclust": 4},
        },
    )

    assert result.status == "done"
    baseline = next(
        branch for branch in result.branches if branch["id"] == "uncorrected"
    )
    summary = baseline["coordinateSummary"]["summary"]
    assert {"sample_id", "library_metric"} == set(summary["technical"]["byCovariate"])
    assert "sex" in summary["protected"]["byCovariate"]
    assert set(pca.diagnostics["cellCycleCovariates"]).issubset(
        summary["protected"]["byCovariate"]
    )
    support = baseline["sampleSupport"]
    assert support["definition"] == "selectedCellCountsByObservationUnit"
    assert support["branchInvariant"] is True
    assert "crossSampleNeighborRate" not in support
    _assert_published(store, result)


def test_stage3_signal_can_select_and_publish_harmony(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, batch_effect=True)
    features = characterize_features(
        store,
        cacheDir=_human_cache(tmp_path),
        allowDownload=False,
    )
    covariates = {
        "status": "done",
        "columns": [
            {"name": "sample_id", "domain": "technical", "kind": "categorical"}
        ],
        "coefficients": [],
        "confounding": [{"observationUnit": "sample_id"}],
        "technicalNesting": [],
    }
    pca = select_pca(
        store,
        features=features,
        covariates=covariates,
        directions={
            "sampleColumn": "sample_id",
            "pcaDims": 5,
            "topN": 10,
            "maxCells": float("inf"),
            "protectSex": True,
            "protectProliferation": False,
        },
    )
    assert pca.status == "done"
    assert pca.diagnostics["selectedSummary"]["technical"]["nFlaggedPcs"] > 0
    for column in pca.diagnostics["cellCycleCovariates"]:
        kind = next(
            item["kind"]
            for item in pca.diagnostics["diagnosticCovariates"]
            if item["name"] == column
        )
        values = (
            np.array(["G1"] * 80)
            if kind == "categorical"
            else np.zeros(80, dtype=float)
        )
        store.cells.insert(column, values, overwrite=True)

    result = select_graph(
        store,
        pca=pca,
        covariates=covariates,
        directions={
            "neighborK": 15,
            "annParams": {"ann_m": 8, "ann_efc": 30, "ann_ef": 30},
            "harmonyBatchColumns": ["sample_id"],
            "harmonyParams": {"nclust": 4, "theta": 4.0},
        },
    )

    assert result.status == "done"
    compact_evidence = [
        {
            "id": branch["id"],
            "technical": branch["coordinateSummary"]["summary"]["technical"],
            "nuisance": branch["coordinateSummary"]["summary"]["nuisance"],
            "ilisi": (branch.get("graphMetrics") or {}).get("ilisiByBatch"),
            "cheapGate": branch.get("cheapGate"),
        }
        for branch in result.branches
    ]
    assert result.selectedBranch == "harmony", {
        "notes": result.notes,
        "evidence": compact_evidence,
    }
    assert result.selectedCoordinates is not None
    assert result.selectedCoordinates["kind"] == "batch_correction"
    state = store.get_assay_state("RNA")
    assert state is not None
    assert state.batch_correction is not None
    assert state.batch_correction.to_dict() == result.selectedCoordinates
    _assert_published(store, result)


def test_phase_harmony_requires_and_uses_guarded_stage3_evidence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    features = characterize_features(
        store,
        cacheDir=_human_cache(tmp_path),
        allowDownload=False,
    )
    covariates = {
        "status": "done",
        "columns": [
            {"name": "sample_id", "domain": "technical", "kind": "categorical"}
        ],
        "coefficients": [],
        "confounding": [{"observationUnit": "sample_id"}],
        "technicalNesting": [],
    }
    pca = select_pca(
        store,
        features=features,
        covariates=covariates,
        directions={
            "sampleColumn": "sample_id",
            "pcaDims": 5,
            "topN": 10,
            "maxCells": float("inf"),
            "protectSex": True,
            "protectProliferation": False,
        },
    )
    phase_column = pca.cellCycle["phaseColumn"]
    store.cells.insert(
        phase_column,
        np.array(["G1", "S"] * 40),
        overwrite=True,
    )
    pca_dump = pca.model_dump()
    pca_dump["diagnostics"]["cellCycleSummary"]["byCovariate"][phase_column] = {
        "nAssociations": 5,
        "meanAssociation": 0.4,
        "maxAssociation": 0.5,
    }
    pca_dump["diagnostics"]["selectedSummary"]["nuisance"]["nFlaggedPcs"] = 1

    result = select_graph(
        store,
        pca=pca_dump,
        covariates=covariates,
        directions={
            "neighborK": 5,
            "harmonyBatchColumns": ["sample_id", phase_column],
            "harmonyParams": {"nclust": 4},
            "allowPhaseHarmony": True,
        },
    )

    assert result.status == "done"
    assert phase_column in result.harmonyBatchColumns
    assert any(item["kind"] == "phaseHarmonyCandidate" for item in result.auditLog)
    assert any(branch["id"] == "harmony" for branch in result.branches)
