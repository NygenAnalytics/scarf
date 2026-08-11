"""Tests for Stage 3 select_pca."""

import importlib
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from scarf.agent import (
    PcaSelectionResult,
    characterize_features,
    select_pca,
)
from scarf.datastore.datastore import DataStore
from scarf.features.gene_reference import write_reference_fixture
from scarf.writers import SparseToZarr

select_pca_module = importlib.import_module("scarf.agent.select_pca")


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


def _store(tmp_path: Path) -> DataStore:
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
    # Make some genes variable and keep overall sparsity realistic.
    dense[:, 9:] = dense[:, 9:] + rng.poisson(1.5, size=(n_cells, n_feats - 9)).astype(
        np.uint16
    )
    # Inflate mitochondrial counts in a subset so percentMito is informative.
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
    sample = np.array(["s1"] * 40 + ["s2"] * 40)
    store.cells.insert("sample_id", sample, overwrite=True)
    return store


def test_select_pca_runs_original_branch(tmp_path: Path) -> None:
    cache = _human_cache(tmp_path)
    store = _store(tmp_path)
    features = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
    )
    input_cells = int(np.asarray(store.cells.fetch_all("I"), dtype=bool).sum())
    result = select_pca(
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
    assert isinstance(result, PcaSelectionResult)
    assert result.status == "done"
    assert result.selectedBranch == "original"
    assert result.selectedPca is not None
    assert result.qcRetention["inputCells"] == input_cells
    assert result.qcRetention["retainedCells"] <= input_cells
    assert result.blacklistIndexes
    mito = next(
        item
        for item in features.assays[0]["families"]
        if item["family"] == "mitochondrial"
    )
    assert set(mito["featureIndexes"]).issubset(set(result.blacklistIndexes))
    assert any(item["action"] == "selectPcaBranch" for item in result.acceptedActions)

    # Consecutive rerun without resetting I should reuse the persisted Stage 3
    # input selection and not progressively tighten QC.
    first_retained = result.qcRetention["retainedCells"]
    second = select_pca(
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
    assert second.status == "done"
    assert second.qcRetention["retainedCells"] == first_retained
    assert any(entry.get("kind") == "reusedStage3Input" for entry in second.auditLog)


def test_select_pca_needs_input_for_ambiguous_sample_units(tmp_path: Path) -> None:
    cache = _human_cache(tmp_path)
    store = _store(tmp_path)
    store.cells.insert(
        "donor_id",
        np.array(["d1"] * 40 + ["d2"] * 40),
        overwrite=True,
    )
    features = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
    )
    covariates = {
        "columns": [
            {
                "name": "sample_id",
                "domain": "technical",
                "kind": "categorical",
            },
            {
                "name": "donor_id",
                "domain": "design",
                "kind": "categorical",
            },
        ],
        "coefficients": [],
        "confounding": [
            {"observationUnit": "sample_id"},
            {"observationUnit": "donor_id"},
        ],
    }
    before = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()
    result = select_pca(
        store,
        features=features,
        covariates=covariates,
    )
    assert result.status == "needsInput"
    assert result.needsInput is not None
    assert set(result.needsInput.options) == {"sample_id", "donor_id"}
    np.testing.assert_array_equal(
        np.asarray(store.cells.fetch_all("I"), dtype=bool),
        before,
    )


def test_select_pca_parses_coefficient_names(tmp_path: Path) -> None:
    cache = _human_cache(tmp_path)
    store = _store(tmp_path)
    features = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
    )
    covariates = {
        "status": "done",
        "columns": [
            {"name": "sample_id", "domain": "technical", "kind": "categorical"},
            {"name": "condition", "domain": "biological", "kind": "categorical"},
            {"name": "sex", "domain": "biological", "kind": "categorical"},
        ],
        "coefficients": [{"name": "condition", "kind": "categorical"}],
        "confounding": [{"observationUnit": "sample_id"}],
    }
    store.cells.insert(
        "condition",
        np.array(["ctrl"] * 40 + ["treat"] * 40),
        overwrite=True,
    )
    store.cells.insert(
        "sex",
        np.array(["F"] * 40 + ["M"] * 40),
        overwrite=True,
    )
    result = select_pca(
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
    assert result.status == "done"
    protected = set(result.diagnostics["protectedCovariates"])
    assert {"condition", "sex"}.issubset(protected)
    assert set(result.diagnostics["cellCycleCovariates"]).issubset(protected)
    roster = {
        item["name"]: item["role"]
        for item in result.diagnostics["diagnosticCovariates"]
    }
    assert roster["condition"] == "protected"
    assert roster["sex"] == "protected"


def test_select_pca_directed_nuisance_indexes_create_alternate_branch(
    tmp_path: Path,
) -> None:
    cache = _human_cache(tmp_path)
    store = _store(tmp_path)
    features = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
    )
    mito = next(
        item
        for item in features.assays[0]["families"]
        if item["family"] == "mitochondrial"
    )
    result = select_pca(
        store,
        features=features,
        directions={
            "sampleColumn": "sample_id",
            "pcaDims": 5,
            "topN": 10,
            "maxCells": float("inf"),
            "protectSex": True,
            "protectProliferation": True,
            "nuisanceGeneIndexes": mito["featureIndexes"],
        },
    )
    assert result.status == "done"
    assert any(branch["id"] == "nuisanceFiltered" for branch in result.branches)
    assert set(mito["featureIndexes"]).issubset(
        set(result.branches[1]["blacklistIndexes"])
    )


def test_select_pca_exports_unprotected_biology_as_nuisance(
    tmp_path: Path,
) -> None:
    cache = _human_cache(tmp_path)
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
        cacheDir=cache,
        allowDownload=False,
    )
    result = select_pca(
        store,
        features=features,
        covariates={
            "status": "done",
            "columns": [
                {
                    "name": "sample_id",
                    "domain": "technical",
                    "kind": "categorical",
                },
                {
                    "name": "library_metric",
                    "domain": "technical",
                    "kind": "continuous",
                },
                {"name": "sex", "domain": "biological", "kind": "categorical"},
            ],
            "coefficients": [],
            "confounding": [{"observationUnit": "sample_id"}],
        },
        directions={
            "sampleColumn": "sample_id",
            "pcaDims": 5,
            "topN": 10,
            "maxCells": float("inf"),
            "protectSex": False,
            "protectProliferation": False,
        },
    )

    assert result.status == "done"
    roster = {
        item["name"]: (item["kind"], item["role"])
        for item in result.diagnostics["diagnosticCovariates"]
    }
    assert roster["library_metric"] == ("continuous", "technical")
    assert roster["sex"] == ("categorical", "nuisance")
    assert result.diagnostics["cellCycleCovariates"]
    assert all(
        roster[column][1] == "nuisance"
        for column in result.diagnostics["cellCycleCovariates"]
    )


def test_select_pca_restores_input_selection_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = _human_cache(tmp_path)
    store = _store(tmp_path)
    features = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
    )
    before = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()

    def retain_two(run, sample_column):
        mask = np.zeros_like(before)
        mask[:2] = True
        run.store.cells.reset_key(key="I")
        run.store.cells.update_key(mask, key="I")
        return {
            "inputCells": int(before.sum()),
            "retainedCells": 2,
            "sampleColumn": sample_column,
            "attrs": [],
        }

    monkeypatch.setattr(select_pca_module, "_run_qc", retain_two)
    result = select_pca(
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

    assert result.status == "failed"
    np.testing.assert_array_equal(
        np.asarray(store.cells.fetch_all("I"), dtype=bool),
        before,
    )
    assert any(item["kind"] == "selectionRolledBack" for item in result.auditLog)


def test_select_pca_branch_gate_rejects_one_protected_covariate_regression() -> None:
    def branch(technical: float, first: float, second: float) -> dict:
        return {
            "diagnostics": {
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
            }
        }

    selected, rationale = select_pca_module._select_branch(
        branch(0.4, 0.3, 0.3),
        branch(0.2, 0.4, 0.2),
    )

    assert selected == "original"
    assert "sex" in rationale


def test_select_pca_rejects_alternate_that_cannot_match_dimensions(
    tmp_path: Path,
) -> None:
    cache = _human_cache(tmp_path)
    store = _store(tmp_path)
    features = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
    )
    result = select_pca(
        store,
        features=features,
        directions={
            "sampleColumn": "sample_id",
            "pcaDims": 5,
            "topN": 10,
            "maxCells": float("inf"),
            "protectSex": True,
            "protectProliferation": True,
            "nuisanceGeneIndexes": list(range(9, 20)),
        },
    )

    assert result.status == "done"
    assert result.selectedBranch == "original"
    alternate = next(
        branch for branch in result.branches if branch["id"] == "nuisanceFiltered"
    )
    assert alternate["status"] == "rejected"
    assert alternate["nFeatures"] <= alternate["requiredDims"]
    assert "pca" not in alternate
