"""Tests for characterize_covariates."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from scipy.sparse import csr_matrix

from scarf.agent import CovariateCharacterization, characterize_covariates
from scarf.agent.characterize_covariates import _is_embedding_column
from scarf.agent.types import Decision
from scarf.datastore.datastore import DataStore
from scarf.writers import SparseToZarr


def _function_model(answers: Mapping[str, Decision]) -> FunctionModel:
    queue = list(answers.items())

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        text = ""
        for message in messages:
            for part in getattr(message, "parts", []):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    text += content
        selected: Decision | None = None
        for key, decision in queue:
            if key in text:
                selected = decision
                break
        if selected is None:
            selected = next(iter(answers.values()))
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=tool.name,
                    args=selected.model_dump(),
                )
            ]
        )

    return FunctionModel(reply)


def _store_with_design(tmp_path: Path) -> DataStore:
    n_cells = 12
    matrix = csr_matrix(np.ones((n_cells, 3), dtype=np.uint16))
    location = tmp_path / "covariates.zarr"
    writer = SparseToZarr(
        matrix,
        str(location),
        cell_ids=[f"cell-{i}" for i in range(n_cells)],
        feature_ids=["g1", "g2", "g3"],
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
    # Two donors, two samples each, disease confounded with batch, cell type within sample.
    donor = np.array(["d1"] * 6 + ["d2"] * 6)
    sample = np.array(["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3 + ["s4"] * 3)
    batch = np.array(["b1"] * 6 + ["b2"] * 6)
    disease = np.array(["case"] * 6 + ["ctrl"] * 6)
    disease_ontology = np.array(["DOID:1"] * 6 + ["DOID:2"] * 6)
    cell_type = np.array(["alpha", "beta", "alpha", "beta", "alpha", "beta"] * 2)
    # Same partition as cell_type under different labels.
    author_cell_type = np.array(["AC", "BC", "AC", "BC", "AC", "BC"] * 2)
    # Constant within sample: valid design-table continuous technical.
    depth = np.array([1000.0] * 3 + [2000.0] * 3 + [3500.5] * 3 + [4800.25] * 3)
    # Varies within sample: must be excluded from the design table.
    umi_noise = np.linspace(100.0, 500.0, n_cells)
    store.cells.insert("donor", donor, overwrite=True)
    store.cells.insert("sample", sample, overwrite=True)
    store.cells.insert("batch", batch, overwrite=True)
    store.cells.insert("disease", disease, overwrite=True)
    store.cells.insert("disease_ontology_term_id", disease_ontology, overwrite=True)
    store.cells.insert("cell_type", cell_type, overwrite=True)
    store.cells.insert("author_cell_type", author_cell_type, overwrite=True)
    store.cells.insert("sequencing_depth", depth, overwrite=True)
    store.cells.insert("umi_noise", umi_noise, overwrite=True)
    # Single-level metadata must be dropped before domain/equivalence work.
    store.cells.insert("protocol", np.array(["v1"] * n_cells), overwrite=True)
    store.cells.insert("X_umap1", np.linspace(0, 1, n_cells), overwrite=True)
    store.cells.insert("X_umap2", np.linspace(1, 0, n_cells), overwrite=True)
    store.cells.insert(
        "X_pca-1", np.random.default_rng(0).normal(size=n_cells), overwrite=True
    )
    return store


def test_embedding_column_name_patterns() -> None:
    assert _is_embedding_column("X_umap1")
    assert _is_embedding_column("X_umap_1")
    assert _is_embedding_column("X_umap-1")
    assert _is_embedding_column("X_pca12")
    assert _is_embedding_column("scVI_3")
    assert _is_embedding_column("PHATE1")
    assert _is_embedding_column("FA_2")
    assert _is_embedding_column("diffmap1")
    assert not _is_embedding_column("donor")
    assert not _is_embedding_column("FA")
    assert not _is_embedding_column("batch")


def test_characterize_covariates_directions_only(tmp_path: Path) -> None:
    store = _store_with_design(tmp_path)
    result = characterize_covariates(
        store,
        studyContext="Case/control retina study across two donors.",
        model=None,
        directions={
            "columnDomains": {
                "donor": "design",
                "sample": "design",
                "batch": "technical",
                "sequencing_depth": "technical",
                "umi_noise": "technical",
                "disease": "biological",
                "cell_type": "biological",
                "author_cell_type": "biological",
            },
            "coefficientsOfInterest": ["disease", "cell_type"],
            "unitsOfInference": {
                "disease": {"observationUnit": "sample", "independentUnit": "donor"},
                "cell_type": {"observationUnit": "sample"},
            },
        },
    )
    assert isinstance(result, CovariateCharacterization)
    assert result.status == "done"
    names = {column["name"]: column for column in result.columns}
    assert names["disease"]["aliases"] == ["disease_ontology_term_id"]
    assert "disease_ontology_term_id" not in {
        column["name"] for column in result.columns if column["domain"] != "ignore"
    }
    assert names["X_umap1"]["domain"] == "ignore"
    assert names["X_pca-1"]["domain"] == "ignore"
    assert names["batch"]["kind"] == "categorical"
    assert names["sequencing_depth"]["kind"] == "continuous"

    drops = {
        entry["column"]: entry["kind"]
        for entry in result.auditLog
        if entry["kind"].startswith("drop")
    }
    assert drops["X_umap1"] == "dropEmbedding"
    assert drops["protocol"] == "dropConstant"
    assert any(kind == "dropAssayStat" for kind in drops.values())
    assert "protocol" not in {
        column["name"] for column in result.columns if column["domain"] != "ignore"
    }

    # donor, batch and disease partition the cells identically but sit in three
    # different domains, so they are a confounding finding rather than aliases.
    across = next(
        entry for entry in result.auditLog if entry["kind"] == "equivalentAcrossDomains"
    )
    assert set(across["columns"]) == {"donor", "batch", "disease"}
    assert across["domains"] == ["biological", "design", "technical"]
    # Same-domain equivalence cannot be resolved without a model, so it holds.
    kept_apart = next(
        entry for entry in result.auditLog if entry["kind"] == "equivalentKeptApart"
    )
    assert set(kept_apart["columns"]) == {"cell_type", "author_cell_type"}
    assert "author_cell_type" in {column["name"] for column in result.columns}

    scopes = {item["name"]: item["scope"] for item in result.coefficients}
    assert scopes["disease"] == "betweenUnit"
    assert scopes["cell_type"] == "withinUnit"
    assert any(entry["kind"] == "withinUnit" for entry in result.auditLog)
    assert len(result.confounding) == 1
    assert result.confounding[0]["coefficient"] == "disease"
    measures = {
        pair["technical"]: pair["association"].get("measure")
        for pair in result.confounding[0]["pairs"]
    }
    assert measures == {"batch": "cramersV", "sequencing_depth": "etaSquared"}
    varying = [
        entry
        for entry in result.auditLog
        if entry["kind"] == "technicalVariesWithinUnit"
    ]
    assert {entry["column"] for entry in varying} == {"umi_noise"}
    assert all(entry["coefficient"] == "disease" for entry in varying)


def test_characterize_covariates_invalid_direction_fails(tmp_path: Path) -> None:
    store = _store_with_design(tmp_path)
    result = characterize_covariates(
        store,
        directions={"columnDomains": {"not_a_column": "biological"}},
    )
    assert result.status == "failed"
    assert any("unknown columns" in note for note in result.notes)


def test_characterize_covariates_rejects_unsupported_domain_value(
    tmp_path: Path,
) -> None:
    store = _store_with_design(tmp_path)
    result = characterize_covariates(
        store,
        directions={"columnDomains": {"batch": "nuisance"}},
    )
    assert result.status == "failed"
    assert any("unsupported values" in note for note in result.notes)


def test_characterize_covariates_stays_headless_without_model(tmp_path: Path) -> None:
    store = _store_with_design(tmp_path)
    result = characterize_covariates(store)
    assert result.status == "done"
    assert result.decisions == []
    assert result.coefficients == []
    assert result.confounding == []
    unresolved = {
        entry["column"] for entry in result.auditLog if entry["kind"] == "domainUnknown"
    }
    assert {"donor", "sample", "batch", "disease"} <= unresolved


def test_characterize_covariates_function_model_path(tmp_path: Path) -> None:
    store = _store_with_design(tmp_path)
    answers = {
        "Assign a domain for cell metadata column donor": Decision(
            selectedId="domain:design",
            rationale="donor is sampling unit",
            evidenceIds=["domain:design"],
        ),
        "Assign a domain for cell metadata column sample": Decision(
            selectedId="domain:design",
            rationale="sample is observation unit",
            evidenceIds=["domain:design"],
        ),
        "Assign a domain for cell metadata column batch": Decision(
            selectedId="domain:technical",
            rationale="batch is technical",
            evidenceIds=["domain:technical"],
        ),
        "Assign a domain for cell metadata column disease": Decision(
            selectedId="domain:biological",
            rationale="disease is biology",
            evidenceIds=["domain:biological"],
        ),
        "Assign a domain for cell metadata column cell_type": Decision(
            selectedId="domain:biological",
            rationale="cell type is biology",
            evidenceIds=["domain:biological"],
        ),
        "Assign a domain for cell metadata column author_cell_type": Decision(
            selectedId="domain:biological",
            rationale="author annotation is biology",
            evidenceIds=["domain:biological"],
        ),
        "assign every cell to the same groups": Decision(
            selectedId="equivalent:cell_type",
            rationale="cell_type carries the readable labels",
            evidenceIds=["equivalent:cell_type"],
        ),
        "Assign a domain for cell metadata column sequencing_depth": Decision(
            selectedId="domain:technical",
            rationale="depth is technical",
            evidenceIds=["domain:technical"],
        ),
        "Assign a domain for cell metadata column umi_noise": Decision(
            selectedId="domain:technical",
            rationale="umi noise is technical",
            evidenceIds=["domain:technical"],
        ),
        "Should biological column disease": Decision(
            selectedId="coefficient:yes",
            rationale="primary contrast",
            evidenceIds=["coefficient:yes"],
        ),
        "Should biological column cell_type": Decision(
            selectedId="coefficient:no",
            rationale="composition only",
            evidenceIds=["coefficient:no"],
        ),
        "Choose the observation unit for coefficient disease": Decision(
            selectedId="unit:sample",
            rationale="one row per sample",
            evidenceIds=["unit:sample"],
        ),
        "Optional independent unit for coefficient disease": Decision(
            selectedId="independentUnit:donor",
            rationale="donor repeats",
            evidenceIds=["independentUnit:donor"],
        ),
    }
    result = characterize_covariates(
        store,
        studyContext="Case control across donors.",
        model=_function_model(answers),
    )
    assert result.status == "done"
    assert any(decision["task"] == "columnDomain" for decision in result.decisions)

    columns = {column["name"]: column for column in result.columns}
    assert columns["cell_type"]["aliases"] == ["author_cell_type"]
    assert "author_cell_type" not in columns
    collapsed = next(
        entry for entry in result.auditLog if entry["kind"] == "equivalentColumns"
    )
    assert collapsed["representative"] == "cell_type"
    assert collapsed["levels"] == "AC = alpha; BC = beta"

    coeffs = {item["name"]: item for item in result.coefficients}
    assert "disease" in coeffs
    assert coeffs["disease"]["scope"] == "betweenUnit"
    assert coeffs["disease"]["observationUnit"] == "sample"
    assert coeffs["disease"]["independentUnit"] == "donor"


def test_characterize_covariates_does_not_mutate_store(tmp_path: Path) -> None:
    store = _store_with_design(tmp_path)
    before = list(store.cells.columns)
    result = characterize_covariates(
        store,
        directions={
            "columnDomains": {"disease": "biological", "batch": "technical"},
            "coefficientsOfInterest": ["disease"],
            "unitsOfInference": {"disease": {"observationUnit": "batch"}},
        },
    )
    assert result.status == "done"
    assert set(store.cells.columns) == set(before)


def test_characterize_covariates_rejects_finer_independent_unit(tmp_path: Path) -> None:
    """Independent unit must be coarser than observation unit, never finer."""
    store = _store_with_design(tmp_path)
    result = characterize_covariates(
        store,
        directions={
            "columnDomains": {
                "donor": "design",
                "sample": "design",
                "batch": "technical",
                "disease": "biological",
            },
            "coefficientsOfInterest": ["disease"],
            # sample is finer than donor: would inflate the design table.
            "unitsOfInference": {
                "disease": {"observationUnit": "donor", "independentUnit": "sample"}
            },
        },
    )
    assert result.status == "done"
    assert any(entry["kind"] == "independentUnitFiner" for entry in result.auditLog)
    coeff = next(item for item in result.coefficients if item["name"] == "disease")
    assert coeff["observationUnit"] == "donor"
    assert coeff["independentUnit"] is None
    assert coeff["designRows"] == 2
    assert len(result.confounding) == 1
    assert result.confounding[0]["nRows"] == 2


def test_characterize_covariates_drops_directed_constant_coefficient(
    tmp_path: Path,
) -> None:
    store = _store_with_design(tmp_path)
    result = characterize_covariates(
        store,
        directions={
            "columnDomains": {"protocol": "technical", "disease": "biological"},
            "coefficientsOfInterest": ["protocol", "disease"],
            "unitsOfInference": {"disease": {"observationUnit": "donor"}},
        },
    )
    assert result.status == "done"
    drops = {
        entry["column"]: entry["kind"]
        for entry in result.auditLog
        if entry["kind"].startswith("drop")
    }
    assert drops["protocol"] == "dropConstant"
    assert any(
        entry["kind"] == "dropConstant" and "coefficientsOfInterest" in entry["detail"]
        for entry in result.auditLog
        if entry.get("column") == "protocol"
    )
    assert "protocol" not in {item["name"] for item in result.coefficients}


def test_characterize_covariates_rejects_vacuous_cell_id_unit(tmp_path: Path) -> None:
    store = _store_with_design(tmp_path)
    cell_ids = store.cells.fetch_all("ids")
    store.cells.insert("barcode", cell_ids, overwrite=True)
    result = characterize_covariates(
        store,
        directions={
            "columnDomains": {
                "barcode": "design",
                "disease": "biological",
                "batch": "technical",
            },
            "coefficientsOfInterest": ["disease"],
            "unitsOfInference": {"disease": {"observationUnit": "barcode"}},
        },
    )
    assert result.status == "done"
    assert any(
        entry["kind"] == "invalidObservationUnit"
        and entry.get("observationUnit") == "barcode"
        for entry in result.auditLog
    )
    coeff = next(item for item in result.coefficients if item["name"] == "disease")
    assert coeff["scope"] == "unresolvedUnit"
    assert coeff.get("observationUnit") is None


def test_characterize_covariates_rejects_auto_unique_per_cell_unit(
    tmp_path: Path,
) -> None:
    store = _store_with_design(tmp_path)
    # Only vacuous unit candidates: unique per cell identifiers.
    n = store.cells.N
    store.cells.insert(
        "barcode", np.array([f"bc{i}" for i in range(n)]), overwrite=True
    )
    result = characterize_covariates(
        store,
        directions={
            "columnDomains": {
                "barcode": "design",
                "disease": "biological",
            },
            "coefficientsOfInterest": ["disease"],
        },
    )
    assert result.status == "done"
    assert any(entry["kind"] == "noValidObservationUnit" for entry in result.auditLog)
    coeff = next(item for item in result.coefficients if item["name"] == "disease")
    assert coeff["scope"] == "unresolvedUnit"


def test_characterize_covariates_respects_subset_cell_key(tmp_path: Path) -> None:
    store = _store_with_design(tmp_path)
    keep = np.zeros(store.cells.N, dtype=bool)
    keep[:6] = True  # donor d1 only; disease is constant (case)
    store.cells.insert("subset", keep, overwrite=True)
    result = characterize_covariates(
        store,
        cellKey="subset",
        directions={
            "columnDomains": {
                "donor": "design",
                "sample": "design",
                "batch": "technical",
                "disease": "biological",
            },
            "coefficientsOfInterest": ["disease"],
            "unitsOfInference": {"disease": {"observationUnit": "sample"}},
        },
    )
    assert result.status == "done"
    assert "disease" not in {item["name"] for item in result.coefficients}
    assert any(
        entry["kind"].startswith("drop") and entry.get("column") == "disease"
        for entry in result.auditLog
    )
    disease_col = next(col for col in result.columns if col["name"] == "disease")
    assert "single-level" in disease_col["summary"]
    assert disease_col["domain"] == "ignore"


def test_characterize_covariates_avoids_bulk_metadata_loads(
    tmp_path: Path, monkeypatch
) -> None:
    store = _store_with_design(tmp_path)
    original_fetch = store.cells.fetch
    original_fetch_all = store.cells.fetch_all
    original_blocks = store.cells.iter_row_blocks
    inside_fetch = False
    direct_full_fetches: list[str] = []
    block_widths: list[int] = []

    def reject_frame(*_args, **_kwargs):
        raise AssertionError("characterization must not build a cell-level dataframe")

    def fetch_spy(column: str, key: str = "I"):
        nonlocal inside_fetch
        inside_fetch = True
        try:
            return original_fetch(column, key=key)
        finally:
            inside_fetch = False

    def fetch_all_spy(column: str):
        if not inside_fetch and column != "I":
            direct_full_fetches.append(column)
        return original_fetch_all(column)

    def blocks_spy(*, cell_key="I", columns=None, block_rows=None):
        names = tuple(columns or ())
        block_widths.append(len(names))
        return original_blocks(
            cell_key=cell_key,
            columns=names,
            block_rows=block_rows,
        )

    monkeypatch.setattr(store.cells, "to_pandas_dataframe", reject_frame)
    monkeypatch.setattr(store.cells, "fetch", fetch_spy)
    monkeypatch.setattr(store.cells, "fetch_all", fetch_all_spy)
    monkeypatch.setattr(store.cells, "iter_row_blocks", blocks_spy)
    result = characterize_covariates(
        store,
        directions={
            "columnDomains": {
                "donor": "design",
                "sample": "design",
                "batch": "technical",
                "disease": "biological",
            },
            "coefficientsOfInterest": ["disease"],
            "unitsOfInference": {
                "disease": {
                    "observationUnit": "sample",
                    "independentUnit": "donor",
                }
            },
        },
    )
    assert result.status == "done"
    assert direct_full_fetches == []
    assert block_widths
    assert max(block_widths) < len(store.cells.columns)
