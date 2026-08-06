"""Tests for characterize_features."""

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from scarf.agent import FeatureCharacterization, characterize_features
from scarf.datastore.datastore import DataStore
from scarf.features.gene_reference import write_reference_fixture
from scarf.writers import SparseToZarr


def _store(
    tmp_path: Path,
    *,
    feature_ids: list[str],
    feature_names: list[str] | None = None,
    assay_name: str = "RNA",
) -> DataStore:
    n_cells = 6
    n_feats = len(feature_ids)
    matrix = csr_matrix(np.ones((n_cells, n_feats), dtype=np.uint16))
    location = tmp_path / f"{assay_name.lower()}.zarr"
    writer = SparseToZarr(
        matrix,
        str(location),
        cell_ids=[f"cell-{i}" for i in range(n_cells)],
        feature_ids=feature_ids,
        feature_names=feature_names,
        assay_name=assay_name,
        mem_budget=64 * 1024 * 1024,
        nthreads=1,
    )
    writer.dump()
    return DataStore(
        str(location),
        default_assay=assay_name,
        min_features_per_cell=0,
        min_cells_per_feature=0,
        nthreads=1,
        mem_budget=64 * 1024 * 1024,
    )


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
            ("ENSG00000071082", "RPL31", "2"),
            ("ENSG00000229807", "XIST", "X"),
            ("ENSG00000129824", "RPS4Y1", "Y"),
            ("ENSG00000184640", "HIST1H4A", "6"),
            ("ENSG00000012048", "BRCA1", "17"),
        ],
    )
    return cache


def test_characterize_features_prefix_species_and_families(tmp_path: Path) -> None:
    cache = _human_cache(tmp_path)
    store = _store(
        tmp_path,
        feature_ids=[
            "ENSG00000075624",
            "ENSG00000198727",
            "ENSG00000071082",
            "ENSG00000229807",
            "ENSG00000129824",
            "ENSG00000184640",
            "ERCC-00002",
        ],
        feature_names=[
            "ACTB",
            "MT-CYB",
            "RPL31",
            "XIST",
            "RPS4Y1",
            "HIST1H4A",
            "ERCC-00002",
        ],
    )
    result = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
        directions={"maxExogenousCandidates": 5},
    )
    assert isinstance(result, FeatureCharacterization)
    assert result.status == "done"
    assay = result.assays[0]
    assert assay["species"] == "homo_sapiens"
    assert assay["speciesMethod"] == "ensemblPrefix"
    families = {item["family"]: item for item in assay["families"]}
    assert families["mitochondrial"]["examples"] == ["MT-CYB"]
    assert "RPL31" in families["ribosomal"]["examples"]
    assert set(families["sex"]["examples"]) == {"RPS4Y1", "XIST"}
    assert families["sex"]["method"] == "chromosome"
    assert families["sex"]["defaultExclude"] is False
    assert families["mitochondrial"]["defaultExclude"] is True
    assert families["histone"]["examples"] == ["HIST1H4A"]
    assert any(item["name"] == "ERCC-00002" for item in assay["exogenous"])
    assert any(entry["kind"] == "sexChromosomeTracked" for entry in result.auditLog)


def test_characterize_features_backfill_blank_names(tmp_path: Path) -> None:
    cache = _human_cache(tmp_path)
    ids = ["ENSG00000075624", "ENSG00000198727", "ENSG00000071082"]
    store = _store(tmp_path, feature_ids=ids, feature_names=["", "", ""])
    result = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
    )
    assay = result.assays[0]
    assert assay["symbolBackfill"]["nRecovered"] == 3
    families = {item["family"]: item for item in assay["families"]}
    assert families["mitochondrial"]["examples"] == ["MT-CYB"]


def test_characterize_features_backfill_when_names_are_ids(tmp_path: Path) -> None:
    cache = _human_cache(tmp_path)
    ids = [
        "ENSG00000075624",
        "ENSG00000198727",
        "ENSG00000071082",
    ]
    store = _store(tmp_path, feature_ids=ids, feature_names=ids)
    result = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
    )
    assay = result.assays[0]
    assert assay["symbolBackfill"]["nRecovered"] == 3
    families = {item["family"]: item for item in assay["families"]}
    assert families["mitochondrial"]["examples"] == ["MT-CYB"]


def test_characterize_features_directions_species(tmp_path: Path) -> None:
    cache = _human_cache(tmp_path)
    store = _store(
        tmp_path,
        feature_ids=["g1", "g2", "g3"],
        feature_names=["ACTB", "GAPDH", "BRCA1"],
    )
    result = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
        directions={"speciesByAssay": {"RNA": "homo_sapiens"}},
    )
    assert result.assays[0]["species"] == "homo_sapiens"
    assert result.assays[0]["speciesMethod"] == "directions"


def test_characterize_features_unknown_species_skips_families(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        feature_ids=["g1", "g2"],
        feature_names=["foo", "bar"],
    )
    result = characterize_features(
        store,
        cacheDir=tmp_path / "empty-cache",
        allowDownload=False,
    )
    assert result.status == "done"
    assay = result.assays[0]
    assert assay["species"] == "unknown"
    assert all(item.get("skipped") == "speciesUnknown" for item in assay["families"])


def test_characterize_features_does_not_mutate_store(tmp_path: Path) -> None:
    cache = _human_cache(tmp_path)
    store = _store(
        tmp_path,
        feature_ids=["ENSG00000075624", "ENSG00000111640"],
        feature_names=["ACTB", "GAPDH"],
    )
    before_ids = list(store.RNA.feats.fetch_all("ids"))
    before_names = list(store.RNA.feats.fetch_all("names"))
    result = characterize_features(store, cacheDir=cache, allowDownload=False)
    assert result.status == "done"
    assert list(store.RNA.feats.fetch_all("ids")) == before_ids
    assert list(store.RNA.feats.fetch_all("names")) == before_names


def test_characterize_features_invalid_decision_is_audited(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from scarf.agent.types import Decision

    cache = tmp_path / "cache"
    cache.mkdir()
    write_reference_fixture(
        cache / "homo_sapiens.tsv",
        species="homo_sapiens",
        release="test",
        rows=[("ENSG00000075624", "ACTB", "7")],
    )
    write_reference_fixture(
        cache / "mus_musculus.tsv",
        species="mus_musculus",
        release="test",
        rows=[("ENSMUSG00000029580", "Actb", "5")],
    )
    store = _store(
        tmp_path,
        feature_ids=["ENSG00000075624", "ENSMUSG00000029580"],
        feature_names=["ACTB", "Actb"],
    )

    def reply(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        bad = Decision(
            selectedId="species:not-a-species",
            rationale="invalid",
            evidenceIds=["species:homo_sapiens"],
        )
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args=bad.model_dump())]
        )

    result = characterize_features(
        store,
        cacheDir=cache,
        allowDownload=False,
        model=FunctionModel(reply),
    )
    assert result.status == "done"
    assert any(
        entry["kind"] == "decisionInvalid" and entry.get("task") == "species"
        for entry in result.auditLog
    )
