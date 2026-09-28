"""Tests for characterize_features."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from scarf.agent.data_enrichment.characterization import (
    FeatureCharacterization,
    characterize_features,
)
from scarf.agent.data_enrichment.characterization import _load_or_fetch_reference
from scarf.datastore.datastore import DataStore
from scarf.features.gene_reference import GeneReference
from scarf.writers import SparseToZarr

from .gene_reference_fixtures import write_reference_fixture


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
    result = characterize_features(store, cacheDir=cache, allowDownload=False)
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
    assert all(item["class"] == "unresolved" for item in assay["exogenous"])
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


def test_characterize_features_validates_assays_before_loading() -> None:
    store = SimpleNamespace(assay_names=["RNA"])

    unknown_assay = characterize_features(store, assays=["ADT"])
    assert unknown_assay.status == "failed"
    assert unknown_assay.notes == ["unknown assays: ['ADT']"]


def test_characterize_features_stops_after_non_rna_identity_audit(
    tmp_path: Path,
) -> None:
    class FeatureTable:
        @staticmethod
        def fetch_all(column: str) -> np.ndarray:
            values = {
                "ids": np.array(["adt-1", "adt-2"]),
                "names": np.array(["CD3", "CD19"]),
            }
            return values[column]

    assay = SimpleNamespace(feats=FeatureTable())
    store = SimpleNamespace(
        assay_names=["ADT"],
        get_assay=lambda _name: assay,
    )

    result = characterize_features(store, cacheDir=tmp_path, allowDownload=False)

    assert result.status == "done"
    assert result.assays[0]["skipped"] == "familyPlanningNotApplicable"
    assert result.auditLog[0]["kind"] == "nonRnaAssay"


def test_characterize_features_audits_offline_unassessable_families(
    tmp_path: Path,
) -> None:
    ids = ["ENSG90000000001", "ENSG90000000002", "ENSG90000000003"]
    store = _store(tmp_path, feature_ids=ids, feature_names=ids)

    result = characterize_features(
        store,
        cacheDir=tmp_path / "empty-cache",
        allowDownload=False,
    )

    assert result.status == "done"
    assert result.assays[0]["species"] == "homo_sapiens"
    kinds = {entry["kind"] for entry in result.auditLog}
    assert {
        "referenceUnavailable",
        "familiesNotAssessable",
        "exogenousUnresolved",
    } <= kinds


def test_reference_loading_audits_mocked_download_success_and_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    characterize_features_module = import_module(
        "scarf.agent.data_enrichment.characterization"
    )
    reference = GeneReference(
        species="homo_sapiens",
        release="test",
        geneId=("ENSG1",),
        symbol=("GENE1",),
        chromosome=("1",),
    )
    monkeypatch.setattr(
        characterize_features_module,
        "load_reference",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        characterize_features_module,
        "ensure_reference",
        lambda *_args, **_kwargs: reference,
    )
    audit_log: list[dict] = []

    loaded = _load_or_fetch_reference(
        "homo_sapiens",
        cache_dir=tmp_path,
        allow_download=True,
        audit_log=audit_log,
        assay="RNA",
    )

    assert loaded is reference
    assert audit_log == [
        {
            "kind": "referenceDownloaded",
            "detail": "Cached gene reference for homo_sapiens release test",
            "assay": "RNA",
            "species": "homo_sapiens",
            "release": "test",
        }
    ]

    def fail_download(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        characterize_features_module,
        "ensure_reference",
        fail_download,
    )
    audit_log = []
    assert (
        _load_or_fetch_reference(
            "mus_musculus",
            cache_dir=tmp_path,
            allow_download=True,
            audit_log=audit_log,
            assay="RNA",
        )
        is None
    )
    assert audit_log[0]["kind"] == "referenceDownloadFailed"
    assert "offline" in audit_log[0]["detail"]


def test_reference_loading_skips_unknown_species(tmp_path: Path) -> None:
    audit_log: list[dict] = []
    assert (
        _load_or_fetch_reference(
            "unknown",
            cache_dir=tmp_path,
            allow_download=False,
            audit_log=audit_log,
            assay="RNA",
        )
        is None
    )
    assert audit_log == []


def test_characterize_features_audits_reference_release_misses(
    tmp_path: Path,
) -> None:
    cache = _human_cache(tmp_path)
    store = _store(
        tmp_path,
        feature_ids=[
            "ENSG00000075624",
            "ENSG00000111640",
            "ENSG00000012048",
            "ENSG99999999999",
        ],
        feature_names=["ACTB", "GAPDH", "BRCA1", "NOVEL"],
    )

    result = characterize_features(store, cacheDir=cache, allowDownload=False)

    miss = next(entry for entry in result.auditLog if entry["kind"] == "referenceMiss")
    assert miss["count"] == 1
    assert miss["examples"] == ["NOVEL"]
