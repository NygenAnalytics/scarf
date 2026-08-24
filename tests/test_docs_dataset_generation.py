import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from scripts import regenerate_docs_datasets as generator


def test_labelled_cluster_mask_rejects_blank_and_nan_values():
    mask = generator._labelled_cluster_mask(
        ["B", "nan", "NaN", "", "  ", "CD4 T"],
    )

    assert mask.tolist() == [True, False, False, False, False, True]


def test_kang_recipes_physically_subset_using_legacy_annotations():
    for dataset in (
        generator.KANG_CONTROL_DATASET,
        generator.KANG_STIMULATED_DATASET,
    ):
        recipe = generator.RECIPES[dataset]

        assert isinstance(recipe, generator.DerivedDatasetRecipe)
        assert recipe.source_datasets == (f"{dataset}{generator.LEGACY_SUFFIX}",)
        assert recipe.derive is generator._derive_labelled_kang_store


def test_derived_sources_prefer_stores_built_in_same_run(tmp_path):
    local = tmp_path / "local"
    (local / generator.STORE_NAME).mkdir(parents=True)
    remote = tmp_path / "remote"
    (remote / generator.STORE_NAME).mkdir(parents=True)

    class Repository:
        def __init__(self) -> None:
            self.downloads: list[str] = []

        def download_dataset(
            self,
            name: str,
            destination: str | Path,
            *,
            zarr: bool,
        ) -> Path:
            assert Path(destination) == tmp_path / "work"
            assert zarr is True
            self.downloads.append(name)
            return remote

    repository = Repository()
    resolved = generator._resolve_derived_sources(
        repository=repository,
        source_datasets=["local_source", "remote_source"],
        local_sources={"local_source": local},
        work=tmp_path / "work",
    )

    assert resolved == {
        "local_source": local,
        "remote_source": remote,
    }
    assert repository.downloads == ["remote_source"]


def test_external_source_download_verifies_checksum(tmp_path, monkeypatch):
    payload = b"checksum-pinned source"
    source = generator.ExternalSource(
        filename="source.bin",
        url="https://example.test/source.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    requested: list[tuple[str, int]] = []

    def open_source(url: str, *, timeout: int) -> BytesIO:
        requested.append((url, timeout))
        return BytesIO(payload)

    monkeypatch.setattr(generator, "urlopen", open_source)
    downloaded = generator._download_external_source(source, tmp_path)

    assert downloaded.read_bytes() == payload
    assert requested == [
        (source.url, generator.EXTERNAL_DOWNLOAD_TIMEOUT_SECONDS),
    ]

    requested.clear()
    assert generator._download_external_source(source, tmp_path) == downloaded
    assert requested == []


def test_external_source_checksum_failure_removes_partial_file(
    tmp_path,
    monkeypatch,
):
    source = generator.ExternalSource(
        filename="source.bin",
        url="https://example.test/source.bin",
        sha256="0" * 64,
    )
    monkeypatch.setattr(
        generator,
        "urlopen",
        lambda _url, **_kwargs: BytesIO(b"wrong"),
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        generator._download_external_source(source, tmp_path)

    assert not (tmp_path / "source.bin").exists()
    assert not (tmp_path / "source.bin.partial").exists()


def test_teaseq_recipe_is_explicit_but_excluded_from_all():
    recipe = generator.RECIPES[generator.TEASEQ_DATASET]

    assert isinstance(recipe, generator.ExternalDatasetRecipe)
    assert generator.TEASEQ_DATASET not in generator._datasets_for_all()
    assert {source.sha256 for source in recipe.sources} == {
        "501a1716a370a3958a71a1aec8e8620f1496d115329d6943ed2bfa450eefac9f",
        "012e6a61de2a79bd96302353536d0a8e44f527007df8f32a4c44417e2bfc1197",
    }
    parameters = {stage: dict(values) for stage, values in recipe.analysis_parameters}
    assert parameters["integration"]["assayOrder"] == ("RNA", "ATAC", "ADT")
    assert parameters["neighborhood"]["selfFreeNeighbors"] == 20


def test_all_cannot_silently_ignore_named_external_dataset():
    with pytest.raises(SystemExit):
        generator.main(["--all", generator.TEASEQ_DATASET])


def test_teaseq_annotations_map_original_barcodes_through_well_suffix(tmp_path):
    n_cells = generator.TEASEQ_TOTAL_CELLS
    n_publication_cells = generator.TEASEQ_MATCHED_PUBLICATION_CELLS
    original = np.asarray([f"BC{index:05d}-1" for index in range(n_cells)])
    publication = pd.DataFrame(
        {
            "barcode": [f"BC{index:05d}-3" for index in range(n_publication_cells)],
            "seurat_pbmc_cell_type": ["T cell"] * n_publication_cells,
            "seurat_pbmc_type_color": ["#123456"] * n_publication_cells,
            "predicted.celltype.l2": ["CD4 T"] * n_publication_cells,
            "predicted.celltype.l2.score": [0.9] * n_publication_cells,
        }
    )
    archive_path = tmp_path / generator.TEASEQ_ANNOTATIONS_NAME
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "Figure4_SourceData2_TypeLabelsUMAP.csv",
            publication.to_csv(index=False),
        )

    class Cells:
        def __init__(self) -> None:
            self.columns = {
                "original_barcodes": original,
                "well_id": np.full(n_cells, "X066-MP0C1W3"),
            }

        def fetch_all(self, name: str) -> np.ndarray:
            return self.columns[name]

        def insert(
            self,
            name: str,
            values: np.ndarray,
            *,
            overwrite: bool,
        ) -> None:
            assert overwrite is True
            self.columns[name] = np.asarray(values)

        def update_key(self, values: np.ndarray, key: str) -> None:
            self.columns[key] = np.asarray(values) & self.columns[key]

    class Store:
        cells = Cells()

    store = Store()
    store.cells.columns["I"] = np.ones(n_cells, dtype=bool)
    generator._add_teaseq_annotations(store, tmp_path)

    assert int(store.cells.columns["I"].sum()) == n_publication_cells
    assert store.cells.columns["publication_barcode"][0] == "BC00000-3"
    assert store.cells.columns["tea_cell_type"][0] == "T cell"
    assert store.cells.columns["tea_cell_type"][-1] == ""
    assert np.isnan(store.cells.columns["tea_prediction_score"][-1])
