from pathlib import Path

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
