#!/usr/bin/env python3
"""Rebuild the analyzed Zarr stores that Cytebase publishes for the documentation.

Source stores are built from raw counts. Derived stores are rebuilt from their
declared published inputs. Every result uses the current Zarr layout and carries
artifact provenance. This script only writes to `build/cytebase`;
`scripts/publish_docs_datasets.py` uploads the result.

Example:
    uv run python scripts/regenerate_docs_datasets.py tenx_5K_pbmc_rnaseq
    uv run python scripts/regenerate_docs_datasets.py --all
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "docs/source/developers/dataset_manifests"
STORE_NAME = "data.zarr"
ARCHIVE_NAME = f"{STORE_NAME}.tar.gz"
LEGACY_SUFFIX = "_legacy_master"

PBMC_FILTERS = {
    "method": "manual",
    "attrs": ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    "highs": [15000, 4000, 15],
    "lows": [1000, 500, 0],
}

KANG_CONTROL_DATASET = "kang_15K_pbmc_rnaseq"
KANG_STIMULATED_DATASET = "kang_14K_ifnb-pbmc_rnaseq"
KANG_INTEGRATED_DATASET = "kang_29K_ctrl-ifnb_pbmc_rnaseq"


def _convert_cellranger_h5(source: Path, store: Path) -> None:
    import scarf

    reader = scarf.CrH5Reader(str(source / "data.h5"))
    scarf.CrToZarr(reader, zarr_loc=str(store)).dump()


def _convert_h5ad(source: Path, store: Path) -> None:
    import scarf

    inspection = scarf.inspect_h5ad(str(source / "data.h5ad"))
    reader = scarf.H5adReader.from_inspect(inspection)
    scarf.H5adToZarr(reader, zarr_loc=str(store)).dump()


def _labelled_cluster_mask(values: Any) -> np.ndarray:
    labels = np.asarray(values).astype(str)
    labels = np.char.strip(labels)
    return (labels != "") & (np.char.lower(labels) != "nan")


def _derive_labelled_kang_store(
    source_paths: dict[str, Path],
    store: Path,
) -> None:
    import scarf

    if len(source_paths) != 1:
        raise ValueError("A labelled Kang store requires exactly one source")
    source_dataset, source_path = next(iter(source_paths.items()))
    source = scarf.DataStore(
        str(source_path / STORE_NAME),
        nthreads=4,
        zarr_mode="r",
    )
    keep = _labelled_cluster_mask(source.cells.fetch_all("cluster_labels"))
    if not keep.any():
        raise RuntimeError(f"No labelled cells remain in {source_dataset}")

    scarf.SubsetZarr(
        zarr_loc=str(store),
        assays=[source.RNA],
        cell_idx=np.flatnonzero(keep),
        reset_cell_filter=True,
        overwrite_existing_file=True,
        nthreads=4,
    ).dump()
    kept = int(keep.sum())
    removed = int((~keep).sum())
    print(
        f"Derived {store} from {source_dataset}: physically removed "
        f"{removed} unlabelled cells and retained {kept}"
    )


def _analyze_pbmc(store: Any) -> None:
    store.pipeline.run(
        filtering=PBMC_FILTERS,
        highly_variable_features={
            "min_cells": 20,
            "top_n": 500,
            "min_mean": -3,
            "max_mean": 2,
            "max_var": 6,
        },
        pca={"dims": 15, "n_centroids": 100},
        neighbors={"k": 11},
        umap={"n_epochs": 250, "spread": 5, "min_dist": 1, "parallel": True},
        leiden={0.5: {"label": "leiden_cluster"}},
        paris=False,
        markers={},
    )
    store.run_paris_clustering(n_clusters=15)


def _analyze_pancreas(store: Any) -> None:
    store.pipeline.run(
        filtering=False,
        cell_cycle_scoring=False,
        highly_variable_features={"min_cells": 20, "top_n": 2000},
        pca={"dims": 15, "n_centroids": 100},
        neighbors={"k": 11},
        umap={"n_epochs": 250, "spread": 5, "min_dist": 1, "parallel": True},
        leiden={0.5: {"label": "leiden_cluster"}},
        paris=False,
        doublet_scoring=False,
        markers=False,
    )
    store.run_marker_search(group_key="clusters")


def _analyze_kang(store: Any) -> None:
    store.pipeline.run(
        filtering={
            "method": "manual",
            "attrs": ["RNA_nCounts", "RNA_nFeatures"],
            "highs": [15000, 4000],
            "lows": [500, 200],
        },
        cell_cycle_scoring=False,
        highly_variable_features={
            "min_cells": 10,
            "top_n": 2000,
            "min_mean": -3,
            "max_mean": 2,
            "max_var": 6,
        },
        pca={"dims": 25, "n_centroids": 100},
        neighbors={"k": 21},
        umap={"n_epochs": 250, "spread": 5, "min_dist": 1, "parallel": True},
        leiden={1.0: {"label": "leiden_cluster"}},
        paris=False,
        doublet_scoring=False,
        markers=False,
    )


def _merge_kang(source_paths: dict[str, Path], store: Path) -> None:
    import scarf

    sources = [
        scarf.DataStore(
            str(source_paths[dataset] / STORE_NAME),
            nthreads=4,
        )
        for dataset in (KANG_CONTROL_DATASET, KANG_STIMULATED_DATASET)
    ]

    scarf.DataStoreMerge(
        datasets=sources,
        zarr_path=str(store),
        names=["ctrl", "stim"],
        assays=["RNA"],
        prepend_text="orig",
        reset_cell_filter=False,
        source_column="sample_id",
        overwrite=True,
        nthreads=4,
    ).dump()


def _analyze_kang_integration(store: Any) -> None:
    store.pipeline.run(
        filtering=False,
        cell_cycle_scoring=False,
        highly_variable_features={
            "min_cells": 10,
            "top_n": 2000,
            "min_mean": -3,
            "max_mean": 2,
            "max_var": 6,
        },
        pca={"dims": 25},
        neighbors={"k": 21},
        umap={"n_epochs": 250, "spread": 5, "min_dist": 1, "parallel": True},
        leiden={1.0: {"label": "integration_clusters"}},
        paris=False,
        doublet_scoring=False,
        markers=False,
    )


def _analyze_citeseq(store: Any) -> None:
    import numpy as np

    store.auto_filter_cells(show_qc_plots=False)
    store.pipeline.run(
        filtering=False,
        cell_cycle_scoring=False,
        highly_variable_features={
            "min_cells": 20,
            "top_n": 1000,
            "min_mean": -3,
            "max_mean": 2,
            "max_var": 6,
        },
        pca={"dims": 15, "n_centroids": 100},
        neighbors={"k": 21},
        umap={"n_epochs": 250, "spread": 5, "min_dist": 1, "parallel": True},
        leiden={1.0: {"label": "leiden_cluster"}},
        paris=False,
        doublet_scoring=False,
        markers=False,
    )

    names = np.asarray(store.ADT.feats.fetch_all("names")).astype(str)
    is_control = np.char.find(np.char.lower(names), "control") >= 0
    store.ADT.feats.update_key(~is_control, "I")

    normalized = store.run_normalization(from_assay="ADT", feat_key="I")
    n_features = int(store.load_artifact(normalized)["data"].shape[1])
    reduction = store.run_custom_reduction(
        np.eye(n_features, dtype=np.float64),
        normalized,
        from_assay="ADT",
    )
    store.build_embedding_initialization(reduction, n_centroids=100)
    ann = store.build_ann_index(reduction)
    neighbors = store.query_neighbors(ann, k=21)
    graph = store.build_connectivity_map(neighbors)
    store.run_umap(graph, n_epochs=250, spread=5, min_dist=1, parallel=True)
    store.run_leiden_clustering(graph, resolution=1.0, label="leiden_cluster")

    for label, method in (("RNA+ADT", "snn"), ("RNA+ADT_wnn", "wnn")):
        integrated = store.integrate_assays(
            assays=["RNA", "ADT"],
            label=label,
            method=method,
        )
        store.run_umap(integrated, n_epochs=250, spread=5, min_dist=1, parallel=True)
        store.run_leiden_clustering(integrated, resolution=1.75, label="leiden_cluster")


def _analyze_atac(store: Any) -> None:
    store.auto_filter_cells(show_qc_plots=False)
    store.mark_prevalent_peaks(top_n=25000)
    normalized = store.run_normalization(feat_key="prevalent_peaks")
    reduction = store.run_lsi(normalized, dims=50, skip_first=True)
    store.build_embedding_initialization(reduction)
    ann = store.build_ann_index(reduction)
    neighbors = store.query_neighbors(ann, k=21)
    graph = store.build_connectivity_map(neighbors)
    store.run_umap(graph, n_epochs=500, min_dist=0.1, spread=1, parallel=True)
    store.run_leiden_clustering(graph, resolution=0.6, label="leiden_cluster")


@dataclass(frozen=True, slots=True)
class RawDatasetRecipe:
    """One publishable store: where its counts come from and how it is analyzed."""

    sources: tuple[str, ...]
    convert: Callable[[Path, Path], None]
    analyze: Callable[[Any], None]
    summary: str
    default_assay: str = "RNA"
    drop_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DerivedDatasetRecipe:
    """One publishable store derived from other published stores."""

    source_datasets: tuple[str, ...]
    derive: Callable[[dict[str, Path], Path], None]
    analyze: Callable[[Any], None]
    summary: str
    default_assay: str = "RNA"


type DatasetRecipe = RawDatasetRecipe | DerivedDatasetRecipe


RECIPES: dict[str, DatasetRecipe] = {
    "tenx_5K_pbmc_rnaseq": RawDatasetRecipe(
        sources=("data.h5",),
        convert=_convert_cellranger_h5,
        analyze=_analyze_pbmc,
        summary=(
            "Manual QC filter, 500 HVGs, PCA 15, k=11 graph, UMAP, "
            "Leiden 0.5, Paris, doublet scores, markers"
        ),
    ),
    "bastidas-ponce_4K_pancreas-d15_rnaseq": RawDatasetRecipe(
        sources=("data.h5ad",),
        convert=_convert_h5ad,
        analyze=_analyze_pancreas,
        summary=(
            "2000 HVGs, PCA 15, k=11 graph, UMAP, Leiden 0.5, markers on the "
            "published cell-type annotations"
        ),
        drop_columns=("X_pca*",),
    ),
    "tenx_8K_pbmc_citeseq": RawDatasetRecipe(
        sources=("data.h5",),
        convert=_convert_cellranger_h5,
        analyze=_analyze_citeseq,
        summary=(
            "RNA and ADT chains with UMAP and Leiden, plus SNN and WNN "
            "integrated graphs"
        ),
    ),
    KANG_CONTROL_DATASET: DerivedDatasetRecipe(
        source_datasets=(f"{KANG_CONTROL_DATASET}{LEGACY_SUFFIX}",),
        derive=_derive_labelled_kang_store,
        analyze=_analyze_kang,
        summary=(
            "Physical removal of unlabelled cells, manual QC, 2000 HVGs, "
            "PCA 25, k=21 graph, UMAP, and Leiden 1.0"
        ),
    ),
    KANG_STIMULATED_DATASET: DerivedDatasetRecipe(
        source_datasets=(f"{KANG_STIMULATED_DATASET}{LEGACY_SUFFIX}",),
        derive=_derive_labelled_kang_store,
        analyze=_analyze_kang,
        summary=(
            "Physical removal of unlabelled cells, manual QC, 2000 HVGs, "
            "PCA 25, k=21 graph, UMAP, and Leiden 1.0"
        ),
    ),
    KANG_INTEGRATED_DATASET: DerivedDatasetRecipe(
        source_datasets=(KANG_CONTROL_DATASET, KANG_STIMULATED_DATASET),
        derive=_merge_kang,
        analyze=_analyze_kang_integration,
        summary=(
            "Control and IFNB assay merge, 2000 HVGs, PCA 25, k=21 graph, "
            "UMAP, and Leiden 1.0"
        ),
    ),
    "tenx_10K_pbmc-v1_atacseq": RawDatasetRecipe(
        sources=("data.h5",),
        convert=_convert_cellranger_h5,
        analyze=_analyze_atac,
        summary=(
            "25000 prevalent peaks, TF-IDF normalization, LSI 50, k=21 graph, "
            "UMAP, Leiden 0.6"
        ),
        default_assay="ATAC",
    ),
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _artifact_inventory(store: Any) -> list[dict[str, object]]:
    refs = list(store.list_artifacts(scope="datastore", complete_only=True))
    for assay in store.assay_names:
        refs.extend(
            store.list_artifacts(
                from_assay=assay,
                complete_only=True,
            )
        )
    inventory: list[dict[str, object]] = []
    for ref in refs:
        status = store.inspect_artifact(ref)
        inventory.append(
            {
                "kind": ref.kind,
                "artifactId": ref.artifact_id,
                "assay": ref.assay,
                "scope": ref.scope,
                "operation": status.operation,
                "path": status.path,
            }
        )
    return inventory


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _drop_columns(
    *,
    store: Path,
    patterns: Sequence[str],
    default_assay: str,
) -> None:
    """Remove imported columns that would only add noise to metadata tables."""
    from fnmatch import fnmatch

    from scarf import DataStore

    target = DataStore(str(store), default_assay=default_assay, nthreads=1)
    dropped = [
        column
        for column in target.cells.columns
        if any(fnmatch(column, pattern) for pattern in patterns)
    ]
    for column in dropped:
        target.cells.drop(column)
    print(f"Dropped {len(dropped)} imported column(s) from {store}")


def _resolve_derived_sources(
    *,
    repository: Any,
    source_datasets: Sequence[str],
    local_sources: dict[str, Path],
    work: Path,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for source_dataset in source_datasets:
        if source_dataset in local_sources:
            source_path = local_sources[source_dataset]
            if not (source_path / STORE_NAME).is_dir():
                raise FileNotFoundError(
                    f"Local source {source_dataset} has no {STORE_NAME}: {source_path}"
                )
            resolved[source_dataset] = source_path
            continue
        resolved[source_dataset] = repository.download_dataset(
            source_dataset,
            destination=str(work),
            zarr=True,
        )
    return resolved


def build_store(
    *,
    dataset: str,
    recipe: DatasetRecipe,
    destination: Path,
    repository_name: str,
    local_sources: dict[str, Path] | None = None,
) -> Path:
    import scarf
    from scarf import DataStore

    started = datetime.now(UTC)
    work = destination / "_source"
    repository = scarf.cytebase.connect(repository_name)

    output = destination / dataset
    output.mkdir(parents=True, exist_ok=True)
    store = output / STORE_NAME
    if store.exists():
        shutil.rmtree(store)
    if isinstance(recipe, RawDatasetRecipe):
        for name in recipe.sources:
            repository.download(f"{dataset}/{name}", destination=str(work))
        recipe.convert(work / dataset, store)

        if recipe.drop_columns:
            _drop_columns(
                store=store,
                patterns=recipe.drop_columns,
                default_assay=recipe.default_assay,
            )

        source_files = list(recipe.sources)
        source_datasets: list[str] = []
        carried_columns: list[str] = []
    else:
        source_paths = _resolve_derived_sources(
            repository=repository,
            source_datasets=recipe.source_datasets,
            local_sources=local_sources or {},
            work=work,
        )
        recipe.derive(source_paths, store)
        source_files = []
        source_datasets = list(recipe.source_datasets)
        carried_columns = []

    datastore = DataStore(
        str(store),
        default_assay=recipe.default_assay,
        nthreads=4,
    )
    recipe.analyze(datastore)

    archive = output / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(store, arcname=STORE_NAME)

    store_bytes = _directory_bytes(store)
    archive_bytes = archive.stat().st_size
    cells_total = int(datastore.cells.N)
    cells_active = int(datastore.cells.active_index("I").size)
    artifacts = _artifact_inventory(datastore)

    manifest = {
        "dataset": dataset,
        "repository": repository_name,
        "recipe": recipe.summary,
        "generatedAt": started.isoformat(),
        "generatorCommit": _git_commit(),
        "scarfVersion": getattr(scarf, "__version__", "unknown"),
        "sourceFiles": source_files,
        "sourceDatasets": source_datasets,
        "carriedColumns": carried_columns,
        "nCellsTotal": cells_total,
        "nCellsActive": cells_active,
        "storeBytes": store_bytes,
        "archiveBytes": archive_bytes,
        "archiveSha256": _file_digest(archive),
        "cellColumns": sorted(datastore.cells.columns),
        "artifacts": artifacts,
        "publishNotes": [
            f"Publish with: uv run python scripts/publish_docs_datasets.py {dataset}",
            f"That preserves the published archive as {dataset}_legacy_master "
            f"and swaps {ARCHIVE_NAME} in place.",
        ],
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{dataset}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(f"Built {dataset} in {elapsed:.0f}s")
    print(f"  store    {store} ({store_bytes / 1e6:.1f} MB)")
    print(f"  archive  {archive} ({archive_bytes / 1e6:.1f} MB)")
    print(f"  manifest {manifest_path}")
    print(f"  cells    {cells_active} of {cells_total} active")
    print(f"  results  {len(artifacts)} complete artifact(s)")
    return store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=[*RECIPES, []],
        help="Datasets to rebuild",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rebuild every dataset in the recipe table",
    )
    parser.add_argument(
        "--repository",
        default="scarf_docs",
        help="Cytebase repository holding the declared source inputs",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPO_ROOT / "build/cytebase",
        help="Output directory for stores, archives, and downloads",
    )
    args = parser.parse_args(argv)

    selected = list(RECIPES) if args.all else list(args.datasets)
    if not selected:
        parser.error("name at least one dataset or pass --all")
    args.destination.mkdir(parents=True, exist_ok=True)
    local_sources: dict[str, Path] = {}
    for dataset in selected:
        store = build_store(
            dataset=dataset,
            recipe=RECIPES[dataset],
            destination=args.destination,
            repository_name=args.repository,
            local_sources=local_sources,
        )
        local_sources[dataset] = store.parent
    return 0


if __name__ == "__main__":
    sys.exit(main())
