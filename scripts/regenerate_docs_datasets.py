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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "docs/source/developers/dataset_manifests"
STORE_NAME = "data.zarr"
ARCHIVE_NAME = f"{STORE_NAME}.tar.gz"
LEGACY_SUFFIX = "_legacy_master"
# Convert and analysis share one process. Two workers pin the Zarr
# executor to one codec thread and async concurrency 1, which every
# later stage can reuse.
ZARR_NTHREADS = 2
DOCS_RUN_LABEL = "docs_default"

PBMC_FILTERS = {
    "method": "manual",
    "attrs": ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    "highs": [15000, 4000, 15],
    "lows": [1000, 500, 0],
}

KANG_CONTROL_DATASET = "kang_15K_pbmc_rnaseq"
KANG_STIMULATED_DATASET = "kang_14K_ifnb-pbmc_rnaseq"
KANG_INTEGRATED_DATASET = "kang_29K_ctrl-ifnb_pbmc_rnaseq"
TEASEQ_DATASET = "swanson_7K_pbmc_teaseq"
TEASEQ_RDS_NAME = "GSM5123951_PBMC_permcells_TEA-seq_SeuratObject.rds"
TEASEQ_ANNOTATIONS_NAME = "elife-63632-fig4-data2-v1.zip"
TEASEQ_TOTAL_CELLS = 7_069
TEASEQ_MATCHED_PUBLICATION_CELLS = 6_194
EXTERNAL_DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True, slots=True)
class ExternalSource:
    filename: str
    url: str
    sha256: str


def _download_external_source(
    source: ExternalSource,
    destination: Path,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.filename
    if target.exists() and _file_digest(target) == source.sha256:
        return target

    partial = destination / f"{source.filename}.partial"
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with (
            urlopen(
                source.url,
                timeout=EXTERNAL_DOWNLOAD_TIMEOUT_SECONDS,
            ) as response,
            partial.open("wb") as handle,
        ):
            for block in iter(lambda: response.read(1 << 20), b""):
                digest.update(block)
                handle.write(block)
        actual = digest.hexdigest()
        if actual != source.sha256:
            raise ValueError(
                f"Checksum mismatch for {source.filename}: "
                f"expected {source.sha256}, found {actual}"
            )
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def _convert_cellranger_h5(source: Path, store: Path) -> None:
    import scarf

    reader = scarf.CrH5Reader(str(source / "data.h5"))
    scarf.CrToZarr(reader, zarr_loc=str(store), nthreads=ZARR_NTHREADS).dump()


def _convert_h5ad(source: Path, store: Path) -> None:
    import scarf

    inspection = scarf.inspect_h5ad(str(source / "data.h5ad"))
    reader = scarf.H5adReader.from_inspect(inspection)
    scarf.H5adToZarr(reader, zarr_loc=str(store), nthreads=ZARR_NTHREADS).dump()


def _labelled_cluster_mask(values: Any) -> np.ndarray:
    labels = np.asarray(values).astype(str)
    labels = np.char.strip(labels)
    return (labels != "") & (np.char.lower(labels) != "nan")


def _set_prepared_cell_selection(store: Any, values: Any) -> None:
    """Make one exact analysis selection the teaching store's literal ``I``."""
    selected = np.asarray(values, dtype=bool)
    if selected.shape != (store.cells.N,):
        raise ValueError("Prepared cell selection must align with the full cell axis")
    store.cells.reset_key("I")
    store.cells.update_key(selected, "I")


def _materialize_run_cell_columns(
    store: Any,
    run: Any,
    columns: Mapping[str, str],
    *,
    set_selection: bool,
) -> None:
    """Copy explicitly named frozen run fields into a prepared teaching store."""
    for target, source in columns.items():
        if source not in run.cells.columns:
            raise KeyError(f"Pipeline run has no prepared field {source!r}")
        values = np.asarray(run.cells.fetch_all(source))
        if values.shape != (store.cells.N,):
            raise ValueError(f"Pipeline field {source!r} is not a full-axis vector")
        fill_value = 0 if np.issubdtype(values.dtype, np.integer) else np.nan
        store.cells.insert(
            target,
            values,
            fill_value=fill_value,
            overwrite=True,
        )
    if set_selection:
        _set_prepared_cell_selection(store, run.cells.fetch_all("I"))


def _prepared_full_axis_values(
    values: np.ndarray,
    indices: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    compact = np.asarray(values)
    if compact.ndim != 1 or compact.shape != (len(indices),):
        raise ValueError("Prepared artifact values do not align with their selection")
    if compact.dtype.kind in {"f", "c"}:
        output = np.full(n_cells, np.nan, dtype=compact.dtype)
    elif compact.dtype.kind == "u":
        compact = compact.astype(np.int64)
        output = np.full(n_cells, -1, dtype=np.int64)
    elif compact.dtype.kind == "i":
        output = np.full(n_cells, -1, dtype=compact.dtype)
    elif compact.dtype.kind == "b":
        output = np.zeros(n_cells, dtype=bool)
    else:
        output = np.full(n_cells, "", dtype=compact.dtype)
    output[indices] = compact
    return output


def _materialize_artifact_cell_columns(
    store: Any,
    artifact: Any,
    columns: Mapping[str, tuple[str, int | None]],
) -> None:
    """Project exact compact artifact arrays into literal teaching columns."""
    from scarf.storage import ArtifactRef
    from scarf.storage.selections import read_stored_selection_indices

    status = store.inspect_artifact(artifact)
    if not status.complete:
        raise ValueError("Prepared artifact must be complete")
    raw_selection = (status.inputs or {}).get("cell_selection")
    if not isinstance(raw_selection, Mapping):
        raise ValueError("Prepared artifact has no exact cell-selection input")
    selection = ArtifactRef.from_dict(raw_selection)
    indices = read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    group = store.load_artifact(artifact)
    arrays: dict[str, np.ndarray] = {}
    for target, (source, value_index) in columns.items():
        if source not in group:
            raise KeyError(f"Prepared artifact has no array {source!r}")
        if source not in arrays:
            arrays[source] = np.asarray(group[source][:])
        values = arrays[source]
        if value_index is None:
            compact = values
        else:
            if values.ndim != 2 or not 0 <= value_index < values.shape[1]:
                raise ValueError(
                    f"Prepared artifact array {source!r} has no component {value_index}"
                )
            compact = values[:, value_index]
        full_values = _prepared_full_axis_values(
            compact,
            indices,
            store.cells.N,
        )
        fill_value = 0 if np.issubdtype(full_values.dtype, np.integer) else np.nan
        store.cells.insert(
            target,
            full_values,
            fill_value=fill_value,
            overwrite=True,
        )


def _drop_retired_assay_state(root: Any) -> tuple[str, ...]:
    """Delete leftover `{assay}/state` groups so the archived store can reopen."""
    removed: list[str] = []
    for name in list(root.group_keys()):
        child = root[name]
        if not hasattr(child, "group_keys"):
            continue
        if child.attrs.get("is_assay") and "state" in child:
            del child["state"]
            removed.append(f"{name}/state")
    return tuple(removed)


def _verify_store_opens(store: Path, *, default_assay: str) -> None:
    from scarf import DataStore

    DataStore(
        str(store),
        default_assay=default_assay,
        nthreads=ZARR_NTHREADS,
        zarr_mode="r",
    )


def _openable_rna_store(source_path: Path, work: Path) -> Path:
    """Return a store DataStore can open, repacking a legacy snapshot if needed."""
    import zarr

    from scarf.storage.counts_t_contract import inspect_counts_t
    from scarf.tools.repack_zarr import repack_store

    store = source_path / STORE_NAME
    root = zarr.open_group(str(store), mode="r")
    if inspect_counts_t(root, "RNA").status == "ready":
        return store

    work.mkdir(parents=True, exist_ok=True)
    repacked = work / f"{source_path.name}_repacked.zarr"
    if repacked.exists():
        shutil.rmtree(repacked)
    print(f"Repacking {store} so the current RNA layout can open")
    repack_store(str(store), str(repacked), nthreads=ZARR_NTHREADS)
    return repacked


def _derive_labelled_kang_store(
    source_paths: dict[str, Path],
    store: Path,
) -> None:
    import scarf

    if len(source_paths) != 1:
        raise ValueError("A labelled Kang store requires exactly one source")
    source_dataset, source_path = next(iter(source_paths.items()))
    source = scarf.DataStore(
        str(_openable_rna_store(source_path, store.parent / "_repack")),
        nthreads=ZARR_NTHREADS,
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
        nthreads=ZARR_NTHREADS,
    ).dump()
    kept = int(keep.sum())
    removed = int((~keep).sum())
    print(
        f"Derived {store} from {source_dataset}: physically removed "
        f"{removed} unlabelled cells and retained {kept}"
    )


def _analyze_pbmc(store: Any) -> None:
    run = store.pipeline.run(
        label=DOCS_RUN_LABEL,
        filtering=PBMC_FILTERS,
        hvg_count=500,
        pca_dims=15,
        neighbors_k=11,
        leiden={"partitions": [0.5]},
        paris=False,
        snapshot_columns=("RNA_nCounts", "RNA_nFeatures"),
    )
    paris = store.run_paris_clustering(run["connectivity_map"], n_clusters=15)
    _materialize_run_cell_columns(
        store,
        run,
        {
            "RNA_UMAP1": "umap_1",
            "RNA_UMAP2": "umap_2",
            "RNA_leiden_cluster": "leiden_0.5",
            "RNA_clusters": "clusters",
            "RNA_S_score": "s_score",
            "RNA_G2M_score": "g2m_score",
            "RNA_cell_cycle_phase": "cell_cycle_phase",
            "RNA_doublet_score": "doublet_score",
        },
        set_selection=True,
    )
    _materialize_artifact_cell_columns(
        store,
        paris,
        {"RNA_paris_cluster": ("labels", None)},
    )


def _analyze_pancreas(store: Any) -> None:
    run = store.pipeline.run(
        label=DOCS_RUN_LABEL,
        filtering=False,
        cell_cycle=False,
        hvg_count=2000,
        pca_dims=15,
        neighbors_k=11,
        leiden={"partitions": [0.5]},
        paris=False,
        doublets=False,
        markers=True,
    )
    _materialize_run_cell_columns(
        store,
        run,
        {
            "RNA_UMAP1": "umap_1",
            "RNA_UMAP2": "umap_2",
            "RNA_leiden_cluster": "leiden_0.5",
            "RNA_clusters": "clusters",
        },
        set_selection=True,
    )


def _analyze_kang(store: Any) -> None:
    run = store.pipeline.run(
        label=DOCS_RUN_LABEL,
        filtering={
            "method": "manual",
            "attrs": ["RNA_nCounts", "RNA_nFeatures"],
            "highs": [15000, 4000],
            "lows": [500, 200],
        },
        cell_cycle=False,
        hvg_count=2000,
        pca_dims=25,
        neighbors_k=21,
        leiden={"partitions": [1.0]},
        paris=False,
        doublets=False,
        markers=True,
    )
    _materialize_run_cell_columns(
        store,
        run,
        {
            "RNA_UMAP1": "umap_1",
            "RNA_UMAP2": "umap_2",
            "RNA_leiden_cluster": "leiden_1.0",
            "RNA_clusters": "clusters",
        },
        set_selection=True,
    )


def _merge_kang(source_paths: dict[str, Path], store: Path) -> None:
    import scarf

    sources = [
        scarf.DataStore(
            str(source_paths[dataset] / STORE_NAME),
            nthreads=ZARR_NTHREADS,
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
        nthreads=ZARR_NTHREADS,
    ).dump()


def _analyze_kang_integration(store: Any) -> None:
    run = store.pipeline.run(
        label=DOCS_RUN_LABEL,
        filtering=False,
        cell_cycle=False,
        hvg_count=2000,
        pca_dims=25,
        neighbors_k=21,
        leiden={"partitions": [1.0]},
        paris=False,
        doublets=False,
        markers=False,
        snapshot_columns=("sample_id", "orig_cluster_labels"),
    )
    _materialize_run_cell_columns(
        store,
        run,
        {
            "RNA_UMAP1": "umap_1",
            "RNA_UMAP2": "umap_2",
            "RNA_integration_clusters": "leiden_1.0",
            "RNA_clusters": "clusters",
        },
        set_selection=True,
    )


def _analyze_citeseq(store: Any) -> None:
    import numpy as np

    rna_run = store.pipeline.run(
        label=DOCS_RUN_LABEL,
        cell_cycle=False,
        hvg_count=1000,
        pca_dims=15,
        neighbors_k=21,
        leiden={"partitions": [1.0]},
        paris=False,
        doublets=False,
        markers=False,
    )
    cell_selection = rna_run["analysis_cell_selection"]

    names = np.asarray(store.ADT.feats.fetch_all("names")).astype(str)
    is_control = np.char.find(np.char.lower(names), "control") >= 0
    adt_features = store.set_feature_selection(
        from_assay="ADT",
        mask=~is_control,
    )

    normalized = store.run_normalization(
        cell_selection,
        adt_features,
    )
    n_features = int(store.load_artifact(normalized)["data"].shape[1])
    reduction = store.run_custom_reduction(
        np.eye(n_features, dtype=np.float64),
        normalized,
    )
    initialization = store.build_embedding_initialization(reduction)
    ann = store.build_ann_index(reduction)
    neighbors = store.query_neighbors(ann, k=21)
    graph = store.build_connectivity_map(neighbors)
    adt_umap = store.run_umap(graph, initialization)
    adt_clusters = store.run_leiden_clustering(graph, resolution=1.0)
    _materialize_artifact_cell_columns(
        store,
        adt_umap,
        {
            "ADT_UMAP1": ("values", 0),
            "ADT_UMAP2": ("values", 1),
        },
    )
    _materialize_artifact_cell_columns(
        store,
        adt_clusters,
        {"ADT_leiden_cluster": ("values", None)},
    )

    for method, sources in (
        ("snn", [rna_run["connectivity_map"], graph]),
        ("wnn", [rna_run["neighbors"], neighbors]),
    ):
        integrated = store.integrate_assays(
            sources,
            method=method,
        )
        embedding = store.run_umap(
            integrated,
            rna_run["embedding_initialization"],
        )
        clusters = store.run_leiden_clustering(integrated, resolution=1.75)
        prefix = "RNA+ADT" if method == "snn" else "RNA+ADT_wnn"
        _materialize_artifact_cell_columns(
            store,
            embedding,
            {
                f"{prefix}_UMAP1": ("values", 0),
                f"{prefix}_UMAP2": ("values", 1),
            },
        )
        _materialize_artifact_cell_columns(
            store,
            clusters,
            {f"{prefix}_leiden_cluster": ("values", None)},
        )

    _materialize_run_cell_columns(
        store,
        rna_run,
        {
            "RNA_UMAP1": "umap_1",
            "RNA_UMAP2": "umap_2",
            "RNA_leiden_cluster": "leiden_1.0",
            "RNA_clusters": "clusters",
        },
        set_selection=True,
    )


def _analyze_atac(store: Any) -> None:
    cell_selection = store.auto_filter_cells()
    prevalent_peaks = store.select_prevalent_peaks(
        cell_selection,
        top_n=25000,
    )
    normalized = store.run_normalization(cell_selection, prevalent_peaks)
    reduction = store.run_lsi(normalized, dims=50, skip_first=True)
    initialization = store.build_embedding_initialization(reduction)
    ann = store.build_ann_index(reduction)
    neighbors = store.query_neighbors(ann, k=21)
    graph = store.build_connectivity_map(neighbors)
    embedding = store.run_umap(
        graph,
        initialization,
        n_epochs=500,
        min_dist=0.1,
        spread=1,
        parallel=True,
    )
    clusters = store.run_leiden_clustering(graph, resolution=0.6)
    _materialize_artifact_cell_columns(
        store,
        embedding,
        {
            "ATAC_UMAP1": ("values", 0),
            "ATAC_UMAP2": ("values", 1),
        },
    )
    _materialize_artifact_cell_columns(
        store,
        clusters,
        {"ATAC_leiden_cluster": ("values", None)},
    )
    from scarf.storage.selections import read_stored_selection_mask

    _set_prepared_cell_selection(
        store,
        read_stored_selection_mask(
            store.zw,
            cell_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ),
    )


def _add_teaseq_annotations(store: Any, source: Path) -> None:
    import pandas as pd

    with ZipFile(source / TEASEQ_ANNOTATIONS_NAME) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if csv_names != ["Figure4_SourceData2_TypeLabelsUMAP.csv"]:
            raise RuntimeError("Unexpected TEA-seq annotation archive contents")
        with archive.open(csv_names[0]) as handle:
            annotations = pd.read_csv(handle)
    if not annotations["barcode"].is_unique:
        raise RuntimeError("TEA-seq publication barcodes are not unique")

    original_barcodes = pd.Series(
        store.cells.fetch_all("original_barcodes"),
        dtype="string",
    ).str.replace(r"-\d+$", "", regex=True)
    well_suffixes = pd.Series(
        store.cells.fetch_all("well_id"),
        dtype="string",
    ).str.extract(r"W(\d+)$", expand=False)
    if well_suffixes.isna().any():
        raise RuntimeError("TEA-seq well_id values do not end in a well number")
    publication_barcodes = original_barcodes + "-" + well_suffixes
    positions = pd.Index(annotations["barcode"].astype(str)).get_indexer(
        publication_barcodes,
    )
    publication_cells = positions >= 0
    matched_cells = int(publication_cells.sum())
    if (
        len(publication_cells) != TEASEQ_TOTAL_CELLS
        or matched_cells != TEASEQ_MATCHED_PUBLICATION_CELLS
    ):
        raise RuntimeError(
            "TEA-seq publication mapping must retain "
            f"{TEASEQ_TOTAL_CELLS:,} cells and select "
            f"{TEASEQ_MATCHED_PUBLICATION_CELLS:,}; found "
            f"{len(publication_cells):,} and {matched_cells:,}"
        )

    def matched_text(column: str) -> np.ndarray:
        values = np.full(len(positions), "", dtype=object)
        values[publication_cells] = (
            annotations[column].astype(str).to_numpy()[positions[publication_cells]]
        )
        return values.astype(str)

    def matched_float(column: str) -> np.ndarray:
        values = np.full(len(positions), np.nan, dtype=np.float64)
        values[publication_cells] = annotations[column].to_numpy(dtype=np.float64)[
            positions[publication_cells]
        ]
        return values

    store.cells.insert(
        "publication_barcode",
        publication_barcodes.to_numpy(dtype=str),
        overwrite=True,
    )
    store.cells.insert(
        "tea_cell_type",
        matched_text("seurat_pbmc_cell_type"),
        overwrite=True,
    )
    store.cells.insert(
        "tea_cell_type_color",
        matched_text("seurat_pbmc_type_color"),
        overwrite=True,
    )
    store.cells.insert(
        "tea_predicted_cell_type",
        matched_text("predicted.celltype.l2"),
        overwrite=True,
    )
    store.cells.insert(
        "tea_prediction_score",
        matched_float("predicted.celltype.l2.score"),
        overwrite=True,
    )
    store.cells.update_key(publication_cells, "I")
    active_cells = int(np.asarray(store.cells.fetch_all("I"), dtype=bool).sum())
    if active_cells != TEASEQ_MATCHED_PUBLICATION_CELLS:
        raise RuntimeError(
            "TEA-seq imported cell filter removed publication matches: "
            f"expected {TEASEQ_MATCHED_PUBLICATION_CELLS:,}, found {active_cells:,}"
        )


def _convert_teaseq(source: Path, store: Path) -> None:
    from scarf import SeuratReader
    from scarf.writers.seurat import SeuratToZarr

    expected_dimensions = {
        "RNA": (36_601, 7_069),
        "ATAC": (240_122, 7_069),
        "ADT": (46, 7_069),
    }
    with SeuratReader(
        source / TEASEQ_RDS_NAME,
        temp_dir=source,
        assays=list(expected_dimensions),
        reductions=[],
    ) as reader:
        dimensions = {
            assay: reader.get_assay(assay).dimensions for assay in expected_dimensions
        }
        if dimensions != expected_dimensions:
            raise RuntimeError(f"Unexpected TEA-seq assay dimensions: {dimensions!r}")
        SeuratToZarr(
            reader,
            str(store),
            mem_budget=8 * (1 << 30),
            nthreads=ZARR_NTHREADS,
        ).dump()

    import scarf

    imported = scarf.DataStore(
        str(store),
        default_assay="RNA",
        nthreads=1,
        mem_budget=4 * (1 << 30),
    )
    _add_teaseq_annotations(imported, source)


def _build_teaseq_graph(
    store: Any,
    *,
    reduction: Any,
    prefix: str,
) -> dict[str, Any]:
    initialization = store.build_embedding_initialization(
        reduction,
        n_centroids=100,
        rand_state=4466,
    )
    ann = store.build_ann_index(
        reduction,
        ann_parallel=False,
        rand_state=4466,
    )
    neighbors = store.query_neighbors(
        ann,
        coordinates=reduction,
        k=20,
    )
    graph = store.build_connectivity_map(neighbors)
    embedding = store.run_umap(
        graph,
        initialization,
        n_epochs=250,
        spread=1,
        min_dist=0.1,
        random_seed=4444,
        parallel=False,
    )
    clusters = store.run_leiden_clustering(
        graph,
        resolution=1.0,
        random_seed=4444,
    )
    _materialize_artifact_cell_columns(
        store,
        embedding,
        {
            f"{prefix}_UMAP1": ("values", 0),
            f"{prefix}_UMAP2": ("values", 1),
        },
    )
    _materialize_artifact_cell_columns(
        store,
        clusters,
        {f"{prefix}_leiden_cluster": ("values", None)},
    )
    return {
        "initialization": initialization,
        "neighbors": neighbors,
        "graph": graph,
    }


def _analyze_teaseq(store: Any) -> None:
    cell_selection = store.snapshot_cell_selection("I")
    hvgs = store.select_hvgs(
        cell_selection,
        from_assay="RNA",
        min_cells=20,
        top_n=2_000,
        show_plot=False,
    )
    rna_normalized = store.run_normalization(
        cell_selection,
        hvgs,
    )
    rna_reduction = store.run_pca(
        rna_normalized,
        dims=30,
        feat_scaling=True,
    )
    rna_graph = _build_teaseq_graph(
        store,
        reduction=rna_reduction,
        prefix="RNA",
    )

    prevalent_peaks = store.select_prevalent_peaks(
        cell_selection,
        from_assay="ATAC",
        top_n=25_000,
    )
    atac_normalized = store.run_normalization(
        cell_selection,
        prevalent_peaks,
    )
    atac_reduction = store.run_lsi(
        atac_normalized,
        dims=30,
        skip_first=True,
        rand_state=4466,
        solver="streaming",
    )
    atac_graph = _build_teaseq_graph(
        store,
        reduction=atac_reduction,
        prefix="ATAC",
    )

    adt_names = np.asarray(store.ADT.feats.fetch_all("names")).astype(str)
    adt_controls = np.char.find(np.char.lower(adt_names), "control") >= 0
    adt_features = store.set_feature_selection(
        from_assay="ADT",
        mask=~adt_controls,
    )
    adt_normalized = store.run_normalization(
        cell_selection,
        adt_features,
    )
    adt_reduction = store.run_pca(
        adt_normalized,
        dims=15,
        feat_scaling=True,
    )
    adt_graph = _build_teaseq_graph(
        store,
        reduction=adt_reduction,
        prefix="ADT",
    )

    for method, sources in (
        (
            "snn",
            [rna_graph["graph"], atac_graph["graph"], adt_graph["graph"]],
        ),
        (
            "wnn",
            [
                rna_graph["neighbors"],
                atac_graph["neighbors"],
                adt_graph["neighbors"],
            ],
        ),
    ):
        integrated = store.integrate_assays(
            sources,
            method=method,
        )
        embedding = store.run_umap(
            integrated,
            rna_graph["initialization"],
            n_epochs=250,
            spread=1,
            min_dist=0.1,
            random_seed=4444,
            parallel=False,
        )
        clusters = store.run_leiden_clustering(
            integrated,
            resolution=1.0,
            random_seed=4444,
        )
        prefix = "RNA+ATAC+ADT" if method == "snn" else "RNA+ATAC+ADT_wnn"
        _materialize_artifact_cell_columns(
            store,
            embedding,
            {
                f"{prefix}_UMAP1": ("values", 0),
                f"{prefix}_UMAP2": ("values", 1),
            },
        )
        _materialize_artifact_cell_columns(
            store,
            clusters,
            {f"{prefix}_leiden_cluster": ("values", None)},
        )


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
class ExternalDatasetRecipe:
    """A publishable store built from checksum-pinned public source files."""

    sources: tuple[ExternalSource, ...]
    convert: Callable[[Path, Path], None]
    analyze: Callable[[Any], None]
    summary: str
    attribution: tuple[str, ...]
    cell_selection: str
    import_memory_bytes: int
    analysis_memory_bytes: int
    analysis_parameters: tuple[
        tuple[
            str,
            tuple[tuple[str, str | int | float | bool | tuple[str, ...]], ...],
        ],
        ...,
    ]
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


type DatasetRecipe = RawDatasetRecipe | ExternalDatasetRecipe | DerivedDatasetRecipe


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
            "pipeline-selected clustering"
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
    TEASEQ_DATASET: ExternalDatasetRecipe(
        sources=(
            ExternalSource(
                filename=TEASEQ_RDS_NAME,
                url=(
                    "https://zenodo.org/api/records/6360802/files/"
                    f"{TEASEQ_RDS_NAME}/content"
                ),
                sha256=(
                    "501a1716a370a3958a71a1aec8e8620f1496d115329d6943ed2bfa450eefac9f"
                ),
            ),
            ExternalSource(
                filename=TEASEQ_ANNOTATIONS_NAME,
                url=(
                    "https://cdn.elifesciences.org/articles/63632/"
                    "elife-63632-fig4-data2-v1.zip"
                ),
                sha256=(
                    "012e6a61de2a79bd96302353536d0a8e44f527007df8f32a4c44417e2bfc1197"
                ),
            ),
        ),
        convert=_convert_teaseq,
        analyze=_analyze_teaseq,
        summary=(
            "RNA PCA, ATAC TF-IDF and LSI, ADT CLR and PCA, modality UMAPs, "
            "three-way SNN and WNN, integrated UMAPs, and Leiden labels"
        ),
        attribution=(
            "Swanson et al. 2021, eLife 10:e63632",
            "GEO accession GSM5123951",
            "eLife Figure 4 source data 2",
        ),
        cell_selection=(
            "Retain all 7,069 imported cells and activate the 6,194 exact "
            "matches to the 6,333 Figure 4 labels from well W3. The pinned "
            "Zenodo RDS omits 139 publication-labelled barcodes."
        ),
        import_memory_bytes=8 * (1 << 30),
        analysis_memory_bytes=8 * (1 << 30),
        analysis_parameters=(
            (
                "rna",
                (
                    ("normalization", "librarySizeLog1p"),
                    ("sizeFactor", 1_000),
                    ("hvgMinCells", 20),
                    ("hvgTopN", 2_000),
                    ("pcaDimensions", 30),
                    ("featureScaling", True),
                ),
            ),
            (
                "atac",
                (
                    ("normalization", "tfIdf"),
                    ("prevalentPeaks", 25_000),
                    ("lsiDimensions", 30),
                    ("skipFirst", True),
                    ("lsiSolver", "streaming"),
                ),
            ),
            (
                "adt",
                (
                    ("excludedNameSubstring", "control"),
                    ("normalization", "clr"),
                    ("pcaDimensions", 15),
                    ("featureScaling", True),
                ),
            ),
            (
                "neighborhood",
                (
                    ("selfFreeNeighbors", 20),
                    ("embeddingCentroids", 100),
                    ("graphSeed", 4_466),
                    ("annParallel", False),
                ),
            ),
            (
                "integration",
                (
                    ("assayOrder", ("RNA", "ATAC", "ADT")),
                    ("methods", ("snn", "wnn")),
                    ("wnnL2Normalize", True),
                ),
            ),
            (
                "layoutAndClustering",
                (
                    ("umapEpochs", 250),
                    ("umapSpread", 1.0),
                    ("umapMinDist", 0.1),
                    ("umapSeed", 4_444),
                    ("umapParallel", False),
                    ("leidenResolution", 1.0),
                    ("leidenSeed", 4_444),
                ),
            ),
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


def _pipeline_inventory(store: Any) -> list[dict[str, object]]:
    return [
        {
            "runId": run.run_id,
            "label": run.label,
            "recipe": run.recipe,
            "status": run.status,
        }
        for run in store.pipeline.list_runs(limit=2**31 - 1)
    ]


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

    output = destination / dataset
    output.mkdir(parents=True, exist_ok=True)
    store = output / STORE_NAME
    if store.exists():
        shutil.rmtree(store)
    if isinstance(recipe, RawDatasetRecipe):
        repository = scarf.cytebase.connect(repository_name)
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
        external_sources: list[dict[str, str]] = []
    elif isinstance(recipe, ExternalDatasetRecipe):
        source_directory = work / dataset
        for source in recipe.sources:
            _download_external_source(source, source_directory)
        recipe.convert(source_directory, store)

        if recipe.drop_columns:
            _drop_columns(
                store=store,
                patterns=recipe.drop_columns,
                default_assay=recipe.default_assay,
            )

        source_files = [source.filename for source in recipe.sources]
        source_datasets = []
        carried_columns = []
        external_sources = [
            {
                "filename": source.filename,
                "url": source.url,
                "sha256": source.sha256,
            }
            for source in recipe.sources
        ]
    else:
        repository = scarf.cytebase.connect(repository_name)
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
        external_sources = []

    datastore_options = (
        {"mem_budget": recipe.analysis_memory_bytes}
        if isinstance(recipe, ExternalDatasetRecipe)
        else {}
    )
    datastore = DataStore(
        str(store),
        default_assay=recipe.default_assay,
        nthreads=ZARR_NTHREADS,
        **datastore_options,
    )
    recipe.analyze(datastore)
    retired_state = _drop_retired_assay_state(datastore.z)
    if retired_state:
        print(
            "Dropped leftover assay state before archiving: " + ", ".join(retired_state)
        )
    _verify_store_opens(store, default_assay=recipe.default_assay)

    archive = output / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(store, arcname=STORE_NAME)

    store_bytes = _directory_bytes(store)
    archive_bytes = archive.stat().st_size
    cells_total = int(datastore.cells.N)
    cells_active = int(datastore.cells.active_index("I").size)
    artifacts = _artifact_inventory(datastore)
    pipeline_runs = _pipeline_inventory(datastore)

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
        "pipelineRuns": pipeline_runs,
        "publishNotes": [
            f"Publish with: uv run python scripts/publish_docs_datasets.py {dataset}",
            f"That preserves the published archive as {dataset}_legacy_master "
            f"and swaps {ARCHIVE_NAME} in place.",
        ],
    }
    if isinstance(recipe, ExternalDatasetRecipe):
        manifest["externalSources"] = external_sources
        manifest["attribution"] = list(recipe.attribution)
        manifest["cellSelection"] = recipe.cell_selection
        manifest["memoryBudgets"] = {
            "importBytes": recipe.import_memory_bytes,
            "analysisBytes": recipe.analysis_memory_bytes,
        }
        manifest["analysisParameters"] = {
            stage: dict(parameters) for stage, parameters in recipe.analysis_parameters
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


def _datasets_for_all() -> list[str]:
    return [
        name
        for name, recipe in RECIPES.items()
        if not isinstance(recipe, ExternalDatasetRecipe)
    ]


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
        help="Rebuild every Cytebase-backed dataset, excluding external sources",
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

    if args.all and args.datasets:
        parser.error("--all cannot be combined with named datasets")
    selected = _datasets_for_all() if args.all else list(args.datasets)
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
