---
description: Decide where Scarf fits and translate familiar Scanpy and Seurat workflows to its store-backed API.
---

(scanpy_and_seurat)=
# Scarf for Scanpy and Seurat users

If you already analyze single-cell data with [Scanpy](https://scanpy.readthedocs.io/) or [Seurat](https://satijalab.org/seurat/), use this page to decide where Scarf fits and to translate the workflow stages you already know.

The biological workflow remains familiar: quality control, feature selection, normalization, PCA, a neighbourhood graph, embeddings, clustering, and marker genes.
The main change is how the work is executed.
A Scarf `DataStore` points to a Zarr store, a chunked format that works on local disk or object storage.
Analysis methods stream data from that store and write results back as they complete.

If you want to run Scarf before comparing APIs, start with the {ref}`Quick start <quickstart>`.

## Where Scarf fits

Scarf is most useful when:

- the count matrix is too large for convenient in-memory analysis
- counts should stay on local disk or object storage while the analysis runs
- completed steps should persist so they can be inspected, reused, or resumed
- RNA, ATAC, or CITE-seq assays should live in one analysis store

You do not need to move an entire project to Scarf.
A common pattern is to run the large, graph-based part of an analysis in Scarf, then export data for a method from another single-cell ecosystem.
Scarf does not include scVI, Scanorama, RNA velocity, or replicate-aware differential expression.
Marker search does apply within-group Benjamini-Hochberg adjustment over tested features.

## The key mental-model change

Start with the object and workflow concepts that are familiar from Scanpy or Seurat:

| Scanpy or Seurat | Scarf |
|---|---|
| An `AnnData` or `SeuratObject` holds counts, metadata, and results | A `DataStore` opens a Zarr store containing counts, metadata, and results |
| Analysis changes an object in the current session, which you save explicitly | Each analysis step writes its result to the store as it completes |
| Filtering commonly subsets an object or creates a separate view | Filtering updates the boolean cell selection `I`; excluded cells remain in the store |
| Neighbour graphs and embeddings occupy named object slots | Graphs and embeddings are stored results that later methods can reuse |
| Larger data needs a backed mode or an additional on-disk backend | Store-backed execution is Scarf's default path |

A `DataStore` can contain several assays, such as `RNA` and `ADT`.
Most methods use the default assay unless you select another one.
Local paths and `s3://` or `gs://` locations use the same analysis API.

## Scanpy workflow map

Scanpy commonly composes the stages as separate `sc.pp`, `sc.tl`, and `sc.pl` calls.
Scarf provides the same level of control through individual methods, but `ds.pipeline.run()` is the shortest path through the standard RNA recipe.

The rows below map intent, not identical statistical implementations:

| Goal | Scanpy | Scarf |
|---|---|---|
| Load counts | `sc.read_*` returns an `AnnData` | A reader and `*ToZarr` writer create the store; `DataStore` opens it |
| Calculate QC metrics | `sc.pp.calculate_qc_metrics` | Opening a new `DataStore` calculates `RNA_nCounts`, `RNA_nFeatures`, `percentMito`/`percentRibo` percentages (0-100) when available, and feature cell counts |
| Filter cells | `sc.pp.filter_cells` or an `obs` mask | `ds.filter_cells` or `ds.auto_filter_cells`; cells are marked inactive rather than deleted |
| Select and normalize features | `sc.pp.normalize_total`, `sc.pp.log1p`, `sc.pp.highly_variable_genes` | `ds.mark_hvgs`, then `ds.run_normalization` |
| Run PCA and find neighbours | `sc.pp.pca`, then `sc.pp.neighbors` | `ds.run_pca`, then the approximate-nearest-neighbour (ANN), neighbour-query, and connectivity methods; {doc}`tutorials/graph_construction` shows the full chain |
| Embed the graph | `sc.tl.umap` | `ds.run_umap` |
| Cluster cells | `sc.tl.leiden` | `ds.run_leiden_clustering` or `ds.run_paris_clustering` |
| Find marker genes | `sc.tl.rank_genes_groups` | `ds.run_marker_search`, then `ds.get_markers`; Scarf reports AUC, two-sided Mann-Whitney p-values, and within-group Benjamini-Hochberg adjustment over tested features. This is not replicate-aware differential expression |
| Plot results | `sc.pl.*` | `ds.plots.embedding`, `ds.plots.dotplot`, and other `ds.plots` methods |
| Export an assay | `adata.write_h5ad` | `ds.to_anndata()` or `scarf.to_h5ad` |

## Seurat workflow map

This map translates common Seurat concepts.
The methods are approximate counterparts rather than one-to-one implementations.

| Goal | Seurat | Scarf |
|---|---|---|
| Hold the analysis | `SeuratObject` | `DataStore` |
| Work with modalities | Assays such as `RNA` and `ADT` | Assays in the same Zarr store |
| Select and normalize features | `NormalizeData`, `FindVariableFeatures` | `ds.mark_hvgs`, then `ds.run_normalization` |
| Scale, reduce, and find neighbours | `ScaleData`, `RunPCA`, `FindNeighbors` | `ds.run_pca` standardizes features by default, followed by Scarf's neighbour-graph methods |
| Embed and cluster | `RunUMAP`, `FindClusters` | `ds.run_umap`, `ds.run_leiden_clustering` |
| Correct batches with Harmony | Harmony integration after PCA | `ds.run_harmony` after PCA, followed by the graph-building methods |
| Integrate modalities with WNN | WNN | `ds.integrate_assays(..., method="wnn")` |
| Select representative cells | Sketching | TopACeDo through {doc}`tutorials/downsampling` |

## Move data between tools

### Scanpy and H5AD

Scarf reads and writes H5AD, so Scarf and Scanpy can be used at different stages of one project.

Import an H5AD file into a Scarf store:

```python
import scarf

inspection = scarf.inspect_h5ad("data.h5ad")
reader = scarf.H5adReader.from_inspect(inspection)
scarf.H5adToZarr(reader, zarr_loc="data.zarr").dump()
ds = scarf.DataStore("data.zarr")
```

Export to an in-memory `AnnData` object or directly to H5AD:

```python
adata = ds.to_anndata()
scarf.to_h5ad(ds.RNA, "analysis.h5ad")
```

`ds.to_anndata()` defaults to active cells (`I`) and all features.
Pass `feature_indexes` or `feature_names` to subset features.
For a large store, export only what the next method needs.
`scarf.to_h5ad` writes the full assay to disk (all cells, including those with `I=False`, and all features) without first creating an in-memory `AnnData`.

Counts and metadata transfer, but Scarf's neighbourhood graphs, provenance records, and multimodal relationships do not map directly to AnnData.
The exported H5AD may therefore need a new neighbour graph in Scanpy.
See {doc}`tutorials/import_and_export` for format details and export options.

### Seurat

Scarf can import a serialized Seurat object from an `.rds` file.

Inspect the RDS file, select importable assays and reductions, then write a Zarr store:

```python
import scarf

inspection = scarf.inspect_seurat("pbmc.rds")
with scarf.SeuratReader(
    "pbmc.rds",
    assays=["RNA"],
    reductions=["pca"],
) as reader:
    scarf.SeuratToZarr(reader, zarr_loc="pbmc.zarr").dump()
ds = scarf.DataStore("pbmc.zarr")
```

The importer brings across supported count layers, cell metadata, `active.ident`, and selected reductions.
Neighbour graphs, images, commands, and most tool slots stay behind.
Prefer original 10x HDF5 or Matrix Market counts when they are available and you only need raw matrices.

To return to Seurat, write H5AD or Matrix Market from Scarf and convert or import it with the tools used by your R workflow.
See {doc}`tutorials/import_and_export` for the full Seurat import contract and the other format paths.

## Choose a workflow

- Run a first analysis: {ref}`Quick start <quickstart>`
- Tune each stage of an RNA workflow: {doc}`tutorials/scrna_seq`
- Control graph construction step by step: {doc}`tutorials/graph_construction`
- Understand stored results and reuse: {doc}`concepts/provenance`
