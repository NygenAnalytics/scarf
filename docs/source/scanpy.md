---
description: Translate a Scanpy workflow to Scarf and move data through H5AD.
---

(scanpy-users)=
# Scarf for Scanpy users

This guide maps familiar Scanpy stages to Scarf's store-backed API. Read
{doc}`scanpy_and_seurat` first for the shared execution and artifact model, or start with the
{ref}`Quick start <quickstart>` to inspect a familiar prepared result.

## Workflow map

Scanpy commonly composes stages as separate `sc.pp`, `sc.tl`, and `sc.pl` calls. Scarf provides
the same level of control through individual methods, while `ds.pipeline.run()` is the shortest
path through the standard RNA workflow. It returns a durable `PipelineRun` with frozen views and
leaves live metadata unchanged.

The rows below map intent, not identical statistical implementations. Scarf selects highly
variable genes before normalizing on that feature set, which differs from the common Scanpy order
of normalization, log transformation, and then HVG selection.

| Goal | Scanpy | Scarf |
|---|---|---|
| Load counts | `sc.read_*` returns an `AnnData` | A reader and `*ToZarr` writer create the store; `DataStore` opens it |
| Calculate QC metrics | `sc.pp.calculate_qc_metrics` | Opening a new `DataStore` calculates count and feature metrics, plus detected RNA mitochondrial and ribosomal percentages |
| Filter cells | `sc.pp.filter_cells` or an `obs` mask | `ds.filter_cells` or `ds.auto_filter_cells` returns a cell-selection artifact without deleting cells or changing `I` |
| Select and normalize features | `sc.pp.normalize_total`, `sc.pp.log1p`, `sc.pp.highly_variable_genes` | `ds.select_hvgs`, then `ds.run_normalization`, using the same cell-selection ref |
| Run PCA and find neighbours | `sc.pp.pca`, then `sc.pp.neighbors` | `ds.run_pca`, then Scarf's neighbour-graph methods |
| Embed the graph | `sc.tl.umap` | `ds.run_umap` |
| Cluster cells | `sc.tl.leiden` | `ds.run_leiden_clustering`; Paris hierarchy diagnostics are covered in the advanced clustering guide |
| Find marker genes | `sc.tl.rank_genes_groups` | `ds.run_marker_search`, then `ds.get_markers` |
| Plot results | `sc.pl.*` | `ds.plots.embedding`, `ds.plots.dotplot`, and other `ds.plots` methods |
| Export an assay | `adata.write_h5ad` | `ds.to_anndata()` or `scarf.to_h5ad` |

Scarf marker search reports AUC, two-sided Mann-Whitney p-values, and within-group
Benjamini-Hochberg adjustment over tested features. It is not replicate-aware differential
expression. See {doc}`tutorials/graph_construction` for the complete manual graph chain and
{doc}`tutorials/scrna_seq` for stage-by-stage biological interpretation.

## Move data through H5AD

Scarf reads and writes H5AD, so Scarf and Scanpy can be used at different stages of one project.
Import an H5AD file into a Scarf store:

```python
import scarf

inspection = scarf.inspect_h5ad("data.h5ad")
reader = scarf.H5adReader.from_inspect(
    inspection,
    embedding_roles={"X_umap": "umap"},
    cluster_keys=("clusters",),
)
imported = scarf.H5adToZarr(
    reader,
    zarr_loc="data.zarr",
    analysis_assay="RNA",
).dump()
ds = scarf.DataStore("data.zarr")
ds.plots.embedding(
    layout=imported.embeddingArtifacts["X_umap"],
    color_by=imported.clusterArtifacts["clusters"],
)
```

Selected embeddings and clusterings become immutable artifacts rather than live metadata
columns. Use their returned refs directly or inspect a payload with `ds.load_artifact(ref)`.

Export to an in-memory `AnnData` object or directly to H5AD:

```python
adata = ds.to_anndata()
scarf.to_h5ad(ds.RNA, "analysis.h5ad")
```

To export a completed pipeline's frozen cells, feature universe, and result fields, pass its run:

```python
adata = ds.to_anndata(run=run)
scarf.to_h5ad(ds.RNA, "pipeline-analysis.h5ad", run=run)
```

Run export writes frozen UMAP coordinates to `obsm["X_umap"]` and frozen cluster labels to
`obs["clusters"]`. A run with `umap=False` does not invent an embedding.

`ds.to_anndata()` defaults to active cells and all features. Pass `feature_indexes` or
`feature_names` to subset features. Without `run`, `scarf.to_h5ad` writes the full assay to disk,
including cells with `I=False`, without first constructing an in-memory `AnnData`.

Counts and metadata transfer, but Scarf's neighbourhood graphs, provenance records, and
multimodal relationships do not map directly to AnnData. The exported H5AD may therefore need a
new neighbour graph in Scanpy. See {doc}`tutorials/import_and_export` for format details and export
options.
