(scarf_and_scanpy)=
# Scarf and Scanpy

Scarf and [Scanpy](https://scanpy.readthedocs.io/) both support core scRNA-seq analysis.
They differ in data model, memory behaviour, and which methods they ship. This page maps
stages between the two tools and notes when to use each. A short Seurat subsection follows
the Scanpy material.

## What each tool is built for

| | Scarf | Scanpy |
|---|---|---|
| Primary object | `DataStore` over a Zarr store | `AnnData` in memory (or backed modes) |
| Design focus | Lower-memory neighbourhood-graph workflows | Broad method catalog and ecosystem |
| Modalities in-package | scRNA-seq, scATAC-seq, CITE-seq | scRNA-seq; spatial and other modalities via the wider scverse stack |
| Results persistence | Written into the Zarr store as you run steps | Held on the `AnnData` object (save explicitly) |

## Stage mapping

Approximate mapping of common steps. Rows marked approximate are not one-to-one.
Store-backed plotting calls use `ds.plots`; the same functions can be called
standalone after `import scarf.plotting as splt`.

| Stage | Scanpy | Scarf |
|---|---|---|
| Load counts | `sc.read_*` → `AnnData` | Readers (`CrH5Reader`, `H5adReader`, …) → `*ToZarr` → `DataStore` |
| QC metrics | `sc.pp.calculate_qc_metrics` | Computed on `DataStore` creation (`RNA_nCounts`, `RNA_nFeatures`, mito/ribo fractions) |
| Filter cells | `sc.pp.filter_cells` | `filter_cells` / `auto_filter_cells` (marks cells inactive via cell key `I`, does not delete) |
| Normalize | `sc.pp.normalize_total`, `sc.pp.log1p` | Runs inside `make_graph` (library-size style normalization; `log_transform=True` by default) |
| Highly variable genes | `sc.pp.highly_variable_genes` | `mark_hvgs` |
| PCA / neighbours | `sc.pp.pca`, `sc.pp.neighbors` | `make_graph` (PCA or LSI, ANN, KNN graph, centroids) |
| UMAP | `sc.tl.umap` | `run_umap` |
| Clustering | `sc.tl.leiden` | `run_leiden_clustering` (also Paris hierarchical clustering) |
| Markers | `sc.tl.rank_genes_groups` | `run_marker_search` / `get_markers` (Mann-Whitney U scores and p-values; no FDR correction) |
| Gene-set activity | `decoupler` or `sc.tl.score_genes` | `run_waggr` / `run_aucell` |
| Plotting | `sc.pl.*` | `ds.plots.embedding`, `ds.plots.dotplot`, or equivalent `scarf.plotting` functions |
| Export | write H5AD | `to_anndata`, `scarf.to_h5ad`, MTX helpers |

## Scarf-specific mechanics

- **Assays**: a `DataStore` can hold multiple assays (for example `RNA` and `ADT`). Methods take an assay name when needed.
- **Filtering marks cells inactive**: filtered cells stay in the store; the boolean cell key `I` controls which cells participate in later steps.
- **Graph-centric pipeline**: embeddings, clustering, mapping, and multimodal integration reuse the neighbourhood graph from `make_graph`.
- **On-disk by default**: intermediate matrices and embeddings live under the Zarr hierarchy.

## When to use Scarf, Scanpy, or both

Use **Scarf** when memory is limited, when you want built-in Harmony / WNN / mapping / TopACeDo
downsampling, or when you prefer results that persist in Zarr as you work.

Use **Scanpy** when you need the wider ecosystem (for example scVI, CellRank, Squidpy) or when
the dataset is small enough that in-memory AnnData is simplest.

Use **both** when Scarf handles large-scale UMAP and clustering, then you export a subset with
`to_anndata` / `to_h5ad` for methods Scarf does not include.

Do not treat runtime or memory numbers on this page as benchmarks. Relative behaviour depends
on dataset size, hardware, and parameters.

## H5AD and AnnData round-trip

Import:

```python
reader = scarf.H5adReader("data.h5ad")
scarf.H5adToZarr(reader, zarr_loc="data.zarr").dump()
ds = scarf.DataStore("data.zarr")
```

Export:

```python
adata = ds.to_anndata()
# or
scarf.to_h5ad(ds.RNA, "subset.h5ad")
```

Not everything round-trips. Neighbourhood graphs, Scarf-specific metadata keys, and multimodal
layouts may need to be recomputed after export. Prefer exporting the assay and cell metadata you
need for the next tool.

## Checklist for Scanpy users

1. Convert counts to Zarr once; open a `DataStore` pointing at that store.
2. Inspect QC columns (`RNA_nCounts`, `RNA_nFeatures`, …) before filtering.
3. Call `mark_hvgs`, then `make_graph` (normalization and PCA happen here).
4. Run `run_umap` and `run_leiden_clustering` on the graph.
5. Use `ds.plots.embedding` for UMAP figures.
6. Export with `to_anndata` when you need a Scanpy-only method.

## If you know Seurat

| Seurat | Scarf (approximate) |
|---|---|
| `SeuratObject` | `DataStore` |
| Assays (`RNA`, `ADT`) | Assays on the store |
| `NormalizeData` / `ScaleData` / `RunPCA` / `FindNeighbors` | Mostly inside `make_graph` |
| `RunUMAP` / `FindClusters` | `run_umap` / `run_leiden_clustering` |
| Harmony integration | `make_graph(..., harmonize=True, batch_columns=...)` |
| WNN | `integrate_assays(..., method="wnn")` |
| Sketching / representative cells | TopACeDo via the downsampling tutorial |

## Next steps

- {ref}`Quick start <quickstart>`
- {doc}`tutorials/scrna_seq`
- {doc}`tutorials/gene_set_enrichment`
- {doc}`tutorials/import_and_export`
