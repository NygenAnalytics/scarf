(scarf_and_scanpy)=
# Coming from Scanpy or Seurat

If you already analyze single-cell data with [Scanpy](https://scanpy.readthedocs.io/) or
Seurat, this page maps familiar stages onto Scarf's API. It is a migration guide, not a
ranking. Scarf covers the core scRNA-seq path and reads and writes AnnData, so the two
fit together in one workflow. A short Seurat subsection follows the Scanpy material.

## What changes when you switch

| | Scarf | Scanpy |
|---|---|---|
| Primary object | `DataStore` over a Zarr store | `AnnData` in memory, or backed / Dask modes for larger data |
| Where data lives | On disk or object storage, read in place | Arrays on the `AnnData` object; backed modes keep matrices on disk |
| Modalities in-package | scRNA-seq, scATAC-seq, CITE-seq | scRNA-seq; spatial and other modalities via the wider scverse stack |
| Results persistence | Written into the Zarr store as you run steps | Held on the `AnnData` object (save explicitly) |
| Provenance | Artifacts with operation, parameters, inputs; reuse and `invalidate_cache` | Workflow history depends on how you record notebooks and `adata.uns` |

The main adjustment is that Scarf keeps counts and results on a store rather than in a
single in-memory object, and it reuses one neighbourhood graph across embedding,
clustering, and mapping. AnnData backed and Dask paths also avoid loading everything at
once; Scarf's default path is store-native Zarr with published artifact state. The stage
mapping below covers the rest.

## Stage mapping

Approximate mapping of common steps. Rows marked approximate are not one-to-one.
Store-backed plotting calls use `ds.plots`; the same functions can be called
standalone after `import scarf.plotting as splt`.

| Stage | Scanpy | Scarf |
|---|---|---|
| Load counts | `sc.read_*` → `AnnData` | Readers (`CrH5Reader`, `H5adReader`, …) → `*ToZarr` → `DataStore` |
| QC metrics | `sc.pp.calculate_qc_metrics` | Single-pass initialization on `DataStore` creation (`RNA_nCounts`, `RNA_nFeatures`, mito/ribo fractions, feature cell counts) |
| Filter cells | `sc.pp.filter_cells` | `filter_cells` / `auto_filter_cells` (marks cells inactive via cell key `I`, does not delete) |
| Normalize | `sc.pp.normalize_total`, `sc.pp.log1p` | `run_normalization` (also inside `ds.pipeline.run`) |
| Highly variable genes | `sc.pp.highly_variable_genes` | `mark_hvgs` |
| PCA / neighbours | `sc.pp.pca`, `sc.pp.neighbors` | `run_pca` → `build_embedding_initialization` → `build_ann_index` → `query_neighbors` → `build_connectivity_map`, or `ds.pipeline.run` |
| UMAP | `sc.tl.umap` | `run_umap` |
| Clustering | `sc.tl.leiden` | `run_leiden_clustering` (also `run_paris_clustering`) |
| Markers | `sc.tl.rank_genes_groups` | `run_marker_search` / `get_markers` (Mann-Whitney U scores and p-values; no FDR correction) |
| Gene-set activity | `decoupler` or `sc.tl.score_genes` | `run_waggr` / `run_aucell` |
| Plotting | `sc.pl.*` | `ds.plots.embedding`, `ds.plots.dotplot`, or equivalent `scarf.plotting` functions |
| Export | write H5AD | `to_anndata`, `scarf.to_h5ad`, MTX helpers |

## Scarf-specific mechanics

- **Assays**: a `DataStore` can hold multiple assays (for example `RNA` and `ADT`). Methods take an assay name when needed.
- **Filtering marks cells inactive**: filtered cells stay in the store; the boolean cell key `I` controls which cells participate in later steps.
- **Graph-centric pipeline**: embeddings, clustering, mapping, and multimodal integration reuse the neighbourhood graph published in `AssayState`.
- **Artifacts**: intermediate matrices and embeddings live under the Zarr hierarchy with provenance; see {doc}`concepts/provenance`.
- **On-disk / remote by default**: open local paths or `s3://` / `gs://` URIs with `storage_options`.

## Using Scarf and Scanpy together

Scarf reads and writes AnnData, so you can move between the two in one project. A common
pattern is to run graph-heavy or large-scale steps in Scarf, then export a subset for a
method that lives in the wider scverse ecosystem (for example scVI, CellRank, or Squidpy).

Some methods are not part of Scarf, including scVI, Scanorama, RNA velocity, and full
differential-expression pipelines with multiple-testing correction. When you need one of
these, export the assay and cell metadata with `to_anndata` / `to_h5ad` and continue in
the other tool.

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
3. Call `ds.pipeline.run` (or `mark_hvgs` plus the atomic graph chain).
4. Run `run_umap` and `run_leiden_clustering` if you did not include them in the recipe.
5. Use `ds.plots.embedding` for UMAP figures.
6. Export with `to_anndata` when you need a Scanpy-only method.

## If you know Seurat

| Seurat | Scarf (approximate) |
|---|---|
| `SeuratObject` | `DataStore` |
| Assays (`RNA`, `ADT`) | Assays on the store |
| `NormalizeData` / `ScaleData` / `RunPCA` / `FindNeighbors` | `run_normalization` → `run_pca` → ANN / neighbors / connectivity (or `ds.pipeline.run`) |
| `RunUMAP` / `FindClusters` | `run_umap` / `run_leiden_clustering` |
| Harmony integration | `run_harmony` after PCA, then continue the atomic chain |
| WNN | `integrate_assays(..., method="wnn")` |
| Sketching / representative cells | TopACeDo via the downsampling tutorial |

## Further reading

- [Scanpy documentation](https://scanpy.readthedocs.io/en/stable/)
- [Seurat PBMC clustering vignette](https://satijalab.org/seurat/articles/pbmc3k_tutorial)
- [Seurat integration introduction](https://satijalab.org/seurat/articles/integration_introduction)

## Next steps

- {ref}`Quick start <quickstart>`
- {doc}`whats_new_in_1_0`
- {doc}`tutorials/atomic_graph_operations`
- {doc}`tutorials/scrna_seq`
- {doc}`tutorials/gene_set_enrichment`
- {doc}`tutorials/import_and_export`
