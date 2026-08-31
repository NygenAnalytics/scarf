---
description: Interpret prepared scATAC-seq clusters with GeneScore marker maps.
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Identify accessibility populations with scATAC-seq

Use a prepared LSI analysis to find broad PBMC accessibility states, then test that
interpretation with gene-score marker maps. The page follows one path and does not repeat the
expensive wide-matrix reduction.

## Open the prepared ATAC result

```{code-cell} ipython3
import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_10K_pbmc-v1_atacseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="ATAC",
    nthreads=4,
)
```

The prepared store contains one complete Leiden result and its linked UMAP. Exact provenance
filters reopen them without walking the full LSI ancestry in notebook code.

```{code-cell} ipython3
[clusters] = ds.list_artifacts(
    from_assay="ATAC",
    kind="cluster_labels",
    operation="run_leiden_clustering",
    complete_only=True,
)
[umap] = ds.list_artifacts(
    from_assay="ATAC",
    kind="embedding",
    operation="run_umap",
    complete_only=True,
)
```

### Question: does the accessibility graph contain distinct populations?

```{code-cell} ipython3
ds.plots.embedding(
    layout=umap,
    color_by=clusters,
    legend_loc="on_data",
)
```

The graph separates several accessibility states. Cluster numbers alone are not biological names,
so the next figure asks whether known lineage loci support a PBMC interpretation.

## Build GeneScores from peak coordinates

GeneScores summarize TF-IDF-normalized accessibility over gene bodies and their promoter regions.
The downloaded BED file uses the same GRCh37 coordinate build as this peak matrix.

```{code-cell} ipython3
annotations = scarf.cytebase.connect("scarf_docs").download_dataset(
    "annotations",
    destination="scarf_datasets",
)

ds.add_melded_assay(
    from_assay="ATAC",
    external_bed_fn=(
        f"{annotations}/human_GRCh37_gencode_v38_gene_body.bed.gz"
    ),
    peaks_col="ids",
    renormalization=False,
    assay_label="GeneScores",
    assay_type="RNA",
)
```

Some annotation features have no overlapping peak in this matrix. They remain zero-count features;
the interpretation below relies only on the displayed loci with observed signal.

### Question: which broad lineages explain the accessibility regions?

```{code-cell} ipython3
ds.plots.embedding(
    layout=umap,
    from_assay="GeneScores",
    color_by=["CD3D", "MS4A1", "LEF1", "NKG7", "TREM1", "LYZ"],
    clip_fraction=0.01,
    n_columns=3,
    sort_values=True,
)
```

CD3D and LEF1 support T-cell accessibility, MS4A1 supports B cells, NKG7 highlights cytotoxic or
NK-like regions, and TREM1 with LYZ supports myeloid populations. These maps justify broad lineage
interpretation, but they do not support assigning every cluster a definitive cell type from this
small panel alone.

## Substitute your own input

For another peak-count matrix, the corresponding atomic path is:

```python
cells = own_ds.auto_filter_cells()
peaks = own_ds.select_prevalent_peaks(cells, top_n=25_000)
normalized = own_ds.run_normalization(cells, peaks)
lsi = own_ds.run_lsi(normalized, dims=50, skip_first=True)
initialization = own_ds.build_embedding_initialization(lsi)
neighbors = own_ds.query_neighbors(own_ds.build_ann_index(lsi), k=21)
graph = own_ds.build_connectivity_map(neighbors)
layout = own_ds.run_umap(graph, initialization)
clusters = own_ds.run_leiden_clustering(graph)
```

Convert and open the new count matrix as shown in {doc}`import_and_export`, then choose filtering,
peak count, LSI dimensions, and clustering policy for that dataset. The {doc}`quality_control`,
{doc}`feature_selection`, {doc}`dimensionality_reduction`,
{doc}`graph_construction`, and {doc}`clustering` guides own those decisions. Use
{doc}`reuse_and_tracing` when you need to inspect the exact artifact lineage.

## Limits of this result

- GeneScores are accessibility proxies, not measured RNA expression.
- Gene annotation and peak coordinates must use compatible genome builds.
- Marker accessibility supports broad states here; more loci and external evidence are needed for
  final annotation.
