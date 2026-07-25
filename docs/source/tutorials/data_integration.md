---
description: Merge scRNA-seq datasets with Harmony batch correction, partial PCA, and LISI integration metrics.
jupytext:
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

(harmony_batch_correction)=
(integration_guide)=
(lisi_metrics)=
(integration_metrics)=

# Merge, Harmony, and partial PCA

This tutorial merges datasets from different Zarr files, corrects batch effects with
partial PCA or Harmony, and quantifies integration with LISI and related metrics.

**When to use which approach**

| Goal | Approach |
|---|---|
| Merge two scRNA-seq batches in one object | `AssayMerge`, then this page |
| Correct batch effects after merge | Atomic Harmony (`run_harmony`) or partial PCA (`pca_cell_key`) |
| Integrate RNA + ADT in the same cells | {ref}`CITE-seq SNN / WNN <multimodal_integration>` |
| Map query cells onto a reference | {doc}`mapping_and_label_transfer` |
| Measure integration quality | `metric_*` helpers in the final section below |

Scarf does not ship Scanorama, BBKNN, scVI, ComBat, or other external integrators.
Export subsets with `to_anndata` or `SubsetZarr` when you need those tools. See
{doc}`../reference/faq`.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Two or more Zarr stores with a shared RNA feature space

## What you will learn

- Merge assays from separate Zarr files with `AssayMerge`
- Compare a naive joint analysis with partial PCA and Harmony using atomic graph ops
- Quantify mixing and label preservation with LISI and related metrics


## Dataset

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')
```

---
## 1) Fetch datasets in Zarr format

Use the same Kang PBMC datasets as in {ref}`mapping and label transfer <data_projection>`.
Download the prepared Zarr stores from the `scarf_docs` Cytebase catalog.

```{code-cell} ipython3
repository = scarf.cytebase.connect("scarf_docs")
ctrl_path = repository.download_dataset(
    name='kang_15K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True
)
stim_path = repository.download_dataset(
    name='kang_14K_ifnb-pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True
)
```

The Zarr files need to be loaded as a DataStore before they can be merged:

```{code-cell} ipython3
ds_ctrl = scarf.DataStore(f'{ctrl_path}/data.zarr', nthreads=4)

ds_ctrl
```

```{code-cell} ipython3
ds_stim = scarf.DataStore(f'{stim_path}/data.zarr', nthreads=4)

ds_stim
```

---
## 2) Merging datasets

The merging step will make sure that the features are in the same order as in the merged file. The merged data will be dumped into a new Zarr file. Use `AssayMerge` to merge multiple samples.

```{code-cell} ipython3
scarf.AssayMerge(
    zarr_path='scarf_datasets/kang_merged_pbmc_rnaseq.zarr',
    assays=[ds_ctrl.RNA, ds_stim.RNA],
    names=['ctrl', 'stim'],
    merge_assay_name='RNA',
    prepend_text='orig',
    reset_cell_filter=False,
    source_column='sample_id',
    overwrite=True,
).dump()
```

Load the merged Zarr file as a DataStore:

```{code-cell} ipython3
ds = scarf.DataStore(
    'scarf_datasets/kang_merged_pbmc_rnaseq.zarr',
    nthreads=4
)
```

The merge removes calculated graphs and embeddings. It keeps each input cell filter because `reset_cell_filter=False`, stores the input metadata with the `orig_` prefix, and writes the corresponding entry from `names` to `sample_id`. Counts and cell metadata receive the same row permutation, so these columns remain aligned.

```{code-cell} ipython3
ds
```

The cell table now contains `sample_id`, the aligned `orig_cluster_labels`, and the preserved `I` filter. The source name is also prepended to each barcode in `ids`.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    ['ids', 'sample_id', 'orig_cluster_labels', 'I']
).head()
```

Now we can check the number of cells from each of the samples:

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    ['sample_id'],
    key='I'
)['sample_id'].value_counts()
```

---
## 3) Naive analysis of merged datasets

By naive, we mean that we make no attempt to remove/account for the latent factors that might contribute to batch effect or treatment-specific effect.
It is usually a good idea to perform a 'naive' pipeline to get an idea about the degree of batch effects.

+++

We start with detecting the highly variable genes:

```{code-cell} ipython3
ds.mark_hvgs(
    min_cells=10,
    top_n=2000,
    min_mean=-3, 
    max_mean=2,
    max_var=6
)
```

Next, build the neighbourhood graph with the atomic chain: normalization, PCA, embedding
initialization, the ANN index, the neighbour query, and the connectivity map.

```{code-cell} ipython3
normalized = ds.run_normalization(feat_key='hvgs')
pca = ds.run_pca(normalized, dims=25)
ds.build_embedding_initialization(pca)
ann = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(ann, k=21)
ds.build_connectivity_map(neighbors)

ds.load_graph()
```

Each step returns a reference to the artifact it wrote. Unlike {doc}`scrna_seq`, this page
captures those references because later sections reuse the same normalized counts with a
different PCA basis, and Harmony has to sit between PCA and the neighbour index.


Calculating UMAP embedding of cells:

```{code-cell} ipython3
ds.run_umap(
    n_epochs=250, 
    spread=5,
    min_dist=1,
    parallel=True
)
```

The coordinates are now cell metadata columns:

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    ['sample_id', 'RNA_UMAP1', 'RNA_UMAP2'],
    key='I'
).head()
```

Visualization of cells from the two samples in the 2D UMAP space:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='sample_id',
    legend_loc='right',
)
```

Visualization of cluster labels in the 2D UMAP space:

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='orig_cluster_labels',
    legend_loc='right',
)
```

---
## 4) Partial PCA training to reduce batch effects

The plots above show that cells from the two source datasets are separated. These datasets
also represent different treatment conditions, so source, treatment, and any technical effects
are confounded. This example demonstrates the correction APIs, but it cannot determine which
part of the separation is technical rather than a true interferon response.

Partial PCA trains the basis on a trusted reference subset. Variation absent from that subset
contributes less to the resulting graph, including both technical variation and real
condition-specific biology. Do not use the corrected coordinates as input for condition-level
differential expression.

+++

First, we need to create a boolean column in the cell attribute table. This column will indicate whether a cell belongs to one of the samples. Here we will create a new column `is_ctrl` and mark the values as True when a cell belongs to the `ctrl` sample.

```{code-cell} ipython3
ds.cells.insert(
    column_name=f'is_ctrl',
    values=(ds.cells.fetch_all('sample_id') == 'ctrl'),
    overwrite=True
)
```

Pass `pca_cell_key='is_ctrl'` to `run_pca` so only control cells define the PCA basis.
Reuse the same normalized artifact from the naive analysis, then run UMAP with
`label='pUMAP'` so the new coordinates land in `RNA_pUMAP` instead of overwriting the naive
`RNA_UMAP` columns. The `RNA` prefix comes from the assay name.

```{code-cell} ipython3
pca_partial = ds.run_pca(normalized, dims=25, pca_cell_key='is_ctrl')
ds.build_embedding_initialization(pca_partial)
ann = ds.build_ann_index(pca_partial)
neighbors = ds.query_neighbors(ann, k=21)
ds.build_connectivity_map(neighbors)

ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label='pUMAP'
)
```

Visualize the new UMAP

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_pUMAP',
    color_by='sample_id',
    legend_loc='right',
)
```

Cluster labels on the new UMAP often mix more by cell type than before, but treatment and
batch remain confounded, so improved mixing is not proof that biology was preserved.

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_pUMAP',
    color_by='orig_cluster_labels',
    legend_loc='right',
)
```

---
## 5) Harmony batch correction

Harmony corrects the PCA embedding before ANN and neighbor query. Call `run_harmony`
between PCA and `build_ann_index`. The graph is then built from the corrected coordinates,
while `build_embedding_initialization` still takes the uncorrected PCA because it accepts a
reduction artifact. That only seeds the starting positions for UMAP, which reads its edges
from the corrected graph.

```{warning}
Here `sample_id` distinguishes control from stimulated cells as well as the source dataset.
Harmony can therefore remove genuine treatment signal. In a real study, use technical batch
columns that are not perfectly confounded with the biological comparison.
```

```{code-cell} ipython3
pca_full = ds.run_pca(normalized, dims=25)
corrected = ds.run_harmony(['sample_id'], pca_full)
ds.build_embedding_initialization(pca_full)
ann = ds.build_ann_index(corrected)
neighbors = ds.query_neighbors(ann, k=21)
ds.build_connectivity_map(neighbors)

ds.run_umap(
    n_epochs=250,
    spread=5,
    min_dist=1,
    parallel=True,
    label='hUMAP'
)

ds.run_leiden_clustering(resolution=1.0)
```


```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_hUMAP',
    color_by='sample_id',
    legend_loc='right',
)

ds.plots.embedding(
    layout_key='RNA_hUMAP',
    color_by='orig_cluster_labels',
    legend_loc='right',
)
```

Harmony is often preferred when multiple batches need to mix without choosing a single reference sample. Partial PCA (section 4) is lighter when one sample defines the embedding.

---
## 6) Quantifying integration quality

The UMAP plots suggest that partial PCA and Harmony mix the two samples, but a visual read is
not enough. Scarf provides several metrics that quantify integration from different angles.
The code below evaluates the latest graph, which is the Harmony graph at this point. To compare
the naive, partial PCA, and Harmony results, calculate and retain these metrics immediately
after building each graph.

**LISI** measures how well a label mixes inside each cell's KNN neighborhood. Running it on
`sample_id` tells us whether batches are mixed, while running it on `orig_cluster_labels`
checks that cell types are still grouped. Good integration raises batch LISI while keeping
cell-type LISI low. With `save_result=True` the per-cell scores are written back as
`lisi__sample_id__*` columns, which you can overlay on the UMAP layouts.

Default `perplexity=30` expects roughly `3 * perplexity` graph neighbors. This tutorial uses
`k=21`, so the call sets `perplexity=7` explicitly. If you omit `perplexity` on a smaller
graph, Scarf reduces it automatically and prints a warning. The scIB benchmark convention
compares neighborhoods with 15, 50, or 90 neighbors, so use a matching `k` when comparing
Scarf results to a published benchmark.


`metric_lisi` returns one score per cell for each label, which is what the saved columns
hold. Summarize them rather than printing the raw arrays.

```{code-cell} ipython3
import numpy as np
import pandas as pd

lisi = ds.metric_lisi(
    label_colnames=['sample_id', 'orig_cluster_labels'],
    save_result=True,
    perplexity=7,
)
pd.DataFrame(
    [
        {
            'label': label,
            'median': float(np.median(scores)),
            'mean': float(np.mean(scores)),
        }
        for label, scores in lisi
    ]
)
```

A batch median well above 1 means neighbourhoods contain both samples, while a cell-type
median near 1 means neighbourhoods stay within one annotated type. Those are the two
objectives, and they pull against each other.

The remaining metrics each reduce to a single number, so collect them in one table.
**iLISI** summarizes batch mixing with scIB scaling and **cLISI** summarizes preservation of
the imported labels, both higher-is-better. **Proportional batch mixing** uses mean batch
LISI with observed batch sizes, which helps when batches are imbalanced. **Graph
connectivity** reports how much of each imported label stays in its largest connected
component. **Label concordance** compares two labelings of the same cells, here fresh Leiden
clusters against imported annotations, which measures label agreement and not batch mixing.

```{code-cell} ipython3
pd.Series(
    {
        'iLISI (batch mixing)': ds.metric_ilisi(
            batch_colname='sample_id',
            perplexity=7,
        ),
        'cLISI (label purity)': ds.metric_clisi(
            label_colname='orig_cluster_labels',
            perplexity=7,
        ),
        'proportional batch mixing': ds.metric_proportional_batch_mixing(
            label_colname='sample_id',
            perplexity=7,
        ),
        'graph connectivity': ds.metric_graph_connectivity(
            label_colname='orig_cluster_labels',
        ),
        'label concordance (ARI)': ds.metric_label_concordance(
            label_columns=['RNA_leiden_cluster', 'orig_cluster_labels'],
            metric='ari',
        ),
    }
).round(3)
```

**Graph silhouette** is per cluster rather than per dataset, scoring how separated each
cluster is from its nearest neighbour, from -1 to 1. Read it alongside the batch metrics,
because over-correction shows up here as clusters losing separation.

```{code-cell} ipython3
silhouette = ds.metric_graph_silhouette(res_label='leiden_cluster')
pd.Series(silhouette).describe().round(3)
```

Negative values flag clusters that sit closer to a neighbouring cluster than to their own
cells, which is a signal to revisit the clustering resolution.

## Common mistakes and limitations

- Treating sample identity as a technical batch when it is confounded with treatment
- Using corrected embeddings as input for condition-level differential expression
- Judging integration from UMAP alone without batch and label metrics
- Comparing LISI scores across graphs built with different `k` or perplexity

## Saved results

The merged Zarr store holds aligned counts and metadata. Each graph rebuild writes its own
UMAP/Leiden columns (`RNA_UMAP`, `RNA_pUMAP`, `RNA_hUMAP`, and so on). With
`save_result=True`, per-cell LISI scores are stored as `lisi__*` columns.

## Further reading

- Korsunsky et al. 2019, Harmony: https://doi.org/10.1038/s41592-019-0619-0

## Next steps

- {doc}`cite_seq`
- {doc}`mapping_and_label_transfer`
- {doc}`../reference/faq`
- {doc}`../reference/api/integration`
