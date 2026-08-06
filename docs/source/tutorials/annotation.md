---
description: Interpret marker statistics, inspect known markers, and assign cell labels.
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

(annotation)=

# Interpreting markers and assigning cell types

This chapter starts from a clustered PBMC store and shows how to read marker tables, plot known markers, and assign labels without treating cluster IDs as cell types.

## Prerequisites

- {doc}`scrna_seq` (or an equivalent clustered Zarr store)
- Familiarity with cell keys (`I` and custom boolean columns)

## What you will learn

- Retrieve markers with `get_markers`
- Color embeddings by gene expression
- Write annotation columns into cell metadata
- Distinguish cell-level marker evidence from replicate-aware differential expression

## Standalone setup

Annotation starts from clusters and a marker table.
The published PBMC store has both, so this page opens it and goes straight to reading the evidence.
{doc}`clustering` covers how the partition is chosen and scored.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import scarf

scarf.configure_output(level='WARNING', progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
)
```

`RNA_clusters` holds the silhouette-selected partition (Leiden or Paris), and the marker table is indexed under the same column.
Confirm cluster sizes and their layout before reading markers.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    columns=['RNA_clusters'],
    key='I',
)['RNA_clusters'].value_counts().sort_index()
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_clusters',
)
```

## 1. Marker tables

`get_markers` returns genes ranked by marker score.
Pass a `group_id` for one cluster, or `group_id=None` for every cluster in one long table with a `group_id` column.
Columns include scores, expression fractions, fold change, a two-sided Mann-Whitney `p_value`, AUC, and `p_value_adjusted`.

```{code-cell} ipython3
markers = ds.get_markers(
    group_key='RNA_clusters',
    group_id='1',
    min_score=-1,
    min_frac_exp=-1,
)
markers[
    [
        'feature_name',
        'score',
        'frac_exp',
        'fold_change',
        'auc',
        'p_value',
        'p_value_adjusted',
    ]
].head(10)
```

Interpret the columns together:

- `score` is the group's mean dense-rank as a share of the sum of mean dense-ranks across all groups.
- `frac_exp` is the fraction of target-group cells with detected expression.
- `fold_change` compares average expression in the target and reference cells.
- `auc` is the probability that a randomly selected target cell has a higher value than a randomly selected reference cell.
  Values near 0.5 provide little separation.
- `p_value` is the two-sided Mann-Whitney result.
- `p_value_adjusted` applies Benjamini-Hochberg correction within this one-versus-rest group over all tested features.

`fold_change`, `auc`, and the Mann-Whitney columns cover one-versus-rest expression contrast.
Both p-value columns treat cells as observations.
They are useful for marker ranking but are not replicate-aware differential expression.
Groups need at least two target and two reference cells; smaller comparisons fail rather than returning unstable statistics.
Older marker tables remain readable.
AUC may be missing until marker search is rerun.
`p_value_adjusted` can be synthesized on read with Benjamini-Hochberg correction when raw `p_value` is present.

```{code-cell} ipython3
ds.plots.marker_heatmap(
    group_key='RNA_clusters',
    topn=5,
    figsize=(5, 9),
)
```

Rows are top markers per cluster; use them with known lineage genes, not as FDR DE.

## 2. Known markers on the embedding

Before assigning labels, check where the panel genes rank across clusters, then confirm them on the UMAP.

```{code-cell} ipython3
markers = ds.get_markers(
    group_key='RNA_clusters',
    group_id=None,
    min_score=-1,
    min_frac_exp=-1,
)
(
    markers[markers['feature_name'].astype(str).isin(['CD14', 'MS4A1', 'CD3D'])]
    .sort_values(['feature_name', 'score'], ascending=[True, False])
    .groupby('feature_name', as_index=False)
    .head(1)[['feature_name', 'group_id', 'score', 'auc', 'frac_exp']]
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['CD14', 'MS4A1', 'CD3D'],
    n_columns=3,
    sort_values=True,
)
```

CD14, MS4A1, and CD3D mark monocyte-, B-, and T-cell-like regions when those lineages are present.
The lookup above names the highest-scoring cluster for each gene; the UMAP shows whether that signal is spatially coherent.

## 3. Assign labels

Map `RNA_clusters` to names using the marker UMAPs and marker tables.
Cluster IDs are not stable across parameter changes, so this cell picks the cluster where each lineage gene ranks highest among markers, then leaves other clusters as `Cluster {id}`.

```{code-cell} ipython3
cluster_ids = ds.cells.fetch_all('RNA_clusters')
unique = sorted(
    {str(c) for c in cluster_ids},
    key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
)

label_map = {c: f'Cluster {c}' for c in unique}
for gene, name in [('CD14', 'Monocytes'), ('MS4A1', 'B cells'), ('CD3D', 'T cells')]:
    hit = markers[markers['feature_name'].astype(str) == gene]
    if hit.empty:
        continue
    cid = str(hit.sort_values('score', ascending=False).iloc[0]['group_id'])
    label_map[cid] = name

label_map
```

```{code-cell} ipython3
labels = np.array([label_map[str(c)] for c in cluster_ids], dtype=object)
ds.cells.insert(column_name='cell_type', values=labels, overwrite=True)
ds.cells.to_pandas_dataframe(
    columns=['cell_type'],
    key='I',
)['cell_type'].value_counts()
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='cell_type',
)
```

Assigned labels replace numeric cluster IDs where the panel genes ranked highest.
Compare `label_map` and the `cell_type` counts with the gene UMAPs before treating unnamed `Cluster {id}` groups as distinct types.

## 4. Relabel clusters with `smart_label`

`smart_label` renames values in one categorical column from the most frequent overlap with another column.
Here `RNA_clusters` IDs are rewritten from the `cell_type` labels just assigned.
Letter suffixes are always appended (`Monocytes` becomes `Monocytesa`); when several clusters share a base label they get ordered suffixes such as `a`, `b`.

```{code-cell} ipython3
ds.smart_label(
    to_relabel='RNA_clusters',
    base_label='cell_type',
    new_col_name='leiden_by_type',
)
pd.crosstab(
    pd.Series(ds.cells.fetch('RNA_clusters'), name='RNA_clusters'),
    pd.Series(ds.cells.fetch('leiden_by_type'), name='leiden_by_type'),
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='leiden_by_type',
)
```

`leiden_by_type` is a convenience labeling of clusters, not an automated ontology annotation.
The crosstab shows which cluster IDs were rewritten and how letter suffixes were assigned.

## 5. Annotate scATAC-seq with gene scores

Peak IDs are difficult to interpret directly.
`add_melded_assay` can combine ATAC peaks that overlap gene bodies and promoter regions into a `GeneScores` assay, which can then be plotted like RNA features.

This page stays on the RNA store.
An executable GeneScores path, including the prepared BED download and marker panel on an ATAC UMAP, is in {doc}`scatac_seq`.
The API sketch below is for coordinate melding on an ATAC assay you already have.

The external BED file has no header and uses tab-separated columns in this order: chromosome, start, end, gene ID, gene name, and optional strand.
Its genome build must match the peak coordinates.
Promoter offsets should be chosen before melding and reported with the annotation source.

```python
ds.add_melded_assay(
    from_assay="ATAC",
    external_bed_fn="genes_with_promoters.bed.gz",
    peaks_col="ids",
    renormalization=False,
    assay_label="GeneScores",
    assay_type="RNA",
)
```

`renormalization=False` retains the summed TF-IDF-normalized peak signal before the resulting RNA-like assay applies its own normalization.
Set it differently only when a constant total across melded features matches the intended interpretation.
Features with no overlapping peaks remain present but invalid.

Gene scores are accessibility summaries, not measured RNA expression.
Confirm cell identities with several loci, known chromatin biology, and the coordinate overlap rate.
A flat marker panel can indicate a genome-build mismatch or insufficient peak overlap.

```{raw} html
<span id="subset-and-recluster"></span>
```

## 6. Subset and recluster

Subset graph construction and validation now live in {doc}`clustering`.
This page keeps annotation focused on evidence and label assignment.

## Choose an annotation path

Use the marker workflow above when assigning labels within the store being analysed.
Use {doc}`mapping_and_label_transfer` when query cells should inherit evidence or labels from a fixed external reference.
Use {doc}`reference_atlases` when that reference must be built, serialized, reloaded, and checked for repeated mapping.

## Common mistakes and limitations

- Treating cluster IDs as biologically stable across resolutions
- Overwriting annotation columns without keeping the clustering key you used
- Claiming replicate-aware DE from `run_marker_search` alone; use `p_value_adjusted` only as a within-group cell-level marker correction
- Assigning a cell type from one marker gene or one cluster statistic
- Interpreting ATAC gene scores as measured transcript abundance

Scarf does not ship an automated ontology annotator.
Labels here are assigned from marker evidence and must be reviewed against the study context.
