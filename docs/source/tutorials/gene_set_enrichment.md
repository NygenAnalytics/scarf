---
description: Calculate per-cell gene-set activity with WAGGR or AUCell.
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

(gene_set_enrichment)=

# Gene-set activity scoring

Score pathway or cell-state signatures per cell with WAGGR or AUCell. Both methods stream
RNA counts from the Zarr store and persist activity scores for later use. These
methods do not calculate enrichment p-values.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with feature names that match the identifiers in your gene sets
- A basic understanding of cell metadata and embeddings

## What you will learn

- Read gene sets from GMT and inspect feature overlap
- Score weighted signatures with WAGGR
- Score rank-based signatures with AUCell
- Load selected score columns without materializing the full result

## Dataset

This page opens the published 5K PBMC store, which already carries the UMAP the
score plots are drawn on. Signature scoring itself reads normalized counts, not
the graph.

```{code-cell} ipython3
from pathlib import Path

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


## 1) Read and inspect gene sets

GMT stores one source per line. The first field is the source name, the second is a
description, and the remaining fields are target genes. `read_gmt` returns one
source-target row per gene.

```{code-cell} ipython3
gmt_path = Path('scarf_datasets/pbmc_signatures.gmt')
gmt_path.write_text(
    'T_cell\tna\tCD3D\tCD3E\tTRAC\tLTB\tIL7R\n'
    'B_cell\tna\tMS4A1\tCD79A\tCD37\tCD74\tHLA-DRA\n'
    'Myeloid\tna\tLST1\tS100A8\tS100A9\tCTSS\tFCER1G\n',
    encoding='utf-8',
)
gene_sets = scarf.read_gmt(gmt_path)
gene_sets
```

Targets are matched to active RNA feature names without case sensitivity. `tmin` is applied
after matching, so a source is retained only when enough of its targets are present. Missing
targets do not need to be removed from the input table first.

```{code-cell} ipython3
available = {str(name).upper() for name in ds.RNA.feats.fetch('names')}
(
    gene_sets.assign(
        matched=gene_sets['target'].str.upper().isin(available),
    )
    .groupby('source')['matched']
    .agg(['sum', 'count'])
)
```

## 2) Score weighted signatures with WAGGR

WAGGR applies edge weights to library-size-normalized expression. `wmean` divides each
weighted sum by the sum of absolute weights, while `wsum` leaves the weighted sum
unscaled. Signed weights are supported.

```{code-cell} ipython3
weighted_sets = gene_sets.assign(weight=1.0)
weighted_sets.loc[
    weighted_sets['target'].isin(['S100A8', 'S100A9']),
    'weight',
] = 1.5

waggr = ds.run_waggr(
    weighted_sets,
    label='pbmc_waggr',
    mode='wmean',
    tmin=3,
    overwrite=True,
)
waggr.source_names, waggr.data.shape
```

WAGGR uses Scarf's default RNA library-size normalization. Set `log_transform=True` to
apply `log1p` before aggregation.

## 3) Score rank recovery with AUCell

AUCell ranks the selected RNA features within each cell and measures how early a source's
targets are recovered. Scores range from zero to one. Network weights are ignored.

The {term}`feat_key` defines the ranking universe. Here the default `I` feature key ranks all active
features. `n_up=500` evaluates recovery within the top 500 ranks.

```{code-cell} ipython3
aucell = ds.run_aucell(
    gene_sets,
    label='pbmc_aucell',
    tmin=3,
    n_up=500,
    tie_seed=0,
    overwrite=True,
)
aucell.source_names, aucell.data.shape
```

The same `tie_seed` gives a deterministic global ordering for equal expression values.
Changing `n_up`, `tie_seed`, the feature selection, or the network creates a different
execution.

## 4) Load selected sources and visualize scores

`get_enrichment` returns a lazy result. Selecting sources first avoids loading unrelated
columns. The values below are activity scores, not p-values.

```{code-cell} ipython3
aucell_cols = []
for source in ['T_cell', 'B_cell', 'Myeloid']:
    result = ds.get_enrichment('pbmc_aucell', sources=[source])
    col = f'{source}_AUCell'
    ds.cells.insert(
        col,
        result.data.compute().ravel(),
        key=result.cell_key,
        overwrite=True,
    )
    aucell_cols.append(col)

present = [c for c in aucell_cols if c in ds.cells.columns]
if present:
    ds.plots.embedding(
        layout_key='RNA_UMAP',
        color_by=present,
        n_columns=3,
        sort_values=True,
    )
```

AUCell scores highlight lineage-consistent regions: T-cell, B-cell, and Myeloid scores
peak in separate parts of the UMAP when those populations are present.

```{code-cell} ipython3
waggr_myeloid = ds.get_enrichment('pbmc_waggr', sources=['Myeloid'])
ds.cells.insert(
    'Myeloid_WAGGR',
    waggr_myeloid.data.compute().ravel(),
    key=waggr_myeloid.cell_key,
    overwrite=True,
)

compare = [
    c for c in ['Myeloid_WAGGR', 'Myeloid_AUCell'] if c in ds.cells.columns
]
if len(compare) == 2:
    ds.plots.embedding(
        layout_key='RNA_UMAP',
        color_by=compare,
        n_columns=2,
        sort_values=True,
    )
```

WAGGR and AUCell both mark myeloid-like cells here, but the score scales differ because
one aggregates weighted expression and the other measures within-cell rank recovery.

## Choosing a method

- Use WAGGR when edge weights or signed targets carry useful information and expression
  magnitude should affect the score.
- Use AUCell when relative within-cell ranks are preferable to expression magnitude.
- Treat both outputs as activity scores, not p-values. Scores from different feature
  universes or AUCell `n_up` values are not directly interchangeable.

## Common mistakes and limitations

- Using identifiers that do not match the assay feature names
- Setting `tmin` above the number of targets that remain after feature matching
- Passing an HVG key to AUCell without intending to restrict its ranking universe
- Comparing WAGGR runs that use different normalization or log-transform settings
- Reusing a label for different inputs without `overwrite=True`
- Editing the count matrix outside Scarf after a result has been cached

Scarf persists each score matrix. Repeating an identical call reuses its
completed result; `overwrite=True` keeps the previous complete result available
until replacement finishes.
