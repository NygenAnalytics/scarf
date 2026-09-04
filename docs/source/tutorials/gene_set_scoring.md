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

(gene_set_scoring)=

# Gene-set activity scoring

Score pathway or cell-state signatures per cell with WAGGR or AUCell.
Both methods stream RNA counts from the Zarr store and persist activity scores for later use.
These methods do not calculate enrichment p-values.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with feature names that match the identifiers in your gene sets
- A basic understanding of cell metadata and embeddings

## What you will learn

- Read gene sets from GMT and inspect feature overlap
- Score weighted signatures with WAGGR
- Score rank-based signatures with AUCell
- Load selected score sources without materializing the full result

## Dataset

The rebuilt 5K PBMC store contains a completed standard analysis labeled `docs_default`.
Open the downloaded store directly because scoring writes new immutable artifacts. The frozen run
provides the exact analysis cells and UMAP used below.
Signature scoring streams raw counts from `assay.rawData`, not a pre-normalized matrix or the graph.
AUCell ranks those raw counts; WAGGR applies library-size normalization inside the scorer.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd

import scarf

scarf.configure_output(level='WARNING', progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
    zarr=True,
)
ds = scarf.DataStore(f'{dataset}/data.zarr', nthreads=4)
run = ds.pipeline.open(label='docs_default')
```

## 1. Read and inspect gene sets

GMT stores one source per line.
The first field is the source name, the second is a description, and the remaining fields are target genes.
`read_gmt` returns one source-target row per gene.

```{code-cell} ipython3
input_directory = TemporaryDirectory()
gmt_path = Path(input_directory.name) / 'pbmc_signatures.gmt'
gmt_path.write_text(
    'T_cell\tna\tCD3D\tCD3E\tTRAC\tLTB\tIL7R\n'
    'B_cell\tna\tMS4A1\tCD79A\tCD37\tCD74\tHLA-DRA\n'
    'Myeloid\tna\tLST1\tS100A8\tS100A9\tCTSS\tFCER1G\n',
    encoding='utf-8',
)
gene_sets = scarf.read_gmt(gmt_path)
gene_sets
```

Targets are matched to active RNA feature names without case sensitivity.
`tmin` is applied after matching, so a source is retained only when enough of its targets are present.
Missing targets do not need to be removed from the input table first.

```{code-cell} ipython3
available = {str(name).upper() for name in ds.RNA.feats.fetch_all('names')}
(
    gene_sets.assign(
        matched=gene_sets['target'].str.upper().isin(available),
    )
    .groupby('source')['matched']
    .agg(['sum', 'count'])
)
```

## 2. Score weighted signatures with WAGGR

WAGGR applies edge weights to library-size-normalized expression.
`wmean` divides each weighted sum by the sum of absolute weights, while `wsum` leaves the weighted sum unscaled.
Signed weights are supported.

This comparison uses the complete assay feature universe for both methods:

```{code-cell} ipython3
cell_selection = run['analysis_cell_selection']
all_features = ds.select_all_features(from_assay='RNA')
```

```{code-cell} ipython3
weighted_sets = gene_sets.assign(weight=1.0)
weighted_sets.loc[
    weighted_sets['target'].isin(['S100A8', 'S100A9']),
    'weight',
] = 1.5
weighted_sets
```

S100A8 and S100A9 carry weight 1.5; every other edge stays at 1.0.

```{code-cell} ipython3
waggr = ds.run_waggr(
    weighted_sets,
    cell_selection,
    features=all_features,
    mode='wmean',
    tmin=3,
)
score_sources = ['T_cell', 'B_cell', 'Myeloid']
waggr_result = ds.get_enrichment(waggr, sources=score_sources)
waggr_scores = pd.DataFrame(
    waggr_result.data.compute(),
    columns=list(waggr_result.source_names),
)
waggr_scores.describe().loc[['min', '50%', 'max']]
```

Each column is one source.
The ranges show that WAGGR tracks expression magnitude and is not confined to values between zero and one.
The loaded `EnrichmentResult.feature_selection` records the exact normalization universe.

To see what the raised Myeloid weights change, run the same network with every weight at 1.0 and compare Myeloid summaries:

```{code-cell} ipython3
waggr_unweighted = ds.run_waggr(
    gene_sets.assign(weight=1.0),
    cell_selection,
    features=all_features,
    mode='wmean',
    tmin=3,
)
waggr_unweighted_result = ds.get_enrichment(waggr_unweighted)
unweighted_scores = pd.DataFrame(
    waggr_unweighted_result.data.compute(),
    columns=list(waggr_unweighted_result.source_names),
)
pd.DataFrame(
    {
        'weighted': waggr_scores['Myeloid'],
        'unweighted': unweighted_scores['Myeloid'],
    }
).describe().loc[['min', '50%', 'max']]
```

Compare the Myeloid rows: any shift is the effect of raising S100A8 and S100A9.
T_cell and B_cell edges were left at 1.0 in both runs.

WAGGR uses Scarf's default RNA library-size normalization.
Set `log_transform=True` to apply `log1p` before aggregation.

## 3. Score rank recovery with AUCell

AUCell ranks the selected RNA features within each cell and measures how early a source's targets are recovered.
Scores range from zero to one.
Network weights are ignored.

The required `features` argument defines the ranking universe.
Here the `all_features` artifact ranks the complete RNA feature order.
`n_up=500` evaluates recovery within the top 500 ranks.

```{code-cell} ipython3
aucell = ds.run_aucell(
    gene_sets,
    cell_selection,
    features=all_features,
    tmin=3,
    n_up=500,
    tie_seed=0,
)
aucell_result = ds.get_enrichment(aucell, sources=score_sources)
aucell_scores = pd.DataFrame(
    aucell_result.data.compute(),
    columns=list(aucell_result.source_names),
)
aucell_scores.describe().loc[['min', '50%', 'max']]
```

AUCell values stay between zero and one.
The same `tie_seed` gives a deterministic global ordering for equal expression values.
Changing `n_up`, `tie_seed`, the feature selection, or the network creates a different execution.

## 4. Visualize the selected sources

`get_enrichment` requires an exact enrichment ref and returns a lazy result.
The calls above selected only the requested source columns before computing them. Reuse those
loaded tables for every plot and comparison below. The values are activity scores, not p-values.

```{code-cell} ipython3
umap = run.cells.to_pandas_dataframe(['umap_1', 'umap_2'])
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, source in zip(axes, score_sources, strict=True):
    axis.scatter(
        umap['umap_1'],
        umap['umap_2'],
        c=aucell_scores[source],
        s=3,
    )
    axis.set_title(f'{source} AUCell')
figure.tight_layout()
figure
```

AUCell scores highlight lineage-consistent regions: T-cell, B-cell, and Myeloid scores peak in separate parts of the UMAP when those populations are present.

```{code-cell} ipython3
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, source in zip(axes, score_sources, strict=True):
    axis.scatter(
        umap['umap_1'],
        umap['umap_2'],
        c=waggr_scores[source],
        s=3,
    )
    axis.set_title(f'{source} WAGGR')
figure.tight_layout()
figure
```

WAGGR marks the same lineage regions, but the color scale follows expression magnitude rather than rank recovery.

WAGGR and AUCell both mark myeloid-like cells here, but the score scales differ because one aggregates weighted expression and the other measures within-cell rank recovery.
Quantify that difference cell by cell:

```{code-cell} ipython3
myeloid_compare = pd.DataFrame(
    {
        'Myeloid_WAGGR': waggr_scores['Myeloid'],
        'Myeloid_AUCell': aucell_scores['Myeloid'],
    }
)
myeloid_compare.describe()
```

```{code-cell} ipython3
figure, axis = plt.subplots(figsize=(4, 4))
axis.scatter(
    myeloid_compare['Myeloid_WAGGR'],
    myeloid_compare['Myeloid_AUCell'],
    s=4,
    alpha=0.35,
)
axis.set_xlabel('Myeloid WAGGR')
axis.set_ylabel('Myeloid AUCell')
plt.show()
```

Cells that rank high for Myeloid under AUCell also tend to score high under WAGGR, while the absolute values stay on different scales.

## Choosing a method

- Use WAGGR when edge weights or signed targets carry useful information and expression magnitude should affect the score.
- Use AUCell when relative within-cell ranks are preferable to expression magnitude.
- Treat both outputs as activity scores, not p-values.
  Scores from different feature universes or AUCell `n_up` values are not directly interchangeable.

## Common mistakes and limitations

- Using identifiers that do not match the assay feature names
- Setting `tmin` above the number of targets that remain after feature matching
- Passing an HVG selection to AUCell without intending to restrict its ranking universe
- Comparing WAGGR runs that use different normalization or log-transform settings
- Editing the count matrix outside Scarf after a result has been cached

Scarf persists each score matrix.
Repeating an identical call reuses its completed result. `invalidate_cache=True` creates another
immutable result without replacing the earlier one.
