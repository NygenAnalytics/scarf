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

# Gene-set activity scoring primer

Single cell readouts on singular genes can be sparse and heavily noisy. This is often driven by dropout, which can zero out a marker in cells that express it, and then one highly expressed gene can dominate expression. Gene-set scoring fixes this by collapsing a whole program; a pathway, a cell state, a lineage signature; into one number per cell that is more robust than any member gene alone.

Scarf offers two ways to compute that number. WAGGR (weight aggregate) takes a weighted mean of normalized gene expression, so strongly expressed genes can pull harder and can push in opposite directions. AUCell ignores expression values entirely after ranking each cell's genes, then measures how early the signature's targets appear among the top ranks, with its scores falling betweem 0-1. WAGGR can be used to gain an idea of the magnitude of a program, whereas AUcell can tell you about rank recovery. Rank recovery can be conceptualized as how early the signature of interest appears among that cell's top-ranked genes.

## Load the prepared dataset

We first begin by loading in the prepared dataset. For context, when we do our activity scoring, the data is streamed directly from the raw counts: AUCell ranks those raw counts, while WAGGR applies library-size normalization for its own scoring function.

```{code-cell}
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

## Read and inspect gene sets

For reading and inspecting your gene sets, you must use **your own Gene Matrix Transposed (GMT) files.** When reading the file, the matrix stores one gene program per line. The first column is the name, the second is a description, and the remaining are target genes.


```{code-cell}
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

Targets are matched to active RNA feature names without case sensitivity, meaning that any match regardless of captilization will be selected. Missing targets do not need to be removed from the input table, as SCARF automatically handles this. A target that matches several active features, such as a gene symbol shared by two feature ids, is ambiguous (ignored)

```{code-cell}
available = {str(name).upper() for name in ds.RNA.feats.fetch_all('names')}
(
    gene_sets.assign(
        matched=gene_sets['target'].str.upper().isin(available),
    )
    .groupby('source')['matched']
    .agg(['sum', 'count'])
)
```

## Start with equal-weight scores

WAGGR calculates a weighted mean of library-size-normalized expression by default.
When the input has no weight column, every target gene has weight one.
We use `tmin=3` for these short teaching signatures so that a signature can still be scored if
one or two of its five genes are missing. The default requires five matched targets.

This comparison uses the complete assay feature universe for both methods:

```{code-cell}
cell_selection = run['analysis_cell_selection']
all_features = ds.select_all_features(from_assay='RNA')
```

```{code-cell}
waggr = ds.run_waggr(
    gene_sets,
    cell_selection,
    features=all_features,
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

## 3. Score rank recovery with AUCell

AUCell ranks the selected RNA features within each cell and measures how early a source's targets are recovered.
Scores range from zero to one.
Network weights are ignored.

The required `features` argument defines the ranking universe.
Here the `all_features` artifact ranks the complete RNA feature order.
By default, AUCell evaluates the top 5% of that ranking universe.

```{code-cell}
aucell = ds.run_aucell(
    gene_sets,
    cell_selection,
    features=all_features,
    tmin=3,
)
aucell_result = ds.get_enrichment(aucell, sources=score_sources)
aucell_scores = pd.DataFrame(
    aucell_result.data.compute(),
    columns=list(aucell_result.source_names),
)
aucell_scores.describe().loc[['min', '50%', 'max']]
```

AUCell values stay between zero and one. The default seed keeps the ordering of tied expression
values reproducible. Use `n_up` only when you want a different rank window, and keep it fixed when
comparing scores from the same feature universe.

## 4. Visualize the selected sources

The tables above contain only the three requested signatures. Reuse them for plotting rather than
loading the scores again. These values describe gene-set activity; they are not p-values.

```{code-cell}
umap = run.cells.to_pandas_dataframe(['umap_1', 'umap_2'])
figure, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, source in zip(axes, score_sources, strict=True):
    points = axis.scatter(
        umap['umap_1'],
        umap['umap_2'],
        c=aucell_scores[source],
        s=3,
        vmin=0,
        vmax=1,
    )
    axis.set_title(f'{source} AUCell')
    figure.colorbar(points, ax=axis, label='AUCell score')
figure.tight_layout()
plt.show()
```

AUCell scores highlight lineage-consistent regions: T-cell, B-cell, and Myeloid scores peak in separate parts of the UMAP when those populations are present.

Compare the two methods for the Myeloid signature. WAGGR follows expression magnitude, while
AUCell measures recovery among the highest-ranked genes, so their numerical scales differ.

```{code-cell}
myeloid_compare = pd.DataFrame(
    {
        'Myeloid_WAGGR': waggr_scores['Myeloid'],
        'Myeloid_AUCell': aucell_scores['Myeloid'],
    }
)
myeloid_compare.describe()
```

```{code-cell}
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

Look for cells with high scores under both methods and cells where the methods disagree.
A change in score scale alone is not a biological difference.

## Optional: use a signature with weights

Use weights when the source of a signature provides a reason for particular genes to contribute
more, less, or in opposite directions. Add a `weight` column to the input table; without it, all
weights are one. WAGGR's default `mode="wmean"` divides the weighted sum by the sum of absolute
weights. `mode="wsum"` leaves it unscaled. AUCell ignores these weights.

WAGGR also accepts `log_transform=True` to apply `log1p` before aggregation. Changing weights or
the expression transform changes the meaning of the score, so choose them before comparing cells
or conditions.

## Common mistakes and limitations

- Using identifiers that do not match the assay feature names
- Setting `tmin` above the number of targets that remain after feature matching
- Passing an HVG selection to AUCell without intending to restrict its ranking universe
- Comparing WAGGR runs that use different normalization or log-transform settings
- Editing the count matrix outside Scarf after a result has been cached

Scarf persists each score matrix.
Repeating an identical call reuses its completed result. `invalidate_cache=True` creates another
immutable result without replacing the earlier one.
