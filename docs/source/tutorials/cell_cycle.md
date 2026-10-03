---
description: Score S and G2M gene programs and assign cell-cycle phases.
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
(cell_cycle)=

# Cell cycle primer

With sc-RNA seq only capturing a snapshot of a cell at a certain point in time, it can often be useful to determine what sage of the cell-cycle a cell is based on that snapshot. With cell division happening in a cycle, growth in G1, replication of their DNA in S phase, preparation in G2 for the split during mitosis, we can estimate their stage based on the gene programs associated with the states. Each stage switches on a characteristic gene program, so measuring S-phase and G2M-phase program activity reveals which cells are cycling: information that matters twice over, because cycling cells can cluster together regardless of cell type (a confounder to check) and because proliferation itself is often the biology of interest.

SCARF infers the cell cycle by scoring each program by averaging its marker genes and subtracting matched control genes sampled from the same expression range, so technical level and dropout do not inflate the score. Built-in human and mouse S/G2M gene lists come inbuilt. Each cell is then assigned one phase: G1 when both scores are negative, otherwise whichever program scores higher.

Here, we score the prepared pancreas store, and then map phases and scores onto its UMAP.

## Open the pre-analyzed store

To begin, we take the downloaded store [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors. We use its selected cells and UMAP so we can focus on cell-cycle scoring.

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    nthreads=4,
)
analysis_run = ds.pipeline.open(label="docs_default")
```

```{code-cell}
ds.plots.embedding(
    run=analysis_run,
    color_by="clusters",
)
```

## Run cell-cycle scoring

Scarf's scorer follows the same general strategy as [Scanpy&#39;s cell-cycle scorer](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes_cell_cycle.html) which:

- Match the supplied S and G2M markers, using Scarf's human-and-mouse lists by default.
- Groups genes into bins with similar mean log-normalized expression across the selected cells.
- Sample control genes from the same expression bins as each phase's markers.
- Subtract mean control expression from mean marker expression for each cell and phase.

As stated earlier, in SCARF, cells with two negative scores are assigned G1; Otherwise, the G2M phase is assigned when its score exceeds the S score, and the remaining cells are assigned S.

```{code-cell}
cell_cycle_ref = ds.run_cell_cycle_scoring(analysis_run["analysis_cell_selection"])
cell_cycle_values = ds.load_artifact(cell_cycle_ref)
s_score = np.asarray(cell_cycle_values["s_score"][:])
g2m_score = np.asarray(cell_cycle_values["g2m_score"][:])
phase = np.asarray(cell_cycle_values["phase"][:]).astype(str)
```

Two markers in the bundled G2M list are absent from this assay; The warning is expected, and SCARF scores the cells with the remaining markers.

## Visualize cell-cycle phases

Pass the result directly to the UMAP plot to color cells by what phase they have been assigned

```{code-cell}
ds.plots.embedding(
    layout=analysis_run["umap"],
    color_by=cell_cycle_ref,
)
```

Look for populations enriched for S or G2M. Cycling cells may be concentrated in one population
or spread across several, depending on the tissue and experimental conditions.

We can also the see the cell cycle phase composition for the datasets completed to see what groups are enriched for what for S or G2M relative to G1:

```{code-cell}
pd.crosstab(analysis_run.cells.fetch("clusters"), phase, normalize="index")
```

We can also choose to visualize cells enrichment for phases by simply plotting their S or G2M score as the color scale upon the UMAP

```{code-cell}
umap = analysis_run.cells.to_pandas_dataframe(["umap_1", "umap_2"])
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
for axis, values, title in (
    (axes[0], s_score, "S score"),
    (axes[1], g2m_score, "G2M score"),
):
    points = axis.scatter(umap["umap_1"], umap["umap_2"], c=values, s=3)
    axis.set_title(title)
    figure.colorbar(points, ax=axis)
figure.tight_layout()
figure
```

## Important caveats to consider regarding cell cycle

- **Incompatible gene identifiers and silent marker dropout:** Scarf's bundled S and G2M signatures rely on standard human and mouse gene nomenclature. Supplying datasets with non-conforming identifiers (e.g., Ensembl IDs, discordant case-sensitivity, or unmapped orthologs) causes the gene markers to drop silently, compromising expression binning and distorting background control subtraction, misrepresenting the final scores.
- **Conflating negative scores with active G1 (the G0 vs. G1 blindspot):** Cells are assigned to G1 by default whenever both S and G2M scores are negative. This heuristic cannot distinguish actively cycling G1 cells from quiescent (G0), senescent, or post-mitotic differentiated states, and severe technical dropout can artificially depress scores into negative values.
- **Treating relative scores as definitive proof of proliferation:** Cell-cycle scores measure the relative enrichment of phase-associated transcripts against expression-matched background bins, not absolute mitotic rates. Without confirming key driver genes (e.g., MKI67, TOP2A, PCNA) and checking whether cell-cycle signatures are confounding unsupervised clustering, stress programs or lineage-specific transcripts can easily be misinterpreted as active proliferation.
