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

With sc-RNA seq only capturing a snapshot of a cell at a certain point in time, it can often be useful to determine what stage of the cell-cycle a cell is in based on that snapshot. With cell division happening in a cycle, growth in G1, replication of their DNA in S phase, preparation in G2 for the split during mitosis, we can estimate their stage based on the gene programs associated with the states. Each stage switches on a characteristic gene program, so measuring S-phase and G2M-phase program activity reveals which cells are cycling: information that matters twice over, because cycling cells can cluster together regardless of cell type (a confounder to check) and because proliferation itself is often the biology of interest.

SCARF infers the cell cycle by scoring each program by averaging its marker genes and subtracting matched control genes sampled from the same expression range, reducing the influence of background expression without eliminating technical effects or dropout. Built-in human and mouse S/G2M gene lists come inbuilt. Each cell is then assigned one phase: G1 when both scores are negative, otherwise whichever program scores higher.

Here, we score the prepared pancreas store, and then map phases and scores onto its UMAP.

## Open the pre-analyzed store

To begin, we take the downloaded store [Bastidas-Ponce et al., 2019 Development](https://journals.biologists.com/dev/article/146/12/dev173849/19483/) for E15.5 stage of differentiation of endocrine cells from a pool of endocrine progenitors-precursors. We use its selected cells and UMAP so we can focus on cell-cycle scoring.

```{code-cell}
# Import plotting, array, and table tools.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import Scarf to open the prepared store and score its cells.
import scarf

# Keep warnings visible while hiding routine progress messages.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared pancreas store and its saved analysis.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    name="bastidas-ponce_4K_pancreas-d15_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the store so scoring can save its results.
ds = scarf.DataStore(f"{dataset}/data.zarr", nthreads=4)
# Reuse the saved analysis and its selected cells.
analysis_run = ds.pipeline.open(label="docs_default")
# Inspect the opened assays and their dimensions.
ds
```

```{code-cell}
# Locate the saved pancreas clusters before scoring.
ds.plots.embedding(run=analysis_run, color_by="clusters")
```

## Run cell-cycle scoring

Scarf's scorer follows the same general strategy as [Scanpy's cell-cycle scorer](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes_cell_cycle.html):

- Match the supplied S and G2M markers, using Scarf's human-and-mouse lists by default.
- Group genes into bins with similar mean log-normalized expression across the selected cells.
- Sample control genes from the same expression bins as each phase's markers.
- Subtract mean control expression from mean marker expression for each cell and phase.

As stated earlier, in SCARF, cells with two negative scores are assigned G1; Otherwise, the G2M phase is assigned when its score exceeds the S score, and the remaining cells are assigned S.

```{code-cell}
# Score S and G2M programs in the saved analysis cells.
cell_cycle_ref = ds.run_cell_cycle_scoring(analysis_run["analysis_cell_selection"])
# Open the saved scores and phase assignments.
cell_cycle_values = ds.load_artifact(cell_cycle_ref)
# Read each selected cell's S-phase score.
s_score = np.asarray(cell_cycle_values["s_score"][:])
# Read each selected cell's G2M-phase score.
g2m_score = np.asarray(cell_cycle_values["g2m_score"][:])
# Read phase labels in the same cell order as the scores.
phase = np.asarray(cell_cycle_values["phase"][:]).astype(str)
# Preview the two scores alongside each cell's assigned phase.
pd.DataFrame({"S score": s_score, "G2M score": g2m_score, "phase": phase}).head()
```

Two markers in the bundled G2M list are absent from this assay; The warning is expected, and SCARF scores the cells with the remaining markers.

## Visualize cell-cycle phases

Pass the result directly to the UMAP plot to color cells by what phase they have been assigned

```{code-cell}
# Color the saved UMAP by each cell's assigned phase.
ds.plots.embedding(layout=analysis_run["umap"], color_by=cell_cycle_ref)
```

Look for populations enriched for S or G2M. Cycling cells may be concentrated in one population
or spread across several, depending on the tissue and experimental conditions.

We can also inspect the cell-cycle phase fractions within each cluster to see which groups are enriched for S or G2M relative to G1:

```{code-cell}
# Compare phase fractions within each saved cluster.
pd.crosstab(analysis_run.cells.fetch("clusters"), phase, normalize="index")
```

We can also choose to visualize cells enrichment for phases by simply plotting their S or G2M score as the color scale upon the UMAP

```{code-cell}
# Read UMAP coordinates in the same cell order as the scores.
umap = analysis_run.cells.to_pandas_dataframe(["umap_1", "umap_2"])
# Create one panel for each cell-cycle score.
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
# Plot S and G2M scores on their respective panels.
for axis, values, title in (
    (axes[0], s_score, "S score"),
    (axes[1], g2m_score, "G2M score"),
):
    # Color each cell by this panel's phase score.
    points = axis.scatter(umap["umap_1"], umap["umap_2"], c=values, s=3)
    # Identify the phase score shown in the panel.
    axis.set_title(title)
    # Show the score scale used to color the cells.
    figure.colorbar(points, ax=axis)
# Make room for panel titles and color scales.
figure.tight_layout()
# Display the completed score comparison.
figure
```

## Important caveats to consider regarding cell cycle

- **Incompatible gene identifiers and missing markers:** Scarf's bundled S and G2M signatures rely on standard human and mouse gene symbols and match names without case sensitivity. Supplying datasets with incompatible identifiers (e.g., Ensembl IDs or unmapped orthologs) leaves markers unmatched. Scarf warns about missing markers and raises an error if either phase has no matching markers. Partial matches can still distort the final scores, so inspect the warnings and gene identifiers.
- **Conflating negative scores with active G1 (the G0 vs. G1 blindspot):** Cells are assigned to G1 by default whenever both S and G2M scores are negative. This heuristic cannot distinguish actively cycling G1 cells from quiescent (G0), senescent, or post-mitotic differentiated states, and severe technical dropout can artificially depress scores into negative values.
- **Treating relative scores as definitive proof of proliferation:** Cell-cycle scores measure the relative enrichment of phase-associated transcripts against expression-matched background bins, not absolute mitotic rates. Without confirming key driver genes (e.g., MKI67, TOP2A, PCNA) and checking whether cell-cycle signatures are confounding unsupervised clustering, stress programs or lineage-specific transcripts can easily be misinterpreted as active proliferation.
