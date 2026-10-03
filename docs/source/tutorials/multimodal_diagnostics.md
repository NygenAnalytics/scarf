---
description: Diagnose modality agreement and compare SNN with WNN integration.
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

# Diagnose multimodal integration

After the core {doc}`cite_seq` workflow, check whether RNA and ADT support similar populations.
Then inspect where each assay contributes to the default WNN integration. An optional comparison
with SNN shows what changes when the two assay graphs have equal standing.

## Open the matched results

```{code-cell} ipython3
# Enumerate each pair of clusterings once.
from itertools import combinations

# Arrange and save Matplotlib figures.
import matplotlib.pyplot as plt
# Summarize cells and results in tables.
import pandas as pd

# Open count stores and run Scarf analyses.
import scarf
# Select explicit fields and display options for plots.
from scarf.plotting import FeatureRef

# Keep routine logs and progress bars out of the results.
scarf.configure_output(level="WARNING", progress=False)

# Download the prepared example store.
dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_8K_pbmc_citeseq",
    destination="scarf_datasets",
    zarr=True,
)
# Open the count store for this analysis.
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
# Open the saved RNA analysis.
rna_run = ds.pipeline.open(label="docs_default")
# Inspect the opened store's cells and features.
ds
```

The prepared store already contains RNA, ADT, WNN, and SNN results. Start by reopening the
ADT results to compare them with RNA. If you are continuing your own analysis, keep the
references returned by its analysis steps instead.

The searches specify which assay, method, and upstream graph each result belongs to.
Each search expects one match in this prepared store. If you have added more analyses, narrow
the search to the result you intend to compare.

```{code-cell} ipython3
# Select the saved ADT layout; require exactly one match.
[adt_layout] = ds.list_artifacts(
    from_assay="ADT",
    kind="embedding",
    operation="run_umap",
    complete_only=True,
)
# Select the saved ADT clusters; require exactly one match.
[adt_clusters] = ds.list_artifacts(
    from_assay="ADT",
    kind="cluster_labels",
    operation="run_leiden_clustering",
    complete_only=True,
)

# Inspect the two ADT results selected for comparison with RNA.
{"ADT layout": adt_layout, "ADT clusters": adt_clusters}
```

## 1. Check RNA and ADT concordance

### Question: where do the assay-specific partitions agree or disagree?

```{code-cell} ipython3
# Create axes for the comparison panels.
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
# Draw each result on its comparison axes.
for axis, layout, labels, title in (
    (axes[0], rna_run["umap"], adt_clusters, "RNA layout, ADT clusters"),
    (axes[1], adt_layout, rna_run["clusters"], "ADT layout, RNA clusters"),
):
    # Color each assay's layout by clusters from the other assay.
    ds.plots.embedding(
        layout=layout,
        color_by=labels,
        legend_loc="on_data",
        show_titles=False,
        target=axis,
        show=False,
    )
    # Label the panel with the result it shows.
    axis.set_title(title)
# Adjust spacing between the comparison panels.
figure.tight_layout()
```

Broad agreement supports a shared population structure. Local differences are not automatically
errors: protein can resolve a population whose transcript is sparse. Large contradictory regions
should be investigated before integration.

## 2. Inspect WNN modality weights

Where does each assay contribute most strongly to the integrated graph? Open the WNN graph
and the layout made from it:

```{code-cell} ipython3
# Select the saved WNN graph; require exactly one match.
[wnn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "wnn"},
    complete_only=True,
)
# Select the saved WNN layout; require exactly one match.
[wnn_layout] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": wnn_graph},
    complete_only=True,
)
# Inspect the WNN layout associated with the selected graph.
wnn_layout
```

```{code-cell} ipython3
# Show how much each modality contributes across the joint map.
ds.plots.modality_weights(graph=wnn_graph, layout=wnn_layout)
```

Spatial shifts show where RNA or ADT contributes more strongly. Check unexpected shifts
against markers and assay quality; noisy features or retained control antibodies can also
change the weights.

## Optional: compare WNN with SNN

SNN merges connectivity maps with equal standing. WNN consumes neighbour artifacts and learns a
per-cell contribution for each modality. Both preserve their exact source references; neither
becomes an implicit active graph.

Open the SNN result only when making this comparison:

```{code-cell} ipython3
# Select the saved SNN graph; require exactly one match.
[snn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "snn"},
    complete_only=True,
)
# Select the saved SNN layout; require exactly one match.
[snn_layout] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": snn_graph},
    complete_only=True,
)
# Inspect the SNN layout associated with the selected graph.
snn_layout
```

Retrieve the cluster labels from each integrated graph:

```{code-cell} ipython3
# Select the saved SNN clusters; require exactly one match.
[snn_clusters] = ds.list_artifacts(
    scope="datastore",
    kind="cluster_labels",
    operation="run_leiden_clustering",
    inputs={"graph": snn_graph},
    complete_only=True,
)
# Select the saved WNN clusters; require exactly one match.
[wnn_clusters] = ds.list_artifacts(
    scope="datastore",
    kind="cluster_labels",
    operation="run_leiden_clustering",
    inputs={"graph": wnn_graph},
    complete_only=True,
)
# Inspect the two clusterings selected for comparison.
{"SNN clusters": snn_clusters, "WNN clusters": wnn_clusters}
```

### Question: does either integration preserve CD16 protein geography better?

```{code-cell} ipython3
# Select the CD16 protein measurement by its feature identifier.
cd16 = FeatureRef("CD16", assay="ADT", by="id", label="CD16")
# Create axes for the comparison panels.
figure, axes = plt.subplots(2, 2, figsize=(9, 8))
# Draw each result on its comparison axes.
for row, layout, labels, method in (
    (0, wnn_layout, wnn_clusters, "WNN"),
    (1, snn_layout, snn_clusters, "SNN"),
):
    # Show the clusters on this integration's own layout.
    ds.plots.embedding(
        layout=layout,
        color_by=labels,
        legend_loc="on_data",
        show_titles=False,
        target=axes[row, 0],
        show=False,
    )
    # Label the panel with the result it shows.
    axes[row, 0].set_title(f"{method} clusters")
    # Show CD16 protein on the same integration layout.
    ds.plots.embedding(
        layout=layout,
        color_by=cd16,
        sort_values=True,
        show_titles=False,
        target=axes[row, 1],
        show=False,
    )
    # Label the panel with the result it shows.
    axes[row, 1].set_title(f"{method}: CD16 protein")
# Adjust spacing between the comparison panels.
figure.tight_layout()
```

The marker should remain localized rather than being spread across unrelated integrated groups.
Use several markers and known populations in a real study; one visually compact layout is not a
selection criterion.

Partition concordance quantifies similarity without declaring a winner:

```{code-cell} ipython3
# Collect the four clusterings to compare.
partitions = {
    "RNA": rna_run["clusters"],
    "ADT": adt_clusters,
    "SNN": snn_clusters,
    "WNN": wnn_clusters,
}
# Collect one agreement summary per pair of clusterings.
concordance = []
# Compare each pair of clusterings once.
for first, second in combinations(partitions, 2):
    # Measure agreement with the adjusted Rand index.
    ari = ds.metric_label_concordance(partitions[first], partitions[second], metric="ari")
    # Measure agreement with normalized mutual information.
    nmi = ds.metric_label_concordance(partitions[first], partitions[second], metric="nmi")
    # Keep both agreement measures with the comparison name.
    concordance.append({"comparison": f"{first} vs {second}", "ARI": ari, "NMI": nmi})
# Display agreement scores for each pair of clusterings.
pd.DataFrame(concordance)
```

ARI and NMI describe agreement. Interpret them beside marker coherence and assay design rather than
maximizing them mechanically.

## Decision guide

- Prefer WNN when the relative local informativeness of matched modalities varies across cells.
- Use SNN when equal graph support is the scientific comparison you intend.
- Reject either result if marker geography, known populations, cell alignment, or graph quality is
  inconsistent.
- Use {doc}`../reference/api/integration` for algorithm and input contracts, and
  {doc}`reuse_and_tracing` for full lineage inspection.
