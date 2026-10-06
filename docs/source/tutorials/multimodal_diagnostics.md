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
# Compare multimodal integration methods

After the core {doc}`cite_seq` workflow, it may be useful to compare and determine whether the RNA and ADT data support similar populations through formal checkpoints.
Then inspect where each assay contributes to the default WNN integration. An optional comparison
with SNN shows what changes when the two assay graphs have equal standing.

## Open the matched results

```{code-cell} ipython3
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

import scarf
from scarf.plotting import FeatureRef

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_8K_pbmc_citeseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="RNA",
    nthreads=4,
)
rna_run = ds.pipeline.open(label="docs_default")
ds
```

The prepared store already contains RNA, ADT, WNN, and SNN results. Start by reopening the
ADT results to compare them with RNA. If you are continuing your own analysis, keep the
references prepared instead of rerunning then,

The searches specify which assay, method, and upstream graph each result belongs to.
Each search expects one match in this prepared store; if you have added more analyses, narrow
the search to the result you intend to compare.

```{code-cell} ipython3
[adt_layout] = ds.list_artifacts(
    from_assay="ADT",
    kind="embedding",
    operation="run_umap",
    complete_only=True,
)
[adt_clusters] = ds.list_artifacts(
    from_assay="ADT",
    kind="cluster_labels",
    operation="run_leiden_clustering",
    complete_only=True,
)

{"ADT layout": adt_layout, "ADT clusters": adt_clusters}
```

## Check RNA and ADT concordance

## Where do the assay-specific partitions agree or disagree?

```{code-cell} ipython3
figure, axes = plt.subplots(1, 2, figsize=(9, 4))
for axis, layout, labels, title in (
    (axes[0], rna_run["umap"], adt_clusters, "RNA layout, ADT clusters"),
    (axes[1], adt_layout, rna_run["clusters"], "ADT layout, RNA clusters"),
):
    ds.plots.embedding(
        layout=layout,
        color_by=labels,
        legend_loc="on_data",
        show_titles=False,
        target=axis,
        show=False,
    )
    axis.set_title(title)
figure.tight_layout()
```

Broad agreement supports a shared population structure, and local differences are not automatically true error, as the correlation between transcriptomic expression and protein presence for a gene may not be the strongest; furthermore, some transcripts may simply be sparse. 

## Inspect WNN modality weights

Where does each assay contribute most strongly to the integrated graph of both forms of data? To figure this out open the WNN graph and the layout made from it:

```{code-cell} ipython3
[wnn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "wnn"},
    complete_only=True,
)
[wnn_layout] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": wnn_graph},
    complete_only=True,
)
wnn_layout
```

```{code-cell} ipython3
ds.plots.modality_weights(graph=wnn_graph, layout=wnn_layout)
```

Spatial shifts show where RNA or ADT contributes more strongly. You should check unexpected shifts
against markers and assay quality; noisy features or retained control antibodies can also
change the weights.

## Compare WNN with SNN

SNN merges connectivity maps with equal contributions from the data, whereas WNN consumes information from both neighbor graphs and learns a per-cell contribution for each modality. Both preserve their exact source references; neither becomes an implicit active graph.

Open the SNN result only when making this comparison:

```{code-cell} ipython3
[snn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "snn"},
    complete_only=True,
)
[snn_layout] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": snn_graph},
    complete_only=True,
)
snn_layout
```

Retrieve the cluster labels from each integrated graph:

```{code-cell} ipython3
[snn_clusters] = ds.list_artifacts(
    scope="datastore",
    kind="cluster_labels",
    operation="run_leiden_clustering",
    inputs={"graph": snn_graph},
    complete_only=True,
)
[wnn_clusters] = ds.list_artifacts(
    scope="datastore",
    kind="cluster_labels",
    operation="run_leiden_clustering",
    inputs={"graph": wnn_graph},
    complete_only=True,
)
{"SNN clusters": snn_clusters, "WNN clusters": wnn_clusters}
```

### Which integration method preserves CD16 protein geography better?

```{code-cell} ipython3
cd16 = FeatureRef("CD16", assay="ADT", by="id", label="CD16")
figure, axes = plt.subplots(2, 2, figsize=(9, 8))
for row, layout, labels, method in (
    (0, wnn_layout, wnn_clusters, "WNN"),
    (1, snn_layout, snn_clusters, "SNN"),
):
    ds.plots.embedding(
        layout=layout,
        color_by=labels,
        legend_loc="on_data",
        show_titles=False,
        target=axes[row, 0],
        show=False,
    )
    axes[row, 0].set_title(f"{method} clusters")
    ds.plots.embedding(
        layout=layout,
        color_by=cd16,
        sort_values=True,
        show_titles=False,
        target=axes[row, 1],
        show=False,
    )
    axes[row, 1].set_title(f"{method}: CD16 protein")
figure.tight_layout()
```

The marker should remain localized rather than being spread across unrelated integrated groups. Use several markers and known populations in a real studies, one marker will never be enough; Furthermore, one visually compact layout is not a selection criterion.

Partition concordance quantifies similarity without declaring a winner:

```{code-cell} ipython3
partitions = {
    "RNA": rna_run["clusters"],
    "ADT": adt_clusters,
    "SNN": snn_clusters,
    "WNN": wnn_clusters,
}
concordance = []
for first, second in combinations(partitions, 2):
    ari = ds.metric_label_concordance(partitions[first], partitions[second], metric="ari")
    nmi = ds.metric_label_concordance(partitions[first], partitions[second], metric="nmi")
    concordance.append({"comparison": f"{first} vs {second}", "ARI": ari, "NMI": nmi})
pd.DataFrame(concordance)
```

The Adjusted Rand Index (ARI) counts the cell pairs that both partitions group together, corrected for what random labels would share by chance. Normalized Mutual Information (NMI) asks how much knowing one partition's labels reduces uncertainty about the other, and it is more forgiving when the two partitions use different numbers of clusters. Both score near 1 when the partitions group cells the same way and near 0 when they share no more than chance, so they measure similarity of groupings, not correctness. Interpret them beside marker coherence and assay design rather than maximizing them mechanically through manipulatin the data

## Decision 

### Choosing WNN versus SNN

- **WNN (adaptive weighting), the default:** Use it when information content is asymmetric
  across cell states, for example when ADT cleanly splits T-cell subsets while RNA resolves
  rare cell types that the antibody panel does not cover. It learns cell-specific weights from
  local prediction accuracy, which keeps a sparse or noisy modality from diluting
  high-confidence topology.
- **SNN (equal standing), the diagnostic baseline:** Use it when both assays have balanced
  feature depth and signal-to-noise ratios, or when you explicitly want an unweighted baseline.
  It tests whether fine WNN structures reflect genuine shared topology or come from extreme
  weight skew toward one modality.

### Validation checklist

- **Modality weights (`ds.plots.modality_weights`):** Expect local, biologically plausible
  shifts, such as high ADT weight in lymphoid cells and high RNA weight in lineages the panel
  does not profile. As a rule of thumb, one modality holding more than 90% of the weight
  across all cells flags severe technical dropout or failed normalization in the downweighted
  assay.
- **Marker localization:** Overlay canonical markers such as CD16, CD4, and CD19. Expression
  should remain tightly clustered. Smearing across unrelated populations indicates false
  nearest-neighbor bridging.
- **Partition concordance (ARI/NMI):** As a rule of thumb, moderate agreement (ARI 0.35 to
  0.65) is normal and reflects complementary biology, such as post-transcriptional differences.
  Very low agreement (ARI below 0.15) suggests uncorrected batch effects or cell-indexing
  mismatches.

### When to reject both integrated graphs

- **Ambient ADT background:** Uncorrected nonspecific antibody binding introduces spurious
  graph edges between unrelated cell types.
- **Uncorrected batch effects:** Modality-specific technical batch variation bleeds directly
  into the integrated connectivity map. Correct batch effects before building the integrated
  graph.
- **Artificial lineage collapse:** Mutually exclusive populations, such as B cells and T cells,
  fuse into a single cluster. This indicates graph neighbor parameters are overly permissive.

Use {doc}`../reference/api/integration` for correction for batch effects, and {doc}`reuse_and_tracing` for full inspection of the different results.
