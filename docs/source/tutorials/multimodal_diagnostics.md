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
```

The prepared store already contains RNA, ADT, WNN, and SNN results. The setup below reopens those
saved results. If you are continuing your own analysis, keep the references returned by its
analysis steps instead.

The searches specify which assay, method, and upstream graph each result belongs to.
Each search expects one match in this prepared store. If you have added more analyses, narrow
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

[snn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "snn"},
    complete_only=True,
)
[wnn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "wnn"},
    complete_only=True,
)

[snn_layout] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": snn_graph},
    complete_only=True,
)
[wnn_layout] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": wnn_graph},
    complete_only=True,
)
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
```

## 1. Check RNA and ADT concordance

### Question: where do the assay-specific partitions agree or disagree?

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

Broad agreement supports a shared population structure. Local differences are not automatically
errors: protein can resolve a population whose transcript is sparse. Large contradictory regions
should be investigated before integration.

## 2. Inspect WNN modality weights

Where does each assay contribute most strongly to the integrated graph?

```{code-cell} ipython3
ds.plots.modality_weights(graph=wnn_graph, layout=wnn_layout)
```

Spatial shifts show where RNA or ADT contributes more strongly. Check unexpected shifts
against markers and assay quality; noisy features or retained control antibodies can also
change the weights.

## Optional: compare WNN with SNN

SNN merges connectivity maps with equal standing. WNN consumes neighbour artifacts and learns a
per-cell contribution for each modality. Both preserve their exact source references; neither
becomes an implicit active graph.

### Question: does either integration preserve CD16 protein geography better?

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

The marker should remain localized rather than being spread across unrelated integrated groups.
Use several markers and known populations in a real study; one visually compact layout is not a
selection criterion.

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
    concordance.append(
        {
            "comparison": f"{first} vs {second}",
            "ARI": ds.metric_label_concordance(
                partitions[first], partitions[second], metric="ari"
            ),
            "NMI": ds.metric_label_concordance(
                partitions[first], partitions[second], metric="nmi"
            ),
        }
    )
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
