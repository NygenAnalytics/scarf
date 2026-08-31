---
description: Interpret a prepared CITE-seq WNN map with protein-marker evidence.
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

(multimodal_integration)=
(wnn_integration)=

# Integrate RNA and protein with CITE-seq

Use weighted nearest neighbours (WNN) to interpret matched RNA and antibody-derived tag (ADT)
measurements from the same PBMCs. This is the one recommended CITE-seq path: reopen the prepared
WNN result, inspect its joint populations, and test them with protein markers.

SNN comparison, modality-specific alternatives, integration metrics, and modality weights belong
in {doc}`multimodal_diagnostics`.

## Open the prepared multimodal result

```{code-cell} ipython3
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
```

The store contains both assay-specific neighbour results and a previously computed WNN graph.
Focused provenance predicates reopen the WNN graph and only the UMAP and Leiden result produced
from that graph. Destructuring each result fails loudly if the prepared store is missing a result
or has more than one match.

```{code-cell} ipython3
[wnn_graph] = ds.list_artifacts(
    scope="datastore",
    kind="integrated_graph",
    operation="integrate_assays",
    parameters={"method": "wnn"},
    complete_only=True,
)
[wnn_umap] = ds.list_artifacts(
    scope="datastore",
    kind="embedding",
    operation="run_umap",
    inputs={"graph": wnn_graph},
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

### Question: what populations does the joint RNA and protein graph separate?

```{code-cell} ipython3
ds.plots.embedding(
    layout=wnn_umap,
    color_by=wnn_clusters,
    legend_loc="on_data",
)
```

The WNN graph separates several broad immune populations while allowing RNA and protein to
contribute differently from cell to cell. Cluster numbers are only partitions; marker evidence is
needed before attaching biological names.

### Question: do measured proteins support the population structure?

This store uses the concise antibody labels as feature IDs, so the typed references make that
lookup explicit while keeping the panel titles readable.

```{code-cell} ipython3
protein_panel = [
    FeatureRef(marker, assay="ADT", by="id", label=marker)
    for marker in ("CD3", "CD4", "CD8a", "CD14", "CD19", "CD56")
]
ds.plots.embedding(
    layout=wnn_umap,
    color_by=protein_panel,
    n_columns=3,
    sort_values=True,
)
```

CD3 with CD4 or CD8a identifies T-cell regions, CD14 supports monocytes, CD19 supports B cells,
and CD56 highlights NK-like cells. Their coherent localization on the same WNN map provides the
biological payoff that the cluster-only view cannot.

## Substitute your own matched assays

WNN consumes exact neighbour artifacts built over the same cells. Once RNA and ADT have each
reached that stage, integration is one call. WNN is now the public default:

```python
wnn_graph = own_ds.integrate_assays([rna_neighbors, adt_neighbors])
wnn_umap = own_ds.run_umap(wnn_graph, rna_initialization)
wnn_clusters = own_ds.run_leiden_clustering(wnn_graph)
```

The source refs stay explicit because choosing the assay-specific representations is a scientific
decision. Use {doc}`graph_construction` for the RNA and ADT neighbour chains,
{doc}`multimodal_diagnostics` to compare integration behavior, and
{doc}`../reference/api/integration` for the WNN contract.

## Limits of this result

- WNN combines neighbourhood evidence; it does not prove that a cluster is biologically valid.
- Control antibodies and assay-specific normalization must be reviewed before building ADT
  neighbours for another dataset.
- The displayed labels remain broad interpretations of this marker panel, not an automated cell
  ontology assignment.
