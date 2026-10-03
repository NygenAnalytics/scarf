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

# Read RNA and protein together with CITE-seq

CITE-seq measures RNA and selected surface proteins in the same cells. Antibodies carry
DNA barcodes, called antibody-derived tags (ADTs), that let us count their signal alongside
RNA. The two measurements can help resolve cell types when one marker is hard to detect
in RNA alone.

Here we use a prepared PBMC analysis to ask whether RNA and protein markers support the
same cell identities. Scarf combines the measurements with weighted nearest neighbors
(WNN): each cell receives an RNA weight and a protein weight based on their relative
ability to predict its local neighborhood.

## Open the prepared result

```{code-cell}
import scarf
from scarf.plotting import FeatureRef

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_8K_pbmc_citeseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(f"{dataset}/data.zarr", default_assay="RNA", nthreads=4)
```

The store already contains a WNN graph, UMAP, and Leiden clusters. We select the UMAP
and clusters made from that same graph. Each returned reference identifies one saved result.

```{code-cell}
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

## Find populations to investigate

```{code-cell}
ds.plots.embedding(
    layout=wnn_umap,
    color_by=wnn_clusters,
    legend_loc="on_data",
)
```

Each point is a cell. Nearby points tend to have similar RNA and protein profiles.
The numbered groups give us populations to investigate; their separation alone does not
tell us which cell types they contain.

## Compare RNA and protein markers

We put each protein beside a related RNA marker on the same map. This dataset stores
short antibody names as feature IDs, so `by="id"` selects the protein explicitly.
`FeatureRef` also supplies the assay and a readable panel title.

```{code-cell}
protein_panel = [
    FeatureRef(marker, assay="ADT", by="id", label=f"{marker} protein")
    for marker in ("CD3", "CD4", "CD8a", "CD14", "CD19", "CD56")
]
rna_panel = [
    FeatureRef(gene, assay="RNA", label=f"{gene} RNA")
    for gene in ("CD3D", "CD4", "CD8A", "CD14", "CD19", "NCAM1")
]
paired_panel = [
    marker for pair in zip(protein_panel, rna_panel, strict=True) for marker in pair
]
ds.plots.embedding(
    layout=wnn_umap,
    color_by=paired_panel,
    n_columns=2,
    sort_values=True,
)
```

Start with CD3 protein and CD3D RNA: their shared region supports a T-cell population.
Within that region, CD4 and CD8a help distinguish T-cell subsets. CD14 supports monocytes,
CD19 supports B cells, and CD56/NCAM1 supports NK-like cells. Use several markers together;
{doc}`annotation` explains how to check an interpretation with additional markers.

The RNA panels show library-size-normalized expression. The ADT panels
show centered-log-ratio (CLR) normalized values. Compare where each marker is high,
rather than comparing RNA and protein colorbar numbers. CLR rescales ADT counts; it does
not remove signal from ambient antibodies or nonspecific binding.

## Build a WNN result for your own data

In your own open datastore, `ds`, build RNA and ADT neighbor results over the
**same selected cells**, following
{doc}`graph_construction`. Keep the returned `rna_neighbors` and `adt_neighbors` references
and the RNA embedding initialization, `rna_initialization`. WNN is the default, so the
joint analysis then needs only these calls:

```python
wnn_graph = ds.integrate_assays([rna_neighbors, adt_neighbors])
wnn_umap = ds.run_umap(wnn_graph, rna_initialization)
wnn_clusters = ds.run_leiden_clustering(wnn_graph)
```

Assay-specific preprocessing matters: RNA and a small antibody panel need different
feature choices. Review those choices before integration. Continue to
{doc}`multimodal_diagnostics` to inspect weights and compare integration methods, or
{doc}`tea_seq` to add a third modality.

## What this result can tell us

- Agreement between RNA and protein supports a cell-type interpretation, but does not
  establish every cluster's identity.
- The antibody panel measures only its selected targets. Background binding can produce
  signal even when the target protein is absent; review assay controls.
- RNA and protein are measured at one time point. Their abundance can differ because of
  translation, turnover, and technical effects.
