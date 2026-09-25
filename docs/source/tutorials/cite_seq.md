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

# scCITE-seq Primer

Cellular Indexing of Transcriptomes and Epiptoes by Sequencing (CITE-seq) is a multimodal single-cell method that allows you to measure both **the transcriptome** (intracellular mRNA expression) and **epitopes** (cell-surface protein abundance) in the exact same single cell. Data in CITE-seq has 2 distinct features for each cell, with the first one being the measured **mRNA** **expression of the genes**; The secondary feature is measured **cell-surface protein abundance**. Cells are incubated with antibodies targeted against specific surface markers of interest, such as CD4; CD8; or CD19. Since each antibody is conjugated to a corresponding unique DNA barcode, by counting these sequenced barcodes, we can yield an Antibody-Derived Tag (ADT) count; The ADT count directly reflects the abundance of the protein on the cell's surface. Generally, CITE-seq is dominantly performed in immune/PBMC contexts, but it is not restricted to this realm.

# Integrate RNA and Protein information with CITE-seq

To interpret matched RNA and ADT measurements from the same PBMCs in order to identify cell identities, we used a completed analysis. SCARF uses a weighted-nearest-neighbor (WNN) based approach, which builds a joint neighbour graph by letting each cell weigh RNA versus protein evidence according to how well each modality predicts its own neighbours, so the final graph is built off both modalities.

## Open the prepared multimodal result

```{code-cell}
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

The prepared result already contains all of the complete analysis, thus all we do is grab the WNN graph, the UMAP, and the [Leiden] clustering results.

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

### What unique populations do the joint RNA and protein graph identify?

Before clustering, single-cell workflows construct a k-nearest-neighbors graph, connecting cells that share similar profiles. In CITE-seq, we have 2 different views, that when combined, provide more thorough pieces of information. The information from the RNA provides information of thousands of genes, but specific marker genes may often drop out, and not be sequenced. CITE-seq comes in here to provide a clean, and stable way to detect the protein expression of these genes on the surface of the cell, but you only get a minuscule portion of genes in comparison to the RNA. When combined, differentiating between cell identities becomes more smooth, as we now possess protein evidence.

```{code-cell}
ds.plots.embedding(
    layout=wnn_umap,
    color_by=wnn_clusters,
    legend_loc="on_data",
)
```

In this UMAP embedding, which is simply a 2D representation of the WNN graph, we can notice that certain groups of cell separate into distinct, well-defined clusters rather than one continous smear. This separation once again indicates that the combined RNA and protein evidence suggest distinctly resolved cellular states. But to actually identify the different cell identites, we can visualize the RNA and protein expession.

### Does the measured protein and RNA expression support the population structure?

One important thing to consider before you visualize the protein expression is that the ADT counts rarely contain true zeros. Unbound antibodies stick nonspecifically to every droplet (background binding), so each cell carries low-level signal for every antibody. This is because every antibody is highly **specific to one sequence**, but that sequence is **not highly specific**.

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
    panel for pair in zip(protein_panel, rna_panel, strict=True) for panel in pair
]
ds.plots.embedding(
    layout=wnn_umap,
    color_by=paired_panel,
    n_columns=2,
    sort_values=True,
)
```

To generally interpret the values on the graph, in our example, the panels with RNA expression are log1p library-sized-normalization expression; a standarf approach for scRNA-seq data. For the protein (ADT) panels, we have centered-log-ratio normalized abundance. Higher expression on the protein panels means their is more surface protein expression releative to the background, in which the background is the extremely low values nearing zero, but neber true zero.

Here, our co-expression of CD3, CD4 and CD8a on both the RNA and protein graph indicates the cluster being T-cells, with CD14 supporting regions of monocytes, CD19 supporting the regions of B cells, and CD56/NCAM1 highlighting NK-like cells. The coherent localization on the same embedding provides further evidence of identifying cell states.

For further information regarding the markers chosen for our cell identification purposes, refer to resources in {doc}`annotation`.

## Limits of this result

- WNN combines neighbourhood evidence; it does not prove that a cluster is biologically valid.
- Control antibodies and assay-specific normalization must be reviewed before building ADT
  neighbours for another dataset.
- The displayed labels remain broad interpretations of this marker panel, not an automated cell
  ontology assignment.

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
