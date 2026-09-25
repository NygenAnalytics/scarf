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

Cellular Indexing of Transcriptomes and Epitopes by Sequencing (CITE-seq) is a multimodal single-cell method that allows you to measure both **the transcriptome** (intracellular mRNA expression) and **epitopes** (cell-surface protein abundance) in the exact same single cell. Data in CITE-seq has 2 distinct features for each cell, with the first one being the measured **mRNA expression** of the genes; The secondary feature is measured **cell-surface protein abundance**. Cells are incubated with antibodies targeted against specific surface markers of interest, such as CD4, CD8, or CD19. Since each antibody is conjugated to a corresponding unique DNA barcode, by counting these sequenced barcodes, we can yield an Antibody-Derived Tag (ADT) count; The ADT count directly reflects the abundance of the protein on the cell's surface. Generally, CITE-seq is dominantly performed in immune/PBMC contexts, but it is not restricted to this realm.

# Integrate RNA and protein information with CITE-seq

To interpret matched RNA and ADT measurements from the same PBMCs in order to identify cell identities, we used a completed analysis. SCARF uses a weighted-nearest-neighbor (WNN) based approach, which builds a joint neighbor graph by letting each cell weigh RNA versus protein evidence according to how well each modality predicts its own neighbors, so the final graph is built off both modalities.

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

The prepared result already contains all of the complete analysis, thus all we do is grab the WNN graph, the UMAP, and the Leiden clustering results.

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

### What unique populations does the joint RNA and protein graph identify?

Before clustering, single-cell workflows construct a k-nearest-neighbors graph, connecting cells that share similar profiles. In CITE-seq, we have 2 different views, that when combined, provide more thorough pieces of information. The information from the RNA provides information of thousands of genes, but specific marker genes may often drop out, and not be sequenced. CITE-seq comes in here to provide a clean, and stable way to detect the protein expression of these genes on the surface of the cell, but you only get a minuscule portion of genes in comparison to the RNA. When combined, differentiating between cell identities becomes smoother, as we now possess protein evidence.

```{code-cell}
ds.plots.embedding(
    layout=wnn_umap,
    color_by=wnn_clusters,
    legend_loc="on_data",
)
```

In this UMAP embedding, which is simply a 2D representation of the WNN graph, we can notice that certain groups of cells separate into distinct, well-defined clusters rather than one continuous smear. This separation once again indicates that the combined RNA and protein evidence suggest distinctly resolved cellular states. But to actually identify the different cell identities, we can visualize the RNA and protein expression.

### Does the measured protein and RNA expression support the population structure?

One important thing to consider before you visualize the protein expression is that the ADT counts rarely contain true zeros. Unbound antibodies stick nonspecifically to every droplet (background binding), so each cell carries low-level signal for every antibody. This is because the antibodies may get trapped in the droplet alongside cell debris, the antibodies sticking nonspecifically to the cell membranes, or antibodies binding to different receptors than intended. 

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

To generally interpret the values on the graph, in our example, the panels with RNA expression are log1p library-size normalization expression; a standard approach for scRNA-seq data. For the protein (ADT) panels, we have centered-log-ratio normalized abundance. Higher expression on the protein panels means there is more surface protein expression relative to the background, in which the background is the extremely low values nearing zero, but never truly zero.

Here, our co-expression of CD3 on the RNA and protein indicates the cluster is likely T cells. Our co-expression of CD4 and CD8a helps in differentiating between subtypes of T cells. Genes and proteins like CD14 support regions of monocytes, while CD19 supports the regions of B cells, and CD56/NCAM1 highlights NK-like cells. The coherent localization on the same embedding provides further evidence of identifying cell states.

For further information regarding the markers chosen for our cell identification purposes, refer to {doc}`annotation`.

## Limits of this result

- **Panel Pre-Selection & Biological Blind Spots:** Unlike RNA-seq, which measures the whole transcriptome (~20,000 genes) without bias, CITE-seq surface protein panels are strictly targeted to a set of proteins (typically 10 to 200 antibodies). If a novel cell type or activation state is driven by a surface marker not included in your selected panel, the protein modality is completely blind to it and integration must rely solely on RNA expression.
- **Ambient Antibodies & Non-Specific Background Binding:** ADT counts do not equal zero even in cells that do not express the protein. High ambient antibody concentrations or unblocked Fc receptors can create false-positive protein signals. Advanced workflows often require isotype controls or ambient-subtraction algorithms to adjust for technical effects like this.
- **Temporality of RNA expression vs. Protein expression:** CITE-seq is transcriptomics performed on dead cells, meaning we only capture a single snapshot of the cell's state. High mRNA abundance does not guarantee high surface protein levels. Differences in translation efficiency, post-transcriptional repression, and protein half-lives mean RNA and protein operate on different biological timelines, thus why we may see differences in our data. It's important to keep this idea in mind when interpreting results, and identify if this is a question that can answer the potential observations in your data.

## Substitute your own input

The workflow for CITE-seq can be described as the following:

When processing your own CITE-seq dataset from scratch, the referenced workflow below can be used as a brief example:

```python
# Firstly prepare the RNA Neighborhood Graph
# Filter cells and select highly variable genes (HVGs)
own_ds.auto_filter_cells()
own_ds.mark_hvgs(from_assay="RNA", top_n=2000)

# Perform dimensionality reduction with PCA and find k-nearest neighbors
rna_pca = own_ds.run_pca(from_assay="RNA", dims=30)
rna_ann = own_ds.build_ann_index(rna_pca)
rna_neighbors = own_ds.query_neighbors(rna_ann, k=21)

# Build a K-means starting layout from the RNA PCA to anchor the
# global manifold
rna_initialization = own_ds.build_embedding_initialization(rna_pca)

# Secondly prepare the ADT (Protein) Neighborhood Graph
# Normalize ADT counts (e.g., CLR) to account for background staining
own_ds.run_normalization(from_assay="ADT")

# For targeted antibody panels (typically < 100 markers), calculate neighbors 
# directly on normalized protein features or after minor dimensionality reduction
adt_ann = own_ds.build_ann_index(from_assay="ADT")
adt_neighbors = own_ds.query_neighbors(adt_ann, k=21)


# Thirdly, we perform multimodal integration with WNN
# WNN computes cell-specific weights based on how reliably each modality 
# predicts neighbors, producing a single consensus neighborhood graph.
wnn_graph = own_ds.integrate_assays([rna_neighbors, adt_neighbors], method="wnn")


# Finally, visualize the joint graph as a 2D UMAP and clustering
# Project the joint graph into 2D space (using RNA initialization for global orientation)
# Then perform Leiden clustering on the joint graph to identify potential cell types

wnn_umap = own_ds.run_umap(wnn_graph, rna_initialization)
wnn_clusters = own_ds.run_leiden_clustering(wnn_graph)
```

Some references you can access for a deeper explanation of the graph building can be found at {doc}`graph_construction`. SCARF also has other integration methods for CITE-seq data, such as shared nearest-neighbor (SNN) integration, which gives RNA and protein equal weight instead of learning per-cell weights. To compare the best integration methods for your CITE-seq analysis, refer to {doc}`../reference/api/integration` and {doc}`multimodal_diagnostics` to compare integration predictions.
