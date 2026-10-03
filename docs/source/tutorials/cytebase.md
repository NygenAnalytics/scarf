---
description: Connect directly to cloud-hosted Scarf DataStores, explore annotations, and plot gene expression without downloading a complete dataset first.
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

(cytebase_tutorial)=

# Explore Cytebase

Cytebase provides cloud-hosted Scarf DataStores that you can connect to directly
for exploration and analysis, without first downloading an H5AD or a complete
`data.zarr`. Search the catalog, choose a dataset, and open its published store
from a local notebook or a Python session on cloud compute.

Scarf reads metadata and count blocks over the network as needed. Computation
runs where your Python session runs; connecting to Cytebase does not provision
a cloud worker. Shared stores are read-only. A mount lets you save new analysis
results on your execution machine while the counts stay in Cytebase.

This tutorial uses datasets sourced from CELLxGENE. It searches the catalog,
reads cell annotations, and plots the UMAP coordinates supplied by CELLxGENE.
Gene expression comes from the published RNA counts; no new embedding or
analysis pipeline is run in this walkthrough.

The saved results on this page were generated against the public
`Nygen/cytebase` bucket, which readers can access without credentials. The
catalog grows as datasets are added, so discovery results can differ when you
rerun the page.

{nb-download}`Download the executed Jupyter notebook <cytebase.ipynb>`.
For longer worked examples, see the
{ref}`example notebooks <cytebase_example_notebooks>` published in the bucket.

## Prerequisites

Install Scarf's SDK and plotting dependencies in your notebook environment:

```bash
uv pip install --prerelease allow 'scarf[cytebase]' jupyterlab
```

When working from a Scarf checkout, use `uv sync --extra cytebase --extra extra`
and select its Python kernel. `cytebase.Catalog()` connects to the public bucket
by default. To use another bucket, set `CYTEBASE_BUCKET` before starting Jupyter,
or pass `bucket=` to `Catalog`. The explicit argument takes precedence over the
environment variable. For a private bucket, supply `HF_TOKEN` through your
environment or use your existing Hugging Face login. Keep credentials and
private bucket names out of notebook source and outputs.

## What you will learn

- Search by text or exact ontology labels
- Open an RNA assay read-only and read selected cell metadata
- Plot imported UMAP coordinates by cell type and gene expression
- Save a figure and mount a datastore for writable analysis
- Use exact labels or SQL for more detailed searches

## 1. Connect and search

```{code-cell} ipython3
import logging
from IPython.display import Markdown, display

import scarf
from scarf import cytebase

# Keep routine progress messages out of the teaching output.
scarf.configure_output(level="WARNING", progress=False)
# HF retry messages include private bucket URLs; keep them out of shared outputs.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# Connect to the public Cytebase catalog.
catalog = cytebase.Catalog()  # Public Cytebase unless CYTEBASE_BUCKET is set.
```

`Catalog()` connects to the public catalog and keeps a small local copy. Searches use this
catalog, so they do not read any count data. By default, they return datasets ready to open.

The example is a single-nucleus RNA-seq dataset of human kidney cortex from
donors with and without diabetic kidney disease (Wilson et al., 2022). Search
for its author and tissue, then keep the returned identifier rather than
constructing an ID yourself.

```{code-cell} ipython3
# Search the catalog for the dataset used in this example.
matches = catalog.search("wilson kidney cortex")
# Stop if the expected kidney dataset is not available.
if not matches:
    raise RuntimeError("The example kidney dataset is not ready in this bucket")
# Summarize dataset titles and sizes while keeping full IDs in the result rows.
match_preview = matches.to_markdown(
    columns=["cytebase_id", "title", "cell_count"], max_cell_chars=32
)
# Display the search summary before choosing a dataset.
display(Markdown(match_preview))
```

```{code-cell} ipython3
# Keep the dataset identifier returned by the catalog.
dataset_id = matches[0]["cytebase_id"]
# Read the dataset's scientific metadata and citation.
entry = catalog.dataset(dataset_id)
# Inspect the dataset record and source citation.
entry
```

`entry` is a `DatasetEntry` containing metadata about the scientific dataset.
Displaying it reads the catalog and dataset record. It does not open the remote
Zarr hierarchy. The citation and CELLxGENE links identify the original study;
retain these when using the data.

## 2. Open the remote DataStore

`catalog.open_datastore(entry.id)` validates the published dataset version and
build receipt, then returns a `DataStore` with `zarr_mode="r"`. Counts remain
remote and the SDK does not write back to the bucket. Keep this object as `ds`
for metadata queries, plotting, and other datastore operations.

```{code-cell} ipython3
# Open the datastore for the following analysis.
ds = catalog.open_datastore(entry.id)
# Inspect the store's assays and dimensions.
ds
```

Small Zarr metadata objects are cached in memory for this open store. Count
chunks are fetched when needed; the SDK does not maintain a persistent local
count cache. Opening and plotting still depend on network latency.

The source H5AD may contain more embeddings than the pipeline imported. Check
the imported keys before plotting:

```{code-cell} ipython3
# List the embeddings recorded in the source dataset.
print("Source embedding keys:", entry.source_embeddings())
# List the embeddings actually imported into the Scarf store.
print("Imported embedding keys:", sorted(cytebase.embeddings(ds)))
# Select the imported UMAP without recomputing its coordinates.
umap_ref = cytebase.embedding(ds, "X_umap")
```

The current pipeline imports `obsm/X_umap`. An `X_pca` key in the source list
alone does not mean its coordinates are available in the Scarf store.

## 3. Read cell metadata

CELLxGENE annotations are stored beside Scarf's QC columns, such as
`RNA_nCounts` and `RNA_nFeatures`. Request only the columns needed for a figure:

```{code-cell} ipython3
import pandas as pd

# Choose the metadata fields needed for exploration.
wanted = ["cell_type", "tissue", "disease", "donor_id", "sex", "assay"]
# Keep only metadata fields available in this store.
columns = [column for column in wanted if column in ds.cells.columns]
# Read cell metadata and index it by cell ID.
meta = ds.cells.to_pandas_dataframe(["ids", *columns], key="I").set_index("ids")
# Preview cell types and donor groups without truncating annotation names.
with pd.option_context("display.max_colwidth", None):
    display(meta[["cell_type", "disease", "donor_id"]].head())
```

This materializes the selected metadata columns as a pandas DataFrame, not the
expression matrix. On a large atlas, metadata and coordinates still require
memory proportional to the selected cells.

Use `ds.cells.columns` to check the available names. Scarf replaces `/` and `\` in imported
column names with `_` so they can be stored safely.

```{code-cell} ipython3
# Count cells assigned to each published cell type.
meta["cell_type"].value_counts().rename("cells").to_frame()
```

Compare cell-type composition between the donor groups:

```{code-cell} ipython3
# Compare the cell counts or fractions across the selected groups.
pd.crosstab(meta["cell_type"], meta["disease"])
```

## 4. Plot the stored UMAP

`ds.plots.embedding(layout=umap_ref, ...)` uses the exact imported coordinates
without recomputing UMAP. With a moderate number of cell types, the default
legend placement writes each label on its cluster. A large atlas with many
labels is better viewed with selected groups or `legend_loc="none"`.

```{code-cell} ipython3
# Plot the published cell types and retain the figure for export.
cell_type_plot = ds.plots.embedding(
    layout=umap_ref, color_by="cell_type", figsize=(10, 7)
)
```

Focus on the five most common cell types without changing the source dataset:

```{code-cell} ipython3
# Select the five most common cell types for a closer view.
top_types = meta["cell_type"].value_counts().head(5).index.tolist()
# Focus the UMAP on the five most abundant cell types.
ds.plots.embedding(
    layout=umap_ref,
    color_by="cell_type",
    groups=top_types,
    figsize=(10, 6),
)
```

### Gene expression

Scarf looks up gene symbols without regard to case. Here we use UMOD for the thick ascending
limb of the loop of Henle, SLC34A1 for the proximal tubule, NPHS1 for podocytes, and PECAM1 for
endothelial cells. An unknown or ambiguous name raises an error, so check the feature metadata
when adapting this panel to another dataset.

```{code-cell} ipython3
# Choose representative markers for the kidney populations.
genes = ["UMOD", "SLC34A1", "NPHS1", "PECAM1"]
```

Expression values are normalized on read; this example explicitly requests a
`log1p` transform. Neither the count matrix nor the imported coordinates are
modified. Drawing high values last makes expressing cells easier to see. Each
gene is read from the remote counts, so every panel adds network reads.

```{code-cell} ipython3
from scarf.plotting import NormalizationSpec

# Plot the four kidney markers using log-normalized expression.
ds.plots.embedding(
    layout=umap_ref,
    color_by=genes,
    normalization=NormalizationSpec(transform="log1p"),
    sort_values=True,
    n_columns=2,
    figsize=(12, 10),
)
```

## 5. Other plots from the same assay

The read-only DataStore also supports plots from existing metadata and counts.
For example, compare a QC measure across cell types:

```{code-cell} ipython3
from scarf.plotting import CellField

# Compare distributions within the stated groups.
ds.plots.distribution(
    "RNA_nCounts",
    grouping=CellField("cell_type"),
    groups=top_types,
    figsize=(10, 5),
)
```

Gene distributions use the same interface; this optional example reads
additional expression values:

```python
# Compare distributions within the stated groups.
ds.plots.distribution(
    genes[:2],
    grouping=CellField("cell_type"),
    groups=top_types,
    figsize=(12, 5),
)
```

## 6. Coordinates for custom plots

Coordinates follow the embedding artifact's frozen cell selection and are
indexed by cell ID. Join on those IDs instead of assuming the same row order as
a separate metadata table:

```{code-cell} ipython3
# Read coordinates indexed by the embedding's cell IDs.
coords = cytebase.embedding_coordinates(ds, umap_ref)
# Align metadata and coordinates by cell ID.
frame = coords.join(meta)
# Preview joined coordinates and full cell-type names in the same rows.
with pd.option_context("display.max_colwidth", None):
    display(frame[["umap_1", "umap_2", "cell_type"]].head())
```

The joined table is ready for a custom figure or for export. For most views, the Scarf plotting
calls above already handle group selection and legends.

## 7. Save a figure

Scarf returns a `PlotResult`. Save the first cell-type figure without repeating
its remote reads, and close the result when finished. For a new figure that
should only be saved, pass `show=False` to the plotting call.

```{code-cell} ipython3
from pathlib import Path

# Choose a local folder for exported figures.
out = Path("figures")
# Create the local folder if it does not already exist.
out.mkdir(exist_ok=True)
# Save the existing figure without repeating its data reads.
cell_type_plot.save(out / "cytebase_cell_types.png", dpi=150)
# Close the figure or reader after its final use.
cell_type_plot.close()
# Show the local filename of the exported figure.
print("Saved figures/cytebase_cell_types.png")
```

## 8. Writable analysis and larger datasets

The snippets in this section are optional and were not executed for this page.
Run them in your local Python environment or on cloud compute. Create a mount
before running Scarf operations that save new results:

```python
# Mount a writable analysis store while keeping counts remote.
analysis_ds = catalog.mount_datastore(entry.id, at="./analysis.zarr")
```

The mount stores metadata and new analysis results at `./analysis.zarr` on the
machine running Python, including when that machine is a cloud worker. Counts
remain remote. Reopening through `catalog.mount_datastore` checks the source build identity.
If that build has changed, choose a new mount directory. Latest-only remote
storage cannot guarantee an immutable source during an already-open session;
a new `catalog.open_datastore` call revalidates the current published build.

A fresh mount copies cell and feature metadata and resolves the published
analysis artifacts read only, so `cytebase.embeddings(analysis_ds)` also finds
the imported UMAP and `analysis_ds.plots.embedding(layout=...)` works on the
mount. New artifacts you create are written to `./analysis.zarr`.

The same search and plotting calls work for larger studies. A Tabula Sapiens
plot can be substantially slower and require more memory. Hide its long tissue
legend to preserve the plotting area:

```python
# Find the larger atlas in the same catalog.
atlases = catalog.search("tabula sapiens", ready_only=True)
# Open the atlas only when the catalog search returned a match.
if atlases:
    # Open the selected atlas without downloading all counts.
    atlas_ds = catalog.open_datastore(atlases[0]["cytebase_id"])
    # Reuse the atlas's imported UMAP.
    atlas_umap_ref = cytebase.embedding(atlas_ds, "X_umap")
    # Plot atlas tissues without an oversized legend.
    atlas_ds.plots.embedding(
        layout=atlas_umap_ref,
        color_by="tissue",
        legend_loc="none",
        figsize=(8, 8),
    )
```

## 9. Search by exact labels or SQL

`search` matches words without regard to case. `find_datasets` matches exact
ontology labels. `list_terms` supplies the available labels in natural order;
its counts cover registered datasets, including those that are not ready.

```{code-cell} ipython3
# List the catalog labels available for an exact search.
catalog.list_terms("disease")
```

Take an exact tissue label from the selected dataset:

```{code-cell} ipython3
# Use an exact tissue label supplied by the catalog.
tissue = matches[0]["tissue_labels"][0]
# Find datasets using the exact tissue and organism labels.
kidney_datasets = catalog.find_datasets(tissue=tissue, organism="Homo sapiens")
# Summarize dataset sizes and disease labels with shortened display IDs.
kidney_preview = kidney_datasets.to_markdown(
    columns=["cytebase_id", "cell_count", "disease_labels"], max_cell_chars=32
)
# Display the exact-match search results.
display(Markdown(kidney_preview))
```

SQL queries run against the verified local catalog. Select just the columns you
need and bind values with `parameters`:

```{code-cell} ipython3
# Run the parameterized catalog query on the local metadata copy.
small_datasets = catalog.query(
    """
    SELECT cytebase_id, first_author, year, cell_count, n_genes,
           len(cell_type_labels) AS n_cell_types
    FROM datasets
    WHERE status = ? AND processed_version_id = latest_version_id
      AND zarr_uri IS NOT NULL
    ORDER BY cell_count
    LIMIT 10
    """,
    parameters=["ready"],
)
# Keep all query columns while shortening only their displayed values.
display(Markdown(small_datasets.to_markdown(max_cell_chars=24)))
```

Pass `ready_only=False` to discover datasets that are still being prepared. For more SQL
examples and a survey of the collection, see the catalog tour linked below.

(cytebase_example_notebooks)=

## Example notebooks

The bucket's
[`notebooks` folder](https://huggingface.co/buckets/Nygen/cytebase/tree/notebooks)
holds three executed notebooks that extend this walkthrough. Each one opens
public datasets without credentials and keeps its outputs, so you can read the
results before running anything. The Hugging Face file viewer does not display
notebooks of this size, so the links below open rendered copies on
[nbviewer](https://nbviewer.org). The case study is also available as the
{doc}`cytebase_covid19` page in this documentation.

| Notebook | What it shows |
| --- | --- |
| [Catalog tour](https://nbviewer.org/urls/huggingface.co/buckets/Nygen/cytebase/resolve/notebooks/cytebase_01_catalog_tour.ipynb) | The collection at a glance by organism, assay, tissue, disease and publication year; search by text, exact labels and SQL across studies; then one dataset's UMAP |
| [COVID-19 case study](https://nbviewer.org/urls/huggingface.co/buckets/Nygen/cytebase/resolve/notebooks/cytebase_02_covid19_pbmc_case_study.ipynb) | COVID-19 and healthy blood from Wilk et al. (2020): study design from metadata, composition per donor, a marker dot plot that checks the published labels, and an interferon response compared between donors |
| [UMAP gallery](https://nbviewer.org/urls/huggingface.co/buckets/Nygen/cytebase/resolve/notebooks/cytebase_03_umap_gallery.ipynb) | Published UMAPs from several tissues and species in one figure; labels, palettes, highlights, facets, density contours and themes; blockwise rasters of 1.1 million Tabula Sapiens cells; and a custom matplotlib figure |

Download a notebook from the folder and open it in the environment from the
prerequisites above:

```bash
curl -LO https://huggingface.co/buckets/Nygen/cytebase/resolve/notebooks/cytebase_01_catalog_tour.ipynb
```

The catalog tour runs in about two minutes, the case study in about five, and
the gallery in about fifteen; most of that time is spent reading from the
network. Their saved outputs reflect the catalog when they were executed.

## Common issues

- **No matching dataset:** check bucket selection, access, and whether the desired
  version is ready. `ready_only=False` includes unfinished registrations.
- **Missing embedding:** use `cytebase.embeddings(ds)` to check imported coordinates.
  `entry.source_embeddings()` also lists source keys that were not imported.
- **Crowded figure:** increase `figsize`, reduce the number of groups or panels,
  or hide the legend. The examples use explicit sizes instead of suppressing
  plotting warnings.
- **Unexpected expression costs:** count blocks are remote, and a single-gene
  request may read a larger storage chunk. Start with a small dataset and a few genes.
- **Writing to a read-only store:** use a local mount for analysis that saves results.
- **`429 Too Many Requests`:** anonymous Hugging Face access allows 500 API
  requests per 5 minutes per IP address, and opening a store and reading its
  metadata uses a share of them. Wait for the window to reset and rerun the
  cell, or sign in with `hf auth login`. `Catalog()` then reads with your token,
  which has a higher limit.

See the [Cytebase API reference](../reference/api/cytebase.md) for the full SDK
and [Remote stores](remote_stores.md) for mounted analysis mechanics. Saved
documentation outputs are a snapshot; rerunning checks the current catalog and
may produce different discovery results as the bucket changes.
