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

- Summarize the collection and run local catalog SQL across studies
- Search by text or exact ontology labels
- Open an RNA assay read-only and read selected cell metadata
- Plot imported UMAP coordinates by cell type and gene expression
- Export coordinates for custom figures and mount a datastore for writable analysis

## 1. Connect and search

```{code-cell} ipython3
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import scarf
from scarf import cytebase
from scarf.plotting import CellField, NormalizationSpec

scarf.configure_output(level="WARNING", progress=False)
# HF retry messages include private bucket URLs; keep them out of shared outputs.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
catalog = cytebase.Catalog()  # Public Cytebase unless CYTEBASE_BUCKET is set.
```

The catalog is a local DuckDB database. Setup and each new catalog query check
the published SHA-256 and verify the local copy. An unchanged catalog is reused
silently. Its files are `~/.scarf/cytebase.duckdb` and
`~/.scarf/cytebase.duckdb.sha256` on Unix, or under `%LOCALAPPDATA%\scarf` on
Windows. This is a catalog cache, not a download of the count matrices.

### The collection at a glance

The catalog has two tables. `datasets` holds one row per dataset, with its title,
citation, cell count, and a list of labels for each facet. `dataset_terms` holds
one row per dataset and label, with columns `cytebase_id`, `facet`, `term_id`,
`label`, and `label_rank`. A study is a CELLxGENE collection, usually one
publication, and many studies publish several datasets.

```{code-cell} ipython3
READY = (
    "status = 'ready' AND processed_version_id = latest_version_id"
    " AND zarr_uri IS NOT NULL"
)
catalog.query(
    f"""
    SELECT count(*) AS datasets,
           count(DISTINCT collection_id) AS studies,
           sum(cell_count) AS cells
    FROM datasets
    WHERE {READY}
    """
)
```

`catalog.connect_catalog()` returns a read-only DuckDB connection, which is
convenient when you want a pandas DataFrame, for example to plot it. Close it
after use with a `with` block. A dataset with several labels in one facet, such
as a multi-tissue atlas, counts once toward each label.

```{code-cell} ipython3
def top_labels(facet, n=10):
    with catalog.connect_catalog() as connection:
        return connection.execute(
            f"""
            SELECT t.label, count(*) AS datasets
            FROM dataset_terms AS t JOIN datasets USING (cytebase_id)
            WHERE t.facet = ? AND {READY}
            GROUP BY t.label
            ORDER BY datasets DESC, t.label
            LIMIT ?
            """,
            [facet, n],
        ).df()


fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
for ax, facet in zip(axes, ["organism", "assay", "tissue"]):
    counts = top_labels(facet)
    ax.barh(counts["label"], counts["datasets"], color="#2a78d6")
    ax.invert_yaxis()
    ax.bar_label(ax.containers[0], fmt="{:,.0f}", padding=2, fontsize=8)
    ax.margins(x=0.12)
    ax.set(title=f"Top {facet} labels", xlabel="Datasets")
    ax.spines[["top", "right"]].set_visible(False)
plt.show()
plt.close(fig)
```

By default, discovery returns datasets whose latest registered version is ready
to open. Pass `ready_only=False` to also discover registered datasets that have
not finished processing.

```{code-cell} ipython3
ready = catalog.find_datasets(ready_only=True)
ready
```

Results display as Markdown tables and behave as lists of complete row
dictionaries. Display truncation does not change the underlying values.
`max_cell_chars=None` shows complete titles, IDs, and label lists.

The example is a single-nucleus RNA-seq dataset of human kidney cortex from
donors with and without diabetic kidney disease (Wilson et al., 2022). Search
for its author and tissue, then keep the returned identifier rather than
constructing an ID yourself.

```{code-cell} ipython3
matches = catalog.search("wilson kidney cortex", ready_only=True, max_cell_chars=None)
if not matches:
    raise RuntimeError("The example kidney dataset is not ready in this bucket")
matches
```

```{code-cell} ipython3
dataset_id = matches[0]["cytebase_id"]
entry = catalog.dataset(dataset_id)
entry
```

`entry` is a `DatasetEntry` containing metadata about the scientific dataset.
Displaying it reads the catalog and dataset record. It does not open the remote
Zarr hierarchy. The citation and CELLxGENE links identify the original study;
retain these when using the data.

### Exact labels and SQL

`search` matches words without regard to case. `find_datasets` matches exact
ontology labels. `list_terms` supplies the available labels in natural order;
its counts cover registered datasets, including those that are not ready.

```{code-cell} ipython3
catalog.list_terms("disease")
```

Take an exact tissue label from the selected dataset:

```{code-cell} ipython3
tissue = matches[0]["tissue_labels"][0]
catalog.find_datasets(tissue=tissue, organism="Homo sapiens", ready_only=True)
```

SQL queries run against the verified local catalog. Select just the columns you
need and bind values with `parameters`:

```{code-cell} ipython3
catalog.query(
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
```

### SQL across studies

`dataset_terms` keeps questions that span studies short. Which human tissues have
datasets that contain microglia, the resident immune cells of the brain and
retina? The catalog records which cell types each dataset contains, not how many
cells of each type, so `cells` is the total size of the matching datasets.

```{code-cell} ipython3
catalog.query(
    f"""
    WITH hits AS (
        SELECT DISTINCT cytebase_id FROM dataset_terms
        WHERE facet = 'cell_type' AND label = ?
    )
    SELECT t.label AS tissue,
           count(DISTINCT d.collection_id) AS studies,
           count(*) AS datasets,
           sum(d.cell_count) AS cells
    FROM hits
    JOIN datasets AS d USING (cytebase_id)
    JOIN dataset_terms AS t ON t.cytebase_id = d.cytebase_id AND t.facet = 'tissue'
    WHERE {READY} AND list_contains(d.organism_labels, ?)
    GROUP BY t.label
    ORDER BY datasets DESC, cells DESC
    LIMIT 10
    """,
    parameters=["microglial cell", "Homo sapiens"],
)
```

## 2. Open the remote DataStore

`catalog.open_datastore(entry.id)` validates the published dataset version and
build receipt, then returns a `DataStore` with `zarr_mode="r"`. Counts remain
remote and the SDK does not write back to the bucket. Keep this object as `ds`
for metadata queries, plotting, and other datastore operations.

```{code-cell} ipython3
ds = catalog.open_datastore(entry.id)
ds
```

Small Zarr metadata objects are cached in memory for this open store. Count
chunks are fetched when needed; the SDK does not maintain a persistent local
count cache. Opening and plotting still depend on network latency.

The source H5AD may contain more embeddings than the pipeline imported. Check
the imported keys before plotting:

```{code-cell} ipython3
print("Source embedding keys:", entry.source_embeddings())
print("Imported embedding keys:", sorted(cytebase.embeddings(ds)))
umap_ref = cytebase.embedding(ds, "X_umap")
```

The current pipeline imports `obsm/X_umap`. An `X_pca` key in the source list
alone does not mean its coordinates are available in the Scarf store.

## 3. Read cell metadata

CELLxGENE annotations are stored beside Scarf's QC columns, such as
`RNA_nCounts` and `RNA_nFeatures`. Request only the columns needed for a figure:

```{code-cell} ipython3
wanted = ["cell_type", "tissue", "disease", "donor_id", "sex", "assay"]
columns = [column for column in wanted if column in ds.cells.columns]
meta = ds.cells.to_pandas_dataframe(["ids", *columns], key="I").set_index("ids")
meta.head()
```

This materializes the selected metadata columns as a pandas DataFrame, not the
expression matrix. On a large atlas, metadata and coordinates still require
memory proportional to the selected cells.

Zarr reads `/` and `\` in a column name as path separators, so Scarf stores a
source column whose name contains either one under the name with `_` in their
place. For example, `Baseline eGFR (ml/min/1.73m2) (Binned)` becomes
`Baseline eGFR (ml_min_1.73m2) (Binned)` in `ds.cells.columns`. The build's
`scarf_ingest.json` keys its `obs_summary` by these Scarf names, while its
`h5ad_keys` listing keeps the source names.

```{code-cell} ipython3
meta["cell_type"].value_counts().rename("cells").to_frame()
```

Compare cell-type composition between the donor groups:

```{code-cell} ipython3
if "disease" in meta:
    display(pd.crosstab(meta["cell_type"], meta["disease"]))
```

## 4. Plot the stored UMAP

`ds.plots.embedding(layout=umap_ref, ...)` uses the exact imported coordinates
without recomputing UMAP. With a moderate number of cell types, the default
legend placement writes each label on its cluster. A large atlas with many
labels is better viewed with selected groups or `legend_loc="none"`.

```{code-cell} ipython3
cell_type_plot = ds.plots.embedding(
    layout=umap_ref, color_by="cell_type", figsize=(10, 7)
)
```

Focus on the five most common cell types without changing the source dataset:

```{code-cell} ipython3
top_types = meta["cell_type"].value_counts().head(5).index.tolist()
ds.plots.embedding(
    layout=umap_ref, color_by="cell_type", groups=top_types, figsize=(10, 6)
);
```

### Gene expression

Gene names come from the published assay feature metadata. Resolve the symbols
against that table first; not every requested gene is present in every dataset.
Here the explicit lookup handles differences in letter case. The markers are
UMOD for the thick ascending limb of the loop of Henle, SLC34A1 for the
proximal tubule, NPHS1 for podocytes, and PECAM1 for endothelial cells.

```{code-cell} ipython3
requested_genes = ["UMOD", "SLC34A1", "NPHS1", "PECAM1"]
by_upper = {str(name).upper(): str(name) for name in ds.RNA.feats.fetch_all("names")}
genes = [by_upper[name.upper()] for name in requested_genes if name.upper() in by_upper]
print("Found:", genes)
print("Missing:", [name for name in requested_genes if name.upper() not in by_upper])
```

Expression values are normalized on read; this example explicitly requests a
`log1p` transform. Neither the count matrix nor the imported coordinates are
modified. Drawing high values last makes expressing cells easier to see. Each
gene is read from the remote counts, so every panel adds network reads.

```{code-cell} ipython3
if genes:
    ds.plots.embedding(
        layout=umap_ref,
        color_by=genes,
        normalization=NormalizationSpec(transform="log1p"),
        sort_values=True,
        n_columns=2,
        figsize=(12, 10),
    );
```

## 5. Other plots from the same assay

The read-only DataStore also supports plots from existing metadata and counts.
For example, compare a QC measure across cell types:

```{code-cell} ipython3
if "RNA_nCounts" in ds.cells.columns:
    ds.plots.distribution(
        "RNA_nCounts",
        grouping=CellField("cell_type"),
        groups=top_types,
        figsize=(10, 5),
    );
```

Gene distributions use the same interface; this optional example reads
additional expression values:

```python
ds.plots.distribution(
    genes[:2], grouping=CellField("cell_type"), groups=top_types, figsize=(12, 5)
)
```

## 6. Coordinates for custom plots

Coordinates follow the embedding artifact's frozen cell selection and are
indexed by cell ID. Join on those IDs instead of assuming the same row order as
a separate metadata table:

```{code-cell} ipython3
coords = cytebase.embedding_coordinates(ds, umap_ref)
frame = coords.join(meta)
frame.head()
```

For a custom figure, reserve space for a short legend and show the other cells
in grey:

```{code-cell} ipython3
x, y = coords.columns[:2]
top = frame["cell_type"].value_counts().head(5).index
rest = frame[~frame["cell_type"].isin(top)]

fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
ax.scatter(rest[x], rest[y], s=1, c="lightgrey", label="other", rasterized=True)
for label in top:
    part = frame[frame["cell_type"] == label]
    ax.scatter(part[x], part[y], s=1, label=label, rasterized=True)
ax.set(xlabel=x, ylabel=y, title="Five most common cell types")
ax.legend(
    markerscale=6, fontsize=8, bbox_to_anchor=(1, 1), loc="upper left", frameon=False
)
plt.show()
plt.close(fig)
```

## 7. Save a figure

Scarf returns a `PlotResult`. Save the first cell-type figure without repeating
its remote reads, and close the result when finished. For a new figure that
should only be saved, pass `show=False` to the plotting call.

```{code-cell} ipython3
out = Path("figures")
out.mkdir(exist_ok=True)
cell_type_plot.save(out / "cytebase_cell_types.png", dpi=150)
cell_type_plot.close()
print("Saved figures/cytebase_cell_types.png")
```

## 8. Writable analysis and larger datasets

The remaining snippets are optional and were not executed for this page.
Run them in your local Python environment or on cloud compute. Create a mount
before running Scarf operations that save new results:

```python
analysis_ds = catalog.mount_datastore(entry.id, at="./analysis.zarr")
```

The mount stores metadata and new analysis results at `./analysis.zarr` on the
machine running Python, including when that machine is a cloud worker. Counts
remain remote. Reopening through `catalog.mount_datastore` checks the source build identity.
If that build has changed, choose a new mount directory. Latest-only remote
storage cannot guarantee an immutable source during an already-open session;
a new `catalog.open_datastore` call revalidates the current published build.

A fresh mount copies cell and feature metadata, but does not copy the source
analysis artifacts. Keep using `ds` to plot the imported UMAP; `analysis_ds`
holds the new artifacts you create. `cytebase.embeddings(analysis_ds)` reports
only imported embeddings present in that local datastore.

The same search and plotting calls work for larger studies. A Tabula Sapiens
plot can be substantially slower and require more memory. Hide its long tissue
legend to preserve the plotting area:

```python
atlases = catalog.search("tabula sapiens", ready_only=True)
if atlases:
    atlas_ds = catalog.open_datastore(atlases[0]["cytebase_id"])
    atlas_umap_ref = cytebase.embedding(atlas_ds, "X_umap")
    atlas_ds.plots.embedding(
        layout=atlas_umap_ref, color_by="tissue", legend_loc="none", figsize=(8, 8)
    )
```

## Planned: agent-processed and annotated stores

A future Cytebase update will also host `data.zarr` stores processed and annotated
through Scarf's agent workflow. The aim is to let users connect to published
analysis results and annotations as well as the underlying counts, and mount
those stores for further analysis. This is a planned addition; the examples
above explore the currently imported CELLxGENE annotations and embeddings.

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
