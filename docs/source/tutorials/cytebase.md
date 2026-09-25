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

**Development preview:** the saved results on this page were generated against
a development bucket. The SDK now defaults to the public `Nygen/cytebase`
bucket, which readers can access without credentials. Available datasets may
differ from this saved snapshot; a configured bucket must contain the example
dataset to rerun the full walkthrough.

{nb-download}`Download the executed Jupyter notebook <cytebase.ipynb>`.

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

- Search by text or exact ontology labels and run local catalog SQL
- Open an RNA assay read-only and read selected cell metadata
- Plot imported UMAP coordinates by cell type and gene expression
- Export coordinates for custom figures and mount a dataset for writable analysis

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

The example is the small Solé-Boldo human skin dataset. Search its author name,
then keep the returned identifier rather than constructing an ID yourself.

```{code-cell} ipython3
matches = catalog.search("soleboldo", ready_only=True, max_cell_chars=None)
if not matches:
    raise RuntimeError("The example skin dataset is not ready in this bucket")
matches
```

```{code-cell} ipython3
dataset_id = matches[0]["cytebase_id"]
dataset = catalog.dataset(dataset_id)
dataset
```

Displaying this handle reads the catalog and dataset record. It does not open
the remote Zarr hierarchy yet. The citation and CELLxGENE links identify the
original study; retain these when using the data.

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

## 2. Open the remote DataStore

`open()` validates the published dataset version and build receipt, then opens
the store with `zarr_mode="r"`. Counts remain remote and the SDK does not write
back to the bucket. Repeated calls on the same handle reuse this DataStore.

```{code-cell} ipython3
ds = dataset.open()
ds
```

Small Zarr metadata objects are cached in memory for this open store. Count
chunks are fetched when needed; the SDK does not maintain a persistent local
count cache. Opening and plotting still depend on network latency.

The source H5AD may contain more embeddings than the pipeline imported. Check
the imported keys before plotting:

```{code-cell} ipython3
print("Source embedding keys:", dataset.source_embeddings())
print("Imported embedding keys:", sorted(dataset.embeddings()))
```

The current pipeline imports `obsm/X_umap`. An `X_pca` key in the source list
alone does not mean its coordinates are available in the Scarf store.

## 3. Read cell metadata

CELLxGENE annotations are stored beside Scarf's QC columns, such as
`RNA_nCounts` and `RNA_nFeatures`. Request only the columns needed for a figure:

```{code-cell} ipython3
wanted = ["cell_type", "tissue", "disease", "donor_id", "sex", "assay"]
columns = [column for column in wanted if column in ds.cells.columns]
meta = dataset.cell_metadata(["ids", *columns]).set_index("ids")
meta.head()
```

This materializes the selected metadata columns as a pandas DataFrame, not the
expression matrix. On a large atlas, metadata and coordinates still require
memory proportional to the selected cells.

```{code-cell} ipython3
meta["cell_type"].value_counts().rename("cells").to_frame()
```

```{code-cell} ipython3
if "donor_id" in meta:
    display(pd.crosstab(meta["cell_type"], meta["donor_id"]))
```

## 4. Plot the stored UMAP

`plot_embedding` uses the imported coordinates without recomputing UMAP.
Additional options are forwarded to `DataStore.plots.embedding`.
Give categorical legends enough space; a large atlas with many labels is
better viewed with selected groups or `legend_loc="none"`.

```{code-cell} ipython3
cell_type_plot = dataset.plot_embedding(color_by="cell_type", figsize=(10, 6))
```

Focus on the five most common cell types without changing the source dataset:

```{code-cell} ipython3
top_types = meta["cell_type"].value_counts().head(5).index.tolist()
dataset.plot_embedding(
    color_by="cell_type", groups=top_types, figsize=(10, 6)
);
```

### Gene expression

Gene names come from the published assay feature metadata. Resolve the symbols
against that table first; not every requested gene is present in every dataset.
Here the explicit lookup handles differences in letter case.

```{code-cell} ipython3
requested_genes = ["PTPRC", "EPCAM", "COL1A1", "PECAM1"]
by_upper = {str(name).upper(): str(name) for name in ds.RNA.feats.fetch_all("names")}
genes = [by_upper[name.upper()] for name in requested_genes if name.upper() in by_upper]
print("Found:", genes)
print("Missing:", [name for name in requested_genes if name.upper() not in by_upper])
```

Expression values are normalized on read; this example explicitly requests a
`log1p` transform. Neither the count matrix nor the imported coordinates are
modified. Drawing high values last makes expressing cells easier to see.

```{code-cell} ipython3
if genes:
    dataset.plot_embedding(
        color_by=genes[:2],
        normalization=NormalizationSpec(transform="log1p"),
        sort_values=True,
        n_columns=2,
        figsize=(12, 5),
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

Coordinates are indexed by cell ID. Join on those IDs instead of assuming the
same row order as a separate metadata table:

```{code-cell} ipython3
coords = dataset.embedding_coordinates()
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
analysis = dataset.mount("./analysis.zarr")
```

The mount stores metadata and new analysis results at `./analysis.zarr` on the
machine running Python, including when that machine is a cloud worker. Counts
remain remote. Reopening through `dataset.mount` checks the source build identity.
If that build has changed, choose a new mount directory. Latest-only remote
storage cannot guarantee an immutable source during an already-open session;
a fresh dataset handle revalidates the current published build.

The same search and plotting calls work for larger studies. A Tabula Sapiens
plot can be substantially slower and require more memory. Hide its long tissue
legend to preserve the plotting area:

```python
atlases = catalog.search("tabula sapiens", ready_only=True)
if atlases:
    atlas = catalog.dataset(atlases[0]["cytebase_id"])
    atlas.plot_embedding(color_by="tissue", legend_loc="none", figsize=(8, 8))
```

## Planned: agent-processed and annotated stores

A future Cytebase update will also host `data.zarr` stores processed and annotated
through Scarf's agent workflow. The aim is to let users connect to published
analysis results and annotations as well as the underlying counts, and mount
those stores for further analysis. This is a planned addition; the examples
above explore the currently imported CELLxGENE annotations and embeddings.

## Common issues

- **No matching dataset:** check bucket selection, access, and whether the desired
  version is ready. `ready_only=False` includes unfinished registrations.
- **Missing embedding:** use `dataset.embeddings()` to check imported coordinates.
  `source_embeddings()` also lists source keys that were not imported.
- **Crowded figure:** increase `figsize`, reduce the number of groups or panels,
  or hide the legend. The examples use explicit sizes instead of suppressing
  plotting warnings.
- **Unexpected expression costs:** count blocks are remote, and a single-gene
  request may read a larger storage chunk. Start with a small dataset and a few genes.
- **Writing to a read-only store:** use a local mount for analysis that saves results.

See the [Cytebase API reference](../reference/api/cytebase.md) for the full SDK
and [Remote stores](remote_stores.md) for mounted analysis mechanics. Saved
documentation outputs are a snapshot; rerunning checks the current catalog and
may produce different discovery results as the bucket changes.
