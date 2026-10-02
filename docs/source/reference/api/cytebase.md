# Cytebase API reference

The Cytebase SDK connects directly to cloud-hosted Scarf DataStores for exploration and
analysis without downloading a complete dataset first. Search the catalog, open a shared
store read-only, or mount it for writable analysis with remote counts. Install the `cytebase`
extra described in {doc}`../../installation` and follow {doc}`../../tutorials/cytebase` for
an executable walkthrough. Its {ref}`example notebooks <cytebase_example_notebooks>` show
longer use cases.

Use the same SDK from a local notebook or cloud compute. Operations read metadata and count
blocks over the network and run in your Python environment; the SDK does not provision compute.

## Configure a catalog

```python
from scarf.cytebase import Catalog

catalog = Catalog()
```

`Catalog()` defaults to the public `Nygen/cytebase` bucket. Public readers do not
need credentials, and dataset stores are opened read-only. To use another bucket,
set `CYTEBASE_BUCKET` or pass `Catalog(bucket=...)`; the explicit argument takes
precedence over the environment variable. Accepted forms are `namespace/name` and
`hf://buckets/namespace/name`. The default `token=None` uses Hugging Face credentials
when available and anonymous access otherwise; `token=False` explicitly selects
anonymous access. This SDK default does not apply to the ingestion pipeline,
which still requires an explicit `CYTEBASE_BUCKET` for its writes.

Construction downloads or reuses a verified local DuckDB catalog. Each new query checks its
published SHA-256 checksum and refreshes the local copy when needed. Queries therefore require
network access even when a cached catalog exists. Counts remain remote until an operation reads
them. `connect_catalog()` exposes a read-only DuckDB connection; close it after use, for example
with a `with` block.

## Discover datasets

`search(text)` matches every whitespace-separated search word, ignoring case, across dataset
IDs, titles, citations, first authors, and facet labels. Its default result limit is 50;
`limit=None` removes that limit. `find_datasets(**facets)` matches exact labels, combining
different facets with AND and values within a list with OR. Use `list_terms(facet)` to discover
the available labels before filtering.

Supported facets are `tissue`, `organ`, `disease`, `assay`, `organism`, `cell_type`, `sex`,
`development_stage`, and `suspension_type`. Both search methods default to `ready_only=True`,
which selects datasets with a ready store for their latest registered version. Pass
`ready_only=False` to include other registered datasets. Results are ordered by descending cell
count, then Cytebase ID. `list_terms()` summarizes terms across the registered catalog, including
datasets that are not ready.

```python
catalog.list_terms("tissue")
matches = catalog.find_datasets(organism="Homo sapiens", tissue=["lung", "blood"])
catalog.search("lung", limit=10)
```

`query(sql, parameters=None)` supports parameterized SQL over the local catalog. The
`datasets` table holds dataset rows, and `dataset_terms` holds facet labels and ontology IDs.

```{eval-rst}
.. autoclass:: scarf.cytebase.Catalog
   :members: connect_catalog, query, find_datasets, search, dataset, list_terms, open_datastore, mount_datastore
```

## Inspect a catalog entry

`catalog.dataset(cytebase_id)` returns a `DatasetEntry` containing metadata about a scientific
dataset. Its `row` dictionary contains the complete catalog row; `id`, `title`, `citation`,
and `cell_count` provide convenient access to common
fields. Displaying the entry in a notebook renders `describe()`, a Markdown summary that can
fetch the published dataset record. Entries are metadata-only: they do not open a datastore,
read cell arrays, mount an analysis, or plot.

`record()` fetches the published `dataset.json` and caches it on the entry.
`source_embeddings()` lists the source H5AD's embedding keys recorded in that JSON. The source
list describes what was inspected and may include keys whose coordinates were not imported.

```{eval-rst}
.. autoclass:: scarf.cytebase.DatasetEntry
   :members: id, title, citation, cell_count, describe, record, source_embeddings
   :undoc-members:
```

## Open a datastore

```python
from scarf import cytebase

entry = catalog.dataset(matches[0]["cytebase_id"])
ds = catalog.open_datastore(entry.id)
umap_ref = cytebase.embedding(ds, "X_umap")
ds.plots.embedding(layout=umap_ref, color_by="cell_type", figsize=(10, 6))
```

`open_datastore(cytebase_id, **datastore_options)` checks the current published provenance and
returns a read-only `DataStore`. Each call opens and validates a datastore; retain the returned
`ds` for subsequent operations. Published cell annotations and imported embeddings are source
results, so displaying them does not recompute an analysis.

Read cell metadata through `ds.cells.to_pandas_dataframe(columns, key="I")`. This returns a
pandas DataFrame for the cells selected by `I`. Select the needed columns to limit metadata
reads. Plot through `ds.plots.embedding(layout=ref, ...)`, using an exact embedding artifact
reference. See {doc}`plotting` for its controls and returned `PlotResult`.

`mount_datastore(cytebase_id, at, **datastore_options)` creates or reopens a writable local
`DataStore` with remote counts pinned to one verified source build. The destination must be a
local path. Keep its adjacent
`.cytebase.json` receipt with the mount; a changed source build requires a new destination.
Mounting keeps counts remote and requires network access for subsequent reads. See
{doc}`../../tutorials/remote_stores` for working with remote matrices and local results.

A mount copies cell and feature metadata and resolves the published source artifacts, such as
imported embeddings, read only. The embedding helpers and plots therefore work on the mount as
on the read-only datastore. New analysis artifacts are written to the mount.

## Resolve imported embeddings

`cytebase.embeddings(ds, *, assay="RNA")` maps imported source embedding keys to exact
`ArtifactRef` values in the open datastore. `cytebase.embedding(ds, key="X_umap", *, assay="RNA")`
selects one of these references. These helpers inspect the imported artifacts the datastore
resolves, which on a mount include its source's.

`cytebase.embedding_coordinates(ds, ref)` returns a DataFrame of the complete referenced
embedding, indexed by cell ID. Row alignment follows the artifact's frozen cell selection,
including when the live `I` column has changed. Join coordinates to metadata by cell ID for
custom plotting. Metadata tables and coordinate exports materialize the requested columns or
coordinates in memory; they do not materialize the complete count matrix.

```{eval-rst}
.. autofunction:: scarf.cytebase.embeddings

.. autofunction:: scarf.cytebase.embedding

.. autofunction:: scarf.cytebase.embedding_coordinates
```

## Prerelease API change

This prerelease replaces `CytebaseDataset` with the metadata-only `DatasetEntry`.
`Catalog.open_datastore` and `Catalog.mount_datastore` replace `Catalog.open_dataset` and
`Catalog.mount_dataset`. The former dataset-handle methods for opening, mounting, metadata,
embeddings, and plotting are removed. These names and methods have no compatibility aliases.
Update existing notebooks to keep the catalog entry in `entry`, open a `DataStore` as `ds`,
and use the datastore and module-level helpers shown above.

## Work with result tables

Catalog discovery and SQL methods return `CatalogResults`, a list-compatible collection of
complete row dictionaries. Indexing and iteration expose full values; slicing produces a plain
list. Notebook output displays at most 20 rows and shortens cells to 100 characters by default.
These display limits do not remove rows or shorten values in the underlying dictionaries.

```python
matches[0]["cytebase_id"]
matches.to_markdown(columns=["cytebase_id", "title"], max_rows=None, max_cell_chars=None)
```

Pass `max_cell_chars=None` to a catalog query to show complete cell values in its default
notebook representation. Use `to_markdown()` to choose displayed columns and row limits.

```{eval-rst}
.. autoclass:: scarf.cytebase.display.CatalogResults
   :members: to_markdown
```

## Public example repositories

The repository interface lists and downloads Scarf's public example files. It uses repository
and dataset directory names, while `Catalog` discovers catalog datasets by Cytebase ID.
The examples in other tutorials use `connect("scarf_docs")` and
`Repository.download_dataset(..., zarr=True)` to download and extract prepared Scarf stores.

```{eval-rst}
.. autofunction:: scarf.cytebase.list_repositories

.. autofunction:: scarf.cytebase.connect

.. autoclass:: scarf.cytebase.Repository
   :members:
```
