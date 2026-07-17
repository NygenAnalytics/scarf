---
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

(data_organization)=

# Data organization

Scarf keeps counts, metadata, graphs, and analysis results in a Zarr store. This chapter shows
where those values live and how to inspect or update cell and feature metadata.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- Basic familiarity with cell and feature metadata

## What you will learn

- Inspect the Zarr hierarchy
- Read and write cell or feature metadata
- Locate cached graph and marker results

## Dataset

The following figure shows the relationship between a `DataStore`, its assays, and metadata.
`DataStore` is the main object used to access data and analysis methods. Each assay has its own
feature metadata, normalization, and feature-selection methods.

+++

<img src="https://raw.githubusercontent.com/parashardhapola/scarf/master/docs/source/_static/scarf_organization.png" alt="Scarf Class design" width="800px">

```{code-cell} ipython3
import scarf
scarf.__version__
```

This chapter uses the prepared CITE-seq store from {doc}`cite_seq`.

```{code-cell} ipython3
scarf.fetch_dataset(
    dataset_name='tenx_8K_pbmc_citeseq',
    save_path='scarf_datasets',
    as_zarr=True
)
```

```{code-cell} ipython3
ds = scarf.DataStore(
    'scarf_datasets/tenx_8K_pbmc_citeseq/data.zarr',
    nthreads=4
)

ds.mark_hvgs(min_cells=20, top_n=500)
ds.make_graph(feat_key='hvgs', k=11, dims=15)
ds.run_leiden_clustering(resolution=0.5)
ds.run_marker_search(group_key='RNA_leiden_cluster', gene_batch_size=100)
```

## Guided steps

### 1. Inspect Zarr trees

Scarf uses [Zarr format](https://zarr.readthedocs.io/en/stable/) to store raw counts as dense, chunked matrices. The Zarr file is in fact a directory that organizes the data in form of a tree. The count matrices, cell and features attributes, and all the data is stored in this Zarr hierarchy. Some of the key benefits of using Zarr over other formats like HDF5 are:
- Parallel read and write access
- Availability of compression algorithms like [LZ4](https://github.com/lz4/lz4) that provides very high compression and decompression speeds.
- Automatic storage of intermediate and processed data

In Scarf, the data is always synchronized between hard disk and RAM and as such there is no need to save data manually or synchronize the store by hand.

We can inspect how the data is organized within the Zarr file using `show_zarr_tree` method of the `DataStore`. 

```{code-cell} ipython3
ds.show_zarr_tree(depth=1)
```

By setting `depth=1` we get a look of the top-levels in the Zarr hierarchy: 'RNA', 'ADT' and 'cellData'. The top levels here are hence composed of two assays 
and the `cellData` level, which will be explained below. Scarf attempts to store most of the data on disk immediately after it is processed. Since, that data that we have loaded is already preprocessed, below we can see that the calculated cell attributes can now be found under the `cellData` level. The cell-wise statistics that were calculated using 'RNA' assay have 'RNA' preppended to the name of the column. Similarly the columns stating with 'ADT' were created using 'ADT' assay data.

```{code-cell} ipython3
ds.show_zarr_tree(start='cellData')
```

**The `I` column** : This column is used to keep track of valid and invalid cells. The values in this column are boolean indicating which cells where filtered out (hence False) during filtering process. One of can think of column `I` as the default subset of cells that will be used for analysis if any other subset is not specifially chosen explicitly by the users. Most methods of `DataStore` object accept a parameter called `cell_key` and the default value for this paramter is `I`, which means that only cells with True in `I` column will be used.

+++

If we inspect one of the assay, we will see the following zarr groups. The raw data matrix is stored under `counts` group (more on it in the next section). `featureData` is the feature level equivalent of `cellData`. `featureData` is explained in further detail in the next section. The `markers` level will be explained further in the last section. `summary_stats_I` contains statistics about features. These statistics are generally created during feature selection step. 'normed__I__hvgs' is explained in detail in the fourth section of this documentation.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA', depth=1)
```

The feature wise attributes for each assay are stored under the assays `featureData` level. The output below shows some of the feature level statistics.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/featureData', depth=1)
```

### 2. Inspect cell and feature attributes

The cell and feature level attributes can be accessed through `DataStore`, for example, using `ds.cells` and `ds.RNA.feats` respectively (both are objects of `Metadata` class). In this section we dive deeper into these attribute tables and try to perform [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) operations on them. 

+++

`head` provides a quick look at the attribute tables. 

```{code-cell} ipython3
ds.cells.head()
```

The feature attribute table from any assay can similarly be quickly inspected

```{code-cell} ipython3
ds.RNA.feats.head()
```

Even though the above 'head' command may make you think that `ds.cells` and `ds.RNA.feats` are Pandas dataframe, they in fact are not. If you wish to obtain a full table as Pandas dataframe, then you can export the columns of your choice as shown below.

```{code-cell} ipython3
ds.cells.to_pandas_dataframe(
    columns=['ids', 'RNA_nCounts', 'RNA_nFeatures', 'RNA_leiden_cluster']
).set_index('ids')
```

If you wish to export all the columns as a dataframe then simply provide `ds.cells.columns` or `ds.RNA.feats.columns` as an argument rather than a list of columns. If you are interested in one of the columns then it can fetched using either `fetch` or `fetch_all` command.

`fetch` will provide values for a subset of cells (by default, only for those that have True value in column `I`, but any other boolean column can be given).

```{code-cell} ipython3
cluster_labels = ds.cells.fetch('RNA_leiden_cluster')
cluster_labels.shape
```

`fetch_all` will return values for all the cells.

```{code-cell} ipython3
cluster_labels_all = ds.cells.fetch_all('RNA_leiden_cluster')
cluster_labels_all.shape
```

If you wish to add a new column then `insert` command can be used for either cell or feature attributes. `insert` method will take care of inserting the values in the right rows even when a subset of values are provided. The default value for `key` parameter is `I` in `insert`method, so it will add values in same order as cells that have True value for `I`

```{code-cell} ipython3
is_cluster_1 = cluster_labels == 1
ds.cells.insert(
    column_name='is_cluster_1',
    values=is_cluster_1
)
```

If we try to add values for a column that already exists then Scarf will throw an error. For example, if we simply rerun the command above, we should get an error

```{code-cell} ipython3
:tags: [raises-exception]

ds.cells.insert(
    column_name='is_cluster_1',
    values=is_cluster_1
)
```

To override this behaviour, you can use 'overwrite' parameter and set it to True

```{code-cell} ipython3
ds.cells.insert(
    column_name='is_cluster_1',
    values=is_cluster_1,
    overwrite=True
)
```

Please checkout the API of the `Metadata` class to get information on how to perform other operations like delete and update columns.

+++

### 3. Inspect count matrices and normalization

+++

Scarf uses Zarr format so that data can be stored in rectangular chunks. The raw data is saved in the `counts` level within each assay level in the Zarr hierarchy. It can easily be accessed as a chunked array using the `rawData` attribute of the assay. This chunked array exposes a NumPy-like interface (indexing, reductions, ufuncs) but evaluates lazily by streaming over row chunks, which keeps the memory requirement bounded regardless of dataset size. Note that for a standard analysis one would not interact with the raw data directly. Scarf internally optimizes the use of this chunked array to minimize the memory requirement of all operations.

```{code-cell} ipython3
ds.RNA.rawData
```

The normalized data can be accessed through the `normed` method of the assay. In Scarf a user doesn't need to perform normalization step manually, Scarf only stores the raw data and generates normalized data whenever needed. This means that Scarf may need to perform normalization several times. However, in practise we have noted that time spent in normalizing the data is only a small fraction of routine workflows. 

```{code-cell} ipython3
ds.RNA.normed()
```

Users can override how the normalization is performed in Scarf. Normalization is performed using the `normMethod` attribute which references the function responsible for performing the normalization.

```{code-cell} ipython3
ds.RNA.normMethod
```

Let's checkout the source of function that is referenced by ds.RNA.normMethod

```{code-cell} ipython3
import inspect 

print (inspect.getsource(ds.RNA.normMethod))
```

Following is an example of how one can override the method of normalization

```{code-cell} ipython3
def my_cool_normalization_method(assay, counts):
    import numpy as np

    # Calculate total counts for each cell
    lib_size = counts.sum(axis=1).reshape(-1, 1) 
    
    # Library size normalization followed by log2 transformation
    return np.log2(counts/lib_size)

ds.RNA.normMethod = my_cool_normalization_method
```

Now whenever Scarf internally requires normalized values, this function will be used. Scarf provides a dummy normalization function (`scarf.assay.norm_dummy`) that does not perform normalization. This function can be useful if you have pre-normalized data and need to disable default normalization.

```{code-cell} ipython3
ds.RNA.normMethod = scarf.assay.norm_dummy
```

Please note: If you are using a custom function or disabling normalization, then everytime you load a DataStore object you will need to reassign `normMethod` to the function of your choice.

+++

### 4. Inspect graph caching

+++

All the results of `make_graph` step are saved under a name on the form '*normed\_\_{cell key}\_\_{feature key}*' (placeholders used in brackets here). In this case, since we did not provide a cell key it takes default value of `I`, which means all the cells that were not filtered out. The feature key (`feat_key`) was set to `hvgs`. The Zarr directory is organized such that all the intermediate data is also saved. The intermediate data is organized in a hierarchy which triggers recomputation when upstream changes are detected. The parameter values are also saved in hierarchy level names. For example, 'reduction_pca_31_I' means that PCA linear dimension reduction with 31 PC axes was used and the PCA was fit across all the cells that have `True` value in column **I**.

```{code-cell} ipython3
ds.show_zarr_tree(start='RNA/normed__I__hvgs')
```

The graph calculated by `make_graph` can be easily loaded using the `load_graph` method, like below. The graph is loaded as a sparse matrix of the cells that were used for creating a graph.

Next, we show how the graph can be accessed if required. However, as stated above, normally Scarf handles the graph loading internally where required. 

Because Scarf saves all the intermediate data, it might be the case that a lot of graphs are stored in the Zarr hierarchy. `load_graph` will load only the latest graph that was computed (for the given assay, cell key and feat key). 

```{code-cell} ipython3
ds.load_graph(
    from_assay='RNA',
    cell_key='I',
    feat_key='hvgs',
    symmetric=False,
    upper_only=False
)
```

The sparse matrix above is the latest graph. To use a graph built with other parameters,
run `make_graph` again with those settings. If that graph already exists, Scarf skips
recalculation and marks it as the latest graph. Load it with `load_graph`.

+++

### 5. Fetch marker features

+++

Marker features (ex. marker genes) for a selection of cells can be calculated using `run_marker_search`. The markers for each group are stored under 'marker' group under the assay's group. Within the '{assay}/markers' level there are sublevels (only 1 here) created based on which cell set and cell groups were used: {cell_key}__{group_key}. Stored below this Zarr level are the individual group labels that contain the maker feature ids and their corresponding scores.

```{code-cell} ipython3
ds.show_zarr_tree(
    start='RNA/markers',
    depth=2
)
```

Marker list for any group (e.x. cell cluster) can be fetched like below

```{code-cell} ipython3
ds.get_markers(
    group_key='RNA_leiden_cluster',
    group_id='1',
    min_score=0.1,
    min_frac_exp=0.1
)
```

One can also export the names of markers for all the groups to a CSV file like below:

```{code-cell} ipython3
ds.export_markers_to_csv(
    group_key='RNA_leiden_cluster',
    csv_filename='test.csv',
    min_score=0.2,
    min_frac_exp=0.1
)
```

---
### 6) Zarr v3 and sharded count arrays

Scarf 0.33+ writes new datasets as Zarr v3. Existing v2 stores remain readable; new metadata written into a v2 store still uses v2-compatible codecs.

Count matrices created by the writers are finalized as sharded arrays (default profile `fast_local`: chunks 512×512, shards 4096×4096). Inspect layout with `show_zarr_tree`, which prints chunk and shard info for arrays at each node.

Set the storage profile with the `SCARF_ZARR_PROFILE` environment variable (`fast_local` or `cloud`) or the `zarrProfile` argument when opening a `DataStore`. To convert an older store to v3 with sharding:

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local
```

### Memory budget and remote stores

Bound streaming memory when opening a store:

```python
ds = scarf.DataStore('path/to/data.zarr', mem_budget='8G', nthreads=4)
```

For object-storage backed Zarr, use `zarrProfile='cloud'` or set `SCARF_ZARR_PROFILE=cloud`. Pass `local_cache=True` to `make_graph` to stage normalized data locally before PCA and KNN, which reduces repeated remote reads.

After Harmony batch correction, corrected embeddings are stored under the PCA reduction group as `harmonizedData` with attribute `isHarmonized=True`.

### Metadata query helpers

Subset cells programmatically without exporting full tables:

```python
# Indices of cells whose label matches one of the targets (string match).
idx = ds.cells.get_index_by(['3'], 'RNA_leiden_cluster')
# Boolean mask for a numeric column between bounds (exclusive by default).
keep = ds.cells.sift('RNA_nCounts', min_v=1000, max_v=15000)
```

## Common mistakes

- Expecting filtered cells to be deleted instead of marked `False` in `I`
- Treating `Metadata` as an in-memory pandas DataFrame
- Fetching a column with `fetch` when values for inactive cells are also required

## Saved results

Metadata changes, graph caches, and marker results are written to the Zarr store. Exported marker
tables are written to the path passed to `csv_filename`.

## Next steps

- {doc}`import_and_export`
- {doc}`plotting`
- {doc}`dimensionality_reduction_and_clustering`
