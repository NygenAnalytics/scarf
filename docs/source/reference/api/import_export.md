# Import and export API reference

## Readers

```{eval-rst}
.. autoclass:: scarf.readers.CrReader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.CrH5Reader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.CrDirReader
    :members:
    :inherited-members:
    :show-inheritance:
```

```{eval-rst}
.. autoclass:: scarf.readers.MtxReader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.H5adReader
    :members:
```

```{eval-rst}
.. autoclass:: scarf.readers.CSVReader
    :members:
```

## H5AD inspection

Use this before `H5adReader` when you do not know which matrix and metadata keys an H5AD file uses.

`H5adReader` checks the matrix shape against the lengths of `obs` and `var` when it is
constructed, and a mismatch raises `ValueError`. Overriding `matrix_key`, for example with
`raw/X`, in the constructor or through `H5adReader.from_inspect`, therefore also needs the
matching `feature_attrs_key`, here `raw/var`. `inspect_h5ad(path, matrix_key="raw/X")` returns
reader arguments that pair them.

`H5adReader` decodes AnnData categorical columns and pandas nullable columns, including the
nullable string arrays that AnnData writes for dataframe indexes. By default it reads the index
that the dataframe's `_index` attribute names.
Missing categorical or object values become `None`; missing numeric values become `NaN`.
Unsupported group encodings and multi-dimensional columns are skipped with a warning.
`H5adToZarr` keeps missing rows under a linked missing mask and stores nullable booleans as
booleans.

Pass `embedding_roles` and `cluster_keys` to select analytical H5AD values for artifact import.
Selected `obsm` arrays and cluster labels are excluded from live metadata and returned as exact
refs by `H5adToZarr.dump()`. Other supported `obs` columns remain literal metadata.

CSC input is converted to temporary row storage on local disk. Pass `temp_dir` to
`H5adReader` or `H5adReader.from_inspect` to choose its parent directory; the default uses
the system temporary directory, including `TMPDIR`. The temporary files are removed when the
reader closes.

```{eval-rst}
.. autofunction:: scarf.inspect_h5ad
```

```{eval-rst}
.. autoclass:: scarf.H5adInspectResult
    :members:
```

```{eval-rst}
.. autoclass:: scarf.H5adImportResult
    :members:
```

## Matrix Market inspection

`inspect_mtx` reports every complete matrix, feature, and cell triplet in a supported source.
Pass one returned candidate to `MtxReader`.

```{eval-rst}
.. autofunction:: scarf.inspect_mtx
```

```{eval-rst}
.. autoclass:: scarf.readers.MtxCandidate
```

## Seurat import

Import a serialized Seurat object from an `.rds` file.
Inspect with `inspect_seurat`, open `SeuratReader`, then write with `SeuratToZarr`.
This path does not attach to a live R session and does not read or write `.h5seurat`.
Sidecar matrices such as BPCells directories and HDF5 files resolve inside `sidecar_root`, which
defaults to the directory of the `.rds` file. A reader opened from a stream needs an explicit
`sidecar_root` before it reads sidecar-backed layers.
See {doc}`../../tutorials/import_and_export` for the worked contract and {doc}`../../seurat` for
workflow mapping.

```{eval-rst}
.. autoclass:: scarf.SeuratReader
    :members:

.. autofunction:: scarf.inspect_seurat

.. autoclass:: scarf.SeuratInspectResult
    :members:

.. autoclass:: scarf.SeuratImportResult
    :members:

.. autoclass:: scarf.SeuratToZarr
    :members:
```

## Storage controls

Writers, merge, subset, and ``DataStore`` accept the same optional storage
controls. ``profile`` chooses the physical encoding. ``policy`` chooses paired
count-matrix geometry, and a writer uses it exactly. Without it, the imports
write the default geometry, and an import whose counts write or ``countsT``
transpose does not fit ``mem_budget`` raises ``MemoryError`` naming the largest
policy, with the default ``unitBytes`` and ``chunkBytes`` halved together, that
fits. A smaller geometry makes every later ``countsT`` read slower. Subset,
``repack_zarr``, merge, and ``add_grouped_assay`` instead fit the geometry to
``mem_budget``: they halve the default ``unitBytes`` and ``chunkBytes`` together
until their writes fit. Writers that choose their source batches (the sparse
and Seurat imports, and merge) admit batches of one destination row band, so a
geometry never leaves the write narrower batches than its bands. A write that
does not fit, with one-row count shards or with its explicit ``policy``, fails
before it creates the destination: writers raise when they are constructed,
and merge when it plans. ``MtxToZarr`` and ``CrToZarr`` take ``lines_in_mem`` in the
constructor, so the fit reserves the Matrix Market parse buffer that the write
uses. A resumed merge keeps the geometry of its completed counts. The geometry
never changes the store's identity. ``io`` overrides automatic read, compute,
and write widths. Unset values stay under automatic planning from
``mem_budget`` and ``nthreads``.

```{eval-rst}
.. autoclass:: scarf.storage.io_policy.StorageIoPolicy
    :members:

.. autoclass:: scarf.storage.count_matrix.CountMatrixPolicy
    :members:
```

### Count storage dtype

Every import stores each assay's counts in a dtype resolved from the values of
that assay, whatever the reader and source encoding, and no import takes a count
dtype argument. When every canonical (duplicate-summed) value is a non-negative
integer, the counts are stored in the narrowest of ``uint8``, ``uint16``,
``uint32``, and ``uint64`` that holds them. Other counts keep their source
dtype in native byte order, with ``float16`` read as ``float32``. Assays split from one source
matrix (10x HDF5 and Matrix Market feature types, H5AD ``assay_split_key``)
each resolve their own dtype. Each import reads every count once before it
creates the destination: ``CrReader`` subclasses and ``H5adReader`` report the
range of each group of features over the selected cells through
``count_value_ranges``, ``CSVReader`` records it as ``countRange`` in its first
pass, and ``SparseToZarr`` and ``SeuratToZarr`` scan their sources. Count
matrices hold finite values, so imports reject NaN and infinity before they
create the destination. Writers cast through a checked cast, so a count that
the stored dtype cannot hold raises instead of wrapping.

Subset and repack keep the source dtype. Merge stores the common type of the
source count dtypes, widened so that features summed by name cannot overflow
it, and rejects integer sources without a common integer dtype. Grouped and
melded assays keep their ``float64`` values.

```{eval-rst}
.. autofunction:: scarf.storage.count_dtype.count_storage_dtype

.. autoclass:: scarf.utils.count_values.CountValueRange
    :members:
```

## Writers

Every writer owns the reserved metadata columns `ids`, `names` and `I` and the
`__scarf_missing__` prefix of missing-value masks. Source cell or feature metadata columns
with these names are skipped with a warning, so imported identifiers are never replaced.
Zarr reads `/` and `\` in a column name as path separators, so writers store a source column
whose name contains either one under the name with `_` in their place and log the rename. A
source name that is already valid keeps its name; a renamed column whose name is taken gets
the first free `_2`, `_3`, and so on. `SeuratToZarr` raises for reserved names instead of
skipping them.
The H5AD, CSV, Matrix Market, and Cell Ranger readers reject missing or repeated cell and
feature IDs.

The cell column `<assay>_I` of every assay that an import writes is reserved for the membership
of that assay ({ref}`assay_membership`), because a merge reads a column of that name as which
cells the assay measured. `H5adToZarr` restores such a column as the membership of the assay
only when the file declares it in `uns["scarf"]["assayMembership"]`, which maps each assay to
its column, as Scarf's exports write it; the declared column must be boolean without missing
values, and a declaration that names another column or a column that `obs` lacks raises
`ValueError` before the destination is created. Every other source column named for an
imported assay, in an H5AD file without the declaration or from `CSVtoZarr`, `CrToZarr`, or
`MtxToZarr`, is skipped with a warning, as `I` is. `SeuratToZarr` writes the membership itself
and raises for a metadata column named for any imported assay. A column named for an assay that
the import does not write, such as `ADT_I` in an RNA-only import, stays an ordinary column.
`SparseToZarr` imports no cell metadata.

Import writers take their assay type from the assay name. Pass `assay_type` (or
`assay_types` on `CrToZarr`) to declare a custom-named assay as a preset such as `RNA`. A value
that is not a preset raises `ValueError`; use `Assay` for a generic assay. `DataStore(assay_types=...)`
and `add_melded_assay(assay_type=...)` apply the same rule. The presets and the persisted
`assayTypes` attribute are described in {ref}`assay_types_attribute`.

Writers create a new store and replace only a store that no `DataStore` has opened. `CrToZarr`,
`MtxToZarr`, `H5adToZarr`, `SparseToZarr`, `CSVtoZarr`, `SeuratToZarr`, and `SubsetZarr` accept an
absent path, an empty directory, or an empty store, and raise `FileExistsError` for a destination
that holds any key, such as an earlier store or a `.DS_Store` file. With `overwrite=True`
(`overwrite_existing_file=True` for `SubsetZarr`) they replace a store whose root group holds no
prepared assay, no `matrixSource`, and only members that Scarf writes (`cellData`, `matrices`,
`artifacts`, `pipeline`, assays, and workspaces of them).
A store that a `DataStore` has opened and content that Scarf did not write, also inside its groups
and arrays, are never replaced, and `SubsetZarr` refuses a destination that overlaps a source. A
local destination inside a Zarr store, such as `s.zarr/RNA` or `s.zarr/new.zarr`, given as a path
or a local store, and a destination whose root group is an assay group raise `ValueError`.
`CrToZarr`, `MtxToZarr`, `H5adToZarr`, and `SeuratToZarr` check their destination before they
read the counts, and `SubsetZarr` when it is constructed. Each checks again as it creates the store
with Zarr mode `"w-"`, which refuses keys that appeared since the check, or with mode `"w"` for the
store that `overwrite=True` replaces. A local path is opened as it is written, also when it holds
`#`, `?`, or `;`. `subset_assay_zarr` adds one count array to an existing store and replaces
nothing: `out_grp` must not exist and must not be, contain, or lie inside `in_grp`, and its parent
group must not already record the layout of another count matrix.

`SubsetZarr`, `DataStoreMerge`, `mount_datastore`, and `repack_store` refuse a source store that
holds a pending derived assay, one that `add_grouped_assay` or `add_melded_assay` has not
published, and raise `ValueError` naming it before they write anything. Once no process is
writing it, remove it with `discard_interrupted_assay` on a `DataStore` of its workspace.

```{eval-rst}
.. autofunction:: scarf.storage.destinations.check_destination

.. autofunction:: scarf.storage.destinations.create_destination

.. autofunction:: scarf.storage.destinations.refuse_pending_assays
```

```{eval-rst}
.. autoclass:: scarf.writers.CrToZarr
    :members:
```

`MtxToZarr` is an alias of `CrToZarr` for use with `MtxReader`.

```{eval-rst}
.. autoclass:: scarf.writers.MtxToZarr
    :members:
```

```{eval-rst}
.. autoclass:: scarf.writers.H5adToZarr
    :members:
```

```{eval-rst}
.. autoclass:: scarf.writers.SparseToZarr
    :members:
```

```{eval-rst}
.. autoclass:: scarf.writers.CSVtoZarr
    :members:
```

```{eval-rst}
.. autoclass:: scarf.writers.SubsetZarr
    :members:
```

```{eval-rst}
.. autofunction:: scarf.writers.to_h5ad
```

```{eval-rst}
.. autofunction:: scarf.writers.to_mtx
```

```{eval-rst}
.. autofunction:: scarf.writers.chunked_to_zarr
```

```{eval-rst}
.. autofunction:: scarf.writers.create_zarr_dataset
```

```{eval-rst}
.. autofunction:: scarf.writers.create_zarr_obj_array
```

```{eval-rst}
.. autofunction:: scarf.writers.create_zarr_count_assay
```

```{eval-rst}
.. autofunction:: scarf.writers.subset_assay_zarr
```

```{eval-rst}
.. autofunction:: scarf.writers.write_renorm_subset_to_zarr
```

## Selection and layout behavior

{py:meth}`scarf.datastore.datastore.DataStore.to_anndata` supports cell selection plus either `feature_names` or `feature_indexes`.
The two feature selectors are mutually exclusive.
`to_anndata` needs the optional `anndata` package and raises `ImportError` without it; install it
with `pip install 'scarf[extra]'`. `to_h5ad` writes with h5py, which Scarf always installs, and
needs no `anndata`, with or without a `run`.
`SubsetZarr` selects cells of one `DataStore` but retains every feature in each assay it writes.
It keeps the membership column `<assay>_I` of each assay it writes, for the selected cells, and
does not copy the membership column of an assay that `assays=` leaves out
({ref}`assay_membership`).

Without a `run`, `to_h5ad` and `to_mtx` export a complete assay.
For feature-selective disk export outside a pipeline run, call `to_anndata` and use AnnData's
writer.
A Matrix Market directory holds no cell metadata, so `to_mtx` cannot say which cells an assay
measured, and a cell that it did not measure would be written as a zero row that `MtxToZarr`
reads back as a measured cell. `to_mtx` therefore raises
{py:exc}`~scarf.metadata.membership.UnmeasuredCellsError` before it writes anything when the
assay's membership column `<assay>_I` marks a cell unmeasured ({ref}`assay_membership`). Export
the measured cells from a subset that `SubsetZarr(..., cell_key="<assay>_I")` writes, or use
`to_h5ad` or `to_anndata` without layers, which declare the membership.

`to_h5ad` streams the matrix into the file one block of rows at a time, so it never holds the
complete matrix, and it converts a dense block to CSR a bounded step of rows at a time, which the
block's stream charges against the memory budget. It writes a temporary file in the same directory
and moves it onto the target once complete, so a failed export leaves no partial file and keeps an
earlier file of the same name; a replaced file keeps its permissions. Without a `run`, text without missing values is a string array and
both indexes are stored as `_index`, as earlier releases wrote them.

Nullable metadata columns keep a stored placeholder in each row that their linked missing mask
flags. `to_anndata` and `to_h5ad` export those rows as missing values with or without `run`:
numeric columns become float64 with `NaN`, boolean columns become nullable booleans, and other
columns hold a missing value, which H5AD files store as a missing category.

Without `run`, `to_anndata` and `to_h5ad` export the membership column `<assay>_I` of the
exported assay as an ordinary `obs` column and declare it in `uns["scarf"]["assayMembership"]`,
for example `{"RNA": "RNA_I"}`, so `H5adToZarr` restores it as the membership of an imported
assay of that name and a merged store survives export, import, and a second merge. The membership
columns of other assays, including the assays of `to_anndata` `layers`, describe assays that the
file does not hold and are not exported, as a merge or subset that leaves an assay out does not
copy its membership column; an import would otherwise keep them as ordinary metadata, which a
later merge with `prepend_text=None` that includes those assays would refuse. A run's frozen fields
declare no membership. Because a layer of another assay cannot declare its membership, every
exported cell must be one that the layer's assay measured: `to_anndata` with such a layer, with or
without `run`, raises `UnmeasuredCellsError` before it reads any value, and a `cell_key` of the
cells that the layer's assay measured exports them. A layer of the exported assay is declared with
it.

Pass a completed run to write its frozen cells, feature universe, and result fields directly:

```python
scarf.to_h5ad(ds.RNA, "analysis.h5ad", run=run)
scarf.to_h5ad(ds.RNA, "analysis_normalized.h5ad", run=run, matrix="normed")
```

The assay must be the exact assay object owned by the datastore that opened the run. A run export
to a file holds exactly what {py:meth}`~scarf.datastore.datastore.DataStore.to_anndata` returns
for the same `run` and `matrix`, because one resolution of the run's rows, columns, and values
serves both: the run's cells, its frozen cell fields in `obs` with the consecutive UMAP fields in
`obsm["X_umap"]` and `clusters` in `obs`, and its frozen feature fields in `var`. `to_anndata`
holds that object in memory; `to_h5ad` streams it into the file. The run's plan decides which text
fields are categoricals: text that repeats or has missing values is a categorical whose categories
are in natural order (`d2` before `d10`, with a superscript or circled digit kept as text), and text
missing in every row is a categorical without categories. `to_anndata` returns those fields as
pandas categoricals with exactly those categories, and the file stores the same categories and
codes, so the object, the file, and a file that `AnnData.write_h5ad` writes from the object agree
whatever the AnnData version. Other text is a string array, and the indexes are named `ids` and
`gene_ids`. Earlier releases returned such text as plain strings and left the categories to
AnnData's writer, which sorts them with natsort and could not write text missing in every row;
this release stores `X` compressed, as a live export does, stores distinct text as a string array
where AnnData under pandas 3 wrote a nullable string array, and leaves out the empty `layers`,
`obsp`, `uns`, `varm`, and `varp` groups. `embeddings_cols`, feature-count recalculation, and a
writer-specific thread override apply only to ordinary full-assay export and are rejected when
`run` is supplied.

With `matrix="raw"`, the default, `X` holds the raw counts of the run's cells over its feature
universe. With `matrix="normed"`, `X` is the run's `normalized` artifact: the float32 values that
its PCA read, as {py:meth}`~scarf.DataStore.run_normalization` stored them for
the run, over the run's highly variable features in stored feature order. `var` then holds the
frozen fields of those features only, and `to_anndata` `layers` align to them. The values are read
in the artifact's stored row bands and never converted to float64. The export checks that the
store's dataset fingerprint still matches the artifact and raises `ValueError` when the run has
no normalized output, or when the artifact's cell selection is not the run's cell selection or its
feature selection is not the run's highly variable features. Earlier releases normalized again
for such an export, with the live size factor, no log transform, and no renormalization over the
selected features, over the whole feature universe, so its `X` never equaled the values that the
run computed. Without a `run`, `to_anndata(matrix="normed")` still normalizes the selected cells
and features now with the assay's `normed`, and `to_h5ad` rejects `matrix="normed"`.

`H5adToZarr.dump()` returns an `H5adImportResult` containing the written assays, analysis assay,
all-cell selection, and imported embedding and clustering refs. Set `analysis_assay` when a
multi-assay import selects analytical values. Use `DataStore.load_artifact(ref)` for payload access
or pass the exact ref to a consumer. Import does not flatten these results into metadata columns.

Ordinary `to_h5ad` export writes a complete assay and live metadata. Run-aware export reads only
the completed run's frozen selections and fields, and for normalized values its `normalized`
artifact, so export does not require physical result columns.

`CrToZarr`, `MtxToZarr`, `H5adToZarr`, `SparseToZarr`, and `SeuratToZarr` select source batch
rows automatically when `batch_size` is omitted, starting from the smallest destination
row-shard height, which the fitted count layout admits. An explicit positive `batch_size` is
capped at that height.

## Merge

Use `DataStoreMerge` to merge DataStores.
Pass `assays=["RNA"]` when only one assay type is needed.
Features are matched by exact feature ID by default. Gene symbols are display labels and do not
merge distinct IDs; suffixes such as `_1` are preserved. An assay present in multiple inputs
raises an error if none of its IDs overlap.
When inputs use different identifier conventions for the same genes, for example Ensembl IDs in
one input and gene symbols in another, pass `feature_key="names"` to match features by name.
The merged feature IDs are then the names, and features that share a name within one input are
summed.
Feature annotation columns are merged when the inputs agree. A column whose values differ for a
shared feature, such as per-dataset highly variable gene flags, is left out with a warning. The same
applies when features that share a name within one input disagree.
Merged cell IDs are `{name}__{cell_id}`, so source names cannot contain `__`. Cell metadata columns
keep their attributes when the inputs agree. Differing `levels` of an unordered categorical column
become their union in first-seen order; any other differing attribute, including ordered levels, is
dropped with a warning.
Each merged assay keeps the type that its sources declare in `assayTypes`, such as `HTO` or
`GeneActivity`, rather than the preset of its assay class, so a merged HTO assay can be
demultiplexed. Sources that declare different types for one assay are rejected before anything is
written; reopen the source whose declaration is wrong with `zarr_mode="r+"` and a matching
`assay_types`.
RNA assays write both `counts` and a gene-major `countsT` copy, which roughly doubles stored counts for those assays.
Non-RNA assays never write `countsT`.
Every merged assay gets the membership column `<assay>_I` described in {ref}`assay_membership`. A
merged cell keeps its source's membership: False when the source lacks the assay, True when the
source measured every cell with it, and the source's own value when the source records partial
membership. A source's membership column is not merged again as an ordinary column such as
`orig_<assay>_I`, and the membership column of a source assay that `assays=` leaves out is not
merged at all, so a merge with `assays=["RNA"]` writes no `ADT_I` or `orig_ADT_I` that a later
merge including ADT would trip over. Planning rejects a malformed source membership column, an
ordinary column whose merged name equals a membership column, and a source whose cells outside an
assay have counts of it. Merges whose sources recorded partial membership before this release
marked those cells as members; redo them from the original sources. A source column named
`<assay>_I` for a merged assay that lacks the membership attributes raises `Cell column
'<assay>_I' is reserved for the membership of assay ...`. Such a column comes from an import by an
earlier release, which kept the column of an exported H5AD file as ordinary metadata, or from a
column inserted by hand before this release. Import the file again with this release, which
restores a declared membership and otherwise skips the column with a warning, or drop the plain
column from the source with `source.cells.drop("<assay>_I")`, after which the source counts every
cell of the assay as measured. `MetaData` writes no membership column itself: `insert`,
`update_key`, and `reset_key` refuse `<assay>_I` for any assay of the store, and `drop` refuses one
that carries the membership role ({ref}`assay_membership`).
The destination must be empty or hold a merge written by `DataStoreMerge`. A destination that
aliases, contains, or lies inside a source store, or that already holds other content, is refused.
`overwrite=True` replaces only a merge-owned destination whose assays have not been prepared by
opening it as a `DataStore`, and it clears the recorded default assay.
Interrupted merges resume at whole-component boundaries (`cellData`, each assay `counts`, and each RNA `countsT`) rather than mid-matrix.
A resume requires the same configuration, the same source counts, assay types, and membership; a
completed `countsT` whose layout differs from the plan is not rewritten in place. The manifest
records `assayTypes` and `sourceMembership`, so a merge interrupted under an earlier release
restarts with `overwrite=True`. Completed counts keep the layout persisted
with them, so a resume under another `mem_budget` reuses them, while counts that a resume rewrites
get a layout fitted to its budget.

```{eval-rst}
.. autoclass:: scarf.merge.DataStoreMerge
    :members:

.. autoclass:: scarf.merge.MergePlan
    :members:

.. autoclass:: scarf.merge.AssayMergePlan
    :members:

.. autoclass:: scarf.merge.MergeResult
    :members:

.. autoclass:: scarf.merge.ComponentResult
    :members:
```
