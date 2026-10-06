(zarr_internals)=
# Zarr internals

This page is for contributors and advanced users who need the on-disk layout.
Analysts should start with {doc}`../tutorials/data_organization`.
Why RNA stores two orientations, and what to do when an older store will not open, is in {doc}`../concepts/memory_and_execution`.

## Layout overview

A Scarf Zarr store is a directory hierarchy.
Typical top-level groups include:

- cell metadata
- one or more assay groups (for example `RNA`, `ADT`, `ATAC`)
- per-assay feature metadata
- immutable feature selections, summaries, normalized matrices, reductions, and graph artifacts under assay-specific artifact paths
- durable pipeline records under `pipeline/runs/{runId}` with ordered stage records below each run

Artifact group names use kind and identity, not encoded graph-stage parameters.
Prefer inspecting a store with `zarr.open` or Scarf's `DataStore` summary rather than hard-coding internal paths in analysis scripts.

Analytical outputs are artifacts only. Feature selection, embedding, clustering, score, and marker
operations leave metadata tables unchanged. Consumers validate exact refs,
artifact completion, lineage, and ordered-axis identity.

A producer creates an artifact group with `complete=False` at a path that ends in its random
256-bit `artifact_id`, so the path belongs to that producer alone. After it validates the payload,
one metadata write sets `complete=True` and publishes the artifact. The producer deletes the group
after a failure only before that write; once the write is issued, the group stays, complete or
not, because the write can persist even when it reports an error.

Producers check the arrays that must hold only finite values as they write them
(`scarf.storage.finite_values`): a NaN or infinite value raises `NonFiniteArtifactError`, a
`ValueError` that names the operation, the array, and the row, and nothing is published.

(normalized_feature_statistics)=

### Normalized feature statistics

A `normalized` artifact holds `data`, the float32 values with one row per selected cell, and two
float64 arrays with one value per feature: `feature_sum`, the sum of the stored values, and
`feature_m2`, the sum of their squared deviations from the feature mean. Feature scaling for PCA
reads the mean `feature_sum / n_cells` and the standard deviation `sqrt(feature_m2 / n_cells)`
from them without reading `data`. Each stored row band adds its sums and merges its `feature_m2`
with the deviation of its mean from the running mean (`scarf.utils.moments.ColumnMoments`), so
the variance never subtracts a squared mean from a mean of squares: it is never negative, and a
feature whose stored values are all equal has a `feature_m2` of exactly zero and a scale of 1.

Normalized artifacts of earlier releases hold `feature_squared_sum`, the sum of squared values,
instead of `feature_m2`. Reuse requires only `data` and `feature_sum`, so they are still reused.
Feature scaling never reads `feature_squared_sum`, because a variance from squared sums loses its
significant digits when a mean is large beside its spread: for such an artifact, as for one
without `feature_sum`, it streams the means and standard deviations from `data`. Running
`run_normalization` with `invalidate_cache=True` writes `feature_m2` and lets scaling read it.

An artifact group's attributes hold `artifact_id`, `kind`, `provenance`, `execution_options`,
`created_at_ns`, `scarf_version`, and `complete`. `provenance` is a mapping with `operation`,
`parameters`, and `inputs`, serialized to canonical JSON values, and with an integer `revision` only
when the operation revision is 2 or more. An artifact that records `revision` 1, or a revision
that is not an integer, is not current and is never reused. An artifact without `revision` is
revision 1, so stores written before operation revisions existed keep their identities. Reuse
compares the whole `provenance` mapping exactly; `execution_options`, `created_at_ns`, and
`scarf_version` never affect it. {doc}`operation_revisions` explains when a revision changes.

## Pipeline records and snapshots

Pipeline run, stage, and label-claim documents use one strict unversioned shape. A run has exactly
`runId`, `recipe`, `requestedLabel`, `label`, `assay`, `startedAtNs`, `finishedAtNs`, `status`,
`complete`, `scarfVersion`, `config`, `stageOrder`, `outputs`, `fields`, `error`, and
`interruption`. A stage has exactly `stage`, `ordinal`, `startedAtNs`, `finishedAtNs`, `status`,
`complete`, `outputs`, `plans`, `metrics`, `error`, and `interruption`. A label claim has exactly
`label` and `runId`. Unknown or missing fields fail closed.

Within 1.x a release may only add fields, and a release that adds one must still read records
written without it. No release removes, renames, retypes, or reinterprets a field, so a run of any
1.x release reads in every later 1.x release. A run's `config` and a field descriptor's `display`
are JSON mappings that readers do not check key by key: a later release may add keys to them, and
code that reads such a key treats its absence as the behavior of the release that wrote the run.

Reopening a run reads its `config` as recorded and never checks it against the current recipe, so
a run whose settings a later release rejects, such as the t-SNE setting `parallel`, still reopens
and reports.
A 2.0 release may change records with an entry in the compatibility inventory.

The run record is created before `input_snapshot`; final outputs and field descriptors are written
only when the complete recipe succeeds. A handled failure or interruption first commits terminal
stage and run details. A hard process death can leave a run or stage incomplete. There is no
on-disk resume, repair, or same-ID retry protocol.

Requested run labels use append-only atomic claims below
`pipeline/runs/.label-claims/{labelDigest}` while a run is finalized. The completed run record is
the public label owner. Failed, interrupted, and removed predecessors can be bypassed by a later
claim. A live or unclean incomplete predecessor blocks the same label and fails closed. After the
operator has confirmed that its process stopped, the exact owner can be marked interrupted with
`pipeline.abandon_label_claim(label=..., run_id=..., reason=...)`; Scarf never infers abandonment
from elapsed time. A storage backend without atomic conditional creation rejects a labeled run
before its run record or any computation is started.

Catalog scans skip run children they cannot read, so one crash cannot hide healthy runs, and log a
warning that names each skipped run ID and its error. Opening such a `run_id` exactly raises that
error.

The first stage stores a cell-selection artifact, a `feature_universe` all-feature selection, and
full-axis cell and feature metadata snapshots. Frozen `run.features["I"]` is backed by that
immutable feature selection rather than the live feature `I` column. Stored selection integrity
compares its Boolean payload and current ordered row-ID fingerprint, not the later value of the
source metadata column. Frozen run views therefore survive live `I` changes but fail if row
identities are replaced or reordered.

Each completed or failed stage stores exact nested artifact-plan dispositions and sampled
process-tree RSS metrics. Artifact reuse comes from planning receipts, never timestamp inference.
Timing, memory, run identity, and reuse state remain in the run ledger and never mutate an artifact.
New artifacts store immutable creation time and creator Scarf version for diagnostics only.

## Removed assay state

The former `{assay}/state` analysis document is not part of this layout. `DataStore` rejects a
store containing that group before it initializes datastore metadata. Scarf does not inspect its
contents, migrate it, or use it to recover a current graph. Re-import or rebuild the dataset with
the current release.

`repack_zarr` copies run records and their append-only label claims because it preserves the axes.
This applies to the root datastore and nested workspaces. Subset and merge outputs do not copy
source runs.
The explicit rewrite omits retired `{assay}/state` groups. Recompute analysis artifacts after the
rewrite; the removed document is never translated into current lineage.
An overwriting merge clears pipeline records and datastore-scoped artifacts in its destination
workspace while preserving unrelated root siblings.

(mounted_targets)=

## Mounted targets

`mount_datastore` creates a target whose root attribute `matrixSource` binds it to the source
that owns the counts. The record holds exactly `location` (an absolute path or URI),
`workspace`, and `assays`, and each assay entry holds exactly `datasetFingerprint`,
`countsFingerprint`, and `requiresTranspose`. Every open checks the source against these
identities. Any other shape, including a field that a later release adds, is rejected rather than
ignored, and a source that is itself a mount is refused.

The target holds copied cell and feature tables, its own artifacts, and its own pipeline runs.
Opening it reopens the target root through a namespace that resolves keys in this order:

1. Keys outside the artifact roots, `[workspace/]artifacts` and `[workspace/]{assay}/artifacts`
   of the mounted assays, belong to the target alone. Cell and feature tables, counts, and
   `pipeline/` never fall back to the source.
2. A key inside an artifact group `{root}/{kind}/{id}` is read from the target when the target
   holds the group's `zarr.json`, and otherwise from the source when the source holds it.
3. Listings of an artifact root and of its kind directories are the union of both stores, and a
   directory document that the target lacks is read from the source.

Every write goes to the target. Writing or deleting inside a source group raises
`PermissionError`, so the source is never modified. Artifact IDs are random 256-bit tokens, so a
group that the source holds is exactly the artifact a copy in the target would be. The target
lists, loads, traces, and reuses source artifacts with their identities unchanged, and a recipe
whose provenance matches the published results creates nothing. Pipeline runs and their atomic
label claims stay per store: open a source run on the source datastore and pass its refs to the
mount.

Refs, provenance, and run records never record a location, and nothing in the target records
which store holds an artifact. Results that use source artifacts therefore need the source, as
counts do, and a source artifact removed by an outside edit fails closed as a missing input. A
target group opened directly with `zarr.open_group` shows only the target's own artifacts.
`repack_zarr` reads a mount through the same namespace, so its output holds the artifacts of both
stores, the target's runs, and copied counts, and no longer needs the source.

(writer_destinations)=

## Writer destinations

Import writers and `SubsetZarr` create their store with
`scarf.storage.destinations.create_destination`, and most of them run `check_destination` first,
before they read their source. A destination without keys is created with mode `"w-"`, which
refuses keys that appear after the check. With `overwrite=True` a destination that holds keys is replaced with
mode `"w"` only when its root group holds no prepared assay and no `matrixSource`, and every
top-level member is a group that Scarf writes: `cellData`, `matrices`, `artifacts`, `pipeline`,
`agent_results`, an assay group (`is_assay`), a pending derived assay, or a workspace group that
holds only such members; inside them, every key is Zarr metadata or an array's chunks. Any other
key, such as a `.DS_Store` file or a root array, and keys without a root group are refused with
`FileExistsError`. Every source that a writer reads is prepared, and `SubsetZarr` also refuses a
destination that overlaps a source, so no writer replaces its source. A local destination below a
directory that holds `zarr.json`, `.zgroup`, or `.zarray`, as a path or a local store, and a
destination whose root group carries `is_assay`, raise `ValueError`. Local paths reach Zarr as `pathlib.Path`, because Zarr reads a string as a URL and
would end the path at `#`, `?`, or `;`.

`DataStoreMerge`, `mount_datastore`, and `repack_store` create their destination with mode `"w-"`
and refuse a destination that overlaps a source. They and `SubsetZarr` refuse a source store that
holds a pending derived assay (`scarf:pending_assay`) in any workspace, through
`refuse_pending_assays`.

(assay_types_attribute)=

## Assay types

The root or workspace attribute `assayTypes` maps each assay name to a preset: `RNA`, `ATAC`,
`ADT`, `HTO`, `CRISPR`, `ANTIGEN`, `CUSTOM`, `GeneActivity`, `GeneScores`, `URNA`, or `Assay`, the
generic type without modality-specific normalization. The preset selects the assay class, and so
the normalization; `HTO` shares the ADT class but stays `HTO`, which `run_hto_demultiplexing`
requires. Import writers seed the entry from an explicit `assay_type` or from the assay name, and
a writable `DataStore` open records the types it resolved. An explicit type that is not a preset,
an `assay_types` key that names no assay of the store, and a recorded type that is not a preset
raise `ValueError` before anything is written. An assay without an explicit or recorded type takes
the preset of its name, or `Assay` with a warning. A read-only open cannot record a type, so it
raises `ValueError` for an explicit type that differs from the type the store declares; open the
store once with `zarr_mode="r+"` to record it. An `assayTypes` attribute that is present but not a
mapping raises the same remedy instead of being read as empty: a writable open whose
`assay_types` names a preset for every assay replaces it.

Each open assay carries the type that its `DataStore` resolved, `Assay.assayType`, which
`scarf.assay.classification.declared_assay_type` returns. Merge, subset, and
`run_hto_demultiplexing` read that type rather than the attribute, so a workspace store opened
with `assay_types={"tags": "HTO"}` declares `HTO` everywhere in the session that recorded it, and
a type written to the attribute behind an open `DataStore` takes effect at its next open. Derived
assays carry the type they were registered with. Subset, repack, and mount copy the declarations,
and `DataStoreMerge` records the one type that every source holding an assay declares, rejecting
sources that declare different types. A mount opens each target assay as its `assay_types` entry,
or else as the source declares it, and requires the source's `countsT` for exactly the assays that
open as RNA.

## Pending derived assays

`add_grouped_assay` and `add_melded_assay` create the logical group `{assay}` or
`{workspace}/{assay}` with `scarf:pending_assay`, the name of the operation, and no `is_assay`, so
assay scans skip it. Its counts (below `matrices/{assay}` in a workspace layout), feature
metadata, RNA `countsT`, and membership column `cellData/{assay}_I` ({ref}`assay_membership`)
follow, and one attribute write then replaces `scarf:pending_assay` with `is_assay=True`.

- Rule A: nothing is deleted once that publication write is issued, even when it fails.
- Rule B: cleanup deletes only the groups that the failed call created.
- Rule C: an ordinary exception deletes them. An interruption (`KeyboardInterrupt`,
  `asyncio.CancelledError`, `SystemExit`) keeps the pending assay, because Zarr may still be
  running the interrupted write, and logs how to remove it.

Every workspace keeps the counts of `{assay}` in `matrices/{assay}`, so a name that is pending in
any workspace is refused in all of them, and `DataStore.discard_interrupted_assay` never removes
counts that another workspace published under that name. It removes the pending assay with its
matrix group, `assayTypes` entry, and membership column; call it only after confirming that no
process is writing it. Repacking refuses a pending assay.

(assay_membership)=

## Assay membership

Every assay of a store spans the store's whole cell axis, so an assay that measured only some cells
still has a `counts` row for every cell. The boolean cell column `<assay>_I` marks the cells that
the assay measured. It has no missing-value mask, and its attributes include
`role="assay_membership"` and `assay="<assay>"`; other attributes are allowed, and a merge does not
carry them into the merged column. A store without the column measured every cell with the assay.
A cell outside an assay holds none of its counts, so its `counts` row is zero and its
`<assay>_nCounts` is 0. Those zeros are no measurement, so analysis fails closed on them.

An operation that reads an assay's values over cells checks the cells against the assay's
membership column after its own argument and assay-type checks and before it plans, reuses, or
writes a result, on a read-only store too: `run_normalization`, `integrate_assays` (the shared cell
selection against every source assay), `select_hvgs`, `select_detected_features`, `run_waggr`,
`run_aucell`, `run_marker_search`, `make_bulk` (the cells whose counts it reads: those of its bulk
columns, or every selected cell for a mean that fits the assay's normalization over them, found
from the live `I` column before a metadata grouping without `cell_selection` snapshots it),
`run_statistical_testing` (feature keys, over the cells that the design keeps),
`select_prevalent_peaks`, `run_cell_cycle_scoring`, `run_feature_percentage`,
`run_hto_demultiplexing`, `run_doublet_detection`, `run_pseudotime_marker_search` and
`run_pseudotime_aggregation` (the valid cells of the pseudotime), `get_imputed` (feature names),
`run_mapping` (the query assay), `to_anndata` with `matrix="normed"` and without a run,
`pipeline.run`, which checks the cells of `cell_key` before any stage runs and never narrows them,
also for each other assay whose metric its filtering stage names, and the quality-control filters
`auto_filter_cells` and `filter_cells` for each metric of an assay that they filter on. A cell column is a metric of the assay whose preparation wrote it:
`<assay>_nCounts`, `<assay>_nFeatures`, and the percentage columns that the assay records in its
`percentFeatures` attribute (`scarf.storage.identity.generated_cell_columns`); the column is found
among each assay's own columns, never by parsing its name. A `quality_metric` artifact scoped to an
assay is a metric of that assay. A cell that the column marks False raises
`scarf.metadata.membership.UnmeasuredCellsError`, a `ValueError` whose `operation`, `assay`,
`column`, `unmeasured`, `selected`, `remedy`, and `cell_key` attributes name the refusal, and whose
message names the remedy for the operation's input: a cell selection from `select_measured_cells`,
passed as `cell_selection=` to `make_bulk`, `run_statistical_testing`, and the QC filters, labels
narrowed with `snapshot_cluster_labels`, a graph built over measured cells, or a `cell_key` column
that is True only for the cells of the caller's column that the assay measured, such as one that
`ds.cells.insert("RNA_measured", ds.cells.fetch_all("I") & ds.cells.fetch_all("RNA_I"))` writes;
the membership column alone also holds cells outside `I`. Operations that read only other results
or metadata, such as PCA, neighbors, clustering, layouts, and `snapshot_cluster_labels`, do not
check; the operations at the roots of their lineage do. Derived assays copy the membership. Raw
exports that can declare it keep unmeasured cells: `to_h5ad` and `to_anndata` without layers. An
export that cannot declare it refuses them before it writes or reads anything: `to_mtx`, whose
zero rows `MtxToZarr` would read back as measured cells, and a `to_anndata` layer of an assay other
than the exported one.

The checks read the membership column in bounded blocks. A boolean column or mask is read beside
the column, block by block; a cell-selection artifact is read as its stored mask. Integer rows of
at least one cell in 16 stream the column once into a copy of one byte per cell; fewer rows read
only the chunks that hold them, in their requested order, repeats included.

`select_measured_cells(assay, cell_selection=...)` keeps the cells of a selection, or of the live
`I` column when it is omitted, that the assay measured, as a datastore `cell_selection` artifact
whose operation is `select_measured_cells`, whose parameters are `{"assay": <assay>}`, and whose
inputs record the prior selection and the fingerprint of the membership column. It returns the
prior selection itself when the assay has no membership column or measured every selected cell,
so a fully measured store keeps its identities, and raises `ValueError` when the assay measured
none of the selected cells. Display reads show unmeasured cells as missing: feature values that
plots fetch, those of `get_cell_vals`, and cluster-tree fill values are NaN for them, which plots
draw in the missing-value color or leave out. Only the measured cells are read and normalized, so a
normalizer fitted over the cells that it reads, such as ATAC TF-IDF, whose document frequency would
otherwise count the unmeasured cells, gives a measured cell the value that a read of the measured
cells alone gives; the stream yields the rows of unmeasured cells as NaN pieces of at most one block
of rows. Summary plots compute their statistics over the cells with a value, which `n_cells`
counts, and the plots record the number of plotted cells that each assay did not measure as
`provenance.extras["unmeasured_cells"]`.

Only imports, `DataStoreMerge`, and derived assays write the column, directly in storage and with
its attributes in its first metadata write, so a reader never finds the column without them; the
copies that `SubsetZarr`, `mount_datastore`, and `repack_zarr` make carry them the same way. A
cell table reserves the name `<assay>_I` of every assay of its store or
workspace, including a pending derived assay, whether or not the column exists:
`MetaData.insert`, with or without `overwrite`, `update_key`, and `reset_key` raise `ValueError`
before they read any value, and `drop` raises for a column that carries the membership `role`,
because a store without the column counts every cell as measured
(`scarf.storage.identity.protect_metadata_column`). A column of that name without the role, such
as a plain column that an earlier release imported or let a user insert, is not a membership
column, and `drop` removes it to repair the store. Feature tables and other groups reserve
nothing.

`SeuratToZarr` writes the column for an `Assay5` whose selected count layers hold only some cells.
Membership is the union of the cells of the selected layers, so with `assay_layers` a cell that only
an unselected layer holds is not a member.

`add_grouped_assay` and `add_melded_assay` write a row for every cell, and a cell that the source
assay did not measure holds none of its counts, so the new assay measures the cells of its source.
When the source has a membership column, the new assay's `<assay_label>_I` is a copy of it, written
while the assay is pending, so a failed write and `discard_interrupted_assay` remove it with the
pending assay. A source without the column gives an assay without one. Both methods refuse, before
they compute anything, a new assay whose membership name is already a cell column; drop that column
or choose another name.

`SubsetZarr` keeps the membership column of each assay that it writes, with the rows of the selected
cells, and leaves out the column of every other assay. `mount_datastore` mounts every assay of its
source, so it copies every membership column with the cell table, as `repack_zarr` does.

`DataStoreMerge` writes the column for every merged assay. A merged cell keeps the value of its
source: False when the source lacks the assay (its counts are zero-filled), True when the source has
no membership column, and the source's own value otherwise. A source's membership column is merged
only into that column, never as an ordinary column such as `orig_<assay>_I`, and ordinary merged
columns never carry the `role` or `assay` attributes. The membership column of a source assay that
`assays=` leaves out is not merged at all, so it cannot take the membership name of an assay that a
later merge includes. Planning raises, before anything is written, for a source column with the
reserved name that breaks the contract (another dtype, a missing-value mask, or `role` and `assay`
attributes that do not name the assay), for an ordinary source column whose merged name is a
membership column, and for a source whose cells outside the assay have a nonzero
`<assay>_nCounts`. The merge manifest records each source's membership state per assay: `missing`,
`all`, or the fingerprint of the source column, so a resume requires the same membership.

Export and import keep membership explicit. Without a pipeline run, `to_h5ad` and `to_anndata`
write the membership column of the exported assay as an ordinary `obs` column and declare it in
`uns["scarf"]["assayMembership"]`, a mapping from the assay to its column. They do not write the
membership columns of other assays, which describe assays that the file does not hold, as a merge or
subset that leaves an assay out does not copy its column. An import reserves the
column name `<assay>_I` of every assay it writes: `H5adToZarr` restores a declared column, which
must be boolean without missing values, with the membership attributes, and skips any other column
of that name with a warning, as `CSVtoZarr`, `CrToZarr`, and `MtxToZarr` always do;
`SeuratToZarr` raises for such a metadata column. A column named for an assay that the import does
not write stays ordinary metadata. A merged store therefore keeps its partial membership through
`to_h5ad`, `H5adToZarr`, and a second merge.

Stores written by earlier releases are not detected or repaired. Results that earlier releases
computed over cells that an assay did not measure stay listable, loadable, and traceable, and
completed pipeline runs over such cells reopen; running such a configuration again raises
`UnmeasuredCellsError` before any stage runs. A merge with a source whose
`<assay>_I` marked some cells False recorded those cells as members and kept the source values
only as `orig_<assay>_I`, or lost them with `prepend_text=None`. A Seurat import whose
`assay_layers` selection excluded cells recorded them as members. Redo such merges and imports from
the original sources. A derived assay that an earlier release built from a partially measured
source has no membership column, so it counts every cell as measured; build it again under a new
label. A subset or a merge that an earlier release wrote without an assay can hold that assay's
membership column, under its own name or as `orig_<assay>_I`; there it is no assay's membership,
and `drop` removes it.

## Count arrays

`counts` is the cell-major assay matrix (`n_cells` × `n_features`).
RNA assays (`RNAassay` and aliases such as `GeneActivity` / `URNA`) also store `countsT`, the same values in gene-major order (`n_features` × `n_cells`).
Import, subset, merge, and `repack_zarr` write both arrays together on Zarr v3.
Non-RNA assays (ATAC, ADT, and similar) write `counts` only.

The two RNA arrays are orientations of one matrix, not independent datastores.
New count arrays are sharded Zarr v3 arrays.
Normalized arrays stay unsharded (`shards=None`).

The matrix group, `counts`, and `countsT` each carry a `scarf:countMatrixLayout` attribute.
Scarf checks that those records agree with each other and with the live arrays.
Missing or mismatched layout metadata is an error.
There is no silent rewrite on open and no automatic upgrade of older count layouts.

Writers compute the raw-count identity while they write `counts`, so no second pass reads the
matrix back. `counts` records it as `content_fingerprint`. The sibling `countSummaries` group holds
per-cell totals (`rowSums`), per-cell detected-feature counts (`rowPositive`), and per-feature
detected-cell counts (`columnPositive`), and its `source_fingerprint` attribute names the counts
it describes. Preparation reads these summaries instead of streaming the matrix. Summaries that are
missing, malformed, or bound to other counts are an error; rebuild the store with `--data-only`.

### Count dtype

`counts` and `countsT` share one dtype, which `scarf.storage.count_dtype` resolves from the
canonical (duplicate-summed) values of the assay. Counts are stored unsigned, in the narrowest of
`uint8`, `uint16`, `uint32`, and `uint64` that holds the largest value, exactly when every value
is a non-negative integer. Other counts keep their source dtype: `float32` or `float64`, or a
signed integer dtype when values are negative. `float16` is not a count storage dtype. The dtype
depends only on the assay's own values, not on the reader, source encoding, index order,
orientation, memory budget, or the other assays of a source matrix, so the same counts imported
from H5AD, 10x HDF5, Matrix Market, CSV, an in-memory sparse matrix, or a Seurat object give the
same store identity. Imports that split one source matrix into assays (10x HDF5 and Matrix Market
feature types, H5AD `assay_split_key`) resolve the dtype of each assay from its own features.
Every import reads all its counts once before it creates the destination to find their ranges:
readers of 10x HDF5, Matrix Market, and H5AD files report the range of each group of features
over the selected cells (`count_value_ranges`), the CSV reader finds it in its first pass, and
`SparseToZarr` and `SeuratToZarr` scan their sources. The scans sum duplicate coordinates as the
writers do, integers exactly in 64 bits and floats in source order. No import takes a count dtype
argument. Count matrices hold finite values: writers reject NaN and infinity, and every import
rejects them before it creates the destination. Writers cast through a checked cast, so a count
that the stored dtype cannot hold raises instead of wrapping.

Writers that rebuild or combine existing counts follow their sources instead. Subset and repack
keep the source dtype, because they rebuild an existing dataset whose identity and copied
artifacts must stay valid. A merge stores the common type of its source count dtypes, widened so
that features summed by name cannot overflow it; integer sources without a common integer dtype
are rejected rather than stored as floats. Derived assays (grouped and melded) keep their
`float64` values.

### Count layout

The layout policy (`unitBytes` and `chunkBytes`) is recorded in `scarf:countMatrixLayout`, and
readers and `countsT` writers replay it. The imports (H5AD, 10x HDF5, Matrix Market, CSV, sparse,
and Seurat) write the default policy unless they are given one, so the layout of an imported store
does not depend on the budget of its import. When the policy's counts write or `countsT` transpose
does not fit `mem_budget`, an import refuses before it creates its destination and names the
largest halving of the policy, with `unitBytes` and `chunkBytes` halved together, that fits; a
smaller layout makes every later `countsT` read slower. Subset, `repack_zarr`, merge, and
`add_grouped_assay` fit the policy to `mem_budget` before they create their destination: when the
default policy's counts write or `countsT` transpose does not fit, they halve `unitBytes` and
`chunkBytes` together, keeping the chunks per shard, down to count shards of one row. Sparse
writers admit the band writes of their sparse sources, and dense writers the dense row bands they
write. Writers that choose their source batches (the sparse imports, the Seurat imports, and
merge) admit batches of one destination row band, the batch their write starts from, so a layout
never leaves the write narrower batches than its bands. An explicit policy is used exactly or
refused before the destination exists, with the largest halving that fits named. `add_melded_assay`
sizes its shards to the melding band that fits `mem_budget`. Writers
that write their assays one at a time (Seurat, subset, repack, and merge) admit each assay on its
own. A resumed merge keeps the layout persisted with its completed counts, so a budget change
between attempts cannot block it, and fits the layout of the counts it rewrites. A store written
with a smaller policy, chosen or fitted, therefore has smaller shards and more objects. The layout
never changes identity: `content_fingerprint`, the counts fingerprint, and the dataset fingerprint
are computed from the stored values, so only the layout fingerprint differs between layouts.

Existing stores are never rewritten under a new dtype or layout rule and keep their dtype, layout,
and identity. A re-import of the same source can store a different dtype, and so get different
fingerprints and artifact ids; mounts and mapping references bound to the replaced store fail
closed.

## Opening an RNA assay

`RNAassay` construction requires a complete `countsT` on Zarr v3 plus matching layout metadata.
The open fails if `countsT` is missing, incomplete, unsharded, a Zarr v2 array, or out of agreement with `counts`.

`repack_zarr --data-only` reads the raw store without constructing `RNAassay`.
The user-facing repair is to re-import the source or run:

```bash
uv run python -m scarf.tools.repack_zarr input.zarr output.zarr --profile fast_local --data-only
```

Without `--data-only`, `repack_zarr` accepts only a prepared source and preserves its results after
verifying that the destination has the same dataset identity.

After a rewrite, recompute HVG, normalization, PCA, graph, and marker artefacts.
Do not resume them from pre-rewrite lineage.

Non-RNA assays in a multi-assay store remain usable when tooling opens without forcing every assay class to construct.

## Zarr versions

New stores default to Zarr v3.
RNA assays require v3 because sharded `countsT` is a v3 feature.
A Zarr v2 RNA store will not open as `RNAassay`.

## Memory controls

`DataStore(..., mem_budget='8G')` bounds streaming and concurrency (blocks, concurrent work, feature batches).
A Zarr read holds more than its result. A read of a sharded array, such as `counts` or `countsT`, fills a shard-level copy of its selection, holds the compressed bytes of every inner chunk it touches in a shard, and decodes those chunks next to them. Plans charge this through `ArrayGeometry.readBytes`, at the decoded size of the compressed bytes because a plan cannot know them before it reads: a `counts` row block of a whole shard reserves twice its bytes, the shard's chunks, and the one chunk that row-block streams decode at a time, and a `countsT` cell band reserves twice its bytes and two of each chunk it touches. An unsharded read, such as one of normalized data, is charged its result and the chunks it decodes; the compressed bytes of a chunk it decodes and a decoded chunk that a codec thread still holds after a read stay outside the plan.
Environment variables used by the documentation executor (`SCARF_MEM_BUDGET`, `SCARF_WORKERS`, …) are also useful for local large runs.

## Related guides

- {doc}`../tutorials/data_organization`
- {doc}`../concepts/memory_and_execution`
- {doc}`contributing`
- {doc}`operation_revisions`
