(architecture)=
# Architecture

Scarf uses concrete domain packages and a one-way module-load dependency
structure. The public API is exposed through package facades, while reusable
computation and storage mechanics live in focused implementation modules.

## Dependency direction

Module-load dependencies point toward the earlier layers in this list:

1. Foundation: `storage`, `matrix`, and `utils`
2. Data model: `metadata` and `assay`
3. Domain algorithms: `neighbors`, `embeddings`, `clustering`, `trajectory`,
   `metrics`, `features`, `quality_control`, and `mapping`
4. Import and export: `readers`, `writers`, and `merge`
5. Orchestration: `datastore` and `datastore._operations`
6. Presentation: `plotting`

The root `scarf` package is a public import facade. It does not form another
runtime layer and must not eagerly import the implementation graph.
Only modules listed by that facade are available as lazy root attributes.
Presentation and lower-level algorithm packages such as `plotting`,
`clustering`, `neighbors`, and `trajectory` require explicit imports. This
keeps the root surface small and avoids loading optional presentation code.

`plotting` does not import `datastore`. Unified plotting consumes a narrow
datastore adapter instead of resolving projection paths. Existing heatmap
functions still read Zarr-backed values from their duck-typed `store` and
`assay` inputs. Removing that remaining storage coupling is deferred.

Function-local imports may cross toward presentation when a public method
explicitly requests a plot. Examples include RNA feature selection and
datastore quality-control helpers. These calls do not create module-load
cycles.

## Package responsibilities

### Foundation

- `storage/` owns stores, layouts, schemas, arrays, sharding, copying,
  resource budgets, storage profiles, materialization, ANN persistence, and
  Zarr runtime guards.
- `matrix/` owns the lazy blockwise matrix abstraction used over NumPy and
  Zarr arrays. Its arithmetic, indexing, and reduction behavior keeps it
  separate from low-level storage mechanics.
- `utils/` owns generic array, compute, logging, prefetch, process, and progress
  helpers. Zarr-specific helpers belong in `storage`, not `utils`.

Facade aliases do not change implementation ownership. `scarf.utils.load_zarr`
remains available for compatibility, but its implementation belongs to
`storage.stores`. Column prefetch uses `storage.parallel` because read-ahead
limits and I/O concurrency are governed by the active storage budget.

### Data model

- `metadata/` owns Zarr-backed metadata tables, row streaming, and table
  queries. It is shared by datastore cell metadata and assay feature metadata,
  so neither `datastore`, `assay`, nor `storage` owns it.
- `assay/` owns assay state, normalization, summary persistence, and the RNA,
  ATAC, and ADT assay types.

Data-model modules may call domain algorithms from the method that needs them.
They must not import those packages at module load time.

### Domain algorithms

- `neighbors/` owns ANN construction, KNN queries, graph operations, diffusion,
  weighted-neighbor integration, and stored KNN graph arrays.
- `embeddings/` owns PCA, LSI, Harmony correction, UMAP, SG-tSNE, and embedding
  initialization.
- `clustering/` owns Leiden clustering and PARIS hierarchy operations.
- `trajectory/` owns pseudotime scoring, feature-profile aggregation, feature
  module clustering, and pseudotime result records.
- `metrics/` owns LISI, silhouette, graph, concordance, and integration scores.
- `features/` owns variability selection, LOWESS trend fitting, feature scoring,
  rank and regression marker searches, GFF parsing, genomic intervals, and
  coordinate-based feature construction.
- `quality_control/` owns filtering, HTO demultiplexing, doublet processing,
  cell-cycle assignment, and the default cell-cycle gene references.
- `mapping/` owns reference artifacts, feature alignment, confidence,
  Symphony-style correction, CORAL compatibility, and mapping results.

Domain algorithm packages must not import `datastore`, `plotting`, or general
import/export packages at module load time. A domain that persists an artifact
may use a narrow, named `storage` adapter.

### Import and export

- `readers/` retrieves example datasets and parses supported input formats.
- `writers/` materializes Scarf stores and exports supported formats.
- `merge/` combines assays and datasets without importing `DataStore` during
  normal module loading.

Readers parse, writers materialize, and merge combines. Format-specific code
belongs in a module named for that format.

### Orchestration

`DataStore` remains the primary workflow API. Its public class chain is kept
for compatibility:

```text
BaseDataStore
  -> GraphDataStore
    -> MappingDatastore
      -> DataStore
```

Method implementations are grouped by responsibility under
`datastore._operations`:

```text
graph
embeddings
clustering
trajectory
mapping
quality_control
features
presentation
```

Operation mixins have no runtime inheritance from datastore facades, no
`__init__`, and no imports of sibling operation mixins. Reusable algorithms
must be placed in their domain package before being exposed through a
datastore method.

### Presentation

`plotting/` is the only plotting package. It has no import dependency on
`datastore`. The removed `scarf.plots`, `scarf.plotting._legacy`, and
`DataStore.plot_*` APIs must not be restored. New plots should return the
established plotting result types, accept documented data contracts, and use
narrow adapters instead of adding storage-path knowledge.

## Public facade policy

The lazy facades in `scarf`, `features`, `readers`, `writers`, `merge`, `utils`,
`neighbors`, `clustering`, `embeddings`, and `trajectory` are architectural
boundaries, not temporary deprecation shims. Their documented 1.x exports
preserve stable import paths and defer optional or expensive implementations
until an export is accessed.

The `assay`, `mapping`, `matrix`, `metadata`, `metrics`, `plotting`, and
`quality_control` package initializers are eager domain facades. API reference
pages and public contract tests define which of their exports carry
compatibility guarantees.

Reloading a lazy facade clears cached exports before resolving them again.
This keeps reload behavior deterministic for tests and interactive work.

Private facade exports used by repository tests are patch seams, not additions
to the documented user API. New production code should import its canonical
implementation directly unless it intentionally needs a public patch seam.

Compatibility policy for the 1.x series:

- `ZarrMerge` is a deprecated subclass of `AssayMerge`.
- `DataStore.metric_integration` is a deprecated name for label concordance.
- Deprecated `run_mapping` flags remain accepted with `DeprecationWarning`.
- These Python API compatibility paths are retained through 1.x and may be
  removed in 2.0.
- Specific older file schemas named in the migration notes remain readable
  while their compatibility tests are maintained. This is not a blanket
  guarantee for every historical artifact.

Retired flat internal modules do not have forwarding shims. Their focused
replacement paths are listed in {doc}`migration_notes`.

## Placement rules

Use these rules when adding code:

1. Put Zarr mechanics in `storage`, blockwise matrix behavior in `matrix`, and
   generic operational helpers in `utils`.
2. Keep metadata and assay focused on state, normalization, and persistence.
3. Put reusable computation in a concrete domain package.
4. Keep domain packages independent of datastore and plotting.
5. Use a named storage adapter when a domain persists an artifact.
6. Put parsing in readers, materialization in writers, and combination in
   merge.
7. Keep plotting free of datastore imports and use narrow adapters for new
   storage-backed inputs.
8. Add compatibility only at an existing public facade.
9. Use a concrete biological or computational package name. Do not introduce
   catch-all packages such as `core` or `analysis`.

Architecture boundaries are enforced in `tests/test_import_architecture.py`.
Public imports, result records, facade behavior, and wheel contents have
separate contract tests.

## Accepted and deferred decisions

Accepted:

- The datastore class chain remains for public compatibility.
- Same-path package facades remain part of the public architecture.
- A small set of domain modules has narrow storage dependencies for persisted
  artifacts.
- Marker statistics and genomic feature construction live under `features`.
- Pseudotime-specific feature aggregation and module clustering live under
  `trajectory.feature_dynamics`.
- `metadata` remains a root data-model package because assays and datastore
  orchestration both depend on it.
- Unified plotting uses a datastore adapter instead of reading Zarr paths.
- Old flat compatibility modules remain deleted.

Deferred to a later structural phase:

- The graph and mapping operation modules remain large. Splitting them requires
  a separate behavioral and performance gate.
- `metadata` and `assay` retain established convenience methods, with their
  domain imports deferred to call time.
- Function-local cycles inside `assay` and `readers` remain accepted because
  there are no module-load cycles.
- Heatmap plotting still reads Zarr-backed values from duck-typed store and
  assay inputs. Replacing those reads requires a separate plotting adapter
  design.

Rejected:

- A vague `core` or `analysis` package.
- Restoring forwarding modules for retired private import paths.
- Restoring legacy plotting modules or datastore plotting methods.
- Moving storage-aware algorithms into datastore orchestration.
