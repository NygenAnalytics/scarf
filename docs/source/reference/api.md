(api)=
# API

Public Scarf surfaces for analysts:

- `DataStore` and its documented methods
- Readers and writers that create or export Zarr stores
- `scarf.plotting`
- Documented integration metrics (`DataStore.metric_*`; `scarf.metrics` holds the underlying functions)
- `MappingReference` / `MappingResult` for atlas-style mapping

Inheritance helpers (`BaseDataStore`, `GraphDataStore`, `MappingDatastore`) are listed under
{doc}`api/datastore` for completeness. Prefer calling methods on `DataStore`.

## By analysis stage

| Stage | Page |
|---|---|
| Import and export | {doc}`api/import_export` |
| DataStore (all stages) | {doc}`api/datastore` |
| Assays and metadata | {doc}`api/assays` |
| Integration and metrics | {doc}`api/integration` |
| Mapping | {doc}`api/mapping` |
| Plotting | {doc}`api/plotting` |
| Utilities and datasets | {doc}`api/utilities` |

Stage names match {doc}`../scarf_and_scanpy` and {doc}`../tutorials/scrna_seq`.
