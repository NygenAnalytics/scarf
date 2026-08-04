(api)=
# API reference

Public Scarf surfaces for analysts:

- `DataStore` and its documented methods
- Graph-construction methods and `ds.pipeline.run`
- `ArtifactRef`, `ArtifactStatus`, and `AssayState`
- `EnrichmentResult` and `read_gmt` for gene-set scoring
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
| Graph construction | {doc}`api/graph_construction` |
| Artifacts and assay state | {doc}`api/artifacts` |
| Analysis pipeline | {doc}`api/pipeline` |
| Assays and metadata | {doc}`api/assays` |
| Integration and metrics | {doc}`api/integration` |
| Mapping | {doc}`api/mapping` |
| Plotting | {doc}`api/plotting` |
| Cytebase and utilities | {doc}`api/utilities` |

Stage names match {doc}`../scanpy_and_seurat` and {doc}`../tutorials/scrna_seq`.
