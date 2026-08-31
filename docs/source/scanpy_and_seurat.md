---
description: Decide where Scarf fits, then choose the focused guide for Scanpy or Seurat.
---

(scanpy_and_seurat)=
# Coming from Scanpy or Seurat

Scarf keeps the familiar biological sequence of quality control, feature selection,
normalization, reduction, neighbourhood graphs, embeddings, clustering, and marker discovery.
The main change is how the work is executed. A Scarf `DataStore` opens a Zarr store on local disk
or object storage, streams bounded blocks, and writes each completed result back as an immutable
artifact.

Choose the guide for the ecosystem you already use:

- {doc}`scanpy` maps Scanpy stages and explains H5AD transfer.
- {doc}`seurat` maps Seurat stages, RDS import, and multimodal WNN analysis.

If you want to run Scarf before comparing APIs, start with the {ref}`Quick start <quickstart>`.

## Where Scarf fits

Scarf is most useful when:

- the count matrix is too large for convenient in-memory analysis
- counts should stay on local disk or object storage while the analysis runs
- completed steps should persist so they can be inspected or reused
- RNA, ATAC, or CITE-seq assays should live in one analysis store

You do not need to move an entire project to Scarf. A common pattern is to run the large,
graph-based part of an analysis in Scarf, then export the required data for a method in another
single-cell ecosystem. Scarf does not include scVI, Scanorama, RNA velocity, or replicate-aware
differential expression.

## The mental-model change

| Familiar concept | Scarf |
|---|---|
| An `AnnData` or `SeuratObject` holds counts, metadata, and results | A `DataStore` opens a Zarr store containing counts, metadata, and results |
| Analysis changes an object in the current session | Each supported analysis step persists an immutable artifact |
| Filtering subsets an object or creates a view | Filtering returns an immutable selection artifact without deleting cells |
| Graphs and embeddings occupy named object slots | Graphs and embeddings are passed by exact artifact reference |
| One object exposes an active result set | A `PipelineRun` exposes one durable frozen result set without replacing other runs |

Feature selection is artifact-only. A call such as `features = ds.select_hvgs(cells, ...)` returns
an immutable reference that normalization and downstream methods consume. Related analysis
branches can therefore coexist in one datastore without an implicit active result replacing an
earlier one.

See {doc}`concepts/provenance` for the result model and {doc}`tutorials/import_and_export` for the
full format contracts.
