---
description: SNN and WNN multimodal graph integration for assays measured in the same cells.
---

(multimodal_integration)=

# Multimodal integration

For CITE-seq and other multi-assay experiments measured in the same cells, Scarf can
merge per-assay neighbourhood graphs with SNN or WNN.

The full executable walkthrough lives in {doc}`cite_seq`. This page summarizes the API
choices.

## When to use which method

Each assay must already have a KNN graph. `label` names the integrated graph; pass that
same string to later UMAP and Leiden calls via `integrated_graph`.

| Method | Call | Notes |
|---|---|---|
| SNN | `ds.integrate_assays(assays=['RNA', 'ADT'], label='RNA+ADT', method='snn')` | Default. Supports two or more assays. |
| WNN | `ds.integrate_assays(assays=['RNA', 'ADT'], label='RNA+ADT_wnn', method='wnn')` | Exactly two assays. Useful when modalities differ in sparsity or signal strength. |

After integration, run UMAP and clustering on the integrated graph (see {doc}`cite_seq`
for parameter examples).

## HTO demultiplexing

`DataStore.mark_hto_identities` assigns hashtag identities when an HTO assay is present
(default assay name `HTO`). A public executable HTO dataset is not yet in the Scarf
catalog; see {doc}`../reference/api/datastore` until one is added.

## Next steps

- {doc}`cite_seq`
- {doc}`choosing_integration_methods`
- {doc}`../reference/api/datastore`
