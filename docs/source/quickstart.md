---
description: Minimal scRNA-seq workflow in Scarf from count matrix to UMAP and clustering.
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

(quickstart)=

# Quick start

This page runs a minimal scRNA-seq pipeline on a 5K PBMC dataset using
`ds.pipeline.run`. For the full workflow, see {doc}`tutorials/scrna_seq`. If you
know Scanpy, see {doc}`scarf_and_scanpy` first.

The pipeline recipe currently accepted is `basic_rna_analysis` only.

## Load counts into Zarr

```{code-cell} ipython3
import scarf

scarf.set_verbosity('WARNING')

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    'tenx_5K_pbmc_rnaseq',
    destination='scarf_datasets',
)
reader = scarf.CrH5Reader(f'{dataset}/data.h5')
scarf.CrToZarr(
    reader,
    zarr_loc=f'{dataset}/data.zarr',
).dump(batch_size=1000)
```

## Open the store and inspect QC

QC thresholds are dataset-specific. Inspect distributions before choosing
cutoffs. Filtering marks cells inactive (cell key `I`) rather than deleting them.

Opening the store prints a message that the smallest cell count is below the RNA
size factor of 1000. The filter thresholds set below remove those cells.

```{code-cell} ipython3
ds = scarf.DataStore(
    f'{dataset}/data.zarr',
    nthreads=4,
    min_features_per_cell=10,
)
ds.plots.distribution(
    keys=['RNA_nCounts', 'RNA_nFeatures'],
    kind='violin',
    max_points=2000,
)
```

## Run the standard RNA recipe

Pass manual filter thresholds through `filtering=`. Opt out of optional score
and marker steps with `False` so this page stays a short path to UMAP and
Leiden. Capture the returned artifact refs.

```{code-cell} ipython3
artifacts = ds.pipeline.run(
    pipeline_id='basic_rna_analysis',
    filtering={
        'method': 'manual',
        'attrs': ['RNA_nCounts', 'RNA_nFeatures'],
        'highs': [15000, 4000],
        'lows': [1000, 500],
    },
    cell_cycle_scoring=False,
    highly_variable_features={
        'min_cells': 20,
        'top_n': 500,
        'show_plot': False,
    },
    pca={'dims': 15, 'n_centroids': 100},
    neighbors={'k': 11},
    umap={
        'n_epochs': 250,
        'spread': 5,
        'min_dist': 1,
        'parallel': True,
    },
    leiden={0.5: {'label': 'leiden_cluster'}},
    paris=False,
    doublet_scoring=False,
    markers=False,
)
list(artifacts)
```

```{note}
By default the pipeline names each Leiden column after its resolution, giving
`RNA_leiden_0.5`. The `label` above renames it to `RNA_leiden_cluster`, which is what
`run_leiden_clustering` writes and what the rest of the documentation uses. The key in the
returned dictionary stays resolution-based (`leiden_0.5`) either way.
```

## Plot the embedding

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by='RNA_leiden_cluster',
)
```

Each colour is a Leiden cluster on the UMAP built from the neighbourhood graph.

## Inspect what was saved

List assay artifacts and inspect one result from the pipeline return value:

```{code-cell} ipython3
print('Artifact kinds:', sorted({ref.kind for ref in ds.list_artifacts()}))

status = ds.inspect_artifact(artifacts['connectivity_map'])
print('Graph complete:', status.complete)
print('Graph operation:', status.operation)
print('Graph parameters:', status.parameters)
```

Typical columns written by this configured run:

- Cell QC: `RNA_nCounts`, `RNA_nFeatures` (and mito/ribo fractions when patterns match)
- Active cells: boolean key `I`
- Embedding: `RNA_UMAP1`, `RNA_UMAP2`
- Clusters: `RNA_leiden_cluster`

For provenance concepts, see {doc}`concepts/provenance`. For atomic control of
the graph chain, see {doc}`tutorials/atomic_graph_operations`.

## Further reading

- [Single-cell best practices: quality control](https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html)
- [Single-cell best practices: clustering](https://www.sc-best-practices.org/cellular_structure/clustering.html)
- [Scanpy clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html)

## Next steps

- Full scRNA-seq chapter: {doc}`tutorials/scrna_seq`
- Atomic graph operations: {doc}`tutorials/atomic_graph_operations`
- Scanpy and Seurat mapping: {doc}`scarf_and_scanpy`
- Publication plotting with `ds.plots`: {ref}`plotting showcase <plotting_showcase>`
