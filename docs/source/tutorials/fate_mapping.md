---
jupytext:
  formats: ipynb,md:myst
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

# Multi-sink fate mapping

Pseudotime places cells along one progression axis. Fate mapping complements that
ordering with a probability for each user-provided terminal state. This notebook
uses pancreatic endocrine differentiation to estimate Alpha, Beta, and Delta
fates without CellRank or Scanpy.

## Prerequisites

- Scarf installed with the `extra` optional dependencies
- An RNA assay with a neighbourhood graph and cluster annotations for sources and sinks

## What you will learn

- Define progenitor and terminal groups for multi-sink PBA
- Estimate a shared pseudotime across branches
- Compute absorption probabilities for each terminal fate

## Dataset

```{code-cell} ipython3
from pathlib import Path

import numpy as np
import pandas as pd

import scarf

scarf.set_verbosity('WARNING')
```

## 1. Load the preprocessed dataset

The prepared Zarr store from the `scarf_docs` Cytebase catalog contains a KNN graph,
UMAP coordinates, Scarf clusters, and the published cell-type annotations.

```{code-cell} ipython3
dataset_root = Path('./scarf_datasets')
dataset_path = (
    dataset_root
    / 'bastidas-ponce_4K_pancreas-d15_rnaseq'
    / 'data.zarr'
)
if not dataset_path.exists():
    scarf.cytebase.connect("scarf_docs").download_dataset(
        name='bastidas-ponce_4K_pancreas-d15_rnaseq',
        destination=str(dataset_root),
        zarr=True,
    )
```

```{code-cell} ipython3
ds = scarf.DataStore(
    str(dataset_path),
    nthreads=4,
    default_assay='RNA',
)
if 'RNA_paris_cluster' not in ds.cells.columns:
    ds.run_paris_clustering(n_clusters=10)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=['clusters', 'RNA_paris_cluster'],
    legend_loc='on_data',
)
```

## 2. Define progenitor and terminal groups

The existing pseudotime tutorial uses Scarf cluster 1 as the progenitor group.
For each terminal cell type that is present in the annotations, we assign an
unused Paris cluster that still contains that label, preferring higher counts.

```{code-cell} ipython3
annotations = ds.cells.to_pandas_dataframe(
    columns=['RNA_paris_cluster', 'clusters'],
    key='I',
)
requested_terminals = ['Alpha', 'Beta', 'Delta']
available_labels = set(annotations['clusters'].unique())
missing_labels = [label for label in requested_terminals if label not in available_labels]
if missing_labels:
    raise ValueError(
        f"Missing terminal labels in clusters: {missing_labels}. "
        f"Available labels: {sorted(available_labels)}"
    )

source_clusters = [1]
if source_clusters[0] not in set(annotations['RNA_paris_cluster']):
    raise ValueError(
        f"Source cluster {source_clusters[0]} is not present in RNA_paris_cluster"
    )

# Prefer a distinct Paris cluster per terminal type. Shared modal clusters are skipped
# when another unused cluster still contains that annotation.
used_clusters = set(source_clusters)
terminal_clusters = {}
for cell_type in requested_terminals:
    counts = annotations.loc[
        annotations['clusters'] == cell_type,
        'RNA_paris_cluster',
    ].value_counts()
    for cluster_id in counts.index:
        cluster_id = int(cluster_id)
        if cluster_id not in used_clusters:
            terminal_clusters[cell_type] = cluster_id
            used_clusters.add(cluster_id)
            break

if len(terminal_clusters) < 2:
    raise ValueError(
        "Could not assign at least two distinct sink clusters for "
        f"{requested_terminals}. Paris clusters may mix these labels too strongly "
        "for multi-sink PBA on this store."
    )

terminal_cell_types = list(terminal_clusters)
sink_clusters = list(terminal_clusters.values())
terminal_clusters
```

## 3. Estimate a shared pseudotime

PBA supplies the developmental direction. The assigned terminal clusters are used
as sinks so the ordering covers the branches analyzed below.

```{code-cell} ipython3
pseudotime = ds.run_pseudotime_scoring(
    source_sink_key='RNA_paris_cluster',
    sources=source_clusters,
    sinks=sink_clusters,
    label='fate_pseudotime',
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=pseudotime.pseudotime_key,
    subset_by=pseudotime.validity_key,
)
```

## 4. Compute fate probabilities

`run_fate_mapping` biases the KNN graph toward increasing pseudotime and solves
the absorption probability for each terminal cell type. The Scarf cluster IDs
above direct PBA, while every cell carrying an Alpha, Beta, or Delta annotation
defines the corresponding fate boundary. The PBA validity key excludes any
unscored graph components from the fate calculation. Since that subset key is
not `I`, Scarf includes it in the saved fate column names.

```{code-cell} ipython3
fate = ds.run_fate_mapping(
    cell_key='I',
    subset_cell_key=pseudotime.validity_key,
    pseudotime_key=pseudotime.pseudotime_key,
    sink_key='clusters',
    sinks=terminal_cell_types,
)
```

```{code-cell} ipython3
pd.Series(
    dict(zip(fate.sink_labels, fate.fate_keys, strict=True)),
    name='cell metadata key',
)
```

```{code-cell} ipython3
ds.plots.embedding(
    layout_key='RNA_UMAP',
    color_by=list(fate.fate_keys),
    subset_by=fate.validity_key,
    n_columns=3,
    sort_values=True,
)
```

## 5. Validate the probability simplex

Every valid cell has one probability per sink. The probabilities sum to one.
Cells whose `sink_key` value matches a requested sink have an exact one-hot
boundary.

```{code-cell} ipython3
valid_probabilities = fate.values[fate.valid]
simplex_error = float(
    np.max(np.abs(valid_probabilities.sum(axis=1) - 1.0))
)
assert simplex_error < 1e-5
simplex_error
```

```{code-cell} ipython3
selected_labels = ds.cells.fetch(
    fate.sink_key,
    key=fate.result_cell_key,
)
terminal_checks = []
for index, sink in enumerate(fate.sink_labels):
    rows = (selected_labels == sink) & fate.valid
    own_probability = fate.values[rows, index]
    other_probability = np.delete(fate.values[rows], index, axis=1)
    np.testing.assert_array_equal(
        own_probability,
        np.ones(own_probability.shape[0], dtype=np.float32),
    )
    assert np.count_nonzero(other_probability) == 0
    terminal_checks.append(
        {
            'sink': sink,
            'cells': int(rows.sum()),
            'minimum own probability': float(own_probability.min()),
            'maximum other probability': float(other_probability.max()),
        }
    )

pd.DataFrame(terminal_checks)
```

## Interpretation and limits

The PBA score describes progress along the shared developmental direction.
The fate columns separate that direction into terminal outcomes and quantify
ambiguous intermediate cells. Sink identities remain supervised: this method
does not discover terminal states automatically and does not use RNA velocity.

## Common mistakes and limitations

- Choosing sink clusters that mix several terminal annotations
- Interpreting fate probabilities as lineage commitment without experimental support
- Ignoring the pseudotime validity key when the graph has multiple components
- Expecting the method to find terminal states on its own

## Saved results

Pseudotime and validity columns are written to cell metadata. Fate probabilities are stored
under the keys returned in `fate.fate_keys`, with a matching validity column.

## Further reading

- Weinreb et al. 2018, population balance analysis (PBA): https://doi.org/10.1073/pnas.1714723115
- [PBA reference implementation](https://github.com/AllonKleinLab/PBA)

## Next steps

- {doc}`pseudotime`
- {doc}`pseudotime_modules`
- {doc}`annotation`
- {doc}`plotting`
