---
description: Compare an uncorrected rheumatoid arthritis PBMC graph with Harmony batch correction.
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

(harmony_batch_correction)=

# Correcting batch effects with Harmony

This tutorial compares one uncorrected RNA analysis with the same analysis after Harmony correction.
It uses the Binvignat rheumatoid arthritis PBMC dataset from the
[CELLxGENE collection](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b)
associated with the [study in JCI Insight](https://insight.jci.org/articles/view/178499).
The tutorial constructs equal cell counts for each sequencing-batch and disease combination, so
technical mixing can be assessed without making disease identical to batch. This sampling is not
a donor-balanced biological design.

The code downloads the
[versioned H5AD file](https://datasets.cellxgene.cziscience.com/3b751975-34bb-409a-a9b7-98380f0450ea.h5ad)
to local disk, converts it to a local Zarr store, and runs every analysis locally. Passing the
download URL to `urlretrieve` transfers the file only. It does not make Scarf compute against a
remote dataset.

## 1. Prepare the local source store

Download the H5AD file only when it is absent. Inspecting it before conversion makes the matrix
choice and dimensions explicit. This tutorial requires the integer-like count matrix at `raw/X`
with 108,717 cells and 21,648 features.

```{code-cell} ipython3
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scarf

scarf.configure_output(level="ERROR", progress=False)

dataset_directory = Path(environ.get("SCARF_DOCS_DATA_DIR", "scarf_datasets"))
dataset_directory.mkdir(parents=True, exist_ok=True)

download_url = (
    "https://datasets.cellxgene.cziscience.com/"
    "3b751975-34bb-409a-a9b7-98380f0450ea.h5ad"
)
h5ad_path = dataset_directory / "binvignat_ra_pbmc.h5ad"
source_store = dataset_directory / "binvignat_ra_pbmc.zarr"

if not h5ad_path.exists():
    partial_path = h5ad_path.with_suffix(".h5ad.part")
    urlretrieve(download_url, partial_path)
    partial_path.replace(h5ad_path)

inspection = scarf.inspect_h5ad(str(h5ad_path))
assert inspection.matrixKey == "raw/X"
assert inspection.integerLike is True
assert (inspection.nCells, inspection.nFeatures) == (108_717, 21_648)

if not source_store.exists():
    with TemporaryDirectory(dir=dataset_directory) as conversion_directory:
        staged_store = Path(conversion_directory) / source_store.name
        reader = scarf.H5adReader.from_inspect(inspection)
        try:
            scarf.H5adToZarr(
                reader,
                zarr_loc=str(staged_store),
                nthreads=4,
            ).dump()
        finally:
            reader.h5.close()
        staged_store.replace(source_store)
```

Initialize the source with `min_features_per_cell=0`. This keeps the imported cell axis intact
while the tutorial defines its own exact selection.

```{code-cell} ipython3
source = scarf.DataStore(
    str(source_store),
    default_assay="RNA",
    min_features_per_cell=0,
    nthreads=4,
)
```

Mount the source into a temporary writable analysis store. Count matrices remain in the local
source store, while the selection and pipeline artifacts are written below the temporary
directory. Keep `analysis_directory` bound for as long as the mounted datastore is in use.

```{code-cell} ipython3
analysis_directory = TemporaryDirectory()
analysis_store = Path(analysis_directory.name) / "ra_batch_demo.zarr"

ds = scarf.mount_datastore(
    str(source_store),
    at=str(analysis_store),
    default_assay="RNA",
    min_features_per_cell=0,
    nthreads=4,
)
```

## 2. Create a balanced 9,000-cell analysis

There are three batches and two disease groups. Select 1,500 cells from every batch and disease
combination with one seeded generator. The resulting `docs_ra_batch_demo` column contains exactly
9,000 cells and is reproducible from the same source file.

```{code-cell} ipython3
batch = ds.cells.fetch_all("batch").astype(str)
disease = ds.cells.fetch_all("disease").astype(str)
batch_values = np.unique(batch)
disease_values = np.unique(disease)

assert batch_values.size == 3
assert disease_values.size == 2

rng = np.random.default_rng(42)
selected = np.zeros(ds.cells.N, dtype=bool)

for batch_value in batch_values:
    for disease_value in disease_values:
        candidates = np.flatnonzero(
            (batch == batch_value) & (disease == disease_value)
        )
        assert candidates.size >= 1_500
        chosen = rng.choice(candidates, size=1_500, replace=False)
        selected[chosen] = True

assert selected.sum() == 9_000
ds.cells.insert(
    column_name="docs_ra_batch_demo",
    values=selected,
    overwrite=True,
)

pd.crosstab(batch[selected], disease[selected])
```

The constructed subset has equal cell counts for every batch and disease combination, but donor
representation is not balanced. Disease is a biological condition rather than a correction
covariate, so only `batch` is supplied to Harmony below.

```{raw} html
<span id="harmony"></span>
<span id="harmony-batch-correction"></span>
```

## 3. Run matched uncorrected and Harmony pipelines

Both runs use the same cells, feature count, PCA dimensions, and neighbour count. The only analysis
difference is `harmony_batch_columns=["batch"]` in the second run. Filtering and optional downstream
stages that are not needed for this comparison are disabled.

```{code-cell} ipython3
pipeline_options = {
    "cell_key": "docs_ra_batch_demo",
    "filtering": False,
    "hvg_count": 1_000,
    "pca_dims": 20,
    "neighbors_k": 21,
    "leiden": False,
    "cell_cycle": False,
    "paris": False,
    "doublets": False,
    "markers": False,
    "snapshot_columns": ("batch", "disease", "rough_annot"),
}

uncorrected = ds.pipeline.run(
    label="ra_uncorrected",
    **pipeline_options,
)
harmony = ds.pipeline.run(
    label="ra_harmony",
    harmony_batch_columns=["batch"],
    **pipeline_options,
)
```

Harmony adjusts PCA coordinates before the neighbour graph is built. It does not change the count
matrix. The two frozen runs keep the exact selections, metadata, layouts, and graph artifacts used
for the comparison.

## 4. Compare the layouts

Plot each run once by sequencing batch and once by the imported broad annotation. The 2 by 2 layout
keeps technical mixing and broad cell-type structure visible together.

```{code-cell} ipython3
figure, axes = plt.subplots(2, 2, figsize=(9, 7))
for row, (run_name, run) in enumerate(
    (("Uncorrected", uncorrected), ("Harmony", harmony))
):
    for column, (field, field_name) in enumerate(
        (("batch", "Batch"), ("rough_annot", "Broad cell type"))
    ):
        ds.plots.embedding(
            run=run,
            layout="umap",
            color_by=field,
            target=axes[row, column],
            legend_loc="right" if field == "batch" else "on_data",
            point_alpha=0.8,
            show=False,
        )
        axes[row, column].set_title(f"{run_name}: {field_name}")

figure.tight_layout()
figure
```

(lisi_metrics)=
(integration_metrics)=

## 5. Quantify mixing and structural preservation

iLISI measures local mixing of `batch`. cLISI measures local separation of `rough_annot`, and graph
connectivity measures whether cells sharing that annotation remain connected. Scarf scales all
three metrics so higher values are better. Use the exact neighbour and connectivity artifacts from
each run, with perplexity 7 for both LISI metrics.

```{code-cell} ipython3
def integration_diagnostics(run):
    return {
        "iLISI (batch)": ds.metric_ilisi(
            batch_colname="batch",
            neighbors=run["neighbors"],
            perplexity=7,
        ),
        "cLISI (rough_annot)": ds.metric_clisi(
            annotation_column="rough_annot",
            neighbors=run["neighbors"],
            perplexity=7,
        ),
        "graph connectivity (rough_annot)": ds.metric_graph_connectivity(
            annotation_column="rough_annot",
            graph=run["connectivity_map"],
        ),
    }


score_frame = pd.DataFrame(
    {
        "Uncorrected": integration_diagnostics(uncorrected),
        "Harmony": integration_diagnostics(harmony),
    }
).T
score_frame.round(3)
```

The validated run produced:

| Run | iLISI, batch | cLISI, rough annotation | Graph connectivity, rough annotation |
| --- | ---: | ---: | ---: |
| Uncorrected | 0.159 | 0.988 | 0.975 |
| Harmony | 0.336 | 0.984 | 0.975 |

The higher iLISI after Harmony, together with nearly unchanged cLISI and graph connectivity,
supports improved technical mixing without obvious loss of broad cell-type structure. These metrics
are diagnostics, not proof that correction is biologically valid or that every disease-associated
signal was preserved. The subset equalizes cell counts, not biological replicates, and disease was
not used as a correction covariate. Biological conclusions still require a donor-aware design and
targeted downstream checks.

See {doc}`../reference/api/graph_construction` for the PCA and Harmony contracts, and
{doc}`../reference/api/integration` for the metric definitions.
