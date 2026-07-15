---
description: Publication-oriented figures with scarf.plotting (embedding, dotplot, composition, export).
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

(plotting_showcase)=

# Plotting with scarf.plotting

`scarf.plotting` is the canonical API for new figures. Import it as `splt`, keep
`DataStore.plot_*` for existing notebooks, and opt into the new embedding path
from `plot_layout` when you want shared scales and a `PlotResult`.

```{code-cell} ipython3
from pathlib import Path

import scarf
import scarf.plotting as splt

DATASET = "bastidas-ponce_4K_pancreas-d15_rnaseq"
repo_root = Path(scarf.__file__).resolve().parents[1]
zarr_path = (
    repo_root
    / "docs"
    / "source"
    / "vignettes"
    / "scarf_datasets"
    / DATASET
    / "data.zarr"
)
if not zarr_path.exists():
    scarf.fetch_dataset(
        dataset_name=DATASET,
        save_path="scarf_datasets",
        as_zarr=True,
    )
    zarr_path = Path("scarf_datasets") / DATASET / "data.zarr"

ds = scarf.DataStore(str(zarr_path), nthreads=4, default_assay="RNA")
```

---

## Embedding

Color by a metadata column or by gene names. Multi-gene panels share one color
scale per gene when you facet.

```{code-cell} ipython3
emb = splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by="clusters",
    show=False,
)
emb.figure
```

```{code-cell} ipython3
genes = ["Gcg", "Ins2", "Sst"]
emb2 = splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by=genes,
    normalization=splt.NormalizationSpec(transform="log1p"),
    sort_values=True,
    show=False,
)
emb2.figure
```

Save a 300 DPI TIFF without cropping the figure size, with a JSON provenance
sidecar:

```{code-cell} ipython3
from pathlib import Path

out = Path("scarf_datasets") / "plotting_showcase_embedding.tiff"
out.parent.mkdir(parents=True, exist_ok=True)
emb.save(out, dpi=300, exact_size=True, provenance_sidecar=True)
assert out.exists()
assert out.with_suffix(".tiff.json").exists()
emb.close()
emb2.close()
```

---

## Blockwise raster embedding

For a large dataset, rasterize continuous cell metadata without loading complete
columns into memory. With no `color_by`, the image shows log-transformed cell
counts per pixel.

```{code-cell} ipython3
raster = splt.embedding_raster(
    ds,
    layout_key="RNA_UMAP",
    color_by="RNA_nCounts",
    pixels=400,
    show=False,
)
raster.figure
```

```{code-cell} ipython3
raster.close()
```

---

## Dotplot and matrixplot

Pass an ordered mapping to keep gene-group brackets. Use `sample_by` so each
sample contributes equal weight to the group summary.

```{code-cell} ipython3
# Toy sample labels for the demo (replace with your real sample column).
n = len(ds.cells.active_index("I"))
ds.cells.insert(
    "demo_sample",
    [f"s{i % 8}" for i in range(n)],
    overwrite=True,
)

dp = splt.dotplot(
    ds,
    features={"endocrine": ["Gcg", "Ins2", "Sst"]},
    group_by="clusters",
    sample_by="demo_sample",
    show=False,
)
dp.figure
```

```{code-cell} ipython3
mp = splt.matrixplot(
    ds,
    features=["Gcg", "Ins2", "Sst"],
    group_by="clusters",
    value="mean",
    show=False,
)
mp.figure
mp.close()
dp.close()
```

---

## Composition (including paired subjects)

`kind='per_sample'` plots one point per sample. With subject and condition
fields in `StudyDesign`, lines connect the same subject across conditions
within each category.

```{code-cell} ipython3
ds.cells.insert(
    "demo_subject",
    [f"d{i % 4}" for i in range(n)],
    overwrite=True,
)
ds.cells.insert(
    "demo_condition",
    ["before" if i % 8 < 4 else "after" for i in range(n)],
    overwrite=True,
)

comp = splt.composition(
    ds,
    category_by="clusters",
    study_design=splt.StudyDesign(
        sample_by="demo_sample",
        subject_by="demo_subject",
        condition_by="demo_condition",
    ),
    kind="per_sample",
    show=False,
)
comp.tables["per_sample"].head()
```

```{code-cell} ipython3
comp.figure
comp.close()
```

---

## Caller-owned axes and compatible `plot_layout`

Draw into an existing axes mosaic, or opt into the new embedding renderer from
`DataStore.plot_layout`:

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplot_mosaic([["A", "B"]], figsize=(8, 3.5))
a = splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by="clusters",
    target=axes["A"],
    show=False,
)
b = splt.embedding(
    ds,
    layout_key="RNA_UMAP",
    color_by="Gcg",
    target=axes["B"],
    show=False,
)
splt.label_panels({"A": axes["A"], "B": axes["B"]}, labels=["A", "B"])
fig
```

```{code-cell} ipython3
# Opt-in bridge: returns a PlotResult when the call is compatible.
result = ds.plot_layout(
    layout_key="RNA_UMAP",
    color_by="clusters",
    show_fig=False,
    use_plotting=True,
)
assert isinstance(result, splt.PlotResult)
result.close()
a.close()
b.close()
plt.close(fig)
```

The legacy `scarf.plots` and `DataStore.plot_*` interfaces remain supported
compatibility APIs without plotting deprecation warnings. Prefer
`import scarf.plotting as splt` for new analysis code.

---

## Distributions

```{code-cell} ipython3
dist = splt.distribution(
    ds,
    keys=["RNA_nCounts", "RNA_nFeatures"],
    group_by="clusters",
    kind="violin",
    max_points=2000,
    seed=0,
    show=False,
)
dist.figure
```

```{code-cell} ipython3
hist = splt.distribution(ds, keys="RNA_nCounts", kind="hist", bins=40, show=False)
ecdf = splt.distribution(ds, keys="RNA_nCounts", kind="ecdf", show=False)
hist.close()
ecdf.close()
dist.close()
```
