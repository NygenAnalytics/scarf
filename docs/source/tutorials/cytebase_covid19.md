---
description: Compare COVID-19 and healthy blood in a published Cytebase dataset using remote cell metadata, the imported UMAP, and donor-level summaries of a few genes.
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
(cytebase_covid19_tutorial)=

# Cytebase case study: COVID-19 blood

This page follows one research question from start to finish, using a published
dataset from Cytebase:

> *How does the peripheral blood of COVID-19 patients differ from that of healthy donors?*

This tutorial goes through more of the downstream analysis that is required to answer questions like the one above. This tutorial utilized a disease-versus-control dataset in the Cytebase catalog, opens it remotely, and answers the question with cell metadata and about two dozen genes. With this, we never have to download the file directly; Scarf reads metadata and the needed count blocks over the network. {doc}`cytebase` introduces the catalog and the SDK calls used here.

The saved results on this page were generated against the public `Nygen/cytebase` bucket, which readers can access without credentials.

{nb-download}`Download the executed Jupyter notebook <cytebase_covid19.ipynb>`.

## Prerequisites

```bash
uv pip install --prerelease allow 'scarf[cytebase]' jupyterlab
```

{doc}`cytebase` describes bucket selection and credentials for private buckets.

## Dataset

The dataset used here is Wilk *et al.* (2020), *Nature Medicine*, [doi:10.1038/s41591-020-0944-y](https://doi.org/10.1038/s41591-020-0944-y): Seq-Well profiling of peripheral blood mononuclear cells (PBMCs) from patients hospitalized
with COVID-19 and from healthy donors. It was curated and distributed by CZ CELLxGENE Discover (CZI Cell Science Program *et al.*, *Nucleic Acids Research*
2025, [doi:10.1093/nar/gkae1142](https://doi.org/10.1093/nar/gkae1142)). Cite both the original study and CELLxGENE Discover when you use these data for your own analysis. All cell-type labels below are the published annotations; this page does not re-cluster or re-annotate.

The setup cell switches off Scarf's progress bars and logs only warnings, to keep
the output readable. It also silences `huggingface_hub` retry messages, which can
include full request URLs.

```{code-cell} ipython3
import logging

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy.stats import mannwhitneyu, spearmanr

import scarf
from scarf import cytebase
from scarf.plotting import CategoricalScale, CellField, NormalizationSpec

scarf.configure_output(level="WARNING", progress=False)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
pd.set_option("display.width", 160)

catalog = cytebase.Catalog()  # Public Cytebase unless CYTEBASE_BUCKET is set.
```

## Find the dataset and open it remotely

`Catalog()` simply downloads a small database file list to your machine, not the actual datasets, Searching against the dataset information doesn't touch any of the stored data.

```{code-cell} ipython3
matches = catalog.search("wilk sars cov 2", ready_only=True, max_cell_chars=None)
match_preview = matches.to_markdown(
    columns=["cytebase_id", "title", "cell_count"], max_cell_chars=32
)
display(Markdown(match_preview))

if not matches:
    raise RuntimeError("The Wilk COVID-19 dataset is not ready in this catalog")
dataset_id = matches[0]["cytebase_id"]
entry = catalog.dataset(dataset_id)
entry
```

`entry` is a catalog record: title, citation, CELLxGENE links, size, and ontology
labels. Displaying it reads the catalog and the dataset's build record, not the
Zarr store.

`open_datastore` validates the published build and returns a read-only
`DataStore`. Only small metadata files are fetched now; counts stay in the bucket
until a specific set of information is requested (like information on a gene). The CELLxGENE UMAP was imported with the data, so it can be plotted directly instead of recomputing an embedding.

```{code-cell} ipython3
ds = catalog.open_datastore(entry.id)
print(ds)
print("\nImported embeddings:", sorted(cytebase.embeddings(ds)))
umap_ref = cytebase.embedding(ds, "X_umap")
```

## Study design from metadata alone

Besides the standard CELLxGENE fields (`cell_type`, `disease`, `donor_id`, and so
on), the store keeps the study's own columns: `Ventilated` and `Admission`
(clinical status), `DPS` (days post symptom onset, 0 for healthy donors), the
authors' coarse and fine labels (`cell.type.coarse`, `cell.type.fine`), and
per-cell scores such as `IFN1`.

Read only the columns needed into a pandas dataframe to summarize clinical metadata per donor.

```{code-cell} ipython3
columns = [
    "cell_type",
    "cell.type.coarse",
    "disease",
    "donor_id",
    "Ventilated",
    "Admission",
    "DPS",
    "sex",
    "development_stage",
    "IFN1",
]
meta = ds.cells.to_pandas_dataframe(["ids", *columns], key="I").set_index("ids")
print(f"{len(meta):,} cells, {meta.shape[1]} metadata columns read")
```

Summarize clinical metadata

```{code-cell} ipython3
def distinct(values):
    return " / ".join(map(str, sorted(values.unique())))


donors = meta.groupby("donor_id").agg(
    disease=("disease", "first"),
    ventilated=("Ventilated", distinct),
    admission=("Admission", distinct),
    days_post_symptoms=("DPS", distinct),
    sex=("sex", "first"),
    age=("development_stage", "first"),
    cells=("disease", "size"),
)
donors
```

The design is small, containing **7 COVID-19 patients (C1 to C7) and 6 healthy donors (H1 to
H6)**, mostly male. Six patients were in intensive care and one (C7) on a general
ward. C1 contributes two samples: day 9 (not ventilated) and day 11 (ventilated).
The donor, not the cell, is the biological replicate, which matters for every
comparison below.

Cell counts per donor range from about 1,700 to 8,400. The CELLxGENE `cell_type`
column uses Cell Ontology names; the authors' `cell.type.coarse` column groups them
into 13 shorter labels, which the compact figures use. Each ontology label maps to
exactly one coarse label:

```{code-cell} ipython3
by_type = pd.crosstab(meta["cell_type"], meta["disease"])
coarse_labels = meta.groupby("cell_type")["cell.type.coarse"].first()
by_type.insert(0, "coarse label", coarse_labels)
by_type.sort_values(["coarse label", "COVID-19"], ascending=[True, False])
```

Raw cell counts already hint at differences (for example, many more plasmablasts
from COVID-19 samples), but they mix biology with how many cells each donor
contributed. Later, we analyze per-donor proportions.

## The published UMAP

`ds.plots.embedding(layout=umap_ref, ...)` draws the imported CELLxGENE
coordinates instead of recomputing them. To begin, first visualize the set of completed annotations:

```{code-cell} ipython3
ds.plots.embedding(
    layout=umap_ref,
    color_by="cell_type",
    legend_loc="right",
    figsize=(12, 7),
)
```

Splitting the same layout by condition with `facet_by` shows where each group's
cells fall.

```{code-cell} ipython3
ds.plots.embedding(
    layout=umap_ref,
    color_by="cell.type.coarse",
    facet_by="disease",
    figsize=(12, 6),
)
```

We can immediately start to see 2 key differences: 1) the plasmablast (PB) island at the top is well populated in COVID-19 but sparse in healthy blood; 2) the CD14 monocyte island is
shifted, with healthy monocytes sitting on the left side, while most COVID-19 monocytes
occupy the right side, so the same annotated cell type appears in a different
transcriptional state.

These panels show where the cells lie, but donors contribute different numbers of cells, thus we can't validate these changes without comparing cell-type proportions with each donor.

## Composition per donor

To validate the changes discussed above, we can calculate the proportions, which are computed within each donor, so a donor with many cells does not dominate. The stacked view shows every donor's blood makeup. With `condition_by`, the `kind="per_sample"` view then places one point per donor for each condition: circles for COVID-19 and squares for healthy donors, with the group mean and 95% confidence interval as a diamond.

```{code-cell} ipython3
coarse_order = [
    "CD4 T",
    "CD8 T",
    "gd T",
    "NK",
    "B",
    "PB",
    "CD14 Monocyte",
    "CD16 Monocyte",
    "DC",
    "pDC",
    "Granulocyte",
    "Platelet",
    "RBC",
]
coarse_scale = CategoricalScale(order=tuple(coarse_order))

ds.plots.composition(
    category_by="cell.type.coarse",
    sample_by="donor_id",
    kind="stacked",
    categorical_scale=coarse_scale,
    figsize=(10, 4.5),
)
```

```{code-cell} ipython3
ds.plots.composition(
    category_by="cell.type.coarse",
    sample_by="donor_id",
    condition_by="disease",
    kind="per_sample",
    categorical_scale=coarse_scale,
    figsize=(10, 7),
    max_figure_width=None,
)
```

```{code-cell} ipython3
fractions = pd.crosstab(
    meta["donor_id"], meta["cell.type.coarse"], normalize="index"
)
shown = ["PB", "CD16 Monocyte", "gd T", "pDC", "NK", "RBC"]
fraction_table = fractions[shown].join(donors["disease"])
fraction_table.groupby("disease")[shown].agg(["min", "median", "max"]).T.round(3)
```

What the per-donor proportions show:

- **Plasmablasts (PB)** make up 0.9% to 20% of cells in COVID-19 donors and at most
  0.8% in healthy donors. Every patient is above the healthy range of PB expression, C7 being there only marginally.
- **CD16 monocytes, gamma-delta T cells, pDCs, and NK cells** are lower in most
  patients. For gamma-delta T cells the two groups have minimal to no overlap.
- **Donor C6 is an outlier**: 54% of its cells are annotated as erythrocytes. Proportions
  sum to one, so such a donor lowers every other fraction, which is one reason to
  look at individual donors and not only group means.

With 7 and 6 donors these are descriptive observations about this cohort. For
formal compositional testing, use a method that accounts for the sum-to-one
constraint and donor-level variation.

## Do the published labels match canonical markers?

Before using the annotations, we can check them against well-known PBMC markers. Scarf resolves gene symbols without regard to case and raises an error for missing or ambiguous names. This panel uses genes present in the published dataset.

Values shown in the dotplot are library-size normalized counts with `log1p` transformation.
`standardize="feature"` rescales each gene so weakly and strongly expressed
markers are equally visible.

```{code-cell} ipython3
marker_sets = {
    "T": ["CD3E", "IL7R", "CD8A"],
    "gd T": ["TRDC"],
    "NK": ["NKG7", "GNLY"],
    "B": ["MS4A1"],
    "PB": ["MZB1", "JCHAIN"],
    "Mono": ["CD14", "LYZ", "FCGR3A"],
    "DC": ["FCER1A"],
    "pDC": ["LILRA4"],
    "Gran.": ["CSF3R"],
    "Plt": ["PPBP"],
    "RBC": ["HBB"],
}
isg_genes = ["IFI27", "ISG15", "IFI44L", "IFI6"]
```

```{code-cell} ipython3
ds.plots.dotplot(
    features=marker_sets,
    group_by="cell.type.coarse",
    group_order=coarse_order,
    normalization=NormalizationSpec(transform="log1p"),
    standardize="feature",
    figsize=(11, 5),
    max_figure_width=None,
)
```

Each population is brightest for its expected markers: TRDC in gamma-delta T
cells, GNLY and NKG7 in NK cells, MS4A1 in B cells, MZB1 and JCHAIN in
plasmablasts, CD14 and LYZ in CD14 monocytes, FCGR3A in CD16 monocytes, FCER1A in
DCs, LILRA4 in pDCs, PPBP in platelets, and HBB in erythrocytes. The published labels are
consistent with these markers, so the rest of the page uses them as given.

## Interferon response

Type I interferon-stimulated genes (ISGs) are a common readout of antiviral signalling. This section looks at four interferons: IFI27, ISG15, IFI44L, and IFI6. In the UMAP, passing `sort_values=True` draws high-expressing cells on top that way they aren't hidden by other cells that express it sparsely.

```{code-cell} ipython3
ds.plots.embedding(
    layout=umap_ref,
    color_by=isg_genes,
    normalization=NormalizationSpec(transform="log1p"),
    sort_values=True,
    n_columns=2,
    figsize=(11, 10),
)
```

ISG15, IFI44L, and IFI6 are expressed across many cell types, with the highest
values in the right half of the monocyte island and in parts of the T and NK area.
IFI27 expression is more restricted, with it being concentrated in the same monocyte region and in the erythrocyte cluster. We can facet IFI27 by condition, on a shared color scale, to visualize
where the signal comes from:

```{code-cell} ipython3
ds.plots.embedding(
    layout=umap_ref,
    color_by=isg_genes[0],
    facet_by="disease",
    normalization=NormalizationSpec(transform="log1p"),
    sort_values=True,
    figsize=(12, 6),
)
```

IFI27 is essentially absent from healthy cells. In COVID-19 samples it is strongest in the shifted monocytes noted above and in the erythrocyte cluster.

### Compare donors, not cells

With tens of thousands of cells, almost any cell-level difference looks "significant", yet the cells come from only 13 people. Treating cells as independent replicates (pseudo-replication) overstates the evidence. Passing `sample_by="donor_id"` makes Scarf average expression within each donor first, so each point below is one donor's mean within a cell type; `split_by="disease"` puts the two conditions side by side. After doing this, we adjust for some of the issues that pseudo-replication brings.

```{code-cell} ipython3
main_types = ["CD14 Monocyte", "CD16 Monocyte", "CD4 T", "CD8 T", "NK", "B"]
ds.plots.distribution(
    isg_genes,
    grouping=CellField("cell.type.coarse"),
    groups=main_types,
    split_by="disease",
    sample_by="donor_id",
    normalization=NormalizationSpec(transform="log1p"),
    share_y=False,
    point_size=6,
    point_alpha=0.85,
    figsize=(12, 6.5),
    max_figure_width=None,
)
```

Healthy donors (orange) sit near zero for all four genes in every cell type, whereas the COVID-19 donors (blue) spread widely: several are far above the healthy range, while a few overlap it. The response is heterogeneous between patients.

To actually see which patients differ, we can summarize CD14 monocytes per donor.

```{code-cell} ipython3
expression = pd.DataFrame(
    {
        gene: np.log1p(ds.get_cell_vals(from_assay="RNA", cell_key="I", k=gene))
        for gene in isg_genes
    },
    index=meta.index,
)
expression["ISG mean"] = expression[isg_genes].mean(axis=1)

expression.head()
```

Restrict the summary to CD14 monocytes and compare donors.

```{code-cell} ipython3
monocytes = meta.join(expression).query("`cell.type.coarse` == 'CD14 Monocyte'")
mono_table = monocytes.groupby("donor_id").agg(
    disease=("disease", "first"),
    admission=("Admission", distinct),
    ventilated=("Ventilated", distinct),
    monocytes=("disease", "size"),
    **{column: (column, "mean") for column in [*isg_genes, "ISG mean", "IFN1"]},
)
mono_table.round(3)
```

This table gives one expression summary per donor.

```{code-cell} ipython3
covid = mono_table.loc[mono_table["disease"] == "COVID-19", "ISG mean"]
healthy = mono_table.loc[mono_table["disease"] == "normal", "ISG mean"]
test = mannwhitneyu(covid, healthy, alternative="two-sided")
rho = spearmanr(mono_table["ISG mean"], mono_table["IFN1"]).statistic

n_above = (covid > healthy.max()).sum()
print(f"Patients above the highest healthy donor: {n_above} of {len(covid)}")
print(
    f"Mann-Whitney U on donor means ({len(covid)} vs {len(healthy)} donors): "
    f"U = {test.statistic:.0f}, two-sided p = {test.pvalue:.4f}"
)
print(
    "Spearman correlation of donor ISG mean with the authors' IFN1 score:",
    f"{rho:.2f}",
)
```

In CD14 monocytes, **six of the seven patients have a higher ISG mean than any
healthy donor**. C1, C4, C5, and C6 are highest (about 0.64 to 0.88, against at
most 0.05 in healthy donors); C2 and C3 are modestly elevated, and C7, the one
patient on a general ward rather than in intensive care, lies within the healthy
range. The authors' `IFN1` score ranks donors in a similar order, so this
short summary agrees with the study's own score. C1's two samples (days 9 and
11) are pooled throughout because the summaries group by `donor_id`.

## Takeaways

Working only from the remote store, with cell metadata and 21 genes:

- **Study design** came from metadata: 7 COVID-19 patients and 6 healthy donors, so
  the donor is the unit for every comparison.
- **Composition**: plasmablasts are expanded in every patient (C7 only marginally),
  while CD16 monocytes, gamma-delta T cells, pDCs, and NK cells are lower in most
  patients.
- **Cell state**: CD14 monocytes from COVID-19 samples occupy a distinct region of
  the published UMAP, the same region with the highest interferon-stimulated gene
  expression.
- **Interferon response**: elevated in most but not all patients. Donor-level
  summaries make this heterogeneity visible, where cell-level statistics would
  hide it.
- **Annotations**: the published labels agree with canonical PBMC markers.

## Limitations

- These are descriptive observations on published annotations from one cohort.
  See the original study for its full analysis and clinical context.
- The Mann-Whitney test compares 7 donor means with 6, so its p-value reflects 13
  biological replicates rather than thousands of cells. Treat it as one
  exploratory test: the genes, cell type, and summary were chosen after looking at
  the data, and 13 donors cannot account for age, sex, sampling day, or treatment.
- The read-only store cannot save results. For new embeddings, clustering, marker
  detection, or donor-level testing with `run_statistical_testing`, mount the
  dataset so new results are written locally while the counts stay in the bucket:

```python
analysis_ds = catalog.mount_datastore(entry.id, at="./analysis.zarr")
```

See {doc}`condition_comparisons` and {doc}`pseudobulk_and_differential_expression`
for formal condition analyses, {doc}`remote_stores` for mounted analysis, and the
{ref}`Cytebase example notebooks <cytebase_example_notebooks>` for a catalog tour
and a UMAP gallery.
