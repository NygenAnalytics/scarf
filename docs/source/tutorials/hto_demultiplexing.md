---
description: Assign sample identities from hashtag oligo counts and interpret singlet, negative, and doublet labels.
---

(hto_demultiplexing)=
(hto_demultiplexing_guide)=

# Demultiplexing cells with HTOs

Hashtag oligo (HTO) counts identify the sample assigned to each droplet in a pooled experiment.
This is separate from integrating RNA and ADT measurements: demultiplexing classifies droplets as one sample, negative, or doublet before sample-level comparisons.

## 1. Run HTO demultiplexing

Scarf expects an HTO assay, named `HTO` by default, in the same datastore as the biological assays.
`run_hto_demultiplexing` normalizes the hashtag counts, estimates background, and returns an immutable
identity artifact without changing shared cell metadata.

```python
cell_selection = ds.snapshot_cell_selection("I")
identities = ds.run_hto_demultiplexing(
    cell_selection,
    from_assay="HTO",
)
ds.load_artifact(identities)["values"][:]
```

## 2. Interpret singlet, negative, and doublet labels

Inspect the loaded values and compare identity counts with the experiment's expected loading.
Singlet labels can define downstream selections or pseudobulk groups. To retain all singlets
without creating a metadata column, select the exact HTO identifiers:

```python
singlet_labels = ds.HTO.feats.fetch_all("ids").astype(str).tolist()
singlets = ds.select_cells(identities, include=singlet_labels)
```

Negative cells do not have a confident hashtag assignment.
Doublets carry evidence for more than one hashtag and should not be silently relabelled as one sample.

Thresholds depend on panel chemistry, loading, and background.
Review the hashtag count distributions and manually retain or exclude negative and doublet classes according to the analysis question.
The method does not replace RNA doublet scoring, because homotypic and untagged multiplets can remain.

## 3. Catalog limitation

Scarf's public dataset catalog does not currently contain a cell-hashing dataset, so this page cannot provide an executable result without inventing unrepresentative data.
Adding a licensed public HTO dataset to the catalog is required before this guide can become executable.
See the {doc}`../reference/api/datastore` reference for the current signature.
