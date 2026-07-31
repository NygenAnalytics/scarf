---
description: Assign sample identities from hashtag oligo counts and interpret singlet, negative, and doublet labels.
---

(hto_demultiplexing_guide)=

# Demultiplexing cells with HTOs

Hashtag oligo (HTO) counts identify the sample assigned to each droplet in a
pooled experiment. This is separate from integrating RNA and ADT measurements:
demultiplexing classifies droplets as one sample, negative, or doublet before
sample-level comparisons.

Scarf expects an HTO assay, named `HTO` by default, in the same datastore as the
biological assays. `mark_hto_identities` normalizes the hashtag counts, estimates
background, and writes the resulting identity to shared cell metadata.

```python
identity_key = ds.mark_hto_identities(
    from_assay="HTO",
    label="sample_id",
)
```

The returned value is the cell metadata key, `sample_id` in this example.
Inspect that column and compare identity counts with the
experiment's expected loading. Singlet labels can define sample-aware QC or
pseudobulk groups. Negative cells do not have a confident hashtag assignment.
Doublets carry evidence for more than one hashtag and should not be silently
relabelled as one sample.

Thresholds depend on panel chemistry, loading, and background. Review the
hashtag count distributions and manually retain or exclude negative and doublet
classes according to the analysis question. The method does not replace RNA
doublet scoring, because homotypic and untagged multiplets can remain.

Scarf's public dataset catalog does not currently contain a cell-hashing
dataset, so this page cannot provide an executable result without inventing
unrepresentative data. Adding a licensed public HTO dataset to the catalog is
required before this guide can become executable. See the
{doc}`../reference/api/datastore` reference for the current signature.
