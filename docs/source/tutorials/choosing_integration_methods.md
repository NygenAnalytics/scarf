---
description: Decide whether to merge, correct, map, or combine modalities.
---

(integration_guide)=

# Choosing between integration and mapping

These tasks solve different problems. Start from the relationship between the
cells and the question you need to answer.

## Merge without correction

Use {doc}`data_integration` when datasets use a compatible feature space and you
need one datastore for joint inspection. Merging aligns features and metadata.
It does not claim that source effects are unwanted or make populations
comparable.

## Correct a merged dataset

Use {doc}`batch_correction` when several datasets should share one graph and you
have a defensible technical batch variable.

- Partial PCA learns a basis from a trusted reference subset. It is useful when
  one sample defines the space in which other cells should be represented.
- Harmony adjusts PCA coordinates across one or more batch columns. It is useful
  when several technical batches should contribute symmetrically.

Neither method can distinguish technical from biological variation when the
batch variable is confounded with the condition of interest. Keep uncorrected
counts for differential expression and use {doc}`integration_metrics` to assess
both mixing and biological preservation.

## Map queries to a fixed reference

Use {doc}`mapping_and_label_transfer` when the reference model must remain fixed
and query cells should receive positions, evidence, or labels in that reference
space. Mapping is preferable to joint correction when new queries arrive over
time or must be compared against the same atlas.

Use {doc}`reference_atlases` when the reference itself needs to be built,
serialized, reloaded, and validated for repeated mapping.

## Combine modalities measured in the same cells

Use {doc}`cite_seq` for the recommended RNA and ADT workflow or another
paired-modality design. SNN merges edge support from two or more assays. WNN
accepts exactly two assays and varies their relative contribution by cell.
After building the integrated graph, use {doc}`multimodal_integration` to
diagnose modality agreement and compare the resulting partitions. These methods
combine modalities, not batches of independent cells.

Scarf does not provide Scanorama, BBKNN, scVI, or ComBat. Export a suitable
selection through `to_anndata` or another supported format when an external
method better matches the study design.
