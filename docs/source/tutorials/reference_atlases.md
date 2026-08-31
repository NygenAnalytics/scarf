---
orphan: true
description: Compatibility route for the former standalone reference-atlas tutorial.
---

(reference_atlas_mapping)=

# Reference atlas tutorial moved

Reference construction, reuse, mapping diagnostics, and label transfer now form one workflow in
{doc}`mapping_and_label_transfer`. A reusable atlas is the fixed `MappingReference` prepared in that
tutorial, together with its reference datastore and separately retained layout artifact.

The former Kang-based example was retired because it assigned one constant batch label to every
control reference cell and another constant label to every stimulated query cell. Those labels
represented biological condition, not measured technical batches. They could exercise the
Harmony/Symphony arguments, but they could not demonstrate batch correction or support a biological
interpretation of correction magnitude.

Continue with {doc}`mapping_and_label_transfer` for the plain-PCA workflow and the requirements for
using a Harmony-backed Symphony reference when genuine technical-batch metadata is available.
