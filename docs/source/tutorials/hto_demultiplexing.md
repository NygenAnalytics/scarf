---
description: Assign sample identities from hashtag oligo counts and interpret singlet, negative, and doublet labels.
---
(hto_demultiplexing)=
(hto_demultiplexing_guide)=

# HTO demultiplexing

Cell hashing labels each sample's cells with a sample-specific lipid-tagged antibody oligo, the hashtag, before the samples are pooled into one library. The pool is sequenced as a single experiment, so every droplet carries its RNA reads plus counts of whichever hashtags stuck to it, and those hashtag counts are what tell you which sample a droplet came from. Demultiplexing is the step that reads those counts and sorts the droplets back into their samples: a singlet carries one confident hashtag, a negative carries none, and a doublet carries evidence for more than one hashtag.

Pooling is cheap, and many samples can share one capture, but the savings only pay off if the droplets can be sorted back apart reliably, because a droplet assigned to the wrong sample quietly contaminates the downstream analysis. This is separate from integrating RNA and ADT measurements ({doc}`cite_seq`): demultiplexing classifies droplets as one sample, negative, or doublet before any further downstream analysis.

Scarf follows the strategy of Seurat's HTOdemux function, with hashtag counts being CLR-normalized, cells clustering into one more cluster than there are hashtags so each hashtag has a low-count background cluster. This in turn allows us to fit a negative-binomial model to each hashtag's background counts to set a cutoff for a positive cell. Each droplet is classified by how many hashtags clear their cutoff, with singlets assigned to the hashtag holding the strongest normalized signal.

Here, we run this workflow on a datastore that already carries an HTO assay and inspect the identity labels it returns.

## Run HTO demultiplexing

Scarf expects an HTO assay, named `HTO` by default, in the same datastore format as the biological assays. Open that datastore as `ds` before following the examples. The assay must be declared as type `HTO` when imported. `run_hto_demultiplexing` normalizes the hashtag counts, estimates background, and returns a frozen, unalterable record.

```python
cell_selection = ds.snapshot_cell_selection("I")
identities = ds.run_hto_demultiplexing(cell_selection)
ds.load_artifact(identities)["values"][:]
```

## Interpret singlet, negative, and doublet labels

To interpret the labels that result from running the analysis, we need to inspect the loaded values and compare identity counts with the experiment's expected loading. Singlet labels can define downstream selections or pseudobulk groups; to retain all singlets without creating a new metadata column, select the exact HTO identifiers as shown below:

```python
singlet_labels = ds.HTO.feats.fetch_all("ids").astype(str).tolist()
singlets = ds.select_cells(identities, include=singlet_labels)
int(ds.load_artifact(singlets)["values"][:].sum())
```

Cells marked doublets carry evidence for more than one hashtag and should not be silently relabeled as one sample; though it's often practice to simply drop them before the downstream analysis.

The thresholds for interpreting the singlet, negative, or doublet labels can depend on panel chemistry, loading, and background expression, thus keep this in mind as you remove cells. Review the hashtag count distributions and manually retain or exclude negative and doublet classes according to the analysis question and whether doublet removal is required. The method does not replace RNA doublet scoring because homotypic and untagged multiplets can remain.
