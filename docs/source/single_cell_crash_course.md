---
description: Contextual background for single-cell RNA-seq analysis
---


# Single Cell Crash Course

This course attempts to give you the rough contextual background behind single-cell RNA-seq analysis before you apply yourself.

A quick note on some general vocabulary used here: a *barcode* is one row of the matrix and usually means one droplet and (usually) one cell. A *feature* is one column: a gene, peak, or protein tag. *Active* means selected for analysis. A *result* means a named, stored output with a record of what produced it. See {doc}`reference/glossary` for the full definitions.

## Section 1 — Bulk RNA-seq of a tissue is a mixture; single-cell keeps the distribution of each cell

Bulk RNA-seq typically functions by grinding up an entire tissue, and reporting the average expression of genes in the sample.
The caveat is if one rare cell type turns a gene on completely while everything else stays silent, the average barely moves and you miss it in Bulk RNA-seq.
Single-cell RNA-seq attempts to solve this by isolating cells into droplets, then tagging each cell's molecules with a barcode, sequencing everything, and counting molecules per cell per gene.
For example, say you have 2 T cells that have 12 copies of `CD3D` each, and 1 B cell at 0; in bulk, this would result in an average of 8 'CD3D' counts; instead, single-cell reports `[12, 12, 0]`, the same average but with specificity now.

The 5K PBMC dataset in {ref}`Quick start <quickstart>` holds monocytes, B cells, and several T-cell
states that cannot be identified through Bulk RNA-seq.

In Scarf, everything downstream starts from count matrices of cells and genes, imported with a `*Reader` plus `*ToZarr` writer (see {doc}`tutorials/import_and_export`);
Once you open a store for scarf, the entire analysis can be performed, giving you one item to inspect.

```python
ds = scarf.DataStore("scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr")  # from quickstart
print(ds.RNA.rawData.shape)  # (cells, genes): rows are barcodes, columns are genes
run = ds.pipeline.open(label="docs_default")  # named completed run, reused in every sketch below
```

However it is important to consider that most single-cell distributions are sparse: many zeros of mixed origin,
true silence plus missed capture, with a few high values, so ensure that you keep this in mind while analyzing.

## Section 2 — From droplets to a count matrix, and what each count generally means

During the wet lab process of single cell transcriptomics, cells are often isolated into droplets with barcoded beads, broken open so mRNA binds the bead, and then
reverse-transcribed and amplified with the barcode attached.
Following these steps, the droplets are then sequenced and counted per gene-barcode combination.
Unique molecular identifiers (UMIs) tag individual molecules so PCR duplicates collapse to one count: without them, a molecule copied a thousand times would
outshout a thousand distinct molecules copied once each, and amplification noise would pose
as biology.


```python
reader = scarf.CrH5Reader("counts.h5")  #
scarf.CrToZarr(reader, zarr_loc="data.zarr").dump()
ds = scarf.DataStore("data.zarr") 
```

It is important to note that a count is a sampled molecule, not absolute truth. Twelve counts means 12 repeated captures.
On the other hand, zero CAN BE ambiguous: silence, or a few molecules that missed capture.
Counts also scale with depth, meaning that sequencing with twice as much depth can increase every count with zero *actual* biological change.
Good practice can be considered to plot a histogram of total counts per barcode (cell).

```python
ds.cells.to_pandas_dataframe(
    columns=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"]
).describe().loc[["min", "50%", "max"]]
```

## Section 3 — Technical variation can appear as true biology unless proved otherwise

Take the following thought experiment for example:
Say you have 2 perfectly identical cells, and you sequence them with the same machine, same conditions, yet your final product has 2 different count vectors.

Capture efficiency, sequencing depth, ambient soup from burst cells, doublets sharing a barcode, batch shifts across day
or lane, and flickering sparsity all move and influence counts without touching the actual biology, and each can
impersonate an entirely different cell type during analysis.
Regardless, with advancements in sequencing technology, these large differences occur less often and it's important not to fearmonger over them, but rather be aware of what may be driving deviation in your data.

Consider how easily technical noise mimics real biology. Double T1's sequencing depth relative to T2, and its entire expression profile doubles with absolutely nothing changed in the cell. Drop ambient LYZ at a mere 1 to 2 counts into the mix, and even silent cells seem to whisper it. Real, uncorrected batches will separate slightly on a plot, and burst red blood cells will haunt clean lymphocytes with ghost counts of HBB.

To rule out issues in your analysis, always quantify depth, detected genes, and mitochondrial share per cell. You can do this by coloring the embedding by sequencing depth to get a visual depiction of the data. We will learn about what these depictions are later!

```python
ds.plots.embedding(layout=run["umap"], color_by="RNA_nCounts")
ds.plots.embedding(layout=run["umap"], color_by=run["clusters"])
```

A difference is biological when it aligns with known markers, replicates, and diverse batches. It is technical when it tracks strictly with depth or processing artifact.

For instance, *MKI67* reads 0 in 99.4% of this dataset's 5,025 cells and a handful of counts in just 28, nearly two-thirds of them *CD3D*+ T cells: sparse, marker-consistent, a cycling state worth pursuing rather than noise.

Conversely, monocytes here total nearly double the rest, a median near 14,800 counts against roughly 7,500, on the back of *LYZ* averaging 50 with cells past 900 plus *S100A9* and *S100A8*: normalize before admiring any monocyte "upregulation." Splits that replicate across datasets and donors are biology. Splits that live exclusively in donor could be outliers instead.


```python
import numpy as np

feats = ds.RNA.feats.to_pandas_dataframe(columns=["names", "nCells"]).set_index("names")
print(feats.loc[["MKI67", "RPS27"]])  # 28 vs 4864 cells: sparsity vs ubiquity, one table

order = list(ds.RNA.feats.fetch_all("names"))
def gene(name):
    i = order.index(name)
    return np.asarray(ds.RNA.rawData[:, i:i + 1].compute()).ravel()
mki, cd3 = gene("MKI67"), gene("CD3D")
print("MKI67+:", int((mki > 0).sum()), "of which CD3D+:", int(((mki > 0) & (cd3 > 0)).sum()))

nc = ds.cells.fetch_all("RNA_nCounts")
mono = gene("LYZ") >= np.percentile(gene("LYZ"), 90)
print("monocyte median vs rest:", np.median(nc[mono]), np.median(nc[~mono]))
```

## Section 4 — Quality control is often the most significant part of your analysis

Filtering out cells during quality control (qc) is the most critical part of the analysis.
All of the work done during transcriptomics analysis depends on QC, as it aids in distinguishing between technical artifacts versus true biological significance (see above).
Generally, cells with near-zero counts go, damaged cells with few genes and high mitochondrial share go, and doublets wait because they are a different problem than low quality.
Modern filtering marks rows rather than deleting them (which other software does!), thus thresholds stay inspectable and reversible, and cutoffs come from the tails of your own distributions, never another tissue's numbers. This allows you to try multiple different thresholds to see what fits your data the best.

Generally, work done with PBMCs filters roughly 1000–15000 counts, 500–4000 genes, mito under 15 percent, and then you can visualize your distributions before and after.. This is a general guideline and potential starting point, remember, do what fits your data the best.

```python
qc_sel = ds.filter_cells(attrs=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"], lows=[1000, 500, 0], highs=[15000, 4000, 15])

```

QC also filters columns, not just rows. Genes seen in only 1 or 2 cells inflate sparsity, distort dispersion estimates during later variable-gene selection, and consume memory for zero information. We can adjust for this reducing the amount of cells a gene needs to be inside off to be kept. 

```python
gene_sel = ds.select_detected_features(qc_sel, min_cells=5)  
```

Single-cell data can also contain doublets, which are simply 2 cells appearing together, and thus getting sequenced together, yielding a larger, irregular cell in comparison to the rest of the data. Doublets instead can be addressed in 2 ways, either by removing them before the downstream analysis occurs, or by marking doublets, keeping them in your final embeddings, and during visualization, then potentially remove them. 

```python
doublet_run = ds.pipeline.run( filtering={"method": "manual", "attrs": ["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"], "highs": [15000, 4000, 15], "lows": [1000, 500, 0]}, hvg_count=500, pca_dims=15, leiden={"partitions": [0.5]}, cell_cycle=False, paris=False, doublets=True, markers=False,) 

doublets = doublet_run["doublets"]  # score artifact, not a deletion
scores = np.asarray(doublet_run.cells.fetch("doublet_score"))
pd.Series(scores, name="doublet_score").plot(kind="hist", bins=40)  # read the tail first
```

Path 2 keeps them in and looks first. Color the embedding by score and read where the doublets live: bridges and fringes lighting up together means doublets form their own crowd, while scattered sparks across clusters means random doublets thoroughout:

```python
ds.plots.embedding(run=doublet_run, color_by="doublet_score", sort_values=True)
```

Path 1 removes by percentile threshold before any downstream analysis. Use histograms to get a better idea of what cutoffs you should set upon your own data, taking into account the shape of the tails of your data. 

```python
cut = float(pd.Series(scores).quantile(0.95))  # example cutoff, not a constant
clean = ds.select_cells(doublets, high=cut, keep_bounds=True)  
```

## Section 5 — Normalization allows you to compare cells; Feature Selection lets you put different lenses on

Cells sequenced at different depths cannot have their gene expression counts compared. Scaling each profile to a common
size factor and log-transforming tames the skew so a unit of difference reads roughly as fold-change. Normalized values are more comparable values:

```python
sel = ds.snapshot_cell_selection("I")
norm = ds.run_normalization(sel, run["highly_variable_features"])
```

Most genes are uninformative: silent or constant everywhere, or pure noise. A few hundred to a few thousand vary in ways that track identity, and highly variable gene selection models the mean-variance trend to keep genes varying beyond expectation for their abundance. 

Each default exclusion earns its usecase: mitochondrial patterns track damage and leakage, whereas ribosomal ones track depth and translational bustle.
Cell-cycle genes cluster cell phases instead of types, and sex-linked genes split donors instead of cell states.

```python
hvg = ds.select_hvgs(sel, top_n=500)
ds.inspect_artifact(hvg).parameters  
```


## Section 6 — Principal Component Analysis shrinks the data down, graphs link similar cells, and UMAP draws the map

Genes often move together in programs, so the true number of dimensions is far below the
gene count. Principal Component Analysis (PCA) finds the main axes of variation, ranked by how much variance each explains. This allows us to simplify from higher dimensional data, and get the 2D/3D representations of the data.
Usually, you can use an elbow plot to know where to cut off your data.

```python
pca = ds.run_pca(norm, dims=15, show_elbow_plot=True)
ds.inspect_artifact(pca).parameters  # name each kept axis before using it
```

If you want to get a little more in-depth, in the PCA space, we get each cell's k nearest neighbors by putting them on a weighted graph based on what we get from PCA.
The graph, not the matrix, is what embeddings, clusters, trajectories, and imputation all use.

```python
index = ds.build_ann_index(pca)
neighbors = ds.query_neighbors(index, k=11)
graph = ds.build_connectivity_map(neighbors)  # everything downstream consumes this
```

Most of the plots you see in recent publications use the Uniform Manifold Approximation and Projection (UMAP) method to plot the data in 2 dimensions.

Essentially, all UMAP does is place cells from the PCA space so graph neighbors stay close.
Nearby placements indicate cells similar to one another, with distances between 'islands' showing that one group of cells is different from another. The empty spaces and distances are not measurements. When you select a seed, all it does reshape the drawing without touching the graph. The graph always stays the same, but the way we can project our data differs.

```python
init = ds.build_embedding_initialization(pca)
umap = ds.run_umap(graph, init)  # calculates the coordinates only
```

The difference between PCA and UMAP can be seen below. Layouts must be two columns,
so take the first 2 of the 15 PCA axes explicitly (the UMAP artifact is already 2D;
sliced the same way for symmetry):

```python
import matplotlib.pyplot as plt
import numpy as np

pc = np.asarray(ds.load_artifact(pca)["data"][:, :2])  # first 2 of 15 dims
xy = np.asarray(ds.load_artifact(umap)["values"][:, :2])  # same cells, UMAP plane
lab = np.asarray(ds.load_artifact(run["clusters"])["values"])

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
left.scatter(pc[:, 0], pc[:, 1], c=lab, s=3)
left.set_title("PCA: first 2 of 15 axes")
right.scatter(xy[:, 0], xy[:, 1], c=lab, s=3)
right.set_title("UMAP: same cells, neighborhood drawing")
```

No biological pattern will depend on the seed or parameters you use. All this does is tune how you see the picture.

## Section 7 — Clustering finds the groups, markers identify them

Community detection cuts the graph where edges run sparse, so dense pockets become
clusters.

Resolution sets how fine the groups are, with higher values giving you more communities and groups.
You can use hierarchical Paris clustering as well to get a second view of the same graph with its own flavor. There is no single best resolution, only the resolution that best allows you to represent your data based on what you expect.

```python
leiden = ds.run_leiden_clustering(graph, resolution=0.5)
paris = ds.run_paris_clustering(graph)  # hierarchical second view, same graph
```

Stable blocks across methods and resolutions are populations. Flickering boundaries could be hypotheses to investigate. Extra high-resolution clusters are not new types until proven, and may just be part of a larger population.

To determine the identity of the cluster/community, you can rank genes and identify potential "marker genes" that represent the cell's identity. You can then visualize the spatial orientation of these certain genes on your UMAP to identify what communities may correspond to what cell identity.

```python
labels = run.cells.fetch("clusters")
top = ds.get_markers(marker=run["markers"], group_id=labels[0],
                     min_score=0.1, min_frac_exp=0.1)  # positives AND negatives both matter
```

It is important to note that when you assign an identity to a cluster/community you are applying a heavy subjectivity, as rooting cell type annotations in ground truth can be a difficult task.

## Section 8 — Testing differences, counting cells, and proving things with replicates

Groupwise tests rank candidates with correction across the gene family. It is important
to note that with thousands of cells, even tiny shifts appear extremely statistically significant. 
Take the simplest case to start, you want to see if a gene's expression changes across 2 conditions, take ISG15 in our example.
Here, we have control and stimulated cells, and to see if there may be a difference we can quickly run a Welch's t-test, which stays descriptive at the cell level. This is useful for **quick**, **exploratory** comparisons, such as checking whether a target like ISG15 shifts between control and stimulated cells in a single experiment. Since this is a cell-level test, we may see extreme significance and get a descriptive summary of some potential effect.

```python
from scarf.plotting import CellField
res = ds.run_statistical_testing("ISG15", grouping=CellField("sample_id"), test="welch")
# Effect first: mean_1, mean_2, mean_difference in res.tables["ISG15"], p value is shown second
```
It can also be useful to ask "does the composition of my cell types change across the condition (i.e., treated vs. untreated).
Cell type composition is a different question entirely, one that can be particularly insightful, and it lives at the sample level, not the cell level. Usually, you tally proportions per independent sample directly from raw metadata, which can avoid issues like pseudoreplication.

```python
tally = ds.cells.to_pandas_dataframe(columns=["sample_id"])
tally["cluster"] = run.cells.fetch("clusters")
tally.groupby(["sample_id", "cluster"]).size()  # per-sample, never pooled
```

Condition claims with proper replicates often need one more step to point out key differences/changes. This is where we can aggregate counts within each cell-type into bulk-like profiles to replicate the bulk-RNAseq modality. We can do this because summing the counts for our genes often yields a distribution that we would observe in bulk data for that sample. The downside of this is trading resolution for valid inference of our hypothesis. However, we still get a greater degree of resolution as we can bulk specific cell types of interest that we want to investigate. 

It is impertive to know that when you perform pseudobulking, you must use **raw** counts, not the normalized or log-transformed counts. This must be done because 

```python
bulk = ds.make_bulk(groups=run["clusters"], aggr_type="sum")
# bulk rows are replicates now: export to DESeq2/edgeR, not a cell-level test.
```

## Section 9 — Gene programs, cell states, cell talk, and predicting a cell's "journey" through time

We know that genes, when they result in some effect, don't act alone.
They often set off a cascade, acting in coordinated networks.
These coordinated networks are often gene programs, and program scoring in single-cell transcriptomics can transform the noisy dropout-heavy data into something with potential meaning.
Some gene sets can show high activity with strong confidence even if some individual genes don't reach significance on their own. The concerted shift, however macro or minor, can be found with program scoring.
Below, we use AUCell, which allows us to evaluate whether a gene program is active inside of a cell.

```python
import pandas as pd

net = pd.DataFrame({"source": "T_program", "target": ["CD3D", "CD8A", "IL7R"]})
sel = ds.snapshot_cell_selection("I")
act = ds.run_aucell(net, sel, features=run["highly_variable_features"], tmin=2)
# Using AUCell; allows us to evaluate whether a gene program is active inside of a cell.
# tmin=2 keeps this 3-target demo set: sources with fewer matched targets are removed,
# so every target must sit inside the HVG universe used here.
```

While tools like AUCell can help determine gene programs enabled inside of individual cells, performing cell-cell communication analysis can allow you to speculate what cells may be sending signals and what cells may be receiving these signals.

Cell-cell communication tools match senders and receivers against prior ligand-receptor
databases. Every inferred interaction is speculation rooted in some ground-truth biology. The key thing to note is that with how cells are processed for single-cell transcriptomics, the dissociation destroys our spatial context, so co-expression alone cannot confirm proximity, direction, or downstream activation. Meaning that even if your ligand and receptor pairs appear to be near each other in terms of your 2D visualization, those actual cells may be millimeters apart inside of the actual tissue.

Pseudotime is a statistical measure of transcriptional similarity, not elapsed wall-clock time. Because single-cell sequencing captures a static snapshot of destroyed cells, trajectory inference projects those cells along a path by stretching graph-based similarities into a continuum. That continuum has no intrinsic direction: no algorithm can infer biological origins de novo, and simply swapping the declared root flips the trajectory arrow across completely unchanged data. In classic pancreas development models, progenitors resolve cleanly toward alpha, beta, and delta fates only because the start and endpoints are anchored with externally validated marker phenotypes. To structure this properly from initial graph construction through continuous modeling, see {doc}`tutorials/pseudotime` and track downstream shifts in {doc}`tutorials/expression_dynamics`.

Supervision looks like this in practice: a zero-sum source/sink vector built from declared labels, scored on the graph.

```python
import numpy as np

graph = run["connectivity_map"]  # trails are walked on the graph, not the matrix
labels = ds.cells.fetch("clusters", key="I")
source = labels == "Ductal"  # declared root: supervision, not discovery
sink = np.isin(labels, ["Alpha", "Beta", "Delta"])  # declared termini
ss_vec = np.zeros(len(labels), dtype=float)
ss_vec[source] = -1.0 / source.sum()
ss_vec[sink] = 1.0 / sink.sum()  # must sum to zero: sources negative, sinks positive

ptime_ref = ds.run_pseudotime_scoring(graph, ss_vec=ss_vec)
ptime = ds.load_pseudotime_scoring(ptime_ref)  # .values, .valid mask, .graph, .ref
print("valid cells:", int(ptime.valid.sum()))  # disconnected cells score NaN, valid False
```

Every trajectory claim must explicitly state how the root was supervised. Before presenting any path as real biology, demand four internal validity checks: the graph must maintain connected components across densely populated manifold space rather than jumping across empty voids; established lineage markers must trend monotonically along the axis; co-regulated gene modules must turn on and off in coordinated succession; and branching fate probabilities must resolve into distinct lineages rather than leaking indiscriminately across states. You can evaluate and verify these transitions in {doc}`tutorials/fate_mapping` and {doc}`tutorials/trajectory_validation`. Finally, test your ordering against basic library metrics: if your pseudotime axis strongly correlates with total UMI counts, you have mapped a technical sequencing depth gradient rather than a biological transition. Require graph continuity, monotonic marker trends, coordinated modules, and validated termini before reporting any trajectory conclusion.

## Section 10 — Batch effects, replicates, and knowing when to trust your results

Batches shift measurements vastly, and correction pulls them together while trying to
preserve type structure.
Most batch effects can be described as technical artifacts, such as sequencing the same tissue on different days, sequencing on different lanes, or some minor handling differences in terms of the tissue.
Single-cell datasets that contain multiple donors will require batch-correction methods to regress out the technical effects while also preserving true biological information.
Undercorrecting for batch effects can leave the data plagued with noise, while overcorrecting can remove true biological differences.
One method you can utilize to regress out technical effects is Harmony as shown below.

```python
fixed = ds.run_harmony(run["pca"], batch_columns=["batch"])
# Harmony corrects reduced coordinates between PCA and neighbor search, where the graph is built.
```

We can visualize the before and after of batch effect correction, with the plot on the left being the uncorrected graph, whereas the graph on the right shows the results after Harmony batch correction.

```python
fixed_index = ds.build_ann_index(fixed)
fixed_graph = ds.build_connectivity_map(ds.query_neighbors(fixed_index, k=11))
fixed_init = ds.build_embedding_initialization(fixed)
fixed_umap = ds.run_umap(fixed_graph, fixed_init)

ds.plots.embedding(layout=run["umap"], color_by="batch")  # before: batches apart
ds.plots.embedding(layout=fixed_umap, color_by="batch")  # after: batches mixed, types kept
```

## Conclusion

Single-cell analysis preserves per-cell distributions through careful quality control, normalization, feature selection, graph construction, and validated interpretation. Carry that workflow into the tutorials below.

---

## Where to go next

Finished the short reading? The tool tutorials assume exactly what you now know. Start with
{ref}`Quick start <quickstart>`, then the complete {doc}`tutorials/scrna_seq` workflow.
Coming from another ecosystem, read {doc}`scanpy_and_seurat` first. For method-choice depth,
work through {doc}`tutorials/quality_control`, {doc}`tutorials/feature_selection`,
{doc}`tutorials/graph_construction`, and {doc}`tutorials/clustering` in order, then
{doc}`tutorials/data_organization` for the storage model that makes reverting safe.
