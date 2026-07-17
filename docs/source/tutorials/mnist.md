---
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

## Analysing MNIST image dataset using Scarf

```{code-cell} ipython3
import scarf
import scarf.plotting as splt
import matplotlib.pyplot as plt
import pandas as pd
scarf.__version__
```

---
### 1) Fetch MNIST dataset and convert to Zarr format

The MNIST datasets consists of 60K grayscale images (28x28 pixel) of handwritten digits (0 through 9).
These can be unraveled such that each digit is described by a 784 dimensional vector. 
This dataset is  available to download through Scarf. We saved this data in the same format as
the output of cellranger pipeline with the matrix saved in MTX format.

```{code-cell} ipython3
scarf.fetch_dataset('lecun_60K_mnist_images', save_path='scarf_datasets')
```

```{code-cell} ipython3
reader = scarf.CrDirReader('scarf_datasets/lecun_60K_mnist_images')
writer = scarf.CrToZarr(
    reader,
    zarr_loc='scarf_datasets/lecun_60K_mnist_images/data.zarr',
    chunk_size=(2000, 1000),
)
writer.dump(batch_size=1000)
```

---
### 2) Load the Zarr file into the DataStore object

```{code-cell} ipython3
ds = scarf.DataStore(
    'scarf_datasets/lecun_60K_mnist_images/data.zarr',
    min_cells_per_feature=1000,
    min_features_per_cell=10,
    nthreads=4,
)
ds
```

The labels for each image are embedded in their names. We will extract them add them as a separate column in *cells* attribute table

```{code-cell} ipython3
ds.cells.insert('digit_label',
                [int(x.rsplit('_', 1)[-1])-1 for x
                 in ds.cells.fetch_all('names')], overwrite=True)
```

---
### 3) Creating neighbourhood graph

We will not perform any cell filtering here. We will also skip feature selection step and will use all the valid features. Since, we imported the data as an RNAassay, PCA will be performed on the data. Before we begin the graph creation step, we will turn off the default normlization for an RNAassay.

```{code-cell} ipython3
# Set normalization method to a dummy function that returns unnormalized data
ds.RNA.normMethod = scarf.assay.norm_dummy

ds.make_graph(feat_key='I', k=31, dims=25, n_centroids=100, show_elbow_plot=True)
```

The elbow plot above suggests that taking first 10 PC dimensions might have been optimal this dataset. We wanted to capture even the very fine difference (at risk of larger noise) between the digits and hence do not rerun this with `dims=10`

+++

---
### 4) UMAP embedding and clustering

We will now generate a 2D embedding of the neighbourhood graph of the MNIST images. This will allow to to ascertain visually, how accurate Scarf was in estimating the underlying manifold of this dataset. There are two critical differences between calling UMAP from with Scarf and directly from UMAP library:
- When calling from Scarf, only the neighbourhood graph is provided to the core UMAP algorithm rather a normalized/scaled/reduced data matrix.
- Scarf performs a Kmeans clustering while identifying neighbours, this allows Scarf to generate an informed initial embedding coordinates for the data based on the KMeans centroids. UMAP package, on the other hand, calculates spectral layout for the graph.

```{code-cell} ipython3
ds.run_umap(n_epochs=300, spread=1, min_dist=0.05, parallel=True)
```

Before we visualize the UMAP embeddings, we will also perform clustering on the data (neighbourhood graph) using the default Paris algorithm. Here we choose to perform overclustering of the data so that we can capture fine differences within the individual digit classes.

```{code-cell} ipython3
ds.run_clustering(n_clusters=20)
```

Relabeling the cluster ids to match their corresponding digit labels

```{code-cell} ipython3
ds.smart_label(
    to_relabel='RNA_cluster',
    base_label='digit_label',
    new_col_name='cluster_label',
)
```

```{code-cell} ipython3
splt.embedding(
    ds,
    layout_key='RNA_UMAP',
    color_by=['digit_label', 'cluster_label'],
    legend_loc='on_data',
).figure
```

The UMAP embedding shows that images from the same individual digit classes were grouped together very nicely. We obtained a cluster of digits classes 4,7 and 9 and another of 3, 5 and 8 classses, similar observation has been made before in the link below. Since, the images from classes 0 and 1 are well separated once can infer that the global structure is also well preserved.

https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_mnist_example.html

We can use `splt.cluster_tree` to show an explicit relationship between the clusters of images. This method leverages the hierarchical relationship between the individual images as calculated by the Paris algorithm. Each circle/pie (sized by number of images in the cluster) represents a cluster. The colours inside each pie indicate the proportion actual digit classes.

```{code-cell} ipython3
:tags: []

splt.cluster_tree(
    ds,
    cluster_key='cluster_label',
    fill_by_value='digit_label',
)
```

Finally, lets visualize images from each of the cluster. To do so, we take all images from a given cluster and merge them (think of it can creating an overlay of all the images from that cluster).

```{code-cell} ipython3
clusts = pd.Series(ds.cells.fetch_all('cluster_label'))
digits = pd.Series(ds.cells.fetch_all('digit_label'))
```

```{code-cell} ipython3
fig = plt.figure(figsize=(8,2))
for n,i in enumerate(sorted(clusts.unique())):
    mean_map = ds.RNA.rawData[((clusts == i) & (digits == int(i[0]))).values].mean(axis=0)
    mean_map = mean_map.compute().reshape(28, 28)
    ax = fig.add_subplot(2, 10, n+1)
    ax.imshow(mean_map, cmap='binary')
    ax.set_axis_off()
    ax.set_title(i, fontsize=10)
plt.tight_layout()
plt.show()
```

It is quite clear that Scarf's clsutering was able to identify naunced differences between the images. For example, cluster 1a and 1c captured those images of an 'upright' 1s while cluster 1b and 1d captured 'slanted' 1s.

This vignette helped us learn two things:
- Scarf is a flexible package that can handle analysis of diverse kinds of datasets.
- Results from this dataset show that Scarf can perform quick, memory efficient and meaningful analysis of large-scale datasets.

+++

---
For omics workflows that use the same embedding and clustering steps, see {doc}`scrna_seq`.
