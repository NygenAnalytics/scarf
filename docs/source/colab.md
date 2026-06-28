(colab)=
# Run tutorials on Google Colab

## Wish to try Scarf without installation?

Google Colab allows running Python code directly on Google's server through a notebook interface.
With the following links you can try running any of the vignettes on Colab.

## Before you run notebooks on Colab

Paste the following at the top of the notebook before running any other cell:

    !pip install ipython-autotime
    !pip install "scarf[extra]"

Scarf requires Python 3.14. Use a Colab runtime that provides 3.14, or a local environment instead.
If dependency errors appear, restart the runtime after installing packages.

## Colab links

### Basic pipelines

- [Workflow for scRNA-Seq data](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scRNAseq.ipynb)
- [Workflow for scATAC-Seq count matrices](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scATACseq.ipynb)

### Multi-omics/Multimodal analysis

- [Analysis of Transcriptome + Surface Proteome](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/multiple_modalities.ipynb)

### Data integration tutorials

- [Projection of cells across datasets](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/data_projection.ipynb)
- [Merging datasets and partial training](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/merging_datasets.ipynb)

### Trajectory analysis tutorials

- [Estimating pseudotime ordering and expression dynamics](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/pseudotime_dynamics.ipynb)

### Other Vignettes

- [Understanding how data is organized in Scarf](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/zarr_explanation.ipynb)
- [Getting data in and out of Scarf](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/download_conversion.ipynb)
- [Cell subsampling using TopACeDo](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/cell_subsampling_tutorial.ipynb)
- [Estimate cell-cycle phases](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/cell_cycle.ipynb)
- [Demonstrating Scarf on MNIST dataset](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/mnist.ipynb)
