#!/usr/bin/env Rscript
# Produces the Seurat side of tests/seurat_wnn_5_5_1_golden.json.
# Driven by scripts/export_seurat_wnn_golden.py, which supplies the shared
# embeddings and neighbour indices and consumes the tables written here.
#
# The matched tables come from Seurat's own routines applied to Scarf's
# candidate pool and bandwidth: L2Norm, PredictAssay, impute_dist and NNdist do
# the arithmetic, and the kernel, score and softmax expressions are copied from
# the body of FindModalityWeights. The default tables come from the public
# FindMultiModalNeighbors with its shipped defaults, which Scarf does not try to
# reproduce.

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: export_seurat_wnn_golden.R <work-dir>")
}
work <- args[[1]]

read_matrix <- function(name) {
  as.matrix(read.table(file.path(work, name), header = FALSE, colClasses = "numeric"))
}

write_matrix <- function(values, name) {
  values <- as.matrix(values)
  lines <- apply(values, 1, function(row) paste(sprintf("%.17g", row), collapse = "\t"))
  writeLines(lines, file.path(work, name))
}

rna <- read_matrix("rna_embedding.tsv")
adt <- read_matrix("adt_embedding.tsv")
# Python writes zero-based, self-free neighbour rows.
rna_idx <- read_matrix("rna_indices.tsv") + 1
adt_idx <- read_matrix("adt_indices.tsv") + 1

n_cells <- nrow(rna)
k <- ncol(rna_idx)
cell_names <- paste0("cell", seq_len(n_cells))
rownames(rna) <- rownames(adt) <- cell_names
colnames(rna) <- paste0("PC_", seq_len(ncol(rna)))
colnames(adt) <- paste0("APC_", seq_len(ncol(adt)))

counts <- matrix(
  1,
  nrow = 5,
  ncol = n_cells,
  dimnames = list(paste0("gene", 1:5), cell_names)
)
object <- CreateSeuratObject(counts = counts, assay = "RNA")
object[["pca"]] <- CreateDimReducObject(embeddings = rna, key = "PC_", assay = "RNA")
object[["apca"]] <- CreateDimReducObject(embeddings = adt, key = "APC_", assay = "RNA")

reductions <- c("pca", "apca")
dims <- list(pca = seq_len(ncol(rna)), apca = seq_len(ncol(adt)))
indices <- list(pca = rna_idx, apca = adt_idx)

# Seurat's own row normalization, matching l2_normalize=True on the Scarf side.
normed <- lapply(reductions, function(r) Seurat:::L2Norm(mat = Embeddings(object, r)[, dims[[r]]]))
names(normed) <- reductions
for (r in reductions) {
  object[[paste0(r, ".norm")]] <- CreateDimReducObject(
    embeddings = normed[[r]],
    key = paste0("norm", Key(object[[r]])),
    assay = DefaultAssay(object[[r]])
  )
}

# nearest.dist is the closest non-self neighbour and sigma is the k-th non-self
# neighbour minus that, which is the bandwidth Scarf uses. Seurat reaches the
# same two quantities as Distances(nn)[, 2] and Distances(nn)[, sigma.idx] on a
# neighbour object whose first column is the cell itself.
own_distances <- lapply(reductions, function(r) {
  embedding <- normed[[r]]
  t(sapply(seq_len(n_cells), function(i) {
    sqrt(rowSums((embedding[indices[[r]][i, ], , drop = FALSE] - rep(embedding[i, ], each = k))^2))
  }))
})
names(own_distances) <- reductions
nearest_dist <- lapply(own_distances, function(d) apply(d, 1, min))
sigma <- lapply(reductions, function(r) {
  apply(own_distances[[r]], 1, function(row) sort(row)[k]) - nearest_dist[[r]]
})
names(sigma) <- reductions

within_impute <- lapply(reductions, function(r) {
  PredictAssay(
    object = object,
    nn.idx = indices[[r]],
    assay = DefaultAssay(object[[r]]),
    reduction = paste0(r, ".norm"),
    dims = dims[[r]],
    return.assay = FALSE,
    verbose = FALSE
  )
})
names(within_impute) <- reductions
cross_impute <- lapply(reductions, function(r) {
  other <- setdiff(reductions, r)
  PredictAssay(
    object = object,
    nn.idx = indices[[other]],
    assay = DefaultAssay(object[[r]]),
    reduction = paste0(r, ".norm"),
    dims = dims[[r]],
    return.assay = FALSE,
    verbose = FALSE
  )
})
names(cross_impute) <- reductions

within_dist <- lapply(reductions, function(r) {
  Seurat:::impute_dist(
    x = normed[[r]],
    y = t(within_impute[[r]]),
    nearest.dist = nearest_dist[[r]]
  )
})
cross_dist <- lapply(reductions, function(r) {
  Seurat:::impute_dist(
    x = normed[[r]],
    y = t(cross_impute[[r]]),
    nearest.dist = nearest_dist[[r]]
  )
})
names(within_dist) <- names(cross_dist) <- reductions

# The next three expressions are copied from the body of FindModalityWeights.
within_kernel <- lapply(reductions, function(r) exp(-1 * (within_dist[[r]] / sigma[[r]])))
cross_kernel <- lapply(reductions, function(r) exp(-1 * (cross_dist[[r]] / sigma[[r]])))
names(within_kernel) <- names(cross_kernel) <- reductions
score <- lapply(reductions, function(r) {
  MinMax(data = within_kernel[[r]] / (cross_kernel[[r]] + 1e-04), min = 0, max = 200)
})
names(score) <- reductions
score_total <- rowSums(exp(Reduce(cbind, score)))
weights <- lapply(reductions, function(r) exp(score[[r]]) / score_total)
names(weights) <- reductions

# Blend the two kernels over the union pool and keep the k strongest, breaking
# ties by ascending cell index the way Scarf's lexsort does.
matched_indices <- matrix(0L, nrow = n_cells, ncol = k)
matched_affinity <- matrix(0, nrow = n_cells, ncol = k)
for (i in seq_len(n_cells)) {
  pool <- sort(union(indices$pca[i, ], indices$apca[i, ]))
  affinity <- rep(0, length(pool))
  for (r in reductions) {
    adjusted <- Seurat:::NNdist(
      nn.idx = matrix(pool, nrow = 1),
      embeddings = normed[[r]],
      query.embeddings = normed[[r]][i, , drop = FALSE],
      nearest.dist = nearest_dist[[r]][i]
    )[[1]]
    affinity <- affinity + weights[[r]][i] * exp(-1 * (adjusted / sigma[[r]][i]))
  }
  chosen <- order(-affinity, pool)[seq_len(k)]
  matched_indices[i, ] <- pool[chosen] - 1L
  matched_affinity[i, ] <- affinity[chosen]
}

write_matrix(matched_indices, "matched_indices.tsv")
write_matrix(matched_affinity, "matched_affinities.tsv")
write_matrix(cbind(weights$pca, weights$apca), "matched_modality_weights.tsv")

# Public defaults. Scarf deliberately differs here, so this table only supports
# a drift check on how far apart the two stay.
default_object <- FindMultiModalNeighbors(
  object = object,
  reduction.list = list("pca", "apca"),
  dims.list = list(dims$pca, dims$apca),
  k.nn = k,
  modality.weight.name = c("pca.weight", "apca.weight"),
  verbose = FALSE
)
default_nn <- default_object[["weighted.nn"]]
default_indices <- Indices(default_nn)
write_matrix(default_indices - 1L, "default_indices.tsv")
write_matrix(
  cbind(default_object$pca.weight, default_object$apca.weight),
  "default_modality_weights.tsv"
)

writeLines(
  c(
    paste0("seurat\t", as.character(packageVersion("Seurat"))),
    paste0("seuratObject\t", as.character(packageVersion("SeuratObject"))),
    paste0("r\t", paste(R.version$major, R.version$minor, sep = ".")),
    paste0("kNn\t", k),
    paste0("nCells\t", n_cells)
  ),
  file.path(work, "versions.tsv")
)
