#!/usr/bin/env Rscript
# Produces the Seurat side of tests/seurat_hto_5_5_1_golden.json.
# Driven by scripts/export_seurat_hto_golden.py, which supplies a deterministic
# subset of raw GSE245108 HTO counts and records the returned cell identities.

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(SeuratObject)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: export_seurat_hto_golden.R <work-dir> <random-seed>")
}
work <- args[[1]]
random_seed <- as.integer(args[[2]])

counts_by_cell <- read.delim(
  file.path(work, "hto_counts.tsv"),
  row.names = 1,
  check.names = FALSE
)
hto <- t(as.matrix(counts_by_cell))
storage.mode(hto) <- "numeric"

# Seurat replaces underscores in feature names. Temporary safe names preserve
# an unambiguous mapping back to the names stored in the source matrix.
original_names <- rownames(hto)
safe_names <- sprintf("HTO%02d", seq_along(original_names))
rownames(hto) <- safe_names

object <- CreateSeuratObject(
  counts = Matrix(hto, sparse = TRUE),
  assay = "HTO",
  project = "GSE245108"
)
object <- NormalizeData(
  object,
  assay = "HTO",
  normalization.method = "CLR",
  margin = 1,
  verbose = FALSE
)
object <- HTODemux(
  object,
  assay = "HTO",
  positive.quantile = 0.99,
  init = length(safe_names) + 1,
  nstarts = 100,
  kfunc = "kmeans",
  seed = random_seed,
  verbose = FALSE
)

labels <- as.character(object$hash.ID)
matched <- match(labels, safe_names)
labels[!is.na(matched)] <- original_names[matched[!is.na(matched)]]
global_column <- "HTO_classification.global"
if (!(global_column %in% colnames(object[[]]))) {
  stop("Seurat did not produce HTO_classification.global")
}

calls <- data.frame(
  barcode = colnames(object),
  hashId = labels,
  classificationGlobal = as.character(object[[global_column, drop = TRUE]]),
  stringsAsFactors = FALSE
)
write.table(
  calls,
  file.path(work, "seurat_calls.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

versions <- data.frame(
  key = c("r", "seurat", "seuratObject", "fitdistrplus"),
  value = c(
    R.version$version.string,
    as.character(packageVersion("Seurat")),
    as.character(packageVersion("SeuratObject")),
    as.character(packageVersion("fitdistrplus"))
  )
)
write.table(
  versions,
  file.path(work, "versions.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = FALSE
)
