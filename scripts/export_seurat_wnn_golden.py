#!/usr/bin/env python
"""Regenerate tests/seurat_wnn_5_5_1_golden.json.

Pulls RNA and ADT reductions from the public tenx_8K_pbmc_citeseq store, takes a
deterministic cell subset, computes exact self-free neighbour rows, and hands
both to scripts/export_seurat_wnn_golden.R. The Seurat tables come back as
tab-separated text and are written out as the golden fixture.

Needs an R installation carrying Seurat. Point --rscript at it, for example a
conda environment created with `conda create -p /tmp/scarf-seurat-env
-c conda-forge r-base r-seurat`.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from scarf.storage.types import as_zarr_array

REPO = Path(__file__).resolve().parent.parent
DEFAULT_STORE = (
    REPO / "docs/source/tutorials/scarf_datasets/tenx_8K_pbmc_citeseq/data.zarr"
)
DEFAULT_FIXTURE = REPO / "tests/seurat_wnn_5_5_1_golden.json"
R_SCRIPT = Path(__file__).resolve().parent / "export_seurat_wnn_golden.R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--rscript", type=Path, default=Path("Rscript"))
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--cells", type=int, default=300)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20210601)
    return parser.parse_args()


def load_embeddings(store: Path, assays: tuple[str, str]) -> dict[str, np.ndarray]:
    import scarf

    scarf.configure_output(level="ERROR", progress=False)
    datastore = scarf.DataStore(str(store), default_assay=assays[0], nthreads=4)
    embeddings = {}
    for assay in assays:
        state = datastore.get_assay_state(assay)
        if state is None or state.reduction is None:
            raise RuntimeError(f"Assay {assay!r} has no reduction artifact")
        reference = state.batch_correction or state.reduction
        group = datastore.load_artifact(reference)
        embeddings[assay] = np.asarray(as_zarr_array(group["data"])[:])
    return embeddings


def exact_neighbors(embedding: np.ndarray, k: int) -> np.ndarray:
    """Self-free neighbour rows ordered by ascending float64 distance."""
    gram = embedding @ embedding.T
    square = np.diag(gram)
    distances = np.sqrt(np.maximum(square[:, None] - 2.0 * gram + square[None, :], 0.0))
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :k].astype(np.uint32)


def write_table(path: Path, values: np.ndarray) -> None:
    fmt = "%d" if np.issubdtype(values.dtype, np.integer) else "%.17g"
    np.savetxt(path, np.atleast_2d(values), fmt=fmt, delimiter="\t")


def read_table(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, dtype=np.float64, delimiter="\t"))


def main() -> None:
    args = parse_args()
    embeddings = load_embeddings(args.store, ("RNA", "ADT"))
    generator = np.random.default_rng(args.seed)
    total = embeddings["RNA"].shape[0]
    if args.cells > total:
        raise ValueError(f"Store holds {total} cells, cannot sample {args.cells}")
    selected = np.sort(generator.choice(total, size=args.cells, replace=False))

    # float32 storage values widened to float64 so R and Python start from
    # identical numbers and recompute identical distances.
    rna = np.asarray(embeddings["RNA"][selected], dtype=np.float64)
    adt = np.asarray(embeddings["ADT"][selected], dtype=np.float64)
    rna_indices = exact_neighbors(rna, args.k)
    adt_indices = exact_neighbors(adt, args.k)

    with tempfile.TemporaryDirectory(prefix="scarf-wnn-golden-") as raw_work:
        work = Path(raw_work)
        write_table(work / "rna_embedding.tsv", rna)
        write_table(work / "adt_embedding.tsv", adt)
        write_table(work / "rna_indices.tsv", rna_indices)
        write_table(work / "adt_indices.tsv", adt_indices)
        subprocess.run(
            [str(args.rscript), str(R_SCRIPT), str(work)],
            check=True,
        )
        versions = dict(
            line.split("\t")
            for line in (work / "versions.tsv").read_text().splitlines()
        )
        matched_indices = read_table(work / "matched_indices.tsv")
        matched_affinities = read_table(work / "matched_affinities.tsv")
        matched_weights = read_table(work / "matched_modality_weights.tsv")
        default_indices = read_table(work / "default_indices.tsv")
        default_weights = read_table(work / "default_modality_weights.tsv")

    fixture = {
        "provenance": {
            "package": "Seurat",
            "packageVersion": versions["seurat"],
            "seuratObjectVersion": versions["seuratObject"],
            "rVersion": versions["r"],
            "sourceRepository": "https://github.com/satijalab/seurat",
            "dataset": "tenx_8K_pbmc_citeseq",
            "datasetSource": "scarf.cytebase.connect('scarf_docs')",
            "generator": "scripts/export_seurat_wnn_golden.py",
            "nCells": int(args.cells),
            "kNn": int(args.k),
            "cellSampleSeed": int(args.seed),
            "l2Normalize": True,
            "matchedFunctions": [
                "Seurat:::L2Norm",
                "Seurat:::PredictAssay",
                "Seurat:::impute_dist",
                "Seurat:::NNdist",
                "Seurat::MinMax",
            ],
            "matchedNote": (
                "Seurat routines evaluated on Scarf's union candidate pool and "
                "k-th non-self bandwidth. The kernel, score and softmax "
                "expressions are copied from the body of FindModalityWeights."
            ),
            "defaultFunction": "Seurat::FindMultiModalNeighbors",
            "defaultNote": (
                "Shipped defaults, including the wider knn.range search and the "
                "SNN-far bandwidth. Scarf does not reproduce these, so the "
                "comparison is statistical."
            ),
        },
        "inputs": {
            "rnaEmbedding": rna.tolist(),
            "adtEmbedding": adt.tolist(),
            "rnaIndices": rna_indices.astype(int).tolist(),
            "adtIndices": adt_indices.astype(int).tolist(),
        },
        "matched": {
            "modalityWeights": matched_weights.tolist(),
            "neighborIndices": matched_indices.astype(int).tolist(),
            "neighborAffinities": matched_affinities.tolist(),
        },
        "seuratDefault": {
            "modalityWeights": default_weights.tolist(),
            "neighborIndices": default_indices.astype(int).tolist(),
        },
    }
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
