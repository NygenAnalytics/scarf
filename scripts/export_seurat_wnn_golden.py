#!/usr/bin/env python
"""Regenerate the two-way or synthetic three-way Seurat WNN golden fixture.

The default mode loads explicitly named reduction artifacts from the public
CITE-seq store. Pass one ``--reduction ASSAY=ARTIFACT_ID`` for every assay. The
``--synthetic-three-way`` mode creates deterministic RNA, ATAC, and ADT
embeddings without an external dataset. Both modes compute exact self-free
neighbour rows and hand an arbitrary ordered modality list to the R companion.

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

from scarf.storage import ArtifactRef
from scarf.storage.types import as_zarr_array

REPO = Path(__file__).resolve().parent.parent
DEFAULT_STORE = (
    REPO / "docs/source/tutorials/scarf_datasets/tenx_8K_pbmc_citeseq/data.zarr"
)
DEFAULT_FIXTURE = REPO / "tests/seurat_wnn_5_5_1_golden.json"
DEFAULT_THREE_WAY_FIXTURE = REPO / "tests/seurat_wnn_3way_5_5_1_golden.json"
R_SCRIPT = Path(__file__).resolve().parent / "export_seurat_wnn_golden.R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--rscript", type=Path, default=Path("Rscript"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assays", nargs="+", default=["RNA", "ADT"])
    parser.add_argument(
        "--reduction",
        action="append",
        default=[],
        metavar="ASSAY=ARTIFACT_ID",
        help="Explicit assay-scoped reduction artifact (repeat per assay)",
    )
    parser.add_argument("--synthetic-three-way", action="store_true")
    parser.add_argument("--cells", type=int)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20210601)
    return parser.parse_args()


def parse_reduction_refs(
    values: list[str],
    assays: tuple[str, ...],
) -> dict[str, ArtifactRef]:
    refs: dict[str, ArtifactRef] = {}
    for value in values:
        assay, separator, artifact_id = value.partition("=")
        if not separator or not assay or not artifact_id:
            raise ValueError("--reduction values must use ASSAY=ARTIFACT_ID")
        if assay in refs:
            raise ValueError(f"Duplicate reduction for assay {assay!r}")
        refs[assay] = ArtifactRef(
            scope="assay",
            assay=assay,
            kind="reduction",
            artifact_id=artifact_id,
        )
    expected = set(assays)
    actual = set(refs)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Explicit reductions must match --assays exactly; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return refs


def load_embeddings(
    store: Path,
    reductions: dict[str, ArtifactRef],
) -> dict[str, np.ndarray]:
    import scarf

    scarf.configure_output(level="ERROR", progress=False)
    assays = tuple(reductions)
    datastore = scarf.DataStore(str(store), default_assay=assays[0], nthreads=4)
    embeddings = {}
    for assay, reference in reductions.items():
        group = datastore.load_artifact(reference)
        embeddings[assay] = np.asarray(as_zarr_array(group["data"])[:])
    return embeddings


def synthetic_three_way_embeddings(
    n_cells: int,
    seed: int,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    latent = generator.normal(size=(n_cells, 4))
    return {
        "RNA": latent @ generator.normal(size=(4, 6))
        + generator.normal(scale=0.08, size=(n_cells, 6)),
        "ATAC": latent @ generator.normal(size=(4, 5))
        + generator.normal(scale=0.12, size=(n_cells, 5)),
        "ADT": latent @ generator.normal(size=(4, 3))
        + generator.normal(scale=0.05, size=(n_cells, 3)),
    }


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
    reduction_sources: dict[str, dict[str, str | None]] | None = None
    if args.synthetic_three_way:
        if args.reduction:
            raise ValueError("--reduction is not used with --synthetic-three-way")
        modality_names = ("RNA", "ATAC", "ADT")
        n_cells = args.cells or 48
        selected_embeddings = synthetic_three_way_embeddings(n_cells, args.seed)
        dataset = "synthetic_three_modality"
        dataset_source = "scripts/export_seurat_wnn_golden.py"
        output = args.output or DEFAULT_THREE_WAY_FIXTURE
    else:
        modality_names = tuple(args.assays)
        if len(modality_names) < 2:
            raise ValueError("Golden generation requires at least two assays")
        if len(set(modality_names)) != len(modality_names):
            raise ValueError("Golden generation requires unique assay names")
        if modality_names != ("RNA", "ADT") and args.output is None:
            raise ValueError("--output is required for a non-default assay list")
        reductions = parse_reduction_refs(args.reduction, modality_names)
        embeddings = load_embeddings(args.store, reductions)
        generator = np.random.default_rng(args.seed)
        total = embeddings[modality_names[0]].shape[0]
        n_cells = args.cells or 300
        if n_cells > total:
            raise ValueError(f"Store holds {total} cells, cannot sample {n_cells}")
        selected = np.sort(generator.choice(total, size=n_cells, replace=False))
        selected_embeddings = {
            name: np.asarray(embeddings[name][selected], dtype=np.float64)
            for name in modality_names
        }
        dataset = args.store.parent.name
        dataset_source = (
            "scarf.cytebase.connect('scarf_docs')"
            if args.store == DEFAULT_STORE
            else str(args.store)
        )
        reduction_sources = {
            name: reductions[name].to_dict() for name in modality_names
        }
        output = args.output or DEFAULT_FIXTURE
    if args.k >= n_cells:
        raise ValueError("k must be smaller than the number of fixture cells")

    # Storage values are widened to float64 so R and Python start from
    # identical numbers and recompute identical distances.
    selected_embeddings = {
        name: np.asarray(selected_embeddings[name], dtype=np.float64)
        for name in modality_names
    }
    neighbor_indices = {
        name: exact_neighbors(selected_embeddings[name], args.k)
        for name in modality_names
    }

    with tempfile.TemporaryDirectory(prefix="scarf-wnn-golden-") as raw_work:
        work = Path(raw_work)
        modality_specs = []
        for index, name in enumerate(modality_names):
            stem = f"modality_{index + 1}"
            reduction = f"mod{index + 1}"
            write_table(work / f"{stem}_embedding.tsv", selected_embeddings[name])
            write_table(work / f"{stem}_indices.tsv", neighbor_indices[name])
            modality_specs.append(f"{name}\t{stem}\t{reduction}")
        (work / "modalities.tsv").write_text("\n".join(modality_specs) + "\n")
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

    provenance = {
        "package": "Seurat",
        "packageVersion": versions["seurat"],
        "seuratObjectVersion": versions["seuratObject"],
        "rVersion": versions["r"],
        "sourceRepository": "https://github.com/satijalab/seurat",
        "dataset": dataset,
        "datasetSource": dataset_source,
        "generator": "scripts/export_seurat_wnn_golden.py",
        "nCells": n_cells,
        "kNn": int(args.k),
        "cellSampleSeed": int(args.seed),
        "modalityNames": list(modality_names),
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
            "nearest-to-k-th non-self distance-span bandwidth. The kernel, "
            "score and softmax expressions are copied from the body of "
            "FindModalityWeights."
        ),
        "defaultFunction": "Seurat::FindMultiModalNeighbors",
        "defaultNote": (
            "Shipped defaults, including the wider knn.range search and the "
            "SNN-far bandwidth. Scarf does not reproduce these, so the "
            "comparison is statistical."
        ),
    }
    if reduction_sources is not None:
        provenance["reductions"] = reduction_sources

    fixture = {
        "provenance": provenance,
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
    if modality_names == ("RNA", "ADT") and not args.synthetic_three_way:
        fixture["inputs"] = {
            "rnaEmbedding": selected_embeddings["RNA"].tolist(),
            "adtEmbedding": selected_embeddings["ADT"].tolist(),
            "rnaIndices": neighbor_indices["RNA"].astype(int).tolist(),
            "adtIndices": neighbor_indices["ADT"].astype(int).tolist(),
        }
    else:
        fixture["inputs"] = {
            "modalityNames": list(modality_names),
            "embeddings": {
                name: selected_embeddings[name].tolist() for name in modality_names
            },
            "neighborIndices": {
                name: neighbor_indices[name].astype(int).tolist()
                for name in modality_names
            },
        }
    output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
