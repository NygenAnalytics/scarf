#!/usr/bin/env python
"""Regenerate tests/seurat_hto_5_5_1_golden.json.

Reads the GSE245108 TNC-1-2-1 filtered 10x H5 matrix, takes a deterministic
cell subset, and hands the raw HTO counts to scripts/export_seurat_hto_golden.R.
The Seurat calls and the exact input counts are written to a compact fixture, so
normal CI does not require R, Seurat, network access, or the full GEO matrix.

Download the source matrix from GEO, then provide an R installation carrying
Seurat 5.5.1. For example:

    uv run python scripts/export_seurat_hto_golden.py \
        --h5 /path/to/GSE245108_TNC-1-2-1_filtered_feature_bc_matrix.h5 \
        --rscript /path/to/seurat-environment/bin/Rscript
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO / "tests/seurat_hto_5_5_1_golden.json"
R_SCRIPT = Path(__file__).resolve().parent / "export_seurat_hto_golden.R"
DATASET_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE245nnn/GSE245108/suppl/"
    "GSE245108_TNC-1-2-1_filtered_feature_bc_matrix.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, default=Path("Rscript"))
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--cells", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260731)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(inputs: dict[str, object]) -> str:
    encoded = json.dumps(
        inputs,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_hto_counts(path: Path) -> pd.DataFrame:
    with h5py.File(path) as handle:
        matrix = handle["matrix"]
        sparse = csc_matrix(
            (
                matrix["data"][:],
                matrix["indices"][:],
                matrix["indptr"][:],
            ),
            shape=tuple(int(value) for value in matrix["shape"][:]),
        )
        names = np.asarray(matrix["features/name"]).astype(str)
        feature_types = np.asarray(matrix["features/feature_type"]).astype(str)
        barcodes = np.asarray(matrix["barcodes"]).astype(str)

    selected = (feature_types == "Antibody Capture") & np.char.startswith(names, "TNC_")
    if selected.sum() != 10:
        raise ValueError(
            "Expected exactly 10 TNC_ Antibody Capture features; "
            f"found {int(selected.sum())}"
        )
    return pd.DataFrame(
        sparse[np.flatnonzero(selected)].T.toarray(),
        index=barcodes,
        columns=names[selected],
    )


def main() -> None:
    args = parse_args()
    if not args.h5.is_file():
        raise FileNotFoundError(args.h5)

    counts = read_hto_counts(args.h5)
    if args.cells > len(counts):
        raise ValueError(
            f"Matrix holds {len(counts)} cells, cannot sample {args.cells}"
        )
    generator = np.random.default_rng(args.sample_seed)
    selected = np.sort(generator.choice(len(counts), size=args.cells, replace=False))
    subset = counts.iloc[selected].astype(np.int64)

    with tempfile.TemporaryDirectory(prefix="scarf-hto-golden-") as raw_work:
        work = Path(raw_work)
        subset.to_csv(work / "hto_counts.tsv", sep="\t", index_label="barcode")
        subprocess.run(
            [
                str(args.rscript),
                str(R_SCRIPT),
                str(work),
                str(args.random_seed),
            ],
            check=True,
        )
        calls = pd.read_csv(work / "seurat_calls.tsv", sep="\t")
        versions = dict(
            line.split("\t")
            for line in (work / "versions.tsv").read_text().splitlines()
        )

    barcodes = subset.index.tolist()
    if calls["barcode"].tolist() != barcodes:
        raise RuntimeError("Seurat returned calls in an unexpected cell order")

    input_values: dict[str, object] = {
        "barcodes": barcodes,
        "htoNames": subset.columns.tolist(),
        "rawCounts": subset.to_numpy().tolist(),
    }
    fixture = {
        "provenance": {
            "package": "Seurat",
            "packageVersion": versions["seurat"],
            "seuratObjectVersion": versions["seuratObject"],
            "fitdistrplusVersion": versions["fitdistrplus"],
            "rVersion": versions["r"],
            "sourceRepository": "https://github.com/satijalab/seurat",
            "dataset": "GSE245108",
            "datasetSource": DATASET_URL,
            "sourceFile": args.h5.name,
            "sourceSha256": sha256_file(args.h5),
            "generator": "scripts/export_seurat_hto_golden.py",
            "nCells": int(args.cells),
            "nHtos": int(subset.shape[1]),
            "cellSampleSeed": int(args.sample_seed),
            "randomSeed": int(args.random_seed),
            "normalization": {"method": "CLR", "margin": 1},
            "clustering": {
                "method": "kmeans",
                "clusterCount": int(subset.shape[1] + 1),
                "nStarts": 100,
            },
            "cutoff": {
                "distribution": "negative_binomial",
                "positiveQuantile": 0.99,
                "comparison": "strictly_greater",
            },
            "referenceFunction": "Seurat::HTODemux",
        },
        "inputs": {
            **input_values,
            "sha256": input_fingerprint(input_values),
        },
        "seurat": {
            "hashId": calls["hashId"].tolist(),
            "classificationGlobal": calls["classificationGlobal"].tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
