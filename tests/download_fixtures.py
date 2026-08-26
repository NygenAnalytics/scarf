"""Download bundled test datasets for CI and local development."""

import argparse
import gzip
import hashlib
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import h5py
import pandas as pd
from huggingface_hub import download_bucket_files
from scipy.sparse import csr_matrix

_H5AD_DOWNLOAD_ATTEMPTS = 3

_BUCKET_ID = "Nygen/cytebase"
_REPOSITORY = "scarf_tests"

_CYTEBASE_FIXTURES = {
    "1K_pbmc_citeseq.h5": (
        "e3dd57c5a8c3426dc5a7dc012a78608554facbddf4c6d1d62671089d197b1053"
    ),
    "500_pbmc_atac.zarr.tar.gz": (
        "f54c79229e674ea2956c048a931aadb4a9d54e75778df17b4bd84608c0e06493"
    ),
    "toy_cr_dir.tar.gz": (
        "8f7a509f577d23bb8b14947bf01ae66dcc44cbf0263aa070fe5759d51e7fddac"
    ),
    "toy_cr_dir_empty.tar.gz": (
        "7b0dad810bb395d837f8b17411b2f7a4a6ff7291c145287cb863ab016c312f35"
    ),
    "sympathetic.loom": (
        "63347f66d1180e544ddff257ee3f15934e989e521527d4cdcacbcb733e846120"
    ),
    "cell_attributes.csv": (
        "cf1edebb1db09918d3d8286dcbe1491bcf8ef5fd7e116bf720e9aef6f078a06c"
    ),
    "knn_indices.npy": (
        "92bf88532823033a5d39350423815b3793e9e806ea8734a59aef303032815fcb"
    ),
    "knn_distances.npy": (
        "4bb814d647fe463ae1e72a5422d34af8588b6eb7fb1da5c4d2cebb056dc96bf4"
    ),
    "knn_weights.npy": (
        "aabc5bec6753d24682f5bb1c2dc7042279496ab0bc7eae99c3908ea198ac0c18"
    ),
    "atac_knn_indices.npy": (
        "e2878cf40604383c954bcb1ecb68c2c9da90a27ccdba2a5cb61ee779e4181246"
    ),
    "atac_knn_distances.npy": (
        "3f446f51636bfc5795f42bc94e26bc98b0c50249ed03b205ea9077b3bc5cd7fb"
    ),
    "markers_cluster1.csv": (
        "04afc7fc03475c6d3eeeeefc1d7c985a3c4475d36a8b445368fc7e74482f6056"
    ),
    "pseudotime_markers_r_values.csv": (
        "b65868812f55fd609fc6dad43a47dd3f5e52404164847cd0746446a6343885d0"
    ),
    "aggregated_feat_idx.npy": (
        "08f2539818778aee5473de16b4b008e3b9343ff5ac4c5fc573e9ccf885d9763f"
    ),
    "aggregated_df_top_10.npy": (
        "dc8443d6f1b7f983f28b3e9ae38f121a3c71e7f711a388676a212100169f127a"
    ),
    "pseudotime_clusters.npy": (
        "f721d50edb512b3d02f9ba37e57804733ca096a625e8a7a0f168dfd113fd66f1"
    ),
    "ptime_modules_group_1.npy": (
        "7c77de233841133b69bacc3d38b9e412f37a2caaa77442e8a9f1beaebf3653bf"
    ),
    # Source: harryhaller001/readseurat commit
    # 8a688b47df27f90e98a4c57ddd9e47c0e5ded01e, tests/data/synthetic.rds (MIT).
    "seurat_assay5_synthetic.rds": (
        "f1e6f6fd3e1959452a9ef7e72571a86e1b27a061d8cf00cd28932d8757cdac7c"
    ),
    # Source: https://doi.org/10.5281/zenodo.10944066 (CC0-1.0).
    "seurat_v4_1_3_pbmc_mye.rds": (
        "f84adf523a78aeb6e6681cf09e06a2a2fcd4e3fe857fdd89b17e90a1782fac3d"
    ),
}

FIXTURE_FILES = tuple(_CYTEBASE_FIXTURES)


def datasets_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _remote_path(name: str) -> str:
    return f"{_REPOSITORY}/{name}"


def _verify_digest(path: Path, expected: str) -> None:
    digest = _sha256(path)
    if digest != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Fixture {path.name} has SHA-256 {digest}, expected {expected}"
        )


def _download_cytebase_fixtures(target: Path, *, force: bool) -> None:
    needed: list[tuple[str, Path, str]] = []
    for name, expected in _CYTEBASE_FIXTURES.items():
        dest = target / name
        if dest.is_file() and not force:
            if _sha256(dest) == expected:
                continue
            dest.unlink()
        elif dest.is_file() and force:
            dest.unlink()
        needed.append((name, dest, expected))

    if not needed:
        return

    download_bucket_files(
        _BUCKET_ID,
        files=[(_remote_path(name), dest) for name, dest, _ in needed],
        token=False,
        raise_on_missing_files=True,
    )

    for name, dest, expected in needed:
        if not dest.is_file():
            raise RuntimeError(
                f"Cytebase did not download {_remote_path(name)}. "
                "Publish fixtures with: "
                "uv run python scripts/publish_test_fixtures.py --apply"
            )
        _verify_digest(dest, expected)


def _citeseq_zarr_ready(archive: Path) -> bool:
    """Return True when the archive opens as current RNA count-matrix layout."""
    if not archive.is_file():
        return False
    import zarr

    from scarf.storage.counts_t_contract import inspect_counts_t

    with tempfile.TemporaryDirectory(prefix="scarf_citeseq_zarr_check_") as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        root = zarr.open_group(tmp, mode="r")
        if "RNA" not in root or "assay2" not in root:
            return False
        return inspect_counts_t(root, "RNA").status == "ready"


def build_citeseq_zarr_fixture(*, force: bool = False) -> None:
    """Write ``1K_pbmc_citeseq.zarr.tar.gz`` from the Cell Ranger H5.

    Tests expect the Antibody Capture assay to be named ``assay2``. The
    published Cytebase archive is not rebuilt here because its persisted
    layout no longer replays against the current planner.
    """
    archive = datasets_dir() / "1K_pbmc_citeseq.zarr.tar.gz"
    if not force and _citeseq_zarr_ready(archive):
        return

    h5_path = datasets_dir() / "1K_pbmc_citeseq.h5"
    if not h5_path.is_file():
        raise RuntimeError(
            "Cannot build 1K_pbmc_citeseq.zarr.tar.gz without "
            f"{h5_path.name}; download core fixtures first."
        )

    from scarf.readers import CrH5Reader
    from scarf.writers import CrToZarr

    with tempfile.TemporaryDirectory(prefix="scarf_citeseq_zarr_build_") as tmp:
        store = Path(tmp) / "store.zarr"
        reader = CrH5Reader(str(h5_path))
        if "ADT" in reader.assayFeats.columns:
            reader.rename_assays({"ADT": "assay2"})
        CrToZarr(reader, zarr_loc=str(store), nthreads=2).dump()
        staging = Path(tmp) / "archive.tar.gz"
        with tarfile.open(staging, "w:gz") as tar:
            for item in sorted(store.iterdir(), key=lambda path: path.name):
                tar.add(item, arcname=item.name)
        if not _citeseq_zarr_ready(staging):
            raise RuntimeError(
                "Built 1K_pbmc_citeseq.zarr.tar.gz is not a current RNA store"
            )
        archive.parent.mkdir(parents=True, exist_ok=True)
        # Path.replace is rename(2) and fails when /tmp and the workspace
        # are different filesystems (GitHub Actions: Errno 18).
        shutil.move(staging, archive)


def download_fixtures(*, force: bool = False) -> None:
    target = datasets_dir()
    target.mkdir(parents=True, exist_ok=True)
    _download_cytebase_fixtures(target, force=force)
    build_citeseq_zarr_fixture(force=force)


def build_mtx_dir_fixture() -> None:
    archive = datasets_dir() / "1K_pbmc_citeseq_dir.tar.gz"
    if archive.is_file():
        return

    h5_path = datasets_dir() / "1K_pbmc_citeseq.h5"
    if not h5_path.is_file():
        msg = (
            "Cannot build 1K_pbmc_citeseq_dir.tar.gz without "
            f"{h5_path.name}; download core fixtures first."
        )
        raise RuntimeError(msg)

    from scarf.readers import CrDirReader, CrH5Reader

    reader = CrH5Reader(str(h5_path))
    rna = "RNA"
    assay = reader.assayFeats[rna]
    start, end = int(assay["start"]), int(assay["end"])
    feat_ids = reader.feature_ids(rna)
    feat_names = reader.feature_names(rna)
    feat_types = reader.feature_types()
    cells = reader.cell_names()

    with tempfile.TemporaryDirectory(prefix="scarf_mtx_fixture_") as tmp:
        mtx_dir = Path(tmp) / "1K_pbmc_citeseq_dir"
        mtx_dir.mkdir()

        with gzip.open(mtx_dir / "features.tsv.gz", "wt") as handle:
            for idx in range(start, end):
                handle.write(
                    f"{feat_ids[idx - start]}\t{feat_names[idx - start]}\t{feat_types[idx]}\n"
                )

        with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as handle:
            for cell in cells:
                handle.write(f"{cell}\n")

        with h5py.File(h5_path) as h5:
            matrix = h5["matrix"]
            mat = csr_matrix(
                (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
                shape=(len(cells), reader.nFeatures),
            )
        rna_mat = mat[:, start:end].T.tocoo()

        with gzip.open(mtx_dir / "matrix.mtx.gz", "wt") as handle:
            handle.write(
                "%%MatrixMarket matrix coordinate integer general\n% Generated by Scarf\n"
            )
            handle.write(f"{rna_mat.shape[0]} {rna_mat.shape[1]} {rna_mat.nnz}\n")
            pd.DataFrame(
                {"row": rna_mat.row + 1, "col": rna_mat.col + 1, "d": rna_mat.data}
            ).to_csv(
                handle,
                sep=" ",
                header=False,
                index=False,
                mode="a",
                lineterminator="\n",
            )

        CrDirReader(str(mtx_dir))

        with tarfile.open(archive, "w:gz") as tar:
            tar.add(mtx_dir, arcname=mtx_dir.name)


def download_optional_h5ad(*, attempts: int = _H5AD_DOWNLOAD_ATTEMPTS) -> bool:
    sample = "bastidas-ponce_4K_pancreas-d15_rnaseq"
    local_h5ad = datasets_dir() / sample / "data.h5ad"
    if local_h5ad.is_file():
        return True

    from scarf import cytebase

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            cytebase.connect("scarf_docs").download_dataset(
                sample,
                destination=datasets_dir(),
            )
            if local_h5ad.is_file():
                return True
            last_error = FileNotFoundError(
                f"Cytebase download finished without creating {local_h5ad}"
            )
        except Exception as exc:
            last_error = exc

        if attempt >= attempts:
            break
        delay = min(60.0, 2.0**attempt)
        print(
            f"Warning: h5ad download attempt {attempt}/{attempts} failed "
            f"({last_error}); retrying in {delay:.0f}s",
            file=sys.stderr,
        )
        time.sleep(delay)

    print(
        f"Warning: skipping optional h5ad fixture after {attempts} attempts "
        f"({last_error}). Tests that need {local_h5ad} will be skipped.",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download Cytebase fixtures and rebuild "
            "1K_pbmc_citeseq.zarr.tar.gz even when files already exist."
        ),
    )
    parser.add_argument(
        "--with-h5ad",
        action="store_true",
        help="Also download the bastidas-ponce h5ad from Cytebase.",
    )
    args = parser.parse_args(argv)

    download_fixtures(force=args.force)
    build_mtx_dir_fixture()
    if args.with_h5ad:
        download_optional_h5ad()
    return 0


if __name__ == "__main__":
    sys.exit(main())
