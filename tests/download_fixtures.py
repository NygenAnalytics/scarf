"""Download bundled test datasets for CI and local development."""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

_FIXTURES_BASE_URL = (
    "https://raw.githubusercontent.com/parashardhapola/scarf/master/scarf/tests/datasets"
)

FIXTURE_FILES = (
    "1K_pbmc_citeseq.h5",
    "1K_pbmc_citeseq.zarr.tar.gz",
    "500_pbmc_atac.zarr.tar.gz",
    "toy_cr_dir.tar.gz",
    "toy_cr_dir_empty.tar.gz",
    "sympathetic.loom",
    "cell_attributes.csv",
    "knn_indices.npy",
    "knn_distances.npy",
    "knn_weights.npy",
    "atac_knn_indices.npy",
    "atac_knn_distances.npy",
    "markers_all_clusters.csv",
    "markers_cluster1.csv",
    "unified_UMAP_coords.npy",
    "pseudotime_markers_r_values.csv",
    "aggregated_feat_idx.npy",
    "aggregated_df_top_10.npy",
    "pseudotime_clusters.npy",
    "ptime_modules_group_1.npy",
)


def datasets_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets"


def download_fixtures(*, force: bool = False) -> None:
    target = datasets_dir()
    target.mkdir(parents=True, exist_ok=True)

    missing: list[tuple[str, urllib.error.HTTPError]] = []
    for name in FIXTURE_FILES:
        dest = target / name
        if dest.is_file() and not force:
            continue
        url = f"{_FIXTURES_BASE_URL}/{name}"
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.HTTPError as exc:
            missing.append((name, exc))

    if missing:
        details = "\n".join(f"  {name}: HTTP {exc.code}" for name, exc in missing)
        msg = f"Failed to download {len(missing)} fixture(s):\n{details}"
        raise RuntimeError(msg)


def download_optional_h5ad() -> None:
    sample = "bastidas-ponce_4K_pancreas-d15_rnaseq"
    local_h5ad = datasets_dir() / sample / "data.h5ad"
    if local_h5ad.is_file():
        return

    from scarf.downloader import fetch_dataset

    fetch_dataset(sample, save_path=str(datasets_dir()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download fixtures even when files already exist.",
    )
    parser.add_argument(
        "--with-h5ad",
        action="store_true",
        help="Also fetch bastidas-ponce h5ad from OSF (requires network).",
    )
    args = parser.parse_args(argv)

    download_fixtures(force=args.force)
    if args.with_h5ad:
        download_optional_h5ad()
    return 0


if __name__ == "__main__":
    sys.exit(main())
