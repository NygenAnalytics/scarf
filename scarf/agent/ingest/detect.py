"""Input format detection for ingest."""

from pathlib import Path


def detect_format(path: str | Path) -> str:
    """Detect a Scarf-supported input family from path layout."""
    source = Path(path)
    suffix = "".join(source.suffixes).lower() if source.suffixes else ""
    name = source.name.lower()

    if name.endswith(".h5ad") or suffix.endswith(".h5ad"):
        return "h5ad"
    if name.endswith(".loom") or suffix.endswith(".loom"):
        return "loom"
    if name.endswith(".rds") or name.endswith(".h5seurat") or suffix.endswith(".rds"):
        return "seurat"
    if name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt"):
        return "csv"
    if name.endswith(".zarr") or (source.is_dir() and _looks_like_zarr(source)):
        return "zarr"
    if source.is_file() and name.endswith((".h5", ".hdf5")):
        return "10x_h5"
    if source.is_dir() and _looks_like_10x_dir(source):
        return "10x_dir"
    if source.is_dir() or name.endswith((".mtx", ".mtx.gz")):
        return "mtx"
    return "unknown"


def _looks_like_zarr(path: Path) -> bool:
    return (path / "zarr.json").exists() or (path / "cellData").exists()


def _looks_like_10x_dir(path: Path) -> bool:
    names = {child.name.lower() for child in path.iterdir()} if path.exists() else set()
    has_matrix = any("matrix.mtx" in name for name in names)
    has_barcodes = any("barcode" in name for name in names)
    has_features = any(
        name.startswith("features") or name.startswith("genes") for name in names
    )
    return has_matrix and has_barcodes and has_features
