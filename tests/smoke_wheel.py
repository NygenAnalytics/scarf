import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


_RETIRED_MODULES = {
    "scarf/_types.py",
    "scarf/ann.py",
    "scarf/assay.py",
    "scarf/bio_data.py",
    "scarf/chunked.py",
    "scarf/clustering/feature_graph.py",
    "scarf/cytebase.py",
    "scarf/dendrogram.py",
    "scarf/downloader.py",
    "scarf/doublet_utils.py",
    "scarf/feat_utils.py",
    "scarf/features/lowess.py",
    "scarf/harmony.py",
    "scarf/harmony/__init__.py",
    "scarf/harmony/api.py",
    "scarf/harmony/models.py",
    "scarf/harmony/optimizer.py",
    "scarf/genomics/__init__.py",
    "scarf/genomics/gff.py",
    "scarf/genomics/intervals.py",
    "scarf/genomics/melding.py",
    "scarf/genomics/reference.py",
    "scarf/graph/build.py",
    "scarf/knn_utils.py",
    "scarf/lineage.py",
    "scarf/mapping/coral.py",
    "scarf/mapping_reference.py",
    "scarf/mapping_utils.py",
    "scarf/markers.py",
    "scarf/markers/__init__.py",
    "scarf/markers/batching.py",
    "scarf/markers/rank.py",
    "scarf/markers/regression.py",
    "scarf/markers/search.py",
    "scarf/meld_assay.py",
    "scarf/merge.py",
    "scarf/merge/assays.py",
    "scarf/metadata.py",
    "scarf/metrics.py",
    "scarf/neighbors/graph_store.py",
    "scarf/neighbors/persistence.py",
    "scarf/neighbors/query.py",
    "scarf/parallel.py",
    "scarf/plots.py",
    "scarf/plots/__init__.py",
    "scarf/plotting/_legacy.py",
    "scarf/plotting/_legacy/__init__.py",
    "scarf/plotting/unified.py",
    "scarf/readers.py",
    "scarf/readers/datasets.py",
    "scarf/results.py",
    "scarf/storage/zarr_store.py",
    "scarf/trajectory/aggregation.py",
    "scarf/symphony.py",
    "scarf/umap.py",
    "scarf/utils.py",
    "scarf/utils/blocks.py",
    "scarf/utils/memory.py",
    "scarf/utils/storage.py",
    "scarf/utils/system.py",
    "scarf/utils/windows.py",
    "scarf/writers.py",
}
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "scarf"
_REQUIRED_MODULES = {
    f"scarf/{path.relative_to(_SOURCE_ROOT).as_posix()}"
    for path in _SOURCE_ROOT.rglob("*.py")
}
_SMOKE_CODE = """
import importlib.util
from pathlib import Path

import scarf
import scarf.plotting as plotting
import scarf.cytebase
import scarf.embeddings.harmony
import scarf.features.genomic
import scarf.features.markers
import scarf.features.variability
import scarf.matrix
import scarf.merge
import scarf.metadata
import scarf.readers
import scarf.writers
from scarf.datastore.datastore import DataStore
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.datastore.mapping_datastore import MappingDatastore
from scarf.cytebase import Repository, connect, list_repositories
from scarf.embeddings.harmony import Harmony, HarmonyResult, fit_harmony, run_harmony
from scarf.features import (
    GffReader,
    coordinate_melding,
    find_markers_by_rank,
    fit_lowess,
    select_highly_variable_features,
)
from scarf.matrix import Block, ChunkedArray
from scarf.merge import DataStoreMerge
from scarf.metadata import MetaData, MetaDataRowBlock
from scarf.readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    H5adReader,
    LoomReader,
    SeuratReader,
    inspect_seurat,
)
from scarf.storage.lineage import ArtifactLineage
from scarf.trajectory.feature_dynamics import knn_clustering
from scarf.writers import (
    CSVtoZarr,
    CrToZarr,
    H5adToZarr,
    LoomToZarr,
    SeuratImportResult,
    SeuratToZarr,
    SparseToZarr,
    SubsetZarr,
    create_zarr_count_assay,
    create_zarr_dataset,
    create_zarr_obj_array,
    chunked_to_zarr,
    subset_assay_zarr,
    to_h5ad,
    to_mtx,
    write_renorm_subset_to_zarr,
)

assert "site-packages" in Path(scarf.__file__).as_posix()
assert issubclass(DataStore, MappingDatastore)
assert issubclass(MappingDatastore, GraphDataStore)
for harmony_object in (Harmony, HarmonyResult, fit_harmony, run_harmony):
    assert harmony_object.__module__ == "scarf.embeddings.harmony"
for matrix_class in (Block, ChunkedArray):
    assert matrix_class.__module__ == "scarf.matrix"
for metadata_class in (MetaData, MetaDataRowBlock):
    assert metadata_class.__module__ == "scarf.metadata"
assert scarf.DataStoreMerge is scarf.merge.DataStoreMerge is DataStoreMerge
assert not hasattr(scarf, "AssayMerge")
assert not hasattr(scarf.merge, "AssayMerge")
assert not hasattr(scarf, "DatasetMerge")
assert not hasattr(scarf.merge, "DatasetMerge")
assert not hasattr(scarf, "ZarrMerge")
assert not hasattr(scarf.merge, "ZarrMerge")
assert DataStoreMerge.__module__ == "scarf.merge"
assert scarf.CrH5Reader is scarf.readers.CrH5Reader
assert scarf.CrToZarr is scarf.writers.CrToZarr
assert scarf.cytebase.Repository is Repository
assert scarf.cytebase.connect is connect
assert scarf.cytebase.list_repositories is list_repositories
assert scarf.ArtifactLineage is ArtifactLineage
assert ArtifactLineage.__module__ == "scarf.storage.lineage"
assert scarf.GffReader is GffReader
assert scarf.coordinate_melding is coordinate_melding
for feature_function in (
    coordinate_melding,
    find_markers_by_rank,
    fit_lowess,
    select_highly_variable_features,
):
    assert callable(feature_function)
assert callable(knn_clustering)
for reader_class in (
    CrH5Reader,
    CrDirReader,
    CrReader,
    H5adReader,
    LoomReader,
    SeuratReader,
    CSVReader,
):
    assert reader_class.__module__ == "scarf.readers"
for writer_class in (
    CrToZarr,
    H5adToZarr,
    LoomToZarr,
    SeuratToZarr,
    SparseToZarr,
    SubsetZarr,
    CSVtoZarr,
):
    assert writer_class.__module__ == "scarf.writers"
assert SeuratImportResult.__module__ == "scarf.writers"
assert inspect_seurat.__module__ == "scarf.readers"
for writer_function in (
    create_zarr_dataset,
    create_zarr_obj_array,
    create_zarr_count_assay,
    subset_assay_zarr,
    chunked_to_zarr,
    write_renorm_subset_to_zarr,
    to_h5ad,
    to_mtx,
):
    assert callable(writer_function)
    assert writer_function.__module__ == "scarf.writers"
for method in (
    "run_mapping",
    "run_marker_search",
    "mark_hvgs",
    "run_pseudotime_aggregation",
    "run_pseudotime_marker_search",
    "metric_lisi",
):
    assert callable(getattr(DataStore, method))
for method in (
    "_load_unified_layout_data",
    "load_unified_graph",
    "run_unified_tsne",
    "run_unified_umap",
):
    assert not hasattr(DataStore, method)
assert not hasattr(plotting, "unified_embedding")
for name in (
    "scarf._types",
    "scarf.bio_data",
    "scarf.chunked",
    "scarf.downloader",
    "scarf.doublet_utils",
    "scarf.feat_utils",
    "scarf.harmony",
    "scarf.genomics",
    "scarf.knn_utils",
    "scarf.lineage",
    "scarf.mapping.coral",
    "scarf.mapping_reference",
    "scarf.mapping_utils",
    "scarf.markers",
    "scarf.meld_assay",
    "scarf.plotting.unified",
    "scarf.symphony",
):
    assert importlib.util.find_spec(name) is None, name
for name in (
    "scarf.embeddings.harmony",
    "scarf.features.genomic",
    "scarf.features.markers",
    "scarf.matrix",
    "scarf.metadata",
    "scarf.metrics",
):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.submodule_search_locations is not None, name
"""


def validate_wheel_contents(wheel: Path) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        script_entries = [
            info
            for info in archive.infolist()
            if info.filename.endswith(".data/scripts/sgtsne")
        ]
    retired = sorted(_RETIRED_MODULES.intersection(names))
    missing = sorted(_REQUIRED_MODULES.difference(names))
    if retired:
        raise RuntimeError(f"Wheel contains retired modules: {retired}")
    if missing:
        raise RuntimeError(f"Wheel is missing required modules: {missing}")
    if len(script_entries) != 1:
        raise RuntimeError("Wheel must contain exactly one sgtsne script")
    script_mode = (script_entries[0].external_attr >> 16) & 0o777
    if script_mode != 0o755:
        raise RuntimeError(
            f"Wheel sgtsne script has mode {oct(script_mode)}, expected 0o755"
        )


def smoke_installed_wheel(wheel: Path) -> None:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    env = os.environ.copy()
    env["HNSWLIB_NO_NATIVE"] = "1"
    with tempfile.TemporaryDirectory(prefix="scarf-wheel-smoke-") as temp_dir:
        subprocess.run(
            [
                "uv",
                "run",
                "--isolated",
                "--python",
                python_version,
                "--with",
                str(wheel),
                "python",
                "-c",
                _SMOKE_CODE,
            ],
            cwd=temp_dir,
            env=env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    validate_wheel_contents(wheel)
    smoke_installed_wheel(wheel)
    print(f"Wheel smoke passed: {wheel.name}")


if __name__ == "__main__":
    main()
