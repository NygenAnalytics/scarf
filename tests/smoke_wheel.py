import argparse
import os
import platform
import subprocess
import sys
import tempfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


_RETIRED_MODULES = {
    "scarf/_types.py",
    "scarf/agent/ingest/loom.py",
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
    "scarf/features/markers/batching.py",
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
    "scarf/neighbors/stream.py",
    "scarf/parallel.py",
    "scarf/plots.py",
    "scarf/plots/__init__.py",
    "scarf/plotting/_legacy.py",
    "scarf/plotting/_legacy/__init__.py",
    "scarf/plotting/unified.py",
    "scarf/readers.py",
    "scarf/readers/datasets.py",
    "scarf/readers/loom.py",
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
    "scarf/writers/loom.py",
    "scarf/writers.py",
}
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "scarf"
_WORKFLOW_SCRIPT = Path(__file__).resolve().with_name("smoke_workflow.py")
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
from scarf.embeddings.harmony import Harmony, HarmonyResult, fit_harmony
from scarf.features import (
    RankMarkerResult,
    find_markers_by_rank,
    fit_lowess,
    select_highly_variable_features,
)
from scarf.matrix import ChunkedArray
from scarf.merge import DataStoreMerge
from scarf.metadata import MetaData, MetaDataRowBlock
from scarf.readers import (
    CSVReader,
    CrDirReader,
    CrH5Reader,
    CrReader,
    H5adReader,
    SeuratReader,
    inspect_seurat,
)
from scarf.storage.lineage import ArtifactLineage
from scarf.trajectory.feature_dynamics import knn_clustering
from scarf.writers import (
    CSVtoZarr,
    CrToZarr,
    H5adImportResult,
    H5adToZarr,
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
for harmony_object in (Harmony, HarmonyResult, fit_harmony):
    assert harmony_object.__module__ == "scarf.embeddings.harmony"
assert ChunkedArray.__module__ == "scarf.matrix"
for metadata_class in (MetaData, MetaDataRowBlock):
    assert metadata_class.__module__ == "scarf.metadata"
assert scarf.DataStoreMerge is scarf.merge.DataStoreMerge is DataStoreMerge
assert not hasattr(scarf, "AssayMerge")
assert not hasattr(scarf.merge, "AssayMerge")
assert not hasattr(scarf, "DatasetMerge")
assert not hasattr(scarf.merge, "DatasetMerge")
assert not hasattr(scarf, "ZarrMerge")
assert not hasattr(scarf.merge, "ZarrMerge")
assert not hasattr(scarf, "LoomReader")
assert not hasattr(scarf, "LoomToZarr")
assert DataStoreMerge.__module__ == "scarf.merge"
assert scarf.CrH5Reader is scarf.readers.CrH5Reader
assert scarf.CrToZarr is scarf.writers.CrToZarr
assert scarf.cytebase.Repository is Repository
assert scarf.cytebase.connect is connect
assert scarf.cytebase.list_repositories is list_repositories
assert scarf.ArtifactLineage is ArtifactLineage
assert ArtifactLineage.__module__ == "scarf.storage.lineage"
for feature_function in (
    find_markers_by_rank,
    fit_lowess,
    select_highly_variable_features,
):
    assert callable(feature_function)
assert RankMarkerResult.__module__ == "scarf.features.markers.table"
assert scarf.features.markers.RankMarkerResult is RankMarkerResult
assert callable(knn_clustering)
for reader_class in (
    CrH5Reader,
    CrDirReader,
    CrReader,
    H5adReader,
    SeuratReader,
    CSVReader,
):
    assert reader_class.__module__ == "scarf.readers"
for writer_class in (
    CrToZarr,
    H5adToZarr,
    SeuratToZarr,
    SparseToZarr,
    SubsetZarr,
    CSVtoZarr,
):
    assert writer_class.__module__ == "scarf.writers"
assert H5adImportResult.__module__ == "scarf.writers"
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
    "select_hvgs",
    "run_pseudotime_aggregation",
    "run_pseudotime_marker_search",
):
    assert callable(getattr(DataStore, method))
for method in (
    "_load_unified_layout_data",
    "load_metric_lisi",
    "load_unified_graph",
    "metric_lisi",
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
    "scarf.readers.loom",
    "scarf.symphony",
    "scarf.writers.loom",
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


# Scarf is pure Python. Its one wheel installs on every platform, so it must
# carry no native executable or library and nothing outside the import root.
_PURE_TAG = "py3-none-any"
_MACH_O_MAGIC = frozenset(
    {
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        # Universal binaries, in both byte orders and both offset widths.
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf",
        b"\xbf\xba\xfe\xca",
    }
)


def _native_format(data: bytes) -> str | None:
    """Name the executable format whose magic bytes start ``data``, if any."""
    if data.startswith(b"\x7fELF"):
        return "ELF"
    if data[:4] in _MACH_O_MAGIC:
        return "Mach-O"
    # A DOS header names the offset of the PE signature at byte 0x3C, so text
    # that merely starts with "MZ" is not mistaken for a Windows binary.
    if data.startswith(b"MZ") and len(data) >= 0x40:
        offset = int.from_bytes(data[0x3C:0x40], "little")
        if data[offset : offset + 4] == b"PE\0\0":
            return "PE"
    return None


def _wheel_tag_problems(wheel: Path, archive: ZipFile) -> list[str]:
    problems: list[str] = []
    file_tag = "-".join(wheel.name.removesuffix(".whl").split("-")[-3:])
    if file_tag != _PURE_TAG:
        problems.append(f"file name tag {file_tag!r}, expected {_PURE_TAG!r}")
    wheel_files = [
        name
        for name in archive.namelist()
        if len(PurePosixPath(name).parts) == 2
        and PurePosixPath(name).parts[0].endswith(".dist-info")
        and PurePosixPath(name).name == "WHEEL"
    ]
    if len(wheel_files) != 1:
        problems.append("the wheel must contain exactly one .dist-info/WHEEL file")
        return problems
    metadata = Parser().parsestr(archive.read(wheel_files[0]).decode("utf-8"))
    tags = [str(tag).strip() for tag in metadata.get_all("Tag") or []]
    if tags != [_PURE_TAG]:
        problems.append(f"WHEEL tags {tags}, expected [{_PURE_TAG!r}]")
    purelib = str(metadata.get("Root-Is-Purelib", "")).strip().lower()
    if purelib != "true":
        problems.append("WHEEL must declare Root-Is-Purelib: true")
    return problems


def validate_wheel_contents(wheel: Path) -> None:
    """Check that ``wheel`` is the complete pure-Python Scarf wheel."""
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        problems = _wheel_tag_problems(wheel, archive)
        for info in archive.infolist():
            if info.is_dir():
                continue
            if PurePosixPath(info.filename).parts[0].endswith(".data"):
                problems.append(
                    f"{info.filename} is install-time data outside the import root"
                )
            with archive.open(info) as member:
                native = _native_format(member.read())
            if native is not None:
                problems.append(f"{info.filename} is a native {native} binary")
    retired = sorted(_RETIRED_MODULES.intersection(names))
    missing = sorted(_REQUIRED_MODULES.difference(names))
    if retired:
        problems.append(f"retired modules are present: {retired}")
    if missing:
        problems.append(f"required modules are missing: {missing}")
    if problems:
        raise RuntimeError(
            "\n".join(
                ["Wheel contents violate the pure-Python contract:"]
                + [f"- {problem}" for problem in problems]
            )
        )


def installs_tsne_extra() -> bool:
    """Whether the smoke installs the tsne extra on this platform.

    sgtsnepi publishes Linux x86_64 wheels for every supported Python, and the
    test and docs extras install it there, so the smoke expects t-SNE on Linux
    x86_64 and the installation guidance everywhere else.
    """
    return sys.platform == "linux" and platform.machine() == "x86_64"


def _environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def smoke_installed_wheel(wheel: Path) -> None:
    """Install ``wheel`` into a clean environment, then import and use it.

    The environment holds only the wheel, its dependencies, and on Linux
    x86_64 the ``tsne`` extra. Its interpreter runs in isolated mode from a
    scratch directory, so neither ``PYTHONPATH`` nor a source checkout can
    stand in for the installed package.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    with_tsne = installs_tsne_extra()
    requirement = f"scarf[tsne] @ {wheel.as_uri()}" if with_tsne else str(wheel)
    env = os.environ.copy()
    env["HNSWLIB_NO_NATIVE"] = "1"
    with tempfile.TemporaryDirectory(
        prefix="scarf-wheel-smoke-", ignore_cleanup_errors=True
    ) as temp_dir:
        root = Path(temp_dir)
        environment = root / "environment"
        subprocess.run(
            ["uv", "venv", "--python", python_version, str(environment)],
            cwd=root,
            env=env,
            check=True,
        )
        python = _environment_python(environment)
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), requirement],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", _SMOKE_CODE],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                str(_WORKFLOW_SCRIPT),
                "with-tsne" if with_tsne else "without-tsne",
            ],
            cwd=root,
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
