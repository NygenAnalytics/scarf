import json
import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import cast


_EXPECTED_EXPORTS = {
    "ArtifactLineage": "scarf.storage.lineage",
    "ArtifactRef": "scarf.storage.refs",
    "ArtifactResolutionError": "scarf.storage.errors",
    "ArtifactStatus": "scarf.storage.artifacts",
    "CSVReader": "scarf.readers",
    "CSVtoZarr": "scarf.writers",
    "CrDirReader": "scarf.readers",
    "CrH5Reader": "scarf.readers",
    "CrReader": "scarf.readers",
    "CrToZarr": "scarf.writers",
    "DataStore": "scarf.datastore.datastore",
    "DataStoreSummary": "scarf.datastore.summary",
    "DataStoreMerge": "scarf.merge",
    "EnrichmentResult": "scarf.features.enrichment.results",
    "FateMappingResult": "scarf.trajectory.results",
    "GffReader": "scarf.features.genomic.gff",
    "H5adInspectResult": "scarf.readers",
    "H5adImportResult": "scarf.writers",
    "H5adReader": "scarf.readers",
    "H5adToZarr": "scarf.writers",
    "LoomReader": "scarf.readers",
    "LoomToZarr": "scarf.writers",
    "MtxReader": "scarf.readers",
    "MtxToZarr": "scarf.writers",
    "SeuratImportResult": "scarf.writers",
    "SeuratInspectResult": "scarf.readers",
    "SeuratReader": "scarf.readers",
    "SeuratToZarr": "scarf.writers",
    "MappingReference": "scarf.mapping.reference",
    "MappingResult": "scarf.mapping.models",
    "mount_datastore": "scarf.datastore.datastore",
    "PseudotimeAggregationResult": "scarf.trajectory.results",
    "PseudotimeMarkerResult": "scarf.trajectory.results",
    "PseudotimeScoreResult": "scarf.trajectory.results",
    "PipelineExecutionError": "scarf.datastore.pipeline_run",
    "PipelineRun": "scarf.datastore.pipeline_run",
    "SparseToZarr": "scarf.writers",
    "SubsetZarr": "scarf.writers",
    "clean_array": "scarf.utils",
    "configure_output": "scarf.utils",
    "controlled_compute": "scarf.utils",
    "coordinate_melding": "scarf.features.genomic.melding",
    "create_zarr_count_assay": "scarf.writers",
    "create_zarr_dataset": "scarf.writers",
    "create_zarr_obj_array": "scarf.writers",
    "chunked_to_zarr": "scarf.writers",
    "get_log_level": "scarf.utils",
    "inspect_h5ad": "scarf.readers",
    "inspect_mtx": "scarf.readers",
    "inspect_seurat": "scarf.readers",
    "load_zarr": "scarf.utils",
    "logger": "scarf.utils",
    "permute_into_chunks": "scarf.utils",
    "read_gmt": "scarf.features.enrichment.net",
    "rescale_array": "scarf.utils",
    "rolling_window": "scarf.utils",
    "set_verbosity": "scarf.utils",
    "compute_with_progress": "scarf.utils",
    "subset_assay_zarr": "scarf.writers",
    "system_call": "scarf.utils",
    "to_h5ad": "scarf.writers",
    "to_mtx": "scarf.writers",
    "tqdmbar": "scarf.utils",
    "tqdm_params": "scarf.utils",
    "write_renorm_subset_to_zarr": "scarf.writers",
}

_EXPECTED_MODULE_ATTRIBUTES = {
    "assay": "scarf.assay",
    "cytebase": "scarf.cytebase",
    "datastore": "scarf.datastore",
    "embeddings": "scarf.embeddings",
    "features": "scarf.features",
    "mapping": "scarf.mapping",
    "matrix": "scarf.matrix",
    "merge": "scarf.merge",
    "metadata": "scarf.metadata",
    "metrics": "scarf.metrics",
    "quality_control": "scarf.quality_control",
    "readers": "scarf.readers",
    "storage": "scarf.storage",
    "utils": "scarf.utils",
    "writers": "scarf.writers",
}

_EXPECTED_UTILS_EXPORTS = [
    "logger",
    "tqdmbar",
    "tqdm_params",
    "configure_output",
    "set_verbosity",
    "get_log_level",
    "system_call",
    "rescale_array",
    "clean_array",
    "load_zarr",
    "permute_into_chunks",
    "compute_with_progress",
    "controlled_compute",
    "iter_column_blocks",
    "process_rss_mb",
    "rss_peak_tracker",
    "array_digest",
    "rolling_window",
]

_EXPECTED_PLOTTING_EXPORTS = (
    "CategoricalScale",
    "CellField",
    "ColorScale",
    "DensityOverlay",
    "FeatureRef",
    "FeatureSummary",
    "Highlight",
    "LegendSpec",
    "NormalizationSpec",
    "PlotProvenance",
    "PlotOutput",
    "PlotOutputSettings",
    "PlotPanelTarget",
    "PlotRecipe",
    "PlotRecipeResult",
    "PlotResult",
    "PlotStep",
    "SizeScale",
    "StudyDesign",
    "THEMES",
    "cluster_tree",
    "cluster_connectivity",
    "collect_legends",
    "compose_results",
    "composition",
    "distribution",
    "dotplot",
    "elbow",
    "embedding",
    "embedding_raster",
    "graph_qc",
    "highly_variable_features",
    "label_panels",
    "marker_heatmap",
    "mapping_calibration",
    "mapping_confusion",
    "mapping_evidence",
    "mapping_score",
    "matrixplot",
    "modality_weights",
    "pseudotime_heatmap",
    "qc",
    "register_theme",
    "run_recipe",
    "theme_context",
)


def _run_probe(source: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, object], json.loads(result.stdout))


def _missing_distribution(_name: str) -> str:
    raise PackageNotFoundError


def test_version_resolution_prefers_distribution(tmp_path: Path):
    import scarf

    assert (
        scarf._resolve_version(lambda _name: "9.8.7", tmp_path / "missing.py")
        == "9.8.7"
    )


def test_version_resolution_reads_generated_file(tmp_path: Path):
    import scarf

    version_path = tmp_path / "_version.py"
    version_path.write_text("__version__ = version = '9.8.7'\n", encoding="utf-8")

    assert scarf._resolve_version(_missing_distribution, version_path) == "9.8.7"


def test_version_resolution_rejects_missing_or_invalid_file(tmp_path: Path):
    import scarf

    version_path = tmp_path / "_version.py"
    assert scarf._resolve_version(_missing_distribution, version_path) == "unavailable"

    version_path.write_text("version is missing\n", encoding="utf-8")
    assert scarf._resolve_version(_missing_distribution, version_path) == "unavailable"


def test_bare_import_is_lazy():
    lazy_names = [*_EXPECTED_EXPORTS, *_EXPECTED_MODULE_ATTRIBUTES]
    result = _run_probe(
        f"""
import json
import sys
import scarf

heavy_modules = {{
    "dask",
    "h5py",
    "hnswlib",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "zarr",
}}
lazy_names = {lazy_names!r}
print(json.dumps({{
    "boundLazyNames": sorted(name for name in lazy_names if name in vars(scarf)),
    "heavyModules": sorted(heavy_modules.intersection(sys.modules)),
    "plotsInDir": "plots" in dir(scarf),
    "plottingInDir": "plotting" in dir(scarf),
    "scarfModules": sorted(
        name for name in sys.modules
        if name == "scarf" or name.startswith("scarf.")
    ),
    "versionIsSet": isinstance(scarf.__version__, str) and bool(scarf.__version__),
}}))
"""
    )

    assert result == {
        "boundLazyNames": [],
        "heavyModules": [],
        "plotsInDir": False,
        "plottingInDir": False,
        "scarfModules": ["scarf"],
        "versionIsSet": True,
    }


def test_utils_package_is_lazy():
    result = _run_probe(
        f"""
import json
import sys
import scarf.utils as utils

exports = {_EXPECTED_UTILS_EXPORTS!r}
print(json.dumps({{
    "advertised": sorted(name for name in exports if name in dir(utils)),
    "bound": sorted(name for name in exports if name in vars(utils)),
    "exports": utils.__all__,
    "heavyModules": sorted({{"numba", "numpy", "scipy", "zarr"}}.intersection(sys.modules)),
}}))
"""
    )

    assert result == {
        "advertised": sorted(_EXPECTED_UTILS_EXPORTS),
        "bound": [],
        "exports": _EXPECTED_UTILS_EXPORTS,
        "heavyModules": [],
    }


def test_clustering_package_is_lazy():
    exports = [
        "BalancedCut",
        "CoalesceTree",
        "ParisClusterDiagnostic",
        "ParisClusteringResult",
        "adaptive_cut",
        "balanced_cut",
        "leiden_membership",
        "make_digraph",
        "paris_dendrogram",
        "straight_cut",
    ]
    result = _run_probe(
        f"""
import json
import sys
import scarf.clustering as clustering

exports = {exports!r}
print(json.dumps({{
    "advertised": sorted(name for name in exports if name in dir(clustering)),
    "bound": sorted(name for name in exports if name in vars(clustering)),
    "exports": clustering.__all__,
    "heavyModules": sorted({{"numba", "numpy", "scipy"}}.intersection(sys.modules)),
}}))
"""
    )

    assert result == {
        "advertised": sorted(exports),
        "bound": [],
        "exports": exports,
        "heavyModules": [],
    }


def test_plotting_package_is_lazy():
    result = _run_probe(
        f"""
import json
import sys
import scarf.plotting as plotting

exports = {_EXPECTED_PLOTTING_EXPORTS!r}
concrete = (
        "scarf.plotting.cluster_connectivity",
    "scarf.plotting.composition",
    "scarf.plotting.diagnostics",
    "scarf.plotting.distribution",
    "scarf.plotting.embedding",
    "scarf.plotting.embedding_raster",
    "scarf.plotting.heatmaps",
    "scarf.plotting.mapping",
    "scarf.plotting.modality_weights",
    "scarf.plotting.recipes",
    "scarf.plotting.summary",
)
print(json.dumps({{
    "advertised": sorted(name for name in exports if name in dir(plotting)),
    "bound": sorted(name for name in exports if name in vars(plotting)),
    "concreteModules": sorted(set(concrete).intersection(sys.modules)),
    "exports": plotting.__all__,
}}))
"""
    )

    assert result == {
        "advertised": sorted(_EXPECTED_PLOTTING_EXPORTS),
        "bound": [],
        "concreteModules": [],
        "exports": list(_EXPECTED_PLOTTING_EXPORTS),
    }


def test_utils_surface_preserves_module_metadata():
    utils = import_module("scarf.utils")

    assert utils.__all__ == _EXPECTED_UTILS_EXPORTS
    for name in _EXPECTED_UTILS_EXPORTS:
        value = getattr(utils, name)
        if callable(value):
            assert value.__module__ == "scarf.utils"


def test_public_exports_match_canonical_objects():
    import scarf

    assert scarf.__all__ == list(_EXPECTED_EXPORTS)
    assert isinstance(scarf.__version__, str)
    assert scarf.__version__
    assert set(_EXPECTED_EXPORTS).issubset(dir(scarf))

    for name, module_name in _EXPECTED_EXPORTS.items():
        canonical = getattr(import_module(module_name), name)
        exported = getattr(scarf, name)
        assert exported is canonical
        assert vars(scarf)[name] is canonical
        if callable(exported):
            assert exported.__module__ == module_name


def test_marker_facade_does_not_export_layout_internals():
    markers = import_module("scarf.features.markers")
    internal_names = {
        "LEGACY_STAT_COLUMNS",
        "MARKER_STAT_COLUMNS",
        "load_marker_table",
        "read_legacy_marker_table",
    }

    assert internal_names.isdisjoint(markers.__all__)
    assert internal_names.isdisjoint(dir(markers))


def test_graph_package_does_not_export_artifact_references():
    graph = import_module("scarf.graph")

    assert "ArtifactRef" not in graph.__all__
    assert not hasattr(graph, "ArtifactRef")


def test_domain_packages_export_canonical_objects():
    exports = {
        (
            "scarf.clustering",
            "ParisClusterDiagnostic",
        ): "scarf.clustering.paris_multiscale",
        (
            "scarf.clustering",
            "ParisClusteringResult",
        ): "scarf.clustering.paris_multiscale",
        ("scarf.clustering", "adaptive_cut"): "scarf.clustering.paris_multiscale",
        ("scarf.embeddings", "run_harmony"): "scarf.embeddings.harmony",
        ("scarf.features", "binned_sampling"): "scarf.features.scoring",
        ("scarf.features", "fit_lowess"): "scarf.features.variability",
        (
            "scarf.features",
            "EnrichmentResult",
        ): "scarf.features.enrichment.results",
        ("scarf.features", "read_gmt"): "scarf.features.enrichment.net",
        (
            "scarf.features",
            "select_highly_variable_features",
        ): "scarf.features.variability",
        ("scarf.features", "GffReader"): "scarf.features.genomic.gff",
        ("scarf.features", "coordinate_melding"): "scarf.features.genomic.melding",
        ("scarf.mapping", "MappingReference"): "scarf.mapping.reference",
        ("scarf.mapping", "MappingResult"): "scarf.mapping.models",
        (
            "scarf.features",
            "find_markers_by_rank",
        ): "scarf.features.markers.search",
        (
            "scarf.features",
            "compare_group_distributions",
        ): "scarf.features.statistical",
        (
            "scarf.features",
            "StatisticalTestResult",
        ): "scarf.features.statistical",
        (
            "scarf.features",
            "GroupComparisonResult",
        ): "scarf.features.statistical",
        (
            "scarf.features",
            "resolve_group_order",
        ): "scarf.features.statistical",
        ("scarf.matrix", "ChunkedArray"): "scarf.matrix.chunked",
        (
            "scarf.metrics",
            "ClusterSeparabilityResult",
        ): "scarf.metrics.cluster_separability",
        ("scarf.metrics", "clisi_knn"): "scarf.metrics.lisi",
        ("scarf.metrics", "compute_lisi"): "scarf.metrics.lisi",
        (
            "scarf.metrics",
            "evaluate_cluster_separability",
        ): "scarf.metrics.cluster_separability",
        ("scarf.metrics", "graph_connectivity"): "scarf.metrics.connectivity",
        ("scarf.metrics", "ilisi_knn"): "scarf.metrics.lisi",
        ("scarf.metrics", "silhouette_scoring"): "scarf.metrics.silhouette",
        (
            "scarf.quality_control",
            "assign_cell_cycle_phase",
        ): "scarf.quality_control.cell_cycle",
        (
            "scarf.quality_control",
            "simulate_doublet_pairs",
        ): "scarf.quality_control.doublets",
        (
            "scarf.quality_control",
            "s_phase_genes",
        ): "scarf.quality_control.cell_cycle_genes",
    }

    for (package_name, symbol), module_name in exports.items():
        package = import_module(package_name)
        canonical = import_module(module_name)
        assert getattr(package, symbol) is getattr(canonical, symbol)


def test_star_import_matches_all():
    result = _run_probe(
        """
import json

namespace = {}
exec("from scarf import *", namespace)
bound = sorted(name for name in namespace if not name.startswith("__"))
print(json.dumps({"bound": bound}))
"""
    )

    assert result["bound"] == sorted(_EXPECTED_EXPORTS)


def test_de_facto_module_attributes_resolve_lazily():
    result = _run_probe(
        f"""
import importlib
import json
import scarf

module_names = {_EXPECTED_MODULE_ATTRIBUTES!r}
before = sorted(name for name in module_names if name in vars(scarf))
advertised = sorted(name for name in module_names if name in dir(scarf))
identical = {{
    name: getattr(scarf, name) is importlib.import_module(module_name)
    for name, module_name in module_names.items()
}}
cached = sorted(name for name in module_names if name in vars(scarf))
print(json.dumps({{
    "advertised": advertised,
    "before": before,
    "cached": cached,
    "identical": identical,
}}))
"""
    )

    expected_names = sorted(_EXPECTED_MODULE_ATTRIBUTES)
    assert result["advertised"] == expected_names
    assert result["before"] == []
    assert result["cached"] == expected_names
    assert result["identical"] == dict.fromkeys(_EXPECTED_MODULE_ATTRIBUTES, True)


def test_legacy_dataset_download_api_is_absent():
    from importlib.util import find_spec

    import scarf

    assert not hasattr(scarf, "fetch_dataset")
    assert not hasattr(scarf, "show_available_datasets")
    assert find_spec("scarf.readers.datasets") is None


def test_retired_merge_names_are_absent():
    import scarf
    import scarf.merge as merge_module

    for name in ("DatasetMerge", "AssayMerge"):
        assert not hasattr(scarf, name)
        assert not hasattr(merge_module, name)


def test_retired_dask_names_are_absent():
    import scarf
    import scarf.storage.materialize as storage_materialize
    import scarf.utils as utils_module
    import scarf.utils.compute as compute_module
    import scarf.writers as writers_module
    import scarf.writers._materialize as writers_materialize

    for name in ("show_dask_progress", "dask_to_zarr"):
        assert not hasattr(scarf, name)
        assert not hasattr(utils_module, name)
        assert not hasattr(writers_module, name)
    assert not hasattr(compute_module, "show_dask_progress")
    assert not hasattr(storage_materialize, "dask_to_zarr")
    assert not hasattr(writers_materialize, "dask_to_zarr")
    assert "compute_with_progress" in dir(utils_module)


def test_lazy_facades_clear_cached_exports_on_reload():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib

cases = (
    ("scarf", "MappingResult"),
    ("scarf.features", "fit_lowess"),
    ("scarf.readers", "CSVReader"),
    ("scarf.writers", "CSVtoZarr"),
    ("scarf.merge", "DataStoreMerge"),
    ("scarf.utils", "clean_array"),
    ("scarf.neighbors", "diffusion_operator"),
    ("scarf.clustering", "balanced_cut"),
    ("scarf.embeddings", "initial_embedding"),
    ("scarf.trajectory", "PseudotimeScoreResult"),
    ("scarf.plotting", "embedding"),
)

for module_name, export_name in cases:
    module = importlib.import_module(module_name)
    original = getattr(module, export_name)
    setattr(module, export_name, object())
    importlib.reload(module)
    assert export_name not in vars(module), (module_name, export_name)
    assert getattr(module, export_name) is original, (module_name, export_name)
""",
        ],
        check=True,
    )


def test_zarr_warning_filter_does_not_make_import_eager():
    result = _run_probe(
        """
import json
import sys
import warnings

import scarf

zarr_was_loaded = "zarr" in sys.modules

import numpy as np
import zarr
from zarr.storage import MemoryStore

with warnings.catch_warnings(record=True) as caught:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_array("text", data=np.asarray(["a", "bb"]))

print(json.dumps({
    "unstableWarnings": [
        type(item.message).__name__
        for item in caught
        if type(item.message).__name__ == "UnstableSpecificationWarning"
    ],
    "zarrWasLoaded": zarr_was_loaded,
}))
"""
    )

    assert result == {"unstableWarnings": [], "zarrWasLoaded": False}


def test_modern_plotting_surface_matches_baseline():
    plotting = import_module("scarf.plotting")

    assert tuple(plotting.__all__) == _EXPECTED_PLOTTING_EXPORTS
    for name in _EXPECTED_PLOTTING_EXPORTS:
        assert hasattr(plotting, name)


def test_plotting_submodule_import_does_not_clobber_function_export():
    plotting = import_module("scarf.plotting")
    canonical_module = import_module("scarf.plotting.embedding")

    assert plotting.embedding is canonical_module.embedding


def test_legacy_plotting_surface_remains_absent():
    from importlib.util import find_spec

    from scarf import DataStore
    from scarf.datastore.plot_accessor import DataStorePlotAccessor

    plotting = import_module("scarf.plotting")

    assert hasattr(DataStore, "plots")
    assert DataStorePlotAccessor.__module__ == "scarf.datastore.plot_accessor"
    assert not hasattr(import_module("scarf"), "DataStorePlotAccessor")
    assert not [name for name in dir(DataStore) if name.startswith("plot_")]
    assert "unified_embedding" not in plotting.__all__
    assert not hasattr(plotting, "unified_embedding")
    assert find_spec("scarf.plots") is None
    assert find_spec("scarf.plotting._legacy") is None
    assert find_spec("scarf.plotting.unified") is None


def test_pipeline_accessor_has_a_public_import_path():
    from scarf import DataStore, PipelineExecutionError, PipelineRun
    from scarf.datastore.pipeline_accessor import PipelineAccessor

    assert hasattr(DataStore, "pipeline")
    assert PipelineAccessor.__module__ == "scarf.datastore.pipeline_accessor"
    assert PipelineRun.__module__ == "scarf.datastore.pipeline_run"
    assert PipelineExecutionError.__module__ == "scarf.datastore.pipeline_run"
    assert {"open", "list_runs", "run"} <= set(vars(PipelineAccessor))
