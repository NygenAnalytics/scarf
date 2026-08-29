import ast
import subprocess
import sys
from functools import cache
from importlib.util import find_spec
from pathlib import Path


_SCARF_ROOT = Path(__file__).resolve().parents[1] / "scarf"
_MOVED_SYMBOLS = {
    "datastore.datastore": {
        "_MARKER_OUT_COLUMNS",
        "_MARKER_STAT_COLUMNS",
        "_feature_column_chunk",
        "_group_assignment_digest",
        "_load_marker_cluster_frame",
        "_marker_stats_matrix",
        "_scatter_feature_clusters",
        "_shared_marker_feature_index",
        "_validated_pseudotime_regressor",
        "_validate_assay_pseudotime",
        "_write_compact_marker_stats",
    },
    "datastore.graph_datastore": {
        "EMBEDDING_CACHE_MAX_BYTES",
        "_make_source_sink_vector",
        "_random_walk_laplacian_transpose",
        "_select_pseudotime_component",
        "_truncated_pba_potential",
        "_validate_source_sink_labels",
        "_validate_source_sink_vector",
    },
    "knn_utils": {
        "_is_umap_version_new",
        "calc_snn",
        "export_knn_to_mtx",
        "merge_graphs",
        "run_sgtsne",
        "self_query_knn",
        "smoothen_dists",
        "weight_sort_indices",
        "wnn_integration",
    },
}


def _upward_imports(
    package_name: str,
    forbidden_packages: set[str],
    allowed_modules: frozenset[str] = frozenset(),
) -> set[tuple[str, str]]:
    package_root = _SCARF_ROOT / package_name
    violations: set[tuple[str, str]] = set()

    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            targets: set[str] = set()
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) > 1 and parts[0] == "scarf":
                        targets.add(".".join(parts[1:]))
            elif isinstance(node, ast.ImportFrom):
                if node.level >= 2:
                    if node.module:
                        targets.add(node.module)
                    else:
                        targets.update(alias.name for alias in node.names)
                elif node.level == 0 and node.module:
                    parts = node.module.split(".")
                    if parts[0] == "scarf":
                        if len(parts) > 1:
                            targets.add(".".join(parts[1:]))
                        else:
                            targets.update(alias.name for alias in node.names)

            for target in targets - allowed_modules:
                root = target.split(".")[0]
                if root in forbidden_packages:
                    violations.add((path.relative_to(package_root).as_posix(), root))

    return violations


def _root_imports(path: Path) -> set[str]:
    parent_parts = list(path.relative_to(_SCARF_ROOT).parent.parts)
    if parent_parts == ["."]:
        parent_parts = []
    imports: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "scarf" and len(parts) > 1:
                    imports.add(".".join(parts[1:]))
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level == 0:
            if not node.module:
                continue
            parts = node.module.split(".")
            if parts[0] != "scarf":
                continue
            relative_parts = parts[1:]
        else:
            up = node.level - 1
            relative_parts = parent_parts[: len(parent_parts) - up]
            if node.module:
                relative_parts += node.module.split(".")

        if relative_parts:
            imports.add(".".join(relative_parts))
        else:
            imports.update(alias.name for alias in node.names)

    return imports


@cache
def _root_imports_by_path() -> dict[str, set[str]]:
    return {
        path.relative_to(_SCARF_ROOT).as_posix(): _root_imports(path)
        for path in _SCARF_ROOT.rglob("*.py")
        if path != _SCARF_ROOT / "__init__.py"
    }


def _facade_importers(facade_name: str) -> set[str]:
    return {
        relative
        for relative, imports in _root_imports_by_path().items()
        if facade_name in imports
    }


def _resolved_module(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        if node.module == "scarf":
            return ""
        if node.module and node.module.startswith("scarf."):
            return node.module.removeprefix("scarf.")
        return None

    parent_parts = list(path.relative_to(_SCARF_ROOT).parent.parts)
    up = node.level - 1
    module_parts = parent_parts[: len(parent_parts) - up]
    if node.module:
        module_parts.extend(node.module.split("."))
    return ".".join(module_parts)


def _runtime_import_modules(
    path: Path,
    *,
    include_function_local: bool = True,
) -> set[str]:
    class RuntimeImportVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.modules: set[str] = set()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if include_function_local:
                self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if include_function_local:
                self.generic_visit(node)

        def visit_If(self, node: ast.If) -> None:
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                for child in node.orelse:
                    self.visit(child)
                return
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name.startswith("scarf."):
                    self.modules.add(alias.name.removeprefix("scarf."))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module_name = _resolved_module(path, node)
            if module_name is not None:
                if node.level > 0 and node.module is None:
                    for alias in node.names:
                        resolved = (
                            f"{module_name}.{alias.name}" if module_name else alias.name
                        )
                        self.modules.add(resolved)
                else:
                    self.modules.add(module_name)

    visitor = RuntimeImportVisitor()
    visitor.visit(ast.parse(path.read_text(), filename=str(path)))
    return visitor.modules


def _attribute_parts(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        parts = _attribute_parts(node.value)
        if parts is not None:
            return [*parts, node.attr]
    return None


def _moved_symbol_imports() -> set[tuple[str, str, str]]:
    violations: set[tuple[str, str, str]] = set()
    for path in _SCARF_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases: dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("scarf."):
                        continue
                    module_name = alias.name.removeprefix("scarf.")
                    if alias.asname:
                        aliases[alias.asname] = module_name
                    else:
                        aliases["scarf"] = ""
                continue
            if not isinstance(node, ast.ImportFrom):
                continue

            module_name = _resolved_module(path, node)
            if module_name is None:
                continue
            moved_symbols = _MOVED_SYMBOLS.get(module_name, set())
            for alias in node.names:
                if alias.name in moved_symbols:
                    violations.add(
                        (
                            path.relative_to(_SCARF_ROOT).as_posix(),
                            module_name,
                            alias.name,
                        )
                    )
                imported_module = ".".join(filter(None, [module_name, alias.name]))
                if imported_module in _MOVED_SYMBOLS:
                    aliases[alias.asname or alias.name] = imported_module

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts = _attribute_parts(node)
            if parts is None or parts[0] not in aliases:
                continue
            resolved = [*filter(None, aliases[parts[0]].split(".")), *parts[1:]]
            if len(resolved) < 2:
                continue
            module_name = ".".join(resolved[:-1])
            symbol_name = resolved[-1]
            if symbol_name in _MOVED_SYMBOLS.get(module_name, set()):
                violations.add(
                    (
                        path.relative_to(_SCARF_ROOT).as_posix(),
                        module_name,
                        symbol_name,
                    )
                )

    return violations


def test_storage_has_no_upward_dependencies():
    assert (
        _upward_imports(
            "storage",
            {"assay", "datastore", "plotting", "writers"},
            allowed_modules=frozenset({"assay.classification"}),
        )
        == set()
    )


def test_matrix_has_no_domain_or_orchestration_dependencies():
    assert (
        _upward_imports(
            "matrix",
            {
                "assay",
                "clustering",
                "datastore",
                "embeddings",
                "features",
                "mapping",
                "merge",
                "metadata",
                "metrics",
                "neighbors",
                "plotting",
                "quality_control",
                "readers",
                "trajectory",
                "writers",
            },
        )
        == set()
    )


def test_plotting_does_not_import_datastore():
    assert _upward_imports("plotting", {"datastore"}) == set()


def test_datastore_plot_namespace_defers_plotting_imports():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

from scarf.datastore.datastore import DataStore

optional = ("matplotlib", "seaborn")
assert not any(
    name == prefix or name.startswith(f"{prefix}.")
    for name in sys.modules
    for prefix in ("scarf.plotting", *optional)
)

store = object.__new__(DataStore)
accessor = store.plots

assert type(accessor).__module__ == "scarf.datastore.plot_accessor"
concrete_modules = {
    "scarf.plotting.composition",
    "scarf.plotting.diagnostics",
    "scarf.plotting.distribution",
    "scarf.plotting.embedding",
    "scarf.plotting.embedding_raster",
    "scarf.plotting.heatmaps",
    "scarf.plotting.summary",
}
assert concrete_modules.isdisjoint(sys.modules)
assert not any(
    name == prefix or name.startswith(f"{prefix}.")
    for name in sys.modules
    for prefix in optional
)

import scarf.plotting as plotting

_ = plotting.embedding
assert "scarf.plotting.embedding" in sys.modules
assert not any(
    name == prefix or name.startswith(f"{prefix}.")
    for name in sys.modules
    for prefix in optional
)
""",
        ],
        check=True,
    )


def test_algorithm_domains_do_not_import_orchestration_or_io():
    # storage.refs holds the artifact reference value type and reads no store,
    # so results that a caller persists may name it. Imported-embedding
    # persistence is the one honest embeddings-to-storage adapter. Trajectory
    # artifact contracts are the corresponding narrow persistence adapter for
    # validating domain payloads without moving their semantics into DataStore.
    forbidden = {"datastore", "plotting", "readers", "writers"}
    storage_exceptions = {
        "clustering": set(),
        "embeddings": {"imported_storage.py"},
        "trajectory": {"artifacts.py"},
    }
    for package_name in ("clustering", "embeddings", "trajectory"):
        package_root = _SCARF_ROOT / package_name
        if package_root.is_dir():
            assert _upward_imports(package_name, forbidden) == set()
            storage_edges = _upward_imports(
                package_name,
                {"storage"},
                frozenset({"storage.refs"}),
            )
            assert {path for path, _target in storage_edges} == storage_exceptions[
                package_name
            ]
    assert {path for path, _target in _upward_imports("clustering", {"storage"})} == {
        "paris_multiscale.py"
    }


def test_artifact_reference_module_has_no_storage_dependencies():
    path = _SCARF_ROOT / "storage" / "refs.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= {"re", "collections", "dataclasses", "typing"}


def test_metrics_and_embedding_harmony_avoid_runtime_orchestration_and_io_imports():
    forbidden_roots = {
        "datastore",
        "merge",
        "plotting",
        "readers",
        "storage",
        "writers",
    }
    paths = [
        *_SCARF_ROOT.joinpath("metrics").glob("*.py"),
        *_SCARF_ROOT.joinpath("embeddings", "harmony").glob("*.py"),
    ]
    for path in paths:
        runtime_imports = _runtime_import_modules(path)
        assert not {
            module_name
            for module_name in runtime_imports
            if module_name.split(".", 1)[0] in forbidden_roots
        }


def test_pca_and_lsi_implementations_live_under_embeddings():
    neighbor_source = (_SCARF_ROOT / "neighbors" / "stream.py").read_text()
    reduction_source = (_SCARF_ROOT / "embeddings" / "reduction.py").read_text()

    assert "sklearn.decomposition" not in neighbor_source
    assert "IncrementalPCA" in reduction_source
    assert "TruncatedSVD" in reduction_source


def test_extracted_domains_have_only_narrow_storage_dependencies():
    forbidden = {"datastore", "plotting", "readers", "writers"}
    storage_exceptions = {
        "features": {
            "enrichment/results.py",
            "genomic/melding.py",
            "markers/batching.py",
            "markers/search.py",
            "statistical.py",
        },
        "neighbors": set(),
        "quality_control": {"doublets.py"},
    }
    for package_name, allowed_storage_importers in storage_exceptions.items():
        assert _upward_imports(package_name, forbidden) == set()
        storage_edges = _upward_imports(package_name, {"storage"})
        storage_importers = {path for path, _target in storage_edges}
        assert storage_importers == allowed_storage_importers
        if package_name == "features":
            statistical_imports = _root_imports(
                _SCARF_ROOT / "features" / "statistical.py"
            )
            assert {
                target
                for target in statistical_imports
                if target == "storage" or target.startswith("storage.")
            } == {"storage.refs"}


def test_read_paths_take_chunk_geometry_only_from_the_storage_geometry_module():
    # scarf/storage/geometry.py owns the one read of an array's chunk grid.
    # Write-path chunk specs in layout.py and sharding.py are a separate concern
    # and stay out of this guard.
    read_path_modules = (
        "assay/rna.py",
        "datastore/_operations/graph.py",
        "datastore/_operations/mapping.py",
        "mapping/confidence.py",
        "mapping/hashing.py",
        "matrix/chunked.py",
        "metadata/rows.py",
        "storage/artifacts.py",
        "storage/copy.py",
        "storage/feature_stream.py",
        "storage/partition.py",
    )
    offenders = set()
    for relative_path in read_path_modules:
        path = _SCARF_ROOT / relative_path
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            reads_subscript = (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in {"chunks", "shards"}
            )
            reads_getattr = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in {"chunks", "shards"}
            )
            if reads_subscript or reads_getattr:
                offenders.add((relative_path, node.lineno))

    assert offenders == set()


def test_mapping_does_not_import_orchestration_or_general_io():
    assert (
        _upward_imports(
            "mapping",
            {"datastore", "plotting", "readers", "writers"},
        )
        == set()
    )
    storage_importers = {
        path for path, target in _upward_imports("mapping", {"storage"})
    }
    assert storage_importers <= {
        "artifact.py",
        "confidence.py",
        "features.py",
        "hashing.py",
        "models.py",
        "projection.py",
        "reference.py",
    }


def test_internal_modules_use_canonical_storage_and_utility_paths():
    for facade_name in (
        "ann",
        "dendrogram",
        "knn_utils",
        "results",
        "umap",
        "writers",
        "parallel",
        "storage.zarr_store",
        "utils",
        "bio_data",
        "doublet_utils",
        "feat_utils",
        "meld_assay",
        "utils.blocks",
        "utils.memory",
        "utils.storage",
        "utils.system",
        "utils.windows",
    ):
        assert _facade_importers(facade_name) == set()


def test_internal_modules_do_not_use_moved_symbols_from_hybrid_facades():
    assert _moved_symbol_imports() == set()


def test_compatibility_only_modules_are_removed():
    retired = {
        _SCARF_ROOT / "_types.py",
        _SCARF_ROOT / "ann.py",
        _SCARF_ROOT / "bio_data.py",
        _SCARF_ROOT / "chunked.py",
        _SCARF_ROOT / "dendrogram.py",
        _SCARF_ROOT / "downloader.py",
        _SCARF_ROOT / "doublet_utils.py",
        _SCARF_ROOT / "feat_utils.py",
        _SCARF_ROOT / "harmony.py",
        _SCARF_ROOT / "harmony" / "__init__.py",
        _SCARF_ROOT / "harmony" / "api.py",
        _SCARF_ROOT / "harmony" / "models.py",
        _SCARF_ROOT / "harmony" / "optimizer.py",
        _SCARF_ROOT / "genomics" / "__init__.py",
        _SCARF_ROOT / "genomics" / "gff.py",
        _SCARF_ROOT / "genomics" / "intervals.py",
        _SCARF_ROOT / "genomics" / "melding.py",
        _SCARF_ROOT / "genomics" / "reference.py",
        _SCARF_ROOT / "knn_utils.py",
        _SCARF_ROOT / "mapping" / "coral.py",
        _SCARF_ROOT / "mapping_reference.py",
        _SCARF_ROOT / "mapping_utils.py",
        _SCARF_ROOT / "markers.py",
        _SCARF_ROOT / "markers" / "__init__.py",
        _SCARF_ROOT / "markers" / "batching.py",
        _SCARF_ROOT / "markers" / "rank.py",
        _SCARF_ROOT / "markers" / "regression.py",
        _SCARF_ROOT / "markers" / "search.py",
        _SCARF_ROOT / "meld_assay.py",
        _SCARF_ROOT / "metadata.py",
        _SCARF_ROOT / "metrics.py",
        _SCARF_ROOT / "neighbors" / "persistence.py",
        _SCARF_ROOT / "features" / "lowess.py",
        _SCARF_ROOT / "clustering" / "_paris_mdl.py",
        _SCARF_ROOT / "clustering" / "feature_graph.py",
        _SCARF_ROOT / "clustering" / "hierarchy.py",
        _SCARF_ROOT / "graph" / "imported_storage.py",
        _SCARF_ROOT / "parallel.py",
        _SCARF_ROOT / "plotting" / "unified.py",
        _SCARF_ROOT / "results.py",
        _SCARF_ROOT / "storage" / "zarr_store.py",
        _SCARF_ROOT / "trajectory" / "aggregation.py",
        _SCARF_ROOT / "symphony.py",
        _SCARF_ROOT / "umap.py",
    }
    assert not {
        path.relative_to(_SCARF_ROOT).as_posix() for path in retired if path.exists()
    }


def test_retired_root_import_paths_do_not_resolve():
    for module_name in (
        "scarf._types",
        "scarf.chunked",
        "scarf.downloader",
        "scarf.harmony",
        "scarf.genomics",
        "scarf.knn_utils",
        "scarf.mapping.coral",
        "scarf.markers",
        "scarf.plotting.unified",
        "scarf.clustering._paris_mdl",
        "scarf.clustering.feature_graph",
        "scarf.clustering.hierarchy",
        "scarf.features.lowess",
        "scarf.graph.imported_storage",
        "scarf.trajectory.aggregation",
        "scarf.lineage",
    ):
        assert find_spec(module_name) is None


def test_cytebase_and_lineage_live_in_packages():
    cytebase_root = _SCARF_ROOT / "cytebase"
    assert cytebase_root.is_dir()
    assert not (_SCARF_ROOT / "cytebase.py").exists()
    assert (cytebase_root / "__init__.py").is_file()
    assert not (_SCARF_ROOT / "lineage.py").exists()
    assert (_SCARF_ROOT / "storage" / "lineage.py").is_file()


def test_utility_modules_use_domain_names():
    retired_files = {
        "blocks.py",
        "memory.py",
        "storage.py",
        "system.py",
        "windows.py",
    }
    assert not retired_files.intersection(
        path.name for path in (_SCARF_ROOT / "utils").glob("*.py")
    )


def test_data_model_defers_domain_algorithms_until_method_calls():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import scarf.metadata

assert not any(
    name == "scarf.features" or name.startswith("scarf.features.")
    for name in sys.modules
)

import scarf.assay

assert not any(
    name == "scarf.trajectory" or name.startswith("scarf.trajectory.")
    for name in sys.modules
)
""",
        ],
        check=True,
    )


def test_features_facades_defer_nested_implementations():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import scarf
import scarf.features as features

assert "scarf.features.variability" not in sys.modules
assert "scarf.features.genomic.gff" not in sys.modules
assert "scarf.features.genomic.melding" not in sys.modules
assert "scarf.features.markers.search" not in sys.modules
assert "scarf.features.enrichment" not in sys.modules
assert "scarf.features.enrichment.net" not in sys.modules
assert "scarf.features.enrichment.results" not in sys.modules
assert "scarf.features.enrichment.aucell" not in sys.modules
assert "scarf.features.enrichment.waggr" not in sys.modules
assert "scarf.features.statistical" not in sys.modules

_ = scarf.read_gmt
assert "scarf.features.enrichment.net" in sys.modules
assert "scarf.features.enrichment.results" not in sys.modules
assert "scarf.features.enrichment.aucell" not in sys.modules
assert "scarf.features.enrichment.waggr" not in sys.modules
assert features.read_gmt is scarf.read_gmt

_ = features.EnrichmentResult
assert "scarf.features.enrichment.results" in sys.modules
assert "scarf.features.enrichment.aucell" not in sys.modules
assert "scarf.features.enrichment.waggr" not in sys.modules

_ = features.find_markers_by_rank
assert "scarf.features.markers.search" in sys.modules
assert "scarf.features.markers.batching" not in sys.modules

_ = features.compare_group_distributions
assert "scarf.features.statistical" in sys.modules
assert "scarf.features.markers.search" in sys.modules

_ = features.GffReader
assert "scarf.features.genomic.gff" in sys.modules
assert "scarf.features.genomic.melding" not in sys.modules
""",
        ],
        check=True,
    )


def test_metrics_and_merge_do_not_import_datastore_at_runtime():
    for module_name in ("scarf.metrics", "scarf.merge"):
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    f"import {module_name}; import sys; "
                    "assert 'scarf.datastore.datastore' not in sys.modules"
                ),
            ],
            check=True,
        )


def test_merge_implementations_are_runtime_isolated():
    merge_root = _SCARF_ROOT / "merge"
    assert merge_root.is_dir()
    assert not (_SCARF_ROOT / "merge.py").exists()
    required_files = {
        "__init__.py",
        "datasets.py",
        "features.py",
        "metadata.py",
        "models.py",
        "row_plan.py",
        "writer.py",
    }
    assert {path.name for path in merge_root.glob("*.py")} == required_files
    assert (
        _runtime_import_modules(
            merge_root / "__init__.py",
            include_function_local=False,
        )
        == set()
    )

    forbidden_roots = {"datastore", "mapping", "plotting", "readers", "writers"}
    for path in merge_root.glob("*.py"):
        if path.name == "__init__.py":
            continue
        runtime_imports = _runtime_import_modules(path)
        assert not {
            module_name
            for module_name in runtime_imports
            if module_name.split(".", 1)[0] in forbidden_roots
        }


def test_mapping_does_not_import_datastore_at_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import scarf.mapping; import sys; "
                "assert 'scarf.datastore.datastore' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_reader_implementations_are_runtime_isolated():
    readers_root = _SCARF_ROOT / "readers"
    assert readers_root.is_dir()
    assert not (_SCARF_ROOT / "readers.py").exists()
    required_files = {
        "__init__.py",
        "_text.py",
        "cellranger.py",
        "csv.py",
        "h5ad.py",
        "loom.py",
        "mtx.py",
        "seurat.py",
    }
    assert required_files.issubset(path.name for path in readers_root.glob("*.py"))
    assert _runtime_import_modules(
        readers_root / "__init__.py",
        include_function_local=False,
    ) == {"readers._text"}

    forbidden_roots = {"datastore", "merge", "plotting", "storage", "writers"}
    format_modules = {
        "readers.cellranger",
        "readers.csv",
        "readers.h5ad",
        "readers.loom",
        "readers.mtx",
        "readers.seurat",
    }
    reader_edges = {
        "readers.cellranger": {"readers.mtx"},
        "readers.mtx": {"readers.cellranger"},
        "readers.seurat": {"readers._rds", "readers._seurat"},
    }
    format_names = {name.rsplit(".", 1)[-1] for name in format_modules}
    implementation_paths = [
        readers_root / name for name in sorted(required_files - {"__init__.py"})
    ]
    for path in implementation_paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        relative_sibling_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is None
            for alias in node.names
        }
        assert relative_sibling_imports.isdisjoint(format_names)

        runtime_imports = _runtime_import_modules(path)
        assert not {
            module_name
            for module_name in runtime_imports
            if module_name.split(".", 1)[0] in forbidden_roots
        }

        current_module = f"readers.{path.stem}"
        if current_module not in format_modules:
            assert {
                module_name
                for module_name in runtime_imports
                if module_name.startswith("readers")
            } <= {"readers.get_file_handle"}
            continue
        allowed_edges = reader_edges.get(current_module, set())
        sibling_modules = format_modules - {current_module} - allowed_edges
        assert runtime_imports.isdisjoint(sibling_modules)

        allowed_reader_imports: set[str] = set(allowed_edges)
        if path.name == "cellranger.py":
            allowed_reader_imports.update({"readers._assay_names", "readers.read_file"})
        elif path.name == "h5ad.py":
            allowed_reader_imports.update(
                {"readers._assay_names", "readers._h5ad_inspect"}
            )
        assert not {
            module_name
            for module_name in runtime_imports
            if module_name.startswith("readers")
            and module_name not in allowed_reader_imports
        }


def test_writer_implementations_are_runtime_isolated():
    writers_root = _SCARF_ROOT / "writers"
    assert writers_root.is_dir()
    assert not (_SCARF_ROOT / "writers.py").exists()
    required_files = {
        "__init__.py",
        "_materialize.py",
        "_store.py",
        "cellranger.py",
        "counts_t.py",
        "csv.py",
        "export.py",
        "h5ad.py",
        "loom.py",
        "sparse.py",
        "subset.py",
        "seurat.py",
    }
    assert {path.name for path in writers_root.glob("*.py")} == required_files
    assert (
        _runtime_import_modules(
            writers_root / "__init__.py",
            include_function_local=False,
        )
        == set()
    )

    forbidden_roots = {"assay", "datastore", "mapping", "merge", "plotting"}
    # Shared RNA classifier is the intentional write/load boundary for countsT.
    allowed_assay_imports = {"assay.classification"}
    format_modules = {
        "writers.cellranger",
        "writers.csv",
        "writers.h5ad",
        "writers.loom",
        "writers.sparse",
        "writers.subset",
        "writers.seurat",
    }
    format_names = {name.rsplit(".", 1)[-1] for name in format_modules}
    facade_edges = {
        "writers.create_cell_data",
        "writers.create_zarr_count_assay",
        "writers.create_zarr_obj_array",
        "writers.load_count_store",
        "writers.load_zarr",
    }
    shared_edges = {"writers._materialize", "writers._store", "writers.counts_t"}
    matching_reader_exports = {
        "cellranger.py": {"CrReader"},
        "csv.py": {"CSVReader"},
        "h5ad.py": {"H5adReader"},
        "loom.py": {"LoomReader"},
        "seurat.py": {"SeuratReader"},
    }

    for path in writers_root.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        relative_sibling_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module in format_names
        }
        assert relative_sibling_imports == set()

        runtime_imports = _runtime_import_modules(path)
        assert not {
            module_name
            for module_name in runtime_imports
            if module_name.split(".", 1)[0] in forbidden_roots
            and module_name not in allowed_assay_imports
        }
        writer_edges = {
            module_name
            for module_name in runtime_imports
            if module_name == "writers" or module_name.startswith("writers.")
        }
        assert writer_edges <= shared_edges | facade_edges

        reader_imports = {
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and _resolved_module(path, node) == "readers"
        }
        imported_reader_exports = {
            alias.name for node in reader_imports for alias in node.names
        }
        assert imported_reader_exports <= matching_reader_exports.get(path.name, set())
        assert not {
            module_name
            for module_name in runtime_imports
            if module_name.startswith("readers.")
            and module_name != f"readers.{path.stem}"
        }


def test_assay_implementations_are_runtime_isolated():
    assay_root = _SCARF_ROOT / "assay"
    assert assay_root.is_dir()
    assert not (_SCARF_ROOT / "assay.py").exists()

    forbidden_roots = {"datastore", "merge", "plotting", "readers", "writers"}
    modality_modules = {"assay.adt", "assay.atac", "assay.rna"}
    modality_names = {name.rsplit(".", 1)[-1] for name in modality_modules}
    for path in assay_root.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        relative_sibling_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is None
            for alias in node.names
        }
        assert relative_sibling_imports.isdisjoint(modality_names)

        module_scope_imports = _runtime_import_modules(
            path,
            include_function_local=False,
        )
        assert not {
            module_name
            for module_name in module_scope_imports
            if module_name.split(".", 1)[0] in forbidden_roots
        }
        current_module = f"assay.{path.stem}"
        sibling_modules = modality_modules - {current_module}
        assert module_scope_imports.isdisjoint(sibling_modules)

        allowed_function_local = {"plotting"}
        if path.name == "base.py":
            allowed_function_local.add("assay.rna")
        if path.name == "classification.py":
            # Classifier resolves preset strings to modality classes.
            allowed_function_local |= modality_modules
        function_local_imports = (
            _runtime_import_modules(path)
            - module_scope_imports
            - allowed_function_local
        )
        assert not {
            module_name
            for module_name in function_local_imports
            if module_name.split(".", 1)[0] in forbidden_roots
        }
        assert function_local_imports.isdisjoint(sibling_modules)


def test_datastore_operation_mixins_are_runtime_isolated():
    from scarf.datastore._operations.clustering import _ClusteringOperationsMixin
    from scarf.datastore._operations.embeddings import _EmbeddingOperationsMixin
    from scarf.datastore._operations.features import _FeatureOperationsMixin
    from scarf.datastore._operations.graph import _GraphOperationsMixin
    from scarf.datastore._operations.integration_metrics import (
        _IntegrationMetricsOperationsMixin,
    )
    from scarf.datastore._operations.mapping import _MappingOperationsMixin
    from scarf.datastore._operations.mapping_reference import (
        _MappingReferenceOperationsMixin,
    )
    from scarf.datastore._operations.presentation import _PresentationOperationsMixin
    from scarf.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )
    from scarf.datastore._operations.trajectory import (
        _TrajectoryFeatureOperationsMixin,
        _TrajectoryOperationsMixin,
    )

    operations_root = _SCARF_ROOT / "datastore" / "_operations"
    facade_modules = {
        "datastore.base_datastore",
        "datastore.datastore",
        "datastore.graph_datastore",
        "datastore.mapping_datastore",
    }
    allowed_operation_helpers = {
        "datastore._operations.enrichment_store",
        "datastore._operations.paris_persistence",
    }
    for path in operations_root.glob("*.py"):
        runtime_imports = _runtime_import_modules(path)
        assert runtime_imports.isdisjoint(facade_modules)
        operation_imports = {
            module_name
            for module_name in runtime_imports
            if module_name == "datastore._operations"
            or module_name.startswith("datastore._operations.")
        }
        assert operation_imports.issubset(allowed_operation_helpers)

    mixins = (
        _EmbeddingOperationsMixin,
        _ClusteringOperationsMixin,
        _TrajectoryOperationsMixin,
        _MappingReferenceOperationsMixin,
        _GraphOperationsMixin,
        _MappingOperationsMixin,
        _QualityControlOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
        _IntegrationMetricsOperationsMixin,
        _PresentationOperationsMixin,
    )
    assert all("__init__" not in mixin.__dict__ for mixin in mixins)
    assert all(mixin.__bases__ == (object,) for mixin in mixins)


def test_analytical_producers_do_not_mutate_live_metadata():
    """Keep analytical results behind immutable refs at the module boundary."""
    operation_paths = sorted((_SCARF_ROOT / "datastore" / "_operations").glob("*.py"))
    producer_paths = [
        *operation_paths,
        _SCARF_ROOT / "embeddings" / "imported.py",
    ]
    forbidden_helpers = {
        "link_cell_data_column",
        "link_feature_data_column",
        "publish_feature_selection_alias",
    }
    forbidden_table_methods = {"drop", "insert", "reset_key", "update_key"}
    violations: list[tuple[str, int, str]] = []

    for path in producer_paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            parts = _attribute_parts(node.func)
            if not parts:
                continue
            called = parts[-1]
            if called in forbidden_helpers or (
                len(parts) >= 2
                and parts[-2] in {"cells", "feats"}
                and called in forbidden_table_methods
            ):
                violations.append(
                    (
                        path.relative_to(_SCARF_ROOT).as_posix(),
                        node.lineno,
                        ".".join(parts),
                    )
                )

    assert violations == []


def test_graph_latest_pointer_reads_are_absent():
    pointer_names = {
        "latest_reduction",
        "latest_ann",
        "latest_knn",
        "latest_graph",
        "latest_kmeans",
    }
    violations: list[tuple[str, int, str]] = []

    for path in _SCARF_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in pointer_names
            ):
                violations.append(
                    (
                        path.relative_to(_SCARF_ROOT).as_posix(),
                        node.lineno,
                        "latest pointer read",
                    )
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in pointer_names
            ):
                violations.append(
                    (
                        path.relative_to(_SCARF_ROOT).as_posix(),
                        node.lineno,
                        "latest pointer read",
                    )
                )

    assert violations == []
