import ast
import json
import os
import subprocess
import sys
from functools import cache
from importlib.util import find_spec
from pathlib import Path

import pytest


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


@cache
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


@cache
def _nodes(path: Path) -> tuple[ast.AST, ...]:
    return tuple(ast.walk(_tree(path)))


@pytest.fixture(scope="module", autouse=True)
def _release_parsed_sources():
    """Parse each source once for this module, then free the ~170 MiB of trees."""
    yield
    _nodes.cache_clear()
    _tree.cache_clear()
    _root_imports_by_path.cache_clear()


def _upward_imports(
    package_name: str,
    forbidden_packages: set[str],
    allowed_modules: frozenset[str] = frozenset(),
) -> set[tuple[str, str]]:
    package_root = _SCARF_ROOT / package_name
    # A renamed or removed package must fail here instead of passing vacuously.
    assert (package_root / "__init__.py").is_file(), package_root
    violations: set[tuple[str, str]] = set()

    for path in package_root.rglob("*.py"):
        for node in _nodes(path):
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

    for node in _nodes(path):
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
    visitor.visit(_tree(path))
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
        aliases: dict[str, str] = {}

        for node in _nodes(path):
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

        for node in _nodes(path):
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


# Scripts that must see ``sys.modules`` as a fresh interpreter would. Each one
# is registered beside the test that asserts its outcome.
_FRESH_IMPORT_CHECKS: dict[str, str] = {}
_LOADED_HELPER = """
import sys


def loaded(*prefixes):
    return sorted(
        name
        for name in sys.modules
        for prefix in prefixes
        if name == prefix or name.startswith(prefix + ".")
    )
"""
# Third-party packages the checked facades load eagerly. The probe imports them
# once, confirms that no Scarf module or package a check forbids came with
# them, and forks one child per check. Every check therefore starts from the
# same state, and none pays for these imports again. Off Linux, where forking a
# process with native libraries loaded is less safe, each check runs in its own
# fresh interpreter instead.
_FRESH_IMPORT_BASE = (
    "h5py",
    "numba",
    "numpy",
    "pandas",
    "scipy.sparse",
    "scipy.stats",
    "sklearn.cluster",
    "zarr",
)
_FRESH_IMPORT_FORBIDDEN_IN_BASE = ("scarf", "matplotlib", "seaborn", "pydantic_ai")
_FRESH_IMPORT_PROBE = """
import importlib
import json
import os
import subprocess
import sys
import traceback

request = json.load(sys.stdin)
for name in request["base"]:
    importlib.import_module(name)
preloaded = sorted(
    name for name in sys.modules if name.split(".")[0] in request["forbidden"]
)
if preloaded:
    raise SystemExit(f"The shared imports loaded {preloaded}")


def run_forked(name, script):
    read_end, write_end = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        os.close(read_end)
        error = ""
        try:
            exec(compile(script, name, "exec"), {"__name__": "__main__"})
        except BaseException:
            error = traceback.format_exc()
        with os.fdopen(write_end, "w") as pipe:
            pipe.write(error)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if error else 0)
    os.close(write_end)
    with os.fdopen(read_end) as pipe:
        error = pipe.read()
    _, status = os.waitpid(pid, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    return error or ("" if exit_code == 0 else f"exited with status {exit_code}")


def run_fresh(name, script):
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    if completed.returncode == 0:
        return ""
    return completed.stderr or f"exited with status {completed.returncode}"


run = run_forked if sys.platform == "linux" else run_fresh
results = {name: run(name, script) for name, script in request["checks"].items()}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(results, handle)
"""


@pytest.fixture(scope="module")
def fresh_import_errors(tmp_path_factory) -> dict[str, str]:
    """Run every registered fresh-interpreter check once and map it to its error."""
    output = tmp_path_factory.mktemp("fresh_imports") / "errors.json"
    request = {
        "base": _FRESH_IMPORT_BASE,
        "forbidden": _FRESH_IMPORT_FORBIDDEN_IN_BASE,
        "checks": {
            name: _LOADED_HELPER + script
            for name, script in _FRESH_IMPORT_CHECKS.items()
        },
    }
    # One BLAS thread keeps the probe single-threaded, so forking it is safe.
    single_threaded = dict.fromkeys(
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"), "1"
    )
    completed = subprocess.run(
        [sys.executable, "-c", _FRESH_IMPORT_PROBE, str(output)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        env=os.environ | single_threaded,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def _assert_fresh_import_check(errors: dict[str, str], name: str) -> None:
    assert errors[name] == "", f"Fresh import check {name!r} failed:\n{errors[name]}"


def test_storage_has_no_upward_dependencies():
    assert (
        _upward_imports(
            "storage",
            {"assay", "datastore", "plotting", "writers"},
        )
        == set()
    )


def test_storage_functions_do_not_hide_assays_behind_unrestricted_arguments():
    violations = []
    for path in (_SCARF_ROOT / "storage").glob("*.py"):
        for node in _nodes(path):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ):
                if argument.arg not in {"assay", "datastore"}:
                    continue
                annotation = (
                    ast.unparse(argument.annotation) if argument.annotation else ""
                )
                if annotation not in {"str", "str | None", "zarr.Group"}:
                    violations.append((path.name, node.name, argument.arg, annotation))
    assert violations == []
    forbidden_attributes = {"rawData", "rawDataT", "normed", "normMethod", "_get_assay"}
    for path in (_SCARF_ROOT / "storage").glob("*.py"):
        assert not any(
            isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
            for node in _nodes(path)
        ), str(path)


def test_execution_uses_one_resolved_contract_and_normalization_has_one_owner():
    from scarf.storage.execution import OperationPlan
    from scarf.storage.async_execution import AsyncStorageRunner
    import inspect

    assert (
        inspect.signature(AsyncStorageRunner).parameters["operation"].annotation
        is OperationPlan
    )
    for path in (_SCARF_ROOT / "storage").glob("*.py"):
        assert not any(
            isinstance(node, ast.ClassDef)
            and node.name in {"ExecutionPlan", "StreamAdmission"}
            for node in _nodes(path)
        )
        assert not any(
            isinstance(node, ast.FunctionDef)
            and node.name == "write_renorm_subset_to_zarr"
            for node in _nodes(path)
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


_FRESH_IMPORT_CHECKS["datastore_plot_namespace"] = """
from scarf.datastore.datastore import DataStore

optional = ("matplotlib", "seaborn")
assert not loaded("scarf.plotting", *optional), loaded("scarf.plotting", *optional)

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
assert concrete_modules.isdisjoint(sys.modules), concrete_modules & set(sys.modules)
assert not loaded(*optional), loaded(*optional)

import scarf.plotting as plotting

_ = plotting.embedding
assert "scarf.plotting.embedding" in sys.modules
assert not loaded(*optional), loaded(*optional)
"""


def test_datastore_plot_namespace_defers_plotting_imports(fresh_import_errors):
    _assert_fresh_import_check(fresh_import_errors, "datastore_plot_namespace")


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
    for package_name, allowed_storage_importers in storage_exceptions.items():
        assert _upward_imports(package_name, forbidden) == set()
        storage_edges = _upward_imports(
            package_name,
            {"storage"},
            frozenset({"storage.refs"}),
        )
        assert {path for path, _target in storage_edges} == allowed_storage_importers
    assert {path for path, _target in _upward_imports("clustering", {"storage"})} == {
        "paris_multiscale.py"
    }


def test_artifact_reference_module_has_no_storage_dependencies():
    path = _SCARF_ROOT / "storage" / "refs.py"
    imported = {
        alias.name.split(".")[0]
        for node in _nodes(path)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in _nodes(path)
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
    metrics_paths = list(_SCARF_ROOT.joinpath("metrics").glob("*.py"))
    harmony_paths = list(_SCARF_ROOT.joinpath("embeddings", "harmony").glob("*.py"))
    assert metrics_paths and harmony_paths
    for path in [*metrics_paths, *harmony_paths]:
        runtime_imports = _runtime_import_modules(path)
        assert not {
            module_name
            for module_name in runtime_imports
            if module_name.split(".", 1)[0] in forbidden_roots
        }


def test_pca_and_lsi_implementations_live_under_embeddings():
    neighbor_sources = [
        path.read_text() for path in (_SCARF_ROOT / "neighbors").glob("*.py")
    ]
    reduction_source = (_SCARF_ROOT / "embeddings" / "reduction.py").read_text()

    assert neighbor_sources
    assert not any("sklearn.decomposition" in source for source in neighbor_sources)
    assert "IncrementalPCA" in reduction_source
    assert "TruncatedSVD" in reduction_source


def test_extracted_domains_have_only_narrow_storage_dependencies():
    forbidden = {"datastore", "plotting", "readers", "writers"}
    storage_exceptions = {
        "features": {
            "aggregation.py",
            "enrichment/results.py",
            "genomic/melding.py",
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
            # Result records need refs; the shared test identities need the
            # artifact fingerprint helpers.
            statistical_imports = _root_imports(
                _SCARF_ROOT / "features" / "statistical.py"
            )
            assert {
                target
                for target in statistical_imports
                if target == "storage" or target.startswith("storage.")
            } == {"storage.artifacts", "storage.refs"}


def test_read_paths_take_chunk_geometry_only_from_the_storage_geometry_module():
    # scarf/storage/geometry.py owns the one read of an array's chunk grid.
    # Write-path chunk specs in layout.py and sharding.py are a separate concern
    # and stay out of this guard.
    read_path_modules = (
        "assay/rna.py",
        "datastore/_operations/graph.py",
        "datastore/_operations/mapping.py",
        "mapping/confidence.py",
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
        for node in _nodes(path):
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
        "label_transfer.py",
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


_FRESH_IMPORT_CHECKS["agent_facade"] = """
import scarf

assert "scarf.agent" not in sys.modules
assert not loaded("pydantic_ai"), loaded("pydantic_ai")

import scarf.agent as agent

runtime_modules = (
    "scarf.agent.evidence",
    "scarf.agent.execution",
    "scarf.agent.provider",
    "scarf.agent.workflow",
    "scarf.datastore",
    "pydantic_ai",
)
for public_name in (None, "Study", "analyze_rna", "AnalysisRun"):
    if public_name is not None:
        getattr(agent, public_name)
    assert not loaded(*runtime_modules), (public_name, loaded(*runtime_modules))
"""


def test_agent_facade_defers_numerical_and_provider_imports(fresh_import_errors):
    _assert_fresh_import_check(fresh_import_errors, "agent_facade")


_UNIX_ONLY_MODULES = frozenset(
    {"fcntl", "grp", "pty", "pwd", "resource", "termios", "tty"}
)


def _load_time_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Return the modules a source imports while it loads, outside any function."""
    imports: list[tuple[int, str]] = []
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))
        pending.extend(ast.iter_child_nodes(node))
    return imports


def test_unix_only_modules_are_imported_only_at_call_time():
    # Windows lacks these modules. A call-time import behind a platform check,
    # as in scarf.agent.records, keeps every module importable there.
    violations = sorted(
        (path.relative_to(_SCARF_ROOT).as_posix(), lineno, module)
        for path in _SCARF_ROOT.rglob("*.py")
        for lineno, module in _load_time_imports(_tree(path))
        if module.split(".")[0] in _UNIX_ONLY_MODULES
    )
    assert violations == []


def test_agent_support_modules_keep_narrow_dependencies():
    # Decisions stay independent of execution, the provider cannot reach Scarf
    # computation, and reports can be regenerated without opening a store.
    allowed_imports = {
        "models": set(),
        "choices": {"agent.models"},
        "prompts": set(),
        "records": set(),
        "provider": {"agent.models", "agent.prompts", "agent.records"},
        "rendering": {"agent.records", "agent.report_charts", "agent.report_html"},
        "report_html": {
            "agent.report_assets",
            "agent.report_style",
            "agent.report_text",
        },
        "report_assets": set(),
        "report_charts": set(),
        "report_style": set(),
        "report_text": set(),
    }
    for name, allowed in allowed_imports.items():
        path = _SCARF_ROOT / "agent" / f"{name}.py"
        assert _runtime_import_modules(path) <= allowed, name


def test_core_packages_do_not_import_scarf_agent():
    assert not {
        (relative, module)
        for relative, imports in _root_imports_by_path().items()
        if not relative.startswith("agent/")
        for module in imports
        if module == "agent" or module.startswith("agent.")
    }


def test_agent_internal_modules_import_concrete_owners():
    violations: set[tuple[str, str]] = set()
    agent_root = _SCARF_ROOT / "agent"
    for path in agent_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        for module in _runtime_import_modules(path):
            if module == "agent":
                violations.add((path.relative_to(agent_root).as_posix(), module))

    assert violations == set()


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
        _SCARF_ROOT / "neighbors" / "stream.py",
        _SCARF_ROOT / "features" / "lowess.py",
        _SCARF_ROOT / "features" / "markers" / "batching.py",
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
        "scarf.features.markers.batching",
        "scarf.neighbors.stream",
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


_FRESH_IMPORT_CHECKS["data_model"] = """
import scarf.metadata

assert not loaded("scarf.features"), loaded("scarf.features")

import scarf.assay

assert not loaded("scarf.trajectory"), loaded("scarf.trajectory")
"""


def test_data_model_defers_domain_algorithms_until_method_calls(fresh_import_errors):
    _assert_fresh_import_check(fresh_import_errors, "data_model")


_FRESH_IMPORT_CHECKS["features_facades"] = """
import scarf
import scarf.features as features

assert "scarf.features.variability" not in sys.modules
assert "scarf.features.genomic.intervals" not in sys.modules
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

_ = features.compare_group_distributions
assert "scarf.features.statistical" in sys.modules
assert "scarf.features.markers.search" in sys.modules

_ = features.get_feature_mappings
assert "scarf.features.genomic.intervals" in sys.modules
assert "scarf.features.genomic.melding" not in sys.modules
"""


def test_features_facades_defer_nested_implementations(fresh_import_errors):
    _assert_fresh_import_check(fresh_import_errors, "features_facades")


_FRESH_IMPORT_CHECKS |= {
    f"{facade}_runtime_imports": f"""
import scarf.{facade}

assert "scarf.datastore.datastore" not in sys.modules, loaded("scarf.datastore")
"""
    for facade in ("metrics", "merge", "mapping")
}


def test_metrics_and_merge_do_not_import_datastore_at_runtime(fresh_import_errors):
    _assert_fresh_import_check(fresh_import_errors, "metrics_runtime_imports")
    _assert_fresh_import_check(fresh_import_errors, "merge_runtime_imports")


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
    assert _runtime_import_modules(
        merge_root / "__init__.py",
        include_function_local=False,
    ) == {"_facade"}

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


def test_mapping_does_not_import_datastore_at_runtime(fresh_import_errors):
    _assert_fresh_import_check(fresh_import_errors, "mapping_runtime_imports")


def test_reader_implementations_are_runtime_isolated():
    readers_root = _SCARF_ROOT / "readers"
    assert readers_root.is_dir()
    assert not (_SCARF_ROOT / "readers.py").exists()
    required_files = {
        "__init__.py",
        "_text.py",
        "_sparse.py",
        "cellranger.py",
        "csv.py",
        "h5ad.py",
        "mtx.py",
        "seurat.py",
    }
    assert required_files.issubset(path.name for path in readers_root.glob("*.py"))
    assert _runtime_import_modules(
        readers_root / "__init__.py",
        include_function_local=False,
    ) == {"_facade"}

    forbidden_roots = {"datastore", "merge", "plotting", "storage", "writers"}
    format_modules = {
        "readers.cellranger",
        "readers.csv",
        "readers.h5ad",
        "readers.mtx",
        "readers.seurat",
    }
    shared_reader_imports = {
        "cellranger.py": {"readers._assay_names", "readers._sparse", "readers._text"},
        "csv.py": {"readers._text"},
        "h5ad.py": {
            "readers._assay_names",
            "readers._h5ad_columns",
            "readers._h5ad_inspect",
            "readers._sparse",
            "readers._text",
        },
        "mtx.py": {"readers._sparse", "readers._text"},
    }
    reader_edges = {
        "readers.mtx": {"readers.cellranger"},
        "readers.seurat": {
            "readers._rds",
            "readers._seurat",
            "readers._seurat.sources",
        },
    }
    format_names = {name.rsplit(".", 1)[-1] for name in format_modules}
    implementation_paths = [
        readers_root / name for name in sorted(required_files - {"__init__.py"})
    ]
    for path in implementation_paths:
        relative_sibling_imports = {
            alias.name
            for node in _nodes(path)
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
            assert not {
                module_name
                for module_name in runtime_imports
                if module_name.startswith("readers")
            }
            continue
        allowed_edges = reader_edges.get(current_module, set())
        sibling_modules = format_modules - {current_module} - allowed_edges
        assert runtime_imports.isdisjoint(sibling_modules)

        allowed_reader_imports: set[str] = set(allowed_edges)
        allowed_reader_imports.update(shared_reader_imports.get(path.name, set()))
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
        "sparse.py",
        "subset.py",
        "seurat.py",
    }
    assert {path.name for path in writers_root.glob("*.py")} == required_files
    assert _runtime_import_modules(
        writers_root / "__init__.py",
        include_function_local=False,
    ) == {"_facade"}

    forbidden_roots = {"assay", "datastore", "mapping", "merge", "plotting"}
    # Shared RNA classifier is the intentional write/load boundary for countsT.
    allowed_assay_imports = {"assay.classification", "assay.normalization"}
    format_modules = {
        "writers.cellranger",
        "writers.csv",
        "writers.h5ad",
        "writers.sparse",
        "writers.subset",
        "writers.seurat",
    }
    format_names = {name.rsplit(".", 1)[-1] for name in format_modules}
    facade_edges = {
        "writers.create_zarr_count_assay",
        "writers.create_zarr_obj_array",
    }
    shared_edges = {"writers._materialize", "writers._store", "writers.counts_t"}
    matching_reader_exports = {
        "cellranger.py": {"CrReader"},
        "csv.py": {"CSVReader"},
        "h5ad.py": {"H5adReader"},
        "seurat.py": {"SeuratReader"},
    }

    for path in writers_root.glob("*.py"):
        if path.name == "__init__.py":
            continue
        relative_sibling_imports = {
            node.module
            for node in _nodes(path)
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
            for node in _nodes(path)
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
        relative_sibling_imports = {
            alias.name
            for node in _nodes(path)
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
        "datastore._operations.statistical_store",
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
        for node in _nodes(path):
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
        for node in _nodes(path):
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


def _is_attrs(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "attrs"


def _reads_missing_mask_link(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        target, key = node.value, node.slice
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
    ):
        target, key = node.func.value, node.args[0]
    elif (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.In | ast.NotIn)
    ):
        target, key = node.comparators[0], node.left
    else:
        return False
    return (
        _is_attrs(target)
        and isinstance(key, ast.Constant)
        and key.value == "missing_mask"
    )


def test_missing_mask_links_are_resolved_by_the_canonical_reader():
    # Readers resolve a nullable array's attrs["missing_mask"] link through
    # storage.arrays.linked_missing_mask, which accepts only the canonical
    # __scarf_missing__<name> sibling. Storage and the materializers that write
    # and verify that layout (writers and merge) may read the attribute.
    allowed_packages = {"storage", "writers", "merge"}
    offenders = set()
    for path in _SCARF_ROOT.rglob("*.py"):
        relative = path.relative_to(_SCARF_ROOT)
        if relative.parts[0] in allowed_packages:
            continue
        for node in _nodes(path):
            if _reads_missing_mask_link(node):
                offenders.add((relative.as_posix(), node.lineno))

    assert offenders == set()
