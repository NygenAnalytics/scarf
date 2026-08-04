"""Isolation and install-hint tests for scarf.agent."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_SCARF_ROOT = Path(__file__).resolve().parents[1] / "scarf"
_CORE_PACKAGES = (
    "assay",
    "clustering",
    "datastore",
    "embeddings",
    "features",
    "graph",
    "mapping",
    "matrix",
    "merge",
    "metadata",
    "metrics",
    "neighbors",
    "plotting",
    "quality_control",
    "readers",
    "storage",
    "trajectory",
    "utils",
    "writers",
)


def test_require_pydantic_ai_missing_extra_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from scarf.agent import _deps

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "pydantic_ai" or name.startswith("pydantic_ai."):
            raise ImportError("simulated missing pydantic_ai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"scarf\[agent\]") as exc_info:
        _deps.require_pydantic_ai()
    assert "uv sync --extra agent" in str(exc_info.value)


def test_import_scarf_does_not_load_pydantic_ai() -> None:
    script = """
import sys
import scarf
assert "pydantic_ai" not in sys.modules
assert not any(name.startswith("pydantic_ai.") for name in sys.modules)
assert "scarf.agent" not in sys.modules
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_core_packages_do_not_import_scarf_agent() -> None:
    violations: list[str] = []
    for package_name in _CORE_PACKAGES:
        package_root = _SCARF_ROOT / package_name
        if not package_root.exists():
            continue
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "scarf.agent" or alias.name.startswith(
                            "scarf.agent."
                        ):
                            violations.append(path.as_posix())
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "scarf.agent" or node.module.startswith(
                        "scarf.agent."
                    ):
                        violations.append(path.as_posix())
    assert violations == []
