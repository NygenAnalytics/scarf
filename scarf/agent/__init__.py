"""Optional automated RNA analysis with a small, lazy public interface."""

from importlib import import_module
from typing import Any

__all__ = ["analyze_rna", "AutomatedWorkflowResult", "AnalysisError"]

for _export in __all__:
    globals().pop(_export, None)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".orchestrator.api" if name == "analyze_rna" else ".orchestrator.models"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
