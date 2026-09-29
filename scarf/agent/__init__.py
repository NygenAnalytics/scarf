"""Optional automated RNA analysis with a small, lazy public interface."""

from .._facade import lazy_facade as _lazy_facade

__all__ = ["analyze_rna", "AutomatedWorkflowResult", "AnalysisError"]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "analyze_rna": ".orchestrator.api",
        "AutomatedWorkflowResult": ".orchestrator.models",
        "AnalysisError": ".orchestrator.models",
    },
)
