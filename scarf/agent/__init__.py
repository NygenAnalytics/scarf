"""Bounded RNA analysis through existing Scarf pipelines and small decisions.

The prototype orchestrator and its persisted histories are not compatible with
this interface. See the agent-local README for the explicit migration boundary.
"""

from .._facade import lazy_facade as _lazy_facade

__all__ = [
    "analyze_rna",
    "analyze_rna_async",
    "resume_rna",
    "resume_rna_async",
    "open_analysis",
    "AnalysisRun",
    "Study",
    "AnalysisConfig",
    "RuntimeConfig",
]

__getattr__, __dir__ = _lazy_facade(
    __name__,
    {
        "analyze_rna": ".api",
        "analyze_rna_async": ".api",
        "resume_rna": ".api",
        "resume_rna_async": ".api",
        "open_analysis": ".api",
        "AnalysisRun": ".result",
        "Study": ".models",
        "AnalysisConfig": ".models",
        "RuntimeConfig": ".models",
    },
)
