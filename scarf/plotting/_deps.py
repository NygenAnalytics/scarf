"""Lazy loaders for optional plotting dependencies."""

from typing import Any


def require_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Scarf plotting requires matplotlib. Install with: pip install 'scarf[extra]'"
        ) from exc
    return plt, mpl


def require_seaborn() -> Any:
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "Scarf plotting requires seaborn. Install with: pip install 'scarf[extra]'"
        ) from exc
    return sns


def require_kneed() -> Any:
    try:
        from kneed import KneeLocator
    except ImportError as exc:
        raise ImportError(
            "Scarf elbow detection requires kneed. Install with: pip install 'scarf[extra]'"
        ) from exc
    return KneeLocator
