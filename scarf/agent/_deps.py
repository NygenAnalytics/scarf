"""Lazy loaders for optional agent dependencies."""

from typing import Any

AGENT_INSTALL_HINT = (
    "Scarf agent support requires the agent extra. "
    "Install with: pip install 'scarf[agent]' "
    "or: uv sync --extra agent"
)


def require_pydantic() -> tuple[Any, Any]:
    try:
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(AGENT_INSTALL_HINT) from exc
    return BaseModel, Field


def require_pydantic_ai() -> Any:
    try:
        import pydantic_ai
    except ImportError as exc:
        raise ImportError(AGENT_INSTALL_HINT) from exc
    return pydantic_ai
