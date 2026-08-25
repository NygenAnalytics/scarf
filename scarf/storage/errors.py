import math
from collections.abc import Mapping
from typing import Any


type ArtifactErrorContextValue = str | int | float | bool | None


class ArtifactResolutionError(ValueError):
    """A machine-readable failure to resolve or validate an artifact."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, ArtifactErrorContextValue],
    ) -> None:
        if not isinstance(code, str) or not code:
            raise TypeError("Artifact resolution error code must be a non-empty string")
        normalized: dict[str, ArtifactErrorContextValue] = {}
        for key, value in context.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Artifact resolution error context keys must be strings"
                )
            if not isinstance(value, str | int | float | bool | type(None)):
                raise TypeError(
                    "Artifact resolution error context values must be JSON scalars"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    "Artifact resolution error context floats must be finite"
                )
            normalized[key] = value
        super().__init__(message)
        self.code = code
        self.context = normalized

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            _restore_artifact_resolution_error,
            (type(self), str(self), self.code, self.context),
        )


def _restore_artifact_resolution_error(
    error_type: type[ArtifactResolutionError],
    message: str,
    code: str,
    context: Mapping[str, ArtifactErrorContextValue],
) -> ArtifactResolutionError:
    return error_type(message, code=code, context=context)
