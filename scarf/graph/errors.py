import math
from collections.abc import Mapping
from typing import TypeAlias


ErrorContextValue: TypeAlias = str | int | float | bool | None


def _restore_incompatible_analysis_state_error(
    message: str,
    code: str,
    context: dict[str, ErrorContextValue],
) -> "IncompatibleAnalysisStateError":
    return IncompatibleAnalysisStateError(message, code=code, context=context)


class IncompatibleAnalysisStateError(ValueError):
    """Stored analysis provenance cannot satisfy the current contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, ErrorContextValue],
    ) -> None:
        if not isinstance(code, str) or not code:
            raise TypeError("code must be a non-empty string")
        copied_context = dict(context)
        if not all(isinstance(key, str) for key in copied_context):
            raise TypeError("context keys must be strings")
        if not all(
            value is None or isinstance(value, str | int | float | bool)
            for value in copied_context.values()
        ):
            raise TypeError("context values must be JSON scalar values")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in copied_context.values()
        ):
            raise ValueError("context floats must be finite")
        super().__init__(message)
        self.code = code
        self.context = copied_context

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            _restore_incompatible_analysis_state_error,
            (str(self), self.code, self.context),
        )
