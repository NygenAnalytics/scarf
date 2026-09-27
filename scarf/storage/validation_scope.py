"""Reuse artifact validation results inside one operation."""

import functools
import inspect
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar, cast

T = TypeVar("T")

_RESULTS: ContextVar[dict[Hashable, Any] | None] = ContextVar(
    "validated_artifacts", default=None
)


@contextmanager
def validation_scope() -> Iterator[None]:
    """Validate each input once while the scope is open.

    Complete artifacts are immutable and prepared row IDs are protected, so an
    operation that reaches the same input through several lineage paths can
    reuse the first validation instead of re-reading and re-hashing it. Nested
    scopes share the outermost one; failures are never cached.
    """
    if _RESULTS.get() is not None:
        yield
        return
    token = _RESULTS.set({})
    try:
        yield
    finally:
        _RESULTS.reset(token)


def validated_once(key: Hashable, validate: Callable[[], T]) -> T:
    """Return ``validate()``, computed once per open scope for ``key``."""
    results = _RESULTS.get()
    if results is None:
        return validate()
    if key not in results:
        results[key] = validate()
    return cast(T, results[key])


def store_key(group: Any) -> tuple[int, str]:
    """Identify a group within one operation by its store and path."""
    return id(group.store), str(group.path)


def scope_public_methods(cls: type, *, module_prefix: str) -> None:
    """Open one validation scope around every public method of ``cls``.

    Methods are wrapped on the classes that define them, so inheritance,
    signatures and docstrings stay as they were.
    """
    for owner in cls.__mro__:
        if not owner.__module__.startswith(module_prefix):
            continue
        for name, member in list(vars(owner).items()):
            if (
                name.startswith("_")
                or not inspect.isfunction(member)
                or getattr(member, "__validation_scoped__", False)
            ):
                continue
            setattr(owner, name, validation_scoped(member))


def validation_scoped(method: Callable[..., T]) -> Callable[..., T]:
    """Run ``method`` inside one validation scope."""

    @functools.wraps(method)
    def run(*args: Any, **kwargs: Any) -> T:
        with validation_scope():
            return method(*args, **kwargs)

    run.__validation_scoped__ = True  # type: ignore[attr-defined]
    return run
