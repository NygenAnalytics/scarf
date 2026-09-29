"""Lazy attribute loading shared by Scarf's package facades.

A facade maps each public name to the relative module that defines it. Reading
a name imports that module and binds the value in the facade. Exports whose
modules are already imported are bound at the same time, but a bound value is
only replaced when it is a same-named submodule, so monkeypatched attributes
survive. Reloading a facade discards its bound exports.
"""

import sys
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from importlib.util import resolve_name
from types import ModuleType
from typing import Any


class _LazyExports:
    """Resolve the lazy names of one facade module."""

    def __init__(
        self,
        module: ModuleType,
        exports: Mapping[str, str],
        modules: Iterable[str],
        set_module: bool,
    ) -> None:
        self.package = module.__name__
        self.namespace = vars(module)
        self.sources = {
            name: resolve_name(source, self.package) for name, source in exports.items()
        }
        self.modules = frozenset(modules)
        self.set_module = set_module

    def resolve(self, name: str) -> Any:
        """Import and bind one lazy name; the facade's module ``__getattr__``."""
        if name in self.modules:
            value = import_module(f"{self.package}.{name}")
            self.namespace[name] = value
            return value
        source = self.sources.get(name)
        if source is None:
            raise AttributeError(f"module {self.package!r} has no attribute {name!r}")
        value = getattr(import_module(source), name)
        self._bind(name, value)
        for other, other_source in self.sources.items():
            if other in self.namespace and not isinstance(
                self.namespace[other], ModuleType
            ):
                continue
            loaded = sys.modules.get(other_source)
            candidate = None if loaded is None else vars(loaded).get(other)
            if candidate is not None and not isinstance(candidate, ModuleType):
                self._bind(other, candidate)
        return value

    def names(self) -> list[str]:
        """Return bound and lazy names; the facade's module ``__dir__``."""
        return sorted({*self.namespace, *self.sources, *self.modules})

    def _bind(self, name: str, value: Any) -> None:
        if self.set_module:
            _set_owned_module(value, self.package)
        self.namespace[name] = value


def _module_name(value: Any) -> str:
    module = getattr(value, "__module__", None)
    return module if isinstance(module, str) else ""


def _set_owned_module(value: Any, package: str) -> None:
    """Report an object defined inside ``package`` as a member of its facade.

    Methods defined by an owned class move with it. Objects defined elsewhere,
    including inherited or shared functions, keep their own ``__module__``.
    """
    owned = f"{package}."
    if not _module_name(value).startswith(owned):
        return
    value.__module__ = package
    if not isinstance(value, type):
        return
    for member in vars(value).values():
        if isinstance(member, classmethod | staticmethod):
            member = member.__func__
        if callable(member) and _module_name(member).startswith(owned):
            member.__module__ = package


class _FacadeModule(ModuleType):
    """Module type whose lazy exports win over same-named submodules.

    Importing ``package.name`` binds that submodule to ``package.name``. When
    the facade also exports an object called ``name``, reading the attribute
    resolves the export instead of returning the submodule.
    """

    def __getattribute__(self, name: str) -> Any:
        namespace = ModuleType.__getattribute__(self, "__dict__")
        if isinstance(namespace.get(name), ModuleType):
            resolve = namespace.get("__getattr__")
            exports = getattr(resolve, "__self__", None)
            if isinstance(exports, _LazyExports) and name in exports.sources:
                return exports.resolve(name)
        return ModuleType.__getattribute__(self, name)


def lazy_facade(
    module_name: str,
    exports: Mapping[str, str],
    *,
    modules: Iterable[str] = (),
    set_module: bool = False,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Install lazy exports on a package facade and return its module hooks.

    Assign the result to the facade's ``__getattr__`` and ``__dir__``.

    Args:
        module_name: The facade's ``__name__``.
        exports: Public name mapped to the relative module that defines it.
        modules: Submodules that are lazy attributes of the facade.
        set_module: Report objects defined inside the package, and the
            methods of such classes, as members of the facade.

    Returns:
        The module ``__getattr__`` and ``__dir__`` functions.
    """
    module = sys.modules[module_name]
    loader = _LazyExports(module, exports, modules, set_module)
    for name in (*loader.sources, *loader.modules):
        loader.namespace.pop(name, None)
    module.__class__ = _FacadeModule
    return loader.resolve, loader.names
