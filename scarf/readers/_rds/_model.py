from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from ._lazy import LazyStringVector
from ._storage import RandomAccessStorage, TempManager
from ._types import (
    RdsClosedError,
    RdsMetadata,
    RdsSourceMetadata,
    RType,
)


@dataclass(frozen=True, slots=True)
class PairValue:
    """The CAR and CDR of a pairlist or language node."""

    car: "RNode"
    cdr: "RNode"


@dataclass(frozen=True, slots=True)
class ClosureValue:
    """A closure represented without evaluation."""

    environment: "RNode | None"
    formals: "RNode"
    body: "RNode"


@dataclass(frozen=True, slots=True)
class PromiseValue:
    """A promise represented without forcing it."""

    environment: "RNode | None"
    value: "RNode"
    expression: "RNode"


@dataclass(frozen=True, slots=True)
class EnvironmentValue:
    """Serialized fields of an environment."""

    locked: bool
    enclosure: "RNode"
    frame: "RNode"
    hash_table: "RNode"


@dataclass(frozen=True, slots=True)
class ExternalPointerValue:
    """Opaque external-pointer fields that are safe to inspect."""

    protected: "RNode"
    tag: "RNode"


@dataclass(frozen=True, slots=True)
class WeakReferenceValue:
    """Opaque weak-reference state."""

    ready_to_finalize: bool
    finalize_on_exit: bool


@dataclass(frozen=True, slots=True)
class PersistentValue:
    """A persistent-hook name represented as inert data."""

    names: LazyStringVector


@dataclass(frozen=True, slots=True)
class BytecodeValue:
    """Serialized bytecode and constants without execution."""

    code: "RNode"
    constants: "tuple[RNode, ...]"


@dataclass(frozen=True, slots=True)
class AltRepValue:
    """An ALTREP constructor description represented as inert data."""

    info: "RNode"
    state: "RNode"
    class_name: str | None
    package_name: str | None
    known: bool


@dataclass(eq=False, slots=True)
class RNode:
    """One identity-bearing node in an R serialization graph."""

    type: RType
    value: Any = None
    attributes: "RNode | None" = None
    tag: "RNode | None" = None
    object: bool = False
    gp: int = 0
    path: str = "$"
    offset: int = 0

    @property
    def is_null(self) -> bool:
        return self.type in {RType.NIL, RType.NIL_VALUE}

    def attribute(self, name: str) -> "RNode | None":
        return get_attribute(self, name)

    def slot(self, name: str) -> "RNode | None":
        return get_slot(self, name)

    def named(self, name: str) -> "RNode | None":
        return get_named(self, name)

    def attribute_items(self) -> "Iterator[tuple[str, RNode]]":
        return iter_attributes(self)

    def named_items(self) -> "Iterator[tuple[str | bytes | None, RNode]]":
        return iter_named(self)


def symbol_name(node: RNode | None) -> str | None:
    """Return the text carried by a symbol node."""
    if node is None or node.is_null or node.type is not RType.SYMBOL:
        return None
    value = node.value
    if not isinstance(value, RNode) or value.type is not RType.CHAR:
        return None
    if isinstance(value.value, bytes):
        return value.value.decode("utf-8", errors="surrogateescape")
    return value.value if isinstance(value.value, str) else None


def iter_pairlist(node: RNode | None) -> Iterator[RNode]:
    """Iterate pairlist cells while rejecting malformed tails and cycles."""
    current = node
    seen: set[int] = set()
    while current is not None and not current.is_null:
        if current.type not in {RType.PAIRLIST, RType.LANGUAGE, RType.DOTS}:
            raise TypeError(
                f"expected a pairlist at {current.path}, found {current.type.name}"
            )
        identity = id(current)
        if identity in seen:
            raise ValueError(f"cycle in pairlist at {current.path}")
        seen.add(identity)
        if not isinstance(current.value, PairValue):
            raise TypeError(f"pairlist node at {current.path} has no pair value")
        yield current
        current = current.value.cdr


def iter_attributes(node: RNode) -> Iterator[tuple[str, RNode]]:
    """Iterate named attributes or S4 slots in serialized order."""
    for cell in iter_pairlist(node.attributes):
        name = symbol_name(cell.tag)
        if name is None:
            continue
        pair = cell.value
        if not isinstance(pair, PairValue):
            continue
        yield name, pair.car


def get_attribute(node: RNode, name: str) -> RNode | None:
    """Look up an attribute without materializing unrelated values."""
    for candidate, value in iter_attributes(node):
        if candidate == name:
            return value
    return None


def get_slot(node: RNode, name: str) -> RNode | None:
    """Look up one S4 slot stored in the attribute pairlist."""
    return get_attribute(node, name)


def _named_values(node: RNode) -> Sequence[RNode]:
    if node.type not in {RType.VECTOR, RType.EXPRESSION}:
        raise TypeError(f"{node.type.name} is not a generic named vector")
    if not isinstance(node.value, tuple):
        raise TypeError(f"vector node at {node.path} has no node sequence")
    return node.value


def iter_named(node: RNode) -> Iterator[tuple[str | bytes | None, RNode]]:
    """Iterate a generic vector together with its names attribute."""
    values = _named_values(node)
    names_node = get_attribute(node, "names")
    if names_node is None or names_node.type is not RType.STRING:
        raise ValueError(f"named vector at {node.path} has no character names")
    names = names_node.value
    if not isinstance(names, LazyStringVector):
        raise TypeError(f"names at {names_node.path} are not lazily indexed")
    if len(names) != len(values):
        raise ValueError(
            f"name count {len(names)} does not match value count {len(values)} "
            f"at {node.path}"
        )
    for index, value in enumerate(values):
        yield names[index], value


def get_named(node: RNode, name: str) -> RNode | None:
    """Look up the first value with an exact serialized name."""
    for candidate, value in iter_named(node):
        if candidate == name:
            return value
    return None


class RdsDocument:
    """Own an indexed RDS graph and every backing resource it uses."""

    def __init__(
        self,
        root: RNode,
        *,
        metadata: RdsMetadata,
        source: RdsSourceMetadata,
        temp_manager: TempManager,
        direct_storage: RandomAccessStorage | None,
    ) -> None:
        self.root = root
        self.metadata = metadata
        self.source = source
        self._temp_manager = temp_manager
        self._direct_storage = direct_storage
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def temp_paths(self) -> tuple[str, ...]:
        return tuple(str(path) for path in self._temp_manager.paths)

    def close(self) -> None:
        """Close all backings and remove owned temporary files."""
        if self._closed:
            return
        self._closed = True
        if self._direct_storage is not None:
            self._direct_storage.close()
        self._temp_manager.close()

    def __enter__(self) -> "RdsDocument":
        if self.closed:
            raise RdsClosedError("RDS document is closed", path="$")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()
