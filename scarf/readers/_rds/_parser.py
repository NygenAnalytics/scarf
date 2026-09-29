import math
import os
import struct
import sys
from dataclasses import dataclass, field
from types import FrameType

from ._lazy import (
    STRING_SOURCE_DECODED,
    STRING_SOURCE_PAYLOAD,
    LazyAtomicVector,
    LazyStringVector,
    decode_r_string,
    pack_string_descriptor,
)
from ._model import (
    AltRepValue,
    BytecodeValue,
    ClosureValue,
    EnvironmentValue,
    ExternalPointerValue,
    PairValue,
    PersistentValue,
    PromiseValue,
    RNode,
    RdsDocument,
    WeakReferenceValue,
    symbol_name,
)
from ._storage import (
    BufferedTempWriter,
    RandomAccessStorage,
    RdsInput,
    TempManager,
    prepare_payload,
)
from ._types import (
    R_INT_NA,
    RdsEncoding,
    RdsFormatError,
    RdsLimitError,
    RdsLimits,
    RdsMetadata,
    RType,
)


_RDATA_HEADERS = {
    b"RDA2\n",
    b"RDA3\n",
    b"RDB2\n",
    b"RDB3\n",
    b"RDX2\n",
    b"RDX3\n",
}
_PAIR_TYPES = {
    RType.PAIRLIST,
    RType.LANGUAGE,
    RType.CLOSURE,
    RType.PROMISE,
    RType.DOTS,
}
_SINGLETON_TYPES = {
    RType.NIL,
    RType.NIL_VALUE,
    RType.EMPTY_ENVIRONMENT,
    RType.BASE_ENVIRONMENT,
    RType.GLOBAL_ENVIRONMENT,
    RType.UNBOUND_VALUE,
    RType.MISSING_ARGUMENT,
    RType.BASE_NAMESPACE,
}
_BYTECODE_PAIR_MARKERS = {
    RType.LANGUAGE,
    RType.PAIRLIST,
    RType.BYTECODE_DEFINITION,
    RType.BYTECODE_REFERENCE,
    RType.BYTECODE_ATTR_LANGUAGE,
    RType.BYTECODE_ATTR_PAIRLIST,
}
# Nested values recurse through at most two parser frames per level; one more
# keeps a safety margin. The caller's stack plus a margin must also fit under
# the interpreter limit.
_FRAMES_PER_LEVEL = 3
_STACK_MARGIN = 64
_KNOWN_ALTREP = {
    "compact_intseq",
    "compact_realseq",
    "deferred_string",
    "wrap_complex",
    "wrap_integer",
    "wrap_logical",
    "wrap_raw",
    "wrap_real",
    "wrap_string",
}


@dataclass(frozen=True, slots=True)
class _ObjectInfo:
    type: RType
    object: bool
    has_attributes: bool
    has_tag: bool
    gp: int


@dataclass(slots=True)
class _BytecodeRepetitions:
    """Language cells that compiled R code shares within one bytecode tree."""

    count: int
    nodes: dict[int, RNode] = field(default_factory=dict)


def _stack_depth() -> int:
    depth = 0
    frame: FrameType | None = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


class _WireReader:
    def __init__(
        self,
        storage: RandomAccessStorage,
        *,
        position: int,
        encoding: RdsEncoding,
        byte_order: str | None,
    ) -> None:
        self.storage = storage
        self.position = position
        self.end = storage.size()
        self.encoding = encoding
        self.byte_order = byte_order
        self.native_encoding: str | None = None
        self._buffer = b""
        self._buffer_start = position

    def tell(self) -> int:
        return self.position

    def remaining(self) -> int:
        return self.end - self.position

    def _fill_buffer(self) -> None:
        count = min(64 * 1024, self.remaining())
        if count <= 0:
            self._buffer = b""
            self._buffer_start = self.position
            return
        self._buffer_start = self.position
        self._buffer = self.storage.read_at(self.position, count)

    def read_byte(self, *, path: str) -> int:
        relative = self.position - self._buffer_start
        if relative < 0 or relative >= len(self._buffer):
            self._fill_buffer()
            relative = 0
        if not self._buffer:
            raise RdsFormatError(
                "unexpected end of stream",
                path=path,
                offset=self.position,
            )
        result = self._buffer[relative]
        self.position += 1
        return result

    def read_exact(self, size: int, *, path: str) -> bytes:
        if size < 0:
            raise ValueError("size must be nonnegative")
        if size > self.remaining():
            raise RdsFormatError(
                f"unexpected end of stream while reading {size} bytes",
                path=path,
                offset=self.position,
            )
        if size == 0:
            return b""
        relative = self.position - self._buffer_start
        if 0 <= relative and relative + size <= len(self._buffer):
            result = self._buffer[relative : relative + size]
        else:
            result = self.storage.read_at(self.position, size, path=path)
        self.position += size
        return result

    def skip(self, size: int, *, path: str) -> int:
        if size < 0:
            raise ValueError("size must be nonnegative")
        offset = self.position
        if size > self.remaining():
            raise RdsFormatError(
                f"unexpected end of stream while skipping {size} bytes",
                path=path,
                offset=offset,
            )
        self.position += size
        return offset

    def read_word(self, *, path: str) -> bytes:
        while True:
            value = self.read_byte(path=path)
            if not chr(value).isspace():
                break
        result = bytearray()
        while not chr(value).isspace():
            result.append(value)
            if len(result) > 127:
                raise RdsFormatError(
                    "ASCII token exceeds 127 bytes",
                    path=path,
                    offset=self.position - len(result),
                )
            value = self.read_byte(path=path)
        return bytes(result)

    def read_int(self, *, path: str) -> int:
        offset = self.position
        if self.encoding is RdsEncoding.ASCII:
            word = self.read_word(path=path)
            if word == b"NA":
                return R_INT_NA
            try:
                return int(word, 10)
            except ValueError as error:
                raise RdsFormatError(
                    f"invalid ASCII integer {word!r}",
                    path=path,
                    offset=offset,
                ) from error
        order = ">" if self.encoding is RdsEncoding.XDR else self.byte_order
        if order not in {"<", ">"}:
            raise RdsFormatError(
                "native binary byte order is unknown",
                path=path,
                offset=offset,
            )
        return int(struct.unpack(f"{order}i", self.read_exact(4, path=path))[0])

    def read_real(self, *, path: str) -> float:
        offset = self.position
        if self.encoding is RdsEncoding.ASCII:
            word = self.read_word(path=path)
            if word == b"NA":
                return math.nan
            if word == b"NaN":
                return math.nan
            if word == b"Inf":
                return math.inf
            if word == b"-Inf":
                return -math.inf
            try:
                text = word.decode("ascii")
                return (
                    float.fromhex(text)
                    if text.lower().startswith(("0x", "-0x"))
                    else float(text)
                )
            except (UnicodeDecodeError, ValueError) as error:
                raise RdsFormatError(
                    f"invalid ASCII real {word!r}",
                    path=path,
                    offset=offset,
                ) from error
        order = ">" if self.encoding is RdsEncoding.XDR else self.byte_order
        if order not in {"<", ">"}:
            raise RdsFormatError(
                "native binary byte order is unknown",
                path=path,
                offset=offset,
            )
        return float(struct.unpack(f"{order}d", self.read_exact(8, path=path))[0])

    def read_ascii_string(self, length: int, *, path: str) -> bytes:
        if length == 0:
            return b""
        while True:
            value = self.read_byte(path=path)
            if not chr(value).isspace():
                break
        result = bytearray()
        escaped = {
            ord("n"): ord("\n"),
            ord("t"): ord("\t"),
            ord("v"): ord("\v"),
            ord("b"): ord("\b"),
            ord("r"): ord("\r"),
            ord("f"): ord("\f"),
            ord("a"): ord("\a"),
            ord("\\"): ord("\\"),
            ord("?"): ord("?"),
            ord("'"): ord("'"),
            ord('"'): ord('"'),
        }
        while len(result) < length:
            if value != ord("\\"):
                result.append(value)
            else:
                value = self.read_byte(path=path)
                mapped = escaped.get(value)
                if mapped is not None:
                    result.append(mapped)
                elif ord("0") <= value <= ord("7"):
                    digits = [value]
                    while len(digits) < 3:
                        candidate = self.read_byte(path=path)
                        if not ord("0") <= candidate <= ord("7"):
                            self.position -= 1
                            break
                        digits.append(candidate)
                    result.append(int(bytes(digits), 8))
                else:
                    result.append(value)
            if len(result) < length:
                value = self.read_byte(path=path)
        return bytes(result)

    def read_string(self, length: int, *, path: str) -> bytes:
        if self.encoding is RdsEncoding.ASCII:
            return self.read_ascii_string(length, path=path)
        return self.read_exact(length, path=path)


class _Parser:
    def __init__(
        self,
        storage: RandomAccessStorage,
        temp_manager: TempManager,
        limits: RdsLimits,
    ) -> None:
        self.storage = storage
        self.temp_manager = temp_manager
        self.limits = limits
        self.reader: _WireReader
        self.metadata: RdsMetadata
        self._objects = 0
        self._references: list[RNode] = []
        self._singletons: dict[RType, RNode] = {}
        self._maximumDepth = limits.max_depth

    def parse(self) -> RNode:
        # Deep input must end in RdsLimitError rather than RecursionError, so
        # the depth limit never exceeds what the remaining Python stack holds.
        stack_levels = (
            sys.getrecursionlimit() - _stack_depth() - _STACK_MARGIN
        ) // _FRAMES_PER_LEVEL
        self._maximumDepth = max(0, min(self.limits.max_depth, stack_levels))
        self.reader, self.metadata = self._parse_header()
        root = self._parse_node("$", 0)
        self._check_complete()
        self.temp_manager.flush()
        return root

    def _parse_header(self) -> tuple[_WireReader, RdsMetadata]:
        start = self.storage.tell()
        available = min(5, self.storage.size() - start)
        prefix = self.storage.read_at(start, available, path="$header")
        if prefix in _RDATA_HEADERS:
            raise RdsFormatError(
                "RData workspace header is not valid for RDS input",
                path="$header",
                offset=start,
            )
        if prefix.startswith(b"A\r\n"):
            encoding = RdsEncoding.ASCII
            position = start + 3
            byte_order = None
        elif prefix.startswith(b"A\n"):
            encoding = RdsEncoding.ASCII
            position = start + 2
            byte_order = None
        elif prefix.startswith(b"X\n"):
            encoding = RdsEncoding.XDR
            position = start + 2
            byte_order = ">"
        elif prefix.startswith(b"B\n"):
            encoding = RdsEncoding.NATIVE
            position = start + 2
            marker = self.storage.read_at(position, 4, path="$header.formatVersion")
            little = int.from_bytes(marker, "little", signed=True)
            big = int.from_bytes(marker, "big", signed=True)
            if little in {2, 3} and big not in {2, 3}:
                byte_order = "<"
            elif big in {2, 3} and little not in {2, 3}:
                byte_order = ">"
            else:
                raise RdsFormatError(
                    "cannot determine native binary byte order",
                    path="$header",
                    offset=position,
                )
        else:
            raise RdsFormatError(
                "unknown R serialization format marker",
                path="$header",
                offset=start,
            )

        reader = _WireReader(
            self.storage,
            position=position,
            encoding=encoding,
            byte_order=byte_order,
        )
        format_version = reader.read_int(path="$header.formatVersion")
        writer_version = reader.read_int(path="$header.writerVersion")
        minimum_version = reader.read_int(path="$header.minimumReaderVersion")
        if format_version not in {2, 3}:
            raise RdsFormatError(
                f"unsupported R serialization version {format_version}",
                path="$header.formatVersion",
                offset=position,
            )
        native_encoding: str | None = None
        if format_version == 3:
            length = reader.read_int(path="$header.nativeEncoding.length")
            self._check_string_length(
                length,
                path="$header.nativeEncoding",
                offset=reader.tell(),
                allow_missing=False,
            )
            raw = reader.read_string(length, path="$header.nativeEncoding")
            try:
                native_encoding = raw.decode("ascii")
            except UnicodeDecodeError as error:
                raise RdsFormatError(
                    "native encoding name is not ASCII",
                    path="$header.nativeEncoding",
                    offset=reader.tell() - length,
                ) from error
            reader.native_encoding = native_encoding
        return reader, RdsMetadata(
            encoding=encoding,
            format_version=format_version,
            writer_version=writer_version,
            minimum_reader_version=minimum_version,
            native_encoding=native_encoding,
            byte_order=byte_order,
        )

    def _check_complete(self) -> None:
        if self.reader.encoding is not RdsEncoding.ASCII:
            if self.reader.remaining() != 0:
                raise RdsFormatError(
                    f"{self.reader.remaining()} trailing bytes after root object",
                    path="$",
                    offset=self.reader.tell(),
                )
            return
        while self.reader.remaining():
            offset = self.reader.tell()
            chunk = self.reader.read_exact(
                min(64 * 1024, self.reader.remaining()),
                path="$",
            )
            if chunk.strip():
                raise RdsFormatError(
                    "non-whitespace data follows the root object",
                    path="$",
                    offset=offset,
                )

    def _start_object(self, path: str, depth: int) -> int:
        offset = self.reader.tell()
        if depth > self._maximumDepth:
            raise RdsLimitError(
                "max_depth",
                depth,
                self._maximumDepth,
                path=path,
                offset=offset,
            )
        self._objects += 1
        if self._objects > self.limits.max_objects:
            raise RdsLimitError(
                "max_objects",
                self._objects,
                self.limits.max_objects,
                path=path,
                offset=offset,
            )
        return offset

    def _parse_info(self, flags: int, *, path: str, offset: int) -> _ObjectInfo:
        type_code = flags & 0xFF
        try:
            r_type = RType(type_code)
        except ValueError as error:
            raise RdsFormatError(
                f"unknown R node type {type_code}",
                path=path,
                offset=offset,
            ) from error
        if r_type in _SINGLETON_TYPES or r_type is RType.REFERENCE:
            return _ObjectInfo(r_type, False, False, False, 0)
        return _ObjectInfo(
            type=r_type,
            object=bool(flags & (1 << 8)),
            has_attributes=bool(flags & (1 << 9)),
            has_tag=bool(flags & (1 << 10)),
            gp=(flags >> 12) & 0xFFFF,
        )

    def _parse_node(
        self,
        path: str,
        depth: int,
        header: tuple[int, _ObjectInfo, int] | None = None,
    ) -> RNode:
        """Parse one item; ``header`` passes flags already read by the caller."""
        if header is None:
            offset = self._start_object(path, depth)
            flags = self.reader.read_int(path=path)
            info = self._parse_info(flags, path=path, offset=offset)
        else:
            flags, info, offset = header
        r_type = info.type

        if r_type in _SINGLETON_TYPES:
            return self._singleton(r_type, path=path, offset=offset)
        if r_type is RType.REFERENCE:
            return self._read_reference(flags, path=path, offset=offset)
        if r_type in {RType.CLASS_REFERENCE, RType.GENERIC_REFERENCE}:
            raise RdsFormatError(
                f"{r_type.name.lower()} is not a readable R serialization node",
                path=path,
                offset=offset,
            )
        if r_type is RType.SYMBOL:
            name = self._parse_node(f"{path}.name", depth + 1)
            node = RNode(
                r_type,
                value=name,
                object=info.object,
                gp=info.gp,
                path=path,
                offset=offset,
            )
            self._references.append(node)
            return node
        if r_type in {RType.PACKAGE, RType.NAMESPACE, RType.PERSISTENT}:
            names = self._parse_persistent_strings(f"{path}.names", depth + 1)
            node = RNode(
                r_type,
                value=PersistentValue(names),
                object=info.object,
                gp=info.gp,
                path=path,
                offset=offset,
            )
            self._references.append(node)
            return node
        if r_type is RType.ENVIRONMENT:
            return self._parse_environment(info, path=path, depth=depth, offset=offset)
        if r_type in _PAIR_TYPES:
            return self._parse_pair(info, path=path, depth=depth, offset=offset)
        if r_type is RType.ALTREP:
            return self._parse_altrep(info, path=path, depth=depth, offset=offset)

        node = RNode(
            r_type,
            object=info.object,
            gp=info.gp,
            path=path,
            offset=offset,
        )
        if r_type in {RType.SPECIAL, RType.BUILTIN}:
            length = self.reader.read_int(path=f"{path}.length")
            self._check_string_length(
                length,
                path=path,
                offset=self.reader.tell(),
                allow_missing=False,
            )
            raw = self.reader.read_string(length, path=path)
            node.value = raw.decode("utf-8", errors="surrogateescape")
        elif r_type is RType.CHAR:
            node.value = self._parse_char_value(info.gp, path=path)
        elif r_type in {
            RType.LOGICAL,
            RType.INTEGER,
            RType.REAL,
            RType.COMPLEX,
            RType.RAW,
        }:
            node.value = self._parse_atomic_vector(r_type, path=path)
        elif r_type is RType.STRING:
            node.value = self._parse_string_vector(path=path, depth=depth)
        elif r_type in {RType.VECTOR, RType.EXPRESSION}:
            length = self._read_length(path=f"{path}.length")
            self._check_materialized_children(length, path=path)
            values = [
                self._parse_node(f"{path}[{index}]", depth + 1)
                for index in range(length)
            ]
            node.value = tuple(values)
        elif r_type is RType.BYTECODE:
            node.value = self._parse_bytecode(path=path, depth=depth)
        elif r_type is RType.EXTERNAL_POINTER:
            self._references.append(node)
            protected = self._parse_node(f"{path}.protected", depth + 1)
            tag = self._parse_node(f"{path}.externalTag", depth + 1)
            node.value = ExternalPointerValue(protected, tag)
        elif r_type is RType.WEAK_REFERENCE:
            self._references.append(node)
            node.value = WeakReferenceValue(
                ready_to_finalize=bool(info.gp & 1),
                finalize_on_exit=bool(info.gp & 2),
            )
        elif r_type is RType.S4:
            node.value = None
        else:
            raise RdsFormatError(
                f"unsupported R node type {r_type.name}",
                path=path,
                offset=offset,
            )

        if info.has_tag:
            # R writes tags only on pairlist-like cells.
            raise RdsFormatError(
                f"{r_type.name} node cannot have a tag",
                path=path,
                offset=offset,
            )
        if info.has_attributes:
            node.attributes = self._null_to_none(
                self._parse_node(f"{path}.attributes", depth + 1)
            )
        return node

    def _singleton(self, r_type: RType, *, path: str, offset: int) -> RNode:
        node = self._singletons.get(r_type)
        if node is None:
            node = RNode(r_type, path=path, offset=offset)
            self._singletons[r_type] = node
        return node

    def _read_reference(self, flags: int, *, path: str, offset: int) -> RNode:
        index = flags >> 8
        if index == 0:
            index = self.reader.read_int(path=f"{path}.reference")
        if index <= 0 or index > len(self._references):
            raise RdsFormatError(
                f"reference index {index} is out of range",
                path=path,
                offset=offset,
            )
        return self._references[index - 1]

    def _parse_environment(
        self,
        info: _ObjectInfo,
        *,
        path: str,
        depth: int,
        offset: int,
    ) -> RNode:
        node = RNode(
            RType.ENVIRONMENT,
            object=True,
            gp=info.gp,
            path=path,
            offset=offset,
        )
        self._references.append(node)
        locked = self.reader.read_int(path=f"{path}.locked")
        if locked not in {0, 1}:
            raise RdsFormatError(
                f"environment lock flag must be 0 or 1, found {locked}",
                path=f"{path}.locked",
                offset=self.reader.tell() - 4,
            )
        enclosure = self._parse_node(f"{path}.enclosure", depth + 1)
        frame = self._parse_node(f"{path}.frame", depth + 1)
        hash_table = self._parse_node(f"{path}.hashTable", depth + 1)
        attributes = self._parse_node(f"{path}.attributes", depth + 1)
        node.value = EnvironmentValue(locked == 1, enclosure, frame, hash_table)
        node.attributes = self._null_to_none(attributes)
        return node

    def _parse_pair(
        self,
        info: _ObjectInfo,
        *,
        path: str,
        depth: int,
        offset: int,
    ) -> RNode:
        """Parse a pairlist chain, following CDR cells iteratively.

        The cells after the first share one depth, so a long pairlist is
        bounded by ``max_objects`` rather than by the nesting limit.
        """
        head: RNode | None = None
        previous: tuple[RNode, RNode] | None = None
        cell_info, cell_path, cell_depth, cell_offset = info, path, depth, offset
        index = 0
        while True:
            attributes = (
                self._parse_node(f"{cell_path}.attributes", cell_depth + 1)
                if cell_info.has_attributes
                else None
            )
            tag = (
                self._parse_node(f"{cell_path}.tag", cell_depth + 1)
                if cell_info.has_tag
                else None
            )
            car = self._parse_node(f"{cell_path}.car", cell_depth + 1)
            cell = RNode(
                cell_info.type,
                attributes=self._null_to_none(attributes),
                tag=self._null_to_none(tag),
                object=cell_info.object,
                gp=cell_info.gp,
                path=cell_path,
                offset=cell_offset,
            )
            if previous is None:
                head = cell
            else:
                previous[0].value = self._pair_value(previous[0], previous[1], cell)
            previous = (cell, car)

            index += 1
            cdr_path = f"{path}.cdr" if index == 1 else f"{path}.cdr[{index}]"
            cdr_offset = self._start_object(cdr_path, depth + 1)
            flags = self.reader.read_int(path=cdr_path)
            cdr_info = self._parse_info(flags, path=cdr_path, offset=cdr_offset)
            if cdr_info.type not in _PAIR_TYPES:
                cdr = self._parse_node(
                    cdr_path,
                    depth + 1,
                    (flags, cdr_info, cdr_offset),
                )
                cell.value = self._pair_value(cell, car, cdr)
                if head is None:
                    raise AssertionError("pairlist chain lost its head")
                return head
            cell_info, cell_path, cell_depth, cell_offset = (
                cdr_info,
                cdr_path,
                depth + 1,
                cdr_offset,
            )

    @staticmethod
    def _pair_value(
        cell: RNode,
        car: RNode,
        cdr: RNode,
    ) -> PairValue | ClosureValue | PromiseValue:
        if cell.type is RType.CLOSURE:
            return ClosureValue(cell.tag, car, cdr)
        if cell.type is RType.PROMISE:
            return PromiseValue(cell.tag, car, cdr)
        return PairValue(car, cdr)

    def _parse_altrep(
        self,
        info: _ObjectInfo,
        *,
        path: str,
        depth: int,
        offset: int,
    ) -> RNode:
        altrep_info = self._parse_node(f"{path}.info", depth + 1)
        state = self._parse_node(f"{path}.state", depth + 1)
        attributes = self._parse_node(f"{path}.attributes", depth + 1)
        class_name, package_name = self._altrep_names(altrep_info)
        return RNode(
            RType.ALTREP,
            value=AltRepValue(
                info=altrep_info,
                state=state,
                class_name=class_name,
                package_name=package_name,
                known=class_name in _KNOWN_ALTREP,
            ),
            attributes=self._null_to_none(attributes),
            object=info.object,
            gp=info.gp,
            path=path,
            offset=offset,
        )

    def _altrep_names(self, info: RNode) -> tuple[str | None, str | None]:
        if info.type is not RType.PAIRLIST or not isinstance(info.value, PairValue):
            return None, None
        class_name = symbol_name(info.value.car)
        tail = info.value.cdr
        if tail.type is not RType.PAIRLIST or not isinstance(tail.value, PairValue):
            return class_name, None
        return class_name, symbol_name(tail.value.car)

    def _parse_char_value(self, gp: int, *, path: str) -> str | bytes | None:
        length = self.reader.read_int(path=f"{path}.length")
        self._check_string_length(
            length,
            path=path,
            offset=self.reader.tell(),
            allow_missing=True,
        )
        if length == -1:
            return None
        raw = self.reader.read_string(length, path=path)
        return self._decode_char(raw, gp)

    def _decode_char(self, raw: bytes, gp: int) -> str | bytes:
        return decode_r_string(raw, gp, self.metadata.native_encoding)

    def _parse_atomic_vector(self, r_type: RType, *, path: str) -> LazyAtomicVector:
        length = self._read_length(path=f"{path}.length")
        item_size = {
            RType.LOGICAL: 4,
            RType.INTEGER: 4,
            RType.REAL: 8,
            RType.COMPLEX: 16,
            RType.RAW: 1,
        }[r_type]
        if self.reader.encoding is not RdsEncoding.ASCII:
            size = length * item_size
            offset = self.reader.skip(size, path=path)
            byte_order = (
                ">"
                if self.reader.encoding is RdsEncoding.XDR
                else self.metadata.byte_order
            )
            if byte_order is None:
                raise AssertionError("native byte order was not initialized")
            return LazyAtomicVector(
                self.storage,
                offset=offset,
                length=length,
                r_type=r_type,
                byte_order=byte_order,
                path=path,
            )

        backing = self.temp_manager.shared("ascii-vector")
        writer = BufferedTempWriter(self.temp_manager, backing)
        offset = writer.position
        for index in range(length):
            item_path = f"{path}[{index}]"
            if r_type in {RType.LOGICAL, RType.INTEGER}:
                data = struct.pack("<i", self.reader.read_int(path=item_path))
            elif r_type is RType.REAL:
                data = struct.pack("<d", self.reader.read_real(path=item_path))
            elif r_type is RType.COMPLEX:
                real = self.reader.read_real(path=f"{item_path}.real")
                imaginary = self.reader.read_real(path=f"{item_path}.imaginary")
                data = struct.pack("<dd", real, imaginary)
            else:
                word = self.reader.read_word(path=item_path)
                try:
                    value = int(word, 16)
                except ValueError as error:
                    raise RdsFormatError(
                        f"invalid ASCII raw byte {word!r}",
                        path=item_path,
                        offset=self.reader.tell() - len(word),
                    ) from error
                if value < 0 or value > 255:
                    raise RdsFormatError(
                        f"ASCII raw byte {value} is out of range",
                        path=item_path,
                        offset=self.reader.tell() - len(word),
                    )
                data = bytes((value,))
            writer.write(data, path=item_path)
        writer.flush(path=path)
        backing.flush()
        return LazyAtomicVector(
            backing,
            offset=offset,
            length=length,
            r_type=r_type,
            byte_order="<",
            path=path,
        )

    def _parse_string_vector(self, *, path: str, depth: int) -> LazyStringVector:
        length = self._read_length(path=f"{path}.length")
        self._check_materialized_children(length, path=path)
        index_storage = self.temp_manager.shared("string-index")
        descriptor_writer = BufferedTempWriter(self.temp_manager, index_storage)
        descriptor_offset = descriptor_writer.position
        decoded_storage: RandomAccessStorage | None = None
        decoded_writer: BufferedTempWriter | None = None

        for index in range(length):
            item_path = f"{path}[{index}]"
            item_offset = self._start_object(item_path, depth + 1)
            flags = self.reader.read_int(path=item_path)
            info = self._parse_info(flags, path=item_path, offset=item_offset)
            # R never references a CHARSXP and never writes its tag or
            # attributes, so each element is one inline CHAR node.
            if info.type is not RType.CHAR:
                raise RdsFormatError(
                    f"character vector element has type {info.type.name}",
                    path=item_path,
                    offset=item_offset,
                )
            if info.has_tag:
                raise RdsFormatError(
                    "CHAR node cannot have a tag",
                    path=item_path,
                    offset=item_offset,
                )
            if info.has_attributes:
                raise RdsFormatError(
                    "CHAR node cannot have attributes",
                    path=item_path,
                    offset=item_offset,
                )
            string_length = self.reader.read_int(path=f"{item_path}.length")
            self._check_string_length(
                string_length,
                path=item_path,
                offset=self.reader.tell(),
                allow_missing=True,
            )
            if string_length == -1:
                descriptor = pack_string_descriptor(0, 0, -1, info.gp)
            elif self.reader.encoding is RdsEncoding.ASCII:
                raw = self.reader.read_ascii_string(
                    string_length,
                    path=item_path,
                )
                if decoded_writer is None:
                    decoded_storage = self.temp_manager.shared("string-data")
                    decoded_writer = BufferedTempWriter(
                        self.temp_manager,
                        decoded_storage,
                    )
                data_offset = decoded_writer.write(raw, path=item_path)
                descriptor = pack_string_descriptor(
                    STRING_SOURCE_DECODED,
                    data_offset,
                    string_length,
                    info.gp,
                )
            else:
                data_offset = self.reader.skip(string_length, path=item_path)
                descriptor = pack_string_descriptor(
                    STRING_SOURCE_PAYLOAD,
                    data_offset,
                    string_length,
                    info.gp,
                )
            descriptor_writer.write(descriptor, path=item_path)

        descriptor_writer.flush(path=path)
        if decoded_writer is not None:
            decoded_writer.flush(path=path)
        index_storage.flush()
        if decoded_storage is not None:
            decoded_storage.flush()
        return LazyStringVector(
            index_storage,
            descriptor_offset=descriptor_offset,
            length=length,
            payload_storage=self.storage,
            decoded_storage=decoded_storage,
            default_encoding=self.metadata.native_encoding,
            path=path,
        )

    def _parse_persistent_strings(
        self,
        path: str,
        depth: int,
    ) -> LazyStringVector:
        placeholder = self.reader.read_int(path=f"{path}.placeholder")
        if placeholder != 0:
            raise RdsFormatError(
                f"persistent string placeholder must be 0, found {placeholder}",
                path=f"{path}.placeholder",
                offset=self.reader.tell() - 4,
            )
        return self._parse_string_vector(path=path, depth=depth)

    def _parse_bytecode(self, *, path: str, depth: int) -> BytecodeValue:
        repeated = self.reader.read_int(path=f"{path}.repeatedCount")
        self._check_count(repeated, path=f"{path}.repeatedCount")
        return self._parse_bytecode_body(
            _BytecodeRepetitions(repeated),
            path=path,
            depth=depth + 1,
        )

    def _parse_bytecode_body(
        self,
        repetitions: _BytecodeRepetitions,
        *,
        path: str,
        depth: int,
    ) -> BytecodeValue:
        code = self._parse_node(f"{path}.code", depth + 1)
        count = self.reader.read_int(path=f"{path}.constantCount")
        self._check_count(count, path=f"{path}.constantCount")
        self._check_materialized_children(count, path=f"{path}.constants")
        constants: list[RNode] = []
        for index in range(count):
            constant_path = f"{path}.constants[{index}]"
            marker_offset = self.reader.tell()
            marker = self.reader.read_int(path=f"{constant_path}.marker")
            if marker == RType.BYTECODE:
                self._start_object(constant_path, depth + 1)
                value = RNode(
                    RType.BYTECODE,
                    value=self._parse_bytecode_body(
                        repetitions,
                        path=constant_path,
                        depth=depth + 1,
                    ),
                    path=constant_path,
                    offset=marker_offset,
                )
            else:
                value = self._parse_bytecode_language(
                    marker,
                    repetitions,
                    path=constant_path,
                    depth=depth + 1,
                )
            constants.append(value)
        return BytecodeValue(code=code, constants=tuple(constants))

    def _parse_bytecode_language(
        self,
        marker: int,
        repetitions: _BytecodeRepetitions,
        *,
        path: str,
        depth: int,
    ) -> RNode:
        """Parse one bytecode constant, following CDR cells iteratively."""
        head: RNode | None = None
        pending: tuple[RNode, RNode] | None = None
        cell_path = path
        cell_depth = depth
        index = 0
        while True:
            if marker not in _BYTECODE_PAIR_MARKERS:
                # R pads other constants with a zero marker before the item.
                result = self._parse_node(cell_path, cell_depth)
            else:
                marker_offset = self._start_object(cell_path, cell_depth)
                if marker == RType.BYTECODE_REFERENCE:
                    result = self._bytecode_reference(
                        repetitions,
                        path=cell_path,
                        offset=marker_offset,
                    )
                else:
                    cell, car = self._parse_bytecode_cell(
                        marker,
                        repetitions,
                        path=cell_path,
                        depth=cell_depth,
                        offset=marker_offset,
                    )
                    if pending is None:
                        head = cell
                    else:
                        pending[0].value = PairValue(pending[1], cell)
                    pending = (cell, car)
                    index += 1
                    cell_path = f"{path}.cdr" if index == 1 else f"{path}.cdr[{index}]"
                    cell_depth = depth + 1
                    marker = self.reader.read_int(path=f"{cell_path}.marker")
                    continue
            if pending is None:
                return result
            pending[0].value = PairValue(pending[1], result)
            if head is None:
                raise AssertionError("bytecode pair chain lost its head")
            return head

    def _bytecode_reference(
        self,
        repetitions: _BytecodeRepetitions,
        *,
        path: str,
        offset: int,
    ) -> RNode:
        position = self.reader.read_int(path=f"{path}.position")
        node = repetitions.nodes.get(position)
        if position < 0 or position >= repetitions.count or node is None:
            raise RdsFormatError(
                f"bytecode reference {position} is out of range",
                path=path,
                offset=offset,
            )
        return node

    def _parse_bytecode_cell(
        self,
        marker: int,
        repetitions: _BytecodeRepetitions,
        *,
        path: str,
        depth: int,
        offset: int,
    ) -> tuple[RNode, RNode]:
        repetition_position: int | None = None
        if marker == RType.BYTECODE_DEFINITION:
            repetition_position = self.reader.read_int(path=f"{path}.position")
            if not 0 <= repetition_position < repetitions.count:
                raise RdsFormatError(
                    f"bytecode definition {repetition_position} is out of range",
                    path=path,
                    offset=offset,
                )
            marker = self.reader.read_int(path=f"{path}.type")

        has_attributes = marker in {
            RType.BYTECODE_ATTR_LANGUAGE,
            RType.BYTECODE_ATTR_PAIRLIST,
        }
        if marker in {RType.LANGUAGE, RType.BYTECODE_ATTR_LANGUAGE}:
            r_type = RType.LANGUAGE
        elif marker in {RType.PAIRLIST, RType.BYTECODE_ATTR_PAIRLIST}:
            r_type = RType.PAIRLIST
        else:
            raise RdsFormatError(
                f"invalid bytecode pair definition type {marker}",
                path=path,
                offset=offset,
            )

        node = RNode(r_type, path=path, offset=offset)
        if repetition_position is not None:
            repetitions.nodes[repetition_position] = node
        if has_attributes:
            node.attributes = self._null_to_none(
                self._parse_node(f"{path}.attributes", depth + 1)
            )
        node.tag = self._null_to_none(self._parse_node(f"{path}.tag", depth + 1))
        car_marker = self.reader.read_int(path=f"{path}.car.marker")
        car = self._parse_bytecode_language(
            car_marker,
            repetitions,
            path=f"{path}.car",
            depth=depth + 1,
        )
        return node, car

    def _read_length(self, *, path: str) -> int:
        offset = self.reader.tell()
        length = self.reader.read_int(path=path)
        if length == -1:
            upper = self.reader.read_int(path=f"{path}.upper") & 0xFFFFFFFF
            lower = self.reader.read_int(path=f"{path}.lower") & 0xFFFFFFFF
            if upper > 65536:
                raise RdsFormatError(
                    f"long-vector upper length {upper} exceeds R's wire limit",
                    path=path,
                    offset=offset,
                )
            length = (upper << 32) | lower
        elif length < 0:
            raise RdsFormatError(
                f"negative vector length {length}",
                path=path,
                offset=offset,
            )
        self._check_count(length, path=path, offset=offset)
        return length

    def _check_count(
        self,
        count: int,
        *,
        path: str,
        offset: int | None = None,
    ) -> None:
        if count < 0:
            raise RdsFormatError(
                f"negative count {count}",
                path=path,
                offset=self.reader.tell() if offset is None else offset,
            )
        if count > self.limits.max_vector_length:
            raise RdsLimitError(
                "max_vector_length",
                count,
                self.limits.max_vector_length,
                path=path,
                offset=self.reader.tell() if offset is None else offset,
            )

    def _check_materialized_children(self, count: int, *, path: str) -> None:
        projected = self._objects + count
        if projected > self.limits.max_objects:
            raise RdsLimitError(
                "max_objects",
                projected,
                self.limits.max_objects,
                path=path,
                offset=self.reader.tell(),
            )

    def _check_string_length(
        self,
        length: int,
        *,
        path: str,
        offset: int,
        allow_missing: bool,
    ) -> None:
        minimum = -1 if allow_missing else 0
        if length < minimum:
            raise RdsFormatError(
                f"invalid string length {length}",
                path=path,
                offset=offset,
            )
        if length > self.limits.max_string_bytes:
            raise RdsLimitError(
                "max_string_bytes",
                length,
                self.limits.max_string_bytes,
                path=path,
                offset=offset,
            )

    @staticmethod
    def _null_to_none(node: RNode | None) -> RNode | None:
        if node is not None and node.is_null:
            return None
        return node


def open_rds(
    source: RdsInput,
    *,
    limits: RdsLimits | None = None,
    temp_dir: str | os.PathLike[str] | None = None,
) -> RdsDocument:
    """Open and index one RDS stream without evaluating serialized code."""
    active_limits = limits or RdsLimits()
    prepared = prepare_payload(source, active_limits, temp_dir=temp_dir)
    try:
        parser = _Parser(
            prepared.storage,
            prepared.temp_manager,
            active_limits,
        )
        root = parser.parse()
        return RdsDocument(
            root,
            metadata=parser.metadata,
            source=prepared.metadata,
            temp_manager=prepared.temp_manager,
            direct_storage=prepared.direct_storage,
        )
    except Exception:
        if prepared.direct_storage is not None:
            prepared.direct_storage.close()
        prepared.temp_manager.close()
        raise
