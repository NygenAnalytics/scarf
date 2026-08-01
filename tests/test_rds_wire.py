import bz2
import gzip
import hashlib
import io
import lzma
import struct
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rdata
import zstandard

from scarf.readers._rds import (
    AltRepValue,
    BytecodeValue,
    ClosureValue,
    EnvironmentValue,
    ExternalPointerValue,
    LazyAtomicVector,
    LazyStringVector,
    PairValue,
    PersistentValue,
    PromiseValue,
    RdsClosedError,
    RdsCompression,
    RdsFormatError,
    RdsLimitError,
    RdsLimits,
    RType,
    WeakReferenceValue,
    open_rds,
)


class Wire:
    def __init__(self, encoding: str = "xdr", version: int = 3) -> None:
        self.encoding = encoding
        self.version = version

    def integer(self, value: int) -> bytes:
        if self.encoding == "ascii":
            return f"{value}\n".encode()
        order = ">" if self.encoding == "xdr" else "<"
        return struct.pack(f"{order}i", value)

    def real(self, value: float) -> bytes:
        if self.encoding == "ascii":
            return f"{value:.16g}\n".encode()
        order = ">" if self.encoding == "xdr" else "<"
        return struct.pack(f"{order}d", value)

    def string(self, value: bytes) -> bytes:
        if self.encoding != "ascii":
            return value
        escaped = bytearray()
        replacements = {
            ord("\n"): b"\\n",
            ord("\t"): b"\\t",
            ord("\\"): b"\\\\",
            ord("?"): b"\\?",
            ord("'"): b"\\'",
            ord('"'): b'\\"',
        }
        for byte in value:
            replacement = replacements.get(byte)
            if replacement is not None:
                escaped.extend(replacement)
            elif byte <= 32 or byte > 126:
                escaped.extend(f"\\{byte:03o}".encode())
            else:
                escaped.append(byte)
        escaped.append(ord("\n"))
        return bytes(escaped)

    def header(self) -> bytes:
        marker = {"xdr": b"X\n", "native": b"B\n", "ascii": b"A\n"}[self.encoding]
        result = (
            marker
            + self.integer(self.version)
            + self.integer(4 * 65536 + 4 * 256)
            + self.integer(3 * 65536 + 5 * 256)
        )
        if self.version == 3:
            result += self.integer(5) + self.string(b"UTF-8")
        return result

    @staticmethod
    def flags(
        r_type: RType | int,
        *,
        attributes: bool = False,
        tag: bool = False,
        object_: bool = False,
        gp: int = 0,
    ) -> int:
        return (
            int(r_type)
            | (int(object_) << 8)
            | (int(attributes) << 9)
            | (int(tag) << 10)
            | (gp << 12)
        )

    def nil(self) -> bytes:
        return self.integer(RType.NIL_VALUE)

    def char(self, value: bytes | None, *, gp: int = 1 << 6) -> bytes:
        result = self.integer(self.flags(RType.CHAR, gp=gp))
        if value is None:
            return result + self.integer(-1)
        return result + self.integer(len(value)) + self.string(value)

    def symbol(self, name: str) -> bytes:
        return self.integer(RType.SYMBOL) + self.char(name.encode())

    def integer_vector(
        self,
        values: list[int],
        *,
        attributes: bytes | None = None,
        long_length: bool = False,
    ) -> bytes:
        result = self.integer(
            self.flags(RType.INTEGER, attributes=attributes is not None)
        )
        if long_length:
            result += self.integer(-1) + self.integer(0) + self.integer(len(values))
        else:
            result += self.integer(len(values))
        result += b"".join(self.integer(value) for value in values)
        return result + (attributes or b"")

    def real_vector(self, values: list[float]) -> bytes:
        return (
            self.integer(RType.REAL)
            + self.integer(len(values))
            + b"".join(self.real(value) for value in values)
        )

    def raw_vector(self, values: bytes) -> bytes:
        data = (
            b"".join(f"{value:02x}\n".encode() for value in values)
            if self.encoding == "ascii"
            else values
        )
        return self.integer(RType.RAW) + self.integer(len(values)) + data

    def string_vector(self, values: list[bytes | None]) -> bytes:
        return (
            self.integer(RType.STRING)
            + self.integer(len(values))
            + b"".join(self.char(value) for value in values)
        )

    def pair(
        self,
        car: bytes,
        cdr: bytes,
        *,
        tag: bytes | None = None,
        attributes: bytes | None = None,
        r_type: RType = RType.PAIRLIST,
    ) -> bytes:
        return (
            self.integer(
                self.flags(
                    r_type,
                    attributes=attributes is not None,
                    tag=tag is not None,
                )
            )
            + (attributes or b"")
            + (tag or b"")
            + car
            + cdr
        )

    def vector(self, values: list[bytes], *, attributes: bytes | None = None) -> bytes:
        return (
            self.integer(self.flags(RType.VECTOR, attributes=attributes is not None))
            + self.integer(len(values))
            + b"".join(values)
            + (attributes or b"")
        )

    def document(self, root: bytes) -> bytes:
        return self.header() + root


def test_xdr_numeric_vector_is_lazy_and_matches_rdata(tmp_path: Path) -> None:
    wire = Wire()
    payload = wire.document(wire.integer_vector([1, -2, 3, 4]))
    path = tmp_path / "numbers.rds"
    path.write_bytes(payload)

    with open_rds(path) as document:
        vector = document.root.value
        assert isinstance(vector, LazyAtomicVector)
        np.testing.assert_array_equal(vector.read_block(1, 3), [-2, 3])
        assert document.metadata.format_version == 3
        assert document.metadata.native_encoding == "UTF-8"
        assert document.source.compression is RdsCompression.NONE
        assert document.source.spooled is False
        assert document.source.source_sha256 == hashlib.sha256(payload).hexdigest()
        assert document.source.payload_sha256 == hashlib.sha256(payload).hexdigest()
        assert document.temp_paths == ()

        oracle = rdata.parser.parse_data(payload, extension=".rds")
        np.testing.assert_array_equal(vector.materialize(), oracle.object.value)


@pytest.mark.parametrize("encoding", ["xdr", "native", "ascii"])
@pytest.mark.parametrize("version", [2, 3])
def test_wire_encodings_and_versions(encoding: str, version: int) -> None:
    wire = Wire(encoding, version)
    payload = wire.document(wire.real_vector([1.25, -2.5]))
    with open_rds(io.BytesIO(payload)) as document:
        assert document.metadata.format_version == version
        assert document.metadata.encoding.value == encoding
        np.testing.assert_array_equal(document.root.value[:], [1.25, -2.5])


def test_ascii_raw_and_long_vector_lengths() -> None:
    ascii_wire = Wire("ascii")
    with open_rds(
        io.BytesIO(ascii_wire.document(ascii_wire.raw_vector(b"\x00\x7f\xff")))
    ) as document:
        np.testing.assert_array_equal(document.root.value[:], [0, 127, 255])

    xdr_wire = Wire()
    payload = xdr_wire.document(xdr_wire.integer_vector([7, 8, 9], long_length=True))
    with open_rds(io.BytesIO(payload)) as document:
        assert document.root.value.length == 3
        np.testing.assert_array_equal(document.root.value[:], [7, 8, 9])


@pytest.mark.parametrize(
    ("compress", "expected"),
    [
        (gzip.compress, RdsCompression.GZIP),
        (bz2.compress, RdsCompression.BZIP2),
        (lzma.compress, RdsCompression.XZ),
        (zstandard.ZstdCompressor().compress, RdsCompression.ZSTD),
    ],
)
def test_streaming_compression_and_cleanup(
    compress: Callable[[bytes], bytes],
    expected: RdsCompression,
    tmp_path: Path,
) -> None:
    wire = Wire()
    payload = wire.document(wire.string_vector([b"alpha", None, b"beta"]))
    document = open_rds(
        io.BytesIO(compressed := compress(payload)),
        temp_dir=tmp_path,
    )
    strings = document.root.value
    assert isinstance(strings, LazyStringVector)
    assert strings[:] == ["alpha", None, "beta"]
    assert document.source.compression is expected
    assert document.source.spooled is True
    assert document.source.source_sha256 == hashlib.sha256(compressed).hexdigest()
    assert document.source.payload_sha256 == hashlib.sha256(payload).hexdigest()
    temp_paths = tuple(Path(path) for path in document.temp_paths)
    assert temp_paths
    assert all(path.exists() for path in temp_paths)

    document.close()
    document.close()
    assert all(not path.exists() for path in temp_paths)
    with pytest.raises(RdsClosedError, match="RDS document is closed"):
        strings[0]


def test_s4_slots_and_named_list_helpers() -> None:
    wire = Wire()
    slots = wire.pair(
        wire.integer_vector([42]),
        wire.nil(),
        tag=wire.symbol("answer"),
    )
    s4 = wire.integer(wire.flags(RType.S4, attributes=True, object_=True)) + slots
    with open_rds(io.BytesIO(wire.document(s4))) as document:
        assert document.root.type is RType.S4
        answer = document.root.slot("answer")
        assert answer is not None
        assert answer.value[0] == 42

    names = wire.pair(
        wire.string_vector([b"left", b"right"]),
        wire.nil(),
        tag=wire.symbol("names"),
    )
    named = wire.vector(
        [wire.integer_vector([1]), wire.integer_vector([2])],
        attributes=names,
    )
    with open_rds(io.BytesIO(wire.document(named))) as document:
        right = document.root.named("right")
        assert right is not None
        assert right.value[0] == 2
        assert [name for name, _value in document.root.named_items()] == [
            "left",
            "right",
        ]


def test_lazy_indexes_share_a_bounded_number_of_temp_files(tmp_path: Path) -> None:
    wire = Wire()
    payload = wire.document(
        wire.vector(
            [
                wire.string_vector([b"one", b"two"]),
                wire.string_vector([b"three"]),
            ]
        )
    )

    with open_rds(io.BytesIO(payload), temp_dir=tmp_path) as document:
        left, right = document.root.value
        assert left.value[:] == ["one", "two"]
        assert right.value[:] == ["three"]
        assert len(document.temp_paths) == 1
        assert "string-index" in Path(document.temp_paths[0]).name


def test_references_preserve_identity_and_environment_cycles() -> None:
    wire = Wire()
    repeated_symbol = wire.vector(
        [
            wire.symbol("shared"),
            wire.integer((1 << 8) | RType.REFERENCE),
        ]
    )
    with open_rds(io.BytesIO(wire.document(repeated_symbol))) as document:
        first, second = document.root.value
        assert first is second

    frame = wire.pair(
        wire.integer((1 << 8) | RType.REFERENCE),
        wire.nil(),
        tag=wire.symbol("self"),
    )
    environment = (
        wire.integer(RType.ENVIRONMENT)
        + wire.integer(0)
        + wire.integer(RType.BASE_ENVIRONMENT)
        + frame
        + wire.nil()
        + wire.nil()
    )
    with open_rds(io.BytesIO(wire.document(environment))) as document:
        value = document.root.value
        assert isinstance(value, EnvironmentValue)
        assert isinstance(value.frame.value, PairValue)
        assert value.frame.value.car is document.root


def test_structural_executable_and_opaque_nodes_are_not_executed() -> None:
    wire = Wire()
    closure = (
        wire.integer(wire.flags(RType.CLOSURE, tag=True))
        + wire.integer(RType.BASE_ENVIRONMENT)
        + wire.nil()
        + wire.symbol("body")
    )
    with open_rds(io.BytesIO(wire.document(closure))) as document:
        assert isinstance(document.root.value, ClosureValue)
        assert document.root.value.body.type is RType.SYMBOL

    promise = wire.pair(
        wire.integer_vector([3]),
        wire.symbol("expression"),
        tag=wire.integer(RType.BASE_ENVIRONMENT),
        r_type=RType.PROMISE,
    )
    with open_rds(io.BytesIO(wire.document(promise))) as document:
        assert isinstance(document.root.value, PromiseValue)
        assert document.root.value.expression.type is RType.SYMBOL

    external_pointer = (
        wire.integer(RType.EXTERNAL_POINTER)
        + wire.integer((1 << 8) | RType.REFERENCE)
        + wire.nil()
    )
    with open_rds(io.BytesIO(wire.document(external_pointer))) as document:
        assert isinstance(document.root.value, ExternalPointerValue)
        assert document.root.value.protected is document.root

    weak_reference = wire.vector(
        [
            wire.integer(RType.WEAK_REFERENCE),
            wire.integer((1 << 8) | RType.REFERENCE),
        ]
    )
    with open_rds(io.BytesIO(wire.document(weak_reference))) as document:
        first, second = document.root.value
        assert isinstance(first.value, WeakReferenceValue)
        assert first is second

    package = (
        wire.integer(RType.PACKAGE)
        + wire.integer(0)
        + wire.integer(1)
        + wire.char(b"package:base")
    )
    with open_rds(io.BytesIO(wire.document(package))) as document:
        assert isinstance(document.root.value, PersistentValue)
        assert document.root.value.names[0] == "package:base"


def test_bytecode_and_altrep_are_inert_structures() -> None:
    wire = Wire()
    bytecode = (
        wire.integer(RType.BYTECODE)
        + wire.integer(0)
        + wire.integer_vector([12])
        + wire.integer(1)
        + wire.integer(RType.INTEGER)
        + wire.integer_vector([99])
    )
    with open_rds(io.BytesIO(wire.document(bytecode))) as document:
        value = document.root.value
        assert isinstance(value, BytecodeValue)
        assert value.constants[0].value[0] == 99

    info = wire.pair(
        wire.symbol("compact_intseq"),
        wire.pair(wire.symbol("base"), wire.nil()),
    )
    altrep = (
        wire.integer(RType.ALTREP)
        + info
        + wire.real_vector([3.0, 1.0, 1.0])
        + wire.nil()
    )
    with open_rds(io.BytesIO(wire.document(altrep))) as document:
        value = document.root.value
        assert isinstance(value, AltRepValue)
        assert value.class_name == "compact_intseq"
        assert value.package_name == "base"
        assert value.known is True


def test_caps_report_deterministic_object_paths(tmp_path: Path) -> None:
    wire = Wire()
    vector_payload = wire.document(wire.integer_vector([1, 2, 3]))
    with pytest.raises(
        RdsLimitError,
        match=r"max_objects exceeded: 1 > 0 at \$",
    ):
        open_rds(
            io.BytesIO(vector_payload),
            limits=RdsLimits(max_objects=0),
        )

    with pytest.raises(
        RdsLimitError,
        match=r"max_vector_length exceeded: 3 > 2 at \$\.length",
    ):
        open_rds(
            io.BytesIO(vector_payload),
            limits=RdsLimits(max_vector_length=2),
        )

    nested = wire.vector([wire.vector([wire.integer_vector([1])])])
    with pytest.raises(
        RdsLimitError,
        match=r"max_depth exceeded: 2 > 1 at \$\[0\]\[0\]",
    ):
        open_rds(
            io.BytesIO(wire.document(nested)),
            limits=RdsLimits(max_depth=1),
        )

    version_two_wire = Wire(version=2)
    strings = version_two_wire.document(version_two_wire.string_vector([b"too long"]))
    with pytest.raises(
        RdsLimitError,
        match=r"max_string_bytes exceeded: 8 > 3 at \$\[0\]",
    ):
        open_rds(
            io.BytesIO(strings),
            limits=RdsLimits(max_string_bytes=3),
        )

    compressed = gzip.compress(vector_payload)
    with pytest.raises(
        RdsLimitError,
        match=r"max_temp_bytes exceeded: .* at \$source",
    ):
        open_rds(
            io.BytesIO(compressed),
            limits=RdsLimits(max_temp_bytes=10),
            temp_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_scratch_free_space_is_checked_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire()
    payload = gzip.compress(wire.document(wire.integer_vector([1, 2, 3])))
    monkeypatch.setattr(
        "scarf.readers._rds._storage.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(RdsLimitError, match=r"scratch_free_bytes exceeded.*\$source"):
        open_rds(io.BytesIO(payload), temp_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_rejects_rdata_headers_and_bad_references() -> None:
    wire = Wire()
    with pytest.raises(RdsFormatError, match="RData workspace header"):
        open_rds(io.BytesIO(b"RDX3\n" + wire.document(wire.nil())))

    bad_reference = wire.document(wire.integer((3 << 8) | RType.REFERENCE))
    with pytest.raises(
        RdsFormatError,
        match=r"reference index 3 is out of range at \$",
    ):
        open_rds(io.BytesIO(bad_reference))
