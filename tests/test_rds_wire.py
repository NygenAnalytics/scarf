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
    R_INT_NA,
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


def test_ascii_octal_and_string_escapes_are_decoded() -> None:
    wire = Wire("ascii", version=2)
    octal_value = b"AB\x07C!G"
    octal_char = (
        wire.integer(wire.flags(RType.CHAR, gp=1 << 6))
        + wire.integer(len(octal_value))
        + b"\\101B\\7C\\41G\n"
    )
    escaped_value = b"\n\t\v\b\r\f\a\\?'\"q"
    escaped_char = (
        wire.integer(wire.flags(RType.CHAR, gp=1 << 6))
        + wire.integer(len(escaped_value))
        + b"\\n\\t\\v\\b\\r\\f\\a\\\\\\?\\'\\\"\\q\n"
    )
    root = wire.integer(RType.STRING) + wire.integer(2) + octal_char + escaped_char

    with open_rds(io.BytesIO(wire.document(root))) as document:
        strings = document.root.value
        assert isinstance(strings, LazyStringVector)
        assert strings.raw(0) == octal_value
        assert strings.raw(1) == escaped_value
        assert strings[:] == [
            octal_value.decode("ascii"),
            escaped_value.decode("ascii"),
        ]


def test_ascii_numeric_special_tokens_are_parsed() -> None:
    wire = Wire("ascii", version=2)
    integer = wire.integer(RType.INTEGER) + wire.integer(2) + b"NA\n7\n"
    with open_rds(io.BytesIO(wire.document(integer))) as document:
        np.testing.assert_array_equal(document.root.value[:], [R_INT_NA, 7])

    real = (
        wire.integer(RType.REAL) + wire.integer(5) + b"NA\nNaN\nInf\n-Inf\n0x1.8p+1\n"
    )
    with open_rds(io.BytesIO(wire.document(real))) as document:
        values = document.root.value[:]
        assert np.isnan(values[0])
        assert np.isnan(values[1])
        assert np.isposinf(values[2])
        assert np.isneginf(values[3])
        assert values[4] == 3.0

    complex_vector = (
        wire.integer(RType.COMPLEX) + wire.integer(2) + b"1.5\n-2.25\nNA\nInf\n"
    )
    with open_rds(io.BytesIO(wire.document(complex_vector))) as document:
        values = document.root.value[:]
        assert values[0] == complex(1.5, -2.25)
        assert np.isnan(values[1].real)
        assert np.isposinf(values[1].imag)


@pytest.mark.parametrize("case", ["header", "root", "vector"])
def test_truncated_binary_payloads_report_object_paths(case: str) -> None:
    wire = Wire()
    cases = {
        "header": (b"X\n\x00\x00", "$header.formatVersion"),
        "root": (wire.header(), "$"),
        "vector": (
            wire.document(wire.integer_vector([1, 2]))[:-1],
            "$",
        ),
    }
    payload, expected_path = cases[case]

    with pytest.raises(
        RdsFormatError,
        match="unexpected end of stream",
    ) as caught:
        open_rds(io.BytesIO(payload))
    assert caught.value.path == expected_path


@pytest.mark.parametrize("case", ["token", "escape"])
def test_ascii_eof_inside_token_or_escape_is_rejected(case: str) -> None:
    wire = Wire("ascii", version=2)
    roots = {
        "token": wire.integer(RType.NIL_VALUE).rstrip(b"\n"),
        "escape": (
            wire.integer(wire.flags(RType.CHAR, gp=1 << 6)) + wire.integer(1) + b"\\"
        ),
    }

    with pytest.raises(
        RdsFormatError,
        match="unexpected end of stream",
    ) as caught:
        open_rds(io.BytesIO(wire.header() + roots[case]))
    assert caught.value.path == "$"


def test_ascii_tokens_enforce_the_127_byte_boundary() -> None:
    wire = Wire("ascii", version=2)
    accepted = b"0." + b"0" * 124 + b"1"
    assert len(accepted) == 127
    root = wire.integer(RType.REAL) + wire.integer(1) + accepted + b"\n"
    with open_rds(io.BytesIO(wire.document(root))) as document:
        assert document.root.value[0] == float(accepted)

    rejected = accepted + b"0"
    assert len(rejected) == 128
    root = wire.integer(RType.REAL) + wire.integer(1) + rejected + b"\n"
    with pytest.raises(
        RdsFormatError,
        match="ASCII token exceeds 127 bytes",
    ) as caught:
        open_rds(io.BytesIO(wire.document(root)))
    assert caught.value.path == "$[0]"


@pytest.mark.parametrize("index", [0, -1, 2])
def test_extended_reference_indices_are_validated(index: int) -> None:
    wire = Wire()
    reference = wire.integer(RType.REFERENCE) + wire.integer(index)

    with pytest.raises(
        RdsFormatError,
        match=rf"reference index {index} is out of range",
    ) as caught:
        open_rds(io.BytesIO(wire.document(reference)))
    assert caught.value.path == "$"


def test_character_vector_reference_requires_a_char_target() -> None:
    wire = Wire()
    referenced_symbol = wire.integer((1 << 8) | RType.REFERENCE)
    strings = wire.integer(RType.STRING) + wire.integer(1) + referenced_symbol
    root = wire.vector([wire.symbol("not-a-char"), strings])

    with pytest.raises(
        RdsFormatError,
        match="character vector reference does not target CHAR",
    ) as caught:
        open_rds(io.BytesIO(wire.document(root)))
    assert caught.value.path == "$[1][0]"


def test_undefined_bytecode_reference_is_rejected() -> None:
    wire = Wire()
    bytecode = (
        wire.integer(RType.BYTECODE)
        + wire.integer(1)
        + wire.integer_vector([12])
        + wire.integer(1)
        + wire.integer(RType.BYTECODE_REFERENCE)
        + wire.integer(0)
    )

    with pytest.raises(
        RdsFormatError,
        match="bytecode reference 0 is out of range",
    ) as caught:
        open_rds(io.BytesIO(wire.document(bytecode)))
    assert caught.value.path == "$.constants[0]"


@pytest.mark.parametrize("case", ["marker", "version", "native-order"])
def test_malformed_headers_are_rejected(case: str) -> None:
    version_one = Wire(version=1)
    cases = {
        "marker": (
            b"Q\n",
            "unknown R serialization format marker",
            "$header",
        ),
        "version": (
            version_one.document(version_one.nil()),
            "unsupported R serialization version 1",
            "$header.formatVersion",
        ),
        "native-order": (
            b"B\n\x00\x00\x00\x00",
            "cannot determine native binary byte order",
            "$header",
        ),
    }
    payload, message, expected_path = cases[case]

    with pytest.raises(RdsFormatError, match=message) as caught:
        open_rds(io.BytesIO(payload))
    assert caught.value.path == expected_path


def test_non_ascii_native_encoding_header_is_rejected() -> None:
    wire = Wire()
    payload = wire.header()[:-5] + b"\xffTF-8" + wire.nil()

    with pytest.raises(
        RdsFormatError,
        match="native encoding name is not ASCII",
    ) as caught:
        open_rds(io.BytesIO(payload))
    assert caught.value.path == "$header.nativeEncoding"


def test_character_vector_rejects_tagged_char_nodes() -> None:
    wire = Wire()
    tagged_char = (
        wire.integer(wire.flags(RType.CHAR, tag=True, gp=1 << 6))
        + wire.integer(1)
        + wire.string(b"x")
        + wire.nil()
    )
    root = wire.integer(RType.STRING) + wire.integer(1) + tagged_char

    with pytest.raises(
        RdsFormatError,
        match="CHAR node cannot have a tag",
    ) as caught:
        open_rds(io.BytesIO(wire.document(root)))
    assert caught.value.path == "$[0]"


@pytest.mark.parametrize(
    ("type_code", "message"),
    [
        (127, "unknown R node type 127"),
        (
            RType.CLASS_REFERENCE,
            "class_reference is not a readable R serialization node",
        ),
        (
            RType.GENERIC_REFERENCE,
            "generic_reference is not a readable R serialization node",
        ),
    ],
)
def test_invalid_node_type_tags_are_rejected(
    type_code: int,
    message: str,
) -> None:
    wire = Wire()

    with pytest.raises(RdsFormatError, match=message) as caught:
        open_rds(io.BytesIO(wire.document(wire.integer(type_code))))
    assert caught.value.path == "$"


def test_scalar_and_empty_container_nodes_are_preserved() -> None:
    wire = Wire()
    special = wire.integer(RType.SPECIAL) + wire.integer(3) + wire.string(b"sum")
    builtin = wire.integer(RType.BUILTIN) + wire.integer(1) + wire.string(b"+")
    expression = wire.integer(RType.EXPRESSION) + wire.integer(0)
    root = wire.vector(
        [
            wire.char(None),
            wire.char(b""),
            special,
            builtin,
            wire.integer(RType.ANY),
            wire.integer_vector([]),
            wire.string_vector([]),
            wire.vector([]),
            expression,
        ]
    )

    with open_rds(io.BytesIO(wire.document(root))) as document:
        (
            missing_char,
            empty_char,
            special_node,
            builtin_node,
            any_node,
            empty_integer,
            empty_string,
            empty_vector,
            empty_expression,
        ) = document.root.value
        assert missing_char.value is None
        assert empty_char.value == ""
        assert special_node.value == "sum"
        assert builtin_node.value == "+"
        assert any_node.value is None
        assert empty_integer.value[:].size == 0
        assert empty_string.value[:] == []
        assert empty_vector.value == ()
        assert empty_expression.value == ()


@pytest.mark.parametrize(
    "case",
    ["negative-vector", "long-upper", "char-length", "string-child"],
)
def test_invalid_container_shapes_are_rejected(case: str) -> None:
    wire = Wire()
    cases = {
        "negative-vector": (
            wire.integer(RType.VECTOR) + wire.integer(-2),
            "negative vector length -2",
            "$.length",
        ),
        "long-upper": (
            wire.integer(RType.INTEGER)
            + wire.integer(-1)
            + wire.integer(65_537)
            + wire.integer(0),
            "long-vector upper length 65537 exceeds R's wire limit",
            "$.length",
        ),
        "char-length": (
            wire.integer(RType.CHAR) + wire.integer(-2),
            "invalid string length -2",
            "$",
        ),
        "string-child": (
            wire.integer(RType.STRING) + wire.integer(1) + wire.integer(RType.INTEGER),
            "character vector element has type INTEGER",
            "$[0]",
        ),
    }
    root, message, expected_path = cases[case]

    with pytest.raises(RdsFormatError, match=message) as caught:
        open_rds(io.BytesIO(wire.document(root)))
    assert caught.value.path == expected_path


def test_trailing_data_rules_follow_the_wire_encoding() -> None:
    wire = Wire()
    with pytest.raises(
        RdsFormatError,
        match="1 trailing bytes after root object",
    ) as caught:
        open_rds(io.BytesIO(wire.document(wire.nil()) + b"\x00"))
    assert caught.value.path == "$"

    ascii_wire = Wire("ascii", version=2)
    whitespace = ascii_wire.document(ascii_wire.nil()) + b" \t\r\n"
    with open_rds(io.BytesIO(whitespace)) as document:
        assert document.root.is_null

    non_whitespace = ascii_wire.document(ascii_wire.nil()) + b" !"
    with pytest.raises(
        RdsFormatError,
        match="non-whitespace data follows the root object",
    ) as caught:
        open_rds(io.BytesIO(non_whitespace))
    assert caught.value.path == "$"


def test_seekable_stream_range_and_position_are_preserved() -> None:
    wire = Wire()
    payload = wire.document(wire.integer_vector([4, 5]))
    prefix = b"ignored-prefix"
    stream = io.BytesIO(prefix + payload)
    start = stream.seek(len(prefix))

    with open_rds(stream) as document:
        assert document.source.source_bytes == len(payload)
        assert document.source.source_sha256 == hashlib.sha256(payload).hexdigest()
        np.testing.assert_array_equal(document.root.value[:], [4, 5])

    assert stream.tell() == start
    assert stream.closed is False


def test_failed_stream_parse_restores_the_input_position() -> None:
    prefix = b"ignored-prefix"
    stream = io.BytesIO(prefix + b"Q\n")
    start = stream.seek(len(prefix))

    with pytest.raises(
        RdsFormatError,
        match="unknown R serialization format marker",
    ):
        open_rds(stream)

    assert stream.tell() == start
    assert stream.closed is False


def test_extended_references_and_singletons_preserve_identity() -> None:
    wire = Wire()
    extended_reference = wire.integer(RType.REFERENCE) + wire.integer(1)
    root = wire.vector(
        [
            wire.symbol("shared"),
            extended_reference,
            wire.nil(),
            wire.nil(),
            wire.integer(RType.BASE_ENVIRONMENT),
            wire.integer(RType.BASE_ENVIRONMENT),
        ]
    )

    with open_rds(io.BytesIO(wire.document(root))) as document:
        symbol, reference, first_nil, second_nil, first_base, second_base = (
            document.root.value
        )
        assert symbol is reference
        assert first_nil is second_nil
        assert first_base is second_base
