import enum
from dataclasses import dataclass


class RType(enum.IntEnum):
    """R serialization node types."""

    NIL = 0
    SYMBOL = 1
    SYM = 1
    PAIRLIST = 2
    LIST = 2
    CLOSURE = 3
    CLO = 3
    ENVIRONMENT = 4
    ENV = 4
    PROMISE = 5
    PROM = 5
    LANGUAGE = 6
    LANG = 6
    SPECIAL = 7
    BUILTIN = 8
    CHAR = 9
    LOGICAL = 10
    LGL = 10
    INTEGER = 13
    INT = 13
    REAL = 14
    COMPLEX = 15
    CPLX = 15
    STRING = 16
    STR = 16
    DOTS = 17
    ANY = 18
    VECTOR = 19
    VEC = 19
    EXPRESSION = 20
    EXPR = 20
    BYTECODE = 21
    BCODE = 21
    EXTERNAL_POINTER = 22
    EXTPTR = 22
    WEAK_REFERENCE = 23
    WEAKREF = 23
    RAW = 24
    S4 = 25
    ALTREP = 238
    BYTECODE_ATTR_PAIRLIST = 239
    BYTECODE_ATTR_LANGUAGE = 240
    BASE_ENVIRONMENT = 241
    EMPTY_ENVIRONMENT = 242
    BYTECODE_REFERENCE = 243
    BYTECODE_DEFINITION = 244
    GENERIC_REFERENCE = 245
    CLASS_REFERENCE = 246
    PERSISTENT = 247
    PACKAGE = 248
    NAMESPACE = 249
    BASE_NAMESPACE = 250
    MISSING_ARGUMENT = 251
    UNBOUND_VALUE = 252
    GLOBAL_ENVIRONMENT = 253
    NIL_VALUE = 254
    REFERENCE = 255


class RdsEncoding(enum.StrEnum):
    """Serialization payload encoding."""

    XDR = "xdr"
    NATIVE = "native"
    ASCII = "ascii"


class RdsCompression(enum.StrEnum):
    """Compression wrapped around an RDS payload."""

    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"
    ZSTD = "zstd"


@dataclass(frozen=True, slots=True)
class RdsLimits:
    """Finite parser and scratch-space limits."""

    max_objects: int = 100_000_000
    max_depth: int = 512
    max_string_bytes: int = 256 * 1024**2
    max_vector_length: int = 1 << 40
    max_temp_bytes: int = 16 * 1024**3

    def __post_init__(self) -> None:
        for name, value in (
            ("max_objects", self.max_objects),
            ("max_depth", self.max_depth),
            ("max_string_bytes", self.max_string_bytes),
            ("max_vector_length", self.max_vector_length),
            ("max_temp_bytes", self.max_temp_bytes),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class RdsSourceMetadata:
    """Input and decompression details."""

    name: str
    compression: RdsCompression
    source_bytes: int | None
    payload_bytes: int
    source_sha256: str
    payload_sha256: str
    spooled: bool


@dataclass(frozen=True, slots=True)
class RdsMetadata:
    """R serialization header details."""

    encoding: RdsEncoding
    format_version: int
    writer_version: int
    minimum_reader_version: int
    native_encoding: str | None
    byte_order: str | None

    @staticmethod
    def unpack_r_version(packed: int) -> tuple[int, int, int]:
        """Return an R packed version as major, minor, and patch."""
        major, remainder = divmod(packed, 65536)
        minor, patch = divmod(remainder, 256)
        return major, minor, patch

    @property
    def writer_version_tuple(self) -> tuple[int, int, int]:
        return self.unpack_r_version(self.writer_version)

    @property
    def minimum_reader_version_tuple(self) -> tuple[int, int, int]:
        return self.unpack_r_version(self.minimum_reader_version)


class RdsError(Exception):
    """Base class for deterministic RDS errors."""

    def __init__(
        self, message: str, *, path: str = "$", offset: int | None = None
    ) -> None:
        self.message = message
        self.path = path
        self.offset = offset
        location = f" at {path}"
        if offset is not None:
            location += f" (byte {offset})"
        super().__init__(f"{message}{location}")


class RdsFormatError(RdsError):
    """Malformed or unsupported wire data."""


class RdsLimitError(RdsError):
    """A configured parser limit was exceeded."""

    def __init__(
        self,
        limit: str,
        actual: int,
        maximum: int,
        *,
        path: str,
        offset: int | None = None,
    ) -> None:
        self.limit = limit
        self.actual = actual
        self.maximum = maximum
        super().__init__(
            f"{limit} exceeded: {actual} > {maximum}",
            path=path,
            offset=offset,
        )


class RdsClosedError(RdsError):
    """Lazy data was accessed after its document closed."""


R_INT_NA = -(1 << 31)
