"""Optional explicit widths over automatic read, compute, and write planning."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StorageIoPolicy:
    """Requested widths for one store or writer. Unset values mean automatic planning."""

    readWorkers: int | None = None
    computeWorkers: int | None = None
    writeWorkers: int | None = None

    def __post_init__(self) -> None:
        for name in ("readWorkers", "computeWorkers", "writeWorkers"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive when set")


DEFAULT_STORAGE_IO_POLICY = StorageIoPolicy()
