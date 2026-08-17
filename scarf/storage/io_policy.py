"""Operation-local storage I/O admission."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StorageIoPolicy:
    sourceReadsInFlight: int = 1
    sourceGroupChunks: int = 1
    destShardsInFlight: int = 1
    destCommitsInFlight: int = 1
    computeWorkers: int = 1
    groupsInFlight: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "sourceReadsInFlight",
            "sourceGroupChunks",
            "destShardsInFlight",
            "destCommitsInFlight",
            "computeWorkers",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.groupsInFlight is not None and int(self.groupsInFlight) < 1:
            raise ValueError("groupsInFlight must be positive when set")


DEFAULT_STORAGE_IO_POLICY = StorageIoPolicy()
