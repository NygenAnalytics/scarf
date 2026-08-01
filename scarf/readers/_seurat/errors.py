class MatrixSourceError(ValueError):
    """Base error for matrix source validation."""


class ResourceLimitError(MatrixSourceError):
    """A source or requested block exceeds configured limits."""


class UnsafeSidecarError(MatrixSourceError):
    """A sidecar uses a path or HDF5 feature that is not safe to load."""


class UnsupportedMatrixOperation(NotImplementedError):
    """A serialized matrix operation has no safe local implementation."""

    def __init__(
        self,
        object_path: str,
        operation: str,
        class_name: str | None = None,
        reason: str | None = None,
    ) -> None:
        details = f" at {object_path}: operation {operation!r}"
        if class_name is not None:
            details += f", class {class_name!r}"
        if reason is not None:
            details += f" ({reason})"
        super().__init__("Unsupported matrix operation" + details)
        self.objectPath = object_path
        self.object_path = object_path
        self.operation = operation
        self.className = class_name
        self.class_name = class_name
        self.reason = reason
