from collections.abc import Generator
from typing import IO


def get_file_handle(fn: str) -> IO:
    """Returns a file object for the given file name.

    Args:
        fn: The path to the file (file name).
    """
    import gzip

    try:
        if fn.rsplit(".", 1)[-1] == "gz":
            return gzip.open(fn, mode="rt")
        else:
            return open(fn, "r")
    except (OSError, IOError, FileNotFoundError):
        raise FileNotFoundError("ERROR: FILE NOT FOUND: %s" % fn)


def read_file(fn: str) -> Generator[str, None, None]:
    """Yields the lines from the file the given file name points to.

    Args:
        fn: The path to the file (file name).
    """
    from . import get_file_handle

    fh = get_file_handle(fn)
    try:
        for line in fh:
            yield line.rstrip()
    finally:
        fh.close()
