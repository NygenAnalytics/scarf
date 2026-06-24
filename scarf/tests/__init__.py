import os
import shutil
import sys

from ..utils import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")

__all__ = ["full_path", "remove", "dask_total_sum"]


def dask_total_sum(raw_data) -> int:
    total = 0
    for block in raw_data.blocks:
        total += int(block.compute().sum())
    return total


def full_path(fn, *args):
    if fn == "" or fn is None:
        return os.path.join("scarf", "tests", "datasets")
    else:
        return os.path.join("scarf", "tests", "datasets", fn, *args)


def remove(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    elif os.path.exists(dir_path):
        os.unlink(dir_path)
    else:
        pass
