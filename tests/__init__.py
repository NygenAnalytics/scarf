import os
import shutil

__all__ = ["full_path", "remove", "dask_total_sum"]

_DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def dask_total_sum(raw_data) -> int:
    total = 0
    for block in raw_data.blocks:
        total += int(block.compute().sum())
    return total


def full_path(fn, *args):
    if fn == "" or fn is None:
        return _DATASETS_DIR
    return os.path.join(_DATASETS_DIR, fn, *args)


def remove(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    elif os.path.exists(dir_path):
        os.unlink(dir_path)
