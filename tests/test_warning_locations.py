"""Scarf's warnings point at the line of the code that calls Scarf.

Each test notes the line where its own call into Scarf starts and checks that
every warning names this file and that line, however many Scarf frames lie
between the call and the warning.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np

import scarf
from scarf.metadata.selection import CellField
from scarf.utils.warnings import warn
from tests.qc_helpers import fresh_qc_store, open_qc_store

_THIS_FILE = Path(__file__).resolve()
_CELL_LEVEL = "Cell-level statistical testing treats each cell"
_REMOVED_GROUP = "Requested group 'g2' was removed"


def _next_line() -> int:
    """Return the line after the caller's current line, where its call starts."""
    return sys._getframe(1).f_lineno + 1


def _assert_point_at(caught: list[warnings.WarningMessage], line: int) -> None:
    assert caught
    for item in caught:
        location = (Path(item.filename).resolve(), item.lineno)
        assert location == (_THIS_FILE, line), str(item.message)


def _matching(caught: list[warnings.WarningMessage], prefix: str) -> list:
    return [item for item in caught if str(item.message).startswith(prefix)]


def test_warn_names_the_first_frame_outside_the_package() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = _next_line()
        warn("a test warning", RuntimeWarning)

    _assert_point_at(caught, line)
    assert [item.category for item in caught] == [RuntimeWarning]

    # A sibling directory whose name extends the package's lies outside it.
    sibling = os.path.dirname(scarf.__file__) + "_plugin" + os.sep + "caller.py"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(compile("warn('from a sibling')", sibling, "exec"), {"warn": warn})
    assert [item.filename for item in caught] == [sibling]


def test_statistical_warnings_name_the_callers_line_through_a_datastore() -> None:
    """A datastore operation warns at its caller, not inside the domain package."""
    store = open_qc_store(fresh_qc_store()[0], min_features_per_cell=-1)
    cells = np.arange(store.cells.N)
    store.cells.insert(
        "warning_group", np.array([f"g{index % 3}" for index in cells], dtype=object)
    )
    store.cells.insert("warning_keep", cells % 3 != 2)
    selection = store.snapshot_cell_selection("I")
    with warnings.catch_warnings(record=True) as through_store:
        warnings.simplefilter("always")
        line = _next_line()
        store.run_statistical_testing(
            ["GENE_A"],
            CellField("warning_group"),
            cell_selection=selection,
            groups=["g0", "g1", "g2"],
            subset_by="warning_keep",
            skip_save=True,
        )
    _assert_point_at(_matching(through_store, _CELL_LEVEL), line)
    _assert_point_at(_matching(through_store, _REMOVED_GROUP), line)
