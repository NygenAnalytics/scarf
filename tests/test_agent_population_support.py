"""Population support uses frozen cells, bounded reads and observed study units."""

from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent.parameter_tuning import diagnostics
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from scarf.storage.refs import ArtifactRef
from tests.agent_examples import example


class _BoundedArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape, self.dtype = values.shape, values.dtype
        self.reads: list[int] = []

    def __getitem__(self, key: Any) -> np.ndarray:
        value = self.values[key]
        assert value.size <= 65_536
        self.reads.append(value.size)
        return value


class _Metadata:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.arrays = {column: _BoundedArray(value) for column, value in values.items()}
        self.columns = [*values, "author_cell_type"]
        self.missing: dict[str, _BoundedArray] = {}
        self.accesses: list[str] = []

    def _get_array(self, column: str) -> _BoundedArray:
        assert column != "author_cell_type"
        self.accesses.append(column)
        return self.arrays[column]

    def _get_missing_mask_array(self, column: str) -> _BoundedArray | None:
        return self.missing.get(column)


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    rows: np.ndarray | None = None,
) -> tuple[Any, ParameterCandidateEvaluation, _BoundedArray]:
    evaluation = example(ParameterCandidateEvaluation)
    selection = ArtifactRef(
        scope="datastore", kind="cell_selection", artifact_id="c" * 64
    )
    evaluation.cellSelection = evaluation.cellSelection.model_copy(
        update={
            "scope": "datastore",
            "assay": None,
            "kind": "cell_selection",
            "artifactId": selection.artifact_id,
        }
    )
    status = SimpleNamespace(
        exists=True, complete=True, inputs={"cell_selection": selection.to_dict()}
    )
    calls: Counter[str] = Counter()
    array = _BoundedArray(labels)

    def inspect(ref: ArtifactRef) -> Any:
        calls["inspect_artifact"] += 1
        return status

    def load(ref: ArtifactRef) -> Any:
        calls["load_artifact"] += 1
        return {"values": array}

    store = SimpleNamespace(
        zw=object(),
        cells=_Metadata(metadata),
        inspect_artifact=inspect,
        load_artifact=load,
        calls=calls,
        status=status,
    )
    monkeypatch.setattr(diagnostics, "as_zarr_array", lambda value, **kwargs: value)

    def selected(*args: Any, **kwargs: Any) -> np.ndarray:
        assert args[1] == selection
        return np.arange(len(labels)) if rows is None else rows

    monkeypatch.setattr(diagnostics, "read_stored_selection_indices", selected)
    return store, evaluation, array


def test_support_exposes_restricted_populations_and_missing_unit_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = np.asarray([1, 3, 4, 6, 8, 9, 10, 11])
    donors = np.full(12, "unused", dtype=object)
    captures = donors.copy()
    donors[rows] = ["d1", "d1", "d1", "d2", "d2", "d2", "d2", "d3"]
    captures[rows] = ["c1", "c1", "c1", "c2", "c3", "c3", "c3", "c4"]
    store, evaluation, _ = _setup(
        monkeypatch,
        np.asarray([1, 1, 1, 1, 2, 2, 2, 3]),
        {"donor": donors, "capture": captures},
        rows,
    )
    mask = np.zeros(12, dtype=bool)
    mask[10] = True
    store.cells.missing["donor"] = _BoundedArray(mask)
    evidence = diagnostics.population_support_evidence(
        store, evaluation, ["donor", "capture"]
    )
    donor = evidence["columns"]["donor"]
    assert evidence["selectedCells"] == 8
    assert evidence["observedPopulations"] == 3
    assert donor["observedGroups"] == 3
    assert donor["coveredCells"] == 7
    assert donor["missingCells"] == 1
    assert donor["coverageFraction"] == 7 / 8
    assert donor["populations"][0]["cluster"] == "3"
    populations = {row["cluster"]: row for row in donor["populations"]}
    assert populations["1"]["supportingGroups"] == 2
    assert populations["1"]["largestGroupFraction"] == 3 / 4
    assert populations["1"]["topGroups"][1] == {
        "value": "d2",
        "valueType": "str",
        "cells": 1,
        "fractionOfPopulation": 1 / 4,
        "fractionOfGroup": 1 / 3,
    }
    assert populations["2"]["supportingGroups"] == 1
    assert populations["2"]["coverageFraction"] == 2 / 3
    assert populations["2"]["largestGroupFraction"] == 2 / 3
    assert populations["3"]["groupsWithAtLeast5Cells"] == 0
    assert evidence["columns"]["capture"]["observedGroups"] == 4
    assert store.calls == {"inspect_artifact": 1, "load_artifact": 1}
    assert set(store.cells.accesses) == {"donor", "capture"}


def test_support_reads_large_selected_axes_in_bounded_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 70_000
    labels = np.zeros(n, dtype=np.int32)
    labels[-5:] = 1
    groups = np.full(n, "main", dtype=object)
    groups[-5:] = "rare"
    store, evaluation, array = _setup(monkeypatch, labels, {"donor": groups})
    evidence = diagnostics.population_support_evidence(store, evaluation, ["donor"])
    assert array.reads == [65_536, n - 65_536]
    assert store.cells.arrays["donor"].reads == [65_536, n - 65_536]
    rare = evidence["columns"]["donor"]["populations"][0]
    assert rare["cluster"] == "1"
    assert rare["cells"] == 5
    assert rare["supportingGroups"] == rare["groupsWithAtLeast5Cells"] == 1
    assert rare["largestGroupFraction"] == 1.0


def test_group_display_truncation_keeps_exact_counts_and_fractions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = np.asarray([f"d{i}" for i in range(70)])
    store, evaluation, _ = _setup(
        monkeypatch, np.ones(70, dtype=int), {"donor": groups}
    )
    evidence = diagnostics.population_support_evidence(store, evaluation, ["donor"])
    column = evidence["columns"]["donor"]
    population = column["populations"][0]
    assert column["observedGroups"] == population["supportingGroups"] == 70
    assert population["omittedGroups"] == population["omittedCells"] == 65
    assert len(population["topGroups"]) == 5
    assert population["largestGroupFraction"] == 1 / 70
    assert (
        sum(row["cells"] for row in population["topGroups"])
        + population["omittedCells"]
        == population["cells"]
    )


def test_population_and_column_display_omissions_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, evaluation, _ = _setup(
        monkeypatch, np.arange(70), {"donor": np.asarray(["one"] * 70)}
    )
    evidence = diagnostics.population_support_evidence(
        store, evaluation, ["donor", "absent", "unused"]
    )
    column = evidence["columns"]["donor"]
    assert evidence["observedPopulations"] == 70
    assert column["omittedPopulations"] == column["omittedPopulationCells"] == 6
    assert len(column["populations"]) == 64
    assert evidence["omittedColumns"] == ["unused"]
    assert evidence["columns"]["absent"]["status"] == "unavailable"
    assert store.cells.accesses == ["donor"]


def test_missing_nonfinite_and_typed_unit_values_are_not_silently_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, evaluation, _ = _setup(
        monkeypatch,
        np.ones(7, dtype=int),
        {
            "donor": np.asarray(
                [1, "1", True, None, float("nan"), float("inf"), ""], dtype=object
            )
        },
    )
    column = diagnostics.population_support_evidence(store, evaluation, ["donor"])[
        "columns"
    ]["donor"]
    assert column["observedGroups"] == 3
    assert column["coveredCells"] == 3
    assert column["missingCells"] == 4
    assert {row["valueType"] for row in column["populations"][0]["topGroups"]} == {
        "str",
        "int",
        "bool",
    }


@pytest.mark.parametrize(
    "damage", ["failed", "incomplete", "wrongCells", "shape", "nonfinite", "fractional"]
)
def test_invalid_population_evidence_fails_before_metadata_reads(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    labels = np.asarray([1.0, 2.0])
    store, evaluation, array = _setup(
        monkeypatch, labels, {"donor": np.asarray(["a", "b"])}
    )
    if damage == "failed":
        evaluation.status = "failed"
    elif damage == "incomplete":
        store.status.complete = False
    elif damage == "wrongCells":
        store.status.inputs["cell_selection"]["artifact_id"] = "d" * 64
    elif damage == "shape":
        array.shape = (3,)
    elif damage == "nonfinite":
        labels[0] = np.nan
    else:
        labels[0] = 1.5
    with pytest.raises(ValueError):
        diagnostics.population_support_evidence(store, evaluation, ["donor"])
    assert store.cells.accesses == []
