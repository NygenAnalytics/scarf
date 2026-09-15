"""Scientific metadata reuse keeps typed inputs and QC policies attributable."""

from collections import Counter
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf.agent.experimental_context import characterization as characterization_module
from scarf.agent.experimental_context import qc_evidence
from scarf.agent.experimental_context.contracts import CovariateCharacterization
from tests.test_agent_experimental_context import _Store, _context, _replace_store_cells


def _capture_context() -> tuple[Any, CovariateCharacterization]:
    store = _Store()
    donors = np.repeat(["d1", "d2", "d3"], 8)
    condition = np.tile(np.repeat(["case", "control"], 4), 3)
    values = {
        "I": np.ones(24, dtype=bool),
        "ids": np.asarray([f"cell{i}" for i in range(24)]),
        "names": np.asarray([f"cell{i}" for i in range(24)]),
        "capture": donors.copy(),
        "donor": donors,
        "condition": condition,
        "sample": np.asarray(
            [f"{d}:{c}" for d, c in zip(donors, condition, strict=True)]
        ),
        "RNA_nCounts": np.tile(np.arange(8, dtype=float) + 100, 3),
        "RNA_nFeatures": np.tile(np.arange(8, dtype=float) + 50, 3),
    }
    _replace_store_cells(store, values)
    deps = _context(store, directions={"physicalCaptureColumn": "capture"}).deps
    characterized = CovariateCharacterization(
        status="done",
        columns=[
            {"name": name, "kind": "categorical"}
            for name in ("condition", "sample", "donor")
        ],
        coefficients=[
            {
                "name": "condition",
                "scope": "betweenUnit",
                "observationUnit": "sample",
                "independentUnit": "donor",
                "pairedCoverage": {"design": "paired"},
            }
        ],
    )
    return deps, characterized


def test_capture_design_is_computed_once_across_real_policy_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps, characterized = _capture_context()
    captures = deps.cells.fetch("capture")
    expected = {
        label: qc_evidence._capture_design_safety(deps, characterized, captures, label)
        for label in ("d1", "d2", "d3")
    }
    calls: Counter[str] = Counter()
    reads: Counter[str] = Counter()
    compute = qc_evidence._compute_capture_design_safety
    fetch = qc_evidence._QcDesignData.fetch

    def counted(*args: Any) -> Any:
        calls[args[-1]] += 1
        return compute(*args)

    def fetched(self: Any, column: str) -> Any:
        if column not in self.values:
            reads[column] += 1
        return fetch(self, column)

    monkeypatch.setattr(qc_evidence, "_compute_capture_design_safety", counted)
    monkeypatch.setattr(qc_evidence._QcDesignData, "fetch", fetched)
    profiles = qc_evidence._offered_qc_profiles(deps, characterized)
    assert sum(bool(profile.captureFailureEvidence) for profile in profiles) >= 4
    assert calls == {"d1": 1, "d2": 1, "d3": 1}
    assert reads == {"condition": 1, "sample": 1, "donor": 1}
    assert deps.qcDesignData is None
    for profile in profiles:
        for failure in profile.captureFailureEvidence:
            rows, condition_safe, unit_safe = expected[failure.capture]
            assert failure.conditionAndUnitSafety == rows
            assert failure.preservesConditionCoverage == condition_safe
            assert failure.preservesIndependentUnitCoverage == unit_safe
    # A later projection sees changed metadata instead of stale cached safety.
    deps.store.cells._values["condition"][:8] = "only_here"
    updated = qc_evidence._offered_qc_profiles(deps, characterized)
    assert calls == {"d1": 2, "d2": 2, "d3": 2}
    assert all(
        not failure.preservesConditionCoverage
        for profile in updated
        for failure in profile.captureFailureEvidence
        if failure.capture == "d1"
    )


@pytest.mark.parametrize("kind", ["continuous", "unknown"])
def test_continuous_or_unknown_capture_protection_cannot_become_categorical(
    kind: str,
) -> None:
    deps, characterized = _capture_context()
    deps.store.cells._values["condition"] = np.linspace(20, 80, 24)
    deps.store.cells._values["condition"][0] = np.nan
    characterized.columns[0]["kind"] = kind
    rows, preserves_conditions, preserves_units = qc_evidence._capture_design_safety(
        deps, characterized, deps.cells.fetch("capture"), "d1"
    )
    assert not preserves_conditions and not preserves_units
    assert "requiredGroups" not in rows[0]
    if kind == "continuous":
        assert rows[0]["matchedRowsBeforeExclusion"] == 23
        assert rows[0]["matchedRowsAfterExclusion"] == 16
        assert rows[0]["independentUnitsAfterExclusion"] == 2
        assert len(rows[0]["quantilesAfterExclusion"]) == 5
        assert "not been established" in rows[0]["reason"]
    else:
        assert rows[0]["reason"] == "unknownCovariateKind"


def test_column_inventory_reuse_checks_values_kinds_missingness_and_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    inventory: dict[str, Any] = {}
    calls: Counter[str] = Counter()
    digest = characterization_module.column_partition_digest

    def counted(cells: Any, column: str, **kwargs: Any) -> Any:
        calls[column] += 1
        return digest(cells, column, **kwargs)

    monkeypatch.setattr(characterization_module, "column_partition_digest", counted)

    def characterize(**directions: Any) -> Any:
        return characterization_module.characterize_covariates(
            store,
            cellSelection=store.cell_selection,
            directions=directions,
            inventory=inventory,
        )

    first = characterize()
    baseline = calls.copy()
    assert first.status == "done"
    characterize(columnDomains={"sequencing_depth": "technical"})
    assert calls == baseline
    store.cells._values["sequencing_depth"][0] += 1
    characterize()
    assert calls == baseline + Counter({"sequencing_depth": 1})
    characterize(columnKinds={"sequencing_depth": "categorical"})
    assert calls == baseline + Counter({"sequencing_depth": 2})
    store.cells._values["sequencing_depth"][0] = np.nan
    characterize(columnKinds={"sequencing_depth": "categorical"})
    assert calls == baseline + Counter({"sequencing_depth": 3})
    store.cells._values["I"][0] = False
    store.refresh_cell_selection()
    before = calls.copy()
    characterize()
    assert all(calls[column] > before[column] for column in baseline)


def test_qc_design_scope_is_cleared_after_projection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps, characterized = _capture_context()

    def fail(*args: Any, **kwargs: Any) -> Any:
        assert isinstance(deps.qcDesignData, qc_evidence._QcDesignData)
        raise ValueError("invalid quality metric")

    monkeypatch.setattr(qc_evidence, "_project_qc_profiles", fail)
    with pytest.raises(ValueError, match="invalid quality metric"):
        qc_evidence._offered_qc_profiles(deps, characterized)
    assert deps.qcDesignData is None


def test_nullable_design_metadata_retains_missingness_as_unresolved_protection() -> (
    None
):
    deps, characterized = _capture_context()
    deps.store.cells._values["condition"] = np.linspace(20, 80, 24).astype(object)
    deps.store.cells._values["condition"][0] = pd.NA
    characterized.columns[0]["kind"] = "continuous"
    rows, protected, _ = qc_evidence._capture_design_safety(
        deps, characterized, deps.cells.fetch("capture"), "d3"
    )
    assert not protected and rows[0]["missingRowsAfterExclusion"] == 1
    deps.store.cells._values["donor"] = deps.store.cells._values["donor"].astype(object)
    deps.store.cells._values["donor"][0] = None
    retention = qc_evidence._design_retention(
        deps, characterized, np.ones(24, dtype=bool), np.ones(24, dtype=bool)
    )
    assert "donor:missingValues" in retention["unsafeRetentionGroups"]
    assert "None" not in retention["retainedCellsByColumn"]["donor"]


@pytest.mark.parametrize("missing", [pd.NaT, b"", b" ", b"\xff"])
def test_unusable_imported_unit_labels_remain_missing_in_retention(
    missing: Any,
) -> None:
    deps, characterized = _capture_context()
    deps.store.cells._values["donor"] = deps.store.cells._values["donor"].astype(object)
    deps.store.cells._values["donor"][0] = missing
    retention = qc_evidence._design_retention(
        deps, characterized, np.ones(24, dtype=bool), np.ones(24, dtype=bool)
    )
    assert "donor:missingValues" in retention["unsafeRetentionGroups"]
    assert retention["retainedCellsByColumn"]["donor"] == {"d1": 7, "d2": 8, "d3": 8}


def _joint_capture_context() -> tuple[Any, CovariateCharacterization]:
    deps, characterized = _capture_context()
    deps.store.cells._values["condition"][:8] = "case"
    deps.store.cells._values["time"] = np.concatenate(
        [np.repeat("late", 8), np.tile(np.repeat(["early", "late"], 4), 2)]
    )
    characterized.columns.append({"name": "time", "kind": "categorical"})
    time = deepcopy(characterized.coefficients[0])
    time["name"] = "time"
    characterized.coefficients.append(time)
    deps.protectedCombinations = [["condition", "time"]]
    return deps, characterized


def test_joint_population_loss_is_visible_despite_preserved_marginal_conditions() -> (
    None
):
    deps, characterized = _joint_capture_context()
    captures = deps.cells.fetch("capture")
    rows, condition_safe, unit_safe = qc_evidence._capture_design_safety(
        deps, characterized, captures, "d1"
    )
    assert all(row["preservesConditionCoverage"] for row in rows[:-1])
    assert all(row["preservesIndependentUnitCoverage"] for row in rows[:-1])
    assert not rows[-1]["preservesConditionCoverage"]
    assert not condition_safe and not unit_safe
    retention = qc_evidence._design_retention(
        deps, characterized, np.ones(24, dtype=bool), captures != "d1"
    )
    for column in ("condition", "time"):
        assert all(retention["retainedCellsByColumn"][column].values())
    assert any(
        value.startswith("combination:") for value in retention["unsafeRetentionGroups"]
    )
    deps.qcDesignData = qc_evidence._QcDesignData(deps.cells)
    cached = qc_evidence._design_retention(
        deps, characterized, np.ones(24, dtype=bool), captures != "d1"
    )
    assert cached == retention


@pytest.mark.parametrize("missing", ["value", "column", "independentColumn"])
def test_missing_joint_design_inputs_cannot_establish_safe_capture_exclusion(
    missing: str,
) -> None:
    deps, characterized = _joint_capture_context()
    if missing == "value":
        deps.store.cells._values["time"] = deps.store.cells._values["time"].astype(
            object
        )
        deps.store.cells._values["time"][0] = None
    elif missing == "column":
        del deps.store.cells._values["time"]
    else:
        del deps.store.cells._values["donor"]
    rows, condition_safe, unit_safe = qc_evidence._capture_design_safety(
        deps, characterized, deps.cells.fetch("capture"), "d1"
    )
    assert not unit_safe
    if missing != "independentColumn":
        assert not condition_safe
        assert rows[-1]["reason"] == "missingProtectedCombination"
        retention = qc_evidence._design_retention(
            deps, characterized, np.ones(24, dtype=bool), np.ones(24, dtype=bool)
        )
        assert any(
            entry.startswith("combination:") and entry.endswith(":missingValues")
            for entry in retention["unsafeRetentionGroups"]
        )
    else:
        assert all(row["reason"] == "missingDesignColumn" for row in rows[:-1])


def test_joint_replication_uses_independent_donors_and_scoped_capture_identity() -> (
    None
):
    deps, characterized = _capture_context()
    deps.store.cells._values["time"] = np.tile(["early", "late"], 12)
    characterized.columns.append({"name": "time", "kind": "categorical"})
    deps.protectedCombinations = [["condition", "time"]]
    deps.qcDesignData = qc_evidence._QcDesignData(deps.cells)
    captures = deps.cells.fetch("capture")
    rows, condition_safe, unit_safe = qc_evidence._capture_design_safety(
        deps, characterized, captures, "d1"
    )
    assert condition_safe and unit_safe
    assert rows[-1]["preservesIndependentUnitCoverage"]
    changed_captures = captures.copy()
    changed_captures[changed_captures == "d2"] = "d1"
    rows, condition_safe, unit_safe = qc_evidence._capture_design_safety(
        deps, characterized, changed_captures, "d1"
    )
    assert condition_safe and not unit_safe
    assert not rows[-1]["preservesIndependentUnitCoverage"]
