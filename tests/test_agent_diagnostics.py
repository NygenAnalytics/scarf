"""Descriptive science diagnostics stay bounded and distinguish authority."""

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf.agent.diagnostics import (
    design_diagnostics,
    family_audit,
    finalist_composition,
    representation_diagnostics,
    resolved_roles,
)
from scarf.agent.evidence import _blacklist, _qc_flags, _qc_projections
from scarf.agent.execution import compare_candidates
from scarf.agent.models import AnalysisConfig, AnalysisInputError, Candidate
from tests.test_agent_evidence import study
from tests.test_agent_science_edges import Metadata


def test_qc_projections_show_actual_group_retention_without_changing_selection() -> (
    None
):
    frame = pd.DataFrame(
        {
            "I": [True] * 6,
            "RNA_nCounts": [9, 10, 15, 20, 21, np.nan],
            "RNA_nFeatures": [2, 3, 4, 4, 5, 5],
            "group": ["a", "a", "b", "b", "b", None],
        }
    )
    original = frame.copy(deep=True)
    store = SimpleNamespace(cells=Metadata(frame))
    active = frame.I.to_numpy()
    flags = _qc_flags(store, active, "RNA")
    config = AnalysisConfig(qcPolicy="manual", qcBounds={"RNA_nCounts": (10, 20)})
    results = _qc_projections(
        store, active, "RNA", config, flags, resolved_roles(study(sampleColumn="group"))
    )
    assert [item["executed"] for item in results] == [False, False, True]
    assert results[0]["retainedCells"] == 6
    assert results[2]["retainedCells"] == 3
    assert results[2]["removedCells"] == 3
    grouped = {item["value"]: item for item in results[2]["byGroup"]["group"]["levels"]}
    assert grouped["a"]["retainedCells"] == 1
    assert grouped["b"]["retainedCells"] == 2
    assert results[2]["byGroup"]["group"]["missingInputCells"] == 1
    assert results[2]["byGroup"]["group"]["missingRetainedCells"] == 0
    pd.testing.assert_frame_equal(frame, original)
    no_manual = _qc_projections(store, active, "RNA", AnalysisConfig(), flags, [])
    assert not no_manual[2]["available"]
    assert no_manual[2]["retainedCells"] is None


def test_family_audit_distinguishes_registry_from_executed_exact_name_policy() -> None:
    names = np.array(
        ["MT-CO1", "mt-Co2", "RPL3", "HLA-DRA", "XIST", "G.3", "g.3", "Gx3"]
    )
    result = family_audit(names, _blacklist(["G.3"]))
    assert result["excludedFeatures"] == 3
    assert result["standardExcludedFeatures"] == 5
    assert result["standardOnlyExcludedFeatures"] == 3
    families = {row["family"]: row for row in result["families"]}
    assert families["mitochondrial"]["excludedFeatures"] == 2
    assert families["ribosomal"]["matchedFeatures"] == 1
    assert families["ribosomal"]["excludedFeatures"] == 0
    assert families["hla"]["excludedFeatures"] == 0


def test_inferred_unit_roles_remain_diagnostic_and_crosstabs_do_not_need_correction() -> (
    None
):
    supplied = study(sampleColumn="donor", protectedColumns=["donor", "disease"])
    roles = resolved_roles(supplied, {"library": "capture"})
    assert {row["role"] for row in roles if row["column"] == "donor"} == {
        "sample",
        "protected",
    }
    library = next(row for row in roles if row["column"] == "library")
    assert library == {
        "column": "library",
        "role": "capture",
        "source": "inferred",
        "authority": "diagnosticOnly",
        "evidenceIds": [],
    }
    metadata = Metadata(
        pd.DataFrame(
            {
                "donor": ["a", "a", "b", "b"],
                "disease": ["x", "x", "y", "y"],
                "library": ["l1", "l2", "l1", "l2"],
            }
        )
    )
    diagnostics = design_diagnostics(metadata, np.ones(4, dtype=bool), roles)
    donor_disease = next(
        row
        for row in diagnostics["crossTabs"]
        if {row["leftColumn"], row["rightColumn"]} == {"donor", "disease"}
    )
    assert not donor_disease["fullyCrossed"]
    assert sum(row["count"] for row in donor_disease["counts"]) == 4


class BoundedArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.reads: list[tuple[int, int]] = []

    def get_orthogonal_selection(self, item: tuple[Any, Any]) -> np.ndarray:
        rows, columns = item
        result = self.values[rows, columns]
        self.reads.append(result.shape)
        assert result.shape[0] <= 10_000
        assert result.shape[1] <= 30
        return result


def test_pc_associations_and_loading_families_use_only_bounded_frozen_evidence() -> (
    None
):
    class Run(dict[str, Any]):
        pass

    n = 12_001
    x = np.arange(n, dtype=float)
    group = np.where(x < n / 2, "a", "b")
    coordinates = BoundedArray(
        np.column_stack([x, (group == "a").astype(float), *[np.ones(n)] * 33])
    )
    metadata = Metadata(
        pd.DataFrame(
            {
                "ids": [f"cell{i}" for i in range(n)],
                "counts": x,
                "group": group,
                "constant": np.ones(n),
            }
        )
    )
    run = Run(pca="pca", highly_variable_features="hvg")
    run.cells = metadata
    run.features = Metadata(pd.DataFrame({"names": ["RPL3", "GENE", "MT-CO1"]}))
    loadings = np.zeros((2, 35))
    loadings[0] = 0.9
    loadings[1] = 0.1
    store = SimpleNamespace(
        load_artifact=lambda ref: (
            {"data": coordinates, "loadings": loadings}
            if ref == "pca"
            else {"values": np.array([True, True, False])}
        )
    )
    result = representation_diagnostics(
        store,
        run,
        {
            "diagnosticColumns": ["counts", "group", "constant"],
            "blacklist": _blacklist([]),
        },
        42,
    )
    assert coordinates.reads == [(10_000, 30)]
    assert result["actualHvgCount"] == 2
    assert len(result["loadingFamilies"]) == 10
    assert result["loadingFamilies"][0]["families"]["ribosomal"] == 1
    associations = {
        (row["column"], row["component"]): row
        for row in result["covariateAssociations"]
    }
    assert associations["counts", 1]["association"] == pytest.approx(1)
    assert associations["group", 2]["association"] == pytest.approx(1)
    assert associations["constant", 1]["association"] is None
    assert result["pcaDiagnosticScope"]["sampleCells"] == 10_000


def test_pc_associations_preserve_core_missing_statuses_and_reject_saturation() -> None:
    class Run(dict[str, Any]):
        pass

    run = Run(pca="pca", highly_variable_features="hvg")
    run.cells = Metadata(
        pd.DataFrame(
            {
                "ids": [f"cell{i}" for i in range(6)],
                "uniqueLabels": list("abcdef"),
                "counts": [0.0, 1.0, np.nan, 3.0, np.inf, 5.0],
                "partialGroups": ["a", "a", None, "b", "", "b"],
            }
        )
    )
    run.features = Metadata(pd.DataFrame({"names": ["G1", "G2"]}))
    store = SimpleNamespace(
        load_artifact=lambda ref: (
            {
                "data": np.arange(6, dtype=float).reshape(-1, 1),
                "loadings": np.array([[1.0], [0.1]]),
            }
            if ref == "pca"
            else {"values": np.ones(2, dtype=bool)}
        )
    )
    result = representation_diagnostics(
        store,
        run,
        {
            "diagnosticColumns": ["uniqueLabels", "counts", "partialGroups"],
            "blacklist": _blacklist([]),
        },
        42,
    )
    rows = {row["column"]: row for row in result["covariateAssociations"]}
    assert rows["uniqueLabels"]["association"] is None
    assert rows["uniqueLabels"]["status"] == "notComputed"
    assert rows["uniqueLabels"]["limitation"] == "saturatedCategories"
    assert rows["counts"]["rowsUsed"] == rows["partialGroups"]["rowsUsed"] == 4
    assert rows["counts"]["rowsMissing"] == rows["partialGroups"]["rowsMissing"] == 2
    assert rows["counts"]["association"] == pytest.approx(1)
    assert rows["counts"]["status"] == "ok"


def test_full_cohort_composition_preserves_rare_clusters_missingness_and_qc() -> None:
    n = 10_005
    labels = np.array(["major"] * (n - 1) + ["rare"])
    run = SimpleNamespace(
        cells=Metadata(
            pd.DataFrame(
                {
                    "donor": ["a"] * (n - 1) + ["b"],
                    "RNA_nCounts": [10.0] * (n - 1) + [np.nan],
                }
            )
        )
    )
    prepared = {
        "resolvedRoles": resolved_roles(study(), {"donor": "sample"}),
        "diagnosticColumns": ["donor", "RNA_nCounts"],
        "qcFlags": {"RNA_nCounts": {}},
    }
    result = finalist_composition(run, prepared, labels)
    assert result["rare"]["groupComposition"]["donor"]["levels"] == [
        {"value": "b", "count": 1, "fraction": 1.0}
    ]
    assert result["rare"]["qc"]["RNA_nCounts"]["missing"] == 1
    assert result["rare"]["qc"]["RNA_nCounts"]["median"] is None
    assert result["major"]["qc"]["RNA_nCounts"]["median"] == 10


def test_equal_resolution_comparison_handles_label_permutations_and_splits() -> None:
    selection = SimpleNamespace(to_dict=lambda: {"artifact_id": "same"})

    class Run(dict[str, Any]):
        status = "completed"
        run_id = "test-comparison"

        def __init__(self, labels: list[str]) -> None:
            super().__init__(
                {"analysis_cell_selection": selection, "leiden_0.5": "partition"}
            )
            self.cells = Metadata(
                pd.DataFrame({"ids": np.arange(8), "leiden_0.5": labels})
            )

    parent = Run(["a"] * 4 + ["b"] * 4)
    permuted = Run(["x"] * 4 + ["y"] * 4)
    baseline = Candidate(candidateId="c0", hvgCount=10, pcaDims=3, neighborsK=3)
    probe = Candidate(
        candidateId="c1", parentId="c0", hvgCount=20, pcaDims=3, neighborsK=3
    )
    config = AnalysisConfig(resolutions=(0.5,))
    comparison = compare_candidates(None, parent, permuted, baseline, probe, config)[0]
    assert comparison["adjustedRandIndex"] == 1
    assert comparison["selection"] == {"artifact_id": "same"}
    assert all(row["fraction"] == 1 for row in comparison["parentToCandidate"])
    split = Run(["x"] * 2 + ["z"] * 2 + ["y"] * 4)
    result = compare_candidates(None, parent, split, baseline, probe, config)[0]
    assert result["adjustedRandIndex"] < 1
    assert result["parentToCandidate"][0]["fraction"] == 0.5
    assert all(row["fraction"] == 1 for row in result["candidateToParent"])
    broad_config = AnalysisConfig(resolutions=(0.5, 0.75))
    with pytest.raises(AnalysisInputError, match="missing a matched resolution"):
        compare_candidates(None, parent, split, baseline, probe, broad_config)
    assert (
        len(
            compare_candidates(
                None,
                parent,
                split,
                baseline,
                probe,
                broad_config.model_copy(update={"resolutions": (0.5,)}),
            )
        )
        == 1
    )
    split["analysis_cell_selection"] = "different"
    with pytest.raises(AnalysisInputError, match="exact frozen cohort"):
        compare_candidates(None, parent, split, baseline, probe, config)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("unfinished", "require completed pipelines"),
        ("reordered", "cell ordering differs"),
        ("truncated", "partition lengths differ"),
    ],
)
def test_comparison_refuses_unfinished_reordered_or_misaligned_partitions(
    damage: str, message: str
) -> None:
    selection = SimpleNamespace(to_dict=lambda: {"artifact_id": "same"})

    class Run(dict[str, Any]):
        status = "completed"
        run_id = "test-comparison"

        def __init__(self, ids: np.ndarray, labels: np.ndarray) -> None:
            super().__init__(
                {"analysis_cell_selection": selection, "leiden_0.5": "partition"}
            )
            columns = {"ids": ids, "leiden_0.5": labels}
            self.cells = SimpleNamespace(fetch=columns.__getitem__)

    ids = np.arange(8)
    labels = np.repeat(["a", "b"], 4)
    parent = Run(ids, labels)
    if damage == "unfinished":
        probe = Run(ids, labels)
        probe.status = "failed"
    elif damage == "reordered":
        probe = Run(ids[::-1], labels)
    else:
        probe = Run(ids, labels[:-1])
    baseline = Candidate(candidateId="c0", hvgCount=10, pcaDims=3, neighborsK=3)
    candidate = baseline.model_copy(update={"candidateId": "c1", "parentId": "c0"})
    config = AnalysisConfig(resolutions=(0.5,))
    with pytest.raises(AnalysisInputError, match=message):
        compare_candidates(None, parent, probe, baseline, candidate, config)
    # The same comparison is accepted once the runs are aligned and complete.
    aligned = compare_candidates(
        None, parent, Run(ids, labels), baseline, candidate, config
    )
    assert aligned[0]["adjustedRandIndex"] == 1.0


@pytest.fixture
def diagnostic_artifacts() -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Public artifact views with independent cell and feature axes."""

    class Run(dict[str, Any]):
        pass

    run = Run(pca="pca", highly_variable_features="hvg")
    run.cells = Metadata(
        pd.DataFrame(
            {
                "ids": [f"cell{i}" for i in range(65)],
                "counts": np.arange(65, dtype=float),
            }
        )
    )
    run.features = Metadata(pd.DataFrame({"names": ["G1", "G2"]}))
    payloads = {
        "pca": {
            "data": np.arange(65, dtype=float).reshape(-1, 1),
            "loadings": np.array([[1.0], [0.1]]),
        },
        "hvg": {"values": np.ones(2, dtype=bool)},
    }
    store = SimpleNamespace(load_artifact=payloads.__getitem__)
    prepared = {"diagnosticColumns": ["counts"], "blacklist": _blacklist([])}
    return store, run, prepared, payloads


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("coordinate", "coordinates are non-finite"),
        ("feature-mask", "HVG selection does not align"),
        ("loading-axis", "loadings do not align"),
        ("loading-value", "loadings contain non-finite"),
    ],
)
def test_invalid_public_artifact_geometry_never_supplies_scientific_evidence(
    diagnostic_artifacts: tuple[Any, Any, dict[str, Any], dict[str, Any]],
    damage: str,
    message: str,
) -> None:
    store, run, prepared, payloads = diagnostic_artifacts
    if damage == "coordinate":
        payloads["pca"]["data"][10, 0] = np.inf
    elif damage == "feature-mask":
        payloads["hvg"]["values"] = np.ones(3, dtype=bool)
    elif damage == "loading-axis":
        payloads["pca"]["loadings"] = np.ones((3, 1))
    else:
        payloads["pca"]["loadings"][1, 0] = np.nan
    with pytest.raises(AnalysisInputError, match=message):
        representation_diagnostics(store, run, prepared, 42)


def test_diagnostic_metadata_budget_and_unique_capture_ids_are_explicit(
    diagnostic_artifacts: tuple[Any, Any, dict[str, Any], dict[str, Any]],
) -> None:
    store, run, prepared, _ = diagnostic_artifacts
    frame = run.cells.frame.copy()
    frame["capture"] = [f"capture{i}" for i in range(65)]
    columns = ["capture", *[f"covariate{i}" for i in range(49)]]
    for column in columns[1:]:
        frame[column] = np.arange(65, dtype=float)
    read_columns = []

    class TrackedMetadata(Metadata):
        def to_pandas_dataframe(self, columns: list[str]) -> pd.DataFrame:
            read_columns.extend(columns)
            return super().to_pandas_dataframe(columns)

    run.cells = TrackedMetadata(frame)
    prepared["diagnosticColumns"] = columns
    result = representation_diagnostics(store, run, prepared, 42)
    assert read_columns == columns[:48]
    assert result["pcaDiagnosticScope"]["columns"] == columns[:48]
    assert "capture" not in {row["column"] for row in result["covariateAssociations"]}
    assert any(
        "capture" in item and "64" in item
        for item in result["pcaDiagnosticScope"]["limitations"]
    )


def test_design_summary_bounds_many_pairs_and_preserves_confirmed_capture_authority() -> (
    None
):
    metadata = Metadata(
        pd.DataFrame(
            {
                "capture": [f"capture{i}" for i in range(65)],
                **{f"group{i}": np.arange(65) % 2 for i in range(7)},
            }
        )
    )
    roles = resolved_roles(
        study(captureColumn="capture", protectedColumns=[f"group{i}" for i in range(7)])
    )
    capture = next(row for row in roles if row["role"] == "capture")
    assert capture["source"] == "supplied"
    assert capture["authority"] == "confirmed"
    assert capture["evidenceIds"] == ["study:capture"]
    result = design_diagnostics(metadata, np.ones(65, dtype=bool), roles)
    assert len(result["crossTabs"]) == 24
    assert any("24 of 28" in item for item in result["limitations"])
    involving_capture = [
        row
        for row in result["crossTabs"]
        if "capture" in {row["leftColumn"], row["rightColumn"]}
    ]
    assert involving_capture
    assert all(
        not row["available"] and "64" in row["reason"] for row in involving_capture
    )
    assert all(row["rowsUsed"] == 65 for row in result["crossTabs"])


def test_nonfinite_upstream_association_is_missing_evidence_and_finite_json(
    diagnostic_artifacts: tuple[Any, Any, dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, run, prepared, _ = diagnostic_artifacts
    # A provider-independent numerical boundary failure must not leak Infinity
    # into immutable JSON or masquerade as a strong scientific association.
    monkeypatch.setattr(
        "scarf.metrics.spearman_rho",
        lambda *args: {
            "status": "ok",
            "value": float("inf"),
            "rowsUsed": 65,
            "rowsMissing": 0,
        },
    )
    result = representation_diagnostics(store, run, prepared, 42)
    row = result["covariateAssociations"][0]
    assert row["association"] is None
    assert row["status"] == "notComputed"
    assert row["limitation"] == "nonfiniteAssociation"
    json.dumps(result, allow_nan=False)


def test_finalist_composition_refuses_a_different_metadata_cohort() -> None:
    run = SimpleNamespace(cells=Metadata(pd.DataFrame({"donor": ["a", "b", "a"]})))
    prepared = {
        "resolvedRoles": resolved_roles(study(sampleColumn="donor")),
        "diagnosticColumns": ["donor"],
    }
    with pytest.raises(AnalysisInputError, match="metadata does not align"):
        finalist_composition(run, prepared, np.array(["cluster1", "cluster2"]))
