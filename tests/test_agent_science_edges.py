"""Scientific input failures and incomplete diagnostics stay explicit and bounded."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf.agent import evidence, execution
from scarf.agent.models import (
    AnalysisConfig,
    AnalysisInputError,
    Candidate,
    ContextDecision,
    NeedsInput,
    RuntimeConfig,
)
from tests.test_agent_evidence import study
from tests.test_agent_execution import _matched


class Metadata:
    """The public, nullable metadata interface without a count-matrix reader."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    @property
    def columns(self) -> list[str]:
        return self.frame.columns.tolist()

    def get_dtype(self, name: str) -> Any:
        return self.frame[name].dtype

    def to_pandas_dataframe(self, columns: list[str]) -> pd.DataFrame:
        return self.frame[columns].copy()

    def fetch_all(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy()

    fetch = fetch_all


@pytest.fixture
def metadata_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Any]:
    """Isolate metadata policy from the already-covered real-store I/O path."""
    import scarf.assay
    from zarr.storage import LocalStore

    class PreparedRna:
        feats = Metadata(
            pd.DataFrame(
                {
                    "ids": ["a", "b", "c", "d"],
                    "names": ["MT-CO1", "HLA-DRA", "G.3", "G4"],
                    "I": [True] * 4,
                }
            )
        )
        matrixGroup = SimpleNamespace(store=LocalStore(tmp_path))

    monkeypatch.setattr(scarf.assay, "RNAassay", PreparedRna)
    assay = PreparedRna()
    cells = Metadata(
        pd.DataFrame(
            {
                "ids": [f"cell{i}" for i in range(8)],
                "names": [f"cell{i}" for i in range(8)],
                "I": [True] * 8,
                "batch": ["a", "b"] * 4,
                "condition": ["control"] * 4 + ["treated"] * 4,
                "RNA_nCounts": [10, 20, 30, 40, 50, 60, 70, 80],
                "RNA_nFeatures": [4] * 8,
            }
        )
    )
    descriptor = SimpleNamespace(
        name="RNA", dataset_fingerprint="finalized-counts", total_features=4
    )
    store = SimpleNamespace(
        assay_names=["RNA"],
        get_assay=lambda name: assay,
        list_artifacts=lambda **kwargs: [],
        cells=cells,
        summary=lambda: SimpleNamespace(
            default_assay="RNA", assays=[descriptor], total_cells=8
        ),
    )
    monkeypatch.setattr(evidence, "open_store", lambda *args, **kwargs: store)
    return tmp_path, store


def test_manual_qc_uses_exact_inclusive_bounds_and_preserves_live_selection() -> None:
    frame = pd.DataFrame(
        {"I": [True] * 6, "RNA_nCounts": [9.0, 10.0, 15.0, 20.0, 21.0, np.nan]}
    )
    before = frame.copy(deep=True)
    config = AnalysisConfig(qcPolicy="manual", qcBounds={"RNA_nCounts": (10, 20)})
    options, retained, warnings, flags = evidence._filtering(
        SimpleNamespace(cells=Metadata(frame)), study(), config, "RNA"
    )
    assert retained.tolist() == [False, True, True, True, False, False]
    assert options == {
        "method": "manual",
        "attrs": ["RNA_nCounts"],
        "lows": [10.0],
        "highs": [20.0],
        "keep_bounds": True,
    }
    assert flags["RNA_nCounts"]["outlierRows"]["missing"] == [5]
    assert any("biological populations" in warning for warning in warnings)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    ("selection", "config", "message", "error"),
    [
        ([1, 1, 0], AnalysisConfig(), "complete boolean", AnalysisInputError),
        ([True, None, False], AnalysisConfig(), "complete boolean", AnalysisInputError),
        ([False] * 3, AnalysisConfig(), "no cells", NeedsInput),
        (
            [True] * 3,
            AnalysisConfig(qcPolicy="gentleMad5"),
            "metrics are absent",
            NeedsInput,
        ),
        (
            [True] * 3,
            AnalysisConfig(qcPolicy="manual", qcBounds={"RNA_missing": (0, 10)}),
            "QC columns are absent",
            NeedsInput,
        ),
    ],
)
def test_invalid_qc_inputs_do_not_silently_choose_a_cohort(
    selection: list[Any], config: AnalysisConfig, message: str, error: type[Exception]
) -> None:
    store = SimpleNamespace(cells=Metadata(pd.DataFrame({"I": selection})))
    with pytest.raises(error, match=message):
        evidence._filtering(store, study(), config, "RNA")


def test_missing_metrics_and_zero_mad_are_flags_without_invented_bounds() -> None:
    active = np.array([True, True, True, False])
    store = SimpleNamespace(
        cells=Metadata(
            pd.DataFrame(
                {
                    "RNA_nCounts": [np.nan, np.inf, -1, 3],
                    "RNA_nFeatures": [8, 8, 8, 20],
                }
            )
        )
    )
    flags = evidence._qc_flags(store, active, "RNA")
    assert flags["RNA_nCounts"]["missing"] == 3
    assert flags["RNA_nCounts"]["outlierRows"]["missing"] == [0, 1, 2]
    assert flags["RNA_nCounts"]["low"] is flags["RNA_nCounts"]["high"] is None
    assert flags["RNA_nFeatures"]["zeroMad"]
    assert flags["RNA_nFeatures"]["lowFlags"] == 0
    assert flags["RNA_nFeatures"]["highFlags"] == 0


_NOT_CROSSED = "Technical 'batch' is not fully crossed with protected 'condition'."
_LEVELS = "Correction metadata 'batch' needs between 2 and 64 categorical levels."


@pytest.mark.parametrize(
    ("damage", "reasons"),
    [
        (None, []),
        (
            "authorization",
            ["Technical batch correction lacks caller-supplied experimental evidence."],
        ),
        (
            "protected",
            [
                "Correction has no explicitly protected biological metadata for validation."
            ],
        ),
        (
            "missing",
            [
                "Correction metadata 'batch' contains missing labels.",
                "Correction metadata 'condition' contains missing labels.",
                _NOT_CROSSED,
            ],
        ),
        ("one", [_LEVELS]),
        ("many", [_LEVELS, _NOT_CROSSED]),
    ],
)
def test_uncertain_experimental_design_never_authorizes_correction(
    damage: str | None, reasons: list[str]
) -> None:
    frame = pd.DataFrame(
        {"batch": ["a", "b"] * 34, "condition": ["x"] * 34 + ["y"] * 34}
    )
    authorization: str | None = "The two library preparations contain both conditions."
    protected = ["condition"]
    if damage == "authorization":
        authorization = None
    elif damage == "protected":
        protected = []
    elif damage == "missing":
        frame.loc[0, "batch"] = " "
        frame.loc[1, "condition"] = None
    elif damage == "one":
        frame["batch"] = "one"
    elif damage == "many":
        frame["batch"] = [f"batch{i}" for i in range(len(frame))]
    eligible, design, limitations = evidence._design(
        SimpleNamespace(cells=Metadata(frame)),
        np.ones(len(frame), dtype=bool),
        ["batch"],
        protected,
        authorization,
    )
    # Only the undamaged, authorized and crossed design permits correction.
    assert eligible is (damage is None)
    assert limitations == reasons
    assert design["limitations"] == limitations


@pytest.mark.parametrize("damage", ["missing", "nonutf8", "budget"])
def test_local_references_reject_unavailable_or_unbounded_evidence(
    tmp_path: Path, damage: str
) -> None:
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    if damage == "nonutf8":
        first.write_bytes(b"\xff\xfe")
        files = [str(first)]
        message = "UTF-8"
    elif damage == "budget":
        first.write_text("a" * 8192)
        second.write_text("b" * 8193)
        files = [str(first), str(second)]
        message = "16 KiB"
    else:
        files = [str(first)]
        message = "unavailable"
    with pytest.raises(NeedsInput, match=message):
        evidence._references(study(referenceFiles=files))


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("assay", "Choose one RNA assay"),
        ("column", "Declared source columns"),
        ("cells", "At least four"),
        ("exclusion", "Exact feature exclusions"),
        ("features", "At least three features"),
        ("identity", "finalized dataset fingerprint"),
    ],
)
def test_prepared_source_failures_report_the_missing_scientific_requirement(
    metadata_source: tuple[Path, Any], damage: str, message: str
) -> None:
    path, store = metadata_source
    supplied = study()
    config = AnalysisConfig()
    if damage == "assay":
        config = AnalysisConfig(assay="ATAC")
    elif damage == "column":
        supplied = study(captureColumn="missing_capture")
    elif damage == "cells":
        store.cells.frame.loc[3:, "I"] = False
    elif damage == "exclusion":
        supplied = study(featureExclusions=["invented"])
    elif damage == "features":
        supplied = study(featureExclusions=["G.3"])
    else:
        store.summary().assays[0].dataset_fingerprint = None
    with pytest.raises(ValueError, match=message):
        evidence.inspect_source(path, supplied, config, RuntimeConfig())


def test_context_metadata_cap_preserves_declared_roles_and_reports_remote_reads(
    metadata_source: tuple[Path, Any],
) -> None:
    path, store = metadata_source
    for i in range(60):
        store.cells.frame[f"extra{i}"] = i
    store.get_assay("RNA").matrixGroup = SimpleNamespace(store=object())
    prepared = evidence.inspect_source(
        path, study(sampleColumn="extra59"), AnalysisConfig(), RuntimeConfig()
    )
    assert len(prepared["contextEvidence"]["columns"]) == 48
    assert "extra59" in prepared["contextEvidence"]["columns"]
    assert any("48 of" in item for item in prepared["limitations"])
    assert any("remote count blocks" in item for item in prepared["limitations"])
    assert any("RNA_percentMito" in item for item in prepared["limitations"])


def test_fingerprinting_handles_nullable_bytes_and_nonfinite_metadata() -> None:
    assert evidence._json_scalar(np.int64(4)) == 4
    assert evidence._json_scalar(b"batch-a") == "batch-a"
    assert evidence._json_scalar(np.nan) is None
    assert evidence._json_scalar(float("inf")) == {"nonfinite": "inf"}
    assert evidence._json_scalar(pd.Timestamp("2024-01-01")) == "2024-01-01 00:00:00"


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        ({"missing": "protected"}, "offered non-held-out"),
        ({"condition": "technical"}, "new technical batch"),
        ({"batch": "sample"}, "change a supplied technical"),
        ({"condition": "ignore"}, "remove a supplied protected"),
    ],
)
def test_context_cannot_reassign_authoritative_roles(
    metadata_source: tuple[Path, Any], roles: dict[str, Any], message: str
) -> None:
    path, _ = metadata_source
    supplied = study(technicalBatchColumns=["batch"], protectedColumns=["condition"])
    prepared = evidence.inspect_source(
        path, supplied, AnalysisConfig(), RuntimeConfig()
    )
    with pytest.raises(AnalysisInputError, match=message):
        evidence.prepare_context(
            prepared,
            ContextDecision(rationale="Suggested metadata roles", columnRoles=roles),
            supplied,
            AnalysisConfig(),
        )


def test_context_adds_protection_but_rechecks_source_identity(
    metadata_source: tuple[Path, Any],
) -> None:
    path, store = metadata_source
    supplied = study()
    prepared = evidence.inspect_source(
        path, supplied, AnalysisConfig(), RuntimeConfig()
    )
    decision = ContextDecision(
        rationale="Preserve measured condition", columnRoles={"condition": "protected"}
    )
    result = evidence.prepare_context(prepared, decision, supplied, AnalysisConfig())
    assert result["protectedColumns"] == ["condition"]
    store.cells.frame.loc[0, "condition"] = "changed"
    with pytest.raises(AnalysisInputError, match="changed while resolving context"):
        evidence.prepare_context(prepared, decision, supplied, AnalysisConfig())
    store.cells.frame = store.cells.frame.drop(columns="condition")
    with pytest.raises(AnalysisInputError, match="metadata columns changed"):
        evidence.verify_source(
            path, prepared, supplied, AnalysisConfig(), RuntimeConfig()
        )


def test_absent_qc_metrics_offer_only_the_retained_projection(
    metadata_source: tuple[Path, Any],
) -> None:
    path, store = metadata_source
    store.cells.frame = store.cells.frame.drop(columns=["RNA_nCounts", "RNA_nFeatures"])
    prepared = evidence.inspect_source(path, study(), AnalysisConfig(), RuntimeConfig())
    assert prepared["qcFlags"] == {}
    projections = {row["policy"]: row for row in prepared["qcProjections"]}
    assert projections["retain"]["available"] is True
    assert projections["retain"]["retainedCells"] == 8
    assert projections["gentleMad5"] == {
        "policy": "gentleMad5",
        "available": False,
        "executed": False,
        "inputCells": 8,
        "retainedCells": None,
        "removedCells": None,
        "bounds": {},
        "byGroup": {},
        "limitations": ["No available QC metrics support this projection."],
    }
    assert projections["manual"]["limitations"] == [
        "No manual thresholds were supplied; none are invented."
    ]
    assert (
        "Optional QC outlier flags are unavailable for missing columns: "
        "['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito']."
    ) in prepared["limitations"]


@pytest.mark.parametrize("role", ["sample", "capture"])
def test_context_cannot_replace_a_supplied_sample_or_capture_column(
    metadata_source: tuple[Path, Any], role: str
) -> None:
    path, _ = metadata_source
    supplied = study(**{f"{role}Column": "batch"})
    prepared = evidence.inspect_source(
        path, supplied, AnalysisConfig(), RuntimeConfig()
    )
    with pytest.raises(
        AnalysisInputError, match="cannot replace a supplied sample or capture column"
    ):
        evidence.prepare_context(
            prepared,
            ContextDecision(
                rationale="Suggested experimental unit",
                columnRoles={"condition": role},
            ),
            supplied,
            AnalysisConfig(),
        )
    # Confirming the supplied column itself is not a replacement.
    confirmed = evidence.prepare_context(
        prepared,
        ContextDecision(rationale="Supplied unit", columnRoles={"batch": role}),
        supplied,
        AnalysisConfig(),
    )
    assert confirmed[f"{role}Column"] == "batch"


@pytest.mark.parametrize("invalid", [np.nan, -1])
def test_writable_open_refuses_invalid_live_feature_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: float
) -> None:
    import scarf

    opened: list[str] = []
    store = SimpleNamespace(
        summary=lambda: SimpleNamespace(default_assay="RNA"),
        cells=Metadata(
            pd.DataFrame({"I": [True, False], "RNA_nFeatures": [invalid, 4]})
        ),
    )

    def datastore(*args: Any, **kwargs: Any) -> Any:
        opened.append(kwargs["zarr_mode"])
        return store

    monkeypatch.setattr(scarf, "DataStore", datastore)
    with pytest.raises(AnalysisInputError, match="invalid feature counts"):
        evidence.open_store(tmp_path, AnalysisConfig(), RuntimeConfig(), writable=True)
    assert opened == ["r"]


@pytest.mark.parametrize("path", ["s3://remote/counts", "/a/nonexistent/local/source"])
def test_sources_must_already_be_local_directories(path: str) -> None:
    with pytest.raises(AnalysisInputError, match="existing local Scarf directory"):
        evidence._local_source(path)


class Reference:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_dict(self) -> dict[str, str]:
        return {"artifact_id": self.name}


class Finalist(dict[str, Any]):
    status = "completed"
    run_id = "frozen-finalist"

    def __init__(self) -> None:
        clustering = Reference("clusters")
        super().__init__(
            {
                "markers": Reference("markers"),
                "pca": Reference("pca"),
                "harmony": Reference("harmony"),
                "analysis_cell_selection": Reference("cohort"),
                "highly_variable_features": Reference("features"),
                "clusters": clustering,
                "leiden_0.5": clustering,
            }
        )
        self.cells = Metadata(
            pd.DataFrame(
                {
                    "ids": [f"cell{i}" for i in range(8)],
                    "clusters": ["a"] * 4 + ["b"] * 4,
                    "batch": ["x", "y"] * 4,
                    "condition": ["normal"] * 4 + ["treated"] * 4,
                    "sample": ["s1", "s2"] * 4,
                    "doublet_score": np.arange(8) / 100,
                }
            )
        )


@pytest.fixture
def finalist_setup() -> tuple[Any, Finalist, Candidate, dict[str, Any], AnalysisConfig]:
    frame = pd.DataFrame(
        {
            "feature_name": ["SUPPORTED", "WEAK", "ABSENT"],
            "score": [0.8, 0.1, 0.9],
            "frac_exp": [0.8, 0.5, 0.0],
            "frac_exp_rest": [0.1, 0.4, 0.0],
        }
    )
    store = SimpleNamespace(get_markers=lambda *args, **kwargs: frame.copy())
    prepared = {
        "technicalBatchColumns": ["batch"],
        "protectedColumns": ["condition"],
        "sampleColumn": "sample",
    }
    return (
        store,
        Finalist(),
        Candidate(candidateId="c0", hvgCount=20, pcaDims=3, neighborsK=5),
        prepared,
        AnalysisConfig(),
    )


def test_finalist_quantifies_mixing_support_and_doublets_from_frozen_rows(
    finalist_setup: tuple[Any, Finalist, Candidate, dict[str, Any], AnalysisConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scarf.metrics

    store, run, candidate, prepared, config = finalist_setup
    graph = tuple(np.zeros((8, 3)) for _ in range(4))
    monkeypatch.setattr(execution, "_diagnostic_graph", lambda *args: graph)

    def lisi(
        distances: Any, indices: Any, labels: pd.DataFrame, columns: list[str]
    ) -> Any:
        assert columns == ["batch"]
        assert labels["batch"].tolist() == ["x", "y"] * 4
        return np.full((8, 1), 1.5)

    monkeypatch.setattr(scarf.metrics, "compute_lisi", lisi)
    monkeypatch.setattr(
        scarf.metrics,
        "lisi_batch_mixing_score",
        lambda values, labels: float(values.mean() - 1),
    )
    monkeypatch.setattr(scarf.metrics, "clisi_knn", lambda *args, **kwargs: 0.9)
    monkeypatch.setattr(
        scarf.metrics, "graph_connectivity", lambda *args, **kwargs: 0.8
    )
    result = execution.finalist_evidence(store, run, candidate, prepared, config)
    assert result["metrics"] == {
        "mixing": {"batch": 0.5},
        "protection": {"condition": {"cLISI": 0.9, "graphConnectivity": 0.8}},
        "markerSupportFraction": 1.0,
        "markerSpecificityMedian": 0.8,
        "doubletHighScoreConcentration": 2.0,
        "crossUnitSupport": 1.0,
    }
    assert result["parameters"]["resolution"] == 0.5
    assert all(cluster["sampleCount"] == 2 for cluster in result["clusters"])
    assert [cluster["doubletScoreMedian"] for cluster in result["clusters"]] == [
        0.015,
        0.055,
    ]
    assert all(cluster["qualifyingMarkerCount"] == 1 for cluster in result["clusters"])
    assert [row["gene"] for row in result["clusters"][0]["weakMarkers"]] == [
        "WEAK",
        "ABSENT",
    ]


@pytest.mark.parametrize("failure", ["graph", "missing-labels", "omitted-level"])
def test_incomplete_diagnostics_are_missing_evidence_not_correction_support(
    finalist_setup: tuple[Any, Finalist, Candidate, dict[str, Any], AnalysisConfig],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    store, run, candidate, prepared, config = finalist_setup
    if failure == "graph":

        def unavailable(*args: Any) -> Any:
            raise RuntimeError("Coordinate graph unavailable")

        monkeypatch.setattr(execution, "_diagnostic_graph", unavailable)
    else:
        monkeypatch.setattr(
            execution,
            "_diagnostic_graph",
            lambda *args: tuple(np.zeros((4, 3)) for _ in range(4)),
        )
        if failure == "missing-labels":
            run.cells.frame.loc[0, ["batch", "condition"]] = None
        else:
            monkeypatch.setattr(
                execution, "_diagnostic_indices", lambda *args: np.array([0, 2])
            )
    result = execution.finalist_evidence(store, run, candidate, prepared, config)
    assert result["metrics"]["mixing"]["batch"] is None
    assert result["metrics"]["protection"]["condition"] == {
        "cLISI": None,
        "graphConnectivity": None,
    }
    assert any("unavailable" in item for item in result["limitations"])
    if failure == "omitted-level":
        assert result["diagnosticScope"]["sampleCells"] == 2
        assert any("rare populations" in item for item in result["limitations"])
        assert result["clusters"][1]["doubletScoreMedian"] is None
        # Sample support remains full-cohort even when the coordinate sample
        # misses an entire cluster.
        assert all(cluster["sampleCount"] == 2 for cluster in result["clusters"])


@pytest.mark.parametrize("failure", ["missing", "blank", "single", "length"])
def test_sample_support_rejects_missing_or_unaligned_sample_identity(
    finalist_setup: tuple[Any, Finalist, Candidate, dict[str, Any], AnalysisConfig],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    store, run, candidate, prepared, config = finalist_setup
    prepared["technicalBatchColumns"] = []
    prepared["protectedColumns"] = []
    if failure in {"missing", "blank"}:
        run.cells.frame.loc[0, "sample"] = None if failure == "missing" else " "
    elif failure == "single":
        run.cells.frame["sample"] = "s1"
    else:
        original = run.cells.to_pandas_dataframe
        monkeypatch.setattr(
            run.cells,
            "to_pandas_dataframe",
            lambda columns: original(columns).iloc[:-1],
        )
        with pytest.raises(AnalysisInputError, match="do not align"):
            execution.finalist_evidence(store, run, candidate, prepared, config)
        return
    result = execution.finalist_evidence(store, run, candidate, prepared, config)
    assert result["metrics"]["crossUnitSupport"] is None
    assert result["requiredSampleSupport"]
    assert any(
        "Cross-unit support is unavailable" in item for item in result["limitations"]
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("native", "one native and one Harmony"),
        ("batch", "technical batch mixing is missing"),
        ("batch-value", "Batch mixing evidence is unavailable"),
        ("protection", "biological protection evidence is missing"),
        ("protection-value", "Biological protection cLISI is unavailable"),
        ("coherence", "Matched markerSupportFraction evidence is unavailable"),
        ("specificity", "Matched marker specificity evidence is unavailable"),
        ("specificity-harm", "Marker specificity worsened"),
    ],
)
def test_correction_gate_rejects_missing_or_wrong_direction_metrics(
    damage: str, message: str
) -> None:
    native, corrected = _matched()
    if damage == "native":
        native["parameters"]["useHarmony"] = True
    elif damage == "batch":
        corrected["metrics"]["mixing"] = {}
    elif damage == "batch-value":
        corrected["metrics"]["mixing"]["batch"] = np.nan
    elif damage == "protection":
        corrected["metrics"]["protection"] = {}
    elif damage == "protection-value":
        corrected["metrics"]["protection"]["condition"]["cLISI"] = None
    elif damage == "coherence":
        corrected["metrics"]["markerSupportFraction"] = None
    elif damage == "specificity":
        corrected["metrics"]["markerSpecificityMedian"] = None
    else:
        corrected["metrics"]["markerSpecificityMedian"] = 0.6
    assert any(
        message in reason for reason in execution.validate_harmony(native, corrected)
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("harmony", "Harmony requires an authorized, crossed experimental design"),
        ("resolution", "The selected resolution is outside the configured panel"),
        ("pca", "PCA dimensions must be smaller than selected cells and features"),
        ("neighbors", "Neighbor count must be smaller than the retained cohort"),
    ],
)
def test_impossible_pipeline_recipes_fail_before_numerical_work(
    damage: str, message: str
) -> None:
    candidate = Candidate(candidateId="c0", hvgCount=20, pcaDims=3, neighborsK=5)
    prepared = {
        "correctionEligible": False,
        "availableFeatures": 20,
        "retainedCells": 8,
    }
    resolution = None
    if damage == "harmony":
        candidate = candidate.model_copy(update={"useHarmony": True})
    elif damage == "resolution":
        resolution = 9.0
    elif damage == "pca":
        candidate = candidate.model_copy(update={"pcaDims": 8})
    else:
        candidate = candidate.model_copy(update={"neighborsK": 8})
    # The store is None: any numerical call would fail with another error.
    with pytest.raises(AnalysisInputError, match=message):
        execution.execute_pipeline(
            None,
            prepared,
            candidate,
            AnalysisConfig(),
            label="rejected",
            resolution=resolution,
        )


def test_incomplete_pipeline_runs_cannot_supply_scientific_evidence(
    finalist_setup: tuple[Any, Finalist, Candidate, dict[str, Any], AnalysisConfig],
) -> None:
    store, run, candidate, prepared, config = finalist_setup
    run.status = "failed"
    with pytest.raises(AnalysisInputError, match="Only completed"):
        execution.summarize_candidate(store, run, candidate, prepared, config)
    with pytest.raises(AnalysisInputError, match="completed run with markers"):
        execution.finalist_evidence(store, run, candidate, prepared, config)
    run.status = "completed"
    run.pop("markers")
    with pytest.raises(AnalysisInputError, match="completed run with markers"):
        execution.finalist_evidence(store, run, candidate, prepared, config)


@pytest.mark.parametrize("damage", ["coordinates", "cohort"])
def test_correction_graph_rejects_invalid_coordinates_or_insufficient_neighbors(
    damage: str,
) -> None:
    coordinates = np.zeros((5, 3))
    rows = np.arange(5)
    if damage == "coordinates":
        coordinates[0, 0] = np.nan
        message = "non-finite"
    else:
        rows = np.arange(3)
        message = "at least three neighbors"
    store = SimpleNamespace(
        load_artifact=lambda ref: {
            "data": SimpleNamespace(
                get_orthogonal_selection=lambda item: coordinates[item]
            )
        }
    )
    with pytest.raises(AnalysisInputError, match=message):
        execution._diagnostic_graph(
            store,
            Finalist(),
            Candidate(candidateId="c0", hvgCount=20, pcaDims=3, neighborsK=5),
            rows,
            4444,
        )


def test_unexpected_silhouette_failures_are_not_reported_as_valid_unscoreable_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.metrics import cluster_selection

    def selector(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("Corrupt coordinate input")

    monkeypatch.setattr(cluster_selection, "select_clusters_by_silhouette", selector)
    store = SimpleNamespace(
        memoryBytes=1024**2,
        load_artifact=lambda ref: {"data": np.zeros((8, 3)), "values": np.zeros(8)},
    )
    with pytest.raises(ValueError, match="Corrupt coordinate"):
        execution.summarize_candidate(
            store,
            Finalist(),
            Candidate(candidateId="c0", hvgCount=20, pcaDims=3, neighborsK=5),
            {},
            AnalysisConfig(),
        )
