"""Source binding, holdouts and conservative preprocessing use real stores."""

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scarf import DataStore
from scarf.agent.evidence import (
    inspect_source,
    open_store,
    prepare_context,
    verify_source,
)
from scarf.agent.models import (
    AnalysisConfig,
    ContextDecision,
    NeedsInput,
    RuntimeConfig,
    Study,
)
from scarf.writers import SparseToZarr


def _counts(*, high_count_outlier: bool = False) -> np.ndarray:
    rng = np.random.default_rng(87)
    matrix = rng.poisson(1.5, (48, 40)).astype(np.uint32)
    matrix[:24, :8] += 8
    matrix[24:, 8:16] += 8
    if high_count_outlier:
        matrix[-1] *= 1000
    return matrix


def make_source(path: Path, *, high_count_outlier: bool = False) -> Any:
    matrix = _counts(high_count_outlier=high_count_outlier)
    genes = ["MT-CO1", "HLA-DRA", "RPL3", "G.3", *[f"G{i}" for i in range(36)]]
    writer = SparseToZarr(
        csr_matrix(matrix),
        str(path),
        [f"cell{i}" for i in range(48)],
        genes,
        nthreads=2,
        mem_budget="256M",
    )
    writer.dump()
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=2,
        mem_budget="256M",
    )
    store.cells.insert("batch", np.tile(["b1", "b2"], 24))
    store.cells.insert("condition", np.repeat(["healthy", "disease"], 24))
    store.cells.insert("sample", np.tile(["s1", "s2", "s3", "s4"], 12))
    store.cells.insert("cell.type.fine", np.repeat(["SECRET_A", "SECRET_B"], 24))
    return store


@pytest.fixture(scope="module")
def source_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("evidence") / "source.zarr"
    make_source(path)
    return path


@pytest.fixture
def source(source_template: Path, tmp_path: Path) -> Path:
    """A private copy of the prepared store; several checks edit it."""
    path = tmp_path / "source.zarr"
    shutil.copytree(source_template, path)
    return path


def study(**kwargs: Any) -> Study:
    return Study(
        context="Human blood RNA counts", objective="Describe populations", **kwargs
    )


def runtime() -> RuntimeConfig:
    return RuntimeConfig(nthreads=2, memBudget="256M")


def _files(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_inspection_and_open_do_not_mutate_source_or_expose_labels(
    source: Path,
) -> None:
    before = _files(source)
    prepared = inspect_source(source, study(), AnalysisConfig(), runtime())
    open_store(source, AnalysisConfig(), runtime(), writable=True)
    assert _files(source) == before
    assert prepared["inputCells"] == prepared["retainedCells"] == 48
    assert prepared["filtering"] is False
    assert "cell.type.fine" in prepared["excludedColumns"]
    assert "SECRET" not in repr(prepared["contextEvidence"])
    assert not prepared["correctionEligible"]
    # Advisory flags use five scaled MADs of log1p total counts.
    totals = _counts().sum(axis=1).astype(float)
    work = np.log1p(totals)
    median = np.median(work)
    deviation = 1.4826 * np.median(np.abs(work - median))
    low = max(0.0, float(np.expm1(median - 5 * deviation)))
    high = float(np.expm1(median + 5 * deviation))
    flags = prepared["qcFlags"]["RNA_nCounts"]
    assert flags["low"] == pytest.approx(low)
    assert flags["high"] == pytest.approx(high)
    assert flags["lowFlags"] == int((totals < low).sum()) == 0
    assert flags["highFlags"] == int((totals > high).sum()) == 0
    assert (flags["missing"], flags["zeroMad"]) == (0, False)
    assert prepared["qcOutliers"]["RNA_nCounts"] == {
        "missing": [],
        "low": [],
        "high": [],
    }


def test_fingerprint_survives_relocation_and_rejects_metadata_edits(
    source: Path, tmp_path: Path
) -> None:
    prepared = inspect_source(source, study(), AnalysisConfig(), runtime())
    moved = tmp_path / "moved.zarr"
    shutil.copytree(source, moved)
    verify_source(moved, prepared, study(), AnalysisConfig(), runtime())
    store = open_store(moved, AnalysisConfig(), runtime(), writable=True)
    changed = store.cells.fetch_all("condition").astype(object)
    changed[0] = "changed"
    store.cells.insert("condition", changed, overwrite=True)
    with pytest.raises(ValueError, match="fingerprint changed"):
        verify_source(moved, prepared, study(), AnalysisConfig(), runtime())


def test_fingerprint_tracks_missing_masks(source: Path) -> None:
    prepared = inspect_source(source, study(), AnalysisConfig(), runtime())
    store = open_store(source, AnalysisConfig(), runtime(), writable=True)
    # Fixture-only construction of a nullable imported metadata column.
    group = store.zw["cellData"]
    mask = np.zeros(48, dtype=bool)
    mask[0] = True
    group.create_array("__scarf_missing__condition", data=mask)
    group["condition"].attrs["missing_mask"] = "__scarf_missing__condition"
    with pytest.raises(ValueError, match="fingerprint changed"):
        verify_source(source, prepared, study(), AnalysisConfig(), runtime())


def test_blacklist_excludes_exact_supplied_features_without_biological_families(
    source: Path,
) -> None:
    supplied = study(featureExclusions=["G.3"])
    prepared = inspect_source(source, supplied, AnalysisConfig(), runtime())
    pattern = prepared["blacklist"]
    assert re.search(pattern, "MT-CO1")
    assert re.search(pattern, "mt-Co1")
    assert re.search(pattern, "G.3")
    assert not any(
        re.search(pattern, gene)
        for gene in ["Gx3", "g.3", "HLA-DRA", "RPL3", "XIST", "CCND1"]
    )
    with pytest.raises(ValueError, match="offered"):
        prepare_context(
            prepared,
            ContextDecision(rationale="remove", excludeFeatures=["HLA-DRA"]),
            supplied,
            AnalysisConfig(),
        )


def test_declared_crossed_design_and_sample_metadata_are_frozen_batch_first(
    source: Path,
) -> None:
    supplied = study(
        technicalBatchColumns=["batch"],
        protectedColumns=["condition"],
        sampleColumn="sample",
        batchCorrectionEvidence="Two independently prepared technical libraries contain both conditions.",
    )
    prepared = inspect_source(source, supplied, AnalysisConfig(), runtime())
    assert prepared["correctionEligible"]
    assert prepared["snapshotColumns"][:3] == ["batch", "condition", "sample"]
    assert set(prepared["diagnosticColumns"]) <= set(prepared["snapshotColumns"])
    result = prepare_context(
        prepared,
        ContextDecision(
            rationale="Retain supplied facts", columnRoles={"condition": "protected"}
        ),
        supplied,
        AnalysisConfig(),
    )
    assert result["fingerprint"] == prepared["fingerprint"]
    assert result["correctionEligible"]


def test_confounded_design_cannot_authorize_harmony(source: Path) -> None:
    store = open_store(source, AnalysisConfig(), runtime(), writable=True)
    store.cells.insert("batch", store.cells.fetch_all("condition"), overwrite=True)
    supplied = study(
        technicalBatchColumns=["batch"],
        protectedColumns=["condition"],
        batchCorrectionEvidence="Imported technical batch",
    )
    prepared = inspect_source(source, supplied, AnalysisConfig(), runtime())
    assert not prepared["correctionEligible"]
    assert any("not fully crossed" in value for value in prepared["limitations"])


def test_inferred_sample_identity_is_frozen_for_diagnostics_without_authority(
    source: Path,
) -> None:
    supplied = study(protectedColumns=["condition"])
    config = AnalysisConfig()
    inspected = inspect_source(source, supplied, config, runtime())
    prepared = prepare_context(
        inspected,
        ContextDecision(
            rationale="The available sample labels support descriptive comparisons.",
            columnRoles={"sample": "sample", "condition": "protected"},
            evidenceIds=["column:sample", "column:condition"],
        ),
        supplied,
        config,
        runtime(),
    )
    assert supplied.sampleColumn is None
    assert prepared["sampleColumn"] is None
    assert "sample" in prepared["snapshotColumns"]
    role = next(row for row in prepared["resolvedRoles"] if row["column"] == "sample")
    assert role["source"] == "inferred"
    assert role["authority"] == "diagnosticOnly"
    assert role["evidenceIds"] == ["column:sample", "column:condition"]
    confirmed = next(
        row for row in prepared["resolvedRoles"] if row["column"] == "condition"
    )
    assert confirmed["evidenceIds"] == ["study:protected"]
    # The inferred unit is summarized for diagnostics on the retained cohort.
    retain = prepared["qcProjections"][0]
    assert retain["policy"] == "retain"
    assert {row["value"]: row for row in retain["byGroup"]["sample"]["levels"]} == {
        f"s{index}": {
            "value": f"s{index}",
            "inputCells": 12,
            "retainedCells": 12,
            "removedCells": 0,
        }
        for index in range(1, 5)
    }
    (table,) = prepared["designDiagnostics"]["crossTabs"]
    assert {table["leftColumn"], table["rightColumn"]} == {"condition", "sample"}
    assert (table["rowsUsed"], table["fullyCrossed"]) == (48, True)
    assert sorted(row["count"] for row in table["counts"]) == [6] * 8
    assert not prepared["correctionEligible"]


def test_annotation_columns_cannot_supply_qc_bounds(source: Path) -> None:
    with pytest.raises(ValueError, match="Held-out annotation columns"):
        inspect_source(
            source,
            study(),
            AnalysisConfig(qcPolicy="manual", qcBounds={"cell.type.fine": (0, 1)}),
            runtime(),
        )


def test_gentle_qc_never_filters_high_count_cells_or_groups_by_sample(
    tmp_path: Path,
) -> None:
    source = tmp_path / "outlier.zarr"
    make_source(source, high_count_outlier=True)
    prepared = inspect_source(
        source,
        study(sampleColumn="sample"),
        AnalysisConfig(qcPolicy="gentleMad5"),
        runtime(),
    )
    options = prepared["filtering"]
    assert options["method"] == "manual"
    assert "sample_column" not in options
    assert options["highs"][options["attrs"].index("RNA_nCounts")] is None
    assert prepared["qcFlags"]["RNA_nCounts"]["highFlags"] == 1
    assert prepared["qcOutliers"]["RNA_nCounts"]["high"] == [47]
    assert prepared["retainedCells"] == 48


def test_reference_evidence_is_bounded_and_bound_on_resume(
    source: Path, tmp_path: Path
) -> None:
    reference = tmp_path / "markers.txt"
    reference.write_text("MS4A1 supports a B-cell hypothesis.")
    supplied = study(referenceFiles=[str(reference)])
    prepared = inspect_source(source, supplied, AnalysisConfig(), runtime())
    assert prepared["contextEvidence"]["references"][0]["text"].startswith("MS4A1")
    reference.write_text("Changed local evidence")
    with pytest.raises(ValueError, match="reference evidence changed"):
        verify_source(source, prepared, supplied, AnalysisConfig(), runtime())


def _merged_source(base: Path) -> Path:
    """Merge 48 cells with RNA and ADT and 16 cells with ADT only.

    The merged RNA assay measured the cells of the first source only, so
    its membership column ``RNA_I`` is False for the other 16 cells.
    """
    from scarf.merge import DataStoreMerge
    from tests.storage_helpers import write_count_store

    rng = np.random.default_rng(5)
    write_count_store(
        str(base / "full.zarr"),
        {"RNA": _counts(), "ADT": rng.integers(1, 20, size=(48, 3))},
        "uint32",
    )
    write_count_store(
        str(base / "adt.zarr"), {"ADT": rng.integers(1, 20, size=(16, 3))}, "uint32"
    )
    sources = [
        DataStore(
            str(base / f"{name}.zarr"),
            default_assay=assay,
            min_features_per_cell=0,
            nthreads=2,
            mem_budget="256M",
        )
        for name, assay in (("full", "RNA"), ("adt", "ADT"))
    ]
    path = base / "merged.zarr"
    DataStoreMerge(sources, str(path), ["full", "adt"], nthreads=2).dump()
    DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=2,
        mem_budget="256M",
    )
    return path


@pytest.mark.slow
def test_inspection_asks_for_a_cell_key_of_measured_cells(tmp_path: Path) -> None:
    """A merged store whose RNA assay measured some cells needs a cell key.

    Every stage reads RNA over the cells of ``cellKey``, and the pipeline
    refuses cells that RNA did not measure. Inspection asks for a cell key
    before any model decision, instead of failing at the first pipeline run.
    """
    source = _merged_source(tmp_path)
    before = _files(source)

    with pytest.raises(NeedsInput) as raised:
        inspect_source(source, study(), AnalysisConfig(), runtime())

    assert raised.value.field == "cellKey"
    question = raised.value.question
    assert "16 of the 64 cells of cellKey 'I'" in question
    assert "RNA_I" in question
    assert (
        "ds.cells.insert('RNA_measured', ds.cells.fetch_all('I') & "
        "ds.cells.fetch_all('RNA_I'))"
    ) in question
    assert _files(source) == before

    # The column that the question names makes the store analyzable.
    store = DataStore(str(source), min_features_per_cell=-1, nthreads=2)
    store.cells.insert(
        "RNA_measured", store.cells.fetch_all("I") & store.cells.fetch_all("RNA_I")
    )
    prepared = inspect_source(
        source, study(), AnalysisConfig(cellKey="RNA_measured"), runtime()
    )
    assert prepared["inputCells"] == prepared["retainedCells"] == 48
