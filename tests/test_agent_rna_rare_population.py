"""Bounded numerical coverage of rare study groups during RNA screening."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from time import perf_counter

import numpy as np
import pytest

from scarf.agent.experimental_context.study import StudyContract
from scarf.agent.ingest import ingest
from scarf.agent.orchestrator import rna_tuning
from scarf.agent.orchestrator.models import (
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    PreprocessedAssayHandoff,
    artifact_model_to_ref,
)
from scarf.agent.parameter_tuning.hvg import core_hvg_evidence
from scarf.agent.types import ArtifactReferenceModel
from scarf.datastore.datastore import DataStore
from scarf.storage.selections import read_stored_selection_indices
from tests.agent_examples import example
from tests.agent_comparison_examples import observed_action
from tests.test_agent_ingest import _write_h5ad
from tests.test_agent_rna_adaptive import checkpoints  # noqa: F401


@pytest.mark.slow
@pytest.mark.usefixtures("checkpoints")
@pytest.mark.parametrize("maximum_sample", [350, 400])
def test_rare_study_group_enlarges_then_retains_full_reference_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    maximum_sample: int,
) -> None:
    """A 6% group needs full evidence, without being discarded as invalid."""
    started = perf_counter()
    rng = np.random.default_rng(4444)
    values = rng.poisson(0.2, (400, 90)).astype(np.uint16)
    values[:188, :12] += rng.poisson(9.0, (188, 12)).astype(np.uint16)
    values[188:376, 12:24] += rng.poisson(9.0, (188, 12)).astype(np.uint16)
    values[376:, 24:36] += rng.poisson(20.0, (24, 12)).astype(np.uint16)
    rare_markers = [
        "NKG7",
        "GNLY",
        "PRF1",
        "KLRD1",
        "FCER1G",
        "TYROBP",
        "CTSW",
        "CST7",
        "GZMB",
        "GZMH",
        "CCL5",
        "FGFBP2",
    ]
    names = [
        *[f"COMMON_A_{i}" for i in range(12)],
        *[f"COMMON_B_{i}" for i in range(12)],
        *rare_markers,
        *[f"BACKGROUND_{i}" for i in range(54)],
    ]
    source, target = tmp_path / "rare_rna.h5ad", tmp_path / "rare_rna.zarr"
    _write_h5ad(
        source,
        values,
        feature_types=[b"Gene Expression"] * 90,
        feature_names=[name.encode() for name in names],
    )
    ingested = ingest(path=source, zarrPath=target, directions={"matrixKey": "X"})
    assert ingested.status == "done", ingested.notes
    store = DataStore(
        str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
    )
    conditions = np.asarray(["common"] * 376 + ["rare_condition"] * 24)
    store.cells.insert("condition", conditions, overwrite=True)
    full_cells = store.snapshot_cell_selection("I")
    live_before = store.cells.fetch_all("I").copy()
    assert int(live_before.sum()) == 400
    features = core_hvg_evidence(store, assay="RNA", cells=full_cells)

    # The separate reference uses the public core recipe on every cell.
    normalized = store.run_normalization(full_cells, features=features["scarfDefault"])
    pca = store.run_pca(normalized)
    ann = store.build_ann_index(
        pca, ann_metric="l2", ann_parallel=False, rand_state=4466
    )
    neighbors = store.query_neighbors(ann, coordinates=pca, k=11)
    graph = store.build_connectivity_map(
        neighbors, local_connectivity=1.0, bandwidth=1.5
    )
    reference = store.run_leiden_clustering(
        graph,
        resolution=0.5,
        backend="igraph",
        symmetric_graph=False,
        graph_upper_only=False,
        random_seed=4444,
    )
    reference_labels = np.asarray(store.load_artifact(reference)["values"][:])
    rare_label, rare_count = np.unique(reference_labels[376:], return_counts=True)
    assert len(rare_label) == 1 and rare_count[0] == 24
    assert int((reference_labels == rare_label[0]).sum()) == 24

    refs = {
        key: ArtifactReferenceModel.from_artifact_ref(ref)
        for key, ref in features.items()
    }
    handoff = PreprocessedAssayHandoff(
        assay="RNA",
        assayType="RNA",
        cellSelection=ArtifactReferenceModel.from_artifact_ref(full_cells),
        graphFeatures=refs["scarfDefault"],
        graphFeatureCandidates=refs,
        markerFeatures=ArtifactReferenceModel.from_artifact_ref(
            store.select_all_features(from_assay="RNA")
        ),
        nCells=400,
        nFeatures=int(
            np.asarray(store.load_artifact(features["scarfDefault"])["values"][:]).sum()
        ),
        reductionMethod="pca",
    )
    plan = example(AutomatedPreprocessingPlan)
    plan.cellQc.attributes = []
    study = StudyContract.get_blank().model_copy(
        update={
            "conditionColumns": ["condition"],
            "protectedColumns": ["condition"],
            "columnKinds": {"condition": "categorical"},
        }
    )
    assessments = []

    def assess(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"][0])
        assessments.append(evidence["scope"])
        assert evidence["scope"] in {"sample1", "full"}
        chosen = next(
            row
            for row in evidence["candidates"]
            if row["parameters"]["leidenResolution"] == 0.5
        )
        assert chosen["eligible"] and chosen["metrics"]["markerCoherence"] is not None
        action = rna_tuning.TuningAction.model_validate(
            observed_action(
                evidence,
                selected=chosen["candidateId"]
                if evidence["comparisonCoverage"]["phase"] != "sensitivity"
                else None,
            )
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", assess)
    execute = rna_tuning.execute_parameter_candidate
    executions = []

    def measured_execute(*args: Any, **kwargs: Any) -> Any:
        result = execute(*args, **kwargs)
        executions.append(result.candidateId)
        return result

    monkeypatch.setattr(rna_tuning, "execute_parameter_candidate", measured_execute)
    run = rna_tuning.RnaTuningRun(
        SimpleNamespace(model=object()),
        store,
        SimpleNamespace(workflowRunId="rare-test"),
        SimpleNamespace(
            config=AutomatedWorkflowConfig(
                screeningCells=100, maxScreeningCells=maximum_sample
            )
        ),
        plan,
        handoff,
        study,
        {},
        {"fixture": "rare-marker-program"},
    )
    report, summary = run.run()
    assert report.status == "done", report.rationale
    assert assessments == ["sample1", "sample1", "full"]
    coverage = [row for row in summary["history"] if "coverage" in row]
    assert [row["scope"] for row in coverage] == ["sample0", "sample1", "full"]
    rare_rows = [
        next(
            group
            for group in row["coverage"]["groups"]["condition"]
            if group["value"] == "rare_condition"
        )
        for row in coverage
    ]
    assert 0 < rare_rows[0]["screeningCells"] < 20 <= rare_rows[1]["screeningCells"]
    assert rare_rows[2]["screeningCells"] == 24
    assert coverage[0]["coverageConcerns"] and not coverage[1]["coverageConcerns"]
    assert run.evaluations["sample0"] == []
    assert summary["budget"]["scopes"]["sample0"]["reserved"]["partitions"] == 0
    assert summary["budget"]["scopes"]["sample1"]["completed"] == {
        "graphs": 5,
        "partitions": 8,
    }
    assert summary["budget"]["scopes"]["full"]["completed"] == {
        "graphs": 1 if maximum_sample < 400 else 0,
        "partitions": 1 if maximum_sample < 400 else 0,
    }
    assert report.cellSelection == handoff.cellSelection
    assert artifact_model_to_ref(report.finalClusterArtifact) == reference
    np.testing.assert_array_equal(store.cells.fetch_all("I"), live_before)
    selected = next(
        row
        for row in report.evaluations
        if row.candidateId == report.recommendedCandidateId
    )
    markers = store.get_markers(
        marker=artifact_model_to_ref(selected.artifacts["markerTable"]),
        min_score=0.0,
        min_frac_exp=0.0,
    )
    rare_table = markers[markers.group_id.astype(str) == str(rare_label[0])]
    assert {"NKG7", "GNLY", "PRF1"}.issubset(set(rare_table.feature_name))
    from sklearn.metrics import adjusted_rand_score

    screened = next(
        row
        for row in run.evaluations["sample1"]
        if row.parameters.leidenResolution == 0.5
        and row.parameters.dimensions == 21
        and row.parameters.neighborsK == 11
    )
    sample_ref = artifact_model_to_ref(screened.cellSelection)
    rows = read_stored_selection_indices(
        store.zw,
        sample_ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    sample_labels = np.asarray(
        store.load_artifact(artifact_model_to_ref(screened.artifacts["clusters"]))[
            "values"
        ][:]
    )
    # This checks numerical transfer on exactly shared cells, independently of the scripted preference.
    assert adjusted_rand_score(reference_labels[rows], sample_labels) > 0.95
    sample_rare_labels = np.unique(sample_labels[rows >= 376])
    assert len(sample_rare_labels) == 1
    assert int((sample_labels == sample_rare_labels[0]).sum()) == int(
        (rows >= 376).sum()
    )
    assert {"NKG7", "GNLY", "PRF1"}.issubset(
        screened.metrics.topMarkerGenes[str(sample_rare_labels[0])]
    )
    assert len(executions) == (9 if maximum_sample < 400 else 8)
    (tmp_path / "rna_comparison_measurements.json").write_text(
        json.dumps(
            {
                "screeningCells": maximum_sample,
                "fullCells": 400,
                "candidateExecutorCalls": len(executions),
                "screeningCompleted": summary["budget"]["scopes"]["sample1"][
                    "completed"
                ],
                "additionalFullCompleted": summary["budget"]["scopes"]["full"][
                    "completed"
                ],
                "sharedCellAdjustedRandIndex": adjusted_rand_score(
                    reference_labels[rows], sample_labels
                ),
                "rareScreeningCells": int((rows >= 376).sum()),
                "rareFullCells": 24,
                "observedRareMarkerGenes": screened.metrics.topMarkerGenes[
                    str(sample_rare_labels[0])
                ],
                "testElapsedSeconds": perf_counter() - started,
                "interpretation": "Local synthetic test including the independent core reference and assertions. This is not model decision agreement or a large-cohort runtime estimate.",
            },
            sort_keys=True,
            indent=2,
        )
    )
