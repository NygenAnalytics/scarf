"""End-to-end scripted decisions execute real Scarf numerical pipelines.

Every real-pipeline check in this module reads one complete analysis: four
native probes, a lenient tie between two neighbor-graph partitions, a declared
sample column, protected biology and a held-out author annotation. Checks that
write to the run history or the store use a private copy of both.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from scarf.agent import (
    AnalysisConfig,
    RuntimeConfig,
    Study,
    analyze_rna,
    open_analysis,
    resume_rna,
)
from scarf.agent.records import RunRecords, digest

_CELLS = 120
_RESOLUTIONS = (0.5, 0.75)
_RUNTIME = RuntimeConfig(nthreads=1, memBudget="512M")
_SELECTED = "c3:r0.5"
_OPERATIONS = [
    "screen_c0",
    "screen_c1",
    "screen_c2",
    "screen_c3",
    "finalist_0",
    "finalist_1",
    "final",
]


def test_finalist_prompt_keeps_measurements_with_bounded_cluster_detail() -> None:
    from scarf.agent.workflow import _selection_evidence

    source = {
        "metrics": {"markerCoherence": 0.8},
        "clusters": [
            {
                "clusterId": str(index),
                "count": index + 1,
                "qualifyingMarkerCount": index % 5,
                "markers": [{"gene": f"G{gene}"} for gene in range(20)],
                "weakMarkers": [{"gene": f"W{gene}"} for gene in range(7)],
            }
            for index in range(100)
        ],
    }
    compact = _selection_evidence(source)
    assert compact["metrics"] == source["metrics"]
    # Four largest, four smallest, then the four with fewest qualifying markers.
    selected = ["99", "98", "97", "96", "3", "2", "1", "0", "5", "10", "15"]
    assert [row["clusterId"] for row in compact["clusters"]] == selected
    assert compact["clusterCount"] == 100
    assert compact["omittedClusterCount"] == 100 - len(selected)
    assert [row["clusterId"] for row in compact["clusterSummaries"]] == [
        str(index) for index in range(64)
    ]
    assert compact["clusterSummaries"][5] == {
        "clusterId": "5",
        "count": 6,
        "qualifyingMarkerCount": 0,
    }
    assert compact["omittedClusterSummaryCount"] == 36
    for row in compact["clusters"]:
        assert [marker["gene"] for marker in row["markers"]] == [
            f"G{gene}" for gene in range(6)
        ]
        assert [marker["gene"] for marker in row["weakMarkers"]] == ["W0", "W1", "W2"]
    assert len(source["clusters"][0]["markers"]) == 20


def scripted_model(observed: Any) -> Any:
    async def choose(messages: Any, info: Any) -> Any:
        payload = json.loads(messages[-1].parts[-1].content)
        evidence = payload["evidence"]
        observed.append(evidence)
        name = info.output_tools[0].parameters_json_schema.get("title")
        answer: dict[str, Any]
        if name == "ContextDecision":
            answer = {
                "rationale": "Supplied study supports descriptive discovery",
                "columnRoles": {},
                "excludeFeatures": [],
            }
        elif name == "AnnotationDecision":
            answer = {
                "annotations": [
                    {
                        "clusterId": row["clusterId"],
                        "identity": "unassigned",
                        "rationale": "Provisional marker evidence requires independent review",
                    }
                    for row in evidence["clusters"]
                ]
            }
        elif "eligibleOptions" in evidence:
            assert all(row["clusters"] for row in evidence["finalists"].values())
            answer = {
                "action": "choose",
                "optionIds": [evidence["eligibleOptions"][0]],
                "rationale": "Selected a complete native finalist",
                "evidenceIds": [evidence["eligibleOptions"][0]],
            }
        elif evidence.get("decisionKind") == "pcProbe":
            answer = {
                "action": "experiment",
                "optionIds": [next(iter(evidence["experiments"]))],
                "rationale": "Measure the registered PC probe",
                "evidenceIds": ["c0"],
            }
        else:
            partitions = evidence["candidates"][0]["partitions"]
            answer = {
                "action": "shortlist",
                "optionIds": [partitions[0]["optionId"]],
                "rationale": "Bounded descriptive baseline",
                "evidenceIds": [partitions[0]["optionId"]],
            }
        return ModelResponse(parts=[ToolCallPart("decision", answer)])

    return FunctionModel(choose)


def _panel_model(observed: list[dict[str, Any]]) -> FunctionModel:
    """Measure the 30-PC probe, shortlist the neighbor graph and defer a tie."""

    async def respond(messages: Any, info: Any) -> ModelResponse:
        evidence = json.loads(messages[-1].parts[-1].content)["evidence"]
        schema = info.output_tools[0].parameters_json_schema["title"]
        observed.append({"schema": schema, "evidence": evidence})
        kind = evidence.get("decisionKind")
        answer: dict[str, Any]
        if schema == "ContextDecision":
            answer = {"rationale": "Retain the supplied cohort and declared roles."}
        elif schema == "AnnotationDecision":
            answer = {
                "annotations": [
                    {
                        "clusterId": row["clusterId"],
                        "identity": "unassigned",
                        "rationale": "Synthetic evidence does not establish a biological identity.",
                    }
                    for row in evidence["clusters"]
                ]
            }
        elif kind == "pcProbe":
            answer = {
                "action": "experiment",
                "optionIds": [
                    key
                    for key, row in evidence["experiments"].items()
                    if row["pcaDims"] == 30
                ],
                "evidenceIds": ["c0"],
                "rationale": "Measure the higher-rank registered probe.",
            }
        elif kind == "nativeShortlist":
            options = ["c3:r0.5", "c3:r0.75"]
            answer = {
                "action": "shortlist",
                "optionIds": options,
                "evidenceIds": options,
                "rationale": "Compare both partitions of the measured neighbor graph.",
            }
        else:
            options = evidence["eligibleOptions"]
            answer = {
                "action": "defer",
                "acceptableOptionIds": options,
                "evidenceIds": options,
                "deferralReason": "ambiguousSelection",
                "question": "Both measured partitions support descriptive discovery; which should be presented?",
                "rationale": "Both measured partitions remain acceptable for this objective.",
            }
        return ModelResponse(parts=[ToolCallPart("decision", answer)])

    return FunctionModel(respond)


def _panel_source(path: Path) -> None:
    """Three marker-gene populations with crossed donors and protected condition."""
    from scipy.sparse import csr_matrix

    from scarf import DataStore
    from scarf.writers import SparseToZarr

    rng = np.random.default_rng(39)
    counts = rng.poisson(1, size=(_CELLS, 2102)).astype(np.uint32)
    for group in range(3):
        counts[group * 40 : (group + 1) * 40, 2 + group * 40 : 42 + group * 40] += (
            rng.poisson(4, size=(40, 40)).astype(np.uint32)
        )
    names = ["MT-CO1", "RPL3", *[f"GENE{i}" for i in range(2100)]]
    SparseToZarr(
        csr_matrix(counts),
        str(path),
        [f"cell{i}" for i in range(_CELLS)],
        names,
        mem_budget="512M",
        nthreads=1,
    ).dump()
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        mem_budget="512M",
    )
    store.cells.insert("donor", np.tile(["donorA", "donorB"], _CELLS // 2))
    store.cells.insert(
        "condition", np.tile(["control", "control", "treated", "treated"], 30)
    )
    store.cells.insert(
        "author_annotation", np.repeat(["SECRET_A", "SECRET_B", "SECRET_C"], 40)
    )


@pytest.fixture(scope="module")
def completed_panel(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One completed analysis with default probes; shared read-only by this module."""
    from pydantic_ai import models
    from threadpoolctl import threadpool_limits

    from scarf import DataStore

    base = tmp_path_factory.mktemp("completed-panel")
    source = base / "source.zarr"
    _panel_source(source)
    before = DataStore(str(source), zarr_mode="r", nthreads=1, mem_budget="512M")
    columns = list(before.cells.columns)
    frame = before.cells.to_pandas_dataframe(columns, key=None)
    observed: list[dict[str, Any]] = []
    # The autouse agent guard is function scoped; apply the same limits here.
    with pytest.MonkeyPatch.context() as patch, threadpool_limits(limits=1):
        patch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
        result = analyze_rna(
            source,
            run_dir=base / "analysis",
            model=_panel_model(observed),
            study=Study(
                context="Synthetic RNA with two explicitly declared donors.",
                objective="Describe populations",
                sampleColumn="donor",
                protectedColumns=["condition"],
            ),
            config=AnalysisConfig(maxCandidates=4, resolutions=_RESOLUTIONS),
            runtime=_RUNTIME,
        )
    records = RunRecords(result.run_dir)
    assert result.status == "completed", records.events()[-5:]
    return SimpleNamespace(
        base=base,
        source=source.resolve(),
        run_dir=result.run_dir,
        observed=observed,
        columns=columns,
        frame=frame,
    )


@pytest.fixture
def panel_copy(completed_panel: SimpleNamespace, tmp_path: Path) -> SimpleNamespace:
    """A private copy of the completed store and history for writing checks."""
    base = tmp_path / "panel"
    shutil.copytree(completed_panel.base, base, symlinks=True)
    return SimpleNamespace(
        source=(base / "source.zarr").resolve(),
        run_dir=(base / "analysis").resolve(),
    )


def _open_store(source: Path) -> Any:
    from scarf import DataStore

    return DataStore(str(source), zarr_mode="r", nthreads=1, mem_budget="512M")


def _decisions(observed: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Evidence shown to the model for one decision kind, in request order."""
    if kind == "finalists":
        return [
            row["evidence"] for row in observed if "eligibleOptions" in row["evidence"]
        ]
    return [
        row["evidence"]
        for row in observed
        if row["evidence"].get("decisionKind") == kind
    ]


@pytest.mark.slow
def test_real_four_representation_panel_and_recorded_lenient_choice(
    completed_panel: SimpleNamespace,
) -> None:
    result = open_analysis(completed_panel.run_dir)
    records = RunRecords(result.run_dir)
    coverage = result.exploration_coverage
    assert coverage is not None
    assert coverage["nativeComplete"] is True
    assert [
        (row["candidateId"], row["axis"], row["parentId"], row["status"])
        for row in coverage["slots"]
    ] == [
        ("c0", "baseline", None, "measured"),
        ("c1", "hvgCount", "c0", "measured"),
        ("c2", "pcaDims", "c0", "measured"),
        ("c3", "neighborsK", "c0", "measured"),
    ]
    candidates = result.candidates
    baseline = candidates[0]["parameters"]
    assert baseline == {
        "hvgCount": 1000,
        "pcaDims": 21,
        "neighborsK": 11,
        "useHarmony": False,
    }
    # Each registered default probe changes exactly one axis of the baseline.
    for row, (axis, value) in zip(
        candidates[1:],
        [("hvgCount", 2000), ("pcaDims", 30), ("neighborsK", 21)],
        strict=True,
    ):
        assert row["parameters"] == {**baseline, axis: value}
        assert row["selection"] == candidates[0]["selection"]
        assert [item["resolution"] for item in row["comparisons"]] == list(_RESOLUTIONS)
        assert all(
            item["parentRunId"] == candidates[0]["runId"]
            and item["runId"] == row["runId"]
            and item["cellCount"] == _CELLS
            for item in row["comparisons"]
        )
    assert [row["actualHvgCount"] for row in candidates[:2]] == [1000, 2000]
    assert [row["operation"] for row in result.pipeline_runs] == _OPERATIONS

    probe = _decisions(completed_panel.observed, "pcProbe")
    shortlist = _decisions(completed_panel.observed, "nativeShortlist")
    finalists = _decisions(completed_panel.observed, "finalists")
    assert len(probe) == len(shortlist) == len(finalists) == 1
    assert probe[0]["allowedActions"] == ["experiment", "defer"]
    assert sorted(row["pcaDims"] for row in probe[0]["experiments"].values()) == [
        10,
        30,
    ]
    # The shortlist is requested only after every independent probe is measured.
    assert shortlist[0]["allowedActions"] == ["shortlist", "defer"]
    assert [row["status"] for row in shortlist[0]["explorationCoverage"]["slots"]] == [
        "measured"
    ] * 4
    assert finalists[0]["allowedActions"] == ["choose", "defer"]
    assert finalists[0]["eligibleOptions"] == ["c3:r0.5", "c3:r0.75"]

    (resolution,) = result.decision_resolutions
    assert resolution["stage"] == "finalists"
    assert resolution["reason"] == "ambiguousSelection"
    assert resolution["rule"] == "orderedAcceptableOption"
    assert resolution["optionOrder"] == ["c3:r0.5", "c3:r0.75"]
    assert resolution["resolved"] == {"action": "choose", "optionIds": [_SELECTED]}
    assert records.read_json("evidence/finalize.json")["selected"] == _SELECTED
    replayed = result.replay_decisions()
    assert [row["source"] for row in replayed].count("policy") == 1
    assert all(row["valid"] for row in replayed)

    finalist = records.read_json("evidence/finalist_0.json")
    assert "markerSupportFraction" in finalist["metrics"]
    assert "markerCoherence" not in finalist["metrics"]
    assessed_run = _open_store(completed_panel.source).pipeline.open(
        run_id=finalist["runId"]
    )
    assert assessed_run["markers"] == result.pipeline["markers"]

    # Held-out labels never reach a prompt and the live source is unchanged.
    assert not any(
        "SECRET_" in json.dumps(row["evidence"]) for row in completed_panel.observed
    )
    after = _open_store(completed_panel.source)
    assert list(after.cells.columns) == completed_panel.columns
    pd.testing.assert_frame_equal(
        after.cells.to_pandas_dataframe(completed_panel.columns, key=None),
        completed_panel.frame,
    )


@pytest.mark.slow
def test_real_measurements_summarize_frozen_partitions_and_declared_roles(
    completed_panel: SimpleNamespace,
) -> None:
    result = open_analysis(completed_panel.run_dir)
    records = RunRecords(result.run_dir)
    store = _open_store(completed_panel.source)
    for candidate in result.candidates:
        run = store.pipeline.open(run_id=candidate["runId"])
        assert candidate["silhouetteSampleCells"] == _CELLS
        assert [row["optionId"] for row in candidate["partitions"]] == [
            f"{candidate['candidateId']}:r{value:g}" for value in _RESOLUTIONS
        ]
        for row in candidate["partitions"]:
            labels = pd.Series(run.cells.fetch(f"leiden_{row['resolution']}"))
            counts = labels.astype(str).value_counts().sort_index()
            assert row["count"] == _CELLS
            assert row["clusterCount"] == len(counts)
            assert row["clusterCounts"] == {
                str(name): int(value) for name, value in counts.items()
            }

    finalist = records.read_json("evidence/finalist_0.json")
    run = store.pipeline.open(run_id=finalist["runId"])
    labels = np.asarray(run.cells.fetch("clusters")).astype(str)
    observed = pd.Series(labels).value_counts().to_dict()
    assert {row["clusterId"]: row["count"] for row in finalist["clusters"]} == (
        observed
    )
    assert finalist["parameters"] == {
        "hvgCount": 1000,
        "pcaDims": 21,
        "neighborsK": 21,
        "useHarmony": False,
        "resolution": 0.5,
    }
    assert finalist["selection"] == result.candidates[3]["selection"]
    scope = finalist["diagnosticScope"]
    assert scope["populationCells"] == scope["sampleCells"] == _CELLS
    assert scope["neighborsK"] == 21
    # Declared protected biology is measured; no technical batch was declared.
    assert finalist["metrics"]["mixing"] == {}
    protection = finalist["metrics"]["protection"]["condition"]
    assert 0 <= protection["cLISI"] <= 1
    assert 0 <= protection["graphConnectivity"] <= 1
    # Donors alternate across every population, so each cluster spans both.
    assert finalist["requiredSampleSupport"] is True
    assert finalist["metrics"]["crossUnitSupport"] == 1.0
    assert {row["sampleCount"] for row in finalist["clusters"]} == {2}
    supported = 0
    for row in finalist["clusters"]:
        table = store.get_markers(
            run["markers"], group_id=row["clusterId"], min_score=-1, min_frac_exp=-1
        )
        qualifying = table[(table["score"] >= 0.25) & (table["frac_exp"] >= 0.2)]
        assert row["qualifyingMarkerCount"] == len(qualifying)
        supported += int(not qualifying.empty)
    assert finalist["metrics"]["markerSupportFraction"] == supported / len(
        finalist["clusters"]
    )


@pytest.mark.slow
def test_registered_neighbor_experiment_keeps_graph_partitions_separate(
    completed_panel: SimpleNamespace,
) -> None:
    result = open_analysis(completed_panel.run_dir)
    records = RunRecords(result.run_dir)
    (shortlist,) = _decisions(completed_panel.observed, "nativeShortlist")
    candidates = shortlist["candidates"]
    assert [row["candidateId"] for row in candidates] == ["c0", "c1", "c2", "c3"]
    assert [row["parameters"]["neighborsK"] for row in candidates] == [11, 11, 11, 21]
    assert all(row["selection"] == candidates[0]["selection"] for row in candidates)
    for candidate in candidates:
        assert {row["resolution"] for row in candidate["partitions"]} == set(
            _RESOLUTIONS
        )
        assert all(
            row["candidateId"] == candidate["candidateId"]
            and row["optionId"].startswith(candidate["candidateId"] + ":")
            for row in candidate["partitions"]
        )
    admissions = [
        row["candidate"]
        for row in records.events()
        if row["kind"] == "candidateAdmitted"
    ]
    assert [row["candidateId"] for row in admissions] == ["c0", "c1", "c2", "c3"]
    assert admissions[3] == {
        "candidateId": "c3",
        "parentId": "c0",
        "hvgCount": 1000,
        "pcaDims": 21,
        "neighborsK": 21,
        "useHarmony": False,
    }
    calls = {
        row["operation"]: row["runId"]
        for row in records.events()
        if row["kind"] == "pipelineCompleted"
    }
    assert list(calls) == _OPERATIONS
    store = _open_store(completed_panel.source)
    runs = {name: store.pipeline.open(run_id=run_id) for name, run_id in calls.items()}
    baseline, features, components, changed = (
        runs[f"screen_c{index}"] for index in range(4)
    )
    finalist, final = runs["finalist_0"], result.pipeline
    # Each probe recomputes only its own axis and reuses everything upstream.
    assert features["highly_variable_features"] != baseline["highly_variable_features"]
    assert (
        components["highly_variable_features"] == (baseline["highly_variable_features"])
    )
    assert components["pca"] != baseline["pca"]
    assert changed["pca"] == baseline["pca"]
    assert changed["neighbors"] != baseline["neighbors"]
    assert changed["connectivity_map"] != baseline["connectivity_map"]
    # The selected neighbor-probe partition is finalized on its own graph.
    selected_key = next(
        key
        for key in changed
        if key.startswith("leiden_") and float(key.removeprefix("leiden_")) == 0.5
    )
    for key in ("analysis_cell_selection", "neighbors", "connectivity_map"):
        assert changed[key] == finalist[key] == final[key]
    assert changed[selected_key] == finalist["clusters"] == final["clusters"]
    assert finalist["markers"] == final["markers"]
    assert records.read_json("evidence/finalize.json")["selected"] == _SELECTED


@pytest.mark.slow
def test_real_numerical_completion_publishes_exact_final_pipeline(
    completed_panel: SimpleNamespace,
) -> None:
    result = open_analysis(completed_panel.run_dir)
    records = RunRecords(result.run_dir)
    compact = result.compact_result
    assert compact is not None
    final = result.pipeline
    assert compact["finalPipelineRunId"] == final.run_id
    assert compact["agentRunId"] == records.manifest["runId"]
    assert compact["assay"] == "RNA"
    assert (
        compact["sourceFingerprint"]
        == (records.read_json("evidence/preprocess.json")["fingerprint"])
    )
    assert compact["externalRunLocator"] == "../analysis"
    selected = compact["selectedParameters"]
    assert selected["pipelineConfig"] == final.report()["run"]["config"]
    assert {
        key: value for key, value in selected.items() if key != "pipelineConfig"
    } == {
        "candidateId": "c3",
        "resolution": 0.5,
        "requestedHvgCount": 1000,
        "actualHvgCount": 1000,
        "pcaDims": 21,
        "neighborsK": 21,
        "useHarmony": False,
    }
    # The lenient tie names its frozen rule next to the model rationale.
    assert compact["selectionRationale"] == (
        "Both measured partitions remain acceptable for this objective."
        " Automatic resolution: orderedAcceptableOption."
    )
    assert records.latest("resultPublished")["resultDigest"] == digest(compact)
    reopened = _open_store(completed_panel.source)
    assert reopened.assay_names == ["RNA"]
    assert (
        reopened.pipeline.open(run_id=compact["finalPipelineRunId"])["clusters"]
        == result.artifacts["clusters"]
    )
    assert not {"calls", "events", "study"} & set(compact)
    assert "thinking" not in json.dumps(compact)


@pytest.mark.slow
def test_real_pipeline_annotations_reuse_and_readonly_export(
    completed_panel: SimpleNamespace, panel_copy: SimpleNamespace, tmp_path: Path
) -> None:
    result = open_analysis(panel_copy.run_dir)
    records = RunRecords(result.run_dir)
    final = result.pipeline
    clusters = {str(value) for value in final.cells.fetch("clusters")}
    assert {row["clusterId"] for row in result.annotations} == clusters
    assert {row["identity"] for row in result.annotations} == {"unassigned"}
    assert "doublets" not in result.artifacts
    for name in ("umap_clusters.png", "marker_dotplot.png"):
        assert (result.run_dir / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert records.latest("reportError") is None
    assert result.report().exists()

    exported = result.export(tmp_path / "export")
    frame = final.cells.to_pandas_dataframe(["ids", "clusters", "umap_1", "umap_2"])
    saved = pd.read_csv(exported / "clusters.csv")
    embedding = pd.read_csv(exported / "umap.csv")
    assert saved["ids"].tolist() == [f"cell{i}" for i in range(_CELLS)]
    assert embedding["ids"].tolist() == saved["ids"].tolist()
    assert (
        saved["clusters"].astype(str).tolist() == frame["clusters"].astype(str).tolist()
    )
    np.testing.assert_allclose(embedding["umap_1"], frame["umap_1"], rtol=1e-6)

    def no_provider(messages: Any, info: Any) -> Any:
        raise AssertionError("A completed analysis must not contact a provider")

    before = records.events()
    resumed = resume_rna(result.run_dir, model=FunctionModel(no_provider))
    assert resumed.status == "completed"
    assert records.events() == before
    assert open_analysis(result.run_dir).status == "completed"
    assert [row["candidateId"] for row in result.candidates] == [
        "c0",
        "c1",
        "c2",
        "c3",
    ]
    replayed = result.replay_decisions()
    assert len(replayed) == len(completed_panel.observed) + 1
    assert all(row["valid"] for row in replayed)


@pytest.mark.slow
def test_real_fresh_mount_cannot_replace_history_but_a_complete_copy_can(
    completed_panel: SimpleNamespace, panel_copy: SimpleNamespace, tmp_path: Path
) -> None:
    from scarf import mount_datastore

    result = open_analysis(panel_copy.run_dir)
    records = RunRecords(result.run_dir)
    original = result.pipeline
    original_refs = dict(original)
    before = records.events()
    mount_path = tmp_path / "fresh-mount.zarr"
    mounted = mount_datastore(
        str(panel_copy.source),
        str(mount_path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        mem_budget="512M",
    )
    assert mounted.pipeline.list_runs() == ()
    assert all(mounted.inspect_artifact(ref).complete for ref in original_refs.values())
    with pytest.raises(ValueError, match="Replacement source cannot open"):
        resume_rna(result.run_dir, model=None, source=mount_path)
    assert records.events() == before
    assert open_analysis(result.run_dir).source == panel_copy.source
    assert open_analysis(result.run_dir).pipeline.run_id == original.run_id

    # A sibling copy keeps the compact result's relative history locator valid.
    relocated = panel_copy.source.parent / "complete-copy.zarr"
    shutil.copytree(panel_copy.source, relocated)
    resumed = resume_rna(result.run_dir, model=None, source=relocated)
    assert resumed.status == "completed"
    assert open_analysis(result.run_dir).source == relocated
    assert dict(resumed.pipeline) == original_refs
    appended = records.events()[len(before) :]
    assert [row["kind"] for row in appended] == ["sourceRebound"]
    assert (
        appended[0]["fingerprint"]
        == (records.read_json("evidence/inspect.json")["fingerprint"])
    )
    copied = _open_store(relocated)
    assert {run.run_id for run in copied.pipeline.list_runs()} == {
        row["runId"] for row in result.pipeline_runs
    }


def test_sync_api_rejects_notebook_loop_before_creating_files(tmp_path: Any) -> None:
    import asyncio

    async def call() -> Any:
        with pytest.raises(RuntimeError, match="event loop"):
            analyze_rna(
                tmp_path,
                run_dir=tmp_path / "run",
                model="unused",
                study=Study(context="Study", objective="Discovery"),
            )

    asyncio.run(call())
    assert not (tmp_path / "run").exists()


def test_provider_failure_is_inspectable_without_store_results(
    agent_rna_source: Any, tmp_path: Any
) -> None:
    from pydantic_ai.exceptions import ModelHTTPError

    async def failure(messages: Any, info: Any) -> Any:
        raise ModelHTTPError(401, "offline-model", {"error": "unauthorized"})

    result = analyze_rna(
        agent_rna_source,
        run_dir=tmp_path / "failed",
        model=FunctionModel(failure),
        study=Study(context="Study", objective="Discovery"),
        config=AnalysisConfig(hvgCount=40, pcaDims=4, neighborsK=7),
        runtime=RuntimeConfig(nthreads=2, memBudget="512M"),
    )
    assert result.status == "failed"
    assert result.report().exists()
    assert (
        len(
            [
                row
                for row in RunRecords(result.run_dir).events()
                if row["kind"] == "modelRequest"
            ]
        )
        == 1
    )
