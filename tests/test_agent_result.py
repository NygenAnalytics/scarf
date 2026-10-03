"""Results stay offline until numerical access and bind exports to the frozen run."""

import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from scarf.agent.models import AnalysisConfig, Study
from scarf.agent.records import RecordError, RunRecords
from scarf.agent.result import AnalysisRun
from scarf.storage.artifacts import ArtifactRef


_TEMPLATES: dict[tuple[object, ...], Path] = {}
_TEMPLATE_ROOTS: list[tempfile.TemporaryDirectory[str]] = []


def _copy_template(
    tmp_path: Path, key: tuple[object, ...], build: Callable[[Path], RunRecords]
) -> RunRecords:
    """Copy a saved history that is built once per process.

    Building appends hash-chained events with a file and directory sync each,
    which dominates these report tests; a copy has identical bytes.
    """
    template = _TEMPLATES.get(key)
    if template is None:
        if not _TEMPLATE_ROOTS:
            _TEMPLATE_ROOTS.append(
                tempfile.TemporaryDirectory(prefix="scarf-agent-records-")
            )
        # A unique base also isolates templates that build on other templates.
        base = Path(tempfile.mkdtemp(dir=_TEMPLATE_ROOTS[0].name))
        template = _TEMPLATES[key] = build(base).path
    destination = tmp_path / "analysis"
    shutil.copytree(template, destination)
    return RunRecords(destination)


def _records(
    tmp_path: Path, status: str = "completed", final: bool = True
) -> RunRecords:
    return _copy_template(
        tmp_path,
        ("records", status, final),
        lambda base: _build_records(base, status, final),
    )


def _build_records(tmp_path: Path, status: str, final: bool) -> RunRecords:
    records = RunRecords.create(
        tmp_path / "analysis",
        {
            "runId": "test-run",
            "source": "../missing-source",
            "study": Study(
                context="One sample <script>alert(1)</script>",
                objective="Discover populations",
            ).model_dump(mode="json"),
            "config": AnalysisConfig().model_dump(mode="json"),
            "procedureIdentity": "test-procedure",
        },
    )
    prepared = {
        "fingerprint": "fixed",
        "inputCells": 8,
        "retainedCells": 3,
        "limitations": ["No biological replication."],
    }
    records.append(
        "stageCompleted",
        stage="inspect",
        evidence=records.write_evidence("inspect", prepared),
    )
    if final:
        refs = {
            "clusters": ArtifactRef("assay", "cluster_labels", "a" * 64, "RNA"),
            "markers": ArtifactRef("assay", "marker_table", "b" * 64, "RNA"),
            "umap": ArtifactRef("assay", "embedding", "c" * 64, "RNA"),
        }
        final_result = {
            "runId": "fixed-final",
            "artifacts": {key: ref.to_dict() for key, ref in refs.items()},
        }
        records.append(
            "pipelineCompleted",
            operation="final",
            runId="fixed-final",
            label="final-label",
            recovered=False,
        )
        records.append(
            "stageCompleted",
            stage="finalize",
            evidence=records.write_evidence("finalize", final_result),
        )
        annotation = {
            "runId": "fixed-final",
            "clusters": refs["clusters"].to_dict(),
            "annotations": [
                {
                    "clusterId": "1",
                    "identity": "T cells",
                    "confidence": "medium",
                    "supportingMarkers": ["CD3D"],
                    "contradictingMarkers": [],
                    "rationale": "Observed CD3D.",
                },
                {
                    "clusterId": "2",
                    "identity": "unassigned",
                    "confidence": "low",
                    "supportingMarkers": [],
                    "contradictingMarkers": [],
                    "rationale": "Insufficient evidence.",
                },
            ],
        }
        records.append(
            "stageCompleted",
            stage="annotate",
            evidence=records.write_evidence("annotate", annotation),
        )
    records.append(
        "status",
        status=status,
        stage="annotate" if final else "inspect",
        questions=[{"questionId": "assay", "question": "Choose an assay."}]
        if status == "needsInput"
        else [],
    )
    return records


class _Run(dict[str, ArtifactRef]):
    def __init__(self, refs: dict[str, ArtifactRef], frame: pd.DataFrame) -> None:
        super().__init__(refs)
        self.run_id = "fixed-final"
        self.status = "completed"
        self.cells = SimpleNamespace(
            to_pandas_dataframe=lambda names: frame[names].copy()
        )


def _bind(
    monkeypatch: pytest.MonkeyPatch, records: RunRecords
) -> tuple[_Run, list[tuple[str, Any]]]:
    import scarf.agent.evidence as evidence

    saved = records.read_json("evidence/finalize.json")
    frame = pd.DataFrame(
        {
            "ids": ["cell-C", "cell-A", "cell-B"],
            "clusters": [2, 1, 1],
            "umap_1": [9.0, 3.0, 1.0],
            "umap_2": [2.0, 4.0, 6.0],
        }
    )
    run = _Run(
        {key: ArtifactRef.from_dict(ref) for key, ref in saved["artifacts"].items()},
        frame,
    )
    observed: list[tuple[str, Any]] = []

    def verify(source: Path, *args: Any) -> None:
        observed.append(("verify", source))

    def open_pipeline(*, run_id: str) -> _Run:
        observed.append(("openPipeline", run_id))
        return run

    def get_markers(**kwargs: Any) -> pd.DataFrame:
        observed.append(("markers", kwargs))
        return pd.DataFrame(
            {
                "group_id": ["1", "2"],
                "feature_name": ["CD3D", "GAPDH"],
                "score": [0.9, 0.1],
            }
        )

    def embedding(**kwargs: Any) -> str:
        observed.append(("plot", kwargs))
        return "plot-result"

    def open_store(source: Path, *args: Any, writable: bool = False) -> Any:
        assert not writable
        observed.append(("openStore", source))
        return SimpleNamespace(
            pipeline=SimpleNamespace(open=open_pipeline),
            inspect_artifact=lambda ref: SimpleNamespace(complete=True),
            get_markers=get_markers,
            plots=SimpleNamespace(embedding=embedding),
        )

    monkeypatch.setattr(evidence, "verify_source", verify)
    monkeypatch.setattr(evidence, "open_store", open_store)
    return run, observed


@pytest.mark.parametrize(
    "status", ["running", "needsInput", "completed", "failed", "interrupted"]
)
def test_all_outcomes_report_offline_and_escape_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    import scarf.agent.evidence as evidence

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Reporting opened the numerical store")

    monkeypatch.setattr(evidence, "open_store", forbidden)
    records = _records(tmp_path, status)
    records.append(
        "decisionAccepted",
        decisionId="context",
        stage="context",
        output={"rationale": "Use supplied metadata."},
    )
    records.append("decisionRejected", decisionId="context", errors=["Unknown choice"])
    before = records.events()
    result = AnalysisRun(records.path)
    report = result.report()
    assert result.status == status
    assert result.source == tmp_path / "missing-source"
    page = report.read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "No biological replication." in page
    assert "Unknown choice" in page
    assert "Use supplied metadata." in page
    assert 'href="events/"' in page
    assert (records.path / "annotations.csv").exists()
    assert (records.path / "report.md").exists()
    assert bool(result.pending_questions) == (status == "needsInput")
    assert records.events() == before
    assert result.report().read_text() == page


def test_pending_questions_clear_after_resume_status(tmp_path: Path) -> None:
    records = _records(tmp_path, "needsInput", final=False)
    result = AnalysisRun(records.path)
    assert result.pending_questions[0]["questionId"] == "assay"
    records.append("status", status="running", stage="inspect")
    assert result.pending_questions == []
    with pytest.raises(ValueError, match="no finalized"):
        _ = result.pipeline


def test_numerical_access_verifies_relocated_source_and_pins_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    run, observed = _bind(monkeypatch, records)
    relocated = tmp_path / "relocated"
    result = AnalysisRun(records.path, source=relocated)
    assert observed == []
    assert result.pipeline is run
    assert observed[:3] == [
        ("verify", relocated),
        ("openStore", relocated),
        ("openPipeline", "fixed-final"),
    ]
    assert result.artifacts["markers"] is run["markers"]
    assert result.pipeline_runs == [
        {
            "operation": "final",
            "runId": "fixed-final",
            "label": "final-label",
            "recovered": False,
        }
    ]
    result.get_markers(min_score=0.5)
    assert observed[-1] == ("markers", {"marker": run["markers"], "min_score": 0.5})
    assert result.plot_embedding(show=False) == "plot-result"
    assert observed[-1] == (
        "plot",
        {
            "run": run,
            "layout": "umap",
            "color_by": "clusters",
            "figsize": (8, 8),
            "theme": "paper",
            "point_edgewidth": 0,
            "point_alpha": 0.85,
            "legend_loc": "right",
            "show_titles": False,
            "show": False,
        },
    )
    with pytest.raises(ValueError, match="pinned"):
        result.get_markers(marker=run["markers"])
    with pytest.raises(ValueError, match="pinned"):
        result.plot_embedding(run=run)


def test_plot_options_override_defaults_without_changing_the_final_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    run, observed = _bind(monkeypatch, records)
    result = AnalysisRun(records.path)
    result.plot_embedding(
        figsize=(5, 5), point_alpha=1, legend_loc="on_data", show=True
    )
    options = observed[-1][1]
    assert options["figsize"] == (5, 5)
    assert options["point_alpha"] == 1
    assert options["legend_loc"] == "on_data"
    assert options["show"] is True
    assert options["run"] is run
    assert options["layout"] == "umap"


def test_marker_plot_uses_the_verified_final_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.plots as plots

    records = _records(tmp_path)
    run, observed = _bind(monkeypatch, records)

    def marker_plot(store: Any, pipeline: Any, **kwargs: Any) -> str:
        assert pipeline is run
        assert kwargs == {"top_n": 3, "max_genes": 12, "show": False}
        return "marker-plot"

    monkeypatch.setattr(plots, "marker_dotplot", marker_plot)
    assert (
        AnalysisRun(records.path).plot_markers(top_n=3, max_genes=12) == "marker-plot"
    )
    assert observed[-1] == ("openPipeline", "fixed-final")


@pytest.mark.parametrize("failure", [None, "save", "markers", "symlink"])
def test_saved_plots_are_high_resolution_atomic_and_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    import scarf.agent.plots as plots

    records = _records(tmp_path)
    result = AnalysisRun(records.path)
    run, _ = _bind(monkeypatch, records)
    closed = []
    saved = []
    expected = b"existing figure"
    umap = records.path / "umap_clusters.png"
    umap.write_bytes(expected)

    class Figure:
        def __init__(self, name: str) -> None:
            self.name = name

        def save(self, path: Path, **kwargs: Any) -> None:
            assert kwargs == {"dpi": 300, "exact_size": False}
            path.write_bytes(b"new " + self.name.encode())
            if failure == "save":
                raise OSError("Disk full")
            saved.append(self.name)

        def close(self) -> None:
            closed.append(self.name)

    def embedding(**kwargs: Any) -> Figure:
        assert kwargs["run"] is run and kwargs["layout"] == "umap"
        assert kwargs["show"] is False
        return Figure("umap")

    def marker_plot(store: Any, pipeline: Any) -> Figure:
        assert pipeline is run
        if failure == "markers":
            raise ValueError("No qualifying markers")
        return Figure("markers")

    store = SimpleNamespace(plots=SimpleNamespace(embedding=embedding))
    monkeypatch.setattr(result, "_bound_pipeline", lambda: (store, run))
    monkeypatch.setattr(plots, "marker_dotplot", marker_plot)
    if failure == "symlink":
        umap.unlink()
        target = tmp_path / "external.png"
        target.write_bytes(expected)
        umap.symlink_to(target)
    if failure:
        error, message = {
            "save": (OSError, "Disk full"),
            "markers": (ValueError, "No qualifying markers"),
            "symlink": (RecordError, "Refusing to replace a symlink plot: umap"),
        }[failure]
        with pytest.raises(error, match=message):
            result.save_plots()
    else:
        paths = result.save_plots()
        assert paths == {
            "umap_clusters": umap,
            "marker_dotplot": records.path / "marker_dotplot.png",
        }
        assert saved == ["umap", "markers"]
    assert umap.read_bytes() == (
        expected if failure in {"save", "symlink"} else b"new umap"
    )
    assert closed == (
        [] if failure == "symlink" else ["umap"] if failure else ["umap", "markers"]
    )
    assert not list(records.path.glob(".*.png"))
    assert result.status == "completed"


def test_plot_failure_keeps_completed_status_and_still_renders_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scarf.agent.api import _report

    records = _records(tmp_path)
    result = AnalysisRun(records.path)

    def fail() -> None:
        raise ValueError("Marker statistics unavailable")

    monkeypatch.setattr(result, "save_plots", fail)
    _report(result, records)
    assert result.status == "completed"
    assert records.latest("reportError")["stage"] == "report"
    assert (
        "A report plot could not be saved" in (records.path / "report.html").read_text()
    )


def test_source_change_prevents_export_before_destination_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.evidence as evidence

    records = _records(tmp_path)
    _bind(monkeypatch, records)

    def changed(*args: Any) -> None:
        raise ValueError("Source fingerprint changed")

    monkeypatch.setattr(evidence, "verify_source", changed)
    with pytest.raises(ValueError, match="fingerprint"):
        AnalysisRun(records.path).export(tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert AnalysisRun(records.path).report().exists()


def test_export_keeps_cell_alignment_and_annotation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    _bind(monkeypatch, records)
    destination = AnalysisRun(records.path).export(tmp_path / "export")
    clusters = pd.read_csv(destination / "clusters.csv")
    coordinates = pd.read_csv(destination / "umap.csv")
    assert clusters["ids"].tolist() == ["cell-C", "cell-A", "cell-B"]
    assert coordinates["ids"].tolist() == clusters["ids"].tolist()
    assert clusters["clusters"].tolist() == [2, 1, 1]
    assert coordinates["umap_1"].tolist() == [9.0, 3.0, 1.0]
    annotations = pd.read_csv(destination / "annotations.csv")
    assert annotations["identity"].tolist() == ["T cells", "unassigned"]
    assert json.loads(annotations.iloc[0]["supportingMarkers"]) == ["CD3D"]
    saved = json.loads((destination / "summary.json").read_text())
    assert saved["final"]["runId"] == "fixed-final"
    assert saved["exportedCells"] == 3
    assert saved["exportedMarkers"] == 2
    with pytest.raises(FileExistsError):
        AnalysisRun(records.path).export(destination)


def test_final_artifact_mismatch_prevents_numerical_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    run, _ = _bind(monkeypatch, records)
    run["markers"] = ArtifactRef("assay", "marker_table", "d" * 64, "RNA")
    with pytest.raises(ValueError, match="artifacts do not match"):
        AnalysisRun(records.path).get_markers()


def test_annotation_mismatch_rejected_offline(tmp_path: Path) -> None:
    records = _records(tmp_path)
    annotated = records.read_json("evidence/annotate.json")
    annotated["runId"] = "other"
    reference = records.write_evidence("wrong-annotation", annotated)
    records.append("stageCompleted", stage="annotate", evidence=reference)
    with pytest.raises(RecordError, match="final frozen"):
        _ = AnalysisRun(records.path).annotations


def test_rendering_failure_preserves_analysis_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.rendering as rendering

    records = _records(tmp_path)

    def failed(*args: Any) -> None:
        raise OSError("Disk full")

    monkeypatch.setattr(rendering, "atomic_text", failed)
    result = AnalysisRun(records.path)
    with pytest.raises(OSError, match="Disk full"):
        result.report()
    assert result.status == "completed"
    assert result.annotations


def test_report_uses_resolved_inputs_and_retains_diagnostic_limitations(
    tmp_path: Path,
) -> None:
    from scarf.agent.rendering import summary

    records = _records(tmp_path, status="failed", final=False)
    resolved = Study(context="Resolved study context", objective="Describe cells")
    records.append(
        "inputsResolved",
        study=resolved.model_dump(mode="json"),
        config=AnalysisConfig(assay="RNA").model_dump(mode="json"),
    )
    prepared = records.read_json("evidence/inspect.json")
    prepared["qcFlags"] = {
        "RNA_nFeatures": {"highFlags": 3, "lowFlags": 1, "missing": 2}
    }
    records.append(
        "stageCompleted",
        stage="preprocess",
        evidence=records.write_evidence("preprocess", prepared),
    )
    records.append(
        "candidateMeasured",
        candidateId="c0",
        evidence=records.write_evidence(
            "candidate",
            {
                "runId": "screen",
                "silhouetteSampleCells": 2000,
                "limitations": ["One partition could not be scored."],
            },
        ),
    )
    records.append(
        "finalistMeasured",
        optionId="c0:r0.5",
        evidence=records.write_evidence(
            "finalist",
            {
                "runId": "finalist",
                "metrics": {"markerCoherence": 0.5},
                "diagnosticScope": {"sampleCells": 10000},
                "limitations": [
                    "Rare populations may be absent from the diagnostic sample."
                ],
            },
        ),
    )
    request = records.write_json(
        "calls/select.request.json",
        {
            "userPrompt": json.dumps(
                {
                    "evidence": {
                        "rejectedOptions": {
                            "c1:r0.5": ["Biological protection worsened."]
                        }
                    }
                }
            )
        },
    )
    records.append("modelRequest", stage="finalists", requestPath=request)

    result = summary(records)
    assert result["study"]["context"] == "Resolved study context"
    assert result["config"]["assay"] == "RNA"
    assert result["qcFlags"]["RNA_nFeatures"]["missing"] == 2
    assert result["rejectedCorrections"] == {
        "c1:r0.5": ["Biological protection worsened."]
    }
    assert len(result["diagnostics"]) == 2
    page = AnalysisRun(records.path).report().read_text()
    assert "Resolved study context" in page
    assert "One partition could not be scored." in page
    assert "Rare populations may be absent" in page
    assert "Biological protection worsened." in page
    assert "Detected genes" in page
    assert "Marker coherence" in page


def test_source_rebinding_uses_latest_locator_and_respects_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    _, observed = _bind(monkeypatch, records)
    result = AnalysisRun(records.path)
    records.append("sourceRebound", source="../old-location")
    records.append("sourceRebound", source="../relocated")
    assert result.source == tmp_path / "relocated"
    assert observed == []
    assert result.pipeline.run_id == "fixed-final"
    assert observed[0] == ("verify", tmp_path / "relocated")
    assert AnalysisRun(records.path, source=tmp_path / "override").source == (
        tmp_path / "override"
    )


def test_missing_fingerprint_prevents_numerical_access_but_preserves_offline_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = _records(tmp_path)
    records = RunRecords.create(tmp_path / "unbound", complete.manifest)
    reference = records.write_evidence(
        "finalize", complete.read_json("evidence/finalize.json")
    )
    records.append("stageCompleted", stage="finalize", evidence=reference)
    records.append("status", status="completed")
    _, observed = _bind(monkeypatch, records)
    result = AnalysisRun(records.path)
    with pytest.raises(RecordError, match="missing its source fingerprint"):
        result.export(tmp_path / "export")
    assert observed == []
    assert not (tmp_path / "export").exists()
    assert result.report().exists()
    assert result.status == "completed"


def test_incomplete_saved_pipeline_cannot_export_numerical_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    run, observed = _bind(monkeypatch, records)
    run.status = "interrupted"
    result = AnalysisRun(records.path)
    with pytest.raises(ValueError, match="saved final pipeline is no longer complete"):
        result.export(tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert not any(name == "markers" for name, _ in observed)
    assert result.report().exists()


def test_incomplete_artifact_cannot_export_even_when_pipeline_is_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.evidence as evidence

    records = _records(tmp_path)
    run, observed = _bind(monkeypatch, records)
    open_store = evidence.open_store

    def with_missing_markers(*args: Any, **kwargs: Any) -> Any:
        store = open_store(*args, **kwargs)
        store.inspect_artifact = lambda ref: SimpleNamespace(
            complete=ref != run["markers"]
        )
        return store

    monkeypatch.setattr(evidence, "open_store", with_missing_markers)
    result = AnalysisRun(records.path)
    with pytest.raises(
        ValueError, match="saved final pipeline has incomplete artifacts"
    ):
        result.export(tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert not any(name == "markers" for name, _ in observed)
    assert result.report().exists()


@pytest.mark.parametrize("relative", [".", "exports/results"])
def test_export_refuses_source_and_nested_directories_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    records = _records(tmp_path)
    _, observed = _bind(monkeypatch, records)
    result = AnalysisRun(records.path)
    destination = result.source / relative
    with pytest.raises(ValueError, match="outside the numerical store"):
        result.export(destination)
    assert not result.source.exists()
    assert not destination.exists()
    assert not any(name == "markers" for name, _ in observed)


def test_export_requires_annotations_for_every_exported_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    _, observed = _bind(monkeypatch, records)
    incomplete = records.read_json("evidence/annotate.json")
    incomplete["annotations"] = incomplete["annotations"][:1]
    records.append(
        "stageCompleted",
        stage="annotate",
        evidence=records.write_evidence("incomplete-annotation", incomplete),
    )
    result = AnalysisRun(records.path)
    with pytest.raises(ValueError, match="do not cover the exported frozen clusters"):
        result.export(tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert not any(name == "markers" for name, _ in observed)
    assert len(result.annotations) == 1


def test_result_repr_reflects_saved_status_without_numerical_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(tmp_path)
    _, observed = _bind(monkeypatch, records)
    result = AnalysisRun(records.path)
    assert repr(result) == (
        f"AnalysisRun(run_dir={str(records.path)!r}, status='completed')"
    )
    records.append("status", status="interrupted")
    assert "status='interrupted'" in repr(result)
    assert observed == []


@pytest.mark.parametrize("name", ["annotations.csv", "report.md", "report.html"])
def test_report_refuses_symlink_outputs_without_changing_the_target(
    tmp_path: Path, name: str
) -> None:
    records = _records(tmp_path)
    protected = tmp_path / "protected-output"
    protected.write_text("Unrelated content must survive.")
    output = records.path / name
    output.symlink_to(protected)
    result = AnalysisRun(records.path)
    with pytest.raises(RecordError, match="must not overwrite a symlink"):
        result.report()
    assert output.is_symlink()
    assert protected.read_text() == "Unrelated content must survive."
    assert result.status == "completed"
    assert not list(records.path.glob(".tmp-*"))
    output.unlink()
    assert result.report().exists()


def test_large_report_truncates_visible_sections_but_preserves_full_saved_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    records = _records(tmp_path)
    _, observed = _bind(monkeypatch, records)
    # Durability is covered by the record tests; skip syncs for this setup only.
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", lambda descriptor: None)
        for index in range(102):
            records.append(
                "decisionAccepted",
                decisionId=f"decision-{index:03d}",
                stage="annotate",
                output={"rationale": f"Observed evidence for batch {index:03d}."},
            )
    before = records.events()
    result = AnalysisRun(records.path)
    html = result.report().read_text()
    markdown = (records.path / "report.md").read_text()
    assert "Additional entries: 2; see saved records." in html
    assert "decision-099" in markdown
    assert "decision-100" not in markdown
    assert records.latest("decisionAccepted")["decisionId"] == "decision-101"
    assert records.events() == before
    assert observed == []
