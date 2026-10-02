"""Saved analyses produce readable, self-contained scientific reports offline."""

import base64
import csv
import io
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from scarf.agent.records import RecordError, RunRecords
from scarf.agent.result import AnalysisRun

from .test_agent_result import _records

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)


class _Document(HTMLParser):
    """Inspect semantics and resource URLs without a browser dependency."""

    def __init__(self, page: str) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []
        self.primary_text: list[str] = []
        self.rows: list[list[str]] = []
        self.headings: list[str] = []
        self._hidden: list[str] = []
        self._details = 0
        self._cell: list[str] | None = None
        self._row: list[str] | None = None
        self._heading: list[str] | None = None
        self.feed(page)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag in {"style", "script", "head"}:
            self._hidden.append(tag)
        if tag == "details":
            self._details += 1
        if tag == "tr":
            self._row = []
        if tag in {"th", "td"}:
            self._cell = []
        if tag in {"h1", "h2", "h3"}:
            self._heading = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"style", "script", "head"} and self._hidden:
            self._hidden.pop()
        if tag == "details":
            self._details -= 1
        if tag in {"th", "td"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join(self._cell))
            self._cell = None
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        if tag in {"h1", "h2", "h3"} and self._heading is not None:
            self.headings.append(" ".join(self._heading))
            self._heading = None

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if not value or self._hidden:
            return
        self.text.append(value)
        if self._details == 0:
            self.primary_text.append(value)
        if self._cell is not None:
            self._cell.append(value)
        if self._heading is not None:
            self._heading.append(value)


def _complete(records: RunRecords, stage: str, name: str, value: Any) -> None:
    records.append(
        "stageCompleted", stage=stage, evidence=records.write_evidence(name, value)
    )


def _rich_records(tmp_path: Path) -> RunRecords:
    records = _records(tmp_path)
    prepared = records.read_json("evidence/inspect.json")
    prepared.update(
        {
            "assay": "RNA",
            "inputCells": 12000,
            "retainedCells": 12000,
            "filtering": False,
            "qcFlags": {
                "RNA_nFeatures": {
                    "low": 200,
                    "high": 6000,
                    "lowFlags": 14,
                    "highFlags": 23,
                    "missing": 7,
                    "outlierRows": {"low": [97], "high": [103]},
                },
                "RNA_percentMito": {
                    "low": None,
                    "high": 25,
                    "lowFlags": 0,
                    "highFlags": 31,
                    "missing": 0,
                },
            },
        }
    )
    _complete(records, "preprocess", "prepared", prepared)
    candidate = {
        "candidateId": "c0",
        "parentId": None,
        "hvgCount": 1000,
        "pcaDims": 21,
        "neighborsK": 11,
        "useHarmony": False,
    }
    partition = {
        "optionId": "c0:r0.5",
        "candidateId": "c0",
        "resolution": 0.5,
        "score": 0.42,
        "count": 12000,
        "clusterCount": 2,
        "clusterCounts": {"1": 8000, "2": 4000},
        "clusterCountsTruncated": False,
    }
    records.append(
        "candidateMeasured",
        candidateId="c0",
        evidence=records.write_evidence(
            "candidate",
            {
                "runId": "candidate-run",
                "parameters": candidate,
                "partitions": [partition],
                "silhouetteSampleCells": 2000,
                "limitations": [],
            },
        ),
    )
    finalist = {
        "runId": "selected-finalist",
        "parameters": candidate,
        "metrics": {
            "markerCoherence": 0.5,
            "markerSpecificityMedian": 0.87,
            "doubletHighScoreConcentration": None,
            "mixing": {},
            "protection": {},
        },
        "diagnosticScope": {"populationCells": 12000, "sampleCells": 10000},
        "clusters": [
            {
                "clusterId": "1",
                "count": 8000,
                "markers": [
                    {"gene": "CD3D", "score": 0.9, "fracExp": 0.75},
                    {"gene": "IL7R", "score": 0.8, "fracExp": 0.5},
                ],
                "qualifyingMarkerCount": 18,
            },
            {
                "clusterId": "2",
                "count": 4000,
                "markers": [],
                "qualifyingMarkerCount": 0,
            },
        ],
        "limitations": ["Rare populations may be absent from the diagnostic sample."],
    }
    records.append(
        "finalistMeasured",
        optionId="c0:r0.5",
        evidence=records.write_evidence("selected-finalist", finalist),
    )
    _complete(
        records,
        "finalists",
        "assessed",
        {
            "selected": "c0:r0.5",
            "finalists": {"c0:r0.5": finalist},
            "rejected": {},
        },
    )
    final = records.read_json("evidence/finalize.json")
    final["selected"] = "c0:r0.5"
    _complete(records, "finalize", "selected-final", final)
    records.append(
        "pipelinePlanned",
        operation="final",
        label="final-label",
        candidate=candidate,
        resolution=0.5,
    )
    records.append(
        "decisionAccepted",
        decisionId="select",
        stage="finalists",
        output={
            "action": "choose",
            "optionIds": ["c0:r0.5"],
            "rationale": "Retain the stable broad populations with measured markers.",
            "evidenceIds": ["c0:r0.5"],
        },
    )
    return records


def test_report_has_scientific_sections_tables_and_collapsed_provenance(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    document = _Document(AnalysisRun(records.path).report().read_text())
    tags = [tag for tag, _ in document.tags]
    identifiers = {attrs.get("id") for _, attrs in document.tags}
    assert "Single-cell RNA analysis" in document.headings
    assert {
        "overview",
        "populations",
        "quality",
        "methods",
        "provenance",
    } <= identifiers
    assert tags.count("table") >= 3
    table_regions = [
        attrs
        for tag, attrs in document.tags
        if tag == "div" and "table-wrap" in str(attrs.get("class", "")).split()
    ]
    assert len(table_regions) == tags.count("table")
    assert all(
        attrs.get("tabindex") == "0"
        and attrs.get("role") == "region"
        and attrs.get("aria-label")
        for attrs in table_regions
    )
    assert "pre" not in tags
    assert "script" not in tags
    assert any(tag == "details" and "open" not in attrs for tag, attrs in document.tags)
    primary = " ".join(document.primary_text)
    assert "12,000" in primary
    assert "T cells" in primary
    assert "unassigned" in primary.lower()
    assert "provisional" in primary.lower()
    for internal_name in (
        "hvgCount",
        "pcaDims",
        "neighborsK",
        "markerCoherence",
        "qcFlags",
        "retainedCells",
    ):
        assert internal_name not in primary
    assert "Marker coherence" in " ".join(document.text)


def test_report_assets_and_typography_are_embedded_without_remote_requests(
    tmp_path: Path,
) -> None:
    page = AnalysisRun(_records(tmp_path).path).report().read_text()
    document = _Document(page)
    images = [attrs for tag, attrs in document.tags if tag == "img"]
    assert any("nygen" in str(attrs.get("alt", "")).lower() for attrs in images)
    assert any("scarf" in str(attrs.get("alt", "")).lower() for attrs in images)
    assert all(str(attrs.get("src", "")).startswith("data:") for attrs in images)
    assert any(
        tag == "link"
        and "icon" in str(attrs.get("rel", ""))
        and str(attrs.get("href", "")).startswith("data:")
        for tag, attrs in document.tags
    )
    assert not any(
        str(attrs.get(key, "")).startswith(("http:", "https:", "//"))
        for _, attrs in document.tags
        for key in ("src", "href")
    )
    assert "@import" not in page
    assert re.search(r"font-family\s*:[^;}]*Inter", page)
    assert "#0077fc" in page.lower()
    assert re.search(r"line-height\s*:\s*1\.2\b", page)
    assert 'href="annotations.csv"' in page
    assert 'href="report.md"' in page


@pytest.mark.parametrize(
    ("asset_name", "filename"),
    [
        ("NYGEN_LOGO", "logo.png"),
        ("SCARF_LOGO", "logo_wide.png"),
        ("FAVICON", "favicon.ico"),
    ],
)
def test_embedded_brand_assets_match_supplied_company_files(
    asset_name: str, filename: str
) -> None:
    from scarf.agent import report_assets

    encoded = getattr(report_assets, asset_name).split(",", 1)[1]
    supplied = Path(__file__).parents[1] / "docs" / "source" / filename
    assert base64.b64decode(encoded, validate=True) == supplied.read_bytes()


def test_embedded_inter_font_retains_its_distribution_license(tmp_path: Path) -> None:
    from scarf.agent.report_assets import INTER_FONT, INTER_FONT_LICENSE

    assert base64.b64decode(INTER_FONT.split(",", 1)[1], validate=True).startswith(
        b"wOF2"
    )
    records = _records(tmp_path)
    page = AnalysisRun(records.path).report().read_text()
    assert "SIL OPEN FONT LICENSE Version 1.1" in INTER_FONT_LICENSE
    assert INTER_FONT_LICENSE in page
    visible = " ".join(_Document(page).text)
    markdown = (records.path / "report.md").read_text()
    for text in (visible, markdown):
        assert "Typography license" not in text
        assert "SIL OPEN FONT LICENSE Version 1.1" not in text
        assert "The Inter Project Authors" not in text


@pytest.mark.parametrize(
    ("status", "label"),
    [
        ("running", "In progress"),
        ("needsInput", "Awaiting input"),
        ("completed", "Completed"),
        ("failed", "Stopped"),
        ("interrupted", "Interrupted"),
    ],
)
def test_report_outcomes_have_readable_status_and_actionable_questions(
    tmp_path: Path, status: str, label: str
) -> None:
    records = _records(tmp_path, status=status, final=status == "completed")
    document = _Document(AnalysisRun(records.path).report().read_text())
    visible = " ".join(document.text)
    assert label in visible
    assert ("Choose an assay." in visible) == (status == "needsInput")
    if status != "completed":
        assert "T cells" not in visible
    assert AnalysisRun(records.path).status == status


def test_report_cluster_sizes_and_markers_use_only_the_selected_finalist(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    assessed = records.read_json("evidence/assessed.json")
    assessed["finalists"]["c1:r0.5"] = {
        "runId": "unselected-finalist",
        "clusters": [
            {
                "clusterId": "1",
                "count": 1234,
                "markers": [{"gene": "WRONG_FINALIST_MARKER", "score": 1.0}],
            }
        ],
    }
    _complete(records, "finalists", "assessed-with-alternative", assessed)
    document = _Document(AnalysisRun(records.path).report().read_text())
    population = next(row for row in document.rows if "T cells" in " ".join(row))
    assert "8,000" in " ".join(population)
    assert "Medium" in " ".join(population)
    assert "CD3D" in " ".join(population)
    assert "1,234" not in " ".join(population)
    assert "WRONG_FINALIST_MARKER" not in " ".join(document.text)
    assert "IL7R" in " ".join(document.text)
    assert "Observed CD3D." in " ".join(document.text)


def test_report_qc_flags_are_named_and_preserve_missing_measurements(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    document = _Document(AnalysisRun(records.path).report().read_text())
    qc_row = next(row for row in document.rows if "Detected genes" in " ".join(row))
    assert {"14", "23", "7"} <= set(qc_row)
    visible = " ".join(document.text)
    assert "Mitochondrial" in visible
    assert "retained" in visible.lower()
    assert "outlierRows" not in visible
    assert "lowFlags" not in visible
    assert "doubletHighScoreConcentration" not in visible


def test_report_missing_counts_and_evidence_are_unknown_not_zero(
    tmp_path: Path,
) -> None:
    records = RunRecords.create(
        tmp_path / "no-evidence",
        {
            "runId": "uninspected",
            "source": "../missing",
            "study": {"context": "Awaiting source validation.", "objective": "Explore"},
            "config": {},
        },
    )
    records.append(
        "status", status="failed", stage="inspect", message="Invalid source."
    )
    page = AnalysisRun(records.path).report().read_text()
    visible = " ".join(_Document(page).text)
    assert "Invalid source." in visible
    assert "Not recorded" in visible or "Not available" in visible
    assert "None" not in visible
    assert re.search(r"\bnan\b", visible, re.IGNORECASE) is None
    assert "T cells" not in visible
    assert (
        len(
            list(
                csv.DictReader(
                    io.StringIO((records.path / "annotations.csv").read_text())
                )
            )
        )
        == 0
    )


def test_report_escapes_supplied_text_in_every_scientific_section(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    attack = '<img src=x onerror="alert(1)"><svg onload="alert(2)">'
    study = records.manifest["study"]
    study.update(context=attack, objective=attack, tissue=attack, organism=attack)
    records.append("inputsResolved", study=study, config=records.manifest["config"])
    records.append(
        "candidateMeasured",
        candidateId=attack,
        evidence=records.write_evidence(
            "unsafe-candidate-name", records.read_json("evidence/candidate.json")
        ),
    )
    annotated = records.read_json("evidence/annotate.json")
    annotated["annotations"][0].update(
        identity=attack,
        supportingMarkers=[attack],
        rationale=attack,
    )
    _complete(records, "annotate", "unsafe-annotations", annotated)
    records.append("limitation", message=attack)
    records.append("modelFailure", stage="annotate", message=attack)
    records.append(
        "status",
        status="needsInput",
        stage="annotate",
        questions=[{"questionId": "clarify", "question": attack}],
    )
    page = AnalysisRun(records.path).report().read_text()
    document = _Document(page)
    assert attack not in page
    assert "&lt;img" in page
    assert "svg" not in [tag for tag, _ in document.tags]
    assert not any(
        name.startswith("on") for _, attrs in document.tags for name in attrs
    )
    assert attack in " ".join(document.text)
    assert "<script>" not in page
    markdown = (records.path / "report.md").read_text()
    assert attack not in markdown
    assert "<img" not in markdown
    assert "<svg" not in markdown
    assert "&lt;img" in markdown
    saved = list(
        csv.DictReader(io.StringIO((records.path / "annotations.csv").read_text()))
    )
    assert saved[0]["identity"] == attack


def test_report_inspection_does_not_claim_filtering_or_doublet_scoring_finished(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path, status="failed", final=False)
    configuration = records.manifest["config"]
    configuration.update(
        qcPolicy="manual", qcBounds={"RNA_nFeatures": [200, None]}, scoreDoublets=True
    )
    records.append(
        "inputsResolved", study=records.manifest["study"], config=configuration
    )
    prepared = records.read_json("evidence/inspect.json")
    prepared["filtering"] = {
        "method": "manual",
        "attrs": ["RNA_nFeatures"],
        "lows": [200],
        "highs": [None],
    }
    _complete(records, "inspect", "inspected-filter-policy", prepared)
    visible = " ".join(_Document(AnalysisRun(records.path).report().read_text()).text)
    assert "configured" in visible.lower()
    assert "filtering was applied" not in visible.lower()
    assert "doublet scoring is requested" in visible.lower()
    assert "scoring was completed" not in visible.lower()
    assert "scores were computed" not in visible.lower()
    assert AnalysisRun(records.path).pipeline_runs == []


def test_report_partial_usage_shows_known_totals_and_unknown_measurements(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    records.append(
        "modelResponse",
        usage={
            "inputTokens": 1000,
            "outputTokens": 25,
            "cacheReadTokens": 0,
            "cacheWriteTokens": None,
        },
    )
    records.append("modelResponse", usage=None)
    records.append("modelResponse", usage={"inputTokens": None, "outputTokens": 75})
    document = _Document(AnalysisRun(records.path).report().read_text())
    inputs = next(row for row in document.rows if "Input tokens" in row)
    outputs = next(row for row in document.rows if "Output tokens" in row)
    cache_reads = next(row for row in document.rows if "Tokens read from cache" in row)
    cache_writes = next(
        row for row in document.rows if "Tokens written to cache" in row
    )
    assert "1,000" in inputs
    assert "1 of 3 recorded responses" in inputs
    assert "100" in outputs
    assert "2 of 3 recorded responses" in outputs
    assert "0" in cache_reads
    assert "Not recorded" in cache_writes or "Not available" in cache_writes
    assert "0" not in cache_writes
    assert "Unavailable usage is unknown." in " ".join(document.text)


def test_report_harmony_assessment_preserves_biological_and_technical_diagnostics(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    finalist = records.read_json("evidence/selected-finalist.json")
    finalist["parameters"]["useHarmony"] = True
    finalist["metrics"].update(
        mixing={"sequencing_batch": 0.71},
        protection={"treatment": {"cLISI": 0.93, "graphConnectivity": 0.86}},
        crossUnitSupport=0.75,
        doubletHighScoreConcentration=1.6,
    )
    records.append(
        "finalistMeasured",
        optionId="c1:r0.5",
        evidence=records.write_evidence("corrected-finalist", finalist),
    )
    final = records.read_json("evidence/selected-final.json")
    final["selected"] = "c1:r0.5"
    _complete(records, "finalize", "corrected-final", final)
    _complete(
        records,
        "finalists",
        "corrected-assessment",
        {
            "selected": "c1:r0.5",
            "finalists": {"c1:r0.5": finalist},
            "rejected": {},
        },
    )
    records.append(
        "pipelinePlanned",
        operation="final",
        label="corrected-final-label",
        candidate={**finalist["parameters"], "candidateId": "c1"},
        resolution=0.5,
    )
    document = _Document(AnalysisRun(records.path).report().read_text())
    visible = " ".join(document.text)
    assert "Harmony" in visible
    for label, value in (
        ("Batch mixing", "0.71"),
        ("Biological separation", "0.93"),
        ("Graph connectivity", "0.86"),
        ("Support across samples", "75.0%"),
        ("Concentration of high doublet scores", "1.6"),
    ):
        rows = [row for row in document.rows if label in " ".join(row)]
        assert any(value in " ".join(row) for row in rows)
    assert "sequencing_batch" in visible or "sequencing batch" in visible
    assert "treatment" in visible
    assert "crossUnitSupport" not in visible
    assert "doubletHighScoreConcentration" not in visible


def test_report_does_not_open_source_or_mutate_frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.evidence as evidence

    records = _rich_records(tmp_path)
    before = {
        path.relative_to(records.path): path.read_bytes()
        for path in records.path.rglob("*")
        if path.is_file()
    }

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Offline rendering must not access a numerical store")

    monkeypatch.setattr(evidence, "open_store", forbidden)
    first = AnalysisRun(records.path).report().read_bytes()
    second = AnalysisRun(records.path).report().read_bytes()
    assert first == second
    for relative, original in before.items():
        assert (records.path / relative).read_bytes() == original
    assert not AnalysisRun(records.path).source.exists()


def test_large_population_report_bounds_visible_rows_but_exports_every_annotation(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    annotated = records.read_json("evidence/annotate.json")
    annotated["annotations"] = [
        {
            "clusterId": str(index),
            "identity": f"Population {index:03d}",
            "confidence": "low",
            "supportingMarkers": [],
            "contradictingMarkers": [],
            "rationale": f"Observed evidence for population {index:03d}.",
        }
        for index in range(102)
    ]
    _complete(records, "annotate", "many-populations", annotated)
    before = records.events()
    page = AnalysisRun(records.path).report().read_text()
    document = _Document(page)
    visible = " ".join(document.text)
    assert "Population 099" in visible
    assert "Population 100" not in visible
    assert "Additional entries: 2; see saved records." in visible
    assert "102" in visible
    saved = list(
        csv.DictReader(io.StringIO((records.path / "annotations.csv").read_text()))
    )
    assert len(saved) == 102
    assert saved[-1]["identity"] == "Population 101"
    assert records.events() == before
    assert len(AnalysisRun(records.path).annotations) == 102


def test_report_sorts_cluster_labels_naturally_without_reordering_saved_annotations(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    annotated = records.read_json("evidence/annotate.json")
    supplied = ["10", "2", "alpha", "1", "Beta", "zeta"]
    expected = ["1", "2", "10", "alpha", "Beta", "zeta"]
    template = annotated["annotations"][0]
    annotated["annotations"] = [
        {**template, "clusterId": identifier, "identity": f"Population {identifier}"}
        for identifier in supplied
    ]
    _complete(records, "annotate", "unordered-clusters", annotated)
    before = records.events()
    document = _Document(AnalysisRun(records.path).report().read_text())
    populations = [
        row
        for row in document.rows
        if len(row) > 1 and row[1].startswith("Population ")
    ]
    assert [row[0] for row in populations] == expected
    details = [
        text
        for text in document.text
        if text.startswith("Cluster ") and "Population " in text
    ]
    assert details == [
        f"Cluster {identifier} · Population {identifier}" for identifier in expected
    ]
    saved = list(
        csv.DictReader(io.StringIO((records.path / "annotations.csv").read_text()))
    )
    assert [row["clusterId"] for row in saved] == supplied
    assert [
        row["clusterId"] for row in AnalysisRun(records.path).annotations
    ] == supplied
    assert records.events() == before


def test_report_embeds_existing_local_umap_without_opening_the_source(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    (records.path / "umap_clusters.png").write_bytes(_PNG)
    document = _Document(AnalysisRun(records.path).report().read_text())
    previews = [
        attrs
        for tag, attrs in document.tags
        if tag == "img" and "umap" in str(attrs.get("alt", "")).lower()
    ]
    assert len(previews) == 1
    assert (
        previews[0]["src"] == "data:image/png;base64," + base64.b64encode(_PNG).decode()
    )
    assert "Saved UMAP preview" in " ".join(document.text)


@pytest.mark.parametrize("preview", ["missing", "symlink", "oversized", "invalid"])
def test_report_omits_unavailable_or_unsafe_previews_without_losing_results(
    tmp_path: Path, preview: str
) -> None:
    records = _rich_records(tmp_path)
    path = records.path / "umap_clusters.png"
    if preview == "symlink":
        outside = tmp_path / "external.png"
        outside.write_bytes(_PNG)
        path.symlink_to(outside)
    elif preview == "oversized":
        with path.open("wb") as stream:
            stream.write(_PNG)
            stream.seek(8 * 1024 * 1024)
            stream.write(b"x")
    elif preview == "invalid":
        path.write_text("<svg onload='alert(1)'>")
    document = _Document(AnalysisRun(records.path).report().read_text())
    assert not any(
        tag == "img" and "umap" in str(attrs.get("alt", "")).lower()
        for tag, attrs in document.tags
    )
    visible = " ".join(document.text)
    assert "T cells" in visible
    assert "Completed" in visible
    assert "preview" in visible.lower()
    assert "not computed" not in visible.lower()


@pytest.mark.parametrize("mismatch", ["runId", "clusters"])
def test_report_refuses_annotations_from_another_frozen_clustering(
    tmp_path: Path, mismatch: str
) -> None:
    records = _rich_records(tmp_path)
    saved = records.read_json("evidence/annotate.json")
    if mismatch == "runId":
        saved["runId"] = "unrelated-run"
    else:
        saved["clusters"] = {**saved["clusters"], "artifact_id": "d" * 64}
    _complete(records, "annotate", "wrong-clustering", saved)
    with pytest.raises(RecordError, match="final frozen clustering"):
        AnalysisRun(records.path).report()
    assert not (records.path / "report.html").exists()
    assert AnalysisRun(records.path).status == "completed"
