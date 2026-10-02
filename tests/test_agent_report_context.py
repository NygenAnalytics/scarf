"""Inline source metadata becomes bounded tables without reopening the source."""

import html
import json
from pathlib import Path

import pytest

from scarf.agent.report_html import _study_context
from scarf.agent.result import AnalysisRun

from .test_agent_report import _Document
from .test_agent_result import _records


def _render(context: str) -> tuple[_Document, str, str]:
    content = _study_context(context)
    page = "".join(content.html)
    return _Document(page), page, "\n".join(content.markdown)


def test_study_metadata_is_readable_in_offline_report(tmp_path: Path) -> None:
    records = _records(tmp_path)
    metadata = {
        "tissue": {
            "distinct": 2,
            "missing": 0,
            "levels": {"sigmoid colon": 33396, "rectum": 4922},
        },
        "donor_id": {
            "distinct": 1,
            "missing": 0,
            "levels": {"pooled": 38318},
        },
    }
    study = records.manifest["study"]
    study["context"] = (
        "**Descriptive discovery.**\nObserved source metadata: "
        + json.dumps(metadata)
        + "\nDonor and capture counts remain uncertain."
    )
    records.append("inputsResolved", study=study, config=records.manifest["config"])
    saved = {path: path.read_bytes() for path in records.path.rglob("*.json")}

    AnalysisRun(records.path).report()

    page = (records.path / "report.html").read_text()
    document = _Document(page)
    overview = " ".join(document.sections["overview"])
    assert "Observed source metadata" in overview
    assert "Observed source metadata: {" not in overview
    assert "Descriptive discovery." in overview
    assert "Donor and capture counts remain uncertain." in overview
    assert ["Tissue", "2", "0"] in document.rows
    assert ["Donor ID", "1", "0"] in document.rows
    assert ["sigmoid colon", "33,396"] in document.rows
    assert ["pooled", "38,318"] in document.rows
    assert "Tissue: category counts" in overview
    assert "<strong>Descriptive discovery.</strong>" in page
    markdown = (records.path / "report.md").read_text()
    assert "| Tissue | 2 | 0 |" in markdown
    assert "| sigmoid colon | 33,396 |" in markdown
    assert all(path.read_bytes() == value for path, value in saved.items())


def test_study_metadata_decoding_precedes_narrative_truncation() -> None:
    metadata = {"sample_id": {"distinct": 1, "levels": {"sample-A": 12000}}}
    context = (
        "Long study description. " * 300
        + "Observed source metadata:\n  "
        + json.dumps(metadata)
        + "\nFollow-up context remains visible."
    )
    document, page, markdown = _render(context)
    assert ["Sample ID", "1", "Not recorded"] in document.rows
    assert ["sample-A", "12,000"] in document.rows
    assert "Follow-up context remains visible." in page
    assert "Follow-up context remains visible." in markdown
    assert "see saved records" in page


def test_study_metadata_preserves_zero_missing_and_unrecorded_counts() -> None:
    metadata = {
        "empty_column": {"distinct": 0, "missing": 15, "levels": {}},
        "unknown_column": {"distinct": None, "missing": None},
        "partial_column": {"levels": {"absent": 0, "unrecorded": None}},
    }
    document, page, _ = _render("Observed source metadata: " + json.dumps(metadata))
    assert ["Empty column", "0", "15"] in document.rows
    assert ["Unknown column", "Not recorded", "Not recorded"] in document.rows
    assert ["absent", "0"] in document.rows
    assert ["unrecorded", "Not recorded"] in document.rows
    assert "No measurements recorded." in page


@pytest.mark.parametrize(
    "context",
    [
        "Ordinary study context without metadata.",
        'Observed source metadata: {"tissue":',
        "Observed source metadata: []",
        'Observed source metadata: {"tissue": []}',
        'Observed source metadata: {"tissue": {"unknown": 4}}',
        'Observed source metadata: {"tissue": {"levels": []}}',
        'Observed source metadata: {"tissue": {"distinct": -1}}',
        'Observed source metadata: {"tissue": {"missing": true}}',
        'Observed source metadata: {"tissue": {"distinct": 1.5}}',
        'Observed source metadata: {"tissue": {"levels": {"a": "four"}}}',
    ],
)
def test_unrecognized_metadata_remains_original_safe_prose(context: str) -> None:
    document, _, markdown = _render(context)
    assert " ".join(document.text) == context
    assert not document.rows
    assert not document.headings
    assert html.unescape(markdown).strip() == context


def test_extremely_nested_metadata_falls_back_to_bounded_prose() -> None:
    _, page, _ = _render("Observed source metadata: " + "[" * 2000 + "]" * 2000)
    assert "Observed source metadata:" in page
    assert "see saved records" in page


def test_metadata_labels_and_values_are_escaped_in_both_formats() -> None:
    metadata = {
        '<script>alert("field")</script>': {
            "distinct": 1,
            "missing": 0,
            "levels": {'<img src=x onerror="alert(1)">|a\nb': 4},
        }
    }
    document, page, markdown = _render(
        "Before <script>alert(2)</script>. Observed source metadata: "
        + json.dumps(metadata)
        + " After <script>alert(3)</script>."
    )
    assert not any(tag in {"script", "img"} for tag, _ in document.tags)
    assert "&lt;script&gt;" in page
    assert "&lt;img" in page
    assert "&lt;script&gt;" in markdown
    assert "\\|a b" in markdown
    assert "alert(3)" in " ".join(document.text)


def test_metadata_fields_and_categories_are_bounded() -> None:
    metadata = {
        f"field_{index}": {"distinct": 1, "missing": 0, "levels": {"known": 1}}
        for index in range(101)
    }
    metadata["field_0"]["distinct"] = 101
    metadata["field_0"]["levels"] = {f"category-{index}": index for index in range(101)}
    document, page, markdown = _render(
        "Observed source metadata: " + json.dumps(metadata) + " Remaining context."
    )
    assert ["Field 99", "1", "0"] in document.rows
    assert "Field 100" not in page
    assert ["category-99", "99"] in document.rows
    assert "category-100" not in page
    assert page.count("Additional entries: 1; see saved records.") == 2
    assert markdown.count("Additional entries: 1; see saved records.") == 2
    assert "Remaining context." in page


def test_empty_metadata_has_explicit_missing_measurements() -> None:
    document, page, markdown = _render("Observed source metadata: {}")
    assert document.headings == ["Observed source metadata"]
    assert "No measurements recorded." in page
    assert "No measurements recorded." in markdown
