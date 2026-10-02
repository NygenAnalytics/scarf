"""Cluster size charts remain exact, accessible and independent of source data."""

import base64
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from scarf.agent.records import RecordError
from scarf.agent.report_charts import cluster_size_svg
from scarf.agent.result import AnalysisRun

from .test_agent_report import _complete, _Document, _rich_records
from .test_agent_result import _records

_SVG = "{http://www.w3.org/2000/svg}"


def _svg_text(svg: str) -> tuple[ElementTree.Element, list[str]]:
    root = ElementTree.fromstring(svg)
    labels = ["".join(node.itertext()) for node in root.iter(f"{_SVG}text")]
    return root, labels


def _report_svg(page: str) -> tuple[_Document, str]:
    document = _Document(page)
    image = next(
        attrs
        for tag, attrs in document.section_tags["results"]
        if tag == "img" and attrs.get("class") == "cluster-size-plot"
    )
    assert image.get("alt")
    assert str(image["src"]).startswith("data:image/svg+xml;base64,")
    return document, base64.b64decode(str(image["src"]).split(",", 1)[1]).decode()


def test_cluster_size_svg_retains_numeric_order_counts_and_zero_origin() -> None:
    rows = [
        {"clusterId": "10", "count": 4000},
        {"clusterId": "2", "count": 2000},
        {"clusterId": "1", "count": 1000},
    ]
    svg = cluster_size_svg(rows)
    assert svg is not None
    root, labels = _svg_text(svg)
    assert [label for label in labels if label in {"1", "2", "10"}] == [
        "1",
        "2",
        "10",
    ]
    assert {"1,000", "2,000", "4,000", "0"} <= set(labels)
    assert "Cells" in labels
    assert [row["clusterId"] for row in rows] == ["10", "2", "1"]
    assert root.find(f"{_SVG}title") is not None
    assert root.find(f"{_SVG}desc") is not None

    # Three measured counts must share a zero origin and preserve their ratios.
    bars = [
        element
        for element in root.iter(f"{_SVG}path")
        if "#0077fc" in element.get("style", "").lower()
    ]
    assert len(bars) == 3
    coordinates = [
        [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", bar.attrib["d"])]
        for bar in bars
    ]
    baselines = [points[1] for points in coordinates]
    heights = [max(points[1::2]) - min(points[1::2]) for points in coordinates]
    assert baselines == pytest.approx([baselines[0]] * 3)
    assert heights[1] / heights[0] == pytest.approx(2)
    assert heights[2] / heights[0] == pytest.approx(4)


@pytest.mark.parametrize(
    "count", [None, -1, float("nan"), float("inf"), True, "12", 1.5]
)
def test_missing_or_invalid_counts_are_never_claimed_as_zero(count: Any) -> None:
    svg = cluster_size_svg(
        [{"clusterId": "zero", "count": 0}, {"clusterId": "unknown", "count": count}]
    )
    assert svg is not None
    root, labels = _svg_text(svg)
    assert labels.count("N/A") == 1
    assert "0" in labels
    description = root.find(f"{_SVG}desc")
    assert description is not None
    assert "not recorded" in "".join(description.itertext()).lower()
    assert {"zero", "unknown"} <= set(labels)


def test_all_unknown_counts_and_empty_evidence_are_explicit() -> None:
    assert cluster_size_svg([]) is None
    svg = cluster_size_svg([{"clusterId": "1"}, {"clusterId": "2", "count": None}])
    assert svg is not None
    _, labels = _svg_text(svg)
    assert labels.count("N/A") == 2


def test_chart_is_deterministic_and_ignores_excess_groups_after_natural_order() -> None:
    rows = [
        {"clusterId": str(index), "count": index + 1000}
        for index in reversed(range(102))
    ]
    first = cluster_size_svg(rows)
    assert first is not None
    assert cluster_size_svg(rows) == first
    root, labels = _svg_text(first)
    assert "1,099" in labels
    assert "1,100" not in labels
    assert "1,101" not in labels
    assert (
        len(
            [
                element
                for element in root.iter(f"{_SVG}path")
                if "#0077fc" in element.get("style", "").lower()
            ]
        )
        == 100
    )
    assert len(first.encode()) < 400_000


def test_cluster_labels_are_safe_plain_svg_text() -> None:
    malicious = r"<script>alert(1)</script> & $\notacommand$"
    math_like = r"$\bad$"
    svg = cluster_size_svg(
        [
            {"clusterId": malicious, "count": 25},
            {"clusterId": math_like, "count": 8},
            {"clusterId": "A&B", "count": 4},
        ]
    )
    assert svg is not None
    root, labels = _svg_text(svg)
    assert {math_like, "A&B"} <= set(labels)
    assert any(label.endswith("…") for label in labels)
    description = root.find(f"{_SVG}desc")
    assert description is not None
    assert malicious in "".join(description.itertext())
    assert not any(
        element.tag.rsplit("}", 1)[-1] in {"script", "foreignObject", "a", "image"}
        for element in root.iter()
    )
    assert not any(
        attribute.lower().startswith("on")
        for element in root.iter()
        for attribute in element.attrib
    )


def test_chart_preserves_existing_figures_and_matplotlib_settings() -> None:
    from matplotlib import pyplot as plt, rcParams

    existing = plt.figure()
    figures = plt.get_fignums()
    settings = dict(rcParams)
    try:
        assert cluster_size_svg([{"clusterId": "1", "count": 20}])
        assert plt.get_fignums() == figures
        assert dict(rcParams) == settings
    finally:
        plt.close(existing)


def test_report_uses_exact_selected_final_counts_without_accessing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.evidence as evidence

    records = _rich_records(tmp_path)
    assessed = records.read_json("evidence/assessed.json")
    assessed["finalists"]["c1:r0.5"] = {
        "clusters": [{"clusterId": "other", "count": 98765}]
    }
    _complete(records, "finalists", "assessed-chart", assessed)
    original = {
        path.relative_to(records.path): path.read_bytes()
        for path in records.path.rglob("*")
        if path.is_file()
    }

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Cluster counts must be rendered from saved evidence only")

    monkeypatch.setattr(evidence, "open_store", forbidden)
    monkeypatch.setattr(evidence, "verify_source", forbidden)
    page = AnalysisRun(records.path).report().read_text()
    document, svg = _report_svg(page)
    assert (records.path / "cluster_sizes.svg").read_text() == svg
    assert (records.path / "cluster_sizes.svg").read_bytes() == svg.encode()
    _, labels = _svg_text(svg)
    assert {"8,000", "4,000"} <= set(labels)
    assert "98,765" not in labels
    assert "other" not in labels
    assert "cluster_sizes.svg" in (records.path / "report.md").read_text()
    assert not any(tag == "table" for tag, _ in document.section_tags["results"])
    assert any(tag == "figcaption" for tag, _ in document.section_tags["results"])
    assert AnalysisRun(records.path).report().read_text() == page
    for relative, value in original.items():
        assert (records.path / relative).read_bytes() == value
    assert not AnalysisRun(records.path).source.exists()


def test_report_shows_bounded_chart_with_explicit_omission_count(
    tmp_path: Path,
) -> None:
    records = _rich_records(tmp_path)
    assessed = records.read_json("evidence/assessed.json")
    assessed["finalists"]["c0:r0.5"]["clusters"] = [
        {"clusterId": str(index), "count": index + 1000}
        for index in reversed(range(103))
    ]
    _complete(records, "finalists", "many-chart-groups", assessed)
    document, svg = _report_svg(AnalysisRun(records.path).report().read_text())
    visible = " ".join(document.sections["results"])
    assert "3" in visible
    assert "additional" in visible.lower()
    _, labels = _svg_text(svg)
    assert "1,099" in labels
    assert "1,100" not in labels
    assert "1,102" not in labels


def test_report_without_selected_finalist_does_not_invent_cluster_counts(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path)
    document = _Document(AnalysisRun(records.path).report().read_text())
    assert not (records.path / "cluster_sizes.svg").exists()
    assert not any(
        attrs.get("class") == "cluster-size-plot"
        for _, attrs in document.section_tags["results"]
    )
    assert "not" in " ".join(document.sections["results"]).lower()


def test_report_refuses_to_replace_a_cluster_chart_symlink(tmp_path: Path) -> None:
    records = _rich_records(tmp_path)
    outside = tmp_path / "preserve.svg"
    outside.write_text("preserve this file")
    (records.path / "cluster_sizes.svg").symlink_to(outside)
    with pytest.raises(RecordError, match="symlink"):
        AnalysisRun(records.path).report()
    assert outside.read_text() == "preserve this file"
    assert (records.path / "cluster_sizes.svg").is_symlink()
