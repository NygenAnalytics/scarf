"""Human-readable presentation of saved evidence, without numerical execution."""

import html
import json
import math
import re
from typing import Any

from .report_assets import (
    FAVICON,
    INTER_FONT,
    INTER_FONT_LICENSE,
    NYGEN_LOGO,
    SCARF_LOGO,
)
from .report_style import STYLE
from .report_text import narrative_html, normalize_narrative

_LIMIT = 100
_UNKNOWN = "Not recorded"
_NYGEN_URL = "https://www.nygen.io/"
_SCARF_URL = "https://github.com/nygenAnalytics/scarf"
_PAPER_URL = "https://doi.org/10.1038/s41467-022-32097-3"
_PAPER_CITATION = (
    "Dhapola, P., Rodhe, J., Olofzon, R. et al. Scarf enables a highly "
    "memory-efficient analysis of large-scale single-cell genomics data. "
    "Nat Commun 13, 4616 (2022)."
)
_STAGES = {
    "inspect": "Inspect the supplied data",
    "context": "Establish study context",
    "preprocess": "Set quality and feature policies",
    "explore": "Explore clustering settings",
    "finalists": "Compare the shortlisted results",
    "finalize": "Finalize clustering and UMAP",
    "annotate": "Assign provisional cell identities",
    "report": "Prepare the report",
}
_STATUSES = {
    "running": "In progress",
    "needsInput": "Awaiting input",
    "completed": "Completed",
    "failed": "Stopped",
    "interrupted": "Interrupted",
}
_LABELS = {
    "hvgCount": "variable genes",
    "pcaDims": "principal components",
    "neighborsK": "neighbors",
    "useHarmony": "Harmony correction",
    "markerCoherence": "marker coherence",
    "markerSupportFraction": "marker support",
    "markerSpecificityMedian": "median marker specificity",
    "fracExpRest": "expression outside the cluster",
    "fracExp": "expression within the cluster",
    "RNA_nFeatures": "detected genes",
    "RNA_nCounts": "total RNA counts",
    "RNA_percentMito": "mitochondrial RNA percentage",
    "RNA_percentRibo": "ribosomal RNA percentage",
    "sampleColumn": "sample grouping",
    "captureColumn": "capture grouping",
    "technicalBatchColumns": "technical batch labels",
    "offeredFeatureExclusions": "offered feature exclusions",
    "mitochondrialFeatures": "mitochondrial genes",
    "qualifyingMarkerCount": "qualifying markers",
}
_QC_LABELS = {
    "nFeatures": "Detected genes",
    "nCounts": "Total RNA counts",
    "percentMito": "Mitochondrial RNA (%)",
    "percentRibo": "Ribosomal RNA (%)",
}


def _text(value: Any, limit: int = 4000) -> str:
    text = _UNKNOWN if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + " … [see saved records]"


def _escape(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _number(value: Any, *, percent: bool = False) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return _UNKNOWN
    if percent:
        return f"{value * 100:,.1f}%"
    return (
        f"{value:,}"
        if isinstance(value, int)
        else f"{value:,.3f}".rstrip("0").rstrip(".")
    )


def _option(value: str) -> str:
    match = re.fullmatch(r"c(\d+)(?::r([\d.]+))?", value)
    if not match:
        return value
    label = "Baseline" if match[1] == "0" else f"Alternative {match[1]}"
    return label + (f", resolution {match[2]}" if match[2] else "")


def _readable(value: Any) -> str:
    text = _text(value)
    for key, label in _LABELS.items():
        text = text.replace(key, label)
    return re.sub(r"\bc\d+(?::r[\d.]+)?\b", lambda match: _option(match[0]), text)


class _Content:
    """Build escaped HTML and readable Markdown from the same displayed values."""

    def __init__(self) -> None:
        self.html: list[str] = []
        self.markdown: list[str] = []

    def paragraph(self, value: Any, style: str = "", limit: int = 4000) -> None:
        text = _text(value, limit)
        self.html.append(f'<div class="narrative {style}">{narrative_html(text)}</div>')
        self.markdown.extend([html.escape(normalize_narrative(text)), ""])

    def heading(self, title: str) -> None:
        self.html.append(f"<h3>{_escape(title)}</h3>")
        self.markdown.extend([f"### {_escape(title)}", ""])

    def omitted(self, total: int) -> None:
        if total > _LIMIT:
            self.paragraph(
                f"Additional entries: {total - _LIMIT}; see saved records.", "caption"
            )

    def bullets(self, values: list[str]) -> None:
        self.html.append("<ul>")
        for value in values[:_LIMIT]:
            text = _text(value)
            self.html.append(f'<li class="narrative">{narrative_html(text)}</li>')
            self.markdown.append(f"- {html.escape(normalize_narrative(text))}")
        self.html.append("</ul>")
        self.markdown.append("")
        self.omitted(len(values))

    def table(
        self, headers: list[str], rows: list[list[Any]], *, prose: tuple[int, ...] = ()
    ) -> None:
        if not rows:
            self.paragraph("No measurements recorded.", "empty")
            return
        self.html.append(
            f'<div class="table-wrap" tabindex="0" role="region" '
            f'aria-label="Table: {_escape(headers[0])}"><table><thead><tr>'
        )
        self.html.extend(f'<th scope="col">{_escape(h)}</th>' for h in headers)
        self.html.append("</tr></thead><tbody>")
        self.markdown.extend(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
        )
        for row in rows[:_LIMIT]:
            self.html.append(
                "<tr>"
                + "".join(
                    f'<td class="narrative">{narrative_html(_text(v))}</td>'
                    if index in prose
                    else f"<td>{_escape(v)}</td>"
                    for index, v in enumerate(row)
                )
                + "</tr>"
            )
            cells = [
                html.escape(
                    normalize_narrative(_text(v)) if index in prose else _text(v)
                )
                .replace("|", "\\|")
                .replace("\n", " ")
                for index, v in enumerate(row)
            ]
            self.markdown.append("| " + " | ".join(cells) + " |")
        self.html.append("</tbody></table></div>")
        self.markdown.append("")
        self.omitted(len(rows))

    def details(self, title: str, content: "_Content") -> None:
        self.html.append(
            f'<details><summary>{_escape(title)}</summary><div class="detail-body">'
            + "".join(content.html)
            + "</div></details>"
        )
        self.markdown.extend([f"### {_escape(title)}", "", *content.markdown])


def _section(
    document: _Content,
    key: str,
    number: int,
    title: str,
    content: _Content,
    *,
    state: str,
    outcome: str,
    expanded: bool,
) -> None:
    state_label = {
        **_STATUSES,
        "partial": "Incomplete",
        "unavailable": "Not recorded",
        "pending": "Not reached",
    }[state]
    document.html.append(
        f'<section id="{key}" aria-labelledby="heading-{key}">'
        f'<details class="workflow-step" data-stage-status="{state}"'
        + (" open" if expanded else "")
        + "><summary>"
        f'<span class="section-number" aria-hidden="true">{number:02}</span>'
        f'<h2 id="heading-{key}">{_escape(title)}</h2>'
        f'<span class="step-status">{state_label}</span>'
        f'<span class="step-outcome">{_escape(outcome)}</span></summary>'
        '<div class="step-body">' + "".join(content.html) + "</div></details></section>"
    )
    document.markdown.extend(
        [
            f"## {number:02}. {title}",
            "",
            f"{state_label}: {_escape(outcome)}",
            "",
            *content.markdown,
        ]
    )


def _step_state(data: dict[str, Any], stages: tuple[str, ...]) -> str:
    completed = set(data["completedStages"])
    if data["stage"] in stages and data["status"] != "completed":
        return str(data["status"])
    if set(stages) <= completed:
        return "completed"
    if set(stages) & (completed | set(data["startedStages"])):
        return "partial"
    order = list(_STAGES)
    later = set(order[order.index(stages[-1]) + 1 :])
    if data["status"] == "completed" or later & (
        completed | set(data["startedStages"])
    ):
        return "unavailable"
    return "pending"


def _outcomes(data: dict[str, Any]) -> dict[str, str]:
    annotations = data["annotations"]
    unassigned = sum(
        row.get("identity", "").lower() == "unassigned" for row in annotations
    )
    candidates = len(data["candidateEvidence"])
    selected = data.get("selectedOption")
    return {
        "overview": f"{_number(data['inputCells'])} supplied cells; study context and metadata.",
        "quality": (
            f"{_number(data['retainedCells'])} of {_number(data['inputCells'])} cells in the prepared cohort."
            if "preprocess" in data["completedStages"]
            else "Quality measurements and preparation policy."
        ),
        "exploration": f"Candidate analyses measured: {candidates}."
        if candidates
        else "No candidate measurements saved yet.",
        "selection": f"Selected: {_option(selected)}."
        if selected
        else "No final clustering selected yet.",
        "results": "Final UMAP, marker expression and cluster sizes."
        if data["final"]
        else "Final numerical results are not yet available.",
        "populations": f"{len(annotations)} clusters reviewed; {unassigned} unassigned."
        if annotations
        else "Provisional cell identities are not yet available.",
    }


def _study_context(text: str) -> _Content:
    """Present a supplied metadata summary without changing the saved context."""
    content = _Content()
    before, marker, after = text.partition("Observed source metadata:")
    if not marker:
        content.paragraph(text, "source-text")
        return content
    after = after.lstrip()
    try:
        metadata, end = json.JSONDecoder().raw_decode(after)
    except (ValueError, RecursionError):
        content.paragraph(text, "source-text")
        return content
    if not _is_metadata_summary(metadata):
        content.paragraph(text, "source-text")
        return content
    if before.strip():
        content.paragraph(before.strip(), "source-text")
    content.heading("Observed source metadata")
    fields = list(metadata.items())
    content.table(
        ["Field", "Distinct values", "Missing values"],
        [
            [
                _metadata_label(name),
                _number(summary.get("distinct")),
                _number(summary.get("missing")),
            ]
            for name, summary in fields
        ],
    )
    for name, summary in fields[:_LIMIT]:
        counts = _Content()
        counts.table(
            ["Value", "Cells"],
            [
                [value, _number(count)]
                for value, count in summary.get("levels", {}).items()
            ],
        )
        content.details(f"{_metadata_label(name)}: category counts", counts)
    if after[end:].strip():
        content.paragraph(after[end:].strip(), "source-text")
    return content


def _metadata_label(value: str) -> str:
    return re.sub(r"\bid\b", "ID", value.replace("_", " ").capitalize())


def _is_metadata_summary(value: Any) -> bool:
    """Recognize only the aggregate schema whose measurements can be displayed."""
    if not isinstance(value, dict):
        return False
    for summary in value.values():
        if not isinstance(summary, dict) or set(summary) - {
            "distinct",
            "missing",
            "levels",
        }:
            return False
        levels = summary.get("levels", {})
        if not isinstance(levels, dict):
            return False
        counts = [summary.get("distinct"), summary.get("missing"), *levels.values()]
        if any(
            count is not None and (type(count) is not int or count < 0)
            for count in counts
        ):
            return False
    return True


def _overview(data: dict[str, Any]) -> _Content:
    content = _Content()
    before = data["inputCells"]
    facts = [
        ("Tissue", data["study"].get("tissue") or "Not supplied"),
        ("Organism", data["study"].get("organism") or "Not supplied"),
        ("Assay", data["prepared"].get("assay")),
        (
            "Input cohort",
            f"{_number(before)} cells" if before is not None else _UNKNOWN,
        ),
        (
            "Progress",
            "Analysis complete"
            if data["status"] == "completed"
            else _STAGES.get(data["stage"], _UNKNOWN),
        ),
    ]
    if "interactionMode" in data["config"]:
        facts.append(
            ("Uncertainty policy", str(data["config"]["interactionMode"]).capitalize())
        )
    content.html.append('<dl class="facts">')
    for label, value in facts:
        content.html.append(f"<div><dt>{label}</dt><dd>{_escape(value)}</dd></div>")
        content.markdown.append(f"- {label}: {_escape(value)}")
    content.html.append("</dl>")
    content.markdown.append("")
    context = _study_context(
        data["study"].get("context") or "No study context supplied."
    )
    content.details("Supplied study context", context)
    roles = data.get("resolvedRoles")
    if roles is not None:
        detail = _Content()
        detail.table(
            ["Metadata field", "Role", "Source", "Authority"],
            [
                [
                    _metadata_label(row["column"]),
                    row.get("role"),
                    row.get("source"),
                    "Diagnostic only"
                    if row.get("authority") == "diagnosticOnly"
                    else row.get("authority"),
                ]
                for row in roles
            ],
        )
        detail.paragraph(
            "Inferred groupings support descriptive checks. They do not authorize technical correction or establish independent biological samples.",
            "caption",
        )
        design = data["prepared"].get("designDiagnostics", {})
        for crossing in design.get("crossTabs", [])[:_LIMIT]:
            detail.heading(
                f"{_metadata_label(crossing['leftColumn'])} and {_metadata_label(crossing['rightColumn'])}"
            )
            detail.table(
                ["First group", "Second group", "Cells"],
                [
                    [row.get("left"), row.get("right"), _number(row.get("count"))]
                    for row in crossing.get("counts", [])
                ],
            )
            detail.paragraph(
                f"Rows with missing labels: {_number(crossing.get('rowsMissing'))}.",
                "caption",
            )
        detail.omitted(len(design.get("crossTabs", [])))
        if design.get("limitations"):
            detail.bullets(design["limitations"])
        content.details("Metadata roles and measured design", detail)
    return content


def _cluster_order(row: dict[str, Any]) -> tuple[int, int | str]:
    identifier = str(row["clusterId"])
    return (
        (0, int(identifier)) if identifier.isdecimal() else (1, identifier.casefold())
    )


def _populations(data: dict[str, Any]) -> _Content:
    content = _Content()
    annotations = sorted(data["annotations"], key=_cluster_order)
    evidence = {str(row["clusterId"]): row for row in data["clusterEvidence"]}
    if not annotations:
        content.paragraph(
            "Provisional cell identities are not yet available. "
            "No partial annotation batch is presented as a completed result.",
            "empty",
        )
        content.heading("Interpretation limitations")
        limitations = _limitations(data)
        content.html.extend(limitations.html)
        content.markdown.extend(limitations.markdown)
        return content
    content.paragraph(
        "Each row is a cluster in the final analysis. Confidence is qualitative, "
        "not a probability. Expand the evidence below to review marker measurements "
        "and the reason for each identity."
    )
    content.paragraph(
        f"Provisional identities are recorded for {len(annotations):,} clusters.",
        "caption",
    )
    content.html.append(
        '<p class="table-hint">Scroll the table sideways to see every column.</p>'
    )
    rows = []
    for annotation in annotations:
        cluster = evidence.get(str(annotation["clusterId"]), {})
        rows.append(
            [
                annotation["clusterId"],
                annotation.get("identity"),
                _number(cluster.get("count")),
                str(annotation.get("confidence", _UNKNOWN)).capitalize(),
                ", ".join(annotation.get("supportingMarkers", []))
                or "No supporting markers recorded",
            ]
        )
    content.table(
        [
            "Cluster",
            "Provisional identity",
            "Cells",
            "Confidence",
            "Supporting markers",
        ],
        rows,
    )
    content.html.append(
        '<div class="actions"><a class="button primary" href="annotations.csv">Download annotations</a></div>'
    )
    content.markdown.extend(["[Download annotations](annotations.csv)", ""])
    content.heading("Evidence for each identity")
    for annotation in annotations[:_LIMIT]:
        detail = _Content()
        cluster = evidence.get(str(annotation["clusterId"]), {})
        detail.paragraph(_readable(annotation.get("rationale")))
        contradictory = annotation.get("contradictingMarkers", [])
        detail.paragraph(
            "Contradicting markers: "
            + (
                ", ".join(contradictory)
                if contradictory
                else "None recorded. Absence of recorded contradiction is not proof of identity."
            )
        )
        markers = cluster.get("markers", [])
        if markers:
            detail.table(
                [
                    "Measured marker",
                    "Marker score",
                    "Expressed in cluster",
                    "Expressed outside cluster",
                ],
                [
                    [
                        m.get("gene"),
                        _number(m.get("score")),
                        _number(m.get("fracExp"), percent=True),
                        _number(m.get("fracExpRest"), percent=True),
                    ]
                    for m in markers
                ],
            )
            detail.paragraph(
                "A bounded selection of measured markers is shown. "
                "These are expression summaries, not differential-expression hypothesis tests.",
                "caption",
            )
        else:
            detail.paragraph(
                "No qualifying marker measurements were saved for this cluster.",
                "muted",
            )
        if "qc" in cluster:
            detail.table(
                [
                    "Cluster QC metric",
                    "Median",
                    "10th percentile",
                    "90th percentile",
                    "Missing",
                ],
                [
                    [
                        _LABELS.get(column, column.replace("_", " ")),
                        *[
                            _number(values.get(key))
                            for key in ("median", "q10", "q90", "missing")
                        ],
                    ]
                    for column, values in cluster["qc"].items()
                ],
            )
        for column, composition in list(cluster.get("groupComposition", {}).items())[
            :_LIMIT
        ]:
            detail.heading(_metadata_label(column) + ": cluster composition")
            detail.table(
                ["Group", "Cells", "Fraction"],
                [
                    [
                        row.get("value"),
                        _number(row.get("count")),
                        _number(row.get("fraction"), percent=True),
                    ]
                    for row in composition.get("levels", [])
                ],
            )
            detail.paragraph(
                f"Missing labels: {_number(composition.get('missing'))}. "
                f"Additional groups omitted: {_number(composition.get('omittedLevels'))}. "
                "Group composition is descriptive and does not establish independent replication.",
                "caption",
            )
        content.details(
            f"Cluster {annotation['clusterId']} · {annotation.get('identity', _UNKNOWN)}",
            detail,
        )
    content.heading("Interpretation limitations")
    limitations = _limitations(data)
    content.html.extend(limitations.html)
    content.markdown.extend(limitations.markdown)
    return content


def _results(data: dict[str, Any]) -> _Content:
    content = _Content()
    if data.get("preview"):
        content.html.append(
            f'<figure><img class="embedding" src="{data["preview"]}" '
            'alt="Saved UMAP preview colored by the analysis clusters">'
            "<figcaption>Saved UMAP preview. Nearby cells have similar profiles in this "
            "projection; cluster identities remain provisional. This existing image "
            "was included without recomputing the analysis.</figcaption></figure>"
        )
        content.markdown.extend(["![Saved UMAP preview](umap_clusters.png)", ""])
    else:
        content.paragraph(
            "A UMAP preview is not available in the saved report files. Numerical results, if finalized, remain available through the analysis.",
            "empty",
        )
    content.heading("Marker expression")
    if data.get("markerPreview"):
        content.html.append(
            '<figure class="marker-figure" tabindex="0" role="region" aria-label="Marker expression plot">'
            f'<img class="marker-plot" src="{data["markerPreview"]}" '
            'alt="Saved marker dotplot across the final clusters">'
            "<figcaption>Saved marker dotplot. Dot size shows the fraction of cells "
            "expressing each gene; color shows log(1 + mean normalized expression). "
            "The panel uses qualifying markers from the final saved marker statistics. "
            "Empty positions indicate zero expression; gray crosses mark unavailable measurements.</figcaption></figure>"
        )
        content.markdown.extend(["![Saved marker dotplot](marker_dotplot.png)", ""])
    else:
        content.paragraph(
            "A marker dotplot preview is not available in the saved report files.",
            "empty",
        )
    sizes = _Content()
    if data.get("clusterSizeChart"):
        sizes.html.append(
            '<figure class="cluster-size-figure" tabindex="0" role="region" aria-label="Cell counts by final cluster">'
            f'<img class="cluster-size-plot" src="{data["clusterSizeChart"]}" '
            'alt="Bar chart of cell counts by final cluster, with counts annotated above each bar">'
            "<figcaption>Cell counts in the final clustering, shown on a common scale starting at zero. "
            "N/A means not recorded; it is not a zero count.</figcaption></figure>"
        )
        sizes.markdown.extend(
            ["![Cell counts by final cluster](cluster_sizes.svg)", ""]
        )
        sizes.omitted(len(data["clusterEvidence"]))
    else:
        sizes.paragraph(
            "Cluster sizes are not available in the saved evidence.", "empty"
        )
    content.details("Cluster sizes", sizes)
    return content


def _quality(data: dict[str, Any]) -> _Content:
    content = _Content()
    prepared = data["prepared"]
    policy = prepared.get("filtering")
    content.paragraph(
        "Policy: retain the supplied cohort. Quality-control flags are advisory and do not remove cells."
        if policy is False
        else "Quality-control filtering is configured for the analysis cohort."
        if policy
        else "The applied filtering policy was not recorded."
    )
    content.paragraph(
        f"Input cells: {_number(data['inputCells'])}. Retained cells: {_number(data['retainedCells'])}."
    )
    rows = []
    for column, flags in data["qcFlags"].items():
        metric = _QC_LABELS.get(column.rsplit("_", 1)[-1], column.replace("_", " "))
        rows.append(
            [
                metric,
                _number(flags.get("lowFlags")),
                _number(flags.get("highFlags")),
                _number(flags.get("missing")),
                _number(flags.get("low")),
                _number(flags.get("high")),
            ]
        )
    content.table(
        [
            "Quality measure",
            "Low flags",
            "High flags",
            "Missing",
            "Lower flag boundary",
            "Upper flag boundary",
        ],
        rows,
    )
    content.paragraph(
        "Flags describe the supplied cohort before additional filtering. Categories can overlap, "
        "so their counts must not be added to estimate a unique number of affected cells. "
        "Flag boundaries are descriptive and are not necessarily removal thresholds.",
        "caption",
    )
    projections = prepared.get("qcProjections")
    if projections is not None:
        content.heading("Quality strategy projections")
        content.table(
            ["Policy", "Projected retained cells", "Projected removed cells", "Use"],
            [
                [
                    row.get("policy"),
                    _number(row.get("retainedCells")),
                    _number(row.get("removedCells")),
                    "Configured for execution"
                    if row.get("executed")
                    else "Unavailable"
                    if row.get("available") is False
                    else "Projection only",
                ]
                for row in projections
            ],
        )
        content.paragraph(
            "These projections apply the recorded bounds to saved quality measurements. "
            "A projected alternative does not mean that its cells were removed or its downstream analysis was run.",
            "caption",
        )
        for row in projections[:_LIMIT]:
            if row.get("limitations") or row.get("bounds") or row.get("byGroup"):
                detail = _Content()
                if row.get("bounds"):
                    detail.table(
                        ["Projected metric", "Lower bound", "Upper bound"],
                        [
                            [
                                _LABELS.get(column, column),
                                _number(bounds[0]),
                                _number(bounds[1]),
                            ]
                            for column, bounds in row["bounds"].items()
                        ],
                    )
                for column, groups in list(row.get("byGroup", {}).items())[:_LIMIT]:
                    detail.heading(_metadata_label(column) + ": projected retention")
                    detail.table(
                        ["Group", "Input cells", "Retained cells", "Removed cells"],
                        [
                            [
                                group.get("value"),
                                *[
                                    _number(group.get(key))
                                    for key in (
                                        "inputCells",
                                        "retainedCells",
                                        "removedCells",
                                    )
                                ],
                            ]
                            for group in groups.get("levels", [])
                        ],
                    )
                    detail.paragraph(
                        f"Missing group labels before filtering: {_number(groups.get('missingInputCells'))}; "
                        f"retained: {_number(groups.get('missingRetainedCells'))}. "
                        f"Additional groups omitted: {_number(groups.get('omittedLevels'))}.",
                        "caption",
                    )
                if row.get("limitations"):
                    detail.bullets(row["limitations"])
                content.details(str(row.get("policy")) + ": projected evidence", detail)
    doublets = data["config"].get("scoreDoublets")
    content.paragraph(
        "Doublet scoring is requested; scores do not automatically remove cells."
        if doublets
        else "Doublet scoring was not requested. Possible doublets remain an interpretation limitation."
        if doublets is False
        else "Doublet scoring policy was not recorded."
    )
    content.heading("Feature selection policy")
    if prepared.get("blacklist"):
        content.paragraph(
            "Mitochondrial genes identified by naming are excluded from variable-gene selection."
        )
        exclusions = (
            prepared.get("contextEvidence", {})
            .get("study", {})
            .get("featureExclusions", [])
        )
        content.paragraph(
            "Additional excluded genes: "
            + (", ".join(exclusions) if exclusions else "None recorded.")
        )
    else:
        content.paragraph(
            "The variable-gene exclusion policy was not recorded.", "muted"
        )
    audit = prepared.get("featureAudit")
    if audit is not None:
        detail = _Content()
        detail.table(
            [
                "Gene family",
                "Matched genes",
                "Excluded from HVGs",
                "Standard blacklist matches",
            ],
            [
                [
                    row.get("family"),
                    _number(row.get("matchedFeatures")),
                    _number(row.get("excludedFeatures")),
                    _number(row.get("standardExcludedFeatures")),
                ]
                for row in audit.get("families", [])
            ],
        )
        detail.paragraph(
            "Family matches inspect the supplied feature names. Standard-blacklist matches are a comparison, "
            "not additional executed exclusions. Families may overlap; counts must not be added.",
            "caption",
        )
        if audit.get("limitations"):
            detail.bullets(audit["limitations"])
        content.details("Gene-family audit", detail)
    return content


def _exploration(data: dict[str, Any]) -> _Content:
    content = _Content()
    coverage = data.get("explorationCoverage")
    if coverage is None:
        content.paragraph(
            "Exploration coverage was not recorded by this procedure.", "caption"
        )
    else:
        content.heading("Exploration coverage")
        content.table(
            ["Trial", "Changed setting", "Parent", "Outcome", "Reason"],
            [
                [
                    _option(row["candidateId"]),
                    _LABELS.get(row.get("axis"), row.get("axis")),
                    _option(row["parentId"]) if row.get("parentId") else "Baseline",
                    row.get("status"),
                    row.get("reason") or "No additional reason recorded.",
                ]
                for row in coverage.get("slots", [])
            ],
        )
        content.paragraph(
            "All planned native comparisons were measured."
            if coverage.get("nativeComplete") is True
            else "Native sensitivity coverage is incomplete. Failed, infeasible, and pending trials do not establish robustness.",
            "caption",
        )
    if not data["candidateEvidence"]:
        content.paragraph("No candidate measurements have been saved yet.", "empty")
    for candidate_data in data["candidateEvidence"][:_LIMIT]:
        content.heading(_option(candidate_data["candidateId"]))
        parameters = candidate_data.get("parameters", {})
        content.paragraph(
            f"{_number(parameters.get('hvgCount'))} variable genes · "
            f"{_number(parameters.get('pcaDims'))} principal components · "
            f"{_number(parameters.get('neighborsK'))} neighbors · "
            + (
                "Harmony correction"
                if parameters.get("useHarmony")
                else "No batch correction"
            )
        )
        content.table(
            ["Resolution", "Clusters", "Silhouette score", "Outcome"],
            [
                [
                    _number(row.get("resolution")),
                    _number(row.get("clusterCount")),
                    _number(row.get("score")),
                    "Evaluated",
                ]
                for row in candidate_data.get("partitions", [])
            ],
        )
        content.paragraph(
            f"Clustering used the full retained cohort. Silhouette calculations used up to "
            f"{_number(candidate_data.get('silhouetteSampleCells'))} cells. "
            "Compare resolutions within this representation; scores alone do not establish "
            "which representation is biologically better.",
            "caption",
        )
        detail = _candidate_diagnostics(candidate_data)
        if detail.html:
            content.details(
                "Representation diagnostics: " + _option(candidate_data["candidateId"]),
                detail,
            )
    content.omitted(len(data["candidateEvidence"]))
    return content


def _candidate_diagnostics(candidate: dict[str, Any]) -> _Content:
    content = _Content()
    associations = candidate.get("covariateAssociations")
    if associations is not None:
        content.table(
            ["Covariate", "Kind", "PC", "Association", "Rows used", "Missing rows"],
            [
                [
                    _metadata_label(row["column"]),
                    row.get("kind"),
                    _number(row.get("component")),
                    _number(row.get("association")),
                    _number(row.get("rowsUsed")),
                    _number(row.get("rowsMissing")),
                ]
                for row in associations
            ],
        )
        content.paragraph(
            "Associations are descriptive and do not establish a technical cause or authorize correction. "
            "They use the recorded diagnostic sample; missing measurements do not mean no association.",
            "caption",
        )
    if "actualHvgCount" in candidate:
        content.paragraph(
            f"Actual selected variable genes: {_number(candidate['actualHvgCount'])}.",
            "caption",
        )
    audit = candidate.get("hvgAudit")
    if audit is not None:
        content.heading("Selected variable-gene families")
        content.table(
            ["Gene family", "Selected genes"],
            [
                [row.get("family"), _number(row.get("matchedFeatures"))]
                for row in audit.get("families", [])
            ],
        )
        if audit.get("limitations"):
            content.bullets(audit["limitations"])
    loadings = candidate.get("loadingFamilies")
    if loadings is not None:
        content.heading("Dominant principal-component features")
        content.table(
            ["PC", "Leading genes", "Gene-family counts among leading genes"],
            [
                [
                    _number(row.get("component")),
                    ", ".join(
                        str(gene["gene"]) for gene in row.get("topGenes", [])[:20]
                    ),
                    "; ".join(
                        f"{family}: {_number(count)}"
                        for family, count in list(row.get("families", {}).items())[
                            :_LIMIT
                        ]
                    ),
                ]
                for row in loadings
            ],
        )
        content.paragraph(
            "Dominant gene families can reflect biology or technical variation. Their presence alone is not a reason to exclude genes.",
            "caption",
        )
    comparisons = candidate.get("comparisons")
    if comparisons is not None:
        content.heading("Matched parent comparisons")
        content.table(
            ["Parent", "Resolution", "Adjusted Rand index", "Compared cells"],
            [
                [
                    _option(row["parentCandidateId"]),
                    _number(row.get("resolution")),
                    _number(row.get("adjustedRandIndex")),
                    _number(row.get("cellCount")),
                ]
                for row in comparisons
            ],
        )
        content.paragraph(
            "Partitions are compared at the same resolution on aligned cells. Agreement measures sensitivity, not biological correctness. "
            "Directional overlaps describe splits and merges without assuming cluster labels match.",
            "caption",
        )
        for comparison in comparisons[:_LIMIT]:
            detail = _Content()
            for key, direction in (
                ("parentToCandidate", "Parent to alternative"),
                ("candidateToParent", "Alternative to parent"),
            ):
                detail.heading(direction)
                detail.table(
                    [
                        "Cluster",
                        "Best matching cluster",
                        "Source cells",
                        "Shared cells",
                        "Source fraction",
                    ],
                    [
                        [
                            row.get("clusterId"),
                            row.get("matchedClusterId"),
                            _number(row.get("sourceCells")),
                            _number(row.get("intersectionCells")),
                            _number(row.get("fraction"), percent=True),
                        ]
                        for row in comparison.get(key, [])
                    ],
                )
            content.details(
                f"Cluster overlap at resolution {_number(comparison.get('resolution'))}",
                detail,
            )
        content.omitted(len(comparisons))
    return content


def _selection(data: dict[str, Any]) -> _Content:
    content = _Content()
    recipe = data["selectedRecipe"] or {}
    candidate = recipe.get("candidate") or {}
    if candidate:
        content.table(
            ["Selected analysis setting", "Value"],
            [
                ["Variable genes", _number(candidate.get("hvgCount"))],
                ["Principal components", _number(candidate.get("pcaDims"))],
                ["Neighbors", _number(candidate.get("neighborsK"))],
                ["Leiden resolution", _number(recipe.get("resolution"))],
                [
                    "Batch correction",
                    "Harmony"
                    if candidate.get("useHarmony")
                    else "Native analysis (no correction)",
                ],
            ],
        )
    else:
        content.paragraph("The final analysis settings are not yet available.", "empty")
    finalists = [
        row for row in data["diagnostics"] if row["kind"] == "finalistMeasured"
    ]
    if finalists:
        content.heading("Marker evidence for the shortlisted results")
        legacy_support = any(
            "markerSupportFraction" not in row.get("metrics", {})
            and "markerCoherence" in row.get("metrics", {})
            for row in finalists
        )
        content.table(
            [
                "Clustering",
                "Marker coherence (marker support)"
                if legacy_support
                else "Marker support",
                "Median marker specificity",
                "Diagnostic sample",
            ],
            [
                [
                    _option(row["optionId"]),
                    _number(_marker_support(row.get("metrics", {})), percent=True),
                    _number(row.get("metrics", {}).get("markerSpecificityMedian")),
                    _number(row.get("diagnosticScope", {}).get("sampleCells")),
                ]
                for row in finalists
            ],
        )
        content.paragraph(
            "Marker support is the fraction of clusters with at least one marker scoring "
            "at least 0.25 and expressed in at least 20% of cells. It is not annotation confidence. "
            "Marker scores depend on the partition and can favor coarser groups. "
            "Diagnostic sampling can omit rare populations. Missing measurements are shown explicitly.",
            "caption",
        )
        if legacy_support:
            content.paragraph(
                "Earlier records named this same measurement marker coherence. "
                "The report reads that saved value without changing the scientific record.",
                "caption",
            )
        for row in finalists[:_LIMIT]:
            metrics = row.get("metrics", {})
            detail = _Content()
            diagnostic_rows = [
                [
                    "Support across samples",
                    "Clusters",
                    _number(metrics.get("crossUnitSupport"), percent=True),
                ],
                [
                    "Concentration of high doublet scores",
                    "Clusters",
                    _number(metrics.get("doubletHighScoreConcentration")),
                ],
            ]
            for group, value in metrics.get("mixing", {}).items():
                diagnostic_rows.append(
                    ["Batch mixing", group.replace("_", " "), _number(value)]
                )
            for group, values in metrics.get("protection", {}).items():
                for key, label in (
                    ("cLISI", "Biological separation"),
                    ("graphConnectivity", "Graph connectivity"),
                ):
                    diagnostic_rows.append(
                        [label, group.replace("_", " "), _number(values.get(key))]
                    )
            detail.table(["Diagnostic", "Grouping", "Value"], diagnostic_rows)
            detail.paragraph(
                "These diagnostics assess technical mixing, preservation of supplied biological "
                "groups, representation across samples, and possible doublets on the saved "
                "diagnostic sample. Missing values do not mean the check passed.",
                "caption",
            )
            content.details(
                "Additional diagnostics: " + _option(row["optionId"]), detail
            )
    return content


def _marker_support(metrics: dict[str, Any]) -> Any:
    """Read the renamed metric; the legacy key is a presentation-only alias."""
    if "markerSupportFraction" in metrics:
        return metrics["markerSupportFraction"]
    return metrics.get("markerCoherence")


def _decisions(data: dict[str, Any], stages: tuple[str, ...]) -> _Content:
    content = _Content()
    decisions = [
        decision
        for decision in data["decisions"]
        if decision.get("stage") in stages
        and (
            "rationale" in decision.get("output", {})
            or decision.get("output", {}).get("question")
        )
    ]
    for decision in decisions[:_LIMIT]:
        output = decision.get("output", {})
        stage = decision.get("stage")
        title = _STAGES.get(stage, "Recorded decision")
        options = output.get("optionIds", [])
        if options:
            title += ": " + "; ".join(_option(option) for option in options)
        detail = _Content()
        detail.paragraph(
            _readable(
                output.get(
                    "rationale",
                    "See the saved model exchange for the accepted response.",
                )
            )
        )
        if output.get("question"):
            detail.paragraph(output["question"])
        detail.paragraph(
            "Decision record: " + str(decision.get("decisionId", _UNKNOWN)), "caption"
        )
        content.details(title, detail)
    content.omitted(len(decisions))
    resolutions = [
        row for row in data.get("decisionResolutions", []) if row.get("stage") in stages
    ]
    for resolution in resolutions[:_LIMIT]:
        detail = _Content()
        reason = resolution.get("reason")
        detail.paragraph(
            {
                "uncertainMetadata": "Optional metadata could not be resolved confidently.",
                "ambiguousSelection": "Several measured choices remained acceptable.",
            }.get(reason, reason or "Conservative policy applied.")
        )
        resolved = resolution.get("resolved", {})
        options = resolved.get("optionIds", [])
        detail.paragraph(
            "Selected: " + "; ".join(_option(option) for option in options)
            if options
            else "Action: "
            + (
                "Retain declared metadata roles and leave uncertain roles unresolved."
                if resolved.get("action") == "retainDeclaredRoles"
                else _readable(resolved.get("action", _UNKNOWN))
            )
        )
        if resolution.get("limitation"):
            detail.paragraph(resolution["limitation"])
        detail.paragraph(
            "This is a recorded policy resolution, separate from the model response. "
            "Rule: " + str(resolution.get("rule", _UNKNOWN)),
            "caption",
        )
        content.details("Automatic conservative resolution", detail)
    content.omitted(len(resolutions))
    return content


def _limitations(data: dict[str, Any]) -> _Content:
    content = _Content()
    # Repeated candidate warnings are one limitation, not independent findings.
    limitations = list(
        dict.fromkeys(
            _readable(re.sub(r"^c\d+(?::r[\d.]+)?: ", "", text))
            for text in data["limitations"]
        )
    )
    if limitations:
        content.bullets(limitations)
    else:
        content.paragraph(
            "No additional limitations were recorded. This does not establish that all sources of uncertainty were assessed."
        )
    return content


def _provenance(data: dict[str, Any]) -> _Content:
    content = _Content()
    detail = _Content()
    detail.table(
        ["Record", "Value"],
        [
            ["Analysis identifier", data.get("runId")],
            ["Procedure identity", data.get("procedureIdentity")],
            ["Final numerical run", (data["final"] or {}).get("runId")],
            ["Observed model requests", _number(data["modelRequests"])],
        ],
    )
    usage = data["usage"]
    detail.heading("Recorded model usage")
    usage_rows = []
    for key, label in (
        ("inputTokens", "Input tokens"),
        ("outputTokens", "Output tokens"),
        ("cacheReadTokens", "Tokens read from cache"),
        ("cacheWriteTokens", "Tokens written to cache"),
    ):
        known = [
            row[key] for row in usage if row is not None and row.get(key) is not None
        ]
        usage_rows.append(
            [
                label,
                _number(sum(known) if known else None),
                f"{len(known)} of {len(usage)} recorded responses",
            ]
        )
    detail.table(["Usage", "Known total", "Availability"], usage_rows)
    detail.paragraph(
        "Unavailable usage is unknown. These totals are not a spending guarantee.",
        "caption",
    )
    detail.heading("Numerical run history")
    detail.table(
        ["Step", "Run identifier", "Recovered after interruption"],
        [
            [
                "Final analysis"
                if row.get("operation") == "final"
                else "Marker assessment"
                if str(row.get("operation", "")).startswith("finalist")
                else "Candidate exploration",
                row.get("runId"),
                "Yes" if row.get("recovered") else "No",
            ]
            for row in data["pipelineRuns"]
        ],
    )
    errors = data["errors"]
    if errors:
        detail.heading(
            "Earlier issues and rejected responses"
            if data["status"] == "completed"
            else "Recorded issues"
        )
        detail.paragraph(
            "The latest status at the top of this report describes the current outcome. Earlier retries and rejected responses remain here for review."
        )
        detail.table(
            ["Step", "Recorded issue"],
            [
                [
                    _STAGES.get(row.get("stage"), "Decision validation"),
                    _readable(
                        row.get("message")
                        or "; ".join(row.get("errors", []))
                        or row.get("errorType", "No message recorded")
                    ),
                ]
                for row in errors
            ],
            prose=(1,),
        )
    detail.heading("Saved files")
    links = [
        ("Study and configuration", "run.json"),
        ("Event history", "events/"),
        ("Measured evidence", "evidence/"),
        ("Model exchanges", "calls/"),
        ("Annotations CSV", "annotations.csv"),
        ("Markdown report", "report.md"),
    ]
    detail.html.append(
        '<div class="record-links">'
        + "".join(f'<a href="{url}">{label}</a>' for label, url in links)
        + "</div>"
    )
    detail.markdown.extend([f"- [{label}]({url})" for label, url in links])
    content.details("Open technical records and run history", detail)
    return content


def build_report(data: dict[str, Any]) -> tuple[str, str]:
    """Return a standalone HTML document and its readable Markdown companion."""
    status = _STATUSES.get(data["status"], "Status unavailable")
    document = _Content()
    document.markdown.extend(
        [
            "# Single-cell RNA analysis",
            "",
            f"[Nygen · nygen.io]({_NYGEN_URL})",
            "",
            f"Status: {status}",
            "",
        ]
    )
    title = (
        data["study"].get("objective")
        or "An evidence-based overview of the supplied RNA cohort."
    )
    study_label = " · ".join(
        str(data["study"][key])
        for key in ("tissue", "organism")
        if data["study"].get(key)
    )
    document.html.append(
        '<header class="hero"><p class="eyebrow">Single-cell study report</p>'
        "<h1>Single-cell RNA analysis</h1>"
        f'<div class="subtitle narrative">{narrative_html(_text(title))}</div><p class="study-label">{_escape(study_label)}</p>'
        "</header>"
    )
    document.markdown.extend(
        [html.escape(normalize_narrative(_text(title))), "", _escape(study_label), ""]
    )
    sections = [
        ("overview", "Study and input data", _overview, ("inspect", "context")),
        ("quality", "Quality and preparation", _quality, ("preprocess",)),
        ("exploration", "Explore clustering", _exploration, ("explore",)),
        ("selection", "Select the final analysis", _selection, ("finalists",)),
        ("results", "Examine the results", _results, ("finalize",)),
        ("populations", "Provisional cell identities", _populations, ("annotate",)),
    ]
    document.html.append(
        '<nav class="contents" aria-label="Report sections">'
        + "".join(
            f'<a href="#{key}"><span>{index:02}</span>{label}</a>'
            for index, (key, label, _, _) in enumerate(sections, 1)
        )
        + '</nav><main id="main">'
    )
    if data["status"] != "completed":
        document.html.append(
            '<aside class="notice" aria-label="Current analysis status">'
        )
        document.paragraph(
            f"{status}: {_STAGES.get(data['stage'], 'analysis')}. This report describes the evidence saved so far."
        )
        if data.get("currentMessage"):
            document.paragraph(_readable(data["currentMessage"]))
        if data["pendingQuestions"]:
            document.bullets([row["question"] for row in data["pendingQuestions"]])
            document.paragraph(
                "Provide the requested facts when resuming this analysis. The saved study and completed work remain available."
            )
        document.html.append("</aside>")
    document.paragraph(
        "Follow the analysis step by step. Open a section to review its measurements and reasoning; cell identities are provisional.",
        "caption",
    )
    opened = next(
        (
            key
            for key, _, _, stages in sections
            if data["status"] != "completed" and data["stage"] in stages
        ),
        "overview",
    )
    outcomes = _outcomes(data)
    for index, (key, label, build, stages) in enumerate(sections, 1):
        content = build(data)
        decisions = _decisions(data, stages)
        content.html.extend(decisions.html)
        content.markdown.extend(decisions.markdown)
        _section(
            document,
            key,
            index,
            label,
            content,
            state=_step_state(data, stages),
            outcome=outcomes[key],
            expanded=key == opened,
        )
    provenance = _provenance(data)
    document.html.append(
        '<section id="provenance" aria-label="Technical records">'
        + "".join(provenance.html)
        + "</section>"
    )
    document.markdown.extend(["## Technical records", "", *provenance.markdown])
    document.markdown.extend(
        [
            "",
            "## About Scarf",
            "",
            f"[Scarf on GitHub]({_SCARF_URL}) · [Scarf paper]({_PAPER_URL})",
            "",
            _PAPER_CITATION,
            "",
        ]
    )
    document.html.append("</main>")
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light">'
        f'<title>Scarf RNA analysis · {_escape(status)}</title><link rel="icon" href="{FAVICON}">'
        f"<!-- Embedded Inter font license:\n{INTER_FONT_LICENSE}\n-->"
        "<style>@font-face {font-family:Inter;font-style:normal;font-weight:100 900;"
        f"font-display:swap;src:url('{INTER_FONT}') format('woff2');}}{STYLE}</style></head>"
        '<body><a class="skip-link" href="#main">Skip to results</a><div class="shell">'
        '<div class="masthead"><div class="brand">'
        f'<img src="{NYGEN_LOGO}" alt="Nygen logo"><div class="brand-name">'
        f'<a href="{_NYGEN_URL}">Nygen<small>nygen.io</small></a></div></div>'
        f'<span class="status">{_escape(status)}</span></div>'
        + "".join(document.html)
        + f'<footer class="footer"><img class="scarf-logo" src="{SCARF_LOGO}" alt="Scarf logo">'
        '<div class="footer-info"><nav class="footer-links" aria-label="Scarf resources">'
        f'<a href="{_SCARF_URL}">Scarf on GitHub</a>'
        f'<a href="{_PAPER_URL}">Scarf paper</a></nav>'
        f'<p class="paper-citation">{_PAPER_CITATION}</p>'
        '<p class="footer-note">Generated from saved analysis records. '
        "Provisional identities require biological review.</p></div>"
        "</footer></div></body></html>\n"
    )
    return page, "\n".join(document.markdown)
