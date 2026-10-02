"""Human-readable presentation of saved evidence, without numerical execution."""

import html
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

_LIMIT = 100
_UNKNOWN = "Not recorded"
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
        text = html.escape(_text(value, limit))
        self.html.append(f'<p class="{style}">{text}</p>')
        self.markdown.extend([text, ""])

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
            self.html.append(f"<li>{_escape(value)}</li>")
            self.markdown.append(f"- {_escape(value)}")
        self.html.append("</ul>")
        self.markdown.append("")
        self.omitted(len(values))

    def table(self, headers: list[str], rows: list[list[Any]]) -> None:
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
                "<tr>" + "".join(f"<td>{_escape(v)}</td>" for v in row) + "</tr>"
            )
            cells = [_escape(v).replace("|", "\\|").replace("\n", " ") for v in row]
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
    document: _Content, key: str, number: int, title: str, content: _Content
) -> None:
    document.html.append(
        f'<section id="{key}" aria-labelledby="heading-{key}">'
        f'<div class="section-heading"><span class="section-number" aria-hidden="true">{number:02}</span>'
        f'<h2 id="heading-{key}">{_escape(title)}</h2></div>'
        + "".join(content.html)
        + "</section>"
    )
    document.markdown.extend([f"## {title}", "", *content.markdown])


def _overview(data: dict[str, Any]) -> _Content:
    content = _Content()
    annotations = data["annotations"]
    clusters = data["clusterEvidence"]
    cluster_count = (
        len(clusters) if clusters else len(annotations) if annotations else None
    )
    unassigned = sum(
        row.get("identity", "").lower() == "unassigned" for row in annotations
    )
    before, after = data["inputCells"], data["retainedCells"]
    retained = after / before if before and after is not None else None
    metrics = [
        ("Cells in analysis cohort", _number(after)),
        ("Supplied cells retained", _number(retained, percent=True)),
        ("Clusters", _number(cluster_count)),
        ("Unassigned clusters", _number(unassigned if annotations else None)),
    ]
    content.html.append('<div class="metrics">')
    for label, value in metrics:
        content.html.append(
            f'<div class="metric"><span class="metric-value">{_escape(value)}</span>'
            f'<span class="metric-label">{label}</span></div>'
        )
        content.markdown.append(f"- {label}: {value}")
    content.html.append("</div>")
    content.markdown.append("")
    content.paragraph(
        "Cell identities are provisional interpretations of measured markers. "
        "Unassigned clusters remain explicit when the evidence is insufficient.",
        "notice",
    )
    content.html.append('<div class="overview-grid">')
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
            "A UMAP preview is not available in the saved report files. "
            "Numerical results, if finalized, remain available through the analysis.",
            "empty",
        )
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
    content.html.append('<dl class="facts">')
    for label, value in facts:
        content.html.append(f"<div><dt>{label}</dt><dd>{_escape(value)}</dd></div>")
        content.markdown.append(f"- {label}: {_escape(value)}")
    content.html.append("</dl></div>")
    content.markdown.append("")
    context = _Content()
    context.paragraph(
        data["study"].get("context") or "No study context supplied.", "source-text"
    )
    content.details("Supplied study context", context)
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
        content.details(
            f"Cluster {annotation['clusterId']} · {annotation.get('identity', _UNKNOWN)}",
            detail,
        )
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
    doublets = data["config"].get("scoreDoublets")
    content.paragraph(
        "Doublet scoring is requested; scores do not automatically remove cells."
        if doublets
        else "Doublet scoring was not requested. Possible doublets remain an interpretation limitation."
        if doublets is False
        else "Doublet scoring policy was not recorded."
    )
    return content


def _methods(data: dict[str, Any]) -> _Content:
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
                    "Selected"
                    if row.get("optionId") == (data["final"] or {}).get("selected")
                    else "Evaluated",
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
    finalists = [
        row for row in data["diagnostics"] if row["kind"] == "finalistMeasured"
    ]
    if finalists:
        content.heading("Marker evidence for the shortlisted results")
        content.table(
            [
                "Clustering",
                "Marker coherence",
                "Median marker specificity",
                "Diagnostic sample",
            ],
            [
                [
                    _option(row["optionId"]),
                    _number(
                        row.get("metrics", {}).get("markerCoherence"), percent=True
                    ),
                    _number(row.get("metrics", {}).get("markerSpecificityMedian")),
                    _number(row.get("diagnosticScope", {}).get("sampleCells")),
                ]
                for row in finalists
            ],
        )
        content.paragraph(
            "Marker coherence is the fraction of clusters with at least one marker scoring "
            "at least 0.25 and expressed in at least 20% of cells. It is not annotation confidence. "
            "Diagnostic sampling can omit rare populations. Missing measurements are shown explicitly.",
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


def _decisions(data: dict[str, Any]) -> _Content:
    content = _Content()
    decisions = data["decisions"]
    if not decisions:
        content.paragraph("No model decisions have been accepted yet.", "empty")
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
        ["# Single-cell RNA analysis", "", f"Status: {status}", ""]
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
        f'<p class="subtitle">{_escape(title)}</p><p class="study-label">{_escape(study_label)}</p>'
        '<div class="actions"><a class="button primary" href="annotations.csv">Download annotations</a>'
        "</div></header>"
    )
    document.markdown.extend([_escape(title), "", _escape(study_label), ""])
    sections = [
        ("overview", "Overview", _overview),
        ("populations", "Cell populations", _populations),
        ("quality", "Quality checks", _quality),
        ("methods", "Analysis choices", _methods),
        ("decisions", "Decision notes", _decisions),
        ("limitations", "Limitations", _limitations),
        ("provenance", "Technical records", _provenance),
    ]
    document.html.append(
        '<nav class="contents" aria-label="Report sections">'
        + "".join(
            f'<a href="#{key}"><span>{index:02}</span>{label}</a>'
            for index, (key, label, _) in enumerate(sections, 1)
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
    for index, (key, label, build) in enumerate(sections, 1):
        _section(document, key, index, label, build(data))
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
        f'<img src="{NYGEN_LOGO}" alt="Nygen logo"><div class="brand-name">Scarf'
        "<small>Single-cell analysis</small></div></div>"
        f'<span class="status">{_escape(status)}</span></div>'
        + "".join(document.html)
        + f'<footer class="footer"><img class="scarf-logo" src="{SCARF_LOGO}" alt="Scarf logo">'
        "<span>Generated from saved analysis records.<br>Provisional identities require biological review.</span>"
        "</footer></div></body></html>\n"
    )
    return page, "\n".join(document.markdown)
