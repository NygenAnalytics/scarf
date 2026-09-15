"""A single readable analysis page, using only recorded scientific evidence."""

import html
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import label, mapping, mappings, scalar, texts

_STYLES = """
:root{color-scheme:light;font-family:Inter,sans-serif;font-size:16px;font-weight:300;line-height:1.2;color:#000000;background:#ffffff;letter-spacing:-.04em}
*{box-sizing:border-box}body{margin:0;background:#ffffff}main{max-width:1120px;margin:auto;padding:48px 32px 72px}
header{border-bottom:1px solid #000000;padding-bottom:32px}h1,h2,h3{line-height:1.2;color:#000000;font-weight:400}
h1{font-size:3.5rem;letter-spacing:0;margin:12px 0 20px}h2{font-size:2rem;letter-spacing:-.04em;margin:56px 0 20px}h3{font-size:1.25rem;letter-spacing:-.04em;margin:32px 0 12px}
p{max-width:82ch}header h1+p{font-size:1.5rem;font-weight:300;margin:0 0 24px}a{color:#0077fc}
small{display:block;color:#b4b4b4;font-size:.75rem;font-weight:400;letter-spacing:-.04em;text-transform:uppercase}.muted{color:#b4b4b4}
.header-links{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:8px 24px;margin-bottom:40px;font-size:.9rem}.header-links a{font-weight:400}
.numbers{display:inline-block;margin:0 0 20px;padding:10px 18px;border-radius:999px;background:#0077fc;color:#ffffff;font-size:1rem;font-weight:400}
.fraction{white-space:nowrap}progress{width:86px;height:8px;border:0;border-radius:999px;overflow:hidden;accent-color:#0077fc;background:#b4b4b4}
progress::-webkit-progress-bar{background:#b4b4b4;border-radius:999px}progress::-webkit-progress-value{background:#0077fc;border-radius:999px}progress::-moz-progress-bar{background:#0077fc;border-radius:999px}
section{margin-top:64px}section>h2:first-child{margin-top:0}.limitations::before{content:"";display:block;width:72px;height:8px;margin-bottom:24px;border-radius:999px;background:#0077fc}
.population-overview{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:20px;align-items:start}.population-overview figure{position:sticky;top:20px}.population-overview>div{min-width:0}
figure{margin:24px 0;background:#ffffff;padding:0}figure img{width:100%;height:auto}
figcaption{color:#b4b4b4;font-size:.9rem;text-align:center}.decision{border-top:1px solid #b4b4b4;padding:14px 0}
.decision h3{margin:0}.decision p{margin:8px 0}details{margin:20px 0}summary{display:inline-block;cursor:pointer;padding:10px 18px;border-radius:999px;box-shadow:inset 0 0 0 2px #0077fc;color:#0077fc;font-weight:400}
details[open] summary{background:#0077fc;color:#ffffff;box-shadow:none}summary:focus-visible{outline:2px solid #000000;outline-offset:3px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem;margin:14px 0}
th,td{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid #b4b4b4}
th,strong{font-weight:400}th{background:#ffffff}td p{margin:0}li{margin:8px 0}
footer{margin-top:64px;border-top:1px solid #000000;padding-top:20px;color:#b4b4b4;font-size:.85rem}
@media(max-width:800px){.population-overview{display:block}.population-overview figure{position:static}}
@media(max-width:600px){main{padding:32px 16px 48px}h1{font-size:2.5rem}h2{font-size:1.75rem}header h1+p{font-size:1.25rem}th,td{padding:8px}}
@media print{body{background:#ffffff}main{padding:0}details{break-inside:avoid}}
"""


def _escape(value: Any) -> str:
    return html.escape(scalar(value))


def _list(items: Sequence[str]) -> str:
    return (
        "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"
        if items
        else ""
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _fraction_bar(value: Any) -> str:
    if (
        not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        return "Unavailable"
    return f'<span class="fraction"><progress value="{value:.6f}" max="1"></progress> {value:.1%}</span>'


def _percentage(value: Any) -> str:
    return (
        f"{value:.1%}"
        if isinstance(value, int | float) and math.isfinite(value) and 0 <= value <= 1
        else "Unavailable"
    )


def _scope(assessment: Mapping[str, Any]) -> str:
    coverage = mapping(assessment.get("coverage"))
    all_cells = (
        assessment.get("scope") == "full"
        or mapping(assessment.get("comparisonCoverage")).get("population") == "allCells"
    )
    name = "Full cohort" if all_cells else "Screening sample"
    size = coverage.get("screeningCells")
    return f"{name} ({size:,} cells)" if isinstance(size, int) else name


def _population_table(payload: Mapping[str, Any]) -> str:
    counts = mapping(payload.get("clusterCounts"))
    total = sum(counts.values())
    markers = mappings(payload.get("markers"))
    support = mapping(payload.get("populationSupport"))
    units = texts(mapping(payload.get("study")).get("independentUnitColumns"))
    columns = mapping(support.get("columns"))
    unit = next((name for name in units if name in columns), None)
    evidence = mapping(columns.get(unit)) if unit is not None else {}
    populations = {
        str(row["cluster"]): row for row in mappings(evidence.get("populations"))
    }
    rows = []
    for cluster, count in counts.items():
        row = populations.get(str(cluster), {})
        genes = ", ".join(
            str(item["feature"])
            for item in markers
            if str(item.get("cluster")) == str(cluster)
        )
        observed = row.get("groupsWithAtLeast5Cells")
        unit_count = str(observed) if isinstance(observed, int) else "Unavailable"
        rows.append(
            f"<tr><td>{_escape(cluster)}</td><td>{count:,}<br>{_fraction_bar(count / total if total else None)}</td>"
            f"<td>{_escape(genes or 'No marker preview available')}</td>"
            f"<td>{unit_count}</td><td>{_fraction_bar(row.get('largestGroupFraction'))}</td></tr>"
        )
    unit_label = label(unit) if unit else "Study unit"
    omitted = evidence.get("omittedPopulations", 0)
    notes = []
    if omitted:
        notes.append(
            f"Support details were not saved for {omitted} populations; their support is unavailable here."
        )
    if evidence.get("missingCells"):
        notes.append(
            f"{evidence['missingCells']:,} cells lack the recorded study-unit metadata."
        )
    if not evidence:
        notes.append("No per-population study-unit evidence was saved.")
    return (
        '<div class="table-wrap"><table><thead><tr><th>Population</th><th>Cells</th><th>Top marker genes</th>'
        f"<th>{_escape(unit_label)} groups with ≥5 cells</th><th>Largest group contribution</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + '<p class="muted">Markers describe gene programs, not validated cell identities. Study-unit counts and concentration describe observed support; five cells is not a replication threshold.</p>'
        + _list(notes)
    )


def _qc_section(payload: Mapping[str, Any]) -> str:
    qc = mapping(payload.get("qc"))
    profiles = mappings(payload.get("qcProfiles"))
    names = {
        "coreGlobalGaussian": "Scarf default global filter",
        "globalGaussian": "Scarf default global filter",
        "coreSampleMad3": "Scarf filter within samples",
        "sampleMad": "Scarf filter within samples",
        "retainWithFlags": "Retain cells with quality flags",
        "skip": "Retain cells without filtering",
        "globalMad5": "Lenient global filter",
        "captureMad5": "Lenient filter within captures",
        "captureMad3Sensitivity": "Stricter filter within captures",
        "pooledReferenceMad5": "Filter using reference captures",
    }
    rows = []
    for profile in profiles:
        name = names.get(
            str(profile.get("registeredProfile") or profile.get("action")),
            "Recorded quality filter",
        )
        if profile.get("sampleColumn"):
            name += f" ({label(str(profile['sampleColumn']))})"
        chosen = profile.get("profileId") == qc.get("profileId")
        rows.append(
            f"<tr><td>{_escape(name)}{' <strong>(selected)</strong>' if chosen else ''}</td>"
            f"<td>{_escape(profile.get('retainedCells'))}</td>"
            f"<td>{_fraction_bar(profile.get('retainedFraction'))}</td></tr>"
        )
    table = (
        '<div class="table-wrap"><table><thead><tr><th>Compared policy</th><th>Projected cells retained</th><th>Retention</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
        if rows
        else "<p>Quality-policy comparisons are unavailable.</p>"
    )
    grouped = []
    for column, groups in mapping(qc.get("retainedCellsByColumn")).items():
        values = list(mapping(groups).values())
        if not values:
            continue
        description = (
            "; ".join(f"{name}: {count:,}" for name, count in groups.items())
            if len(values) <= 8
            else f"{len(values)} groups; smallest {min(values):,}, largest {max(values):,} cells"
        )
        grouped.append([label(column), description])
    decisions = [
        mapping(item.get("record"))
        for item in mappings(payload.get("decisions"))
        if mapping(item.get("record")).get("decisionId")
        in {"cellQuality", "qcGrouping"}
    ]
    quality = [item for item in decisions if item.get("decisionId") == "cellQuality"]
    chosen_reason = f"<p>{_escape(quality[-1].get('rationale'))}</p>" if quality else ""
    explanations = "".join(
        f"<p>{_escape(item.get('rationale'))}</p>"
        for item in decisions
        if item.get("decisionId") == "qcGrouping"
    )
    if explanations:
        explanations = f"<details><summary>Study grouping for quality filtering</summary>{explanations}</details>"
    return f'<section id="quality"><h2>Cell quality</h2>{chosen_reason}{table}{_table(("Retained study groups", "Cells"), grouped)}{explanations}</section>'


def _design_section(payload: Mapping[str, Any]) -> str:
    study = mapping(payload.get("study"))
    context = mapping(payload.get("context"))
    characterization = mapping(context.get("characterization"))
    correction = mapping(payload.get("selectedParameters")).get("useHarmony")
    license = study.get("correctionLicense")
    if correction is True:
        correction_text = "Batch correction was applied to the selected representation."
    elif license == "unsafeConfounded":
        correction_text = "Batch correction was not applied: the recorded design cannot separate the proposed batch effects from protected biology."
    elif correction is False:
        correction_text = "The selected representation uses no batch correction."
    else:
        correction_text = "The correction decision is unavailable."
    rows = []
    for coefficient in mappings(characterization.get("coefficients")):
        replication = mapping(coefficient.get("replication"))
        groups = mappings(replication.get("independentUnitsByGroup"))
        counts = "; ".join(
            f"{item.get('group')}: {item.get('count')}" for item in groups
        )
        paired = mapping(coefficient.get("pairedCoverage"))
        pairing = (
            f"{paired['completePairs']} of {paired.get('pairs', 'unavailable')} complete pairs"
            if paired.get("design") == "mixedOrIncomplete"
            else "Between study units"
            if paired.get("betweenIndependentUnits")
            else ""
        )
        rows.append([label(str(coefficient.get("name", ""))), counts, pairing])
    requirements = {
        item["requirementId"]: item
        for item in mappings(study.get("evidenceRequirements"))
    }
    questions = []
    for item in mappings(study.get("evidenceCoverage")):
        requirement = requirements.get(item.get("requirementId"), {})
        status = {
            "computed": "Assessed",
            "nonIdentifiable": "Not identifiable",
            "unsupported": "Not assessed",
            "failed": "Failed",
        }.get(str(item.get("status")), "Unavailable")
        questions.append(
            [
                requirement.get("question"),
                status,
                "; ".join(label(reason) for reason in texts(item.get("reasons"))),
            ]
        )
    claims = texts(study.get("unsupportedClaims"))
    return f'<section id="design"><h2>Study design and correction</h2><p>{_escape(correction_text)}</p>{_table(("Study factor", "Independent units per group", "Design"), rows)}{_table(("Objective question", "Evidence", "Limit"), questions)}{_list(claims)}</section>'


_AXIS_LABELS = {
    "hvgCount": "Number of variable genes",
    "hvgRanking": "Variable-gene ranking",
    "featurePolicy": "Gene families",
    "pca": "PCA dimensions",
    "dimensions": "PCA dimensions",
    "neighbors": "Neighbors",
    "resolution": "Clustering resolution",
    "partition": "Clustering resolution",
    "batchCorrection": "Batch correction",
    "harmony": "Batch correction",
}


def _comparison_sections(payload: Mapping[str, Any]) -> str:
    assessments = mappings(payload.get("assessments"))
    scope_evidence = {str(item["scope"]): item for item in assessments}
    sections = []
    for title, partition in (("Genes and representation", False), ("Clustering", True)):
        scopes: dict[str, dict[str, Any]] = {}
        for assessment in assessments:
            coverage = mapping(assessment.get("comparisonCoverage"))
            settings = mapping(coverage.get("candidateSettings"))
            for conclusion in mappings(assessment.get("comparisonConclusions")):
                if (conclusion.get("axis") == "partition") != partition:
                    continue
                identities = texts(conclusion.get("candidateIds"))
                for identity in identities:
                    candidate = mapping(settings.get(identity))
                    scope = str(candidate.get("scope", ""))
                    if not candidate or scope not in {"sample0", "sample1", "full"}:
                        raise ValueError(
                            "Reported comparison lacks exact candidate scope and settings"
                        )
                    group = scopes.setdefault(
                        scope,
                        {
                            "assessment": scope_evidence.get(scope, {"scope": scope}),
                            "conclusions": {},
                            "candidates": {},
                            "unavailable": {},
                        },
                    )
                    key = (conclusion["axis"], tuple(sorted(identities)))
                    group["conclusions"][key] = conclusion
                    group["candidates"][identity] = candidate
            for item in mappings(coverage.get("comparisons")):
                if (
                    item.get("status") != "notApplicable"
                    or (item.get("axis") == "partition") != partition
                ):
                    continue
                candidate = mapping(settings.get(str(item.get("baselineCandidateId"))))
                scope = str(candidate.get("scope", ""))
                if scope in scopes:
                    scopes[scope]["unavailable"][
                        (item.get("axis"), item.get("reason"))
                    ] = item
        content = []
        for group in scopes.values():
            content.append(f"<h3>{_escape(_scope(group['assessment']))}</h3>")
            for conclusion in group["conclusions"].values():
                axis = str(conclusion["axis"])
                content.append(
                    f"<p><strong>{_escape(_AXIS_LABELS.get(axis, label(axis)))}.</strong> {_escape(conclusion.get('plainLanguageSummary'))}</p>"
                )
                content.append(
                    f"<details><summary>Evidence behind this choice</summary><p>{_escape(conclusion.get('quantitativeReason'))}</p><p>{_escape(conclusion.get('biologicalReason'))}</p></details>"
                )
                content.append(
                    _list(
                        texts(
                            [
                                item.get("interpretation")
                                for item in mappings(conclusion.get("tradeoffs"))
                            ]
                        )
                    )
                )
            rows = []
            for identity, candidate in group["candidates"].items():
                setting = candidate
                parameters = mapping(candidate.get("parameters"))
                metrics = mapping(candidate.get("metrics"))
                preferred = [
                    _AXIS_LABELS.get(item["axis"], label(item["axis"]))
                    for item in group["conclusions"].values()
                    if item.get("preferredCandidateId") == identity
                ]
                choice = (
                    "Preferred: " + ", ".join(dict.fromkeys(preferred))
                    if preferred
                    else "Compared"
                )
                if identity == mapping(payload.get("accepted")).get(
                    "selectedCandidateId"
                ):
                    choice = "Selected final settings"
                if partition:
                    rows.append(
                        [
                            parameters.get("leidenResolution"),
                            metrics.get("nClusters"),
                            metrics.get("minClusterCells"),
                            metrics.get("seedStability"),
                            metrics.get("subsampleStability"),
                            _percentage(metrics.get("markerCoherence")),
                            choice,
                        ]
                    )
                else:
                    ranking = {"batchAware": "Within batches", "global": "Global"}.get(
                        str(setting.get("ranking")), "Unavailable"
                    )
                    rows.append(
                        [
                            setting.get("hvgCount"),
                            ranking,
                            parameters.get("dimensions"),
                            parameters.get("neighborsK"),
                            metrics.get("nClusters"),
                            metrics.get("seedStability"),
                            _percentage(metrics.get("markerCoherence")),
                            choice,
                        ]
                    )
            headers = (
                (
                    "Resolution",
                    "Populations",
                    "Smallest population",
                    "Repeat agreement",
                    "Subsample agreement",
                    "Clusters with qualifying markers",
                    "Choice",
                )
                if partition
                else (
                    "Variable genes",
                    "Ranking",
                    "PCA dimensions",
                    "Neighbors",
                    "Populations",
                    "Repeat agreement",
                    "Clusters with qualifying markers",
                    "Choice",
                )
            )
            content.append(_table(headers, rows))
            for item in group["unavailable"].values():
                axis = str(item["axis"])
                content.append(
                    f"<p>{_escape(_AXIS_LABELS.get(axis, label(axis)))}: not compared. {_escape(item.get('reason'))}</p>"
                )
        if content:
            sections.append(f"<section><h2>{title}</h2>{''.join(content)}</section>")
    return "".join(sections)


def render_analysis_document(payload: Mapping[str, Any]) -> str:
    final = mapping(payload.get("finalAnalysis"))
    request = mapping(payload.get("request"))
    counts = mapping(payload.get("clusterCounts"))
    total = sum(int(value) for value in counts.values())
    objective = request.get("studyObjective") or request.get("studyContext") or ""
    assay = final.get("primaryAssay") or request.get("primaryAssay") or "RNA"
    qc = mapping(payload.get("qc"))
    qc_text = ""
    if isinstance(qc.get("retainedCells"), int) and isinstance(
        qc.get("retainedFraction"), int | float
    ):
        qc_text = f"<p>QC retained {_escape(qc['retainedCells'])} cells ({float(qc['retainedFraction']):.1%}).</p>"
    accepted = mapping(payload.get("accepted"))
    outcome = str(
        accepted.get("plainLanguageSummary")
        or "The selected populations and their marker programs are recorded below."
    )
    map_markup = ""
    if payload.get("umap"):
        display = int(payload.get("displayedCells") or total)
        map_markup = f'<figure><img src="{html.escape(str(payload["umap"]), quote=True)}" alt="Final UMAP colored by saved population labels"><figcaption>{display:,} of {total:,} cells shown. Counts and markers use the complete selection.</figcaption></figure>'
    limitations = _list(texts(payload.get("limitations")))
    display_notes = _list(texts(payload.get("displayNotes")))
    mode_note = (
        "<p>The model assessed structured evidence. No plots were supplied for visual inspection.</p>"
        if accepted.get("evidenceMode") == "structured"
        else ""
    )
    parameters = mapping(payload.get("selectedParameters"))
    setting = mapping(payload.get("selectedSetting"))
    methods = _table(
        ("Selected setting", "Value"),
        [
            [name, value]
            for name, value in (
                ("Variable genes", setting.get("hvgCount")),
                ("PCA dimensions", parameters.get("dimensions")),
                ("Neighbors", parameters.get("neighborsK")),
                ("Clustering resolution", parameters.get("leidenResolution")),
                ("Batch correction", parameters.get("useHarmony")),
            )
        ],
    )
    usage = mapping(payload.get("modelUsage"))
    usage_note = ""
    if usage.get("invocations"):
        usage_note = (
            f"<p>Recorded model work: {int(usage['invocations']):,} invocations, "
            f"{int(usage.get('failedInvocations', 0)):,} failed; "
            f"{int(usage.get('requests', 0)):,} completed responses and "
            f"{int(usage.get('validationRetries', 0)):,} validation corrections. "
            f"Reported tokens: {int(usage.get('inputTokens', 0)):,} input and "
            f"{int(usage.get('outputTokens', 0)):,} output, including failed invocations.</p>"
        )
        if usage.get("availability") != "reported":
            usage_note += (
                "<p>Provider usage is incomplete or unavailable for some invocations. "
                "Reported totals are known usage only; missing usage is not zero. "
                "Failed requests without a response are not included in the response count.</p>"
            )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scarf analysis summary</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400&amp;display=swap" rel="stylesheet"><style>{_STYLES}</style></head><body><main>
<header><nav class="header-links" aria-label="Nygen links"><a href="https://www.nygen.io/">Nygen Analytics</a><a href="https://www.nygen.io/products/scarfweb">ScarfWeb</a><a href="https://www.nygen.io/products/cytetype">CyteType</a></nav><small>Scarf analysis</small><h1>Analysis summary</h1><p>{_escape(objective)}</p>
<p class="numbers">{total:,} cells · {len(counts):,} clusters · {_escape(assay)}</p>{qc_text}<p>{_escape(outcome)}</p></header>
<section><h2>Populations and markers</h2><div{' class="population-overview"' if map_markup else ""}>{map_markup}<div>{_population_table(payload)}</div></div></section>
{_qc_section(payload)}{_design_section(payload)}{_comparison_sections(payload)}
<details><summary>Selected methods and evidence</summary>{methods}{mode_note}{usage_note}<p>Repeat and subsample agreement use adjusted Rand index. Marker coverage is the fraction of clusters with qualifying markers. These describe the selected analysis; they are not probabilities of biological correctness.</p></details>
{"<details><summary>Unavailable displays</summary>" + display_notes + "</details>" if display_notes else ""}
{'<section class="limitations"><h2>Limits of this analysis</h2>' + limitations + "</section>" if limitations else ""}
<footer>Generated locally by <a href="https://scarf.readthedocs.io/">Scarf</a>. All numerical evidence is read from the saved analysis; report generation makes no analysis or model calls.</footer>
</main></body></html>"""
