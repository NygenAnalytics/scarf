"""Read-only adapters from the authoritative stage journal to a local report."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...storage.refs import ArtifactRef
from ...storage.stores import zarr_root_path
from ..types import ArtifactReferenceModel
from .contracts import label, mapping, mappings, texts

if TYPE_CHECKING:
    from ...datastore.datastore import DataStore


def _local_root(target: "str | Path | DataStore") -> Path:
    """Resolve a local store without accepting remote locations."""
    from ...datastore.datastore import DataStore

    if isinstance(target, DataStore):
        location = zarr_root_path(target.z)
        if location is None:
            raise ValueError("Agent HTML reports require a local filesystem store")
        path = Path(location)
    elif isinstance(target, Path):
        path = target
    elif isinstance(target, str):
        if "://" in target and not target.startswith("file://"):
            raise ValueError("Agent HTML reports require a local filesystem store")
        path = Path(target.removeprefix("file://"))
    else:
        raise TypeError("Agent HTML reports require a local filesystem store")
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def report_directory(root: Path, run_id: str, workspace: str | None) -> Path:
    """Keep derived pages within the exact journal owner's local directory."""
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid analysis run identifier")
    active = root if workspace is None else (root / workspace).resolve()
    if not active.is_relative_to(root):
        raise ValueError("Analysis workspace resolves outside the store")
    destination = (active / "agents" / "orchestrations" / run_id / "report").resolve()
    if not destination.is_relative_to(active):
        raise ValueError("Analysis report resolves outside the store")
    return destination


def artifact_ref(value: Any) -> ArtifactRef:
    ref = ArtifactReferenceModel.model_validate(value)
    return ArtifactRef(
        scope=ref.scope, assay=ref.assay, kind=ref.kind, artifact_id=ref.artifactId
    )


def scientific_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Select recorded scientific values without inferring decision rationales."""
    from ..orchestrator.rna import validate_analysis_evidence

    validate_analysis_evidence(snapshot)
    final = mapping(snapshot.get("finalAnalysis"))
    stages = mappings(snapshot.get("stages"))
    decisions = [
        decision for stage in stages for decision in mappings(stage.get("decisions"))
    ]
    limitations = texts(final.get("limitations")) + texts(snapshot.get("limitations"))
    assessments = mappings(snapshot.get("analysisReviews"))
    full_assessments = [
        item
        for item in assessments
        if item["scope"] == "full" and item.get("action") == "accept"
    ]
    accepted = full_assessments[-1] if full_assessments else {}
    selected: dict[str, Any] = {}
    qc_profiles: list[dict[str, Any]] = []
    qc_profile_id: Any = None
    context: dict[str, Any] = {}
    study: dict[str, Any] = {}
    for stage in stages:
        report = mapping(stage.get("report"))
        stage_name = str(stage.get("stage", ""))
        if stage_name.startswith("parameter_tuning"):
            evaluations = mappings(report.get("evaluations"))
            recommended = report.get("recommendedCandidateId")
            selected = next(
                (
                    item
                    for item in evaluations
                    if item.get("candidateId") == recommended
                ),
                selected,
            )
        if stage_name == "experimental_context":
            context = report
            study = mapping(mapping(stage.get("outputs")).get("studyContract"))
            qc_profile_id = mapping(report.get("cellQc")).get("profileId")
            qc_profiles = mappings(report.get("qcProfiles"))
        outputs = mapping(stage.get("outputs"))
        for plan_key in ("preprocessingPlan", "resolvedPreprocessingPlan"):
            plan = mapping(outputs.get(plan_key))
            if plan:
                qc_profile_id = mapping(plan.get("cellQc")).get("profileId")
    qc = next(
        (item for item in qc_profiles if item.get("profileId") == qc_profile_id), {}
    )
    limitations.extend(texts(study.get("limitations")))
    limitations.extend(
        str(item["explanation"])
        for item in mappings(accepted.get("populationConcerns"))
        if item.get("explanation")
    )
    comparison_limits = {}
    for comparison in mappings(
        mapping(context.get("characterization")).get("comparisons")
    ):
        if comparison.get("status") != "unsupported":
            continue
        proposal = mapping(comparison.get("proposal"))
        question = (
            f"{label(str(proposal.get('response', 'Study factor')))} against "
            + ", ".join(
                label(name) for name in texts(proposal.get("explanatoryColumns"))
            )
        )
        if proposal.get("conditionedOn"):
            question += f" within {label(str(proposal['conditionedOn']))} groups"
        reasons = "; ".join(
            label(reason) for reason in texts(comparison.get("reasons"))
        )
        comparison_limits[str(comparison.get("evidenceId"))] = (
            f"{question}: the requested association was not computed. {reasons}."
        )
    displayed_limits = []
    for limitation in limitations:
        match = next(
            (
                text
                for identity, text in comparison_limits.items()
                if identity and identity in limitation
            ),
            None,
        )
        displayed_limits.append(match or limitation)
    population = mapping(
        mapping(accepted.get("populationSupport")).get(
            str(accepted.get("selectedCandidateId"))
        )
    )
    if population:
        for key in ("cellSelection", "clusters"):
            reference = population.get(key)
            if not isinstance(reference, Mapping):
                raise ValueError(
                    f"Reported population support lacks its {key} reference"
                )
            if ArtifactRef.from_dict(reference) != artifact_ref(final.get(key)):
                raise ValueError(
                    "Reported population support differs from the final analysis"
                )
        if population.get("candidateId") != accepted.get("selectedCandidateId"):
            raise ValueError("Reported population support belongs to another candidate")
    return {
        "request": mapping(snapshot.get("request")),
        "modelUsage": mapping(snapshot.get("modelUsage")),
        "finalAnalysis": final,
        "decisions": decisions,
        "assessments": assessments,
        "accepted": accepted,
        "context": context,
        "study": study,
        "populationSupport": population,
        "limitations": list(dict.fromkeys(displayed_limits)),
        "selectedParameters": mapping(selected.get("parameters")),
        "selectedMetrics": mapping(selected.get("metrics")),
        "selectedSetting": mapping(
            mapping(accepted.get("settings")).get(
                str(accepted.get("selectedCandidateId"))
            )
        ),
        "selectedFeatures": mapping(
            mapping(accepted.get("featureEvidence")).get(
                str(accepted.get("selectedCandidateId"))
            )
        ),
        "qc": qc,
        "qcProfiles": qc_profiles,
    }
