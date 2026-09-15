"""Representation interventions use exact eligible genes and protected programs."""

import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent.orchestrator import rna_tuning
from scarf.agent.orchestrator.models import artifact_model_to_ref
from scarf.agent.types import ArtifactReferenceModel
from scarf.storage.refs import ArtifactRef
from tests.test_agent_rna_adaptive import checkpoints as memory_checkpoints  # noqa: F401
from tests.test_agent_rna_evidence_mode import make_run


@pytest.fixture
def feature_run(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, object())
    names = np.array(["MT-CO1", "RPL3", "XIST", "CD3D", *[f"G{i}" for i in range(96)]])
    ids = np.array([f"ENSG{i}" for i in range(len(names))])
    arrays: dict[str, dict[str, np.ndarray]] = {}

    def selection(mask: Any, **kwargs: Any) -> ArtifactRef:
        mask = np.asarray(mask, dtype=bool)
        identifier = hashlib.sha256(mask.tobytes()).hexdigest()
        arrays[identifier] = {
            "values": mask.copy(),
            "corrected_variance": np.arange(100, dtype=float),
        }
        return ArtifactRef(
            scope="assay", assay="RNA", kind="feature_selection", artifact_id=identifier
        )

    allowed = selection(np.arange(100) != 99)
    eligible = selection((np.arange(100) != 99) & (np.arange(100) != 0))
    setting = run.settings[selected.candidateId]
    setting.features = ArtifactReferenceModel.from_artifact_ref(eligible)
    setting.eligibleFeatures = setting.features
    setting.hvgCount = 98
    run.handoff.graphFeatureCandidates["eligibleAll"] = (
        ArtifactReferenceModel.from_artifact_ref(allowed)
    )
    run.plan.assays[0].featureParameters = {}
    run.family_patterns = {
        "mitochondrial": "^MT-",
        "ribosomal": "^RP[LS]",
        "sexLinked": "^XIST$",
    }
    run.store = SimpleNamespace(
        load_artifact=lambda reference: {
            key: value.copy() for key, value in arrays[reference.artifact_id].items()
        },
        get_assay=lambda _: SimpleNamespace(
            feats=SimpleNamespace(
                N=100, fetch_all=lambda column: names if column == "names" else ids
            )
        ),
        set_feature_selection=selection,
    )

    def rank(
        store: Any, *, eligible: ArtifactRef, top_n: int, **kwargs: Any
    ) -> ArtifactRef:
        mask = store.load_artifact(eligible)["values"].copy()
        mask[np.flatnonzero(mask)[top_n:]] = False
        return selection(mask)

    monkeypatch.setattr(rna_tuning, "rank_core_hvgs", rank)
    return run, selected, arrays, selection


@pytest.mark.parametrize(
    "intervention",
    ["includeFamily", "includeFeature", "excludeFamily", "excludeFeature"],
)
def test_feature_policy_changes_only_representation_eligibility(
    feature_run: Any, intervention: str
) -> None:
    run, selected, arrays, _ = feature_run
    marker_features = run.marker_features
    previous = run.settings[selected.candidateId].model_copy(deep=True)
    value = {
        "includeFamily": "mitochondrial",
        "includeFeature": "ENSG0",
        "excludeFamily": "ribosomal",
        "excludeFeature": "ENSG1",
    }[intervention]
    changed = run.apply_experiment(
        selected, {"parameter": intervention, "value": value}
    )
    old_mask = arrays[previous.eligibleFeatures.artifactId]["values"]
    new_mask = arrays[changed.eligibleFeatures.artifactId]["values"]
    expected = 0 if intervention.startswith("include") else 1
    assert np.flatnonzero(old_mask != new_mask).tolist() == [expected]
    assert not new_mask[99], "Ineligible genes cannot be reinstated"
    assert changed.parameters == previous.parameters
    assert run.settings[selected.candidateId] == previous
    assert run.marker_features == marker_features


@pytest.mark.parametrize(
    "protection",
    [
        {"protectFeatures": ["ENSG1"]},
        {"protectFamilies": ["ribosomal"]},
        {"protectFamilies": ["sexLinked"]},
    ],
)
def test_objective_protection_blocks_exclusion_before_feature_execution(
    feature_run: Any, protection: dict[str, Any]
) -> None:
    run, selected, arrays, _ = feature_run
    run.plan.assays[0].featureParameters = protection
    value = "XIST" if protection.get("protectFamilies") == ["sexLinked"] else "RPL3"
    before = len(arrays)
    with pytest.raises(ValueError, match="objective-protected"):
        run.apply_experiment(selected, {"parameter": "excludeFeature", "value": value})
    assert len(arrays) == before


def test_missing_feature_or_underpowered_representation_cannot_be_executed(
    feature_run: Any,
) -> None:
    run, selected, _, _ = feature_run
    with pytest.raises(ValueError, match="absent from the assay"):
        run.apply_experiment(
            selected, {"parameter": "excludeFeature", "value": "missing"}
        )
    with pytest.raises(ValueError, match="fixed PCA dimension"):
        run.apply_experiment(selected, {"parameter": "hvgCount", "value": 10})
    with pytest.raises(ValueError, match="explicit technical column"):
        run.apply_experiment(
            selected, {"parameter": "hvgRanking", "value": "batchAware"}
        )


def test_nominations_follow_eligible_protected_genes_and_skip_unknown_families(
    feature_run: Any,
) -> None:
    run, selected, _, _ = feature_run
    setting = run.settings[selected.candidateId]
    nominate = rna_tuning.RnaTuningRun._feature_nomination
    run.plan.assays[0].featureParameters = {
        "protectFamilies": ["notRegistered", "mitochondrial"]
    }
    assert nominate(run, setting) == {
        "parameter": "includeFamily",
        "value": "mitochondrial",
    }
    run.plan.assays[0].featureParameters = {
        "protectFeatures": ["ENSG1"],
        "proposedExcludeFeatures": ["RPL3", "missing"],
    }
    assert nominate(run, setting) is None
    run.plan.assays[0].featureParameters = {"proposedExcludeFamilies": ["ribosomal"]}
    assert nominate(run, setting) == {
        "parameter": "excludeFamily",
        "value": "ribosomal",
    }


def test_offered_experiments_respect_population_rank_and_covariate_kind(
    feature_run: Any,
) -> None:
    run, selected, _, _ = feature_run
    setting = run.settings[selected.candidateId]
    setting.hvgCount = 22
    setting.ranking = "batchAware"
    setting.rankingColumn = "capture"
    setting.parameters.useHarmony = True
    run.scope_sizes["full"] = 22
    run.batch_columns = ["capture", "age"]
    run.study.columnKinds = {"capture": "categorical", "age": "continuous"}
    options = run.experiments(selected)
    assert "dimensions:30" not in options
    assert "neighborsK:41" not in options
    assert "hvgRanking:batchAware:age" not in options
    assert "hvgRanking:batchAware:capture" not in options
    assert "hvgRanking:global" in options and "useHarmony:false" in options


def test_batch_ranking_requires_supported_groups_and_reuses_core_variability(
    feature_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scarf.agent.experimental_context import characterization

    run, selected, _, selection = feature_run
    eligible = artifact_model_to_ref(
        run.settings[selected.candidateId].eligibleFeatures
    )
    labels = np.repeat(["a", "b", "small"], [25, 25, 5])
    calls: list[dict[str, Any]] = []
    run.batch_columns = ["capture"]
    run.store.zw = None
    run.store.cells = object()
    monkeypatch.setattr(
        characterization,
        "_SelectionBoundCells",
        lambda *a: SimpleNamespace(fetch=lambda _: labels),
    )

    def filter_cells(columns: Any, lower: Any, upper: Any, **kwargs: Any) -> str:
        assert columns == ["capture"] and lower == upper
        assert kwargs["cell_selection"] == run.cells
        return lower[0]

    core_features = selection(np.ones(100, dtype=bool))
    summary = ArtifactRef(
        scope="assay", assay="RNA", kind="feature_summary", artifact_id="b" * 64
    )
    original_load = run.store.load_artifact
    run.store.load_artifact = lambda ref: (
        {"normed_n": np.repeat(25, 100)} if ref == summary else original_load(ref)
    )
    run.store.filter_cells = filter_cells
    run.store.inspect_artifact = lambda ref: SimpleNamespace(
        inputs={"feature_summary": summary.to_dict()}
    )

    def select_hvgs(cells: Any, **kwargs: Any) -> ArtifactRef:
        calls.append({"cells": cells, **kwargs})
        return core_features

    run.store.select_hvgs = select_hvgs
    with pytest.raises(ValueError, match="approved technical column"):
        run.batch_ranking(eligible, 40, "donor")
    assert not calls
    indices = run.batch_ranking(eligible, 40, "capture")
    assert {call["cells"] for call in calls} == {"a", "b"}
    assert all(call["blacklist"] == "" and call["top_n"] == 100 for call in calls)
    assert len(indices) > 0 and 0 not in indices and 99 not in indices
    labels[:] = "a"
    with pytest.raises(ValueError, match="two groups"):
        run.batch_ranking(eligible, 40, "capture")
