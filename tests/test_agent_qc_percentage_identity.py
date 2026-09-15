"""Exact gene ownership for percentage evidence and QC execution."""

from pathlib import Path

import numpy as np
import pytest

from scarf.agent.experimental_context.characterization import _SelectionBoundCells
from scarf.agent.experimental_context.contracts import ExperimentalContextDependencies
from scarf.agent.experimental_context.qc_evidence import (
    _derive_missing_percentage_artifacts,
    _offered_qc_profiles,
    _qc_metric_sources,
)
from scarf.agent.cell_quality.execution import execute_registered_cell_qc
from scarf.agent.tools import core_artifact_reference
from scarf.agent.parameter_tuning.diagnostics import _covariate_associations
from scarf.datastore.datastore import DataStore
from scarf.metadata.selection import NamedCellArtifact
from tests.agent_orchestrator_store import create_store


def test_exact_mitochondrial_artifact_owns_filtering_without_overwriting_metadata(
    tmp_path: Path,
) -> None:
    path = create_store(tmp_path / "percentages.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    store.get_assay("RNA").feats.insert(
        "names",
        np.asarray(["MT-CO1", "MTAP", "MTOR", "RPS3"]),
        overwrite=True,
    )
    imported = np.asarray([100.0, 60.0, 100.0, 83.3333333333])
    store.cells.insert("RNA_percentMito", imported)
    selected = store.snapshot_cell_selection("I")
    sources = _derive_missing_percentage_artifacts(
        store,
        cell_selection=selected,
        driver=("RNA", "RNA"),
        quality_sources=[],
    )
    deps = ExperimentalContextDependencies(
        store=store,
        cellSelection=selected,
        cells=_SelectionBoundCells(store.zw, store.cells, selected),
        qcAssay="RNA",
        studyContext="Human nuclei with mitochondrial QC.",
        qualityMetricArtifacts=sources,
    )
    values, metadata, artifacts, evidence, concordance, _, _ = _qc_metric_sources(
        deps, ("RNA", "RNA")
    )
    np.testing.assert_array_equal(store.cells.fetch_all("RNA_percentMito"), imported)
    np.testing.assert_allclose(values["RNA_percentMito"], [80, 0, 200 / 3, 0])
    assert "RNA_percentMito" not in metadata
    assert {item.name for item in artifacts} == {"RNA_percentMito", "RNA_percentRibo"}
    mito_sources = [item for item in evidence if item.metricRole == "mitochondrial"]
    assert [(item.sourceType, item.usableForFiltering) for item in mito_sources] == [
        ("metadataColumn", False),
        ("artifact", True),
    ]
    assert mito_sources[1].executionName == "RNA_percentMito"
    assert concordance[0].exactlyEqual is False
    support = {}
    correlation = _covariate_associations(
        store,
        selected,
        np.asarray(values["RNA_percentMito"])[:, None],
        ["RNA_percentMito"],
        ["qc"],
        support=support,
        column_artifacts={
            "RNA_percentMito": core_artifact_reference(sources[0].artifact)
        },
    )
    np.testing.assert_allclose(correlation, [[1.0]])
    assert support["RNA_percentMito"]["kind"] == "continuous"
    feature_ref = next(
        ref for ref in mito_sources[1].inputArtifacts if ref.kind == "feature_selection"
    )
    np.testing.assert_array_equal(
        store.load_artifact(core_artifact_reference(feature_ref))["values"][:],
        [True, False, False, False],
    )
    profile = next(
        value
        for value in _offered_qc_profiles(deps)
        if value.registeredProfile == "globalMad5"
    )
    assert "RNA_percentMito" not in profile.attributes
    assert any(bound["metric"] == "RNA_percentMito" for bound in profile.resolvedBounds)
    # The projected thresholds must also execute under the same metric names.
    filtered, _ = execute_registered_cell_qc(
        store,
        "globalMad5",
        profile_parameters=profile.parameters,
        expected_active_cells=profile.activeCells,
        expected_retained_cells=profile.retainedCells,
        expected_flag_counts=profile.flaggedCells,
        attrs=profile.attributes,
        artifact_metrics=[
            NamedCellArtifact(
                name=item.name, artifact=core_artifact_reference(item.artifact)
            )
            for item in profile.artifactMetrics
        ],
        cell_selection=selected,
    )
    assert (
        int(np.asarray(store.load_artifact(filtered)["values"][:]).sum())
        == profile.retainedCells
    )


def test_changed_gene_definition_invalidates_percentage_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = create_store(tmp_path / "changed.zarr")
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    selected = store.snapshot_cell_selection("I")
    assay = store.get_assay("RNA")
    compute = assay._compute_feature_percentage
    calculated = []

    def counted(cells, genes):
        calculated.append(genes.copy())
        return compute(cells, genes)

    monkeypatch.setattr(assay, "_compute_feature_percentage", counted)
    first = _derive_missing_percentage_artifacts(
        store,
        cell_selection=selected,
        driver=("RNA", "RNA"),
        quality_sources=[],
    )
    repeated = _derive_missing_percentage_artifacts(
        store,
        cell_selection=selected,
        driver=("RNA", "RNA"),
        quality_sources=[],
    )
    assert repeated == first
    assert len(calculated) == 2
    store.get_assay("RNA").feats.insert(
        "names",
        np.asarray(["MT-CO1", "RPS3", "MT-ND1", "GENE2"]),
        overwrite=True,
    )
    changed = _derive_missing_percentage_artifacts(
        store,
        cell_selection=selected,
        driver=("RNA", "RNA"),
        quality_sources=[],
    )
    assert changed[0].artifact != first[0].artifact
    assert changed[1].artifact == first[1].artifact
    assert len(calculated) == 3


def test_unresolved_mitochondrial_definition_cannot_use_imported_percentages(
    tmp_path: Path,
) -> None:
    store = DataStore(
        str(create_store(tmp_path / "unresolved.zarr")),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r+",
    )
    store.get_assay("RNA").feats.insert(
        "names", np.asarray(["MTAP", "MTOR", "RPS3", "ACTB"]), overwrite=True
    )
    imported = np.asarray([90.0, 0.0, 0.0, 0.0])
    store.cells.insert("RNA_percentMito", imported)
    selected = store.snapshot_cell_selection("I")
    sources = _derive_missing_percentage_artifacts(
        store,
        cell_selection=selected,
        driver=("RNA", "RNA"),
        quality_sources=[],
    )
    deps = ExperimentalContextDependencies(
        store=store,
        cellSelection=selected,
        cells=_SelectionBoundCells(store.zw, store.cells, selected),
        qcAssay="RNA",
        studyContext="Mitochondrial genes cannot be identified in these annotations.",
        qualityMetricArtifacts=sources,
    )
    values, metadata, _, evidence, _, notes, _ = _qc_metric_sources(
        deps, ("RNA", "RNA")
    )
    assert "RNA_percentMito" not in values
    assert "RNA_percentMito" not in metadata
    assert all(
        not item.usableForFiltering
        for item in evidence
        if item.metricRole == "mitochondrial"
    )
    assert any(
        "mitochondrial percentage has no exact usable artifact" in n for n in notes
    )
    np.testing.assert_array_equal(store.cells.fetch_all("RNA_percentMito"), imported)
