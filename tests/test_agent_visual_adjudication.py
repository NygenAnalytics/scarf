"""Model-facing plots use matched artifacts and bounded selected-cell reads."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scarf.agent.orchestrator import tuning
from scarf.agent.parameter_tuning.contracts import (
    ArtifactRecord,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterMetrics,
)
from scarf.agent.types import ArtifactReferenceModel


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.rows = []

    def get_orthogonal_selection(self, selection):
        self.rows.append(np.asarray(selection[0]).copy())
        assert len(selection[0]) <= 5000
        return self.values[selection]


def _visual_fixture(monkeypatch, *, batch=True, doublet=False, qc_size=20):
    arrays = {}
    selections = {}

    def ref(kind, index, scope="assay"):
        return ArtifactRecord(
            kind=kind,
            scope=scope,
            assay="RNA" if scope == "assay" else None,
            artifactId=f"{index:064x}",
        )

    cells = ref("cell_selection", 1, "datastore")
    full = ref("cell_selection", 2, "datastore")
    selections[cells.artifactId] = np.arange(5, 15)
    selections[full.artifactId] = np.arange(20)
    coordinates = ref("reduction", 3)
    harmony = ref("reduction", 4)
    clusters = ref("cluster_labels", 5)
    features = ref("feature_selection", 6)
    marker = ref("marker_table", 7)
    qc = ref("quality_metric", 8)
    for item, field, value in (
        (coordinates, "data", np.arange(20).reshape(10, 2)),
        (harmony, "data", np.arange(20).reshape(10, 2) * 0.8),
        (clusters, "values", np.arange(10) % 2),
        (qc, "values", np.linspace(0.1, 1.1, qc_size)),
    ):
        arrays[item.artifactId] = {field: _Array(value)}
    genes = ["MT-CO1", "MRPL1", "RPL1", "CCN1", "HLA-A", "H2-A", "HIST1", "XIST"]
    native = ParameterCandidateEvaluation(
        candidateId="native",
        status="done",
        eligible=True,
        parameters=ParameterCandidate(candidateId="native"),
        cellSelection=ArtifactReferenceModel.model_validate(cells.model_dump()),
        artifacts={
            "pca": coordinates,
            "clusters": clusters,
            "graphFeatures": features,
            "markerTable": marker,
        },
        metrics=ParameterMetrics(
            batchMixing={"batch": 0.5} if batch else {},
            topMarkerGenes={"0": genes, "1": ["MS4A1"]},
        ),
    )
    corrected = native.model_copy(
        deep=True,
        update={
            "candidateId": "harmony",
            "parameters": native.parameters.model_copy(update={"useHarmony": True}),
        },
    )
    corrected.artifacts["harmony"] = harmony
    if doublet:
        score = ref("doublet_score", 9)
        arrays[score.artifactId] = {"values": _Array(np.linspace(0.01, 0.9, 10))}
        native.artifacts["doubletScore:all"] = score
    store = SimpleNamespace(
        zw=object(),
        cells=SimpleNamespace(columns=["batch"] if batch else []),
        load_artifact=lambda item: arrays[item.artifact_id],
        inspect_artifact=lambda item: SimpleNamespace(
            inputs={
                "cell_selection": {
                    "type": "artifact",
                    "scope": full.scope,
                    "kind": full.kind,
                    "artifact_id": full.artifactId,
                }
            }
        ),
        get_markers=lambda *a, **k: pd.DataFrame(
            {
                "group_id": ["0"] * len(genes),
                "feature_name": genes,
                "score": np.linspace(0.3, 0.9, len(genes)),
            }
        ),
    )
    monkeypatch.setattr(tuning, "as_zarr_array", lambda array, **kwargs: array)
    monkeypatch.setattr(
        tuning,
        "read_stored_selection_indices",
        lambda group, item, **kwargs: selections[item.artifact_id],
    )
    monkeypatch.setattr(
        tuning,
        "read_metadata_rows_chunkwise",
        lambda metadata, name, rows: np.asarray(["batch-a", "batch-b"] * 10)[rows],
    )
    return store, native, corrected, qc, arrays, selections


@pytest.mark.parametrize(
    "batch,doublet,qc_size", [(True, True, 20), (False, False, 10)]
)
def test_native_harmony_visuals_preserve_matching_and_exact_qc_projection(
    monkeypatch, batch, doublet, qc_size
):
    store, native, corrected, qc, arrays, _ = _visual_fixture(
        monkeypatch, batch=batch, doublet=doublet, qc_size=qc_size
    )
    captured = []
    savefig = plt.Figure.savefig

    def save(figure, *args, **kwargs):
        captured.extend(
            text.get_text() for axis in figure.axes for text in axis.get_xticklabels()
        )
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", save)
    outputs = tuning._analysis_visual_content(
        store,
        native,
        [native, corrected],
        qc_artifact_metrics=[("Exact mitochondrial fraction", qc)],
    )
    assert {item.identifier for item in outputs} == {
        "analysis-overview",
        "native-harmony-comparison",
        "marker-score-heatmap",
        "qc-doublet-diagnostics",
    }
    assert all(item.data.startswith(b"\x89PNG") for item in outputs)
    assert "MT-CO1 [mitochondrial]" in captured
    assert "XIST [sex-linked]" in captured
    expected = np.arange(5, 15) if qc_size == 20 else np.arange(10)
    assert np.array_equal(arrays[qc.artifactId]["values"].rows[-1], expected)


@pytest.mark.parametrize(
    "damage,error",
    [
        ("missingCoordinates", "lacks visualizable"),
        ("oneCoordinate", "two dimensions"),
        ("clusterLength", "do not align"),
        ("doubletSelection", "lacks its exact cell selection"),
        ("qcSelection", "lacks its cell selection"),
        ("qcCoverage", "does not cover selected cells"),
        ("missingHarmony", "lacks visual artifacts"),
    ],
)
def test_visual_evidence_rejects_unmatched_or_missing_artifact_inputs(
    monkeypatch, damage, error
):
    store, native, corrected, qc, arrays, selections = _visual_fixture(
        monkeypatch, doublet=True
    )
    if damage == "missingCoordinates":
        del native.artifacts["pca"]
    elif damage == "oneCoordinate":
        arrays[native.artifacts["pca"].artifactId]["data"] = _Array(np.ones((10, 1)))
    elif damage == "clusterLength":
        arrays[native.artifacts["clusters"].artifactId]["values"] = _Array(np.arange(9))
    elif damage == "doubletSelection":
        arrays[native.artifacts["doubletScore:all"].artifactId]["values"] = _Array(
            np.arange(9)
        )
    elif damage == "qcSelection":
        store.inspect_artifact = lambda item: SimpleNamespace(inputs={})
    elif damage == "qcCoverage":
        selections[f"{2:064x}"] = np.arange(20, 40)
    elif damage == "missingHarmony":
        del corrected.artifacts["harmony"]
    try:
        with pytest.raises(ValueError, match=error):
            tuning._analysis_visual_content(
                store,
                native,
                [native, corrected],
                qc_artifact_metrics=[("mitochondrial", qc)],
            )
    finally:
        plt.close("all")
