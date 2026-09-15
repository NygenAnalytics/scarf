"""Numerical evidence remains bounded and rejects mismatched scientific inputs."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent.parameter_tuning import diagnostics as d
from scarf.agent.parameter_tuning.contracts import (
    ArtifactRecord,
    ParameterCandidateEvaluation,
)
from scarf.storage.refs import ArtifactRef
from tests.agent_examples import example


class _Blocks:
    def __init__(self, values: np.ndarray, limit: int = 65_536) -> None:
        self.values = values
        self.shape, self.dtype = values.shape, values.dtype
        self.limit = limit
        self.reads: list[int] = []

    def __getitem__(self, rows: slice) -> np.ndarray:
        output = self.values[rows]
        assert len(output) <= self.limit
        self.reads.append(len(output))
        return output

    def __array__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Scientific arrays must use bounded reads")


def _ref(kind: str, value: int = 1) -> ArtifactRef:
    return ArtifactRef(
        "datastore" if kind == "cell_selection" else "assay",
        kind,
        f"{value:064x}",
        None if kind == "cell_selection" else "RNA",
    )


def test_streamed_component_and_loading_statistics_match_dense_reference() -> None:
    rng = np.random.default_rng(7)
    coordinates = _Blocks(rng.normal(size=(70_000, 3)))
    np.testing.assert_allclose(
        d._component_variance(coordinates), coordinates.values.var(axis=0), rtol=1e-12
    )
    assert coordinates.reads == [65_536, 4464]
    loadings = _Blocks(rng.normal(size=(9000, 3)), 8192)
    indices = np.arange(9000) * 2
    family = np.arange(9000) % 3 == 0
    top, magnitude, enrichment = d._top_loadings(loadings, indices, {"family": family})
    expected = np.argsort(-np.abs(loadings.values), axis=0, kind="stable")[:20].T
    np.testing.assert_array_equal(top, indices[expected])
    for column in range(3):
        np.testing.assert_allclose(
            magnitude[column], np.abs(loadings.values[expected[column], column])
        )
        assert enrichment[0, column] == pytest.approx(
            family[expected[column]].mean() / family.mean()
        )
    assert loadings.reads == [8192, 808]


@pytest.mark.parametrize(
    "values", [np.ones(3), np.ones((0, 2)), np.asarray([[1.0, np.nan]])]
)
def test_pca_variance_cannot_come_from_invalid_coordinates(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        d._component_variance(values)


@pytest.mark.parametrize("damage", ["shape", "empty", "familyMask", "nonfinite"])
def test_loading_programs_require_exact_finite_feature_axis(damage: str) -> None:
    values = np.ones((3, 2))
    indices = np.arange(3)
    family = np.ones(3, dtype=bool)
    if damage == "shape":
        values = values[:2]
    elif damage == "empty":
        values = values[:, :0]
    elif damage == "familyMask":
        family = family[:2]
    else:
        values[0, 0] = np.inf
    with pytest.raises(ValueError):
        d._top_loadings(values, indices, {"family": family})


def test_family_influence_distinguishes_representation_programs_without_removing_genes() -> (
    None
):
    names = np.asarray(
        [
            "MT-CO1",
            "MTOR",
            "RPS3",
            "MRPS3",
            "MKI67",
            "HBA1",
            "IGHM",
            "FOS",
            "ATF3",
            "XIST",
            "OTHER",
        ]
    )
    expected = {
        "mitochondrial": {0},
        "ribosomal": {2},
        "mitoribosomal": {3},
        "cellCycle": {4},
        "hemoglobin": {5},
        "immuneReceptor": {6},
        "stress": {7},
        "dissociation": {7, 8},
        "sex": {9},
    }
    for family, positions in expected.items():
        assert set(np.flatnonzero(d._family_mask(names, family))) == positions
    assert d._family_mask(names, "unregisteredFamily") is None
    np.testing.assert_array_equal(
        names,
        [
            "MT-CO1",
            "MTOR",
            "RPS3",
            "MRPS3",
            "MKI67",
            "HBA1",
            "IGHM",
            "FOS",
            "ATF3",
            "XIST",
            "OTHER",
        ],
    )


def _variance_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, ArtifactRef, dict[str, Any]]:
    normalized, cells, features = (
        _ref("normalized"),
        _ref("cell_selection"),
        _ref("feature_selection"),
    )
    data = _Blocks(
        np.column_stack([np.arange(9000), np.ones(9000), np.arange(9000) % 3]).astype(
            float
        ),
        8192,
    )
    payload: dict[str, Any] = {"data": data}
    inputs = {"cell_selection": cells, "feature_selection": features}
    store = SimpleNamespace(
        inspect_artifact=lambda _: SimpleNamespace(inputs=inputs),
        load_artifact=lambda _: payload,
    )
    status = SimpleNamespace(
        inputs={"normalized": normalized, "pca_cell_selection": cells}
    )
    monkeypatch.setattr(d, "as_zarr_array", lambda value, **_: value)
    return store, status, features, payload


def test_scaled_pca_variance_reuses_saved_feature_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, status, features, payload = _variance_inputs(monkeypatch)

    def evaluate() -> float:
        return d._scaled_total_variance(
            store, status, n_rows=9000, n_features=3, feature_selection=features
        )

    assert evaluate() == 2
    assert payload["data"].reads == [8192, 808]
    values = payload["data"].values
    payload.update(
        feature_sum=values.sum(axis=0),
        feature_squared_sum=np.square(values).sum(axis=0),
    )
    payload["data"].reads.clear()
    assert evaluate() == 2
    assert payload["data"].reads == []


@pytest.mark.parametrize(
    "damage",
    [
        "missingInput",
        "differentCells",
        "differentGenes",
        "differentShape",
        "nonfinite",
        "invalidSummaries",
        "constantGenes",
    ],
)
def test_scaled_variance_denominator_requires_exact_normalization(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    store, status, features, payload = _variance_inputs(monkeypatch)
    if damage == "missingInput":
        status.inputs.pop("normalized")
    elif damage == "differentCells":
        status.inputs["pca_cell_selection"] = _ref("cell_selection", 2)
    elif damage == "differentGenes":
        features = _ref("feature_selection", 2)
    elif damage == "differentShape":
        payload["data"].shape = (9000, 4)
    elif damage == "nonfinite":
        payload["data"].values[0, 0] = np.nan
    elif damage == "invalidSummaries":
        payload.update(feature_sum=np.ones(2), feature_squared_sum=np.ones(2))
    else:
        payload["data"].values[:] = 1
    with pytest.raises(ValueError):
        d._scaled_total_variance(
            store, status, n_rows=9000, n_features=3, feature_selection=features
        )


@pytest.mark.parametrize("damage", ["roles", "rows", "kind", "artifactKind"])
def test_pca_covariate_evidence_rejects_wrong_type_or_selection(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    monkeypatch.setattr(
        d,
        "_aligned_metadata_values",
        lambda *_: np.arange(3 if damage == "rows" else 4),
    )
    monkeypatch.setattr(
        d,
        "resolve_cell_aligned_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(values=np.arange(4)),
    )
    with pytest.raises(ValueError):
        d._covariate_associations(
            SimpleNamespace(zw=None),
            _ref("cell_selection"),
            np.ones((4, 2)),
            ["covariate"],
            [] if damage == "roles" else ["technical"],
            column_kinds={
                "covariate": "unknown" if damage == "kind" else "categorical"
            },
            column_artifacts={"covariate": _ref("quality_metric")}
            if damage == "artifactKind"
            else {},
        )


def test_missing_categorical_covariates_have_zero_matched_support_without_coordinate_reads() -> (
    None
):
    coordinates = _Blocks(np.ones((4, 2)))
    np.testing.assert_array_equal(
        d._categorical_association(coordinates, np.asarray([None] * 4)), [0, 0]
    )
    assert coordinates.reads == []


def test_doublet_summary_uses_bounded_scores_and_preserves_extremes() -> None:
    values = _Blocks(np.linspace(0, 1, 70_000))
    summary, sample = d._bounded_score_summary(values, maximum_sample_size=1000)
    assert len(sample) <= 1000 and summary["sampleSize"] == len(sample)
    assert summary["minimum"] == 0 and summary["maximum"] == 1
    assert summary["p90"] == pytest.approx(np.quantile(sample, 0.9))
    assert values.reads == [65_536, 4464]


@pytest.mark.parametrize(
    "values,limit",
    [
        (np.ones(3), 0),
        (np.ones((2, 2)), 10),
        (np.ones(0), 10),
        (np.asarray([1.0, np.nan]), 10),
    ],
)
def test_doublet_summary_cannot_hide_invalid_scores(
    values: np.ndarray, limit: int
) -> None:
    with pytest.raises(ValueError):
        d._bounded_score_summary(values, maximum_sample_size=limit)


@pytest.mark.parametrize("damage", ["gap", "captureSummary", "selection"])
def test_resumed_doublets_require_complete_capture_inventory(damage: str) -> None:
    evaluation = ParameterCandidateEvaluation(
        artifacts={"doubletScore:0": ArtifactRecord.from_ref(_ref("quality_metric"))}
    )
    evaluation.metrics.doubletScoreByCapture = {"capture": {"p90": 0.8}}
    if damage == "gap":
        evaluation.artifacts["doubletScore:1"] = evaluation.artifacts.pop(
            "doubletScore:0"
        )
    elif damage == "captureSummary":
        evaluation.metrics.doubletScoreByCapture = {}
    with pytest.raises(ValueError):
        d.restore_advisory_doublets(evaluation, capture_column="capture")


def _score_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, dict[ArtifactRef, np.ndarray]]:
    parent, selected, score = (
        _ref("cell_selection"),
        _ref("cell_selection", 2),
        _ref("quality_metric"),
    )
    selections = {parent: np.arange(4), selected: np.arange(4)}
    store = SimpleNamespace(
        zw=None, load_artifact=lambda _: {"values": np.asarray([0.1, 0.2, 0.8, 0.9])}
    )
    monkeypatch.setattr(d, "as_zarr_array", lambda value, **_: value)
    monkeypatch.setattr(
        d, "read_stored_selection_indices", lambda _root, ref, **_: selections[ref]
    )
    evidence = d.AdvisoryDoubletScores(
        (score,),
        (selected,),
        _ref("connectivity_map"),
        _ref("cluster_labels"),
        score_summaries=({"p90": 0.75},),
    )
    return store, evidence, selections


@pytest.mark.parametrize(
    "damage",
    [
        "labels",
        "parentOrder",
        "summaryCount",
        "scoreShape",
        "outsideParent",
        "overlap",
        "nonfinite",
    ],
)
def test_doublet_concentration_cannot_compare_different_or_overlapping_cells(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    from dataclasses import replace

    store, evidence, selections = _score_evidence(monkeypatch)
    labels = np.asarray([1, 1, 2, 2])
    if damage == "labels":
        labels = labels[:3]
    elif damage == "parentOrder":
        selections[_ref("cell_selection")] = np.asarray([1, 0, 2, 3])
    elif damage == "summaryCount":
        evidence = replace(evidence, score_summaries=({"p90": 0.5}, {"p90": 0.5}))
    elif damage == "scoreShape":
        store.load_artifact = lambda _: {"values": np.ones(3)}
    elif damage == "outsideParent":
        selections[_ref("cell_selection", 2)] = np.asarray([0, 1, 2, 4])
    elif damage == "overlap":
        evidence = replace(
            evidence,
            scores=evidence.scores * 2,
            cell_selections=evidence.cell_selections * 2,
            score_summaries=evidence.score_summaries * 2,
        )
    else:
        store.load_artifact = lambda _: {"values": np.asarray([np.nan, 0, 0, 1])}
    with pytest.raises(ValueError):
        d._doublet_concentration(store, labels, _ref("cell_selection"), evidence)


def test_doublet_concentration_uses_matched_capture_thresholds_and_empty_evidence_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    store, evidence, _ = _score_evidence(monkeypatch)
    assert (
        d._doublet_concentration(
            store, np.asarray([1, 1, 2, 2]), _ref("cell_selection"), evidence
        )
        == 2
    )
    missing = replace(evidence, scores=(), cell_selections=(), score_summaries=())
    assert (
        d._doublet_concentration(
            store, np.asarray([1, 1, 2, 2]), _ref("cell_selection"), missing
        )
        is None
    )


def test_native_doublet_reconstruction_uses_pca_and_preserves_selected_parameters() -> (
    None
):
    selected = example(ParameterCandidateEvaluation)
    selected.parameters.useHarmony = True
    selected.artifacts["pca"] = ArtifactRecord.from_ref(_ref("reduction"))
    pca = d._artifact_ref(selected, "pca")
    calls = []

    def operation(name: str, kind: str) -> Any:
        def execute(*args: Any, **kwargs: Any) -> ArtifactRef:
            calls.append((name, args, kwargs))
            return _ref(kind)

        return execute

    store = SimpleNamespace(
        build_ann_index=operation("ann", "ann_index"),
        query_neighbors=operation("neighbors", "neighbors"),
        build_connectivity_map=operation("graph", "connectivity_map"),
        run_leiden_clustering=operation("clusters", "cluster_labels"),
    )
    clusters, graph = d.resolve_native_doublet_inputs(store, selected, [])
    assert [row[0] for row in calls] == ["ann", "neighbors", "graph", "clusters"]
    assert calls[0][1] == (pca,)
    assert calls[1][2]["k"] == selected.parameters.neighborsK
    assert calls[-1][2]["resolution"] == selected.parameters.leidenResolution
    assert clusters == _ref("cluster_labels") and graph == _ref("connectivity_map")


def _capture_scoring_store(
    monkeypatch: pytest.MonkeyPatch, labels: np.ndarray, *, capture_known: bool = True
) -> tuple[Any, Any, dict[ArtifactRef, np.ndarray], list[Any]]:
    parent = _ref("cell_selection")
    selections = {parent: np.arange(len(labels))}
    payloads: dict[ArtifactRef, dict[str, np.ndarray]] = {}
    calls = []
    current = [parent]
    serial = [10]

    def reference(kind: str) -> ArtifactRef:
        serial[0] += 1
        return _ref(kind, serial[0])

    def record(name: str, kind: str) -> Any:
        def run(*args: Any, **kwargs: Any) -> ArtifactRef:
            calls.append((name, args, kwargs))
            if name == "normalize":
                current[0] = args[0]
            return reference(kind)

        return run

    def filter_cells(
        columns: Any, lower: Any, upper: Any, **kwargs: Any
    ) -> ArtifactRef:
        assert columns == ["capture"] and lower == upper
        assert kwargs["cell_selection"] == parent
        mask = labels == lower[0]
        selected = reference("cell_selection")
        selections[selected] = np.flatnonzero(mask)
        payloads[selected] = {"values": mask}
        calls.append(("filter", (), {"capture": lower[0]}))
        return selected

    def score(*args: Any, **kwargs: Any) -> ArtifactRef:
        calls.append(("doublet", args, kwargs))
        result = reference("quality_metric")
        payloads[result] = {"values": np.linspace(0, 1, len(selections[current[0]]))}
        return result

    store = SimpleNamespace(
        zw=None,
        cells=SimpleNamespace(
            N=len(labels), columns=["capture"] if capture_known else []
        ),
        get_assay=lambda _: SimpleNamespace(
            feats=SimpleNamespace(
                fetch_all=lambda _: np.asarray(["g1", "g2", "g3", "g4", "g5"])
            )
        ),
        load_artifact=lambda ref: payloads[ref],
        filter_cells=filter_cells,
        run_normalization=record("normalize", "normalized"),
        run_pca=record("pca", "reduction"),
        build_ann_index=record("ann", "ann_index"),
        query_neighbors=record("neighbors", "neighbors"),
        build_connectivity_map=record("graph", "connectivity_map"),
        run_leiden_clustering=record("cluster", "cluster_labels"),
        run_doublet_detection=score,
    )
    monkeypatch.setattr(d, "as_zarr_array", lambda value, **_: value)
    monkeypatch.setattr(
        d, "read_stored_selection_indices", lambda _root, ref, **_: selections[ref]
    )
    monkeypatch.setattr(
        d, "read_metadata_rows_chunkwise", lambda _cells, _column, rows: labels[rows]
    )
    monkeypatch.setattr(d, "read_feature_selection_indices", lambda *_: np.arange(5))
    monkeypatch.setattr(
        d,
        "resolve_native_doublet_inputs",
        lambda *_: (_ref("cluster_labels"), _ref("connectivity_map")),
    )
    selected = example(ParameterCandidateEvaluation)
    from scarf.agent.tools import artifact_reference

    selected.cellSelection = artifact_reference(parent)
    selected.parameters.dimensions = 21
    selected.parameters.neighborsK = 11
    return store, selected, selections, calls


def test_capture_doublet_routing_caps_rank_and_preserves_unscored_small_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.asarray(["tiny"] * 2 + ["small"] * 3 + ["large"] * 5)
    store, selected, selections, calls = _capture_scoring_store(monkeypatch, labels)
    evidence = d.score_advisory_doublets(
        store,
        selected,
        [selected],
        assay="RNA",
        feature_selection=_ref("feature_selection"),
        capture_column="capture",
    )
    assert evidence.capture_values == ("large", "small")
    assert evidence.capture_coverage == 0.8
    assert any("tiny" in note and "only 2" in note for note in evidence.limitations)
    assert [row[2]["dims"] for row in calls if row[0] == "pca"] == [4, 2]
    assert [row[2]["k"] for row in calls if row[0] == "neighbors"] == [4, 2]
    assert sum(row[0] == "doublet" for row in calls) == 2
    np.testing.assert_array_equal(selections[_ref("cell_selection")], np.arange(10))
    for ref, capture in zip(
        evidence.cell_selections, evidence.capture_values, strict=True
    ):
        assert np.all(labels[selections[ref]] == capture)


@pytest.mark.parametrize("capture_known", [False, True])
def test_single_or_unknown_capture_uses_one_native_reference_with_visible_scope(
    monkeypatch: pytest.MonkeyPatch, capture_known: bool
) -> None:
    store, selected, _, calls = _capture_scoring_store(
        monkeypatch, np.asarray(["one"] * 5), capture_known=capture_known
    )
    evidence = d.score_advisory_doublets(
        store,
        selected,
        [selected],
        assay="RNA",
        feature_selection=_ref("feature_selection"),
        capture_column="capture" if capture_known else None,
    )
    assert len(evidence.scores) == 1 and evidence.capture_coverage == 1
    assert [row[0] for row in calls] == ["doublet"]
    assert evidence.capture_values == ("one" if capture_known else "allSelectedCells",)
    assert bool(evidence.limitations) == (not capture_known)


@pytest.mark.parametrize(
    "damage", ["missingSelection", "tooManyCaptures", "allCapturesTooSmall"]
)
def test_doublets_fail_visibly_when_scoring_scope_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    labels = (
        np.arange(513).astype(str)
        if damage == "tooManyCaptures"
        else np.asarray(["a", "a", "b", "b"])
    )
    store, selected, _, calls = _capture_scoring_store(monkeypatch, labels)
    if damage == "missingSelection":
        selected.cellSelection = None
    with pytest.raises(ValueError):
        d.score_advisory_doublets(
            store,
            selected,
            [selected],
            assay="RNA",
            feature_selection=_ref("feature_selection"),
            capture_column="capture",
        )
    assert not any(row[0] == "doublet" for row in calls)


@pytest.mark.parametrize(
    "damage", ["inventory", "emptyParent", "emptyScores", "unalignedScores"]
)
def test_doublet_summary_requires_aligned_nonempty_capture_artifacts(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    store, evidence, selections = _score_evidence(monkeypatch)
    scores = evidence.scores
    if damage == "inventory":
        scores = ()
    elif damage == "emptyParent":
        selections[_ref("cell_selection")] = np.asarray([], dtype=int)
    elif damage == "emptyScores":
        store.load_artifact = lambda _: {"values": np.ones(0)}
    else:
        store.load_artifact = lambda _: {"values": np.ones(3)}
    with pytest.raises(ValueError):
        d._build_advisory_doublet_scores(
            store,
            scores=scores,
            cell_selections=evidence.cell_selections,
            native_graph=evidence.native_graph,
            native_clusters=evidence.native_clusters,
            parent_selection=_ref("cell_selection"),
            capture_values=("one",),
            capture_column="capture",
            limitations=(),
        )


def test_large_doublet_summary_identifies_sampled_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, evidence, selections = _score_evidence(monkeypatch)
    selections[_ref("cell_selection")] = np.arange(70_000)
    selections[evidence.cell_selections[0]] = np.arange(70_000)
    values = _Blocks(np.linspace(0, 1, 70_000))
    store.load_artifact = lambda _: {"values": values}
    result = d._build_advisory_doublet_scores(
        store,
        scores=evidence.scores,
        cell_selections=evidence.cell_selections,
        native_graph=evidence.native_graph,
        native_clusters=evidence.native_clusters,
        parent_selection=_ref("cell_selection"),
        capture_values=("one",),
        capture_column="capture",
        limitations=(),
    )
    assert result.capture_coverage == 1
    assert result.score_summaries[0]["sampleSize"] < 70_000
    assert any("deterministic bounded samples" in note for note in result.limitations)


def test_metadata_diagnostics_preserve_missing_masks_and_reject_misalignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        d, "read_stored_selection_indices", lambda *_args, **_kwargs: np.asarray([1, 3])
    )
    monkeypatch.setattr(
        d, "read_metadata_rows_chunkwise", lambda *_: np.asarray([20.0, 40])
    )
    monkeypatch.setattr(
        d, "read_metadata_missing_rows_chunkwise", lambda *_: np.asarray([False, True])
    )
    store = SimpleNamespace(zw=None, cells=None)
    values = d._aligned_metadata_values(store, _ref("cell_selection"), "age")
    assert values.tolist() == [20.0, None]
    monkeypatch.setattr(
        d, "read_metadata_rows_chunkwise", lambda *_: np.asarray([20.0])
    )
    for read in (d._aligned_metadata_values, d._aligned_metadata):
        with pytest.raises(ValueError, match="align"):
            read(store, _ref("cell_selection"), "age")


def test_topology_overlap_requires_same_neighborhood_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d, "as_zarr_array", lambda value, **_: value)
    left = _ref("neighbors")
    store = SimpleNamespace(
        load_artifact=lambda ref: {
            "indices": np.ones((4, 2) if ref == left else (3, 2), dtype=int)
        }
    )
    with pytest.raises(ValueError, match="align"):
        d._neighbor_overlap(store, left, _ref("neighbors", 2))


@pytest.mark.parametrize("damage", ["artifactKind", "scaling", "loadingShape"])
def test_pca_diagnostic_rejects_unmatched_method_or_feature_axis_before_writing(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    evaluation = ParameterCandidateEvaluation(
        artifacts={
            "pca": ArtifactRecord.from_ref(_ref("reduction")),
            "neighbors": ArtifactRecord.from_ref(_ref("neighbors")),
        }
    )
    store = SimpleNamespace(
        inspect_artifact=lambda _: SimpleNamespace(
            parameters={"feat_scaling": damage != "scaling"}
        ),
        load_artifact=lambda _: {"data": np.ones((4, 2)), "loadings": np.ones((4, 2))},
    )
    monkeypatch.setattr(d, "as_zarr_array", lambda value, **_: value)
    with pytest.raises(ValueError):
        d._write_pca_diagnostic(
            store,
            evaluation,
            feature_selection=_ref("feature_selection"),
            selected_indices=np.arange(3),
            family_masks={},
            covariate_columns=[],
            covariate_roles=[],
            adjacent_overlap=None,
            column_artifacts={"counts": _ref("cluster_labels")}
            if damage == "artifactKind"
            else {},
        )


def test_reused_pca_evidence_rejects_a_changed_saved_payload(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zarr
    from scarf.storage.artifacts import fingerprint_stored_arrays

    root = zarr.open_group(str(tmp_path / "diagnostics.zarr"), mode="w")
    pca = root.create_group("pca")
    pca.create_array("data", data=np.ones((4, 2)))
    pca.create_array("loadings", data=np.ones((3, 2)))
    saved = root.create_group("saved")
    payload = {
        "component_variance": np.ones(2),
        "explained_variance_ratio": np.ones(2),
        "top_loading_feature_indices": np.tile(np.arange(3), (2, 1)),
        "top_loading_values": np.ones((2, 3)),
        "family_enrichment": np.ones((0, 2)),
        "covariate_association": np.ones((0, 2)),
        "adjacent_neighbor_overlap": np.ones(1),
    }
    for name, values in payload.items():
        saved.create_array(name, data=values)
    saved.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        saved, d._PCA_DIAGNOSTIC_ARRAYS
    )
    saved["component_variance"][0] = 2
    evaluation = ParameterCandidateEvaluation(
        artifacts={
            "pca": ArtifactRecord.from_ref(_ref("reduction")),
            "neighbors": ArtifactRecord.from_ref(_ref("neighbors")),
        }
    )
    store = SimpleNamespace(
        zw=root,
        cells=None,
        inspect_artifact=lambda _: SimpleNamespace(parameters={"feat_scaling": True}),
        load_artifact=lambda ref: pca if ref.kind == "reduction" else saved,
    )
    monkeypatch.setattr(
        d,
        "plan_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(
            reused=True, ref=_ref("feature_summary")
        ),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        d._write_pca_diagnostic(
            store,
            evaluation,
            feature_selection=_ref("feature_selection"),
            selected_indices=np.arange(3),
            family_masks={},
            covariate_columns=[],
            covariate_roles=[],
            adjacent_overlap=None,
        )


def test_incomplete_candidates_keep_their_failure_without_running_new_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        d,
        "_selected_feature_names",
        lambda *_: (np.arange(3), np.asarray(["a", "b", "c"])),
    )
    failed = ParameterCandidateEvaluation(
        status="failed", error="PCA evidence unavailable"
    )
    store = SimpleNamespace(cells=SimpleNamespace(columns=[]))
    pca = d.augment_pca_evaluations(
        store,
        [failed],
        feature_selection=_ref("feature_selection"),
        nominated_families=[],
        protected_families=[],
        technical_columns=[],
        protected_columns=[],
        qc_columns=[],
    )
    clusters = d.augment_cluster_evaluations(
        store,
        pca,
        marker_assay="RNA",
        marker_features=_ref("feature_selection"),
        independent_unit_columns=[],
        technical_columns=[],
    )
    assert clusters[0].status == "failed"
    assert clusters[0].error == "PCA evidence unavailable"
    assert clusters[0].artifacts == {}


def test_tiny_population_stability_reclusters_the_entire_supported_graph() -> None:
    from scipy.sparse import csr_matrix

    graph = csr_matrix(np.ones((3, 3)) - np.eye(3))
    assert d._subsample_partition_stability(graph, np.ones(3), 0.5) == 1


@pytest.mark.parametrize("damage", ["labelMatrix", "seedSelection"])
def test_cluster_diagnostics_reject_unmatched_partition_evidence_before_markers(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    from scarf.agent.tools import artifact_reference

    monkeypatch.setattr(
        d,
        "_selected_feature_names",
        lambda *_: (np.arange(3), np.asarray(["a", "b", "c"])),
    )
    monkeypatch.setattr(d, "as_zarr_array", lambda value, **_: value)
    selected = ParameterCandidateEvaluation(
        status="done",
        eligible=True,
        cellSelection=artifact_reference(_ref("cell_selection")),
        artifacts={
            "clusters": ArtifactRecord.from_ref(_ref("cluster_labels")),
            "connectivityMap": ArtifactRecord.from_ref(_ref("connectivity_map")),
        },
    )
    initial = np.ones((3, 2)) if damage == "labelMatrix" else np.asarray([1, 1, 2])
    store = SimpleNamespace(
        load_artifact=lambda ref: {
            "values": initial if ref == _ref("cluster_labels") else np.asarray([1, 2])
        },
        run_leiden_clustering=lambda *_args, **_kwargs: _ref("cluster_labels", 2),
    )
    with pytest.raises(ValueError, match="one-dimensional|align"):
        d.augment_cluster_evaluations(
            store,
            [selected],
            marker_assay="RNA",
            marker_features=_ref("feature_selection"),
            independent_unit_columns=[],
            technical_columns=[],
        )


def test_pca_feature_evidence_requires_an_assay_owned_selection() -> None:
    with pytest.raises(ValueError, match="belong to one assay"):
        d._selected_feature_names(
            None, ArtifactRef("datastore", "feature_selection", "a" * 64)
        )
