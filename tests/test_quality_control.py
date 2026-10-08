from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import chisquare

from scarf.quality_control.cell_cycle import assign_cell_cycle_phase
from scarf.quality_control.doublets import (
    sample_cluster_pool,
    simulate_doublet_pairs,
    sum_doublet_pairs,
)
from scarf.quality_control.filtering import gaussian_quantile_bounds
from scarf.graph.feature_projection import (
    graph_cell_selection,
    resolve_native_graph_inputs,
)
from scarf.metadata.artifacts import (
    artifact_values,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_array,
    fingerprint_strings,
)
from scarf.storage.operation_revisions import effective_revision
from scarf.storage.selections import (
    read_stored_selection_mask,
    resolve_generated_selection_artifact,
)


def test_doublet_pair_counts_are_widened_without_overflow():
    counts = csr_matrix(np.array([[200, 0], [200, 9]], dtype=np.uint8))
    summed = sum_doublet_pairs(counts, np.array([0]), np.array([1]))

    assert summed.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(summed.toarray(), [[400, 9]])


@pytest.mark.parametrize(
    ("dtype", "left", "right"),
    [
        ("uint64", 2**63, 2**63),
        ("int64", 2**62, 2**62),
        ("int64", -(2**62) - 1, -(2**62)),
    ],
)
def test_doublet_pair_counts_reject_unrepresentable_integer_sums(dtype, left, right):
    counts = csr_matrix(np.array([[0, left], [1, right]], dtype=dtype))

    with pytest.raises(OverflowError, match="Synthetic doublet counts exceed"):
        sum_doublet_pairs(counts, np.array([0]), np.array([1]))


@pytest.mark.parametrize(
    ("dtype", "values", "expected", "output_dtype"),
    [
        ("uint64", [[2**64 - 2, 0], [1, 2**64 - 1]], [[2**64 - 1] * 2], "uint64"),
        ("int64", [[-(2**63), 2**63 - 1], [0, -(2**63)]], [[-(2**63), -1]], "int64"),
        ("uint64", [[0, 0], [1, 2]], [[1, 2]], "uint64"),
        ("int8", [[100, -100], [100, -100]], [[200, -200]], "int16"),
        ("float32", [[1.5, 0], [2.25, 3.5]], [[3.75, 3.5]], "float64"),
        ("float32", [[2.0**24, 0], [1.0, 0]], [[2.0**24 + 1, 0]], "float64"),
        ("bool", [[True, False], [True, True]], [[2, 1]], "uint8"),
    ],
)
def test_doublet_pair_counts_preserve_representable_sums(
    dtype, values, expected, output_dtype
):
    counts = csr_matrix(np.array(values, dtype=dtype))

    actual = sum_doublet_pairs(counts, np.array([0]), np.array([1]))

    assert actual.dtype == np.dtype(output_dtype)
    np.testing.assert_array_equal(
        actual.toarray(), np.array(expected, dtype=output_dtype)
    )


def test_simulate_doublet_pairs_is_seeded_and_heterotypic():
    clusters = np.array([0, 0, 1, 1])
    left, right = simulate_doublet_pairs(
        clusters,
        n_sim=12,
        heterotypic_fraction=1.0,
        rng=np.random.default_rng(11),
    )

    # The pairs are pinned: changing them changes every doublet score, which
    # needs a revision of run_doublet_detection.
    np.testing.assert_array_equal(left, [0, 0, 3, 1, 2, 2, 2, 0, 1, 0, 1, 3])
    np.testing.assert_array_equal(right, [3, 3, 1, 3, 1, 1, 0, 2, 2, 3, 3, 0])
    assert np.all(clusters[left] != clusters[right])
    repeated = simulate_doublet_pairs(clusters, 12, 1.0, np.random.default_rng(11))
    np.testing.assert_array_equal(repeated[0], left)
    np.testing.assert_array_equal(repeated[1], right)


@pytest.mark.parametrize(("sizes", "fraction"), [((100, 1), 1.0), ((990, 5, 5), 0.8)])
def test_simulate_doublet_pairs_make_every_forced_doublet_heterotypic(sizes, fraction):
    # A dominant cluster made bounded redraws fall far short of the fraction:
    # 0.20 heterotypic pairs for sizes (100, 1) at 1.0.
    clusters = np.repeat(np.arange(len(sizes)), sizes)
    n_sim = 5_000
    left, right = simulate_doublet_pairs(
        clusters, n_sim, fraction, np.random.default_rng(29)
    )

    # The documented draws: first parents, then the forced doublets.
    draws = np.random.default_rng(29)
    np.testing.assert_array_equal(left, draws.integers(0, len(clusters), size=n_sim))
    forced = draws.random(n_sim) < fraction
    heterotypic = clusters[left] != clusters[right]
    assert np.all(heterotypic[forced])
    # Unconstrained doublets may pair two clusters too.
    assert heterotypic.mean() >= forced.mean()
    if fraction == 1.0:
        assert heterotypic.all()


def test_simulate_doublet_pairs_draw_partners_uniformly_from_eligible_cells():
    # Pool positions interleave the clusters, which hold 5, 3, and 2 cells.
    clusters = np.array(["b", "a", "c", "a", "b", "a", "c", "a", "b", "a"])
    left, right = simulate_doublet_pairs(
        clusters, 60_000, 1.0, np.random.default_rng(7)
    )

    assert chisquare(np.bincount(left, minlength=len(clusters))).pvalue > 1e-3
    for cluster in np.unique(clusters):
        partners = np.bincount(
            right[clusters[left] == cluster], minlength=len(clusters)
        )
        assert not partners[clusters == cluster].any()
        # Every cell outside the cluster is an equally likely partner, so
        # partner clusters are drawn in proportion to their sizes.
        assert chisquare(partners[clusters != cluster]).pvalue > 1e-3

    # The partner of a doublet that is not forced is any pool cell.
    left, right = simulate_doublet_pairs(
        clusters, 60_000, 0.5, np.random.default_rng(13)
    )
    draws = np.random.default_rng(13)
    draws.integers(0, len(clusters), size=60_000)
    free = draws.random(60_000) >= 0.5
    partners = np.bincount(right[free], minlength=len(clusters))
    assert chisquare(partners).pvalue > 1e-3


def test_simulate_doublet_pairs_allows_homotypic_when_fraction_is_zero():
    clusters = np.array([0, 0, 1, 1])
    left, right = simulate_doublet_pairs(
        clusters,
        n_sim=40,
        heterotypic_fraction=0.0,
        rng=np.random.default_rng(3),
    )

    # Without a heterotypic quota the pairs are two plain uniform draws.
    expected = np.random.default_rng(3)
    np.testing.assert_array_equal(left, expected.integers(0, 4, size=40))
    np.testing.assert_array_equal(right, expected.integers(0, 4, size=40))
    assert np.any(clusters[left] == clusters[right])
    # One cluster can form only homotypic pairs, which a zero quota allows.
    single = simulate_doublet_pairs(
        np.zeros(4, dtype=int), 40, 0.0, np.random.default_rng(3)
    )
    np.testing.assert_array_equal(single[0], left)
    np.testing.assert_array_equal(single[1], right)
    empty = simulate_doublet_pairs(clusters, 0, 0.8, np.random.default_rng(3))
    assert [part.shape for part in empty] == [(0,), (0,)]


def test_simulate_doublet_pairs_refuse_heterotypic_doublets_of_one_cluster():
    with pytest.raises(ValueError, match="the pool holds one cluster"):
        simulate_doublet_pairs(np.full(4, "T"), 40, 1e-9, np.random.default_rng(3))


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"n_sim": -1}, ValueError, "n_sim must be at least 0"),
        ({"heterotypic_fraction": 1.5}, ValueError, "from 0 to 1"),
        ({"pool_clusters": np.zeros((2, 2))}, ValueError, "one-dimensional"),
        ({"pool_clusters": np.array([])}, ValueError, "non-empty"),
        ({"n_sim": 2.5}, TypeError, "n_sim must be an integer"),
        ({"heterotypic_fraction": True}, TypeError, "must be a real number"),
    ],
)
def test_simulate_doublet_pairs_validate_arguments(arguments, error, message):
    options = {
        "pool_clusters": np.array([0, 1]),
        "n_sim": 4,
        "heterotypic_fraction": 0.5,
        "rng": np.random.default_rng(0),
    }
    with pytest.raises(error, match=message):
        simulate_doublet_pairs(**(options | arguments))


def test_sample_cluster_pool_respects_fraction_and_cap():
    clusters = np.array([0, 0, 0, 0, 1, 1, 2])
    rng = np.random.default_rng(7)

    pool = sample_cluster_pool(
        clusters,
        fraction=0.5,
        max_per_cluster=2,
        rng=rng,
    )

    np.testing.assert_array_equal(pool, np.sort(pool))
    assert set(pool).issubset(set(range(len(clusters))))
    counts = {int(c): int((clusters[pool] == c).sum()) for c in np.unique(clusters)}
    assert counts == {0: 2, 1: 1, 2: 1}

    repeated = sample_cluster_pool(
        clusters,
        fraction=0.5,
        max_per_cluster=2,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(pool, repeated)

    with pytest.raises(ValueError, match="No cells could be sampled"):
        sample_cluster_pool(
            clusters,
            fraction=0.0,
            max_per_cluster=2,
            rng=np.random.default_rng(0),
        )


def test_doublet_real_arguments_reject_booleans_and_non_numbers():
    from scarf.datastore._operations.quality_control import _validated_real

    for value in (True, np.bool_(False), "0.5", None):
        with pytest.raises(TypeError, match="ratio must be a real number"):
            _validated_real(value, "ratio", low=0.0)
    assert _validated_real(np.float32(0.5), "ratio", low=0.0, high=1.0) == 0.5
    with pytest.raises(ValueError, match=r"ratio must be a finite number > 0"):
        _validated_real(0, "ratio", low=0.0, include_low=False)
    with pytest.raises(ValueError, match=r">= 0 and <= 1"):
        _validated_real(float("nan"), "ratio", low=0.0, high=1.0)


def test_assign_cell_cycle_phase_preserves_rule_precedence():
    phases = assign_cell_cycle_phase(
        s_score=np.array([1.0, 0.1, -2.0, 0.0, -1.0]),
        g2m_score=np.array([0.5, 0.2, -1.0, 0.0, 0.5]),
    )

    np.testing.assert_array_equal(phases, ["S", "G2M", "G1", "S", "G2M"])


def test_gaussian_quantile_bounds_uses_median_and_population_deviation():
    bounds = gaussian_quantile_bounds(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        min_p=0.1,
        max_p=0.9,
    )

    np.testing.assert_allclose(
        bounds,
        (1.1876123951263535, 4.8123876048736465),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("value", [0.0, 10.0, -3.0])
def test_gaussian_constant_metrics_have_finite_equal_bounds(value):
    from scarf.quality_control.filtering import _apply_bounds

    values = np.full(5, value)
    assert gaussian_quantile_bounds(values) == (value, value)
    assert not _apply_bounds(values, value, value).any()


@pytest.mark.parametrize("min_p,max_p", [(0, 0.99), (0.01, 1), (0.9, 0.1), (0.5, 0.5)])
def test_gaussian_constant_metrics_reject_invalid_quantiles(min_p, max_p):
    with pytest.raises(ValueError, match="0 < min_p < max_p < 1"):
        gaussian_quantile_bounds(np.zeros(5), min_p, max_p)


def _snapshot_store(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


def _doublet_clusters(datastore, graph: ArtifactRef) -> ArtifactRef:
    return datastore.run_leiden_clustering(graph, resolution=0.5)


def _fixture_graph(datastore) -> ArtifactRef:
    graphs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(graphs) == 1
    return graphs[0]


def _fixture_quality_metric(
    datastore,
    cell_selection: ArtifactRef,
    values: np.ndarray,
) -> ArtifactRef:
    metric_values = np.asarray(values, dtype=np.float64)
    planned = plan_cell_data_artifact(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind="quality_metric",
        operation="fixture_quality_metric",
        parameters={},
        inputs={"values_fingerprint": fingerprint_array(metric_values)},
        execution_options={},
        cell_selection=cell_selection,
        arrays={"values": (metric_values.shape, "f")},
    )
    write_cell_data_artifact(
        datastore.zw,
        planned,
        {"values": metric_values},
    )
    return planned.ref


def _fixture_categorical_values(
    datastore,
    cell_selection: ArtifactRef,
    values: np.ndarray,
) -> ArtifactRef:
    labels = np.asarray(values)
    fingerprint = (
        fingerprint_strings(labels.astype(str))
        if labels.dtype.kind in {"O", "S", "U"}
        else fingerprint_array(labels)
    )
    planned = plan_cell_data_artifact(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind="hto_identity",
        operation="fixture_categorical_values",
        parameters={},
        inputs={"values_fingerprint": fingerprint},
        execution_options={},
        cell_selection=cell_selection,
        arrays={"values": (labels.shape, None)},
    )
    write_cell_data_artifact(datastore.zw, planned, {"values": labels})
    return planned.ref


def _selection_values(datastore, ref: ArtifactRef) -> np.ndarray:
    return read_stored_selection_mask(
        datastore.zw,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def test_select_cells_thresholds_artifact_values_without_live_metadata_writes(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    source = store.snapshot_cell_selection()
    values = np.linspace(-1.0, 1.0, store.cells.N)
    metric = _fixture_quality_metric(store, source, values)

    drifted = np.ones(store.cells.N, dtype=bool)
    drifted[::3] = False
    store.cells.insert("I", drifted, overwrite=True, force=True)
    metadata_before = _snapshot_store(str(Path(store.zarr_loc) / "cellData"))

    selected = store.select_cells(
        metric,
        low=-0.25,
        high=0.75,
        keep_bounds=True,
    )

    np.testing.assert_array_equal(
        _selection_values(store, selected),
        (values >= -0.25) & (values <= 0.75),
    )
    status = store.inspect_artifact(selected)
    assert status.operation == "select_cells"
    assert ArtifactRef.from_dict(status.inputs["values"]) == metric
    assert ArtifactRef.from_dict(status.inputs["source_cell_selection"]) == source
    assert ArtifactRef.from_dict(status.inputs["prior_cell_selection"]) == source
    assert (
        store.select_cells(
            metric,
            low=-0.25,
            high=0.75,
            keep_bounds=True,
        )
        == selected
    )
    assert _snapshot_store(str(Path(store.zarr_loc) / "cellData")) == metadata_before


def test_select_cells_composes_only_with_a_source_subset(datastore_ephemeral) -> None:
    store = datastore_ephemeral
    source_mask = np.arange(store.cells.N) % 2 == 0
    store.cells.insert("I", source_mask, overwrite=True, force=True)
    source = store.snapshot_cell_selection()
    metric = _fixture_quality_metric(
        store,
        source,
        np.arange(int(source_mask.sum()), dtype=np.float64),
    )

    prior_mask = source_mask & (np.arange(store.cells.N) % 4 == 0)
    store.cells.insert("I", prior_mask, overwrite=True, force=True)
    prior = store.snapshot_cell_selection()
    selected = store.select_cells(metric, low=None, high=None, cell_selection=prior)
    np.testing.assert_array_equal(_selection_values(store, selected), prior_mask)

    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    superset = store.snapshot_cell_selection()
    with pytest.raises(ValueError, match="must be a subset"):
        store.select_cells(metric, cell_selection=superset)

    with pytest.raises(ValueError, match="low cannot exceed high"):
        store.select_cells(metric, low=2.0, high=1.0)
    with pytest.raises(TypeError, match="low must be a finite number"):
        store.select_cells(metric, low=True)


def test_select_cells_includes_categorical_artifact_values(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    source = store.snapshot_cell_selection()
    labels = np.resize(
        np.asarray(["tag-b", "Negative", "tag-a", "Doublet"]),
        store.cells.N,
    )
    identities = _fixture_categorical_values(store, source, labels)

    selected = store.select_cells(identities, include=["tag-b", "tag-a"])

    np.testing.assert_array_equal(
        _selection_values(store, selected),
        np.isin(labels, ["tag-a", "tag-b"]),
    )
    assert store.inspect_artifact(selected).parameters["include"] == [
        "tag-a",
        "tag-b",
    ]
    assert (
        store.select_cells(
            identities,
            include=["tag-a", "tag-b"],
        )
        == selected
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        store.select_cells(identities, include=["tag-a"], low=0)
    with pytest.raises(TypeError, match="numeric unless include"):
        store.select_cells(identities)


def test_select_cells_reads_canonical_label_arrays_and_rejects_empty_results(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    source = store.snapshot_cell_selection()
    n_cells = store.cells.N
    phases = np.resize(np.asarray(["G1", "S", "G2M"]), n_cells)
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind="cell_cycle",
        operation="fixture_cell_cycle",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=source,
        arrays={
            "s_score": ((n_cells,), "f"),
            "g2m_score": ((n_cells,), "f"),
            "phase": ((n_cells,), None),
        },
    )
    write_cell_data_artifact(
        store.zw,
        planned,
        {"s_score": np.zeros(n_cells), "g2m_score": np.zeros(n_cells), "phase": phases},
    )
    metric = _fixture_quality_metric(store, source, np.zeros(n_cells))

    selected = store.select_cells(planned.ref, include=["G1"])

    np.testing.assert_array_equal(_selection_values(store, selected), phases == "G1")
    with pytest.raises(ValueError, match="retained no cells"):
        store.select_cells(planned.ref, include=["M"])
    with pytest.raises(ValueError, match="retained no cells"):
        store.select_cells(metric, low=1.0)


def test_select_cells_rejects_lossy_categorical_include_values(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    store.cells.insert(
        "I",
        np.ones(store.cells.N, dtype=bool),
        overwrite=True,
        force=True,
    )
    source = store.snapshot_cell_selection()
    integer_labels = np.resize(np.asarray([1, 2], dtype=np.int16), store.cells.N)
    values = _fixture_categorical_values(store, source, integer_labels)

    selected = store.select_cells(values, include=[1])

    np.testing.assert_array_equal(
        _selection_values(store, selected),
        integer_labels == 1,
    )
    with pytest.raises(TypeError, match="integers for an integer artifact"):
        store.select_cells(values, include=[True])
    with pytest.raises(TypeError, match="integers for an integer artifact"):
        store.select_cells(values, include=[1, "1"])


def test_selection_equality_uses_validated_immutable_fingerprints(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    values = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    row_ids = np.asarray(store.cells.fetch_all("ids"))
    first = store.snapshot_cell_selection()
    same_values = resolve_generated_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=values,
        row_ids=row_ids,
        operation="fixture_equal_selection",
        parameters={},
        inputs={},
        source_column="artifact",
        invalidate_cache=True,
    )[0]
    changed_values = values.copy()
    changed_values[0] = ~changed_values[0]
    different = resolve_generated_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=changed_values,
        row_ids=row_ids,
        operation="fixture_changed_selection",
        parameters={},
        inputs={},
        source_column="artifact",
    )[0]

    assert first != same_values
    assert store._selection_artifacts_match(first, same_values)
    assert not store._selection_artifacts_match(first, different)

    artifact_group(store.zw, same_values)["values"][0] = ~values[0]
    assert not store._selection_artifacts_match(first, same_values)


def test_doublet_scores_preserve_artifacts_without_materializing_queries(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    selected_connectivity = _fixture_graph(datastore)
    clusters = _doublet_clusters(datastore, selected_connectivity)
    metadata_before = _snapshot_store(str(Path(datastore.zarr_loc) / "cellData"))
    reference_projections = set(
        datastore.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
    )
    from scarf.quality_control import doublets

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Doublet scoring must not create a query store or projection"
        )

    monkeypatch.setattr(type(datastore), "run_mapping", forbidden)
    diffusion_before = set(
        datastore.list_artifacts(kind="diffusion_operator", from_assay="RNA")
    )
    raw_scores: list[np.ndarray] = []
    score_doublets = doublets.score_synthetic_doublets

    def recording_scores(*args, **kwargs):
        raw_scores.append(np.array(score_doublets(*args, **kwargs)))
        return raw_scores[-1].copy()

    monkeypatch.setattr(doublets, "score_synthetic_doublets", recording_scores)

    score_ref = datastore.run_doublet_detection(
        clusters,
        selected_connectivity,
        cluster_sample_fraction=0.01,
        max_cells_per_cluster=2,
        simulation_ratio=0.01,
        save_k=3,
        smoothing_t=1,
        random_seed=19,
    )

    # The record holds the scoring parameters only; it carries no
    # arithmetic salt. Revision 2, direct heterotypic sampling, changed the
    # scores without changing the parameters.
    score_status = datastore.inspect_artifact(score_ref)
    assert set(score_status.parameters) == {
        "cluster_sample_fraction",
        "max_cells_per_cluster",
        "simulation_ratio",
        "heterotypic_fraction",
        "save_k",
        "smoothing_t",
        "normalize_scores",
        "random_seed",
    }
    assert score_status.revision == 2 and score_status.is_current
    assert (
        set(
            datastore.list_artifacts(
                kind="projection",
                from_assay="RNA",
            )
        )
        == reference_projections
    )
    scores = artifact_values(
        artifact_group(datastore.zw, score_ref),
        "values",
    )
    assert scores.ndim == 1
    assert (
        scores.shape
        == artifact_values(
            artifact_group(datastore.zw, clusters),
            "values",
        ).shape
    )
    # Stored scores are one row-normalized diffusion step of the raw mapping
    # scores over the symmetrized graph, min-max scaled to [0, 1].
    (raw,) = raw_scores
    assert raw.shape == scores.shape
    assert np.all(raw >= 0) and np.any(raw > 0)
    graph = datastore.load_graph(
        selected_connectivity, symmetric=True, upper_only=False
    )
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    smoothed = graph.multiply(inverse[:, None]).tocsr() @ raw
    expected = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    np.testing.assert_allclose(scores, expected, rtol=1e-9, atol=1e-12)
    assert scores.min() == 0.0 and scores.max() == 1.0
    assert "RNA_doublet_score__raw" not in datastore.cells.columns
    assert (
        _snapshot_store(str(Path(datastore.zarr_loc) / "cellData")) == metadata_before
    )

    neighbors = ArtifactRef.from_dict(score_status.inputs["neighbors"])
    reference_refs = datastore.list_artifacts(
        kind="mapping_reference",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    references = [datastore.get_mapping_reference(ref) for ref in reference_refs]
    matching = [
        reference for reference in references if reference.neighbors == neighbors
    ]
    assert matching
    reference = matching[-1]
    assert reference.method == "pca"
    assert reference.symphony_state is None
    assert (
        set(datastore.list_artifacts(kind="diffusion_operator", from_assay="RNA"))
        == diffusion_before
    )
    monkeypatch.setattr(doublets, "score_synthetic_doublets", forbidden)

    assert (
        datastore.run_doublet_detection(
            clusters,
            selected_connectivity,
            cluster_sample_fraction=0.01,
            max_cells_per_cluster=2,
            simulation_ratio=0.01,
            save_k=3,
            smoothing_t=1,
            random_seed=19,
        )
        == score_ref
    )
    assert (
        _snapshot_store(str(Path(datastore.zarr_loc) / "cellData")) == metadata_before
    )


@pytest.mark.parametrize(
    ("failure", "error", "message"),
    [
        ("query", RuntimeError, "neighbor query failed"),
        ("features", ValueError, "reference order"),
        ("scores", RuntimeError, "mapping scores do not match"),
        ("graph", ValueError, "graph does not match"),
    ],
)
def test_doublet_failure_leaves_no_complete_score(
    analyzed_datastore_ephemeral,
    monkeypatch,
    failure,
    error,
    message,
) -> None:
    datastore = analyzed_datastore_ephemeral
    graph = _fixture_graph(datastore)
    clusters = _doublet_clusters(datastore, graph)
    from scarf.neighbors.stages import NeighborQueryStage

    metadata_before = _snapshot_store(str(Path(datastore.zarr_loc) / "cellData"))
    scores_before = set(
        datastore.list_artifacts(
            kind="doublet_score", from_assay="RNA", complete_only=True
        )
    )

    def failed_query(*args, **kwargs):
        raise RuntimeError("neighbor query failed")

    if failure == "query":
        monkeypatch.setattr(NeighborQueryStage, "query", failed_query)
    else:
        from scarf.quality_control import doublets

        owner, name = {
            "features": (datastore, "get_mapping_reference"),
            "scores": (doublets, "score_synthetic_doublets"),
            "graph": (datastore, "_load_graph_artifact"),
        }[failure]
        original = getattr(owner, name)

        def corrupt_result(*args, **kwargs):
            result = original(*args, **kwargs)
            if failure == "features":
                return replace(result, feature_ids=result.feature_ids[::-1])
            if failure == "scores":
                return result[:-1]
            return result[:-1, :-1]

        monkeypatch.setattr(owner, name, corrupt_result)

    with pytest.raises(error, match=message):
        datastore.run_doublet_detection(
            clusters,
            graph,
            cluster_sample_fraction=0.01,
            max_cells_per_cluster=2,
            simulation_ratio=0.01,
            save_k=3,
            smoothing_t=1,
            random_seed=23,
        )

    assert (
        set(
            datastore.list_artifacts(
                kind="doublet_score", from_assay="RNA", complete_only=True
            )
        )
        == scores_before
    )
    assert not datastore.list_artifacts(
        kind="projection",
        from_assay="RNA",
    )
    assert "RNA_doublet_score__raw" not in datastore.cells.columns
    assert (
        _snapshot_store(str(Path(datastore.zarr_loc) / "cellData")) == metadata_before
    )


def test_doublet_detection_rejects_symphony_connectivity_chain(
    analyzed_datastore_ephemeral,
) -> None:
    datastore = analyzed_datastore_ephemeral
    native_graph = _fixture_graph(datastore)
    clusters = _doublet_clusters(datastore, native_graph)
    reduction = resolve_native_graph_inputs(datastore.zw, native_graph).coordinates
    datastore.cells.insert(
        "doublet_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    correction = datastore.run_harmony(
        reduction,
        ["doublet_batch"],
        harmony_params={"nclust": 5},
    )
    ann_index = datastore.build_ann_index(
        correction,
    )
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=3,
    )
    connectivity = datastore.build_connectivity_map(neighbors)
    references_before = set(
        datastore.list_artifacts(
            kind="mapping_reference",
            from_assay="RNA",
        )
    )

    with pytest.raises(
        ValueError,
        match="uncorrected PCA graph",
    ):
        datastore.run_doublet_detection(
            clusters,
            connectivity,
            simulation_ratio=0.01,
        )

    assert datastore.inspect_artifact(connectivity).complete
    assert (
        set(
            datastore.list_artifacts(
                kind="mapping_reference",
                from_assay="RNA",
            )
        )
        == references_before
    )


def test_doublet_detection_needs_two_clusters_for_heterotypic_doublets(
    analyzed_datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = analyzed_datastore_ephemeral
    graph = _fixture_graph(datastore)
    datastore.cells.insert(
        "one_cluster", np.full(datastore.cells.N, "all"), overwrite=True
    )
    clusters = datastore.snapshot_cluster_labels(
        "one_cluster", cell_selection=graph_cell_selection(datastore.zw, graph)
    )
    from scarf.datastore._operations import quality_control as operations

    def refuse_planning(*_args, **_kwargs):
        raise AssertionError("a one-cluster request must fail before planning")

    with monkeypatch.context() as patched:
        patched.setattr(operations, "plan_cell_data_artifact", refuse_planning)
        with pytest.raises(
            ValueError,
            match=(
                r"^Doublet detection with heterotypic_fraction=0\.8 pairs cells "
                r"from two different clusters, but every selected cell has the "
                r"same cluster label\. Pass a clustering with two or more "
                r"clusters, or heterotypic_fraction=0 to simulate doublets from "
                r"any two sampled cells\.$"
            ),
        ):
            datastore.run_doublet_detection(clusters, graph)

    # Without forced heterotypic doublets one cluster is enough.
    ref = datastore.run_doublet_detection(
        clusters,
        graph,
        cluster_sample_fraction=0.01,
        max_cells_per_cluster=2,
        simulation_ratio=0.01,
        heterotypic_fraction=0.0,
        save_k=3,
        smoothing_t=1,
    )
    status = datastore.inspect_artifact(ref)
    # Zero-fraction pairs are the draws of earlier releases, so the scores
    # keep revision 1 and their identity.
    assert status.complete and status.revision == 1


@pytest.mark.parametrize(
    ("parameters", "revision"),
    [
        ({"heterotypic_fraction": 0.0}, 1),
        ({"heterotypic_fraction": 0}, 1),
        ({"heterotypic_fraction": 0.8}, 2),
        ({"heterotypic_fraction": False}, 2),
        ({}, 2),
    ],
)
def test_doublet_revision_spares_scores_without_forced_heterotypic_pairs(
    parameters, revision
):
    assert (
        effective_revision("run_doublet_detection", "doublet_score", parameters, {})
        == revision
    )
