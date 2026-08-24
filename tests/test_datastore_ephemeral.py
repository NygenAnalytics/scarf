import numpy as np
import pytest

from scarf.utils import configure_output
from tests.fixtures_datastore import _has_graph, build_neighbourhood_graph

pytestmark = pytest.mark.slow


def _active_cell_count(datastore) -> int:
    return len(datastore.cells.active_index("I"))


def _ensure_graph(datastore):
    assert _has_graph(datastore), "analyzed datastore fixture has no RNA/I/hvgs graph"


def _build_symphony_reference(
    datastore,
    batch_column: str,
    *,
    k: int = 3,
    nclust: int = 5,
):
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    correction = datastore.run_harmony(
        [batch_column],
        state.reduction,
        harmony_params={"nclust": nclust},
        update_state=False,
    )
    ann_index = datastore.build_ann_index(
        correction,
        update_state=False,
    )
    neighbors = datastore.query_neighbors(
        ann_index,
        coordinates=correction,
        k=k,
        update_state=False,
    )
    return datastore.build_mapping_reference(neighbors)


def _clear_umap_columns(datastore):
    for column in ("RNA_UMAP1", "RNA_UMAP2"):
        if column in datastore.cells.columns:
            datastore.cells.drop(column)


def test_run_umap_recomputes_coordinates(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    _clear_umap_columns(ds)

    ds.run_umap(n_epochs=10, parallel=False)

    umap1 = ds.cells.fetch("RNA_UMAP1")
    umap2 = ds.cells.fetch("RNA_UMAP2")
    assert len(umap1) == _active_cell_count(ds)
    assert len(umap2) == _active_cell_count(ds)


def test_run_umap_uses_explicit_progress_for_verbose_output(
    analyzed_datastore_ephemeral,
    monkeypatch,
):
    import scarf.embeddings.umap as umap_module

    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    observed: list[bool] = []

    def fake_fit_transform(*, graph, verbose, **_kwargs):
        observed.append(verbose)
        return np.zeros((graph.shape[0], 2)), None, None

    monkeypatch.setattr(umap_module, "fit_transform", fake_fit_transform)
    try:
        configure_output(progress=False)
        ds.run_umap(
            n_epochs=10,
            label="progress_off_umap",
            invalidate_cache=True,
        )
        configure_output(progress=True)
        ds.run_umap(
            n_epochs=10,
            label="progress_on_umap",
            invalidate_cache=True,
        )
    finally:
        configure_output(progress=False)

    assert observed == [False, True]


def test_run_leiden_writes_cluster_labels(analyzed_datastore_ephemeral):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)

    ds.run_leiden_clustering(label="ephemeral_leiden")

    groups = ds.cells.fetch("RNA_ephemeral_leiden", key="I")
    assert len(groups) == _active_cell_count(ds)
    assert np.unique(groups).size >= 1


def test_cached_harmony_can_rebuild_incomplete_mapping_reference(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    reference = _build_symphony_reference(ds, "mapping_batch")
    state = ds.get_assay_state("RNA")
    assert state is not None
    first_harmony = state.batch_correction
    first_ref = reference.ref
    assert state.named_results["mapping_reference"] == first_ref
    artifact = ds.zw[ds.inspect_artifact(first_ref).path]
    for name in (
        "reference_distance_quantiles",
        "reference_distance_values",
    ):
        values = artifact[name][:]
        del artifact[name]
        with pytest.raises(ValueError, match="missing required arrays"):
            ds.get_mapping_reference(first_ref)
        artifact.create_array(name, data=values)
    del artifact["sigma"]
    with pytest.raises(ValueError, match="missing required arrays"):
        ds.get_mapping_reference(first_ref)

    assert state.neighbors is not None
    rebuilt = ds.build_mapping_reference(state.neighbors)
    loaded = ds.get_mapping_reference(rebuilt.ref)

    assert rebuilt.method == "symphony"
    assert loaded.ref == rebuilt.ref
    assert ds.inspect_artifact(rebuilt.ref).complete
    rebuilt_state = ds.get_assay_state("RNA")
    assert rebuilt_state is not None
    assert rebuilt_state.batch_correction == first_harmony
    assert rebuilt_state.named_results["mapping_reference"] == rebuilt.ref
    assert rebuilt.ref != first_ref


def test_mapping_reference_rebuilds_after_missing_artifact(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    reference = _build_symphony_reference(ds, "mapping_batch")
    first_state = ds.get_assay_state("RNA")
    assert first_state is not None
    first_ref = reference.ref
    assert first_state.named_results["mapping_reference"] == first_ref
    del ds.zw[ds.inspect_artifact(first_ref).path]

    assert first_state.neighbors is not None
    rebuilt = ds.build_mapping_reference(first_state.neighbors)
    loaded = ds.get_mapping_reference(rebuilt.ref)

    assert rebuilt.ref != first_ref
    assert loaded.ref == rebuilt.ref
    assert loaded.method == "symphony"
    assert ds.inspect_artifact(rebuilt.ref).complete
    rebuilt_state = ds.get_assay_state("RNA")
    assert rebuilt_state is not None
    assert rebuilt_state.named_results["mapping_reference"] == rebuilt.ref


def test_mapping_reference_tracks_changed_batch_values(
    analyzed_datastore_ephemeral,
):
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    batches = np.where(np.arange(ds.cells.N) % 2 == 0, "a", "b")
    ds.cells.insert("mapping_batch", batches, overwrite=True)
    first_reference = _build_symphony_reference(ds, "mapping_batch")
    first_state = ds.get_assay_state("RNA")
    assert first_state is not None
    assert first_state.named_results["mapping_reference"] == first_reference.ref

    changed = np.where(np.arange(ds.cells.N) % 3 == 0, "a", "b")
    ds.cells.insert("mapping_batch", changed, overwrite=True)
    rebuilt = _build_symphony_reference(ds, "mapping_batch")
    second_state = ds.get_assay_state("RNA")

    assert second_state is not None
    assert rebuilt.ref != first_reference.ref
    assert rebuilt.batch_correction != first_reference.batch_correction
    assert second_state.named_results["mapping_reference"] == rebuilt.ref
    assert rebuilt.metadata["batch_columns"] == ["mapping_batch"]
    assert ds.get_mapping_reference(rebuilt.ref).ref == rebuilt.ref


def test_mapping_reference_rejects_unscaled_pca_chain(
    analyzed_datastore_ephemeral,
) -> None:
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    build_neighbourhood_graph(
        ds,
        feat_key="hvgs",
        feat_scaling=False,
        dims=5,
        k=3,
        n_centroids=10,
    )
    state = ds.get_assay_state("RNA")
    assert state is not None
    assert state.reduction is not None
    assert state.neighbors is not None
    with pytest.raises(ValueError, match="feature scaling"):
        ds.build_mapping_reference(state.neighbors)
    parameters = ds.inspect_artifact(state.reduction).parameters
    assert parameters is not None
    assert parameters["feat_scaling"] is False
    assert ds.inspect_artifact(state.reduction).operation == "run_pca"


def test_mapping_reference_identical_call_reuses_complete_chain(
    analyzed_datastore_ephemeral,
) -> None:
    ds = analyzed_datastore_ephemeral
    _ensure_graph(ds)
    state = ds.get_assay_state("RNA")
    assert state is not None
    assert state.neighbors is not None

    first_reference = ds.build_mapping_reference(state.neighbors)
    first_state = ds.get_assay_state("RNA")
    second_reference = ds.build_mapping_reference(state.neighbors)
    second_state = ds.get_assay_state("RNA")
    loaded = ds.get_mapping_reference(second_reference.ref)

    assert first_reference.ref == second_reference.ref
    assert loaded.ref == second_reference.ref
    assert ds.inspect_artifact(second_reference.ref).complete
    assert second_state == first_state
