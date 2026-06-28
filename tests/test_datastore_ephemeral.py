from tests.fixtures_datastore import _has_graph


def _active_cell_count(datastore) -> int:
    return len(datastore.cells.active_index("I"))


def _ensure_graph(datastore):
    if not _has_graph(datastore):
        datastore.auto_filter_cells(show_qc_plots=False)
        datastore.mark_hvgs(top_n=100, show_plot=False)
        datastore.make_graph(feat_key="hvgs")


def _clear_umap_columns(datastore):
    for column in ("RNA_UMAP1", "RNA_UMAP2"):
        if column in datastore.cells.columns:
            datastore.cells.drop(column)


def _clear_projection(datastore, name: str):
    projections = datastore.z["RNA"].get("projections")
    if projections is not None and name in projections:
        del projections[name]


def test_run_umap_recomputes_coordinates(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    _clear_umap_columns(ds)

    ds.run_umap(n_epochs=10, parallel=False)

    umap1 = ds.cells.fetch("RNA_UMAP1")
    umap2 = ds.cells.fetch("RNA_UMAP2")
    assert len(umap1) == _active_cell_count(ds)
    assert len(umap2) == _active_cell_count(ds)


def test_run_mapping_writes_projection(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    _clear_projection(ds, "freshmap")

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="freshmap",
        target_feat_key="hvgs_freshmap",
        save_k=3,
    )

    assert "freshmap" in ds.z["RNA"]["projections"]
    assert ds.z["RNA"]["projections"]["freshmap"]["indices"].shape[
        0
    ] == _active_cell_count(ds)


def test_run_mapping_with_coral_writes_corrected_data(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    _clear_projection(ds, "freshmap_coral")

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="freshmap_coral",
        target_feat_key="hvgs_fresh_coral",
        save_k=3,
        run_coral=True,
    )

    normed_loc = "normed__I__hvgs_fresh_coral"
    assert "data_coral" in ds.RNA.z[normed_loc]
    assert "freshmap_coral" in ds.z["RNA"]["projections"]


def test_run_unified_umap_after_mapping(datastore_ephemeral):
    ds = datastore_ephemeral
    _ensure_graph(ds)
    _clear_projection(ds, "freshmap_unified_src")
    _clear_projection(ds, "unified_UMAP")

    ds.run_mapping(
        target_assay=ds.RNA,
        target_name="freshmap_unified_src",
        target_feat_key="hvgs_unified_src",
        save_k=3,
    )
    ds.run_unified_umap(target_names=["freshmap_unified_src"], n_epochs=10)

    coords = ds.z["RNA"]["projections"]["unified_UMAP"][:]
    assert coords.shape[1] == 2
    assert coords.shape[0] >= _active_cell_count(ds)
