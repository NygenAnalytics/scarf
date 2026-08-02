import numpy as np
import pandas as pd
import pytest

import scarf.plotting as splt
from scarf.datastore.datastore import DataStore
from scarf.metadata.artifacts import (
    categorical_display,
    validate_display_metadata,
)
from scarf.plotting._contracts import CategoricalScale, ColorScale
from scarf.plotting.embedding import _continuous_limits
from scarf.storage.artifacts import ArtifactRef, artifact_path
from scarf.storage.selections import resolve_selection_artifact
from tests.fixtures_datastore import build_neighbourhood_graph


def _ensure_graph(datastore) -> None:
    datastore.auto_filter_cells(show_qc_plots=False)
    if "I__metadata_hvgs" not in datastore.RNA.feats.columns:
        datastore.mark_hvgs(
            from_assay="RNA",
            cell_key="I",
            top_n=100,
            hvg_key_name="metadata_hvgs",
            show_plot=False,
            min_cells=int(0.01 * datastore.cells.N),
            max_cells=np.inf,
            blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
        )
    build_neighbourhood_graph(
        datastore,
        from_assay="RNA",
        feat_key="metadata_hvgs",
        dims=5,
        k=3,
        n_centroids=10,
        local_cache=False,
    )


def _column_ref(datastore, column: str) -> ArtifactRef:
    raw_ref = datastore.zw["cellData"][column].attrs["source_artifact"]
    return ArtifactRef.from_dict(raw_ref)


def test_umap_matrix_and_leiden_columns_link_authoritative_artifacts(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    hvg_display = dict(
        datastore.RNA.z["featureData"]["I__metadata_hvgs"].attrs["display"]
    )
    assert hvg_display["kind"] == "categorical"
    returned_umap = datastore.run_umap(n_epochs=10, label="metadata_umap")
    returned_leiden = datastore.run_leiden_clustering(label="metadata_leiden")

    umap1 = datastore.zw["cellData"]["RNA_metadata_umap1"]
    umap2 = datastore.zw["cellData"]["RNA_metadata_umap2"]
    umap_ref = ArtifactRef.from_dict(umap1.attrs["source_artifact"])
    leiden_ref = _column_ref(datastore, "RNA_metadata_leiden")

    assert returned_umap == umap_ref
    assert returned_leiden == leiden_ref
    assert ArtifactRef.from_dict(umap2.attrs["source_artifact"]) == umap_ref
    assert umap1.attrs["value_index"] == 0
    assert umap2.attrs["value_index"] == 1
    first_input = datastore._resolve_cell_data_provenance_input(
        "RNA_metadata_umap1",
        cell_key="I",
    )
    second_input = datastore._resolve_cell_data_provenance_input(
        "RNA_metadata_umap2",
        cell_key="I",
    )
    assert first_input["artifact"] == second_input["artifact"]
    assert first_input["value_index"] == 0
    assert second_input["value_index"] == 1
    assert first_input != second_input
    display = dict(umap1.attrs["display"])
    values = datastore.cells.fetch("RNA_metadata_umap1", key="I")
    assert display["kind"] == "continuous"
    assert display["minimum"] == float(values.min())
    assert display["maximum"] == float(values.max())
    assert datastore.load_artifact(umap_ref)["values"].shape[1] == 2
    assert leiden_ref.kind == "cluster_labels"
    leiden_display = dict(
        datastore.zw["cellData"]["RNA_metadata_leiden"].attrs["display"]
    )
    assert leiden_display["kind"] == "categorical"
    assert all(
        set(category) == {"value", "label", "color"}
        for category in leiden_display["categories"]
    )
    result = splt.embedding(
        datastore,
        layout_key="RNA_metadata_umap",
        color_by="RNA_metadata_umap1",
        show=False,
    )
    continuous = next(scale for scale in result.scales if isinstance(scale, ColorScale))
    assert continuous.vmin == display["minimum"]
    assert continuous.vmax == display["maximum"]
    result.close()
    multi = splt.embedding(
        datastore,
        layout_key="RNA_metadata_umap",
        color_by=[
            "RNA_metadata_umap1",
            "RNA_metadata_leiden",
        ],
        show=False,
    )
    assert any(isinstance(scale, ColorScale) for scale in multi.scales)
    assert any(isinstance(scale, CategoricalScale) for scale in multi.scales)
    multi.close()
    batches = np.asarray(
        ["a" if index % 2 else "b" for index in range(datastore.cells.N)]
    )
    datastore.cells.insert("display_batch", batches, overwrite=True)
    datastore.zw["cellData"]["display_batch"].attrs["display"] = categorical_display(
        datastore.cells.fetch("display_batch", key="I")
    )
    leiden_values = datastore.cells.fetch(
        "RNA_metadata_leiden",
        key="I",
    )
    keep = list(np.unique(leiden_values)[:2])
    grouped = splt.embedding(
        datastore,
        layout_key="RNA_metadata_umap",
        color_by=["RNA_metadata_leiden", "display_batch"],
        groups=keep,
        show=False,
    )
    categorical_scales = [
        scale for scale in grouped.scales if isinstance(scale, CategoricalScale)
    ]
    assert list(categorical_scales[0].order) == keep
    assert set(categorical_scales[1].order) == {"a", "b"}
    grouped.close()


def test_membership_and_smart_labels_are_artifact_backed_and_lisi_is_read_only(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    datastore.run_leiden_clustering(label="independent_leiden")
    cluster_key = "RNA_independent_leiden"

    datastore.calc_membership_strength(
        "RNA",
        "I",
        "metadata_hvgs",
        cluster_key,
    )
    membership_key = "RNA_I_cluster_membership_strength"
    membership_ref = _column_ref(datastore, membership_key)
    assert membership_ref.kind == "membership_strength"
    graph_loc = datastore.get_latest_graph_loc("RNA", "I", "metadata_hvgs")
    n_cells, k = datastore._get_graph_ncells_k(graph_loc)
    edges = np.asarray(datastore.zw[graph_loc]["edges"][:]).reshape(n_cells, k, 2)
    clusters = np.asarray(datastore.cells.fetch(cluster_key, key="I"))
    expected_membership = np.asarray(
        [
            pd.Series(row).value_counts(dropna=False).iloc[0] / k
            for row in clusters[edges[:, :, 1]]
        ]
    ).round(3)
    np.testing.assert_array_equal(
        datastore.cells.fetch(membership_key, key="I"),
        expected_membership,
    )

    columns_before = set(datastore.cells.columns)
    artifacts_before = set(datastore.list_artifacts())
    lisi = datastore.metric_lisi([cluster_key], perplexity=1)
    repeated_lisi = datastore.metric_lisi([cluster_key], perplexity=1)
    np.testing.assert_allclose(lisi[cluster_key], repeated_lisi[cluster_key])
    assert set(datastore.cells.columns) == columns_before
    assert set(datastore.list_artifacts()) == artifacts_before

    read_only = DataStore(
        datastore.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    read_only_lisi = read_only.metric_lisi(
        [cluster_key],
        perplexity=1,
    )
    np.testing.assert_allclose(read_only_lisi[cluster_key], lisi[cluster_key])

    unsaved_smart = datastore.smart_label(
        cluster_key,
        cluster_key,
    )
    assert unsaved_smart == datastore.smart_label(
        cluster_key,
        cluster_key,
    )
    datastore.smart_label(
        cluster_key,
        cluster_key,
        new_col_name="smart_clusters",
    )
    smart_ref = _column_ref(datastore, "smart_clusters")
    assert smart_ref.kind == "smart_label"
    with monkeypatch.context() as cached:
        cached.setattr(
            pd,
            "crosstab",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cached smart labels were recomputed")
            ),
        )
        datastore.smart_label(
            cluster_key,
            cluster_key,
            new_col_name="smart_clusters",
        )
    assert _column_ref(datastore, "smart_clusters") == smart_ref


def test_hto_identity_is_artifact_backed(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    n_active = len(datastore.cells.active_index("I"))
    column_name = "sample_id"
    expected = np.asarray(
        ["negative" if index % 2 else "tag" for index in range(n_active)]
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.quality_control.hto_demux",
        lambda counts, **kwargs: pd.Series(expected[: len(counts)]),
    )

    returned_label = datastore.mark_hto_identities(
        from_assay="assay2",
        cell_key="I",
        label=column_name,
    )

    assert returned_label == column_name
    ref = _column_ref(datastore, column_name)
    assert ref.kind == "hto_identity"
    parameters = datastore.inspect_artifact(ref).parameters
    assert parameters is not None
    assert parameters["method"] == {
        "normalization": "clr_per_hto",
        "clustering": {
            "method": "kmeans",
            "init": "random",
            "n_starts": 100,
            "cluster_count": "n_htos_plus_one",
        },
        "background": "raw_mean",
        "cutoff": {
            "distribution": "negative_binomial_nb2",
            "quantile": 0.99,
            "location": 0,
            "comparison": "strictly_greater",
        },
        "singlet_assignment": "clr_argmax",
    }
    assert "algorithm_version" not in parameters
    np.testing.assert_array_equal(
        datastore.cells.fetch(column_name, key="I"),
        expected,
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.quality_control.hto_demux",
        lambda counts, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached HTO identities were recomputed")
        ),
    )
    cached_label = datastore.mark_hto_identities(
        from_assay="assay2",
        cell_key="I",
        label=column_name,
    )
    assert cached_label == column_name
    assert _column_ref(datastore, column_name) == ref


def test_user_display_metadata_survives_column_refresh(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    datastore.run_umap(n_epochs=10, label="display_umap")
    datastore.run_leiden_clustering(label="display_leiden")
    column = datastore.zw["cellData"]["RNA_display_leiden"]
    custom = {
        "kind": "categorical",
        "categories": [
            {
                "value": int(category),
                "label": f"Group {category}",
                "color": "#123456",
            }
            for category in np.unique(
                datastore.cells.fetch("RNA_display_leiden", key="I")
            )
        ],
    }
    column.attrs["display"] = custom

    datastore.run_leiden_clustering(
        label="display_leiden",
        invalidate_cache=True,
    )

    assert (
        dict(datastore.zw["cellData"]["RNA_display_leiden"].attrs["display"]) == custom
    )
    result = splt.embedding(
        datastore,
        layout_key="RNA_display_umap",
        color_by="RNA_display_leiden",
        show=False,
    )
    categorical = next(
        scale for scale in result.scales if isinstance(scale, CategoricalScale)
    )
    assert categorical.palette is not None
    assert set(categorical.palette.values()) == {"#123456"}
    assert categorical.labels is not None
    assert all(label.startswith("Group ") for label in categorical.labels.values())
    result.close()


def test_malformed_display_is_rejected_before_column_refresh(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    datastore.run_leiden_clustering(label="malformed_display")
    column_name = "RNA_malformed_display"
    column = datastore.zw["cellData"][column_name]
    original_ref = dict(column.attrs["source_artifact"])
    original_values = datastore.cells.fetch_all(column_name).copy()
    column.attrs["display"] = "invalid"

    with pytest.raises(TypeError, match="mapping"):
        datastore.run_leiden_clustering(
            label="malformed_display",
            invalidate_cache=True,
        )

    np.testing.assert_array_equal(
        datastore.cells.fetch_all(column_name),
        original_values,
    )
    assert (
        dict(datastore.zw["cellData"][column_name].attrs["source_artifact"])
        == original_ref
    )


def test_multicolumn_display_validation_is_atomic(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    datastore.run_umap(n_epochs=10, label="atomic_display")
    first = datastore.zw["cellData"]["RNA_atomic_display1"]
    second = datastore.zw["cellData"]["RNA_atomic_display2"]
    first_ref = dict(first.attrs["source_artifact"])
    second_ref = dict(second.attrs["source_artifact"])
    second.attrs["display"] = "invalid"

    with pytest.raises(TypeError, match="mapping"):
        datastore.run_umap(
            n_epochs=10,
            label="atomic_display",
            invalidate_cache=True,
        )

    assert dict(first.attrs["source_artifact"]) == first_ref
    assert dict(second.attrs["source_artifact"]) == second_ref


def test_display_validation_rejects_nonfinite_and_duplicate_values() -> None:
    with pytest.raises(TypeError, match="minimum"):
        validate_display_metadata(
            {
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": np.nan,
                "maximum": 1.0,
                "scale": "linear",
            }
        )
    with pytest.raises(ValueError, match="unique"):
        validate_display_metadata(
            {
                "kind": "categorical",
                "categories": [
                    {"value": 1, "label": "A", "color": "#123456"},
                    {"value": 1, "label": "B", "color": "#654321"},
                ],
            }
        )
    with pytest.raises(ValueError, match="collide"):
        validate_display_metadata(
            {
                "kind": "categorical",
                "categories": [
                    {"value": True, "label": "Yes", "color": "#123456"},
                    {"value": 1, "label": "One", "color": "#654321"},
                ],
            }
        )
    with pytest.raises(ValueError, match="unknown fields"):
        validate_display_metadata(
            {
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": 0.0,
                "maximum": 1.0,
                "scale": "linear",
                "extra": "invalid",
            }
        )


def test_small_constant_log_bounds_remain_positive() -> None:
    limits = _continuous_limits(
        np.asarray([0.1, 0.1]),
        ColorScale(
            cmap="viridis",
            vmin=0.1,
            vmax=0.1,
            scale="log",
        ),
    )

    assert limits[0] > 0
    assert limits[0] < 0.1 < limits[1]


def test_composition_keeps_missing_category_distinct(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    n_cells = len(datastore.cells.active_index("I"))
    values = np.asarray(
        [
            np.nan if index % 3 == 0 else 1.0 if index % 3 == 1 else 2.0
            for index in range(n_cells)
        ],
        dtype=float,
    )
    datastore.cells.insert(
        "composition_missing",
        values,
        key="I",
        overwrite=True,
    )
    result = splt.composition(
        datastore,
        category_by="composition_missing",
        categorical_scale=CategoricalScale(
            order=(1.0, 2.0),
            palette={1.0: "#123456", 2.0: "#654321"},
            missing_color="#ff0000",
            missing_label="Missing",
        ),
        show=False,
    )

    categories = result.tables["aggregate"]["category"].tolist()
    assert 1.0 in categories
    assert any(value is None for value in categories)
    assert all(value != "\x00scarf_missing_category\x00" for value in categories)
    returned = next(
        scale for scale in result.scales if isinstance(scale, CategoricalScale)
    )
    assert returned.missing_color == "#ff0000"
    result.close()


def test_manual_column_edit_is_captured_as_new_artifact(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    datastore.run_leiden_clustering(label="editable_leiden")
    column = "RNA_editable_leiden"
    original = _column_ref(datastore, column)
    edited = np.asarray(datastore.cells.fetch(column, key="I")).copy()
    edited[0] = int(edited.max()) + 1
    datastore.cells.insert(column, edited, key="I", overwrite=True)

    captured = datastore._resolve_cell_data_input(column, cell_key="I")

    assert captured != original
    assert captured.kind == "metadata_snapshot"
    np.testing.assert_array_equal(
        datastore.load_artifact(captured)["values"][:],
        edited,
    )
    assert _column_ref(datastore, column) == captured


def test_cell_cycle_columns_share_one_artifact(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    datastore.auto_filter_cells(show_qc_plots=False)
    datastore.run_cell_cycle_scoring()

    refs = {
        _column_ref(datastore, "RNA_S_score"),
        _column_ref(datastore, "RNA_G2M_score"),
        _column_ref(datastore, "RNA_cell_cycle_phase"),
    }

    assert len(refs) == 1
    ref = refs.pop()
    assert ref.kind == "cell_cycle"
    assert set(datastore.load_artifact(ref).array_keys()) == {
        "s_score",
        "g2m_score",
        "phase",
    }


def test_fate_columns_share_one_reusable_artifact(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    n_cells = len(datastore.cells.active_index("I"))
    pseudotime = np.linspace(0.0, 1.0, n_cells)
    labels = np.full(n_cells, "other", dtype=object)
    labels[-2:] = ["A", "B"]
    datastore.cells.insert(
        "artifact_pseudotime",
        pseudotime,
        key="I",
        overwrite=True,
    )
    datastore.cells.insert(
        "artifact_sinks",
        labels,
        key="I",
        overwrite=True,
    )

    result = datastore.run_fate_mapping(
        pseudotime_key="artifact_pseudotime",
        sink_key="artifact_sinks",
        sinks=["A", "B"],
        label="artifact_fate",
    )
    refs = {
        _column_ref(datastore, column)
        for column in (*result.fate_keys, result.validity_key)
    }
    assert len(refs) == 1
    fate_ref = refs.pop()
    assert fate_ref.kind == "fate_map"

    from scarf.datastore._operations import trajectory as trajectory_operations

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("fate mapping should have been reused")

    monkeypatch.setattr(
        trajectory_operations,
        "_compute_fate_probabilities_impl",
        fail_if_recomputed,
    )
    cached = datastore.run_fate_mapping(
        pseudotime_key="artifact_pseudotime",
        sink_key="artifact_sinks",
        sinks=["A", "B"],
        label="artifact_fate_cached",
    )
    assert _column_ref(datastore, cached.fate_keys[0]) == fate_ref
    np.testing.assert_allclose(cached.values, result.values, equal_nan=True)


def test_graph_outputs_reject_equal_size_different_cell_selection(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    mask = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    selected = np.flatnonzero(mask)
    excluded = np.flatnonzero(~mask)
    assert len(selected) > 0 and len(excluded) > 0
    mask[selected[0]] = False
    mask[excluded[0]] = True
    datastore.cells.insert("I", mask, overwrite=True, force=True)

    with pytest.raises(ValueError, match="no longer matches"):
        datastore.run_leiden_clustering(label="misaligned")
    with pytest.raises(ValueError, match="no longer matches"):
        datastore.run_umap(n_epochs=10, label="misaligned_umap")
    with pytest.raises(ValueError, match="no longer matches"):
        datastore.get_diffusion_operator(invalidate_cache=True)
    with pytest.raises(ValueError, match="no longer matches"):
        datastore.metric_lisi(
            ["names"],
            perplexity=1,
        )


def test_graph_outputs_accept_content_equivalent_selection_artifacts(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    equivalent = resolve_selection_artifact(
        datastore.zw,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray(datastore.cells.fetch_all("I"), dtype=bool),
        row_ids=np.asarray(datastore.cells.fetch_all("ids")),
        operation="equivalent_selection",
        parameters={"source": "test"},
        inputs={},
        source_column="I",
    )
    datastore.zw["cellData/I"].attrs["source_artifact"] = equivalent.to_dict()
    datastore.zw["cellData/I"].attrs["source_value"] = "values"

    datastore.run_leiden_clustering(label="equivalent_selection")
    assert "RNA_equivalent_selection" in datastore.cells.columns


def test_graph_consumers_accept_an_explicit_connectivity_map(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    state = datastore.get_assay_state("RNA")
    assert state is not None
    graph = state.connectivity_map
    assert graph is not None

    implicit = datastore.run_leiden_clustering(label="implicit_leiden")
    explicit = datastore.run_leiden_clustering(graph, label="explicit_leiden")

    assert explicit == implicit
    np.testing.assert_array_equal(
        datastore.cells.fetch("RNA_implicit_leiden", key="I"),
        datastore.cells.fetch("RNA_explicit_leiden", key="I"),
    )
    umap_ref = datastore.run_umap(graph, n_epochs=10, label="explicit_umap")
    assert umap_ref.kind == "embedding"
    assert "RNA_explicit_umap1" in datastore.cells.columns

    side_neighbors = datastore.query_neighbors(
        state.ann_index,
        k=5,
        update_state=False,
    )
    side_graph = datastore.build_connectivity_map(
        side_neighbors,
        update_state=False,
    )
    side = datastore.run_leiden_clustering(side_graph, label="side_leiden")
    assert side != implicit
    assert "RNA_side_leiden" in datastore.cells.columns
    assert datastore.get_assay_state("RNA").connectivity_map == graph

    with pytest.raises(ValueError, match="does not match the graph"):
        datastore.run_leiden_clustering(graph, feat_key="I", label="mismatched")
    with pytest.raises(TypeError, match="artifact reference"):
        datastore.run_umap("RNA/graph", n_epochs=10, label="not_a_ref")
    with pytest.raises(ValueError, match="connectivity map"):
        datastore.run_umap(state.reduction, n_epochs=10, label="wrong_kind")


def test_lisi_rejects_incomplete_ann_dependency(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.ann_index is not None
    assert state.neighbors is not None
    ann_group = datastore.zw[artifact_path(state.ann_index)]
    ann_group.attrs["complete"] = False

    try:
        with pytest.raises(ValueError, match="ann_index artifact"):
            datastore.metric_lisi(
                ["names"],
                use_latest_knn=False,
                knn_loc=artifact_path(state.neighbors),
                perplexity=1,
            )
    finally:
        ann_group.attrs["complete"] = True


def test_unedited_paris_column_retains_cluster_cut_ref(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    result = datastore.run_paris_clustering(
        n_clusters=3,
        label="metadata_paris",
    )
    assert result.label_key is not None
    linked = _column_ref(datastore, result.label_key)

    resolved = datastore._resolve_cell_data_input(
        result.label_key,
        cell_key="I",
    )

    assert linked.kind == "cluster_cut"
    assert resolved == linked
