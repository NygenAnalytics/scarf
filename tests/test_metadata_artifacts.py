import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.metadata.artifacts as metadata_artifacts_module
import scarf.plotting as splt
from scarf.datastore.datastore import DataStore
from scarf.metadata.artifacts import (
    categorical_display,
    continuous_display,
    feature_column_display,
    link_feature_data_column,
    plan_cell_data_artifact,
    validate_display_metadata,
    write_cell_data_artifact,
)
from scarf.plotting._contracts import CategoricalScale, ColorScale
from scarf.plotting.embedding import _continuous_limits
from scarf.storage.artifacts import ArtifactRef, artifact_path, inspect_artifact
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import resolve_selection_artifact
from tests.fixtures_datastore import build_neighbourhood_graph


def _ensure_graph(datastore) -> None:
    datastore.auto_filter_cells(show_qc_plots=False)
    feature_selection = datastore.mark_hvgs(
        from_assay="RNA",
        cell_key="I",
        top_n=100,
        label="metadata_hvgs",
        show_plot=False,
        min_cells=int(0.01 * datastore.cells.N),
        max_cells=np.inf,
        blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
    )
    build_neighbourhood_graph(
        datastore,
        from_assay="RNA",
        features=feature_selection,
        dims=5,
        k=3,
        n_centroids=10,
        local_cache=False,
    )


def _column_ref(datastore, column: str) -> ArtifactRef:
    raw_ref = datastore.zw["cellData"][column].attrs["source_artifact"]
    return ArtifactRef.from_dict(raw_ref)


def _memory_metadata_root() -> tuple[zarr.Group, ArtifactRef]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_group("RNA").create_group("featureData")
    cell_data = root.create_group("cellData")
    cell_ids = np.asarray(["cell-0", "cell-1", "cell-2"])
    selection = np.asarray([True, False, True])
    cell_data.create_array("ids", data=cell_ids)
    cell_data.create_array("I", data=selection)
    selection_ref = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=selection,
        row_ids=cell_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    return root, selection_ref


def test_cell_data_artifact_cache_hit_miss_and_payload_validation(
    monkeypatch,
) -> None:
    root, selection = _memory_metadata_root()
    common = {
        "scope": "datastore",
        "kind": "metadata_snapshot",
        "operation": "cache_metadata",
        "parameters": {"label": "batch"},
        "inputs": {},
        "execution_options": {"source_column": "batch"},
        "cell_selection": selection,
        "arrays": {"values": ((2,), "f")},
    }
    values = np.asarray([0.25, 0.75], dtype=np.float64)
    first = plan_cell_data_artifact(root, **common)
    first_group = write_cell_data_artifact(root, first, {"values": values})

    assert first.reused is False
    assert inspect_artifact(root, first.ref).complete
    np.testing.assert_array_equal(first_group["values"][:], values)

    cached = plan_cell_data_artifact(root, **common)
    assert cached.reused is True
    assert cached.ref == first.ref

    def fail_if_written(*_args, **_kwargs):
        raise AssertionError("a cached metadata artifact was rewritten")

    with monkeypatch.context() as cache_hit:
        cache_hit.setattr(
            metadata_artifacts_module,
            "create_zarr_dataset",
            fail_if_written,
        )
        reused_group = write_cell_data_artifact(root, cached, {"values": values})
    assert reused_group.path == first_group.path

    changed = plan_cell_data_artifact(
        root,
        **{
            **common,
            "parameters": {"label": "condition"},
        },
    )
    invalidated = plan_cell_data_artifact(
        root,
        **{
            **common,
            "invalidate_cache": True,
        },
    )
    assert changed.reused is False
    assert changed.ref != first.ref
    assert invalidated.reused is False
    assert invalidated.ref != first.ref

    del first_group["values"]
    corrupt_miss = plan_cell_data_artifact(root, **common)
    assert inspect_artifact(root, first.ref).complete
    assert corrupt_miss.reused is False
    assert corrupt_miss.ref != first.ref


def test_cell_data_artifact_validation_and_failed_write_status() -> None:
    root, selection = _memory_metadata_root()
    wrong_selection = ArtifactRef(
        scope="datastore",
        kind="metadata_snapshot",
        artifact_id="f" * 64,
    )
    common = {
        "scope": "datastore",
        "kind": "metadata_snapshot",
        "operation": "validate_metadata",
        "parameters": {},
        "inputs": {},
        "execution_options": {},
    }

    with pytest.raises(ValueError, match="cell-selection"):
        plan_cell_data_artifact(
            root,
            **common,
            cell_selection=wrong_selection,
            arrays={"values": ((2,), None)},
        )
    with pytest.raises(ValueError, match="selected cell count"):
        plan_cell_data_artifact(
            root,
            **common,
            cell_selection=selection,
            arrays={"values": ((1,), None)},
        )

    planned = plan_cell_data_artifact(
        root,
        **common,
        cell_selection=selection,
        arrays={"values": ((2, 1), None)},
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        write_cell_data_artifact(
            root,
            planned,
            {"values": np.asarray([["a"], ["b"]])},
        )

    failed_status = inspect_artifact(root, planned.ref)
    assert failed_status.exists
    assert failed_status.complete is False
    retry = plan_cell_data_artifact(
        root,
        **common,
        cell_selection=selection,
        arrays={"values": ((2, 1), None)},
    )
    assert retry.reused is False
    assert retry.ref != planned.ref


def test_corrupt_metadata_links_and_incomplete_artifacts_are_rebuilt(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    column_name = "imported_metadata"
    values = np.asarray([f"batch-{index % 2}" for index in range(datastore.cells.N)])
    datastore.cells.insert(column_name, values, overwrite=True)
    column = datastore.zw["cellData"][column_name]
    column.attrs["source_artifact"] = {
        "type": "artifact",
        "scope": "datastore",
    }
    column.attrs["source_value"] = "values"

    first = datastore._resolve_cell_data_input(column_name, cell_key="I")

    assert first.kind == "metadata_snapshot"
    linked = datastore.zw["cellData"][column_name]
    assert ArtifactRef.from_dict(linked.attrs["source_artifact"]) == first
    np.testing.assert_array_equal(
        datastore.load_artifact(first)["values"][:],
        datastore.cells.fetch(column_name, key="I"),
    )

    datastore.zw[artifact_path(first)].attrs["complete"] = False
    replacement = datastore._resolve_cell_data_input(column_name, cell_key="I")

    assert replacement != first
    assert datastore.inspect_artifact(first).complete is False
    assert datastore.inspect_artifact(replacement).complete
    linked = datastore.zw["cellData"][column_name]
    assert ArtifactRef.from_dict(linked.attrs["source_artifact"]) == replacement
    np.testing.assert_array_equal(
        datastore.load_artifact(replacement)["values"][:],
        datastore.cells.fetch(column_name, key="I"),
    )


def test_datastore_rejects_incomplete_import_status(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    datastore.zw.attrs["scarf:import_source"] = "synthetic"
    datastore.zw.attrs["scarf:import_complete"] = False

    with pytest.raises(RuntimeError, match="synthetic import is incomplete"):
        DataStore(
            datastore.zarr_loc,
            default_assay="RNA",
        )


def test_datastore_rejects_corrupt_imported_metadata(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    datastore.zw.attrs["scarf:import_source"] = "synthetic"
    datastore.zw.attrs["scarf:import_complete"] = True
    datastore.zw["cellData"].create_array(
        "truncated_imported_metadata",
        data=np.asarray(["only-one-row"]),
    )

    with pytest.raises(ValueError, match="Metadata table is corrupted"):
        DataStore(
            datastore.zarr_loc,
            default_assay="RNA",
        )


def test_feature_link_validation_preserves_existing_provenance() -> None:
    root, _selection = _memory_metadata_root()
    feature_data = root["RNA"]["featureData"]
    values = np.asarray([0.25, 0.75])
    feature_data.create_array("score", data=values)
    first = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        artifact_id="a" * 64,
    )
    display = continuous_display(values)
    link_feature_data_column(
        root["RNA"],
        "score",
        first,
        value_name="values",
        default_display=display,
    )
    assert feature_column_display(root["RNA"], "score") == display

    target = feature_data["score"]
    target.attrs["display"] = "invalid"
    before = dict(target.attrs)
    replacement = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="feature_selection",
        artifact_id="b" * 64,
    )
    with pytest.raises(TypeError, match="mapping"):
        link_feature_data_column(
            root["RNA"],
            "score",
            replacement,
            value_name="replacement",
            value_index=1,
        )

    persisted = root["RNA"]["featureData"]["score"]
    assert dict(persisted.attrs) == before


def test_umap_matrix_and_leiden_columns_link_authoritative_artifacts(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    hvg_column = datastore.RNA.z["featureData"]["metadata_hvgs"]
    assert ArtifactRef.from_dict(
        hvg_column.attrs["source_artifact"]
    ) == datastore.resolve_features("RNA", "metadata_hvgs")
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
    assert datastore.inspect_artifact(leiden_ref).parameters["backend"] == "igraph"
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


def test_leiden_backend_is_part_of_artifact_identity(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)

    native = datastore.run_leiden_clustering(label="native_leiden")
    legacy = datastore.run_leiden_clustering(
        backend="leidenalg",
        label="legacy_leiden",
    )

    assert native != legacy
    assert datastore.inspect_artifact(native).parameters["backend"] == "igraph"
    assert datastore.inspect_artifact(legacy).parameters["backend"] == "leidenalg"


def test_run_leiden_rejects_unknown_backend(datastore_ephemeral) -> None:
    with pytest.raises(ValueError, match="backend"):
        datastore_ephemeral.run_leiden_clustering(
            backend="unknown",  # type: ignore[arg-type]
            label="bad_backend",
        )


def test_membership_and_smart_labels_are_artifact_backed_and_lisi_is_read_only(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    _ensure_graph(datastore)
    datastore.run_leiden_clustering(label="independent_leiden")
    cluster_key = "RNA_independent_leiden"

    datastore.calc_membership_strength(cluster_key)
    membership_key = "RNA_I_cluster_membership_strength"
    membership_ref = _column_ref(datastore, membership_key)
    assert membership_ref.kind == "membership_strength"
    state = datastore.get_assay_state("RNA")
    assert state is not None
    assert state.connectivity_map is not None
    graph_loc = artifact_path(state.connectivity_map)
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


@pytest.mark.parametrize(
    ("display", "error_type", "message"),
    [
        (
            {"kind": "continuous"},
            ValueError,
            "incomplete",
        ),
        (
            {
                "kind": "continuous",
                "colormap": 1,
                "minimum": 0.0,
                "maximum": 1.0,
                "scale": "linear",
            },
            TypeError,
            "colormap",
        ),
        (
            {
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": 0.0,
                "maximum": 1.0,
                "scale": "square-root",
            },
            ValueError,
            "scale",
        ),
        (
            {
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": 2.0,
                "maximum": 1.0,
                "scale": "linear",
            },
            ValueError,
            "exceeds",
        ),
        (
            {"kind": "categorical", "categories": "not-a-list"},
            TypeError,
            "must be a list",
        ),
        (
            {"kind": "categorical", "categories": ["not-a-mapping"]},
            TypeError,
            "must be a mapping",
        ),
        (
            {
                "kind": "categorical",
                "categories": [{"value": 1, "label": "one"}],
            },
            ValueError,
            "requires value, label, and color",
        ),
        (
            {
                "kind": "categorical",
                "categories": [{"value": None, "label": "none", "color": "#123456"}],
            },
            TypeError,
            "JSON scalar",
        ),
        (
            {
                "kind": "categorical",
                "categories": [{"value": 1, "label": 1, "color": "#123456"}],
            },
            TypeError,
            "label must be a string",
        ),
        (
            {
                "kind": "categorical",
                "categories": [{"value": 1, "label": "one", "color": "red"}],
            },
            ValueError,
            "hex color",
        ),
        (
            {
                "kind": "categorical",
                "categories": [],
                "missing_label": 1,
            },
            TypeError,
            "missing_label",
        ),
        (
            {
                "kind": "categorical",
                "categories": [],
                "missing_color": "red",
            },
            ValueError,
            "missing_color",
        ),
        (
            {"kind": "unknown"},
            ValueError,
            "continuous or categorical",
        ),
    ],
)
def test_display_validation_rejects_malformed_contracts(
    display: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        validate_display_metadata(display)


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

    with pytest.raises(TypeError, match="feat_key"):
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
        with pytest.raises(
            ArtifactResolutionError,
            match=r"(?i)artifact is incomplete",
        ) as error:
            datastore.metric_lisi(
                ["names"],
                neighbors=state.neighbors,
                perplexity=1,
            )
        assert error.value.code == "incomplete_artifact"
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
