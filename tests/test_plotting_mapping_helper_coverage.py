"""Focused edge coverage for plotting and mapping helper contracts."""

from collections.abc import Mapping
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf.assay import RNAassay, norm_lib_size
from scarf.mapping import artifact as mapping_artifact
from scarf.mapping import features as mapping_features
from scarf.mapping import reference as mapping_reference
from scarf.mapping.models import (
    ScaledPCAProjectionModel,
    SymphonyCorrectionModel,
)
from scarf.mapping.reference import MappingReference
from scarf.plotting import _data as plotting_data
from scarf.plotting._contracts import CellField, ColorScale, NormalizationSpec
from scarf.storage.artifacts import ArtifactRef, callable_identity
from scarf.storage.budget import ResourceBudget


distribution_plot = import_module("scarf.plotting.distribution")


def _ref(
    kind: str,
    token: str,
    *,
    assay: str | None = "RNA",
) -> ArtifactRef:
    return ArtifactRef(
        scope="datastore" if assay is None else "assay",
        assay=assay,
        kind=kind,
        artifact_id=token * 64,
    )


def _model() -> ScaledPCAProjectionModel:
    return ScaledPCAProjectionModel(
        feature_means=np.array([0.0, 1.0]),
        feature_scales=np.array([1.0, 2.0]),
        loadings=np.eye(2),
    )


def _symphony() -> SymphonyCorrectionModel:
    return SymphonyCorrectionModel(
        centroids=np.eye(2),
        raw_centroids=np.eye(2),
        corrected_centroids=np.eye(2) * 2,
        cluster_mass=np.array([1.0, 2.0]),
        sigma=np.array([0.5, 1.0]),
    )


def _reference(*, metadata: Mapping[str, Any] | None = None) -> MappingReference:
    return MappingReference(
        datastore=SimpleNamespace(zw=object(), cells=object()),
        ref=_ref("mapping_reference", "a"),
        assay_name="RNA",
        reduction=_ref("reduction", "b"),
        ann_index=_ref("ann_index", "c"),
        neighbors=_ref("neighbors", "d"),
        cell_selection=_ref("cell_selection", "e", assay=None),
        feature_selection=_ref("feature_selection", "f"),
        batch_correction=None,
        dataset_fingerprint="dataset",
        selected_cell_count=2,
        model=_model(),
        symphony_state=None,
        feature_ids=np.array(["g0", "g1"]),
        metadata=metadata
        or {
            "method": "pca",
            "ann_metric": "l2",
            "normalization_parameters": {"size_factor": 1_000.0},
        },
        reference_distance_quantiles=np.array([0.0, 1.0]),
        reference_distance_values=np.array([0.1, 0.2]),
    )


class _FakeArray:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.dtype = self.values.dtype
        self.attrs: dict[str, Any] = {}

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.values[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.values[key] = value


class _FakeGroup(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        group = _FakeGroup()
        self[name] = group
        return group

    def group_keys(self) -> list[str]:
        return [name for name, value in self.items() if isinstance(value, _FakeGroup)]

    def array_keys(self) -> list[str]:
        return [name for name, value in self.items() if isinstance(value, _FakeArray)]


def _root() -> _FakeGroup:
    return _FakeGroup()


def _array(
    root: _FakeGroup,
    name: str,
    values: Any,
    *,
    chunks: tuple[int, ...] | None = None,
) -> _FakeArray:
    data = np.asarray(values)
    del chunks
    array = _FakeArray(data)
    root[name] = array
    return array


def test_plot_data_rejects_malformed_artifact_inputs(monkeypatch) -> None:
    store = SimpleNamespace(zw=object())
    layout = _ref("embedding", "1")
    selection = _ref("cell_selection", "2", assay=None)

    monkeypatch.setattr(
        plotting_data,
        "inspect_artifact",
        lambda *_: SimpleNamespace(inputs={}),
    )
    with pytest.raises(ValueError, match="no cell-selection"):
        plotting_data._artifact_cell_selection(store, layout, label="Plot")

    monkeypatch.setattr(
        plotting_data,
        "inspect_artifact",
        lambda *_: SimpleNamespace(inputs={"cell_selection": {}}),
    )
    with pytest.raises(ValueError, match="invalid cell-selection"):
        plotting_data._artifact_cell_selection(store, layout, label="Plot")

    with pytest.raises(TypeError, match="ArtifactRef"):
        plotting_data._validated_embedding_selection(store, "umap")  # type: ignore[arg-type]

    monkeypatch.setattr(
        plotting_data,
        "inspect_artifact",
        lambda *_: SimpleNamespace(complete=False, operation="run_umap", inputs={}),
    )
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        plotting_data._validated_embedding_selection(store, layout)

    monkeypatch.setattr(
        plotting_data,
        "_artifact_cell_selection",
        lambda *_args, **_kwargs: selection,
    )
    status = SimpleNamespace(complete=True, operation="run_umap", inputs={})
    monkeypatch.setattr(plotting_data, "inspect_artifact", lambda *_: status)
    with pytest.raises(ValueError, match="no graph input"):
        plotting_data._validated_embedding_selection(store, layout)

    status.inputs = {"graph": {}}
    with pytest.raises(ValueError, match="invalid graph input"):
        plotting_data._validated_embedding_selection(store, layout)

    status.inputs = {"graph": _ref("neighbors", "3", assay="ATAC").to_dict()}
    with pytest.raises(ValueError, match="scope does not match"):
        plotting_data._validated_embedding_selection(store, layout)


def test_plot_data_grouping_and_layout_validation(monkeypatch) -> None:
    store = SimpleNamespace(zw=object())
    groups = _ref("cluster_labels", "4", assay=None)
    selection = _ref("cell_selection", "5", assay=None)

    with pytest.raises(ValueError, match="exactly one"):
        plotting_data._resolve_grouping(
            store,
            group_by=None,
            groups=None,
            cell_key="I",
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        plotting_data._resolve_grouping(
            store,
            group_by=None,
            groups="labels",  # type: ignore[arg-type]
            cell_key="I",
        )
    with pytest.raises(ValueError, match="cannot override"):
        plotting_data._resolve_grouping(
            store,
            group_by=None,
            groups=groups,
            cell_key="custom",
        )

    status = SimpleNamespace(complete=False)
    monkeypatch.setattr(plotting_data, "inspect_artifact", lambda *_: status)
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        plotting_data._resolve_grouping(
            store,
            group_by=None,
            groups=groups,
            cell_key="I",
        )

    status.complete = True
    monkeypatch.setattr(
        plotting_data,
        "_artifact_cell_selection",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        plotting_data,
        "read_stored_selection_indices",
        lambda *_args, **_kwargs: np.array([0, 2]),
    )
    payload: dict[str, Any] = {}
    monkeypatch.setattr(plotting_data, "artifact_group", lambda *_: payload)
    monkeypatch.setattr(plotting_data, "as_zarr_array", lambda value, **_: value)
    with pytest.raises(ValueError, match="canonical 'values'"):
        plotting_data._resolve_grouping(
            store,
            group_by=None,
            groups=groups,
            cell_key="I",
        )

    payload["values"] = np.array(["a"])
    with pytest.raises(ValueError, match="do not align"):
        plotting_data._resolve_grouping(
            store,
            group_by=None,
            groups=groups,
            cell_key="I",
        )

    layout = _ref("embedding", "6")
    monkeypatch.setattr(
        plotting_data,
        "_validated_embedding_selection",
        lambda *_: selection,
    )
    payload.clear()
    with pytest.raises(ValueError, match="no canonical values"):
        plotting_data._resolve_layout(store, layout)

    payload["values"] = np.array([["bad", "coordinates"], ["x", "y"]])
    with pytest.raises(TypeError, match="must be numeric"):
        plotting_data._resolve_layout(store, layout)

    payload["values"] = np.ones((2, 1))
    with pytest.raises(ValueError, match="two columns"):
        plotting_data._resolve_layout(store, layout)

    payload["values"] = np.array([[0.0, 1.0], [np.inf, 2.0]])
    with pytest.raises(ValueError, match="must be finite"):
        plotting_data._resolve_layout(store, layout)

    with pytest.raises(ValueError, match="No cells remain"):
        plotting_data.resolve_cell_selection(
            1,
            subset=np.array([False]),
            category_values=np.array(["a"]),
        )


class _AxisProbe:
    def __init__(self) -> None:
        self.steps: list[tuple[Any, ...]] = []
        self.plots: list[tuple[Any, ...]] = []
        self.texts: list[tuple[Any, ...]] = []
        self.legend_calls = 0
        self.ylim = (0.0, 1.0)
        self.xlim = (0.0, 1.0)

    def hist(self, *args: Any, **kwargs: Any) -> None:
        self.plots.append((args, kwargs))

    def step(self, *args: Any, **kwargs: Any) -> None:
        self.steps.append((args, kwargs))

    def legend(self, *args: Any, **kwargs: Any) -> None:
        self.legend_calls += 1

    def get_ylim(self) -> tuple[float, float]:
        return self.ylim

    def set_ylim(self, *values: float) -> None:
        self.ylim = (values[0], values[1])

    def get_xlim(self) -> tuple[float, float]:
        return self.xlim

    def set_xlim(self, *values: float) -> None:
        self.xlim = (values[0], values[1])

    def plot(self, *args: Any, **kwargs: Any) -> None:
        self.plots.append((args, kwargs))

    def text(self, *args: Any, **kwargs: Any) -> None:
        self.texts.append((args, kwargs))


def test_distribution_fetch_and_drawing_edge_cases(monkeypatch) -> None:
    store = SimpleNamespace(cells=object())
    indices = np.array([0, 1])
    monkeypatch.setattr(
        distribution_plot,
        "read_metadata_rows",
        lambda *_: np.array([1.0, 2.0]),
    )
    monkeypatch.setattr(
        distribution_plot,
        "read_metadata_missing_rows",
        lambda *_: np.array([True]),
    )
    with pytest.raises(ValueError, match="does not match"):
        distribution_plot._fetch_metadata_series(store, "metric", indices)

    monkeypatch.setattr(
        distribution_plot,
        "read_metadata_rows",
        lambda *_: np.array(["a", "missing"], dtype=object),
    )
    monkeypatch.setattr(
        distribution_plot,
        "read_metadata_missing_rows",
        lambda *_: np.array([False, True]),
    )
    values, _ = distribution_plot._fetch_metadata_series(store, "label", indices)
    assert values[0] == "a"
    assert pd.isna(values[1])

    monkeypatch.setattr(
        distribution_plot,
        "_fetch_metadata_series",
        lambda *_: (np.array([3.0, 4.0]), "identity"),
    )
    fetched = distribution_plot._fetch_series(
        store,
        CellField("metric", label="Metric"),
        cell_indices=indices,
        from_assay=None,
        normalization=NormalizationSpec(),
    )
    assert fetched == (
        pytest.approx(np.array([3.0, 4.0])),
        "Metric",
        False,
        "identity",
        None,
    )

    axis = _AxisProbe()
    frame = pd.DataFrame({"group": ["a"], "value": [1.0]})
    distribution_plot._draw_hist(
        axis,
        frame,
        color="black",
        bins=4,
        group_by="group",
        order=["missing"],
        palette=None,
        show_legend=True,
    )
    assert not axis.plots
    assert axis.legend_calls == 1

    empty_x, empty_y = distribution_plot._ecdf_xy(np.array([np.nan, np.inf]))
    assert empty_x.size == empty_y.size == 0

    axis = _AxisProbe()
    was_subsampled = distribution_plot._draw_ecdf(
        axis,
        pd.DataFrame({"group": ["a"] * 4, "value": [1.0, 2.0, 3.0, 4.0]}),
        color="black",
        max_points=2,
        rng=np.random.default_rng(3),
        group_by="group",
        order=["a"],
        palette={"a": "red"},
        show_legend=True,
    )
    assert was_subsampled
    assert len(axis.steps) == 1
    assert axis.legend_calls == 1


def test_distribution_panel_and_color_limit_edges() -> None:
    with pytest.raises(ValueError, match="No finite values"):
        distribution_plot._panel_display_frame(
            np.array([np.nan, np.nan]),
            np.array(["a", "b"]),
            split_arr=None,
            sample_arr=None,
            sample_stat="mean",
            expression_cutoff=0.0,
            row_standardize=False,
        )

    with pytest.raises(ValueError, match="scope"):
        distribution_plot._mean_color_limits(
            [pd.Series([1.0])],
            SimpleNamespace(scope="invalid"),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="No finite expression"):
        distribution_plot._mean_color_limits(
            [pd.Series([np.nan])],
            ColorScale(scope="shared"),
        )

    panel_limits, reference = distribution_plot._mean_color_limits(
        [pd.Series(dtype=float), pd.Series([2.0, 4.0])],
        ColorScale(scope="panel"),
    )
    assert panel_limits == [(0.0, 1.0), (2.0, 4.0)]
    assert reference == (0.0, 1.0)

    invalid_limits = SimpleNamespace(
        scope="shared",
        quantiles=None,
        vmin=2.0,
        vmax=1.0,
        vcenter=None,
    )
    with pytest.raises(ValueError, match="vmax"):
        distribution_plot._mean_color_limits(
            [pd.Series([1.0, 2.0])],
            invalid_limits,  # type: ignore[arg-type]
        )

    invalid_center = SimpleNamespace(
        scope="shared",
        quantiles=None,
        vmin=0.0,
        vmax=1.0,
        vcenter=2.0,
    )
    with pytest.raises(ValueError, match="vcenter"):
        distribution_plot._mean_color_limits(
            [pd.Series([0.0, 1.0])],
            invalid_center,  # type: ignore[arg-type]
        )

    limits, _ = distribution_plot._mean_color_limits(
        [pd.Series([0.0, 1.0])],
        ColorScale(scope="shared", vcenter=2.0),
    )
    assert limits[0][0] == 0.0
    assert limits[0][1] > 2.0
    assert distribution_plot._render_color_limits(3.0, 3.0) == (2.5, 3.5)

    with pytest.raises(ValueError, match="Unknown colormap"):
        distribution_plot._mean_group_palette(
            pd.Series({"a": 1.0}),
            ["a"],
            color_scale=ColorScale(cmap="definitely-not-a-colormap"),
            lo=0.0,
            hi=1.0,
        )


def test_distribution_stat_formatting_and_annotation_edges() -> None:
    assert distribution_plot._format_stat_p_text(0.0001, False) == "***"
    assert distribution_plot._format_stat_p_text(0.005, False) == "**"
    assert distribution_plot._format_stat_p_text(0.03, False) == "*"
    assert distribution_plot._format_stat_p_text(0.2, False) == "ns"
    assert distribution_plot._format_stat_p_text(0.0001, True) == "p=1.0e-04"
    assert distribution_plot._format_stat_p_text(0.02, True) == "p=0.02"

    result = SimpleNamespace(
        tables={"metric": object()},
        tested_features=("identity",),
        source_assays=(3,),
        value_fingerprints=("",),
    )
    assert distribution_plot._result_key_metadata(result, "absent") == (
        None,
        None,
        False,
        None,
        False,
    )
    assert distribution_plot._result_key_metadata(result, "metric") == (
        "identity",
        None,
        False,
        None,
        False,
    )
    assert not distribution_plot._same_category_universe(["a"], ["a", "b"])

    axis = _AxisProbe()
    common = {
        "ax": axis,
        "group_order": ["a", "b"],
        "orientation": "vertical",
        "bracket_height": 0.1,
        "show_p_value": True,
        "annotation_color": "black",
    }
    assert not distribution_plot._annotate_distribution_stats(
        frame=pd.DataFrame({"statistic": [1.0]}),
        method="one_way_anova",
        **common,
    )
    assert not distribution_plot._annotate_distribution_stats(
        frame=pd.DataFrame(
            {"group_1": ["missing"], "group_2": ["b"], "p_value": [0.01]}
        ),
        method="dunn",
        **common,
    )
    assert not distribution_plot._annotate_distribution_stats(
        frame=pd.DataFrame({"p_value": [0.01]}),
        method="unsupported",
        **common,
    )
    assert not distribution_plot._annotate_distribution_stats(
        frame=pd.DataFrame({"p_value": [np.nan]}),
        method="one_way_anova",
        **common,
    )


def _compatible_stats_result() -> SimpleNamespace:
    return SimpleNamespace(
        grouping=None,
        group_field=CellField("group"),
        cell_selection=None,
        n_cells=4,
        n_groups=2,
        sample_by=None,
        pair_by=None,
        sample_fingerprint=None,
        pair_fingerprint=None,
        summary_scope="cell",
        sample_stat="mean",
        expression_cutoff=0.0,
        tables={"metric": object()},
        tested_features=("identity",),
        source_assays=("RNA",),
        value_fingerprints=("values",),
        cell_selection_fingerprint="cells",
        group_fingerprint="groups",
        group_order=("a", "b"),
        normalization={"source": "assay", "transform": "none"},
        normalization_method={"name": "norm"},
        size_factor=1_000.0,
    )


def _compatibility_issue(
    result: SimpleNamespace,
    **overrides: Any,
) -> str | None:
    arguments: dict[str, Any] = {
        "label": "metric",
        "expected_identity": "identity",
        "expected_value_fingerprint": "values",
        "expected_source_assay": "RNA",
        "grouping": CellField("group"),
        "cell_selection": None,
        "n_cells": 4,
        "n_groups": 2,
        "group_order": ("a", "b"),
        "sample_by": None,
        "pair_by": None,
        "sample_fingerprint": None,
        "pair_fingerprint": None,
        "sample_stat": "mean",
        "expression_cutoff": 0.0,
        "normalization": NormalizationSpec(),
        "normalization_method": {"name": "norm"},
        "size_factor": 1_000.0,
        "cell_selection_fingerprint": "cells",
        "group_fingerprint": "groups",
    }
    arguments.update(overrides)
    return distribution_plot._stat_result_compatibility_issue(result, **arguments)


def _changed(result: SimpleNamespace, **values: Any) -> SimpleNamespace:
    fields = vars(result).copy()
    fields.update(values)
    return SimpleNamespace(**fields)


def test_distribution_stats_compatibility_reports_every_identity_mismatch() -> None:
    base = _compatible_stats_result()
    assert _compatibility_issue(base) is None

    artifact_grouping = _ref("cluster_labels", "7", assay=None)
    assert "grouping artifact" in str(
        _compatibility_issue(
            _changed(base, grouping=artifact_grouping),
            grouping=artifact_grouping,
        )
    )
    assert "metadata grouping" in str(
        _compatibility_issue(_changed(base, group_field=CellField("other")))
    )
    assert "cell-selection artifact" in str(
        _compatibility_issue(
            _changed(base, cell_selection=_ref("cell_selection", "8", assay=None))
        )
    )
    assert "computed on 5 cells" in str(_compatibility_issue(_changed(base, n_cells=5)))
    assert "computed on 3 groups" in str(
        _compatibility_issue(_changed(base, n_groups=3))
    )
    assert "sample_by" in str(_compatibility_issue(base, sample_by="sample"))
    assert "pair_by" in str(_compatibility_issue(base, pair_by="pair"))
    assert "does not include sample-value" in str(
        _compatibility_issue(base, sample_fingerprint="expected")
    )
    assert "sample values do not match" in str(
        _compatibility_issue(
            _changed(base, sample_fingerprint="received"),
            sample_fingerprint="expected",
        )
    )
    assert "does not include pair-value" in str(
        _compatibility_issue(base, pair_fingerprint="expected")
    )
    assert "pair values do not match" in str(
        _compatibility_issue(
            _changed(base, pair_fingerprint="received"),
            pair_fingerprint="expected",
        )
    )
    assert "summary_scope" in str(
        _compatibility_issue(_changed(base, summary_scope="sample"))
    )

    sampled = _changed(
        base,
        sample_by="sample",
        sample_fingerprint="sample-values",
        summary_scope="sample",
        sample_stat="median",
    )
    assert "sample_stat" in str(
        _compatibility_issue(
            sampled,
            sample_by="sample",
            sample_fingerprint="sample-values",
        )
    )
    fractional = _changed(sampled, sample_stat="fraction", expression_cutoff=1.0)
    assert "expression_cutoff" in str(
        _compatibility_issue(
            fractional,
            sample_by="sample",
            sample_fingerprint="sample-values",
            sample_stat="fraction",
            expression_cutoff=2.0,
        )
    )

    cases = (
        ({"tested_features": ()}, {}, "tested-value identity"),
        ({"tested_features": ("other",)}, {}, "does not match panel"),
        ({"value_fingerprints": ()}, {}, "realized-value identity"),
        ({"value_fingerprints": ("other",)}, {}, "realized values do not match"),
        ({"source_assays": ()}, {}, "source-assay identity"),
        ({"source_assays": ("ATAC",)}, {}, "source assay does not match"),
        ({"cell_selection_fingerprint": ""}, {}, "cell-selection identity"),
        (
            {"cell_selection_fingerprint": "other"},
            {},
            "cell selection does not match",
        ),
        ({"group_fingerprint": ""}, {}, "group-value identity"),
        ({"group_fingerprint": "other"}, {}, "group values do not match"),
        ({"group_order": ()}, {}, "tested group universe"),
        ({"group_order": ("a", "c")}, {}, "group universe does not match"),
        ({"normalization": {}}, {}, "feature-normalization identity"),
        (
            {"normalization": {"source": "raw", "transform": "none"}},
            {},
            "normalization does not match",
        ),
        (
            {"normalization_method": {"name": "other"}},
            {},
            "normalization method",
        ),
        ({"size_factor": 2_000.0}, {}, "size factor"),
    )
    for changes, call_changes, expected in cases:
        assert expected in str(
            _compatibility_issue(_changed(base, **changes), **call_changes)
        )


def test_mapping_feature_helper_validation_and_lightweight_methods() -> None:
    invalid_ids = (
        (np.ones((1, 1), dtype=str), ValueError, "one-dimensional"),
        (np.array([], dtype=str), ValueError, "cannot be empty"),
        (np.array([1, 2]), TypeError, "contain strings"),
        (np.array(["ok", 3], dtype=object), TypeError, "contain strings"),
    )
    for values, error, message in invalid_ids:
        with pytest.raises(error, match=message):
            mapping_features._feature_ids(values, name="feature IDs")

    method = callable_identity(norm_lib_size)
    normalization = {
        "normalization_method": method,
        "size_factor": 1_000.0,
        "log_transform": False,
        "renormalize_subset": True,
    }
    with pytest.raises(ValueError, match="missing"):
        mapping_features._normalization_parameters({})
    with pytest.raises(ValueError, match="Unsupported reference"):
        mapping_features._normalization_parameters({**normalization, "extra": 1})
    with pytest.raises(TypeError, match="must be a boolean"):
        mapping_features._normalization_parameters(
            {**normalization, "log_transform": 1}
        )

    with pytest.raises(TypeError, match="RNA query assays"):
        mapping_features.AlignedFeatureStream(
            object(),  # type: ignore[arg-type]
            np.array([0]),
            np.array(["g"]),
            np.array([0.0]),
            normalization,
            "zero",
            ResourceBudget(1_000, 1),
        )
    assay = object.__new__(RNAassay)
    with pytest.raises(TypeError, match="ResourceBudget"):
        mapping_features.AlignedFeatureStream(
            assay,
            np.array([0]),
            np.array(["g"]),
            np.array([0.0]),
            normalization,
            "zero",
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="fields must be positive"):
        mapping_features.AlignedFeatureStream(
            assay,
            np.array([0]),
            np.array(["g"]),
            np.array([0.0]),
            normalization,
            "zero",
            ResourceBudget(0, 1),
        )
    with pytest.raises(TypeError, match="must be an integer"):
        mapping_features.AlignedFeatureStream(
            assay,
            np.array([0]),
            np.array(["g"]),
            np.array([0.0]),
            normalization,
            "zero",
            ResourceBudget(1_000, 1),
            reserved_resident_bytes=True,
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        mapping_features.AlignedFeatureStream(
            assay,
            np.array([0]),
            np.array(["g"]),
            np.array([0.0]),
            normalization,
            "zero",
            ResourceBudget(1_000, 1),
            reserved_per_row_bytes=-1,
        )

    assay.rawData = SimpleNamespace(shape=(1,), dtype=np.dtype("f8"))
    with pytest.raises(ValueError, match="two-dimensional"):
        mapping_features.AlignedFeatureStream(
            assay,
            np.array([0]),
            np.array(["g"]),
            np.array([0.0]),
            normalization,
            "zero",
            ResourceBudget(1_000, 1),
        )
    assay.rawData = SimpleNamespace(shape=(1, 1), dtype=np.dtype("O"))
    with pytest.raises(TypeError, match="numeric dtype"):
        mapping_features.AlignedFeatureStream(
            assay,
            np.array([0]),
            np.array(["g"]),
            np.array([0.0]),
            normalization,
            "zero",
            ResourceBudget(1_000, 1),
        )
    assay.rawData = SimpleNamespace(shape=(1, 1), dtype=np.dtype("f8"))
    with pytest.raises(ValueError, match="cannot be empty"):
        mapping_features.AlignedFeatureStream(
            assay,
            np.array([], dtype=np.int64),
            np.array(["g"]),
            np.array([0.0]),
            normalization,
            "zero",
            ResourceBudget(1_000, 1),
        )

    stream = object.__new__(mapping_features.AlignedFeatureStream)
    stream._reference_normalized_means = np.array([1.0])
    stream._normalization_parameters = normalization
    stream._missing_feature_policy = "zero"
    assert stream.reference_normalized_means.tolist() == [1.0]
    assert stream.normalization_parameters == normalization
    assert stream.missing_feature_policy == "zero"

    stream._query_cell_indices = np.array([0])
    stream._resident_bytes = 10
    stream._decode_bytes = 0
    stream._resources = ResourceBudget(10, 1)
    stream._source_geometry = None
    with pytest.raises(MemoryError, match="One aligned row"):
        stream._plan_rows(bytes_per_row=1)

    stream._query_cell_indices = np.array([1, 0])
    stream._raw_backing = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(stream._read_raw(0, 2, np.array([1])), [[4.0], [2.0]])


def test_mapping_reference_metadata_is_recursively_frozen_and_thawed() -> None:
    metadata = {
        "method": "pca",
        "ann_metric": "l2",
        "normalization_parameters": {
            "size_factor": 1_000.0,
            "list": [1, {"tuple": (2, 3)}],
            "array": np.array([4, 5]),
        },
    }
    reference = _reference(metadata=metadata)
    frozen_list = reference.metadata["normalization_parameters"]["list"]
    assert frozen_list == [1, {"tuple": (2, 3)}]
    assert frozen_list != "not-a-list"
    assert reference.normalization_parameters == {
        "size_factor": 1_000.0,
        "list": [1, {"tuple": [2, 3]}],
        "array": [4, 5],
    }

    with pytest.raises(TypeError, match="feature IDs"):
        MappingReference(
            **{
                **{
                    field: getattr(reference, field)
                    for field in reference.__dataclass_fields__
                },
                "feature_ids": np.array([1, 2]),
            }
        )

    invalid = _reference(
        metadata={
            "method": "pca",
            "ann_metric": "l2",
            "normalization_parameters": ["invalid"],
        }
    )
    with pytest.raises(TypeError, match="parameters are invalid"):
        _ = invalid.normalization_parameters


def test_mapping_reference_axis_and_layout_validation(monkeypatch) -> None:
    reference = _reference()
    monkeypatch.setattr(
        mapping_reference,
        "validate_stored_selection_integrity",
        lambda *_args, **_kwargs: SimpleNamespace(selected_count=3),
    )
    with pytest.raises(ValueError, match="cell count has changed"):
        reference.validate_frozen_axes()

    monkeypatch.setattr(
        MappingReference,
        "validate_dataset_fingerprint",
        lambda _self: None,
    )
    monkeypatch.setattr(
        mapping_reference,
        "validate_stored_selection_integrity",
        lambda *_args, **_kwargs: SimpleNamespace(selected_count=2),
    )
    monkeypatch.setattr(
        mapping_reference,
        "read_stored_selection_indices",
        lambda *_args, **_kwargs: np.array([0, 1]),
    )
    monkeypatch.setattr(
        mapping_reference,
        "read_metadata_rows_chunkwise",
        lambda *_args, **_kwargs: np.array(["only-one"]),
    )
    monkeypatch.setattr(
        mapping_reference,
        "read_metadata_missing_rows",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="cell count has changed"):
        reference._selected_cell_values("label", validate_binding=False)

    monkeypatch.setattr(MappingReference, "validate_frozen_axes", lambda _self: None)
    monkeypatch.setattr(
        mapping_artifact,
        "validate_mapping_reference_binding",
        lambda value: value,
    )
    layout = _ref("embedding", "9")
    status = SimpleNamespace(complete=False, inputs={})
    monkeypatch.setattr(mapping_reference, "inspect_artifact", lambda *_: status)
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        reference.fetch_layout(layout)

    status.complete = True
    with pytest.raises(ValueError, match="no cell-selection"):
        reference.fetch_layout(layout)
    status.inputs = {"cell_selection": {}}
    with pytest.raises(ValueError, match="invalid cell-selection"):
        reference.fetch_layout(layout)
    status.inputs = {
        "cell_selection": _ref("cell_selection", "0", assay=None).to_dict()
    }
    with pytest.raises(ValueError, match="different cell selections"):
        reference.fetch_layout(layout)

    status.inputs = {"cell_selection": reference.cell_selection.to_dict()}
    payload: dict[str, Any] = {}
    monkeypatch.setattr(mapping_reference, "artifact_group", lambda *_: payload)
    monkeypatch.setattr(mapping_reference, "as_zarr_array", lambda value, **_: value)
    with pytest.raises(ValueError, match="no canonical values"):
        reference.fetch_layout(layout)

    payload["values"] = np.array([["bad", "coordinates"], ["x", "y"]])
    with pytest.raises(TypeError, match="must be numeric"):
        reference.fetch_layout(layout)
    payload["values"] = np.ones((2, 1))
    with pytest.raises(ValueError, match="two columns"):
        reference.fetch_layout(layout)
    payload["values"] = np.array([[0.0, 1.0], [np.inf, 2.0]])
    with pytest.raises(ValueError, match="infinite"):
        reference.fetch_layout(layout)


def test_mapping_artifact_writer_contract_edges(monkeypatch) -> None:
    def create_dataset(
        group: _FakeGroup,
        name: str,
        _chunks: tuple[int, ...],
        dtype: str,
        shape: tuple[int, ...],
    ) -> _FakeArray:
        array = _FakeArray(np.empty(shape, dtype=dtype))
        group[name] = array
        return array

    def create_object_array(group: _FakeGroup, name: str, values: Any) -> None:
        group[name] = _FakeArray(np.asarray(values).astype(str))

    monkeypatch.setattr(mapping_artifact, "create_zarr_dataset", create_dataset)
    monkeypatch.setattr(mapping_artifact, "create_zarr_obj_array", create_object_array)
    monkeypatch.setattr(mapping_artifact, "_payload_fingerprint", lambda *_: "hash")
    monkeypatch.setattr(mapping_artifact, "array_geometry", lambda _array: None)
    model = _model()
    root = _root()
    with pytest.raises(ValueError, match="unsupported method"):
        mapping_artifact.write_artifact_mapping_reference(
            root.create_group("bad_method"),
            model,
            None,
            np.array(["g0", "g1"]),
            {"method": "invalid"},
            np.array([0.0, 1.0]),
            np.array([0.1, 0.2]),
        )

    means = _array(root, "means", np.array([0.0, 1.0]))
    scales = _array(root, "scales", np.array([1.0, 2.0]))
    loadings = _array(root, "loadings", np.eye(2))
    common = {
        "feature_means": means,
        "feature_scales": scales,
        "loadings": loadings,
        "feature_ids": np.array(["g0", "g1"]),
        "reference_distance_quantiles": np.array([0.0, 1.0]),
        "reference_distance_values": np.array([0.1, 0.2]),
    }
    with pytest.raises(ValueError, match="sources do not match"):
        mapping_artifact.write_artifact_mapping_reference_from_sources(
            root.create_group("bad_sources"),
            symphony_sources={"centroids": loadings},
            metadata={"method": "symphony"},
            **common,
        )
    with pytest.raises(ValueError, match="unsupported method"):
        mapping_artifact.write_artifact_mapping_reference_from_sources(
            root.create_group("bad_source_method"),
            symphony_sources=None,
            metadata={"method": "invalid"},
            **common,
        )
    with pytest.raises(ValueError, match="method and Symphony"):
        mapping_artifact.write_artifact_mapping_reference_from_sources(
            root.create_group("missing_symphony"),
            symphony_sources=None,
            metadata={"method": "symphony"},
            **common,
        )

    high_rank = _array(root, "high_rank", np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="one or two axes"):
        mapping_artifact._write_array_from_source(
            root.create_group("high_rank_target"),
            "array",
            high_rank,
        )


def _symphony_sources(
    root: _FakeGroup,
    prefix: str,
    *,
    dtype: np.dtype[Any] = np.dtype("f8"),
    centroid_shape: tuple[int, int] = (2, 2),
) -> dict[str, _FakeArray]:
    n_dims, n_clusters = centroid_shape
    return {
        "centroids": _array(
            root,
            f"{prefix}_centroids",
            np.ones(centroid_shape, dtype=dtype),
        ),
        "raw_centroids": _array(
            root,
            f"{prefix}_raw",
            np.ones((n_clusters, n_dims), dtype=dtype),
        ),
        "corrected_centroids": _array(
            root,
            f"{prefix}_corrected",
            np.ones((n_clusters, n_dims), dtype=dtype),
        ),
        "cluster_mass": _array(
            root,
            f"{prefix}_mass",
            np.ones(n_clusters, dtype=dtype),
        ),
        "sigma": _array(
            root,
            f"{prefix}_sigma",
            np.ones(n_clusters, dtype=dtype),
        ),
    }


def test_mapping_artifact_source_validation_edges(monkeypatch) -> None:
    monkeypatch.setattr(mapping_artifact, "array_geometry", lambda _array: None)
    root = _root()
    means = _array(root, "means", np.zeros(2))
    scales = _array(root, "scales", np.ones(2))
    loadings = _array(root, "loadings", np.eye(2))

    with pytest.raises(ValueError, match="incompatible dimensions"):
        mapping_artifact.validate_mapping_reference_sources(
            feature_means=means,
            feature_scales=scales,
            loadings=means,
            symphony_sources=None,
        )
    with pytest.raises(ValueError, match="sources do not match"):
        mapping_artifact.validate_mapping_reference_sources(
            feature_means=means,
            feature_scales=scales,
            loadings=loadings,
            symphony_sources={"centroids": loadings},
        )
    with pytest.raises(ValueError, match="float64"):
        mapping_artifact.validate_mapping_reference_sources(
            feature_means=means,
            feature_scales=scales,
            loadings=loadings,
            symphony_sources=_symphony_sources(
                root,
                "f32",
                dtype=np.dtype("f4"),
            ),
        )
    with pytest.raises(ValueError, match="dimensions do not match"):
        mapping_artifact.validate_mapping_reference_sources(
            feature_means=means,
            feature_scales=scales,
            loadings=loadings,
            symphony_sources=_symphony_sources(
                root,
                "wrong_dims",
                centroid_shape=(3, 2),
            ),
        )
    invalid = _symphony_sources(root, "invalid")
    invalid["sigma"][0] = 0.0
    with pytest.raises(ValueError, match="arrays are invalid"):
        mapping_artifact.validate_mapping_reference_sources(
            feature_means=means,
            feature_scales=scales,
            loadings=loadings,
            symphony_sources=invalid,
        )

    with pytest.raises(ValueError, match="sources do not match"):
        mapping_artifact.mapping_reference_source_fingerprint(
            feature_means=means,
            feature_scales=scales,
            loadings=loadings,
            symphony_sources={"centroids": loadings},
        )


def test_mapping_artifact_numeric_and_stored_array_helpers(monkeypatch) -> None:
    monkeypatch.setattr(
        mapping_artifact,
        "as_zarr_array",
        lambda value, **_kwargs: value,
    )
    root = _root()
    _array(root, "integer", np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="invalid dtype or shape"):
        mapping_artifact._numeric_payload_array(root, "integer", ndim=1)

    nan_values = _array(root, "nan", np.array([1.0, np.nan]))
    below = _array(root, "below", np.array([-1.0, 1.0]))
    above = _array(root, "above", np.array([1.0, 3.0]))
    decreasing = _array(root, "decreasing", np.array([1.0, 0.0]), chunks=(1,))
    assert not mapping_artifact._numeric_values_are_valid(nan_values)
    assert not mapping_artifact._numeric_values_are_valid(below, minimum=0.0)
    assert not mapping_artifact._numeric_values_are_valid(above, maximum=2.0)
    assert not mapping_artifact._numeric_values_are_valid(
        decreasing,
        nondecreasing=True,
    )

    stored = _array(root, "stored", np.array([1.0, 2.0]))
    assert not mapping_artifact._stored_array_matches_values(stored, np.ones(3))
    assert not mapping_artifact._stored_array_matches_values(
        stored,
        np.array([1, 2], dtype=np.int64),
    )
    assert not mapping_artifact._stored_array_matches_values(
        stored,
        np.array([1.0, 3.0]),
    )

    short = _array(root, "short", np.array([1.0]))
    other = _array(root, "other", np.array([1.0, 3.0]))
    assert not mapping_artifact._stored_array_matches_array(short, stored)
    assert not mapping_artifact._stored_array_matches_array(stored, other)

    monkeypatch.setattr(mapping_artifact, "array_geometry", lambda _array: None)
    assert not mapping_artifact._stored_string_values_are_unique(
        _FakeArray(np.array(["valid", 3], dtype=object))  # type: ignore[arg-type]
    )
    assert not mapping_artifact._stored_string_values_are_unique(
        _FakeArray(np.array(["duplicate", "duplicate"], dtype=object))  # type: ignore[arg-type]
    )


def test_mapping_artifact_payload_matching_and_contract_helpers(monkeypatch) -> None:
    monkeypatch.setattr(
        mapping_artifact,
        "as_zarr_array",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(mapping_artifact, "array_geometry", lambda _array: None)
    root = _root()
    model = _model()
    metadata = {"method": "pca", "nested": {"value": 1}}
    quantiles = np.array([0.0, 1.0])
    distances = np.array([0.1, 0.2])
    group = root.create_group("payload")
    group.attrs["reference_metadata"] = metadata
    group["feature_ids"] = _FakeArray(np.array(["g0", "g1"]))
    group["feature_means"] = _FakeArray(model.feature_means)
    group["feature_scales"] = _FakeArray(model.feature_scales)
    group["loadings"] = _FakeArray(model.loadings)
    group["reference_distance_quantiles"] = _FakeArray(quantiles)
    group["reference_distance_values"] = _FakeArray(distances)
    assert mapping_artifact.mapping_reference_payload_matches_expected(
        group,
        model=model,
        symphony_state=None,
        feature_ids=np.array(["g0", "g1"]),
        metadata=metadata,
        reference_distance_quantiles=quantiles,
        reference_distance_values=distances,
    )
    del group["loadings"]
    assert not mapping_artifact.mapping_reference_payload_matches_expected(
        group,
        model=model,
        symphony_state=None,
        feature_ids=np.array(["g0", "g1"]),
        metadata=metadata,
        reference_distance_quantiles=quantiles,
        reference_distance_values=distances,
    )

    symphony_group = root.create_group("symphony_payload")
    state = _symphony()
    symphony_metadata = {"method": "symphony"}
    symphony_group.attrs["reference_metadata"] = symphony_metadata
    symphony_group["feature_ids"] = _FakeArray(np.array(["g0", "g1"]))
    symphony_group["feature_means"] = _FakeArray(model.feature_means)
    symphony_group["feature_scales"] = _FakeArray(model.feature_scales)
    symphony_group["loadings"] = _FakeArray(model.loadings)
    symphony_group["reference_distance_quantiles"] = _FakeArray(quantiles)
    symphony_group["reference_distance_values"] = _FakeArray(distances)
    for name in mapping_artifact._SYMPHONY_ARRAYS:
        symphony_group[name] = _FakeArray(getattr(state, name))
    assert mapping_artifact.mapping_reference_payload_matches_expected(
        symphony_group,
        model=model,
        symphony_state=state,
        feature_ids=np.array(["g0", "g1"]),
        metadata=symphony_metadata,
        reference_distance_quantiles=quantiles,
        reference_distance_values=distances,
    )

    status = SimpleNamespace(ref=_ref("neighbors", "1"), inputs={})
    with pytest.raises(ValueError, match="missing from the graph chain"):
        mapping_artifact._ref_from_input(status, "coordinates")
    status.inputs = {"coordinates": {}}
    with pytest.raises(ValueError, match="malformed"):
        mapping_artifact._ref_from_input(status, "coordinates")

    names_group = root.create_group("names")
    for name in mapping_artifact._COMMON_ARRAYS:
        _array(names_group, name, np.ones(1))
    names_group.attrs["reference_metadata"] = {"method": "pca"}
    with pytest.raises(ValueError, match="payload fingerprint is missing"):
        mapping_artifact._validate_payload_names(names_group, "pca")
    names_group.attrs["payload_fingerprint"] = "fingerprint"
    names_group.attrs["unexpected"] = True
    with pytest.raises(ValueError, match="attributes do not match"):
        mapping_artifact._validate_payload_names(names_group, "pca")
    with pytest.raises(ValueError, match="metadata 'assay' is missing"):
        mapping_artifact._metadata_string({}, "assay")

    monkeypatch.setattr(
        mapping_artifact,
        "mapping_reference_source_fingerprint",
        lambda **_: "expected",
    )
    assert not mapping_artifact.mapping_reference_payload_matches_sources(
        symphony_group,
        feature_means=symphony_group["feature_means"],
        feature_scales=symphony_group["feature_scales"],
        loadings=symphony_group["loadings"],
        symphony_sources={"centroids": symphony_group["centroids"]},
        feature_ids=np.array(["g0", "g1"]),
        metadata=symphony_metadata,
        reference_distance_quantiles=quantiles,
        reference_distance_values=distances,
        expected_source_fingerprint="expected",
    )
