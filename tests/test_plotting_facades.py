"""Delegation tests for specialized plotting facades."""

import numpy as np
import pandas as pd

import scarf.plots as legacy_plots
import scarf.plotting as splt


def test_diagnostic_facades_forward_public_arguments(monkeypatch):
    calls = {}

    def plot_qc(**kwargs):
        calls["qc"] = kwargs
        return "qc-result"

    def plot_elbow(values, *, figsize):
        calls["elbow"] = (values, figsize)

    def plot_graph_qc(graph):
        calls["graph"] = graph

    def plot_mean_var(**kwargs):
        calls["hvg"] = kwargs

    monkeypatch.setattr(legacy_plots, "plot_qc", plot_qc)
    monkeypatch.setattr(legacy_plots, "plot_elbow", plot_elbow)
    monkeypatch.setattr(legacy_plots, "plot_graph_qc", plot_graph_qc)
    monkeypatch.setattr(legacy_plots, "plot_mean_var", plot_mean_var)

    frame = pd.DataFrame({"groups": ["a"], "metric": [1.0]})
    assert splt.qc(frame, figsize=(3, 2), show=False) == "qc-result"
    splt.elbow([0.5, 0.3], figsize=(4, 2))
    graph = object()
    splt.graph_qc(graph)
    splt.highly_variable_features(
        np.array([1.0]),
        np.array([2.0]),
        np.array([3]),
        np.array([True]),
        point_sizes=(4, 20),
    )

    assert calls["qc"]["fig_size"] == (3, 2)
    assert calls["qc"]["show_fig"] is False
    assert calls["elbow"] == ([0.5, 0.3], (4, 2))
    assert calls["graph"] is graph
    assert calls["hvg"]["ss"] == (4, 20)


class _FacadeStore:
    def __init__(self):
        self.calls = {}

    def plot_marker_heatmap(self, **kwargs):
        self.calls["marker"] = kwargs

    def plot_cluster_tree(self, **kwargs):
        self.calls["tree"] = kwargs

    def plot_pseudotime_heatmap(self, **kwargs):
        self.calls["pseudotime"] = kwargs


def test_heatmap_facades_forward_arguments():
    store = _FacadeStore()
    splt.marker_heatmap(
        store,
        group_key="cluster",
        topn=7,
        show_fig=False,
        figsize=(4, 6),
    )
    target = object()
    splt.cluster_tree(
        store,
        cluster_key="cluster",
        color_key={"a": "red"},
        ax=target,
        show_fig=False,
    )
    splt.pseudotime_heatmap(
        store,
        pseudotime_key="trajectory",
        show_features=["GeneA"],
        show_fig=False,
    )

    assert store.calls["marker"]["group_key"] == "cluster"
    assert store.calls["marker"]["topn"] == 7
    assert store.calls["marker"]["figsize"] == (4, 6)
    assert store.calls["marker"]["show_fig"] is False
    assert store.calls["tree"]["ax"] is target
    assert store.calls["tree"]["color_key"] == {"a": "red"}
    assert store.calls["pseudotime"]["pseudotime_key"] == "trajectory"
    assert store.calls["pseudotime"]["show_features"] == ["GeneA"]
