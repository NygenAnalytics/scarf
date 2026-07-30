import json
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import scarf.plotting as plotting
from scarf.plotting.recipes import (
    PlotOutputSettings,
    PlotPanelTarget,
    PlotRecipe,
    PlotStep,
    run_plot_recipe,
)


class FakePlotResult:
    def __init__(self, name: str, *, owns_figure: bool = True) -> None:
        self.name = name
        self.owns_figure = owns_figure
        self.saved: list[Path] = []
        self.show_calls = 0
        self.close_calls = 0
        self.usable = True

    def save(self, path: str | Path, **kwargs: object) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.name, encoding="utf-8")
        self.saved.append(output)
        return output

    def show(self) -> None:
        self.show_calls += 1

    def close(self) -> None:
        self.close_calls += 1
        self.usable = False


def test_runner_preserves_order_suppresses_plot_show_and_manages_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = object()
    calls: list[tuple[str, object, dict[str, object]]] = []
    plot_results: dict[str, FakePlotResult] = {}

    def fake_plot(name: str):
        def plot(store_arg: object, **kwargs: object) -> FakePlotResult:
            calls.append((name, store_arg, kwargs))
            result = FakePlotResult(name)
            plot_results[name] = result
            return result

        return plot

    monkeypatch.setattr(plotting, "embedding", fake_plot("embedding"))
    monkeypatch.setattr(plotting, "dotplot", fake_plot("dotplot"))
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                kwargs={"layout_key": "umap", "show": True},
                output_filename="figures/overview.png",
            ),
            PlotStep(
                name="markers",
                plot="dotplot",
                kwargs={"features": ["A"], "group_by": "cluster"},
                output_filename="figures/markers.svg",
            ),
        ]
    )

    result = run_plot_recipe(store, recipe, output_dir=tmp_path, show=True)

    assert [name for name, _, _ in calls] == ["embedding", "dotplot"]
    assert all(store_arg is store for _, store_arg, _ in calls)
    assert all(kwargs["show"] is False for _, _, kwargs in calls)
    assert calls[0][2]["layout_key"] == "umap"
    expected_paths = (
        tmp_path / "figures/overview.png",
        tmp_path / "figures/markers.svg",
    )
    assert result.written_paths == expected_paths
    assert [output.name for output in result.outputs] == ["overview", "markers"]
    assert result.results == (
        plot_results["embedding"],
        plot_results["dotplot"],
    )
    assert all(path.exists() for path in result.written_paths)
    for name, expected_path in zip(("embedding", "dotplot"), expected_paths):
        plot_result = plot_results[name]
        assert plot_result.saved == [expected_path]
        assert plot_result.show_calls == 1
        assert plot_result.close_calls == 1
    assert result.failures == {}


def test_no_output_mode_leaves_returned_owned_figure_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plot_result = FakePlotResult("embedding")

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        return plot_result

    monkeypatch.setattr(plotting, "embedding", embedding)
    recipe = PlotRecipe(
        [PlotStep(name="overview", plot="embedding", kwargs={"layout_key": "umap"})]
    )

    result = run_plot_recipe(object(), recipe)

    assert result.results == (plot_result,)
    assert result.written_paths == ()
    assert plot_result.saved == []
    assert plot_result.show_calls == 0
    assert plot_result.close_calls == 0
    assert plot_result.usable


def test_runner_does_not_close_caller_owned_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plot_result = FakePlotResult("embedding", owns_figure=False)
    monkeypatch.setattr(
        plotting,
        "embedding",
        lambda store, **kwargs: plot_result,
    )
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="overview.png",
            )
        ]
    )

    run_plot_recipe(object(), recipe, output_dir=tmp_path, show=True)

    assert plot_result.saved == [tmp_path / "overview.png"]
    assert plot_result.show_calls == 1
    assert plot_result.close_calls == 0
    assert plot_result.usable


def test_continue_on_error_captures_compact_failures_and_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    successful = FakePlotResult("dotplot")

    def failing(store: object, **kwargs: object) -> FakePlotResult:
        calls.append("bad")
        warnings.warn("first warning", UserWarning)
        raise RuntimeError("plot\nfailed")

    def succeeding(store: object, **kwargs: object) -> FakePlotResult:
        calls.append("good")
        warnings.warn("second warning", RuntimeWarning)
        return successful

    monkeypatch.setattr(plotting, "embedding", failing)
    monkeypatch.setattr(plotting, "dotplot", succeeding)
    recipe = PlotRecipe(
        [
            PlotStep(name="bad", plot="embedding"),
            PlotStep(name="good", plot="dotplot"),
        ]
    )

    with pytest.raises(RuntimeError, match="plot"):
        run_plot_recipe(object(), recipe)
    assert calls == ["bad"]

    calls.clear()
    result = run_plot_recipe(object(), recipe, continue_on_error=True)

    assert calls == ["bad", "good"]
    assert result.failures == {"bad": "RuntimeError: plot failed"}
    assert result.results == (successful,)
    assert result.warnings == (
        "bad: first warning",
        "good: second warning",
    )
    assert successful.usable


def test_json_and_toml_loaders_have_matching_camel_case_schema(
    tmp_path: Path,
) -> None:
    config = {
        "steps": [
            {
                "name": "overview",
                "plot": "embedding",
                "kwargs": {
                    "layoutKey": "umap",
                    "colorBy": "cluster",
                },
                "outputFilename": "figures/overview.png",
            },
            {
                "name": "composition",
                "plot": "composition",
                "kwargs": {"categoryBy": "cluster"},
            },
        ]
    }
    json_path = tmp_path / "recipe.json"
    json_path.write_text(json.dumps(config), encoding="utf-8")
    toml_path = tmp_path / "recipe.toml"
    toml_path.write_text(
        """
[[steps]]
name = "overview"
plot = "embedding"
outputFilename = "figures/overview.png"

[steps.kwargs]
layoutKey = "umap"
colorBy = "cluster"

[[steps]]
name = "composition"
plot = "composition"

[steps.kwargs]
categoryBy = "cluster"
""".strip(),
        encoding="utf-8",
    )

    from_dict = PlotRecipe.from_dict(config)
    from_json = PlotRecipe.from_json(json_path)
    from_toml = PlotRecipe.from_toml(toml_path)

    assert from_dict == from_json == from_toml
    assert from_json.steps[0].output_filename == "figures/overview.png"
    assert dict(from_toml.steps[1].kwargs) == {"category_by": "cluster"}


def test_runner_resolves_json_config_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[dict[str, object]] = []
    plot_result = FakePlotResult("embedding")

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        seen.append(kwargs)
        return plot_result

    monkeypatch.setattr(plotting, "embedding", embedding)
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "kwargs": {"layoutKey": "umap"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_plot_recipe(object(), path)

    assert seen == [{"layout_key": "umap", "show": False}]
    assert result.results == (plot_result,)


@pytest.mark.parametrize(
    "filename",
    [
        "/tmp/plot.png",
        "../plot.png",
        "nested/../../plot.png",
        r"C:\plots\plot.png",
    ],
)
def test_output_filename_rejects_absolute_and_traversal_paths(filename: str) -> None:
    with pytest.raises(ValueError, match="relative|traversal"):
        PlotStep(
            name="overview",
            plot="embedding",
            output_filename=filename,
        )


def test_output_filename_requires_supported_format_and_output_directory() -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        PlotStep(
            name="overview",
            plot="embedding",
            output_filename="overview.txt",
        )

    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="overview.png",
            )
        ]
    )
    with pytest.raises(ValueError, match="output_dir"):
        run_plot_recipe(object(), recipe)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"steps": [], "extra": True}, "Unknown plot recipe"),
        (
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "output_filename": "overview.png",
                    }
                ]
            },
            "Unknown plot step",
        ),
        (
            {"steps": [{"name": "overview", "plot": "embedding", "kwargs": []}]},
            "mapping",
        ),
        ({"steps": [{"name": "overview", "plot": "unknown"}]}, "Unknown plot"),
    ],
)
def test_from_dict_rejects_unknown_keys_and_invalid_values(
    config: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        PlotRecipe.from_dict(config)


def test_recipe_validates_non_empty_unique_names_and_is_frozen() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PlotRecipe([])
    with pytest.raises(ValueError, match="non-empty"):
        PlotStep(name=" ", plot="embedding")

    step = PlotStep(name="overview", plot="embedding")
    with pytest.raises(ValueError, match="unique"):
        PlotRecipe([step, step])
    with pytest.raises(FrozenInstanceError):
        step.name = "changed"
    assert not hasattr(step, "__dict__")


def test_external_kwargs_require_camel_case_and_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="camelCase"):
        PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "kwargs": {"layout_key": "umap"},
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="Unknown kwargs"):
        PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "kwargs": {
                            "layoutKey": "umap",
                            "notAPlotArgument": True,
                        },
                    }
                ]
            }
        )


def test_external_recipe_coerces_plot_contracts_and_feature_refs() -> None:
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "markers",
                    "plot": "dotplot",
                    "kwargs": {
                        "features": [
                            {
                                "value": "CD3D",
                                "assay": "RNA",
                                "by": "name",
                            }
                        ],
                        "groupBy": "cluster",
                        "colorScale": {
                            "cmap": "magma",
                            "quantiles": [0.01, 0.99],
                            "missingColor": "#cccccc",
                        },
                        "sizeScale": {
                            "sizeMin": 5,
                            "sizeMax": 80,
                        },
                    },
                }
            ]
        }
    )

    kwargs = recipe.steps[0].kwargs
    assert kwargs["features"] == [
        plotting.FeatureRef(value="CD3D", assay="RNA", by="name")
    ]
    assert kwargs["color_scale"] == plotting.ColorScale(
        cmap="magma",
        quantiles=(0.01, 0.99),
        missing_color="#cccccc",
    )
    assert kwargs["size_scale"] == plotting.SizeScale(size_min=5, size_max=80)


def test_runner_injects_pipeline_artifacts_without_mutating_recipe_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []
    plot_result = FakePlotResult("embedding")

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        seen.append(kwargs)
        return plot_result

    monkeypatch.setattr(plotting, "embedding", embedding)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                kwargs={"layout_key": "umap"},
                artifact_kwargs={"color_by": "clusterColumn"},
            )
        ]
    )

    result = run_plot_recipe(
        object(),
        recipe,
        artifacts={"clusterColumn": "RNA_leiden_cluster"},
    )

    assert result.results == (plot_result,)
    assert seen == [
        {
            "layout_key": "umap",
            "color_by": "RNA_leiden_cluster",
            "show": False,
        }
    ]
    assert dict(recipe.steps[0].kwargs) == {"layout_key": "umap"}
    with pytest.raises(KeyError, match="clusterColumn"):
        run_plot_recipe(object(), recipe, artifacts={})


def test_recipe_resolves_optional_panel_target_and_output_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[dict[str, object]] = []
    result = FakePlotResult("embedding", owns_figure=False)

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        seen.append(kwargs)
        return result

    monkeypatch.setattr(plotting, "embedding", embedding)
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "overview",
                    "plot": "embedding",
                    "kwargs": {"layoutKey": "umap"},
                    "target": {"panel": "A"},
                    "output": {
                        "filename": "overview.png",
                        "dpi": 180,
                        "transparent": True,
                        "exactSize": False,
                    },
                }
            ]
        }
    )

    assert recipe.steps[0].target == PlotPanelTarget("A")
    assert recipe.steps[0].output == PlotOutputSettings(
        "overview.png",
        dpi=180,
        transparent=True,
        exact_size=False,
    )
    run_plot_recipe(
        object(),
        recipe,
        targets={"A": "axes-a"},
        output_dir=tmp_path,
    )

    assert seen == [
        {
            "layout_key": "umap",
            "target": "axes-a",
            "show": False,
        }
    ]
    with pytest.raises(KeyError, match="requires panel"):
        run_plot_recipe(object(), recipe, targets={}, output_dir=tmp_path)


def test_json_loader_rejects_duplicate_keys_and_non_object_roots() -> None:
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        PlotRecipe.from_json(
            '{"steps": [{"name": "one", "name": "two", "plot": "embedding"}]}'
        )
    with pytest.raises(TypeError, match="mapping"):
        PlotRecipe.from_json("[]")
