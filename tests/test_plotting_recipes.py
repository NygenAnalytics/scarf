import json
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import scarf.plotting as plotting
import scarf.plotting.recipes as recipes_module
from scarf.plotting.recipes import (
    ALLOWED_OUTPUT_FORMATS,
    PlotOutputSettings,
    PlotPanelTarget,
    PlotRecipe,
    PlotStep,
    run_plot_recipe,
    run_recipe,
)


class FakePlotResult:
    def __init__(self, name: str, *, owns_figure: bool = True) -> None:
        self.name = name
        self.owns_figure = owns_figure
        self.saved: list[Path] = []
        self.save_kwargs: list[dict[str, object]] = []
        self.show_calls = 0
        self.close_calls = 0
        self.usable = True

    def save(self, path: str | Path, **kwargs: object) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.name, encoding="utf-8")
        self.saved.append(output)
        self.save_kwargs.append(kwargs)
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
    assert result.outputs[0].step_name == "overview"
    assert result.outputs[0].written_path == expected_paths[0]
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
    assert result.save_kwargs == [
        {
            "dpi": 180,
            "transparent": True,
            "exact_size": False,
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


@pytest.mark.parametrize(
    "filename",
    [
        f"nested/plot.{extension.upper()}"
        for extension in sorted(ALLOWED_OUTPUT_FORMATS)
    ],
)
def test_output_settings_accept_supported_formats_case_insensitively(
    filename: str,
) -> None:
    output = PlotOutputSettings(filename)

    assert output.filename == filename


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"filename": 7}, TypeError, "must be a string"),
        ({"filename": " "}, ValueError, "non-empty"),
        ({"filename": "bad\x00.png"}, ValueError, "null byte"),
        ({"filename": "plot.png", "dpi": True}, ValueError, "positive integer"),
        ({"filename": "plot.png", "dpi": 0}, ValueError, "positive integer"),
        ({"filename": "plot.png", "dpi": 1.5}, ValueError, "positive integer"),
        (
            {"filename": "plot.png", "transparent": 1},
            TypeError,
            "flags must be boolean",
        ),
        (
            {"filename": "plot.png", "exact_size": 0},
            TypeError,
            "flags must be boolean",
        ),
    ],
)
def test_output_settings_reject_invalid_serialized_values(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        PlotOutputSettings(**kwargs)


def test_step_rejects_conflicting_sources_and_output_destinations() -> None:
    with pytest.raises(ValueError, match="cannot both define.*color_by"):
        PlotStep(
            name="overlap",
            plot="embedding",
            kwargs={"color_by": "cluster"},
            artifact_kwargs={"color_by": "clusterArtifact"},
        )
    with pytest.raises(ValueError, match="not both"):
        PlotStep(
            name="output",
            plot="embedding",
            output=PlotOutputSettings("plot.png"),
            output_filename="plot.svg",
        )
    with pytest.raises(ValueError, match="filenames must be unique"):
        PlotRecipe(
            [
                PlotStep(
                    name="first",
                    plot="embedding",
                    output_filename="plot.png",
                ),
                PlotStep(
                    name="second",
                    plot="embedding",
                    output_filename="plot.png",
                ),
            ]
        )


def test_step_and_recipe_reject_invalid_python_contracts() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        PlotPanelTarget(" ")
    with pytest.raises(TypeError, match="name must be a string"):
        PlotStep(name=1, plot="embedding")
    with pytest.raises(TypeError, match="plot must be a string"):
        PlotStep(name="step", plot=1)
    with pytest.raises(ValueError, match="Unknown plot"):
        PlotStep(name="step", plot="unknown")
    with pytest.raises(TypeError, match="kwargs.*mapping"):
        PlotStep(name="step", plot="embedding", kwargs=[])
    with pytest.raises(TypeError, match="kwargs keys.*strings"):
        PlotStep(name="step", plot="embedding", kwargs={1: "value"})
    with pytest.raises(TypeError, match="artifact_kwargs.*mapping"):
        PlotStep(name="step", plot="embedding", artifact_kwargs=[])
    with pytest.raises(TypeError, match="artifact_kwargs must map"):
        PlotStep(
            name="step",
            plot="embedding",
            artifact_kwargs={"color_by": 1},
        )
    with pytest.raises(TypeError, match="target must be"):
        PlotStep(name="step", plot="embedding", target="panel")
    with pytest.raises(TypeError, match="output must be"):
        PlotStep(name="step", plot="embedding", output="plot.png")
    with pytest.raises(TypeError, match="ordered sequence"):
        PlotRecipe("step")
    with pytest.raises(TypeError, match="Every recipe step"):
        PlotRecipe([object()])


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        ({}, ValueError, "requires 'steps'"),
        ({"steps": {}}, TypeError, "ordered sequence"),
        ({"steps": ["step"]}, TypeError, "step 0 must be a mapping"),
        (
            {"steps": [{"plot": "embedding"}]},
            ValueError,
            "step 0 requires: name",
        ),
    ],
)
def test_serialized_recipe_requires_ordered_complete_steps(
    config: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        PlotRecipe.from_dict(config)


@pytest.mark.parametrize(
    ("step_fields", "error_type", "message"),
    [
        ({"target": []}, TypeError, "target must be a mapping"),
        ({"target": {"other": "A"}}, ValueError, "Unknown plot step 0 target"),
        ({"target": {}}, ValueError, "target requires panel"),
        ({"output": []}, TypeError, "output must be a mapping"),
        ({"output": {"dpi": 100}}, ValueError, "output requires filename"),
        (
            {
                "output": {"filename": "plot.png"},
                "outputFilename": "plot.svg",
            },
            ValueError,
            "not both",
        ),
        (
            {
                "output": {
                    "filename": "plot.png",
                    "provenanceSidecar": True,
                }
            },
            ValueError,
            "Unknown plot step 0 output",
        ),
        (
            {"artifactKwargs": {"colorBy": 7}},
            TypeError,
            "values must be artifact names",
        ),
        (
            {
                "kwargs": {"colorBy": "cluster"},
                "artifactKwargs": {"colorBy": "clusterArtifact"},
            },
            ValueError,
            "cannot both define",
        ),
    ],
)
def test_serialized_target_output_and_artifact_validation(
    step_fields: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    step = {"name": "overview", "plot": "embedding", **step_fields}

    with pytest.raises(error_type, match=message):
        PlotRecipe.from_dict({"steps": [step]})


def test_serialized_contracts_cover_color_and_grouped_feature_shapes() -> None:
    normalization = plotting.NormalizationSpec(source="raw")
    recipe = PlotRecipe.from_dict(
        {
            "steps": [
                {
                    "name": "multi-color",
                    "plot": "embedding",
                    "kwargs": {
                        "layoutKey": "umap",
                        "colorBy": [
                            {
                                "key": "cluster",
                                "kind": "categorical",
                            },
                            {"value": "CD3D"},
                        ],
                        "normalization": normalization,
                        "categoricalScale": {"order": ["B", "A"]},
                        "densityOverlay": {
                            "levels": 3,
                            "groupBy": "cluster",
                            "groups": ["A"],
                        },
                        "highlight": {"indices": [1, 2]},
                    },
                },
                {
                    "name": "single-color",
                    "plot": "embedding",
                    "kwargs": {
                        "layoutKey": "umap",
                        "colorBy": {"value": "CD4"},
                    },
                },
                {
                    "name": "grouped-features",
                    "plot": "dotplot",
                    "kwargs": {
                        "features": {
                            "T cells": [
                                {"value": "CD3D"},
                                "IL7R",
                            ]
                        },
                        "groupBy": "cluster",
                    },
                },
            ]
        }
    )

    multi_color = recipe.steps[0].kwargs
    assert multi_color["color_by"] == [
        plotting.CellField(key="cluster", kind="categorical"),
        plotting.FeatureRef(value="CD3D"),
    ]
    assert multi_color["normalization"] is normalization
    assert multi_color["categorical_scale"].order == ("B", "A")
    assert multi_color["density_overlay"] == plotting.DensityOverlay(
        levels=3,
        group_by="cluster",
        groups=("A",),
    )
    assert multi_color["highlight"] == plotting.Highlight(indices=(1, 2))
    assert recipe.steps[1].kwargs["color_by"] == plotting.FeatureRef(value="CD4")
    assert recipe.steps[2].kwargs["features"] == {
        "T cells": [
            plotting.FeatureRef(value="CD3D"),
            "IL7R",
        ]
    }


@pytest.mark.parametrize(
    ("color_scale", "error_type", "message"),
    [
        ("magma", TypeError, "color_scale must be a mapping"),
        ({"missing_color": "gray"}, ValueError, "lower camelCase"),
        ({"unknownField": True}, ValueError, "Unknown color_scale"),
        (
            {"quantiles": [0.9, 0.1]},
            ValueError,
            "Invalid color_scale",
        ),
    ],
)
def test_serialized_contracts_reject_invalid_shapes(
    color_scale: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        PlotRecipe.from_dict(
            {
                "steps": [
                    {
                        "name": "overview",
                        "plot": "embedding",
                        "kwargs": {
                            "layoutKey": "umap",
                            "colorScale": color_scale,
                        },
                    }
                ]
            }
        )


def test_recipe_loaders_normalize_suffixes_and_source_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[dict[str, object]] = []

    def embedding(store: object, **kwargs: object) -> FakePlotResult:
        seen.append(kwargs)
        return FakePlotResult("embedding")

    monkeypatch.setattr(plotting, "embedding", embedding)
    json_path = tmp_path / "recipe.JSON"
    json_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": "json",
                        "plot": "embedding",
                        "kwargs": {"layoutKey": "umap"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    toml_source = """
[[steps]]
name = "toml"
plot = "embedding"

[steps.kwargs]
layoutKey = "umap"
""".strip()
    toml_path = tmp_path / "recipe.ToMl"
    toml_path.write_text(toml_source, encoding="utf-8")

    json_recipe = PlotRecipe.from_json(str(json_path))
    toml_recipe = PlotRecipe.from_toml(str(toml_path))
    bytes_recipe = PlotRecipe.from_toml(toml_source.encode())
    json_result = run_recipe(object(), str(json_path))
    toml_result = run_recipe(object(), toml_path)

    assert json_recipe.steps[0].name == "json"
    assert toml_recipe == bytes_recipe
    assert json_result.results[0].name == "embedding"
    assert toml_result.results[0].name == "embedding"
    assert seen == [
        {"layout_key": "umap", "show": False},
        {"layout_key": "umap", "show": False},
    ]


def test_recipe_loader_rejects_invalid_source_types_and_formats(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="PlotRecipe or a JSON/TOML path"):
        run_plot_recipe(object(), object())
    with pytest.raises(ValueError, match=r"\.json or \.toml"):
        run_plot_recipe(object(), tmp_path / "recipe.yaml")
    with pytest.raises(ValueError, match="Invalid JSON constant"):
        PlotRecipe.from_json(
            b'{"steps": [{"name": "bad", "plot": "embedding", '
            b'"kwargs": {"pointSize": NaN}}]}'
        )


def test_output_path_rejects_symlink_escape_before_plotting(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    (output_root / "linked").symlink_to(outside, target_is_directory=True)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="linked/plot.png",
            )
        ]
    )

    with pytest.raises(ValueError, match="escapes output_dir"):
        run_plot_recipe(object(), recipe, output_dir=output_root)


def test_runner_rejects_target_defined_in_kwargs_and_panel_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        plotting,
        "embedding",
        lambda store, **kwargs: pytest.fail("plot should not be called"),
    )
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                kwargs={"target": "inline"},
                target=PlotPanelTarget("A"),
            )
        ]
    )

    with pytest.raises(ValueError, match="defines target in two places"):
        run_plot_recipe(object(), recipe, targets={"A": "external"})


def test_runner_requires_artifact_mapping_before_plotting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        plotting,
        "embedding",
        lambda store, **kwargs: pytest.fail("plot should not be called"),
    )
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                artifact_kwargs={"color_by": "clusterColumn"},
            )
        ]
    )

    with pytest.raises(ValueError, match="requires an artifact mapping"):
        run_plot_recipe(object(), recipe)


@pytest.mark.parametrize(
    ("failure_point", "show", "output_filename", "error_type"),
    [
        ("save", False, "plot.png", OSError),
        ("show", True, None, RuntimeError),
    ],
)
def test_runner_closes_owned_result_after_save_or_show_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
    show: bool,
    output_filename: str | None,
    error_type: type[Exception],
) -> None:
    result = FakePlotResult("embedding")

    def fail_save(path: str | Path, **kwargs: object) -> Path:
        raise OSError("disk full")

    def fail_show() -> None:
        raise RuntimeError("display failed")

    if failure_point == "save":
        monkeypatch.setattr(result, "save", fail_save)
    else:
        monkeypatch.setattr(result, "show", fail_show)
    monkeypatch.setattr(plotting, "embedding", lambda store, **kwargs: result)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename=output_filename,
            )
        ]
    )

    with pytest.raises(error_type):
        run_plot_recipe(
            object(),
            recipe,
            output_dir=tmp_path if output_filename is not None else None,
            show=show,
        )

    assert result.close_calls == 1
    assert not result.usable


def test_runner_preserves_primary_error_when_owned_result_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = FakePlotResult("embedding")

    def fail_save(path: str | Path, **kwargs: object) -> Path:
        raise OSError("disk full")

    def fail_close() -> None:
        result.close_calls += 1
        raise RuntimeError("close failed")

    monkeypatch.setattr(result, "save", fail_save)
    monkeypatch.setattr(result, "close", fail_close)
    monkeypatch.setattr(plotting, "embedding", lambda store, **kwargs: result)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="plot.png",
            )
        ]
    )

    with pytest.raises(OSError, match="disk full") as error:
        run_plot_recipe(object(), recipe, output_dir=tmp_path)

    assert result.close_calls == 1
    assert error.value.__notes__ == [
        "The plot also failed to close cleanly: RuntimeError: close failed"
    ]


def test_runner_does_not_close_caller_owned_result_after_save_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = FakePlotResult("embedding", owns_figure=False)

    def fail_save(path: str | Path, **kwargs: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(result, "save", fail_save)
    monkeypatch.setattr(plotting, "embedding", lambda store, **kwargs: result)
    recipe = PlotRecipe(
        [
            PlotStep(
                name="overview",
                plot="embedding",
                output_filename="plot.png",
            )
        ]
    )

    with pytest.raises(OSError, match="disk full"):
        run_plot_recipe(object(), recipe, output_dir=tmp_path)

    assert result.close_calls == 0
    assert result.usable


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("show", 1, "show must be a boolean"),
        ("continue_on_error", 1, "continue_on_error must be a boolean"),
        ("artifacts", [], "artifacts must be a mapping"),
        ("targets", [], "targets must be a mapping"),
    ],
)
def test_runner_validates_execution_context(
    argument: str,
    value: object,
    message: str,
) -> None:
    recipe = PlotRecipe([PlotStep(name="overview", plot="embedding")])

    with pytest.raises(TypeError, match=message):
        run_plot_recipe(object(), recipe, **{argument: value})


def test_runner_reports_missing_or_noncallable_plot_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = PlotRecipe([PlotStep(name="overview", plot="embedding")])
    missing_module = object()
    monkeypatch.setattr(recipes_module, "import_module", lambda name: missing_module)

    with pytest.raises(RuntimeError, match="not available"):
        run_plot_recipe(object(), recipe)

    class NonCallablePlotting:
        embedding = object()

    monkeypatch.setattr(
        recipes_module,
        "import_module",
        lambda name: NonCallablePlotting,
    )
    with pytest.raises(RuntimeError, match="not callable"):
        run_plot_recipe(object(), recipe)
