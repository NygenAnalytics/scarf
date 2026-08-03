"""Declarative plotting recipes and their sequential runner."""

import json
import inspect
import re
import tomllib
import warnings as warning_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from importlib import import_module
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from ..utils.logging import logger
from ._contracts import (
    CategoricalScale,
    CellField,
    ColorScale,
    DensityOverlay,
    FeatureRef,
    Highlight,
    NormalizationSpec,
    SizeScale,
    StudyDesign,
)


ALLOWED_PLOTS = frozenset(
    {
        "cluster_connectivity",
        "cluster_tree",
        "composition",
        "distribution",
        "dotplot",
        "embedding",
        "embedding_raster",
        "marker_heatmap",
        "mapping_calibration",
        "mapping_confusion",
        "mapping_evidence",
        "mapping_score",
        "matrixplot",
        "pseudotime_heatmap",
    }
)
ALLOWED_OUTPUT_FORMATS = frozenset({"pdf", "png", "svg", "tif", "tiff"})

_RECIPE_KEYS = frozenset({"steps"})
_STEP_KEYS = frozenset(
    {
        "name",
        "plot",
        "kwargs",
        "artifactKwargs",
        "target",
        "output",
        "outputFilename",
    }
)
_TARGET_KEYS = frozenset({"panel"})
_OUTPUT_KEYS = frozenset({"filename", "dpi", "transparent", "exactSize"})
_LOWER_CAMEL_KEY = re.compile(r"^[a-z][A-Za-z0-9]*$")


def _validate_keys(
    value: Mapping[Any, Any],
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    unknown = [key for key in value if not isinstance(key, str) or key not in allowed]
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise ValueError(f"Unknown {context} key(s): {names}")


def _validate_output_filename(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("outputFilename must be a string")
    if not value.strip():
        raise ValueError("outputFilename must be non-empty")
    if "\x00" in value:
        raise ValueError("outputFilename must not contain a null byte")

    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise ValueError("outputFilename must be a relative path")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise ValueError("outputFilename must not contain parent traversal")

    suffix = Path(value).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_OUTPUT_FORMATS:
        formats = ", ".join(sorted(ALLOWED_OUTPUT_FORMATS))
        raise ValueError(
            f"Unsupported output format {Path(value).suffix!r}; choose from {formats}"
        )


def _camel_to_snake(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _external_kwargs(
    plot_name: str,
    value: Mapping[Any, Any],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(plot_name, str) or plot_name not in ALLOWED_PLOTS:
        choices = ", ".join(sorted(ALLOWED_PLOTS))
        raise ValueError(f"Unknown plot {plot_name!r}; choose from {choices}")
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    converted: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or _LOWER_CAMEL_KEY.fullmatch(key) is None:
            raise ValueError(f"{context} keys must use lower camelCase")
        python_key = _camel_to_snake(key)
        if python_key in converted:
            raise ValueError(f"{context} contains duplicate key {python_key!r}")
        converted[python_key] = item

    plot = _resolve_plot(plot_name)
    signature = inspect.signature(plot)
    accepts_extra = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_extra:
        allowed = set(signature.parameters) - {"store"}
        unknown = sorted(set(converted) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown kwargs for plot {plot_name!r}: "
                + ", ".join(map(repr, unknown))
            )
    return converted


def _contract_from_mapping(
    contract: type[Any],
    value: Any,
    *,
    context: str,
) -> Any:
    if isinstance(value, contract):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    converted: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or _LOWER_CAMEL_KEY.fullmatch(key) is None:
            raise ValueError(f"{context} keys must use lower camelCase")
        converted[_camel_to_snake(key)] = item
    allowed = {item.name for item in fields(contract)}
    unknown = sorted(set(converted) - allowed)
    if unknown:
        raise ValueError(f"Unknown {context} key(s): " + ", ".join(map(repr, unknown)))
    tuple_fields: dict[type[Any], tuple[str, ...]] = {
        CategoricalScale: ("order",),
        ColorScale: ("quantiles",),
        DensityOverlay: ("levels", "groups"),
        Highlight: ("groups", "indices"),
    }
    for name in tuple_fields.get(contract, ()):
        item = converted.get(name)
        if item is not None and not isinstance(item, int):
            converted[name] = tuple(item)
    try:
        return contract(**converted)
    except (TypeError, ValueError) as error:
        raise type(error)(f"Invalid {context}: {error}") from error


def _coerce_feature_reference(value: Any, *, context: str) -> Any:
    if isinstance(value, Mapping):
        return _contract_from_mapping(FeatureRef, value, context=context)
    return value


def _coerce_serialized_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    contract_kwargs: dict[str, type[Any]] = {
        "categorical_scale": CategoricalScale,
        "color_scale": ColorScale,
        "density_overlay": DensityOverlay,
        "highlight": Highlight,
        "normalization": NormalizationSpec,
        "size_scale": SizeScale,
        "study_design": StudyDesign,
    }
    for name, contract in contract_kwargs.items():
        if name in out and out[name] is not None:
            out[name] = _contract_from_mapping(
                contract,
                out[name],
                context=name,
            )
    if "color_by" in out:
        color_by = out["color_by"]

        def coerce_color(value: Any) -> Any:
            if not isinstance(value, Mapping):
                return value
            if "key" in value:
                return _contract_from_mapping(CellField, value, context="colorBy")
            return _contract_from_mapping(FeatureRef, value, context="colorBy")

        if isinstance(color_by, Sequence) and not isinstance(color_by, str | bytes):
            out["color_by"] = [coerce_color(value) for value in color_by]
        else:
            out["color_by"] = coerce_color(color_by)
    if "features" in out:
        feature_value = out["features"]
        if isinstance(feature_value, Mapping):
            out["features"] = {
                group: [
                    _coerce_feature_reference(
                        value,
                        context=f"features.{group}",
                    )
                    for value in values
                ]
                for group, values in feature_value.items()
            }
        elif isinstance(feature_value, Sequence) and not isinstance(
            feature_value,
            str | bytes,
        ):
            out["features"] = [
                _coerce_feature_reference(value, context="features")
                for value in feature_value
            ]
    return out


@dataclass(frozen=True, slots=True)
class PlotPanelTarget:
    """Reference an axes supplied to the recipe runner by panel name."""

    panel: str

    def __post_init__(self) -> None:
        if not isinstance(self.panel, str) or not self.panel.strip():
            raise ValueError("Plot panel target must be a non-empty string")


@dataclass(frozen=True, slots=True)
class PlotOutputSettings:
    """Per-step export settings resolved below the runner output directory."""

    filename: str
    dpi: int | None = None
    transparent: bool = False
    exact_size: bool = True

    def __post_init__(self) -> None:
        _validate_output_filename(self.filename)
        if self.dpi is not None and (
            isinstance(self.dpi, bool) or not isinstance(self.dpi, int) or self.dpi <= 0
        ):
            raise ValueError("Plot output dpi must be a positive integer or None")
        if not isinstance(self.transparent, bool) or not isinstance(
            self.exact_size,
            bool,
        ):
            raise TypeError("Plot output flags must be boolean")


@dataclass(frozen=True, slots=True)
class PlotStep:
    """One named plotting call in a recipe."""

    name: str
    plot: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    artifact_kwargs: Mapping[str, str] = field(default_factory=dict)
    target: PlotPanelTarget | None = None
    output: PlotOutputSettings | None = None
    output_filename: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Plot step name must be a string")
        if not self.name.strip():
            raise ValueError("Plot step name must be non-empty")
        if not isinstance(self.plot, str):
            raise TypeError("Plot step plot must be a string")
        if self.plot not in ALLOWED_PLOTS:
            choices = ", ".join(sorted(ALLOWED_PLOTS))
            raise ValueError(f"Unknown plot {self.plot!r}; choose from {choices}")
        if not isinstance(self.kwargs, Mapping):
            raise TypeError(f"kwargs for step {self.name!r} must be a mapping")
        if any(not isinstance(key, str) for key in self.kwargs):
            raise TypeError(f"kwargs keys for step {self.name!r} must be strings")
        object.__setattr__(
            self,
            "kwargs",
            MappingProxyType(dict(self.kwargs)),
        )
        if not isinstance(self.artifact_kwargs, Mapping):
            raise TypeError(f"artifact_kwargs for step {self.name!r} must be a mapping")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or not value
            for key, value in self.artifact_kwargs.items()
        ):
            raise TypeError(
                "artifact_kwargs must map non-empty plot keyword names "
                "to non-empty artifact names"
            )
        overlap = set(self.kwargs).intersection(self.artifact_kwargs)
        if overlap:
            raise ValueError(
                "kwargs and artifact_kwargs cannot both define: "
                + ", ".join(sorted(overlap))
            )
        object.__setattr__(
            self,
            "artifact_kwargs",
            MappingProxyType(dict(self.artifact_kwargs)),
        )
        if self.target is not None and not isinstance(self.target, PlotPanelTarget):
            raise TypeError("target must be a PlotPanelTarget or None")
        if self.output is not None and not isinstance(
            self.output,
            PlotOutputSettings,
        ):
            raise TypeError("output must be PlotOutputSettings or None")
        if self.output is not None and self.output_filename is not None:
            raise ValueError("Set output or output_filename, not both")
        if self.output_filename is not None:
            _validate_output_filename(self.output_filename)
            object.__setattr__(
                self,
                "output",
                PlotOutputSettings(filename=self.output_filename),
            )
        elif self.output is not None:
            object.__setattr__(self, "output_filename", self.output.filename)


@dataclass(frozen=True, slots=True)
class PlotRecipe:
    """An ordered collection of plotting steps."""

    steps: tuple[PlotStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, Sequence) or isinstance(
            self.steps, (str, bytes, bytearray)
        ):
            raise TypeError("Plot recipe steps must be an ordered sequence")
        steps = tuple(self.steps)
        if any(not isinstance(step, PlotStep) for step in steps):
            raise TypeError("Every recipe step must be a PlotStep")
        if not steps:
            raise ValueError("Plot recipe must contain at least one step")
        names = [step.name for step in steps]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            joined = ", ".join(repr(name) for name in duplicates)
            raise ValueError(f"Plot step names must be unique; duplicates: {joined}")
        output_names = [
            step.output_filename for step in steps if step.output_filename is not None
        ]
        duplicate_outputs = sorted(
            {name for name in output_names if output_names.count(name) > 1}
        )
        if duplicate_outputs:
            raise ValueError(
                "Plot output filenames must be unique; duplicates: "
                + ", ".join(map(repr, duplicate_outputs))
            )
        object.__setattr__(self, "steps", steps)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlotRecipe":
        if not isinstance(value, Mapping):
            raise TypeError("Plot recipe configuration must be a mapping")
        _validate_keys(value, _RECIPE_KEYS, context="plot recipe")
        if "steps" not in value:
            raise ValueError("Plot recipe configuration requires 'steps'")

        raw_steps = value["steps"]
        if not isinstance(raw_steps, Sequence) or isinstance(
            raw_steps, (str, bytes, bytearray)
        ):
            raise TypeError("Plot recipe 'steps' must be an ordered sequence")

        steps: list[PlotStep] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise TypeError(f"Plot recipe step {index} must be a mapping")
            _validate_keys(raw_step, _STEP_KEYS, context=f"plot step {index}")
            missing = {"name", "plot"}.difference(raw_step)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"Plot recipe step {index} requires: {names}")
            plot_name = raw_step["plot"]
            kwargs = _coerce_serialized_kwargs(
                _external_kwargs(
                    plot_name,
                    raw_step.get("kwargs", {}),
                    context=f"kwargs for plot step {index}",
                )
            )
            artifact_kwargs = _external_kwargs(
                plot_name,
                raw_step.get("artifactKwargs", {}),
                context=f"artifactKwargs for plot step {index}",
            )
            if any(not isinstance(value, str) for value in artifact_kwargs.values()):
                raise TypeError("artifactKwargs values must be artifact names")
            target_value = raw_step.get("target")
            target = None
            if target_value is not None:
                if not isinstance(target_value, Mapping):
                    raise TypeError("target must be a mapping")
                _validate_keys(
                    target_value,
                    _TARGET_KEYS,
                    context=f"plot step {index} target",
                )
                if "panel" not in target_value:
                    raise ValueError("target requires panel")
                target = PlotPanelTarget(panel=target_value["panel"])
            output_value = raw_step.get("output")
            output_filename = raw_step.get("outputFilename")
            if output_value is not None and output_filename is not None:
                raise ValueError("Set output or outputFilename, not both")
            output = None
            if output_value is not None:
                if not isinstance(output_value, Mapping):
                    raise TypeError("output must be a mapping")
                _validate_keys(
                    output_value,
                    _OUTPUT_KEYS,
                    context=f"plot step {index} output",
                )
                if "filename" not in output_value:
                    raise ValueError("output requires filename")
                output = PlotOutputSettings(
                    filename=output_value["filename"],
                    dpi=output_value.get("dpi"),
                    transparent=output_value.get("transparent", False),
                    exact_size=output_value.get("exactSize", True),
                )
            steps.append(
                PlotStep(
                    name=raw_step["name"],
                    plot=plot_name,
                    kwargs=kwargs,
                    artifact_kwargs=artifact_kwargs,
                    target=target,
                    output=output,
                    output_filename=output_filename,
                )
            )
        return cls(tuple(steps))

    @classmethod
    def from_json(
        cls,
        source: str | bytes | bytearray | Path,
    ) -> "PlotRecipe":
        raw = _read_json_source(source)
        value = json.loads(
            raw,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        return cls.from_dict(value)

    @classmethod
    def from_toml(cls, source: str | bytes | Path) -> "PlotRecipe":
        if isinstance(source, Path):
            with source.open("rb") as stream:
                value = tomllib.load(stream)
        else:
            raw = source.decode("utf-8") if isinstance(source, bytes) else source
            path = _existing_config_path(raw)
            if path is not None:
                with path.open("rb") as stream:
                    value = tomllib.load(stream)
            else:
                value = tomllib.loads(raw)
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class PlotOutput:
    """The successful result of one recipe step."""

    name: str
    result: Any = field(repr=False)
    path: Path | None = None

    @property
    def step_name(self) -> str:
        return self.name

    @property
    def written_path(self) -> Path | None:
        return self.path


@dataclass(frozen=True, slots=True)
class PlotRecipeResult:
    """Results and diagnostics from a recipe execution."""

    outputs: tuple[PlotOutput, ...] = ()
    written_paths: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    failures: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "written_paths", tuple(self.written_paths))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self,
            "failures",
            MappingProxyType(dict(self.failures)),
        )

    @property
    def results(self) -> tuple[Any, ...]:
        return tuple(output.result for output in self.outputs)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {value}")


def _existing_config_path(value: str) -> Path | None:
    if "\n" in value or "\r" in value:
        return None
    try:
        path = Path(value)
        return path if path.is_file() else None
    except OSError:
        return None


def _read_json_source(
    source: str | bytes | bytearray | Path,
) -> str | bytes | bytearray:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if isinstance(source, str):
        path = _existing_config_path(source)
        if path is not None:
            return path.read_text(encoding="utf-8")
    return source


def _load_recipe(recipe: PlotRecipe | str | Path) -> PlotRecipe:
    if isinstance(recipe, PlotRecipe):
        return recipe
    if not isinstance(recipe, (str, Path)):
        raise TypeError("recipe must be a PlotRecipe or a JSON/TOML path")
    path = Path(recipe)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return PlotRecipe.from_json(path)
    if suffix == ".toml":
        return PlotRecipe.from_toml(path)
    raise ValueError("Recipe path must use a .json or .toml extension")


def _resolve_plot(name: str) -> Any:
    plotting = import_module("scarf.plotting")
    try:
        plot = getattr(plotting, name)
    except AttributeError as error:
        raise RuntimeError(
            f"Plot {name!r} is not available in this Scarf build"
        ) from error
    if not callable(plot):
        raise RuntimeError(f"Plot {name!r} is not callable")
    return plot


def _safe_output_path(output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    root = output_dir.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Output path for {filename!r} escapes output_dir")
    return path


def _compact_failure(error: Exception) -> str:
    detail = " ".join(str(error).split())
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _log_failure(step_name: str, failure: str) -> None:
    logger.exception(f"Plot recipe step {step_name!r} failed: {failure}")


def _run_step(
    store: Any,
    step: PlotStep,
    *,
    output_path: Path | None,
    show: bool,
    artifacts: Mapping[str, Any] | None,
    targets: Mapping[str, Any] | None,
) -> PlotOutput:
    plot = _resolve_plot(step.plot)
    kwargs = dict(step.kwargs)
    if step.artifact_kwargs:
        if artifacts is None:
            raise ValueError(f"Recipe step {step.name!r} requires an artifact mapping")
        missing = [
            artifact_name
            for artifact_name in step.artifact_kwargs.values()
            if artifact_name not in artifacts
        ]
        if missing:
            raise KeyError(
                f"Recipe step {step.name!r} is missing artifacts: "
                + ", ".join(map(repr, missing))
            )
        kwargs.update(
            {
                plot_keyword: artifacts[artifact_name]
                for plot_keyword, artifact_name in step.artifact_kwargs.items()
            }
        )
    if step.target is not None:
        if "target" in kwargs:
            raise ValueError(f"Recipe step {step.name!r} defines target in two places")
        if targets is None or step.target.panel not in targets:
            raise KeyError(
                f"Recipe step {step.name!r} requires panel "
                f"{step.target.panel!r} in targets"
            )
        kwargs["target"] = targets[step.target.panel]
    kwargs["show"] = False
    plot_result = plot(store, **kwargs)
    should_close = output_path is not None or show

    try:
        if output_path is not None:
            assert step.output is not None
            plot_result.save(
                output_path,
                dpi=step.output.dpi,
                transparent=step.output.transparent,
                exact_size=step.output.exact_size,
            )
        if show:
            plot_result.show()
    except Exception as error:
        if should_close and getattr(plot_result, "owns_figure", False):
            try:
                plot_result.close()
            except Exception as close_error:
                error.add_note(
                    "The plot also failed to close cleanly: "
                    f"{type(close_error).__name__}: {close_error}"
                )
        raise

    if should_close and getattr(plot_result, "owns_figure", False):
        plot_result.close()
    return PlotOutput(name=step.name, result=plot_result, path=output_path)


def run_plot_recipe(
    store: Any,
    recipe: PlotRecipe | str | Path,
    *,
    artifacts: Mapping[str, Any] | None = None,
    targets: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
    continue_on_error: bool = False,
) -> PlotRecipeResult:
    """Run recipe steps in order without allowing plots to display themselves."""

    resolved_recipe = _load_recipe(recipe)
    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")
    if not isinstance(continue_on_error, bool):
        raise TypeError("continue_on_error must be a boolean")
    if artifacts is not None and not isinstance(artifacts, Mapping):
        raise TypeError("artifacts must be a mapping or None")
    if targets is not None and not isinstance(targets, Mapping):
        raise TypeError("targets must be a mapping or None")

    has_outputs = any(
        step.output_filename is not None for step in resolved_recipe.steps
    )
    if has_outputs and output_dir is None:
        raise ValueError("output_dir is required when a step declares outputFilename")

    output_paths: dict[str, Path] = {}
    if has_outputs:
        assert output_dir is not None
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_paths = {
            step.name: _safe_output_path(output_root, step.output_filename)
            for step in resolved_recipe.steps
            if step.output_filename is not None
        }

    outputs: list[PlotOutput] = []
    written_paths: list[Path] = []
    captured_warnings: list[str] = []
    failures: dict[str, str] = {}

    for step in resolved_recipe.steps:
        recorded: list[warning_module.WarningMessage] = []
        try:
            with warning_module.catch_warnings(record=True) as recorded:
                warning_module.simplefilter("always")
                output = _run_step(
                    store,
                    step,
                    output_path=output_paths.get(step.name),
                    show=show,
                    artifacts=artifacts,
                    targets=targets,
                )
        except Exception as error:
            captured_warnings.extend(
                f"{step.name}: {warning.message}" for warning in recorded
            )
            if not continue_on_error:
                raise
            failure = _compact_failure(error)
            failures[step.name] = failure
            _log_failure(step.name, failure)
            continue

        captured_warnings.extend(
            f"{step.name}: {warning.message}" for warning in recorded
        )
        outputs.append(output)
        if output.path is not None:
            written_paths.append(output.path)

    return PlotRecipeResult(
        outputs=tuple(outputs),
        written_paths=tuple(written_paths),
        warnings=tuple(captured_warnings),
        failures=failures,
    )


def run_recipe(
    store: Any,
    recipe: PlotRecipe | str | Path,
    *,
    artifacts: Mapping[str, Any] | None = None,
    targets: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
    continue_on_error: bool = False,
) -> PlotRecipeResult:
    """Run a plotting recipe after any analysis pipeline has completed."""
    return run_plot_recipe(
        store,
        recipe,
        artifacts=artifacts,
        targets=targets,
        output_dir=output_dir,
        show=show,
        continue_on_error=continue_on_error,
    )


__all__ = [
    "ALLOWED_OUTPUT_FORMATS",
    "ALLOWED_PLOTS",
    "PlotOutput",
    "PlotOutputSettings",
    "PlotPanelTarget",
    "PlotRecipe",
    "PlotRecipeResult",
    "PlotStep",
    "run_recipe",
    "run_plot_recipe",
]
