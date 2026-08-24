"""Themes and categorical palettes for scarf.plotting."""

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

from ._contracts import FrameStyle, LegendLoc

# Shared Scarf figure defaults used by embedding-like plots.
DEFAULT_POINT_SIZE = 10.0
DEFAULT_POINT_EDGEWIDTH = 0.1
DEFAULT_RASTERIZE_THRESHOLD = 50_000
DEFAULT_PANEL_INCHES = 3.2
MAX_FIGURE_WIDTH_INCHES = 7.5
LEGEND_SIDE_MAX_CATEGORIES = 12
LEGEND_ON_DATA_MAX_CATEGORIES = 40
LEGEND_SIDE_ENTRIES_PER_COLUMN = 20
LEGEND_SIDE_MAX_COLUMNS = 4
LEGEND_SIDE_MAX_ENTRIES = LEGEND_SIDE_ENTRIES_PER_COLUMN * LEGEND_SIDE_MAX_COLUMNS

_LAYOUT_POINT_SIZE_SPECS: WeakKeyDictionary[
    Any,
    tuple[int, float, float, float],
] = WeakKeyDictionary()

# Okabe-Ito plus four high-contrast extensions for categorical figures.
COLORBLIND_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#6F4E7C",
    "#2E8B57",
    "#A05195",
    "#8C564B",
]

# Lifted from scanpy.plotting.palettes.
CUSTOM_PALETTES: dict[int, list[str]] = {
    10: [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#aa40fc",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#b5bd61",
        "#17becf",
    ],
    20: [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ],
    28: [
        "#023fa5",
        "#7d87b9",
        "#bec1d4",
        "#d6bcc0",
        "#bb7784",
        "#8e063b",
        "#4a6fe3",
        "#8595e1",
        "#b5bbe3",
        "#e6afb9",
        "#e07b91",
        "#d33f6a",
        "#11c638",
        "#8dd593",
        "#c6dec7",
        "#ead3c6",
        "#f0b98d",
        "#ef9708",
        "#0fcfc0",
        "#9cded6",
        "#d5eae7",
        "#f3e1eb",
        "#f6c4e1",
        "#f79cd4",
        "#7f7f7f",
        "#c7c7c7",
        "#1CE6FF",
        "#336600",
    ],
    102: [
        "#FFFF00",
        "#1CE6FF",
        "#FF34FF",
        "#FF4A46",
        "#008941",
        "#006FA6",
        "#A30059",
        "#FFDBE5",
        "#7A4900",
        "#0000A6",
        "#63FFAC",
        "#B79762",
        "#004D43",
        "#8FB0FF",
        "#997D87",
        "#5A0007",
        "#809693",
        "#6A3A4C",
        "#1B4400",
        "#4FC601",
        "#3B5DFF",
        "#4A3B53",
        "#FF2F80",
        "#61615A",
        "#BA0900",
        "#6B7900",
        "#00C2A0",
        "#FFAA92",
        "#FF90C9",
        "#B903AA",
        "#D16100",
        "#DDEFFF",
        "#000035",
        "#7B4F4B",
        "#A1C299",
        "#300018",
        "#0AA6D8",
        "#013349",
        "#00846F",
        "#372101",
        "#FFB500",
        "#C2FFED",
        "#A079BF",
        "#CC0744",
        "#C0B9B2",
        "#C2FF99",
        "#001E09",
        "#00489C",
        "#6F0062",
        "#0CBD66",
        "#EEC3FF",
        "#456D75",
        "#B77B68",
        "#7A87A1",
        "#788D66",
        "#885578",
        "#FAD09F",
        "#FF8A9A",
        "#D157A0",
        "#BEC459",
        "#456648",
        "#0086ED",
        "#886F4C",
        "#34362D",
        "#B4A8BD",
        "#00A6AA",
        "#452C2C",
        "#636375",
        "#A3C8C9",
        "#FF913F",
        "#938A81",
        "#575329",
        "#00FECF",
        "#B05B6F",
        "#8CD0FF",
        "#3B9700",
        "#04F757",
        "#C8A1A1",
        "#1E6E00",
        "#7900D7",
        "#A77500",
        "#6367A9",
        "#A05837",
        "#6B002C",
        "#772600",
        "#D790FF",
        "#9B9700",
        "#549E79",
        "#FFF69F",
        "#201625",
        "#72418F",
        "#BC23FF",
        "#99ADC0",
        "#3A2465",
        "#922329",
        "#5B4534",
        "#FDE8DC",
        "#404E55",
        "#0089A3",
        "#CB7E98",
        "#A4E804",
        "#324E72",
    ],
}

THEMES: dict[str, dict[str, Any]] = {
    "paper": {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    },
    "notebook": {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 150,
        "figure.dpi": 100,
    },
    "minimal": {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    },
    "dark": {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": "#e8e8e8",
        "axes.labelcolor": "#e8e8e8",
        "axes.edgecolor": "#e8e8e8",
        "xtick.color": "#e8e8e8",
        "ytick.color": "#e8e8e8",
        "axes.titlecolor": "#e8e8e8",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 150,
        "figure.dpi": 100,
    },
}


def default_point_size(
    n_cells: int,
    *,
    panel_area: float = DEFAULT_PANEL_INCHES**2,
    size_min: float = 1.0,
    size_max: float = 28.0,
) -> float:
    """Marker area derived from selected cells and physical panel area."""
    if panel_area <= 0:
        raise ValueError("panel_area must be positive")
    if size_min <= 0 or size_max < size_min:
        raise ValueError("point-size bounds must satisfy 0 < size_min <= size_max")
    n = max(1, int(n_cells))
    reference_area = DEFAULT_PANEL_INCHES**2
    area_factor = (float(panel_area) / reference_area) ** 0.72
    population_factor = (1_000.0 / n) ** 0.5
    return float(min(size_max, max(size_min, 16.0 * area_factor * population_factor)))


def default_point_edgewidth(
    n_cells: int,
    *,
    point_size: float | None = None,
) -> float:
    """Tune point outlines to marker area and cloud density."""
    n = max(1, int(n_cells))
    area = point_size if point_size is not None else default_point_size(n)
    if n >= 20_000 or area < 2.5:
        return 0.0
    if area < 7.0 or n >= 10_000:
        return 0.05
    return 0.15


def register_layout_point_size(
    collection: Any,
    *,
    n_points: int,
    size_min: float,
    size_max: float,
    multiplier: float = 1.0,
) -> None:
    _LAYOUT_POINT_SIZE_SPECS[collection] = (
        int(n_points),
        float(size_min),
        float(size_max),
        float(multiplier),
    )


def refresh_layout_point_sizes(figure: Any) -> None:
    """Refresh marked scatter artists after figure layout is resolved."""
    marked = [
        (ax, collection, specification)
        for ax in figure.axes
        for collection in ax.collections
        if (specification := _LAYOUT_POINT_SIZE_SPECS.get(collection)) is not None
    ]
    if not marked:
        return
    figure.canvas.draw()
    for ax, collection, specification in marked:
        bbox = ax.get_position()
        width, height = figure.get_size_inches()
        panel_area = float(bbox.width * width * bbox.height * height)
        n_points, size_min, size_max, multiplier = specification
        point_size = default_point_size(
            n_points,
            panel_area=panel_area,
            size_min=size_min,
            size_max=size_max,
        )
        collection.set_sizes(
            np.full(
                len(collection.get_offsets()),
                point_size * multiplier,
                dtype=np.float64,
            )
        )


def resolve_legend_loc(n_categories: int, legend_loc: LegendLoc = "auto") -> LegendLoc:
    """Choose a legend placement that survives many clusters."""
    if legend_loc != "auto":
        if legend_loc not in ("right", "on_data", "none"):
            raise ValueError(
                "legend_loc must be one of 'auto', 'right', 'on_data', 'none'"
            )
        return legend_loc
    if n_categories <= LEGEND_SIDE_MAX_CATEGORIES:
        return "right"
    if n_categories <= LEGEND_ON_DATA_MAX_CATEGORIES:
        return "on_data"
    return "right"


def legend_side_columns(n_entries: int) -> int:
    """Columns for a side legend, bounded so wide category sets stay readable."""
    columns = int(np.ceil(max(int(n_entries), 1) / LEGEND_SIDE_ENTRIES_PER_COLUMN))
    return max(1, min(columns, LEGEND_SIDE_MAX_COLUMNS))


def capped_figsize(
    width: float,
    height: float,
    *,
    max_width: float | None = MAX_FIGURE_WIDTH_INCHES,
) -> tuple[float, float]:
    """Clamp figure width so atlas-scale category counts stay page-sized."""
    resolved_width = float(width)
    if max_width is not None:
        if max_width <= 0:
            raise ValueError("max_width must be positive or None")
        resolved_width = min(resolved_width, float(max_width))
    return (resolved_width, float(height))


def _category_sort_key(value: Any) -> tuple[Any, ...]:
    """Sort key: numbers in numeric order, then natural string order."""
    import math

    import numpy as np
    import pandas as pd

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (2, ())
    try:
        if pd.isna(value):
            return (2, ())
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return (1, (str(bool(value)).lower(),))
    if isinstance(value, (int, np.integer)):
        return (0, (float(value),))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return (0, (float(value),))

    text = str(value)
    try:
        return (0, (float(text),))
    except ValueError:
        pass

    tokens = tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", text)
        if part != ""
    )
    return (1, tokens)


def sort_categories(values: Sequence[Any]) -> list[Any]:
    """Order categories with numeric ids before natural strings (1, 2, 10)."""
    return sorted(values, key=_category_sort_key)


def palette_for_n(
    n: int,
    *,
    palette_name: str = "default",
) -> list[str]:
    if palette_name not in ("default", "colorblind"):
        raise ValueError("palette_name must be 'default' or 'colorblind'")
    if palette_name == "colorblind":
        if n <= len(COLORBLIND_PALETTE):
            return list(COLORBLIND_PALETTE[:n])
        from ..utils.logging import logger
        from ._deps import require_seaborn

        logger.warning(
            f"Requested {n} colorblind-safe colors but only "
            f"{len(COLORBLIND_PALETTE)} are available; "
            "falling back to evenly spaced hues that are distinct but not "
            "guaranteed colorblind safe"
        )
        sns = require_seaborn()
        return list(sns.color_palette("husl", n_colors=n).as_hex())
    if n <= 10:
        return list(CUSTOM_PALETTES[10][:n])
    if n <= 20:
        return list(CUSTOM_PALETTES[20][:n])
    if n <= 28:
        return list(CUSTOM_PALETTES[28][:n])
    if n <= 102:
        return list(CUSTOM_PALETTES[102][:n])
    from ._deps import require_seaborn

    sns = require_seaborn()
    return list(sns.color_palette("husl", n_colors=n).as_hex())


def categorical_color_map(
    categories: list[Any],
    *,
    palette: Mapping[Any, str] | None = None,
    palette_name: str = "default",
    missing_label: str | None = None,
    missing_color: str = "#bdbdbd",
) -> dict[Any, str]:
    cats = list(categories)
    if palette is not None:
        out = dict(palette)
        for cat in cats:
            if cat not in out:
                raise KeyError(f"Category {cat!r} missing from palette")
    else:
        colors = palette_for_n(len(cats), palette_name=palette_name)
        out = dict(zip(cats, colors))
    if missing_label is not None:
        out[missing_label] = missing_color
    return out


def continuous_norm(
    mpl: Any,
    *,
    vmin: float,
    vmax: float,
    vcenter: float | None,
) -> Any:
    if vmax <= vmin:
        vmax = vmin + 1.0
    if vcenter is None:
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if not vmin < vcenter < vmax:
        raise ValueError("vcenter must be strictly between the color limits")
    return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def square_axis_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pad axis limits so the data window is square.

    Keeps equal-aspect embeddings visually square after legends and colorbars
    shrink the available axes width.
    """
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    span = max(x1 - x0, y1 - y0, 1e-12)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half = 0.5 * span
    return (cx - half, cx + half), (cy - half, cy + half)


def scatter_edgecolor(theme: str = "notebook") -> str:
    """Marker edge color that stays readable on light and dark themes."""
    if theme == "dark":
        # Mid grey keeps dark fills legible without turning markers into rings.
        return "#8f8f8f"
    return "#333333"


def foreground_color(theme: str = "notebook") -> str:
    """High-contrast foreground for annotations and segment borders."""
    return "#e8e8e8" if theme == "dark" else "#333333"


def finish_embedding_axes(
    ax: Any,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str = "",
    ylabel: str = "",
    title: str | None = None,
    frame: FrameStyle = "minimal",
) -> None:
    """Apply shared Scarf chrome to a 2D embedding axes."""
    if frame not in ("axes", "minimal", "none"):
        raise ValueError("frame must be one of 'axes', 'minimal', 'none'")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    if frame == "axes":
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    if frame == "none" and hasattr(ax, "spines"):
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        ax.set_title(title)


def apply_figure_chrome(figure: Any, theme: str = "notebook") -> None:
    """Apply Scarf figure background and spine defaults after axes creation."""
    opaque = theme != "dark"
    if opaque:
        figure.patch.set_facecolor("white")
        figure.patch.set_alpha(1.0)
    else:
        figure.patch.set_alpha(0)
    for ax in figure.axes:
        if opaque:
            ax.patch.set_facecolor("white")
            ax.patch.set_alpha(1.0)
        else:
            ax.patch.set_alpha(0)
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)


@contextmanager
def theme_context(name: str = "notebook") -> Iterator[None]:
    if name not in THEMES:
        raise KeyError(f"Unknown theme {name!r}. Choose from: {sorted(THEMES)}")
    from ._deps import require_matplotlib

    _, mpl = require_matplotlib()
    with mpl.rc_context(THEMES[name]):
        yield


def register_theme(
    name: str,
    rcparams: Mapping[str, Any],
    *,
    base: str | None = "notebook",
    overwrite: bool = False,
) -> None:
    """Register a Matplotlib rcParams theme for later plot calls."""
    if not name:
        raise ValueError("Theme name must be non-empty")
    if name in THEMES and not overwrite:
        raise ValueError(f"Theme {name!r} already exists")
    if base is not None and base not in THEMES:
        raise KeyError(f"Unknown base theme {base!r}")
    from ._deps import require_matplotlib

    _, mpl = require_matplotlib()
    invalid = sorted(set(rcparams) - set(mpl.rcParams))
    if invalid:
        raise KeyError(f"Unknown Matplotlib rcParams: {invalid}")
    values = dict(THEMES[base]) if base is not None else {}
    values.update(dict(rcparams))
    THEMES[name] = values
