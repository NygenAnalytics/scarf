"""Themes and categorical palettes for scarf.plotting."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd

from ..utils.arrays import sort_categories
from ._contracts import CategoricalScale, ColorScale, FrameStyle, LegendLoc

# Shared Scarf figure defaults used by embedding-like plots.
DEFAULT_RASTERIZE_THRESHOLD = 50_000
DEFAULT_PANEL_INCHES = 3.2
MAX_FIGURE_WIDTH_INCHES = 7.5
LEGEND_SIDE_MAX_CATEGORIES = 12
LEGEND_ON_DATA_MAX_CATEGORIES = 40
LEGEND_SIDE_ENTRIES_PER_COLUMN = 20
LEGEND_SIDE_MAX_COLUMNS = 4
LEGEND_SIDE_MAX_ENTRIES = LEGEND_SIDE_ENTRIES_PER_COLUMN * LEGEND_SIDE_MAX_COLUMNS


@dataclass(frozen=True, slots=True)
class _PointSizeSpec:
    """How to resize one scatter artist once its panel size is final."""

    n_points: int
    size_min: float
    size_max: float
    multiplier: float
    edgecolor: str | None
    edgewidth: float | None


_LAYOUT_POINT_SIZE_SPECS: WeakKeyDictionary[Any, _PointSizeSpec] = WeakKeyDictionary()

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


def scatter_edges(edgecolor: str, edgewidth: float) -> tuple[str, float]:
    """Return scatter edge color and width, hiding edges of zero width."""
    if edgewidth <= 0:
        return "none", 0.0
    return edgecolor, float(edgewidth)


def panel_area_inches(ax: Any) -> float:
    """Physical axes area in square inches, floored for collapsed panels."""
    bounds = ax.get_position()
    width = max(float(bounds.width * ax.figure.get_figwidth()), 0.1)
    height = max(float(bounds.height * ax.figure.get_figheight()), 0.1)
    return width * height


def register_layout_point_size(
    collection: Any,
    *,
    n_points: int,
    size_min: float,
    size_max: float,
    multiplier: float = 1.0,
    edgecolor: str | None = None,
    edgewidth: float | None = None,
) -> None:
    """Mark a scatter artist whose marker area follows its final panel size.

    ``multiplier`` scales the panel point size, for example for highlighted
    cells. With ``edgecolor``, marker edges are refreshed with the size, using
    ``edgewidth`` or a width derived from the point size and population.
    """
    _LAYOUT_POINT_SIZE_SPECS[collection] = _PointSizeSpec(
        n_points=int(n_points),
        size_min=float(size_min),
        size_max=float(size_max),
        multiplier=float(multiplier),
        edgecolor=edgecolor,
        edgewidth=None if edgewidth is None else float(edgewidth),
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
        point_size = default_point_size(
            specification.n_points,
            panel_area=panel_area_inches(ax),
            size_min=specification.size_min,
            size_max=specification.size_max,
        )
        collection.set_sizes(
            np.full(
                len(collection.get_offsets()),
                point_size * specification.multiplier,
                dtype=np.float64,
            )
        )
        if specification.edgecolor is not None:
            edges, linewidth = scatter_edges(
                specification.edgecolor,
                (
                    specification.edgewidth
                    if specification.edgewidth is not None
                    else default_point_edgewidth(
                        specification.n_points,
                        point_size=point_size,
                    )
                ),
            )
            collection.set_edgecolors(edges)
            collection.set_linewidths(linewidth)


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
) -> dict[Any, str]:
    cats = list(categories)
    if palette is not None:
        out = dict(palette)
        for cat in cats:
            if cat not in out:
                raise KeyError(f"Category {cat!r} missing from palette")
        return out
    colors = palette_for_n(len(cats), palette_name=palette_name)
    return dict(zip(cats, colors))


def colormap_palette(categories: Sequence[Any], cmap: str) -> dict[Any, str]:
    """Assign categories evenly spaced colors from a Matplotlib colormap."""
    from ._deps import require_matplotlib

    _, mpl = require_matplotlib()
    colormap = mpl.colormaps.get_cmap(cmap)
    last = max(len(categories) - 1, 1)
    return {
        category: mpl.colors.to_hex(colormap(index / last))
        for index, category in enumerate(categories)
    }


def resolve_category_scale(
    observed: Sequence[Any] | np.ndarray,
    scale: CategoricalScale | None,
    *,
    context: str = "categorical_scale",
) -> CategoricalScale:
    """Resolve the displayed order and colors of the observed categories.

    An explicit ``order`` must list every observed category and sets their
    display order; categories it lists that no plotted cell carries are left
    out of the display. Generated colors are assigned over the full explicit
    order, so they stay stable when a plot shows a subset. Without an order,
    categories sort naturally. Missing values are never categories.
    """
    present = [
        value
        for value in pd.unique(np.asarray(observed, dtype=object).ravel())
        if not _is_missing_category(value)
    ]
    if scale is not None and scale.order is not None:
        full_order = list(scale.order)
        if len(set(full_order)) != len(full_order):
            raise ValueError(f"{context}.order cannot contain duplicates")
        unlisted = [value for value in present if value not in full_order]
        if unlisted:
            raise ValueError(
                f"{context}.order is missing observed values: "
                + ", ".join(map(str, unlisted[:10]))
            )
        shown = set(present)
        order = [value for value in full_order if value in shown]
    else:
        full_order = sort_categories(present)
        order = list(full_order)
    colors = categorical_color_map(
        full_order,
        palette=scale.palette if scale is not None else None,
        palette_name=scale.palette_name if scale is not None else "default",
    )
    labels = scale.labels if scale is not None else None
    return CategoricalScale(
        order=tuple(order),
        palette={value: colors[value] for value in order},
        labels=(
            None
            if labels is None
            else {value: str(labels.get(value, value)) for value in order}
        ),
        missing_color=scale.missing_color if scale is not None else "#bdbdbd",
        missing_label=scale.missing_label if scale is not None else "NA",
        palette_name=scale.palette_name if scale is not None else "default",
    )


def category_label(scale: CategoricalScale, value: Any) -> str:
    """Display text for one category of a resolved scale."""
    if scale.labels is None:
        return str(value)
    return str(scale.labels.get(value, value))


def _is_missing_category(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def resolve_color_limits(values: Any, scale: ColorScale) -> tuple[float, float]:
    """Resolve continuous color limits under the policy every plot shares.

    Limits come from the finite values, or from their ``quantiles``, and an
    explicit ``vmin`` or ``vmax`` replaces the matching side. A ``vcenter``
    pivot widens derived limits so a diverging map works on one-sided values.
    Tied or inverted limits keep the explicit or lower side and extend the
    other side by one unit, so a constant value takes the low end of the map.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        low, high = 0.0, 1.0
    elif scale.quantiles is not None:
        low, high = (float(value) for value in np.quantile(finite, scale.quantiles))
    else:
        low, high = float(finite.min()), float(finite.max())
    if scale.vmin is not None:
        low = float(scale.vmin)
    if scale.vmax is not None:
        high = float(scale.vmax)
    if scale.vcenter is not None:
        center = float(scale.vcenter)
        if (scale.vmin is not None and center <= low) or (
            scale.vmax is not None and center >= high
        ):
            raise ValueError("vcenter must be strictly between the color limits")
        if not low < center < high:
            margin = max((max(high, center) - min(low, center)) * 1e-6, 1e-9)
            low = min(low, center - margin)
            high = max(high, center + margin)
    elif high <= low:
        if scale.vmax is not None and scale.vmin is None:
            low = high - 1.0
        else:
            high = low + 1.0
    if scale.scale == "log" and low <= 0:
        raise ValueError("Log color scale requires positive values")
    return low, high


def continuous_norm(
    mpl: Any,
    *,
    vmin: float,
    vmax: float,
    vcenter: float | None,
    scale: str = "linear",
) -> Any:
    """Build the Matplotlib norm for resolved limits and a ColorScale scale."""
    if vmax <= vmin:
        vmax = vmin + 1.0
    if scale == "log":
        if vmin <= 0:
            raise ValueError("Log color scale requires positive values")
        return mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    if scale == "symlog":
        return mpl.colors.SymLogNorm(
            linthresh=max(abs(vmax - vmin) * 0.001, 1e-12),
            vmin=vmin,
            vmax=vmax,
        )
    if vcenter is None:
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if not vmin < vcenter < vmax:
        raise ValueError("vcenter must be strictly between the color limits")
    return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def padded_square_limits(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pad finite coordinates by 5% on each axis and square the window."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.shape != yy.shape:
        raise ValueError("Coordinate columns must have matching shapes")
    finite = np.isfinite(xx) & np.isfinite(yy)
    if not finite.any():
        raise ValueError("No finite coordinates are available to plot")
    xx = xx[finite]
    yy = yy[finite]
    x_pad = 0.05 * (float(xx.max() - xx.min()) or 1.0)
    y_pad = 0.05 * (float(yy.max() - yy.min()) or 1.0)
    return square_axis_limits(
        (float(xx.min() - x_pad), float(xx.max() + x_pad)),
        (float(yy.min() - y_pad), float(yy.max() + y_pad)),
    )


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
