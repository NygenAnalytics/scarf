"""Themes and categorical palettes for scarf.plotting."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

# Lifted from scanpy.plotting.palettes (also used by legacy scarf.plots).
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
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 150,
        "figure.dpi": 100,
    },
    "minimal": {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    },
}


def palette_for_n(n: int) -> list[str]:
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
    return list(sns.color_palette(n_colors=n).as_hex())


def categorical_color_map(
    categories: list[Any],
    *,
    palette: Mapping[Any, str] | None = None,
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
        colors = palette_for_n(len(cats))
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


@contextmanager
def theme_context(name: str = "notebook") -> Iterator[None]:
    if name not in THEMES:
        raise KeyError(f"Unknown theme {name!r}. Choose from: {sorted(THEMES)}")
    from ._deps import require_matplotlib

    _, mpl = require_matplotlib()
    with mpl.rc_context(THEMES[name]):
        yield
