"""PlotResult and figure ownership helpers."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Hashable, cast

import numpy as np
import pandas as pd

from ._contracts import PlotProvenance
from ._deps import require_matplotlib
from ._style import theme_context


def _json_ready(value: Any) -> Any:
    if not isinstance(value, type) and is_dataclass(value):
        return _json_ready(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


@dataclass(slots=True)
class LegendSpec:
    """Description of a legend or colorbar attached to a plot."""

    kind: str
    label: str | None = None
    scale_key: Hashable | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlotResult:
    """Return value from scarf.plotting functions.

    ``figure`` is the matplotlib figure. ``axes`` maps panel keys to axes.
    ``tables`` holds any summary data used to build the plot. ``owns_figure``
    is True when Scarf created the figure; call ``close()`` in that case when
    you are done so notebooks do not keep many open figures.
    """

    figure: Any
    axes: dict[Hashable, Any]
    tables: dict[str, pd.DataFrame]
    legends: tuple[LegendSpec, ...]
    scales: tuple[Any, ...]
    provenance: PlotProvenance
    owns_figure: bool
    theme: str = "notebook"

    def show(self) -> None:
        plt, mpl = require_matplotlib()
        backend = str(mpl.get_backend()).lower()
        in_ipython = False
        try:
            from IPython import get_ipython

            in_ipython = get_ipython() is not None
        except ImportError:
            pass
        # myst-nb / docs kernels often use Agg under the hood; prefer explicit
        # IPython display so figures land in notebook outputs.
        if in_ipython or "inline" in backend:
            from IPython.display import display

            display(self.figure)  # type: ignore[no-untyped-call]
            if self.owns_figure:
                plt.close(self.figure)
            return
        plt.show()

    def close(self) -> None:
        if not self.owns_figure:
            return
        plt, _ = require_matplotlib()
        plt.close(self.figure)

    def save_provenance(
        self,
        path: str | Path,
        *,
        figure_path: str | Path | None = None,
        dpi: float | None = None,
    ) -> Path:
        """Write a JSON file describing scales, legends, and cell counts."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        width, height = self.figure.get_size_inches()
        payload = {
            "provenance": self.provenance,
            "figure": {
                "width_inches": float(width),
                "height_inches": float(height),
                "dpi": float(self.figure.dpi),
            },
            "legends": self.legends,
            "scales": [
                {
                    "type": type(scale).__name__,
                    "values": scale,
                }
                for scale in self.scales
            ],
            "tables": {
                name: {
                    "rows": len(table),
                    "columns": [str(column) for column in table.columns],
                }
                for name, table in self.tables.items()
            },
        }
        if figure_path is not None:
            exported = Path(figure_path)
            payload["export"] = {
                "filename": exported.name,
                "format": exported.suffix.lower().lstrip("."),
                "dpi": float(dpi if dpi is not None else self.figure.dpi),
            }
        out.write_text(
            json.dumps(
                _json_ready(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return out

    def save(
        self,
        path: str | Path,
        *,
        dpi: int | None = None,
        transparent: bool = False,
        exact_size: bool = True,
        tiff_compression: str = "tiff_lzw",
        provenance_sidecar: bool | str | Path = False,
    ) -> Path:
        """Write the figure to disk (PNG, PDF, SVG, TIFF, and other backends).

        The default background is opaque white, which is what most journals
        expect. Pass ``transparent=True`` if you need the figure to sit on a
        dark notebook theme. ``exact_size=True`` keeps the inch size you set
        when the figure was created; set it to False only when you want a tight
        crop for display. ``provenance_sidecar=True`` writes a sibling JSON
        file with the plot metadata.
        """
        out = Path(path)
        if not out.suffix:
            raise ValueError("Export path must include a file extension")
        if dpi is not None and dpi <= 0:
            raise ValueError("dpi must be positive")
        file_type = out.suffix.lower().lstrip(".")
        supported = self.figure.canvas.get_supported_filetypes()
        if file_type not in supported:
            raise ValueError(
                f"Unsupported export format {out.suffix!r}; choose from "
                f"{sorted(supported)}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"transparent": transparent}
        if dpi is not None:
            kwargs["dpi"] = dpi
        elif file_type in ("tif", "tiff"):
            kwargs["dpi"] = 300
        if file_type in ("tif", "tiff"):
            kwargs["pil_kwargs"] = {"compression": tiff_compression}
        # Exact-size export must not crop; tight bbox changes physical size.
        kwargs["bbox_inches"] = None if exact_size else "tight"
        _, mpl = require_matplotlib()
        # Honour explicit transparent=False even when the active theme prefers none.
        export_rc: dict[str, Any] = {"savefig.bbox": None} if exact_size else {}
        figure_face = self.figure.patch.get_facecolor()
        figure_alpha = self.figure.patch.get_alpha()
        axes_faces = [
            (ax, ax.patch.get_facecolor(), ax.patch.get_alpha())
            for ax in self.figure.axes
        ]
        if transparent:
            export_rc["savefig.transparent"] = True
            export_rc["savefig.facecolor"] = "none"
        else:
            export_rc["savefig.transparent"] = False
            export_rc["savefig.facecolor"] = "white"
            self.figure.patch.set_facecolor("white")
            self.figure.patch.set_alpha(1.0)
            for ax in self.figure.axes:
                ax.patch.set_facecolor("white")
                ax.patch.set_alpha(1.0)
        try:
            with theme_context(self.theme), mpl.rc_context(export_rc):
                self.figure.savefig(out, **kwargs)
        finally:
            self.figure.patch.set_facecolor(figure_face)
            self.figure.patch.set_alpha(figure_alpha)
            for ax, face, alpha in axes_faces:
                ax.patch.set_facecolor(face)
                ax.patch.set_alpha(alpha)
        if provenance_sidecar:
            sidecar = (
                out.with_suffix(out.suffix + ".json")
                if provenance_sidecar is True
                else Path(provenance_sidecar)
            )
            if sidecar.resolve() == out.resolve():
                raise ValueError("Provenance sidecar path must differ from figure path")
            self.save_provenance(
                sidecar,
                figure_path=out,
                dpi=float(kwargs.get("dpi", self.figure.dpi)),
            )
        return out


def normalize_axes_target(
    target: Any | None,
    *,
    panel_keys: Sequence[Hashable],
    figsize: tuple[float, float] | None,
    n_columns: int | None = None,
) -> tuple[Any, dict[Hashable, Any], bool]:
    """Return (figure, axes_map, owns_figure)."""
    plt, _ = require_matplotlib()
    keys = list(panel_keys)
    if not keys:
        raise ValueError("panel_keys must be non-empty")

    if target is None:
        if figsize is None:
            from ._style import DEFAULT_PANEL_INCHES

            panel = DEFAULT_PANEL_INCHES
            figsize = (
                (panel * len(keys), panel) if len(keys) > 1 else (panel + 0.4, panel)
            )
        if len(keys) == 1:
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
            return fig, {keys[0]: ax}, True
        ncols = n_columns if n_columns is not None else min(len(keys), 4)
        ncols = max(1, min(ncols, len(keys)))
        nrows = int(np.ceil(len(keys) / ncols))
        fig, axs = plt.subplots(
            nrows, ncols, figsize=figsize, squeeze=False, layout="constrained"
        )
        mapping: dict[Hashable, Any] = {}
        for i, key in enumerate(keys):
            mapping[key] = axs[i // ncols, i % ncols]
        for j in range(len(keys), nrows * ncols):
            fig.delaxes(axs[j // ncols, j % ncols])
        return fig, mapping, True

    if figsize is not None:
        raise ValueError("figsize is invalid when a caller-owned target is provided")

    if isinstance(target, Mapping):
        missing = [k for k in keys if k not in target]
        if missing:
            raise KeyError(f"target mapping missing panel keys: {missing}")
        mapping = {k: target[k] for k in keys}
        figures = {id(ax.figure): ax.figure for ax in mapping.values()}
        if len(figures) != 1:
            raise ValueError("All target axes must belong to the same figure")
        return next(iter(figures.values())), mapping, False

    if isinstance(target, (list, tuple)) or (
        isinstance(target, np.ndarray) and target.dtype == object
    ):
        arr = np.asarray(target, dtype=object).ravel()
        if len(arr) != len(keys):
            raise ValueError(f"Expected {len(keys)} axes for panels, got {len(arr)}")
        figures = {id(ax.figure): ax.figure for ax in arr}
        if len(figures) != 1:
            raise ValueError("All target axes must belong to the same figure")
        fig = next(iter(figures.values()))
        return fig, {k: ax for k, ax in zip(keys, arr)}, False

    if len(keys) != 1:
        raise TypeError(
            "A single Axes target is only valid for one-panel plots; "
            "pass a sequence or mapping of axes for faceted plots"
        )
    return target.figure, {keys[0]: target}, False


def label_panels(
    axes: Mapping[Hashable, Any] | Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    fontsize: float = 11,
    fontweight: str = "bold",
) -> None:
    """Add A/B/C panel labels to axes in order."""
    if isinstance(axes, Mapping):
        ax_list = list(axes.values())
    else:
        ax_list = list(axes)
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(ax_list))]
    if len(labels) != len(ax_list):
        raise ValueError("labels length must match number of axes")
    for ax, lab in zip(ax_list, labels):
        ax.text(
            0.02,
            0.98,
            lab,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight=fontweight,
        )


def collect_legends(
    figure: Any,
    results: Sequence[PlotResult],
) -> tuple[LegendSpec, ...]:
    """Combine legend/colorbar descriptors from several plot results."""
    legends: list[LegendSpec] = []
    for result in results:
        legends.extend(result.legends)
    out = tuple(legends)
    figure._scarf_legends = out  # type: ignore[attr-defined]
    return out


def as_2d_axes_array(in_ax: Any, n_columns: int = 1) -> np.ndarray:
    """Normalize legacy in_ax inputs to a 2D object array (rows, cols)."""
    if in_ax is None:
        raise ValueError("in_ax is None")
    if isinstance(in_ax, np.ndarray):
        if in_ax.ndim == 2:
            return in_ax
        if in_ax.ndim == 1:
            return in_ax.reshape(1, -1)
    return np.array([[in_ax]], dtype=object)
