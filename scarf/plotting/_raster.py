"""Blockwise embedding raster (two-pass; no full-column materialization)."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._deps import require_matplotlib
from ._style import continuous_norm


@dataclass(frozen=True, slots=True)
class RasterCanvas:
    """Rasterized embedding panel."""

    image: np.ndarray  # float (H, W), NaN where empty
    counts: np.ndarray  # int (H, W)
    extent: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    vmin: float
    vmax: float
    n_cells: int
    n_blocks: int


def density_canvas_from_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    pixels: int,
) -> RasterCanvas:
    """Bin materialized coordinates into the shared raster canvas contract."""
    if pixels < 8:
        raise ValueError("pixels must be at least 8")
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if len(xx) != len(yy):
        raise ValueError("x and y lengths must match")
    finite = np.isfinite(xx) & np.isfinite(yy)
    xmin, xmax, ymin, ymax = map(float, extent)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("extent must have increasing coordinate limits")
    counts_xy, _, _ = np.histogram2d(
        xx[finite],
        yy[finite],
        bins=pixels,
        range=((xmin, xmax), (ymin, ymax)),
    )
    counts = np.flipud(counts_xy.T).astype(np.int64, copy=False)
    image = np.full(counts.shape, np.nan, dtype=np.float64)
    occupied = counts > 0
    image[occupied] = np.log1p(counts[occupied])
    return RasterCanvas(
        image=image,
        counts=counts,
        extent=(xmin, xmax, ymin, ymax),
        vmin=0.0,
        vmax=float(np.nanmax(image)) if occupied.any() else 1.0,
        n_cells=int(finite.sum()),
        n_blocks=1,
    )


def _finite_minmax(values: np.ndarray) -> tuple[float, float] | None:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return None
    return float(v.min()), float(v.max())


def _priority_sample_update(
    sample_values: np.ndarray,
    sample_priorities: np.ndarray,
    n_sampled: int,
    values: np.ndarray,
    *,
    capacity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Update an exact uniform sample using independent random priorities."""
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return sample_values, sample_priorities, n_sampled

    new_priorities = rng.random(len(vals))
    combined_values = np.concatenate((sample_values[:n_sampled], vals))
    combined_priorities = np.concatenate(
        (sample_priorities[:n_sampled], new_priorities)
    )
    keep = min(capacity, len(combined_values))
    if len(combined_values) > keep:
        selected = np.argpartition(combined_priorities, -keep)[-keep:]
        combined_values = combined_values[selected]
        combined_priorities = combined_priorities[selected]
    sample_values[:keep] = combined_values
    sample_priorities[:keep] = combined_priorities
    return sample_values, sample_priorities, keep


def raster_from_metadata(
    cells: Any,
    *,
    x_key: str,
    y_key: str,
    color_key: str | None = None,
    cell_key: str = "I",
    subset_by: str | None = None,
    pixels: int = 400,
    block_rows: int | None = None,
    quantiles: tuple[float, float] | None = (0.01, 0.99),
    seed: int = 0,
    sample_capacity: int = 50_000,
) -> RasterCanvas:
    """Two-pass raster of metadata columns via ``MetaData.iter_row_blocks``.

    Pass 1: bounds and color limits (exact min/max, or approximate quantiles via
    reservoir sampling). Pass 2: accumulate mean color per pixel.
    """
    if pixels < 8:
        raise ValueError("pixels must be >= 8")
    if sample_capacity < 1:
        raise ValueError("sample_capacity must be >= 1")
    if quantiles is not None:
        q0, q1 = quantiles
        if not (0.0 <= q0 < q1 <= 1.0):
            raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    cols = [x_key, y_key]
    if color_key is not None:
        cols.append(color_key)
    if subset_by is not None:
        if subset_by not in cells.columns:
            raise KeyError(f"subset_by {subset_by!r} not found in cell metadata")
        cols.append(subset_by)
    rng = np.random.default_rng(seed)

    # --- Pass 1: bounds + color scale ---
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    cmin = np.inf
    cmax = -np.inf
    sample_values = np.empty(sample_capacity, dtype=np.float64)
    sample_priorities = np.empty(sample_capacity, dtype=np.float64)
    n_sampled = 0
    n_cells = 0
    n_blocks = 0
    for block in cells.iter_row_blocks(
        cell_key=cell_key, columns=cols, block_rows=block_rows
    ):
        n_blocks += 1
        if len(block.active_global_indices) == 0:
            continue
        x = np.asarray(block.values[x_key], dtype=np.float64)
        y = np.asarray(block.values[y_key], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if subset_by is not None:
            sub = np.asarray(block.values[subset_by])
            if sub.dtype != bool:
                raise TypeError(
                    f"subset_by {subset_by!r} must be boolean; got {sub.dtype}"
                )
            finite &= sub
        if not finite.any():
            continue
        n_cells += int(finite.sum())
        xmin = min(xmin, float(x[finite].min()))
        xmax = max(xmax, float(x[finite].max()))
        ymin = min(ymin, float(y[finite].min()))
        ymax = max(ymax, float(y[finite].max()))
        if color_key is not None:
            c = np.asarray(block.values[color_key], dtype=np.float64)[finite]
            mm = _finite_minmax(c)
            if mm is not None:
                cmin = min(cmin, mm[0])
                cmax = max(cmax, mm[1])
            if quantiles is not None:
                sample_values, sample_priorities, n_sampled = _priority_sample_update(
                    sample_values,
                    sample_priorities,
                    n_sampled,
                    c,
                    capacity=sample_capacity,
                    rng=rng,
                )

    if n_cells == 0 or not np.isfinite(xmin):
        empty = np.full((pixels, pixels), np.nan, dtype=np.float64)
        return RasterCanvas(
            image=empty,
            counts=np.zeros((pixels, pixels), dtype=np.int64),
            extent=(0.0, 1.0, 0.0, 1.0),
            vmin=0.0,
            vmax=1.0,
            n_cells=0,
            n_blocks=n_blocks,
        )

    if color_key is None:
        vmin, vmax = 0.0, 1.0
    elif not np.isfinite(cmin):
        vmin, vmax = 0.0, 1.0
    elif quantiles is not None and n_sampled > 0:
        sample = sample_values[:n_sampled]
        q0, q1 = quantiles
        vmin = float(np.quantile(sample, q0))
        vmax = float(np.quantile(sample, q1))
        if vmax <= vmin:
            vmin, vmax = float(cmin), float(cmax if cmax > cmin else cmin + 1.0)
    else:
        vmin, vmax = float(cmin), float(cmax if cmax > cmin else cmin + 1.0)

    # Pad extent slightly so edge points land inside bins.
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = 0.01 * dx if dx > 0 else 0.5
    pad_y = 0.01 * dy if dy > 0 else 0.5
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0

    sums = np.zeros((pixels, pixels), dtype=np.float64)
    counts = np.zeros((pixels, pixels), dtype=np.int64)

    # --- Pass 2: accumulate ---
    for block in cells.iter_row_blocks(
        cell_key=cell_key, columns=cols, block_rows=block_rows
    ):
        if len(block.active_global_indices) == 0:
            continue
        x = np.asarray(block.values[x_key], dtype=np.float64)
        y = np.asarray(block.values[y_key], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if subset_by is not None:
            sub = np.asarray(block.values[subset_by])
            if sub.dtype != bool:
                raise TypeError(
                    f"subset_by {subset_by!r} must be boolean; got {sub.dtype}"
                )
            finite &= sub
        if not finite.any():
            continue
        x = x[finite]
        y = y[finite]
        ix = np.clip(
            ((x - xmin) / (xmax - xmin) * (pixels - 1e-9)).astype(np.int64),
            0,
            pixels - 1,
        )
        iy = np.clip(
            ((y - ymin) / (ymax - ymin) * (pixels - 1e-9)).astype(np.int64),
            0,
            pixels - 1,
        )
        # Image row 0 is top; flip y for display extent mapping.
        iy_img = pixels - 1 - iy
        if color_key is None:
            np.add.at(counts, (iy_img, ix), 1)
        else:
            c = np.asarray(block.values[color_key], dtype=np.float64)[finite]
            finite_color = np.isfinite(c)
            np.add.at(sums, (iy_img[finite_color], ix[finite_color]), c[finite_color])
            np.add.at(counts, (iy_img[finite_color], ix[finite_color]), 1)

    image = np.full((pixels, pixels), np.nan, dtype=np.float64)
    nonzero = counts > 0
    if color_key is None:
        image[nonzero] = np.log1p(counts[nonzero])
        vmin = 0.0
        vmax = float(image[nonzero].max()) if nonzero.any() else 1.0
    else:
        image[nonzero] = sums[nonzero] / counts[nonzero]
    return RasterCanvas(
        image=image,
        counts=counts,
        extent=(xmin, xmax, ymin, ymax),
        vmin=vmin,
        vmax=vmax,
        n_cells=n_cells,
        n_blocks=n_blocks,
    )


def draw_raster_canvas(
    ax: Any,
    canvas: RasterCanvas,
    *,
    cmap: str = "viridis",
    missing_color: str = "white",
    vcenter: float | None = None,
) -> Any:
    """Draw a ``RasterCanvas`` onto a matplotlib axes; return the mappable."""
    _, mpl = require_matplotlib()
    data = canvas.image.copy()
    cmap_obj = mpl.colormaps.get_cmap(cmap).with_extremes(bad=missing_color)
    norm = continuous_norm(
        mpl,
        vmin=canvas.vmin,
        vmax=canvas.vmax,
        vcenter=vcenter,
    )
    im = ax.imshow(
        data,
        origin="upper",
        extent=[
            canvas.extent[0],
            canvas.extent[1],
            canvas.extent[2],
            canvas.extent[3],
        ],
        cmap=cmap_obj,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_facecolor(missing_color)
    return im
