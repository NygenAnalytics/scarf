import os
from collections.abc import Callable
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..utils.arguments import float_argument, integer_argument
from ..utils.process import suppress_native_output

# Scarf has one t-SNE backend, the optional sgtsnepi package. Its platform
# facts are those of sgtsnepi 0.7, the minimum the tsne extra requires.
SGTSNEPI_GUIDANCE = (
    "t-SNE needs the optional sgtsnepi package, which could not be imported. "
    'Install it with Scarf\'s tsne extra, for example pip install "scarf[tsne]". '
    "sgtsnepi publishes wheels only for Linux x86_64 and for macOS 26 or newer "
    "on arm64. Elsewhere, installing it compiles it from source, which needs a "
    "C++ compiler and FFTW, and sgtsnepi does not support Windows."
)


@dataclass(frozen=True, slots=True)
class SgtsneSettings:
    """SG-t-SNE settings as canonical Python numbers."""

    tsne_dims: int
    lambda_scale: float
    max_iter: int
    early_iter: int
    alpha: int
    box_h: float


def _positive_float(value: object, name: str) -> float:
    resolved = float_argument(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def sgtsne_settings(
    *,
    tsne_dims: object,
    lambda_scale: object,
    max_iter: object,
    early_iter: object,
    alpha: object,
    box_h: object,
) -> SgtsneSettings:
    """Validate SG-t-SNE settings and return them as canonical Python numbers."""
    return SgtsneSettings(
        tsne_dims=integer_argument(tsne_dims, "tsne_dims", minimum=1),
        lambda_scale=_positive_float(lambda_scale, "lambda_scale"),
        max_iter=integer_argument(max_iter, "max_iter", minimum=1),
        early_iter=integer_argument(early_iter, "early_iter", minimum=0),
        alpha=integer_argument(alpha, "alpha", minimum=1),
        box_h=_positive_float(box_h, "box_h"),
    )


def require_sgtsnepi() -> Callable[..., Any]:
    """Return the ``sgtsnepi`` entry point; raise ``ImportError`` if it is missing."""
    try:
        from sgtsnepi import sgtsnepi
    except ImportError as exc:
        raise ImportError(SGTSNEPI_GUIDANCE) from exc
    entry_point: Callable[..., Any] = sgtsnepi
    return entry_point


def run_sgtsne(
    graph: csr_matrix | coo_matrix,
    ini_embed: np.ndarray,
    *,
    tsne_dims: int = 2,
    max_iter: int = 500,
    early_iter: int = 200,
    alpha: int = 10,
    lambda_scale: float = 1.0,
    box_h: float = 0.7,
    verbose: bool = True,
) -> np.ndarray:
    """Run SG-t-SNE using the ``sgtsnepi`` Python backend."""
    settings = sgtsne_settings(
        tsne_dims=tsne_dims,
        lambda_scale=lambda_scale,
        max_iter=max_iter,
        early_iter=early_iter,
        alpha=alpha,
        box_h=box_h,
    )
    n_cells = graph.shape[0]
    ini_embed = np.asarray(ini_embed)
    if ini_embed.shape != (n_cells, settings.tsne_dims):
        raise ValueError(
            f"ini_embed must have shape ({n_cells}, {settings.tsne_dims}), "
            f"got {ini_embed.shape}"
        )
    sgtsnepi = require_sgtsnepi()
    graph = graph.tocsr(copy=True)
    graph.eliminate_zeros()

    # sgtsnepi's own silent mode closes descriptors 1 and 2 for the rest of
    # the process, so quiet runs redirect them around the call instead.
    with ExitStack() as output:
        if not verbose:
            output.enter_context(suppress_native_output())
            stream = output.enter_context(open(os.devnull, "w"))
            output.enter_context(redirect_stdout(stream))
            output.enter_context(redirect_stderr(stream))
        embedding = sgtsnepi(
            graph,
            y0=ini_embed.T,
            d=settings.tsne_dims,
            max_iter=settings.max_iter,
            early_exag=settings.early_iter,
            lambda_par=settings.lambda_scale,
            h=settings.box_h,
            alpha=settings.alpha,
            silent=False,
        )
    return np.asarray(embedding)
