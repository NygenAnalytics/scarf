"""Plot scale and theme behavior tests."""

import matplotlib
import pytest

matplotlib.use("Agg")

from scarf.plotting._deps import require_matplotlib
from scarf.plotting._style import (
    categorical_color_map,
    continuous_norm,
    palette_for_n,
    theme_context,
)


@pytest.mark.parametrize("size", [5, 15, 25, 50, 110])
def test_palette_for_n_returns_requested_colors(size):
    assert len(palette_for_n(size)) == size


def test_categorical_color_map_validates_custom_palette():
    with pytest.raises(KeyError, match="missing from palette"):
        categorical_color_map(["a", "b"], palette={"a": "red"})
    colors = categorical_color_map(
        ["a"],
        palette={"a": "red"},
        missing_label="NA",
        missing_color="gray",
    )
    assert colors == {"a": "red", "NA": "gray"}


def test_continuous_norm_supports_center_and_validates_bounds():
    _, mpl = require_matplotlib()
    norm = continuous_norm(mpl, vmin=-2, vmax=3, vcenter=0)
    assert norm.__class__.__name__ == "TwoSlopeNorm"
    with pytest.raises(ValueError, match="vcenter"):
        continuous_norm(mpl, vmin=0, vmax=3, vcenter=4)


def test_theme_context_rejects_unknown_theme():
    with pytest.raises(KeyError, match="Unknown theme"):
        with theme_context("unknown"):
            pass


def test_theme_context_restores_matplotlib_state():
    _, mpl = require_matplotlib()
    original = mpl.rcParams["font.size"]
    with theme_context("paper"):
        assert mpl.rcParams["font.size"] == 8
    assert mpl.rcParams["font.size"] == original
