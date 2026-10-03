"""Figure export and provenance sidecar tests."""

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from PIL import Image

matplotlib.use("Agg")

import scarf.plotting as splt

_N_CELLS = 30


class _Cells:
    def __init__(self, **columns):
        self._columns = {name: np.asarray(values) for name, values in columns.items()}
        self.columns = tuple(self._columns)
        self.N = _N_CELLS

    def fetch(self, column, key="I"):
        assert key == "I"
        return self._columns[column]

    def fetch_all(self, column):
        return self._columns[column]

    def active_index(self, key="I"):
        assert key == "I"
        return np.arange(self.N)


@pytest.fixture
def embedding_store():
    rng = np.random.default_rng(12)
    return SimpleNamespace(
        cells=_Cells(
            I=np.ones(_N_CELLS, dtype=bool),
            UMAP1=rng.normal(size=_N_CELLS),
            UMAP2=rng.normal(size=_N_CELLS),
            cluster=np.repeat(["a", "b", "c"], 10).astype(object),
        ),
        _defaultAssay="RNA",
        zw=None,
        _stored_display_metadata=lambda _column: None,
    )


def test_tiff_export_exact_size_and_provenance_sidecar(embedding_store, tmp_path):
    result = splt.embedding(
        embedding_store,
        layout_key="UMAP",
        color_by="cluster",
        figsize=(3.0, 2.0),
        show=False,
    )
    path = tmp_path / "figure.tiff"
    with matplotlib.rc_context({"savefig.bbox": "tight"}):
        result.save(path, dpi=120, provenance_sidecar=True)

    with Image.open(path) as image:
        assert image.format == "TIFF"
        # Exact-size export ignores a global tight bounding box.
        assert image.size == (360, 240)
        assert image.tag_v2.get(259) == 5  # LZW compression

    sidecar = tmp_path / "figure.tiff.json"
    payload = json.loads(sidecar.read_text())
    assert payload["provenance"]["renderer"] == "matplotlib"
    assert payload["provenance"]["n_cells"] == _N_CELLS
    assert payload["figure"] == {
        "dpi": 100.0,
        "height_inches": 2.0,
        "width_inches": 3.0,
    }
    assert payload["export"] == {
        "dpi": 120.0,
        "filename": "figure.tiff",
        "format": "tiff",
    }
    assert payload["legends"] == [
        {
            "extras": {"legend_loc": "right"},
            "kind": "categorical",
            "label": "cluster",
        }
    ]
    result.close()


def test_svg_export_preserves_physical_size_under_tight_global_setting(
    embedding_store, tmp_path
):
    result = splt.embedding(
        embedding_store,
        layout_key="UMAP",
        figsize=(3.0, 2.0),
        show=False,
    )
    path = tmp_path / "figure.svg"
    with matplotlib.rc_context({"savefig.bbox": "tight"}):
        result.save(path, exact_size=True)
    root = ET.parse(path).getroot()
    assert root.attrib["width"] == "216pt"
    assert root.attrib["height"] == "144pt"
    result.close()


def test_export_crops_only_without_exact_size(embedding_store, tmp_path):
    result = splt.embedding(
        embedding_store,
        layout_key="UMAP",
        color_by="cluster",
        figsize=(3.0, 2.0),
        show=False,
    )

    exact = result.save(tmp_path / "exact.png", dpi=100)
    cropped = result.save(tmp_path / "cropped.png", dpi=100, exact_size=False)

    with Image.open(exact) as image:
        assert image.size == (300, 200)
    with Image.open(cropped) as image:
        assert image.size != (300, 200)
    result.close()


@pytest.mark.parametrize(
    ("threshold", "rasterized"),
    [(_N_CELLS, True), (_N_CELLS + 1, False)],
)
def test_embedding_rasterizes_from_the_threshold_cell_count(
    embedding_store, threshold, rasterized
):
    result = splt.embedding(
        embedding_store,
        layout_key="UMAP",
        rasterize_threshold=threshold,
        show=False,
    )

    (points,) = next(iter(result.axes.values())).collections
    assert points.get_rasterized() is rasterized
    result.close()
