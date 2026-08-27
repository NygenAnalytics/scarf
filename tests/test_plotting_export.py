"""Milestone D export and provenance tests."""

import json
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
import pytest
from PIL import Image

matplotlib.use("Agg")

import scarf.plotting as splt


def test_tiff_export_exact_size_and_provenance_sidecar(
    umap, leiden_clustering, datastore, tmp_path
):
    result = splt.embedding(
        datastore,
        layout=umap,
        color_by=leiden_clustering,
        figsize=(3.0, 2.0),
        show=False,
    )
    path = tmp_path / "figure.tiff"
    with matplotlib.rc_context({"savefig.bbox": "tight"}):
        result.save(path, dpi=120, provenance_sidecar=True)

    with Image.open(path) as image:
        assert image.format == "TIFF"
        assert image.size == (360, 240)
        assert image.tag_v2.get(259) == 5  # LZW compression

    sidecar = tmp_path / "figure.tiff.json"
    payload = json.loads(sidecar.read_text())
    assert payload["provenance"]["renderer"] == "matplotlib"
    assert payload["provenance"]["n_cells"] == result.provenance.n_cells
    assert payload["figure"]["width_inches"] == pytest.approx(3.0)
    assert payload["export"] == {
        "dpi": 120.0,
        "filename": "figure.tiff",
        "format": "tiff",
    }
    assert "cluster" in payload["legends"][0]["label"]
    result.close()


def test_svg_export_preserves_physical_size_under_tight_global_setting(
    umap, datastore, tmp_path
):
    result = splt.embedding(
        datastore,
        layout=umap,
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


def test_save_provenance_handles_numpy_values(
    umap, leiden_clustering, datastore, tmp_path
):
    result = splt.embedding(
        datastore,
        layout=umap,
        color_by=leiden_clustering,
        show=False,
    )
    result.provenance.extras["array"] = np.array([1, 2], dtype=np.int64)
    result.provenance.extras["missing"] = np.float64(np.nan)
    path = result.save_provenance(tmp_path / "plot.json")
    payload = json.loads(path.read_text())
    assert payload["provenance"]["extras"]["array"] == [1, 2]
    assert payload["provenance"]["extras"]["missing"] is None
    result.close()


def test_export_validates_extension_and_dpi(umap, datastore, tmp_path):
    result = splt.embedding(
        datastore,
        layout=umap,
        show=False,
    )
    with pytest.raises(ValueError, match="file extension"):
        result.save(tmp_path / "no_extension")
    with pytest.raises(ValueError, match="dpi must be positive"):
        result.save(tmp_path / "plot.png", dpi=0)
    with pytest.raises(ValueError, match="Unsupported export format"):
        result.save(tmp_path / "plot.unsupported")
    cropped = result.save(tmp_path / "plot.png", exact_size=False)
    assert cropped.exists()
    result.close()


def test_embedding_rasterization_threshold(umap, datastore):
    vector = splt.embedding(
        datastore,
        layout=umap,
        rasterize_threshold=10**9,
        show=False,
    )
    rasterized = splt.embedding(
        datastore,
        layout=umap,
        rasterize_threshold=0,
        show=False,
    )
    vector_collection = next(iter(vector.axes.values())).collections[0]
    raster_collection = next(iter(rasterized.axes.values())).collections[0]
    assert vector_collection.get_rasterized() is False
    assert raster_collection.get_rasterized() is True
    vector.close()
    rasterized.close()
