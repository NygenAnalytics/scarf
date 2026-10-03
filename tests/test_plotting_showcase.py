"""Showcase generator smoke tests."""

import importlib.util
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image


_EXPECTED_OUTPUTS = {
    "categorical_embedding.png",
    "cell_cycle_scores.png",
    "cluster_connectivity.png",
    "composition.png",
    "continuous_embedding.png",
    "dark_embedding.png",
    "grouped_dotplot.png",
    "highlighted_embedding.png",
    "marker_heatmap.png",
    "matrix_plot.png",
    "publication_composite.png",
    "publication_composite.svg",
    "stacked_violin.png",
}


def _load_showcase_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_plotting_showcase.py"
    )
    spec = importlib.util.spec_from_file_location("scarf_plotting_showcase", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_isolated_store(module, datastore_zarr_root, tmp_path):
    work_directory = tmp_path / "showcase_store"
    work_directory.mkdir()
    return module._prepare_store(Path(datastore_zarr_root), work_directory)


def test_showcase_generator_has_offline_fixture_cli(capsys):
    module = _load_showcase_module()

    with pytest.raises(SystemExit) as exited:
        module.main(["--help"])

    assert exited.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--fixture", "--layout-fixture", "--output-dir"):
        assert option in help_text
    # The defaults point at committed offline fixtures.
    defaults = module._parser().parse_args([])
    repository = Path(__file__).resolve().parents[1]
    assert defaults.fixture == Path("tests/datasets/1K_pbmc_citeseq.zarr.tar.gz")
    assert defaults.layout_fixture == Path(
        "tests/visual/showcase/plotting_showcase_layout.npz"
    )
    assert defaults.output_dir == Path("plotting_showcase")
    assert (repository / defaults.layout_fixture).is_file()


def test_showcase_keeps_analysis_outputs_as_artifacts():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_plotting_showcase.py"
    )
    source = script.read_text(encoding="utf-8")

    assert ".cells.insert(" not in source
    assert ".cells.update_key(" not in source
    assert "layout_key=" not in source


@pytest.mark.slow
def test_generate_showcase_artifacts(
    datastore_zarr_root,
    tmp_path,
):
    requested = os.environ.get("SCARF_SHOWCASE_OUTPUT_DIR")
    output_directory = (
        Path(requested) if requested is not None else tmp_path / "showcase_outputs"
    )
    module = _load_showcase_module()
    store, artifacts = _prepare_isolated_store(
        module,
        datastore_zarr_root,
        tmp_path,
    )
    assert {
        "RNA_G2M_score",
        "RNA_S_score",
        "RNA_UMAP1",
        "RNA_UMAP2",
        "RNA_cell_cycle_phase",
        "RNA_leiden_cluster",
    }.isdisjoint(store.cells.columns)
    columns_before = set(store.cells.columns)
    outputs = module.generate_showcase(
        store,
        output_directory,
        layout=artifacts["layout"],
        graph=artifacts["graph"],
        clusters=artifacts["clusters"],
        cell_cycle=artifacts["cell_cycle"],
        features=artifacts["features"],
    )

    assert {path.name for path in outputs} == _EXPECTED_OUTPUTS
    columns_after = set(store.cells.columns)
    # Plots read analysis outputs as artifacts and never add cell metadata.
    assert columns_after == columns_before
    by_name = {path.name: path for path in outputs}
    for path in outputs:
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
    # The composite keeps its exact 11 x 6.5 inch page at 220 dpi.
    with Image.open(by_name["publication_composite.png"]) as image:
        assert image.size == (2420, 1430)
    svg = ET.parse(by_name["publication_composite.svg"]).getroot()
    assert (svg.attrib["width"], svg.attrib["height"]) == ("792pt", "468pt")


@pytest.mark.slow
@pytest.mark.visual
def test_showcase_matches_visual_references(
    datastore_zarr_root,
    tmp_path,
):
    if os.environ.get("SCARF_RUN_VISUAL_REGRESSION") != "1":
        pytest.skip("Set SCARF_RUN_VISUAL_REGRESSION=1 to compare showcase figures")
    from matplotlib.testing.compare import compare_images

    module = _load_showcase_module()
    store, artifacts = _prepare_isolated_store(
        module,
        datastore_zarr_root,
        tmp_path,
    )
    output_directory = tmp_path / "showcase_outputs"
    outputs = module.generate_showcase(
        store,
        output_directory,
        layout=artifacts["layout"],
        graph=artifacts["graph"],
        clusters=artifacts["clusters"],
        cell_cycle=artifacts["cell_cycle"],
        features=artifacts["features"],
    )
    assert {path.name for path in outputs} == _EXPECTED_OUTPUTS
    reference_dir = Path(__file__).parent / "visual" / "showcase"
    expected_pngs = {
        reference_dir / name for name in _EXPECTED_OUTPUTS if name.endswith(".png")
    }
    assert set(reference_dir.glob("*.png")) == expected_pngs
    tolerance = float(os.environ.get("SCARF_VISUAL_TOLERANCE", "1.5"))
    failures = []
    for actual in outputs:
        if actual.suffix != ".png":
            continue
        expected = reference_dir / actual.name
        if not expected.exists():
            failures.append(f"missing reference: {expected}")
            continue
        comparison = compare_images(
            str(expected),
            str(actual),
            tol=tolerance,
        )
        if comparison is not None:
            failures.append(str(comparison))
    assert not failures, "\n\n".join(failures)
