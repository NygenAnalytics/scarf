"""Showcase generator smoke tests."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_showcase_generator_has_offline_fixture_cli():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_plotting_showcase.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--fixture" in completed.stdout
    assert "--layout-fixture" in completed.stdout
    assert "--output-dir" in completed.stdout


@pytest.mark.slow
def test_generate_requested_showcase_artifacts(
    datastore_zarr_root,
    tmp_path,
):
    requested = os.environ.get("SCARF_SHOWCASE_OUTPUT_DIR")
    if requested is None:
        pytest.skip("Set SCARF_SHOWCASE_OUTPUT_DIR to regenerate review figures")
    module = _load_showcase_module()
    store = _prepare_isolated_store(
        module,
        datastore_zarr_root,
        tmp_path,
    )
    outputs = module.generate_showcase(store, Path(requested))

    assert {path.name for path in outputs} == _EXPECTED_OUTPUTS
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


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
    store = _prepare_isolated_store(
        module,
        datastore_zarr_root,
        tmp_path,
    )
    output_directory = tmp_path / "showcase_outputs"
    outputs = module.generate_showcase(store, output_directory)
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
