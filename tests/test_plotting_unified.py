"""Tests for scarf.plotting.unified_embedding."""

import matplotlib
import pytest

matplotlib.use("Agg")

import scarf.plotting as splt


def test_unified_embedding_matches_projection_rows(run_unified_umap, datastore):
    result = splt.unified_embedding(
        datastore,
        layout_key="unified_UMAP",
        show=False,
    )
    layout = datastore.z["RNA"]["projections"]["unified_UMAP"]
    assert result.provenance.n_cells == layout.shape[0]
    assert result.tables["cells"].shape[0] == layout.shape[0]
    ax = next(iter(result.axes.values()))
    assert ax.get_box_aspect() == pytest.approx(1.0)
    assert result.figure.legends or ax.get_legend() is not None
    result.close()


def test_unified_embedding_show_target_only_and_groups(
    run_unified_umap, paris_clustering, datastore
):
    layout = datastore.z["RNA"]["projections"]["unified_UMAP"]
    n_cells = list(layout.attrs["n_cells"])
    n_target = int(sum(n_cells[1:]))
    groups = [f"g{i % 3}" for i in range(n_target)]
    result = splt.unified_embedding(
        datastore,
        layout_key="unified_UMAP",
        show_target_only=True,
        target_groups=groups,
        show=False,
    )
    assert result.provenance.n_cells == n_target
    assert "reference" not in set(result.tables["cells"]["group"])
    result.close()


def test_unified_embedding_rejects_bad_target_groups(run_unified_umap, datastore):
    with pytest.raises(ValueError, match="target_groups length"):
        splt.unified_embedding(
            datastore,
            layout_key="unified_UMAP",
            target_groups=["only-one"],
            show=False,
        )
