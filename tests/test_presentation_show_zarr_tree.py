"""Regression tests for DataStore presentation helpers."""

import pytest


def test_show_zarr_tree_prints_hierarchy(datastore_ephemeral, capsys):
    datastore_ephemeral.show_zarr_tree(start="/", depth=1)
    captured = capsys.readouterr().out
    assert "RNA" in captured or "cellData" in captured or "matrices" in captured

    with pytest.raises(KeyError):
        datastore_ephemeral.show_zarr_tree(start="does_not_exist", depth=1)
