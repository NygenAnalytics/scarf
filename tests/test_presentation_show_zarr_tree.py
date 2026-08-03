"""Regression tests for DataStore presentation helpers."""

import pytest


def test_show_zarr_tree_prints_hierarchy(datastore_ephemeral, capsys):
    store = datastore_ephemeral
    store.show_zarr_tree(start="/", depth=1)
    captured = capsys.readouterr().out

    assert captured.strip()
    for assay_name in store.assay_names:
        assert assay_name in captured
    root_names = set(store.zw.group_keys()) | set(store.zw.array_keys())
    assert root_names
    assert any(name in captured for name in root_names)

    with pytest.raises(KeyError):
        store.show_zarr_tree(start="does_not_exist", depth=1)
