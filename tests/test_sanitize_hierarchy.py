"""Regression tests for sanitize_hierarchy layout contracts."""

import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.base_datastore import sanitize_hierarchy


def test_sanitize_hierarchy_accepts_legacy_and_workspace_layouts() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    assay = root.create_group("RNA")
    assay.create_group("featureData")
    assay.create_group("counts")
    assert sanitize_hierarchy(root, "RNA", None) is True

    workspace_root = zarr.open_group(store=MemoryStore(), mode="w")
    shell = workspace_root.create_group("ws")
    assay = shell.create_group("RNA")
    assay.create_group("featureData")
    matrices = workspace_root.create_group("matrices")
    matrices.create_group("RNA").create_group("counts")
    assert sanitize_hierarchy(workspace_root, "RNA", "ws") is True


def test_sanitize_hierarchy_rejects_missing_nodes() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(KeyError, match="RNA not found"):
        sanitize_hierarchy(root, "RNA", None)

    assay = root.create_group("RNA")
    with pytest.raises(KeyError, match="featureData"):
        sanitize_hierarchy(root, "RNA", None)

    assay.create_group("featureData")
    with pytest.raises(KeyError, match="counts"):
        sanitize_hierarchy(root, "RNA", None)

    workspace_root = zarr.open_group(store=MemoryStore(), mode="w")
    shell = workspace_root.create_group("ws")
    assay = shell.create_group("RNA")
    assay.create_group("featureData")
    with pytest.raises(KeyError, match="no 'matrices' slot"):
        sanitize_hierarchy(workspace_root, "RNA", "ws")

    matrices = workspace_root.create_group("matrices")
    with pytest.raises(KeyError, match="not found in workspace matrices"):
        sanitize_hierarchy(workspace_root, "RNA", "ws")

    matrices.create_group("RNA")
    with pytest.raises(KeyError, match="counts"):
        sanitize_hierarchy(workspace_root, "RNA", "ws")
