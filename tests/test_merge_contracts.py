import inspect
import pickle
import subprocess
import sys
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import scarf
import scarf.merge as merge_module
from scarf.merge import AssayMerge, DatasetMerge, DummyAssay
from tests.signature_contracts import signature_digest


_PUBLIC_CLASS_METHODS = {
    AssayMerge: (
        "__init__",
        "perform_randomization_rows",
        "check_feat_ids",
        "get_feat_suffix",
        "update_feat_ids",
        "update_feat_ids_for_map",
        "dump",
    ),
    DatasetMerge: (
        "__init__",
        "get_unique_assays",
        "create_merge_generators",
        "generate_dummy_assay",
        "dump",
    ),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    AssayMerge: "28945aa6ef349e023f76969f30fd5b97fd95bb762b6b005106c3cd1ee0d7ac21",
    DatasetMerge: "c556a12844cd893c19ae605443bce2f384fa1d9277dfe157a3b5e1ee456d7189",
}
_FACADE_METHODS = {DummyAssay: ("__init__",), **_PUBLIC_CLASS_METHODS}


def test_merge_facade_surface_is_stable():
    assert merge_module.__all__ == [
        "DatasetMerge",
        "AssayMerge",
    ]
    expected = {
        "AssayMerge",
        "DatasetMerge",
        "DummyAssay",
        "MergeAssay",
        "_RowPlan",
        "controlled_compute",
        "load_zarr",
    }
    assert expected.issubset(dir(merge_module))
    assert all(getattr(merge_module, name) is not None for name in expected)
    assert not hasattr(merge_module, "ZarrMerge")


def test_merge_class_and_method_signatures_are_stable():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        methods = {name: getattr(cls, name) for name in names}
        assert signature_digest(methods) == _PUBLIC_CLASS_SIGNATURE_DIGESTS[cls]


def test_merge_class_and_method_metadata_remains_on_facade():
    for cls, names in _FACADE_METHODS.items():
        assert cls.__module__ == "scarf.merge"
        for name in names:
            descriptor = inspect.getattr_static(cls, name)
            if isinstance(descriptor, staticmethod):
                method = descriptor.__func__
            else:
                method = descriptor
            assert method.__module__ == "scarf.merge"
            assert method.__qualname__.startswith(f"{cls.__name__}.")


def test_merge_descriptor_and_hierarchy_contracts_are_stable():
    assert isinstance(
        inspect.getattr_static(AssayMerge, "_get_feat_ids"),
        staticmethod,
    )


def test_merge_facade_classes_remain_pickle_resolvable():
    for cls in _FACADE_METHODS:
        assert pickle.loads(pickle.dumps(cls)) is cls


def test_merge_datastore_type_hints_resolve_lazily():
    assert get_type_hints(DummyAssay.__init__)["ds"] is scarf.DataStore
    assert get_type_hints(DatasetMerge.__init__)["datasets"] == list[scarf.DataStore]
    assert get_type_hints(DatasetMerge.generate_dummy_assay)["ds"] is scarf.DataStore


def test_dataset_merge_resolves_assay_factory_from_facade(monkeypatch):
    assay = SimpleNamespace(rawData=SimpleNamespace(chunksize=(2, 2)))
    dataset = SimpleNamespace(
        assay_names=["RNA"],
        get_assay=lambda name: assay,
    )
    generator = SimpleNamespace(
        permutations_rows={0: {0: np.array([0])}},
        permutations_rows_offset={0: {0: np.array([0])}},
        coordinates_permutations=np.array([[0, 0]]),
    )
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return generator

    monkeypatch.setattr(merge_module, "AssayMerge", factory)
    merge = object.__new__(DatasetMerge)
    merge.datasets = [dataset]
    merge.names = ["sample"]
    merge.zarr_path = "merged.zarr"
    merge.in_workspaces = None
    merge.out_workspace = None
    merge.dtype = None
    merge.overwrite = False
    merge.prepend_text = "orig"
    merge.reset_cell_filter = True
    merge.seed = 42
    merge.storage_options = None
    merge.source_column = None
    merge.memBudget = 1024**3
    merge.nthreads = 2
    merge.profile = "fast_local"
    merge.targetChunkBytes = None
    merge.targetShardBytes = None
    merge.unique_assays = ["RNA"]

    assert merge.create_merge_generators() == [generator]
    assert captured["assays"] == [assay]
    assert captured["merge_assay_name"] == "RNA"


def test_dataset_merge_resolves_dummy_factory_from_facade(monkeypatch):
    reference_assay = SimpleNamespace(
        rawData=SimpleNamespace(chunksize=(2, 2), dtype=np.dtype("uint16")),
        feats=SimpleNamespace(N=2),
        name="RNA",
    )
    dataset = SimpleNamespace(
        assay_names=["RNA"],
        cells=SimpleNamespace(N=3),
        get_assay=lambda name: reference_assay,
        nthreads=2,
        resources=None,
    )
    sentinel = object()
    captured = {}

    def factory(*args):
        captured["args"] = args
        return sentinel

    monkeypatch.setattr(merge_module, "DummyAssay", factory)
    merge = object.__new__(DatasetMerge)
    merge.datasets = [dataset]

    assert merge.generate_dummy_assay(dataset, "RNA") is sentinel
    assert captured["args"][0] is dataset
    assert captured["args"][2] is reference_assay.feats
    assert captured["args"][3] == "RNA"


def test_merge_facade_loads_implementations_lazily_and_clears_on_reload():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys

import scarf.merge as merge

assert "scarf.merge.assays" not in sys.modules
assert "scarf.merge.datasets" not in sys.modules
assert "scarf.datastore.datastore" not in sys.modules
assert not any(
    name == "scarf.assay" or name.startswith("scarf.assay.")
    for name in sys.modules
)
assert not {"pandas", "scipy", "zarr"}.intersection(sys.modules)

assay_merge = merge.AssayMerge
assert "scarf.merge.assays" in sys.modules
assert "scarf.merge.datasets" not in sys.modules
assert "scarf.datastore.datastore" not in sys.modules
assert {
    "AssayMerge",
    "DummyAssay",
    "MergeAssay",
    "_RowPlan",
    "controlled_compute",
    "load_zarr",
} <= vars(merge).keys()
assert "ZarrMerge" not in vars(merge)

dataset_merge = merge.DatasetMerge
assert "scarf.merge.datasets" in sys.modules
assert "scarf.datastore.datastore" not in sys.modules
for cls in (assay_merge, dataset_merge, merge.DummyAssay):
    assert cls.__module__ == "scarf.merge"
    for descriptor in cls.__dict__.values():
        if isinstance(descriptor, (classmethod, staticmethod)):
            method = descriptor.__func__
        else:
            method = descriptor
        if callable(method) and hasattr(method, "__module__"):
            assert method.__module__ == "scarf.merge"

merge.AssayMerge = object()
importlib.reload(merge)
assert not {
    "AssayMerge",
    "DatasetMerge",
    "DummyAssay",
    "MergeAssay",
    "_RowPlan",
    "controlled_compute",
    "load_zarr",
}.intersection(vars(merge))
assert merge.AssayMerge is assay_merge
""",
        ],
        check=True,
    )
