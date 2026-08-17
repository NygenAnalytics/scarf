import inspect
import pickle
import subprocess
import sys
from dataclasses import fields
from typing import get_type_hints

import scarf.merge as merge_module
from scarf.merge import (
    AssayMergePlan,
    ComponentResult,
    DataStoreMerge,
    MergePlan,
    MergeResult,
)
from tests.signature_contracts import signature_digest


_PUBLIC_CLASS_METHODS = {
    DataStoreMerge: (
        "__init__",
        "plan",
        "dump",
    ),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    DataStoreMerge: "96cfe6d07ac5536f04d1cd333a42f58957fe3ab5d89c69b65acea72b2d5bf5f6",
}
_FACADE_METHODS = {
    MergePlan: (),
    MergeResult: (),
    AssayMergePlan: (),
    ComponentResult: (),
    **_PUBLIC_CLASS_METHODS,
}


def test_merge_facade_surface_is_stable():
    assert merge_module.__all__ == [
        "DataStoreMerge",
        "MergePlan",
        "MergeResult",
        "AssayMergePlan",
        "ComponentResult",
    ]
    expected = {
        "DataStoreMerge",
        "MergePlan",
        "MergeResult",
        "AssayMergePlan",
        "ComponentResult",
    }
    assert expected.issubset(dir(merge_module))
    assert all(getattr(merge_module, name) is not None for name in expected)
    assert not hasattr(merge_module, "AssayMerge")
    assert not hasattr(merge_module, "DatasetMerge")
    assert not hasattr(merge_module, "ZarrMerge")
    assert not hasattr(merge_module, "DummyAssay")


def test_merge_class_and_method_signatures_are_stable():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        methods = {name: getattr(cls, name) for name in names}
        assert signature_digest(methods) == _PUBLIC_CLASS_SIGNATURE_DIGESTS[cls]


def test_merge_plan_and_result_fields_are_stable():
    expected = {
        AssayMergePlan: (
            "assayName",
            "assayType",
            "sourcePresent",
            "missingSources",
            "nFeatures",
            "featureOverlapFraction",
            "dtype",
            "chunks",
            "shards",
            "writeCountsT",
            "estimatedWriteTasks",
            "countsAction",
            "countsTAction",
        ),
        MergePlan: (
            "zarrPath",
            "outWorkspace",
            "sourceNames",
            "nCells",
            "assays",
            "profile",
            "seed",
            "missingAssayPolicy",
            "willResume",
            "canDump",
            "blockedReason",
            "cellDataAction",
            "manifest",
        ),
        ComponentResult: ("name", "action"),
        MergeResult: (
            "zarrPath",
            "nCells",
            "assayNames",
            "components",
            "resumed",
        ),
    }
    for model, field_names in expected.items():
        assert tuple(field.name for field in fields(model)) == field_names


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


def test_merge_facade_classes_remain_pickle_resolvable():
    for cls in _FACADE_METHODS:
        assert pickle.loads(pickle.dumps(cls)) is cls


def test_dataset_merge_type_hints_resolve_lazily():
    hints = get_type_hints(DataStoreMerge.__init__)
    assert "datasets" in hints
    assert "zarr_path" in hints
    assert "counts_t" not in hints
    assert "missing_assay_policy" in hints


def test_merge_facade_loads_implementations_lazily_and_clears_on_reload():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys

import scarf.merge as merge

assert "scarf.merge.datasets" not in sys.modules
assert "scarf.datastore.datastore" not in sys.modules
assert not {"pandas", "scipy", "zarr"}.intersection(sys.modules)

dataset_merge = merge.DataStoreMerge
assert "scarf.merge.datasets" in sys.modules
assert "scarf.datastore.datastore" not in sys.modules
assert {
    "DataStoreMerge",
    "MergePlan",
    "MergeResult",
    "AssayMergePlan",
    "ComponentResult",
} <= vars(merge).keys()
assert "AssayMerge" not in vars(merge)
assert "DatasetMerge" not in vars(merge)
assert "DummyAssay" not in vars(merge)

for cls in (
    dataset_merge,
    merge.MergePlan,
    merge.MergeResult,
    merge.AssayMergePlan,
    merge.ComponentResult,
):
    assert cls.__module__ == "scarf.merge"

merge.DataStoreMerge = object()
importlib.reload(merge)
assert not {
    "DataStoreMerge",
    "MergePlan",
    "MergeResult",
    "AssayMergePlan",
    "ComponentResult",
}.intersection(vars(merge))
assert merge.DataStoreMerge is dataset_merge
""",
        ],
        check=True,
    )
