from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.storage.feature_selection as feature_selection
from scarf.storage.artifacts import (
    artifact_group,
    fingerprint_stored_arrays,
    fingerprint_stored_strings,
    inspect_artifact,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.feature_selection import _ValidatedFeatureSelection
from scarf.storage.refs import ArtifactRef, ExternalArtifactRef
from tests.test_feature_selection_resolution import _selection_store


def _feature_ref(value: str = "a") -> ArtifactRef:
    return ArtifactRef("assay", "feature_selection", value * 64, assay="RNA")


def _summary_ref(value: str = "b") -> ArtifactRef:
    return ArtifactRef("assay", "feature_summary", value * 64, assay="RNA")


class _Keys:
    def __init__(self, *names: str) -> None:
        self.names = names

    def array_keys(self) -> tuple[str, ...]:
        return self.names


def _status(
    *,
    operation: str | None,
    inputs: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    exists: bool = True,
    complete: bool = True,
) -> Any:
    return SimpleNamespace(
        operation=operation,
        inputs={} if inputs is None else inputs,
        parameters={} if parameters is None else parameters,
        exists=exists,
        complete=complete,
    )


def test_feature_selection_write_and_feature_table_contracts() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = feature_selection._feature_selection_plan(
        root,
        assay="RNA",
        n_features=3,
        ordered_feature_ids_fingerprint="rows",
        operation="set_feature_selection",
        parameters={"values_fingerprint": "values"},
        inputs={"all_features": _feature_ref().to_dict()},
        execution_options={},
        payload_names=("values", "corrected_variance"),
        invalidate_cache=True,
    )
    with pytest.raises(ValueError, match="shape"):
        feature_selection._write_feature_selection(
            root,
            planned,
            ordered_feature_ids_fingerprint="rows",
            payload={
                "values": np.ones(3, dtype=bool),
                "corrected_variance": np.ones(2),
            },
            payload_names=("values", "corrected_variance"),
        )

    empty = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._feature_data(empty, "RNA")
    assert caught.value.code == "wrong_assay"
    rna = empty.create_group("RNA")
    rna.create_array("featureData", data=np.arange(2))
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._feature_data(empty, "RNA")
    assert caught.value.code == "corrupt_payload"
    del empty["RNA/featureData"]
    empty.create_group("RNA/featureData")
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._feature_data(empty, "RNA")
    assert caught.value.code == "row_mismatch"


def test_feature_ref_scope_payload_and_local_input_contracts() -> None:
    wrong_scope = ArtifactRef("datastore", "feature_selection", "1" * 64)
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_ref_scope(wrong_scope, "RNA")
    assert caught.value.code == "wrong_scope"
    wrong_assay = ArtifactRef("assay", "feature_selection", "2" * 64, assay="ADT")
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_ref_scope(wrong_assay, "RNA")
    assert caught.value.code == "wrong_assay"

    root = zarr.open_group(store=MemoryStore(), mode="w")
    group = root.create_group("payload")
    group.create_array("values", data=np.ones(2, dtype=bool))
    group.create_array("unexpected", data=np.ones(2))
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._payload_names(group)
    assert caught.value.code == "corrupt_payload"

    assert feature_selection._local_input_ref(1) is None
    assert (
        feature_selection._local_input_ref(
            {"type": "external_artifact", "dataset_fingerprint": "x", "ref": {}}
        )
        is None
    )
    assert feature_selection._local_input_ref({"type": "artifact"}) is None
    invalid = _feature_ref().to_dict()
    invalid["artifact_id"] = "bad"
    assert feature_selection._local_input_ref(invalid) is None


def test_feature_summary_parent_reference_and_status_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_summary_parent(root, "RNA", _feature_ref())
    assert caught.value.code == "wrong_kind"
    wrong_assay = ArtifactRef("assay", "feature_summary", "1" * 64, assay="ADT")
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_summary_parent(root, "RNA", wrong_assay)
    assert caught.value.code == "wrong_assay"

    summary = _summary_ref()
    monkeypatch.setattr(
        feature_selection,
        "inspect_artifact",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_summary_parent(root, "RNA", summary)
    assert caught.value.code == "corrupt_payload"

    for status, code in (
        (_status(operation="summarize_rna_features", exists=False), "missing_artifact"),
        (
            _status(operation="summarize_rna_features", complete=False),
            "incomplete_artifact",
        ),
        (_status(operation="other"), "corrupt_payload"),
        (
            _status(
                operation="summarize_rna_features",
                parameters={"normalization_method": "log"},
            ),
            "corrupt_payload",
        ),
        (
            _status(
                operation="summarize_rna_features",
                parameters={"normalization_method": "log", "size_factor": 1},
                inputs={"cell_selection": {}},
            ),
            "corrupt_payload",
        ),
    ):
        monkeypatch.setattr(
            feature_selection, "inspect_artifact", lambda *_args, s=status: s
        )
        with pytest.raises(ArtifactResolutionError) as caught:
            feature_selection._validate_feature_summary_parent(root, "RNA", summary)
        assert caught.value.code == code


def test_feature_summary_parent_wraps_cell_status_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    summary = _summary_ref()
    cell_ref = ArtifactRef("datastore", "cell_selection", "c" * 64)
    summary_status = _status(
        operation="summarize_rna_features",
        parameters={"normalization_method": "log", "size_factor": 1},
        inputs={"cell_selection": cell_ref.to_dict()},
    )

    def malformed(_root: zarr.Group, ref: ArtifactRef) -> Any:
        if ref == summary:
            return summary_status
        raise ValueError("bad cell record")

    monkeypatch.setattr(feature_selection, "inspect_artifact", malformed)
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_summary_parent(root, "RNA", summary)
    assert caught.value.code == "corrupt_payload"

    for cell_status, code in (
        (_status(operation="selection", exists=False), "missing_artifact"),
        (_status(operation="selection", complete=False), "incomplete_artifact"),
    ):
        monkeypatch.setattr(
            feature_selection,
            "inspect_artifact",
            lambda _root, ref, c=cell_status: summary_status if ref == summary else c,
        )
        with pytest.raises(ArtifactResolutionError) as caught:
            feature_selection._validate_feature_summary_parent(root, "RNA", summary)
        assert caught.value.code == code


def _summary_payload_root() -> tuple[zarr.Group, zarr.Group, ArtifactRef, ArtifactRef]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    feature_data = root.create_group("RNA/featureData")
    feature_data.create_array("ids", data=np.asarray(["g1", "g2"]))
    payload = root.create_group("summary_payload")
    for name in ("normed_tot", "normed_n", "sigmas"):
        payload.create_array(name, data=np.ones(2, dtype=np.float64))
    payload.attrs["ordered_feature_ids_fingerprint"] = fingerprint_stored_strings(
        feature_data["ids"]
    )
    payload.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        payload, ("normed_tot", "normed_n", "sigmas")
    )
    return (
        root,
        payload,
        _summary_ref(),
        ArtifactRef("datastore", "cell_selection", "c" * 64),
    )


def _patch_valid_summary_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    payload: zarr.Group,
    summary: ArtifactRef,
    cell_ref: ArtifactRef,
) -> None:
    summary_status = _status(
        operation="summarize_rna_features",
        parameters={"normalization_method": "log", "size_factor": 1},
        inputs={"cell_selection": cell_ref.to_dict()},
    )
    monkeypatch.setattr(
        feature_selection,
        "inspect_artifact",
        lambda _root, ref: (
            summary_status if ref == summary else _status(operation="selection")
        ),
    )
    monkeypatch.setattr(
        feature_selection, "validate_stored_selection_integrity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(feature_selection, "artifact_group", lambda *_args: payload)


def test_feature_summary_payload_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, payload, summary, cell_ref = _summary_payload_root()
    _patch_valid_summary_dependencies(monkeypatch, payload, summary, cell_ref)
    payload.create_array("unexpected", data=np.ones(2))
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_summary_parent(root, "RNA", summary)
    assert caught.value.code == "corrupt_payload"

    root, payload, summary, cell_ref = _summary_payload_root()
    _patch_valid_summary_dependencies(monkeypatch, payload, summary, cell_ref)
    del payload["sigmas"]
    payload.create_array("sigmas", data=np.ones(2, dtype=np.int64))
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_summary_parent(root, "RNA", summary)
    assert caught.value.code == "corrupt_payload"

    root, payload, summary, cell_ref = _summary_payload_root()
    _patch_valid_summary_dependencies(monkeypatch, payload, summary, cell_ref)
    payload.attrs["ordered_feature_ids_fingerprint"] = "changed"
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_summary_parent(root, "RNA", summary)
    assert caught.value.code == "row_mismatch"

    root, payload, summary, cell_ref = _summary_payload_root()
    _patch_valid_summary_dependencies(monkeypatch, payload, summary, cell_ref)
    payload.attrs["payload_fingerprint"] = "changed"
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_summary_parent(root, "RNA", summary)
    assert caught.value.code == "corrupt_payload"


def test_feature_selection_provenance_contract_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _feature_ref()

    cases = (
        (_status(operation="unknown"), _Keys("values"), "operation"),
        (
            _status(operation="set_feature_selection"),
            _Keys("values"),
            "provenance",
        ),
        (
            _status(
                operation="set_feature_selection",
                inputs={"all_features": _feature_ref("b").to_dict()},
                parameters={"values_fingerprint": "x"},
            ),
            _Keys("other"),
            "payload arrays",
        ),
        (
            _status(
                operation="set_feature_selection",
                inputs={"all_features": {}},
                parameters={"values_fingerprint": "x"},
            ),
            _Keys("values"),
            "universe input",
        ),
        (
            _status(
                operation="select_detected_features",
                inputs={"feature_summary": {}},
                parameters={"min_cells": 1},
            ),
            _Keys("values"),
            "summary input",
        ),
    )
    for status, group, message in cases:
        with pytest.raises(ArtifactResolutionError, match=message):
            feature_selection._validate_feature_selection_provenance(
                root,
                "RNA",
                ref,
                status,
                group,
                seen=set(),  # type: ignore[arg-type]
            )

    all_ref = _feature_ref("b")
    status = _status(
        operation="set_feature_selection",
        inputs={"all_features": all_ref.to_dict()},
        parameters={"values_fingerprint": "x"},
    )
    monkeypatch.setattr(
        feature_selection,
        "_validate_feature_selection",
        lambda *_args, **_kwargs: _ValidatedFeatureSelection(
            all_ref,
            object(),
            "set_feature_selection",  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(ArtifactResolutionError, match="not all_features"):
        feature_selection._validate_feature_selection_provenance(
            root,
            "RNA",
            ref,
            status,
            _Keys("values"),
            seen=set(),  # type: ignore[arg-type]
        )


def test_feature_selection_snapshot_and_mapping_input_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _feature_ref()
    hvg_parameters = {
        "min_cells": 1,
        "max_cells": 10,
        "top_n": 2,
        "min_var": 0,
        "max_var": 1,
        "min_mean": 0,
        "max_mean": 1,
        "n_bins": 2,
        "lowess_frac": 0.3,
        "blacklist": [],
        "keep_bounds": False,
        "bin_strategy": "equal_width",
    }
    status = _status(
        operation="select_hvgs",
        inputs={"feature_snapshot": {}, "feature_summary": _summary_ref().to_dict()},
        parameters=hvg_parameters,
    )
    monkeypatch.setattr(
        feature_selection, "_validate_feature_summary_parent", lambda *_args: None
    )
    with pytest.raises(ArtifactResolutionError, match="snapshot input"):
        feature_selection._validate_feature_selection_provenance(
            root,
            "RNA",
            ref,
            status,
            _Keys("values", "corrected_variance"),  # type: ignore[arg-type]
            seen=set(),
        )

    all_ref = _feature_ref("d")
    monkeypatch.setattr(
        feature_selection,
        "_validate_feature_selection",
        lambda *_args, **_kwargs: _ValidatedFeatureSelection(
            all_ref,
            object(),
            "create_all_features",  # type: ignore[arg-type]
        ),
    )
    mapping_cases: tuple[tuple[Any, str], ...] = (
        (1, "malformed"),
        ({}, "malformed"),
        (
            ExternalArtifactRef(
                "dataset",
                ArtifactRef("assay", "feature_selection", "e" * 64, assay="RNA"),
            ).to_dict(),
            "wrong kind",
        ),
    )
    for raw, message in mapping_cases:
        status = _status(
            operation="select_mapping_overlap",
            inputs={"mapping_reference": raw, "all_features": all_ref.to_dict()},
            parameters={},
        )
        with pytest.raises(ArtifactResolutionError, match=message):
            feature_selection._validate_feature_selection_provenance(
                root,
                "RNA",
                ref,
                status,
                _Keys("values"),  # type: ignore[arg-type]
                seen=set(),
            )


def test_feature_selection_validation_wraps_corrupt_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _store, ref = _selection_store()
    with pytest.raises(ArtifactResolutionError, match="cycle"):
        feature_selection._validate_feature_selection(root, "RNA", ref, seen={ref})

    monkeypatch.setattr(
        feature_selection,
        "inspect_artifact",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_selection(root, "RNA", ref)
    assert caught.value.code == "corrupt_payload"

    monkeypatch.undo()
    group = artifact_group(root, ref)
    del group["values"]
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection._validate_feature_selection(root, "RNA", ref)
    assert caught.value.code == "corrupt_payload"


def test_feature_universe_and_empty_index_contracts() -> None:
    root, _store, ref = _selection_store(np.zeros(4, dtype=bool))
    indices = feature_selection.read_feature_selection_indices(root, "RNA", ref)
    assert indices.dtype == np.intp and indices.size == 0

    status = inspect_artifact(root, ref)
    parent = ArtifactRef.from_dict((status.inputs or {})["all_features"])
    group = artifact_group(root, parent)
    group["values"][0] = False
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, ("values",))
    with pytest.raises(ArtifactResolutionError, match="select every feature"):
        feature_selection.resolve_feature_selection(root, "RNA", parent)


def test_feature_selection_operation_specific_identity_errors() -> None:
    root, _store, ref = _selection_store()
    group = artifact_group(root, ref)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters["values_fingerprint"] = "changed"
    provenance["parameters"] = parameters
    group.attrs["provenance"] = provenance
    with pytest.raises(ArtifactResolutionError, match="value identity"):
        feature_selection.resolve_feature_selection(root, "RNA", ref)

    root, _store, ref = _selection_store()
    status = inspect_artifact(root, ref)
    parent = ArtifactRef.from_dict((status.inputs or {})["all_features"])
    group = artifact_group(root, parent)
    provenance = dict(group.attrs["provenance"])
    parameters = dict(provenance["parameters"])
    parameters["ordered_feature_ids_fingerprint"] = "changed"
    provenance["parameters"] = parameters
    group.attrs["provenance"] = provenance
    with pytest.raises(ArtifactResolutionError) as caught:
        feature_selection.resolve_feature_selection(root, "RNA", parent)
    assert caught.value.code == "row_mismatch"
