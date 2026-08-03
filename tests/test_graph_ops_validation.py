"""Early-exit validation tests for graph operations."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.graph import _sampling_fraction
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.storage.artifacts import ArtifactRef


class _BareGraphStore(GraphDataStore):
    @property
    def assay_names(self) -> list[str]:
        return []


def _bare_store() -> _BareGraphStore:
    store = object.__new__(_BareGraphStore)
    store.z = zarr.open_group(store=MemoryStore(), mode="w")
    store.workspace = None
    store.zarr_mode = "r+"
    store._defaultAssay = None
    store.nthreads = 1
    return store


def test_sampling_fraction_validates_type_and_range():
    assert _sampling_fraction(0.5, "frac") == 0.5
    assert _sampling_fraction(1, "frac") == 1.0
    with pytest.raises(TypeError, match="must be a number"):
        _sampling_fraction(True, "frac")
    with pytest.raises(TypeError, match="must be a number"):
        _sampling_fraction("x", "frac")
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        _sampling_fraction(0.0, "frac")
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        _sampling_fraction(1.1, "frac")
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        _sampling_fraction(float("nan"), "frac")


def test_get_latest_keys_requires_default_assay_and_skips_resolution_when_explicit():
    store = _bare_store()
    with pytest.raises(ValueError, match="No default assay"):
        store._get_latest_keys(None, None, None)

    store._defaultAssay = "RNA"
    store._get_latest_cell_key = Mock(
        side_effect=AssertionError("explicit keys must not resolve cell key")
    )
    store._get_latest_feat_key = Mock(
        side_effect=AssertionError("explicit keys must not resolve feat key")
    )
    assert store._get_latest_keys("ADT", "custom", "feats") == (
        "ADT",
        "custom",
        "feats",
    )
    store._get_latest_cell_key.assert_not_called()
    store._get_latest_feat_key.assert_not_called()


def test_require_complete_artifact_rejects_kind_and_assay_mismatch(monkeypatch):
    store = _bare_store()
    ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="b" * 64,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "storage lookup must not run for local validation failures"
        )

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.require_complete_artifact",
        fail_if_called,
    )
    with pytest.raises(ValueError, match="Expected 'ann_index' artifact"):
        store._require_complete_artifact(ref, "ann_index")

    wrong_assay = ArtifactRef(
        scope="assay",
        assay="ADT",
        kind="neighbors",
        artifact_id="c" * 64,
    )
    with pytest.raises(ValueError, match="must belong to assay 'RNA'"):
        store._require_complete_artifact(wrong_assay, "neighbors", assay="RNA")


def test_artifact_input_ref_requires_named_input():
    store = _bare_store()
    ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="b" * 64,
    )
    store._require_complete_artifact = Mock(return_value=SimpleNamespace(inputs={}))
    with pytest.raises(ValueError, match="has no 'coordinates' input"):
        store._artifact_input_ref(ref, "coordinates", "reduction")


def test_run_lsi_and_custom_reduction_validate_before_work():
    store = _bare_store()
    store._run_reduction_artifact = Mock(
        side_effect=AssertionError("reduction must not run")
    )

    with pytest.raises(ValueError, match="solver must be"):
        store.run_lsi(solver="mystery")
    with pytest.raises(TypeError, match="n_iter must be an integer"):
        store.run_lsi(n_iter=True)
    with pytest.raises(ValueError, match="n_iter must be nonnegative"):
        store.run_lsi(n_iter=-1)
    with pytest.raises(TypeError, match="n_oversamples must be an integer"):
        store.run_lsi(n_oversamples=True)
    with pytest.raises(ValueError, match="n_oversamples must be nonnegative"):
        store.run_lsi(n_oversamples=-2)

    with pytest.raises(ValueError, match="two-dimensional matrix"):
        store.run_custom_reduction(np.arange(4))
    with pytest.raises(ValueError, match="two-dimensional matrix"):
        store.run_custom_reduction(np.zeros((5, 0)))
    store._run_reduction_artifact.assert_not_called()
