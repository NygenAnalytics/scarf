"""Imported embedding helpers consume DataStore artifacts without a catalog."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from scarf.cytebase._embeddings import embedding, embedding_coordinates, embeddings
from scarf.storage.artifacts import ArtifactRef
from tests.fixtures_cytebase import CYTEBASE_ID, UMAP

pytestmark = pytest.mark.usefixtures("cytebase_offline")


def _ref(token="a", assay="RNA"):
    return ArtifactRef(
        scope="assay", kind="embedding", assay=assay, artifact_id=token * 64
    )


@pytest.mark.parametrize("assay", ["RNA", "ATAC"])
def test_discovery_uses_import_provenance_and_requested_assay(assay):
    ref = _ref(assay=assay)
    requests = []

    def list_artifacts(**kwargs):
        requests.append(kwargs)
        if kwargs.get("parameters", {}).get("dimreduc_key", "X_umap") != "X_umap":
            return []
        return [ref]

    store = SimpleNamespace(
        list_artifacts=list_artifacts,
        inspect_artifact=lambda ref: SimpleNamespace(
            parameters={"dimreduc_key": "X_umap"}
        ),
    )
    assert embeddings(store, assay=assay) == {"X_umap": ref}
    assert embedding(store, assay=assay) == ref
    expected = {
        "kind": "embedding",
        "from_assay": assay,
        "operation": "import_dimreduc",
        "complete_only": True,
    }
    assert requests == [expected, expected | {"parameters": {"dimreduc_key": "X_umap"}}]
    with pytest.raises(KeyError, match="'X_pca' was not imported .* X_umap"):
        embedding(store, "X_pca", assay=assay)


def test_missing_embeddings_report_an_empty_inventory():
    store = SimpleNamespace(list_artifacts=lambda **kwargs: [])
    assert embeddings(store) == {}
    with pytest.raises(KeyError, match="available: none"):
        embedding(store)


def test_unique_key_lookup_ignores_ambiguous_unrelated_embeddings():
    refs = {"X_umap": [_ref("a")], "X_tsne": [_ref("b"), _ref("c")]}

    def list_artifacts(**kwargs):
        return refs[kwargs["parameters"]["dimreduc_key"]]

    # Successful lookup needs no metadata inspection or whole-store inventory.
    store = SimpleNamespace(list_artifacts=list_artifacts)
    assert embedding(store) == refs["X_umap"][0]
    with pytest.raises(ValueError, match="'X_tsne' is ambiguous"):
        embedding(store, "X_tsne")


@pytest.mark.parametrize("lookup", [embeddings, embedding])
def test_duplicate_import_keys_require_an_explicit_reference(lookup):
    store = SimpleNamespace(
        list_artifacts=lambda **kwargs: [_ref("a"), _ref("b")],
        inspect_artifact=lambda ref: SimpleNamespace(
            parameters={"dimreduc_key": "X_umap"}
        ),
    )
    with pytest.raises(
        ValueError, match="'X_umap' is ambiguous .* explicit ArtifactRef"
    ):
        lookup(store)


@pytest.mark.parametrize("parameters", [None, {}, {"dimreduc_key": ""}])
def test_imported_artifact_requires_a_source_key(parameters):
    store = SimpleNamespace(
        list_artifacts=lambda **kwargs: [_ref()],
        inspect_artifact=lambda ref: SimpleNamespace(parameters=parameters),
    )
    with pytest.raises(ValueError, match="has no source embedding key"):
        embeddings(store)


@pytest.fixture
def opened_store(ready_dataset):
    from scarf.cytebase import connector

    store = connector.open_datastore(ready_dataset.bucket, CYTEBASE_ID)
    yield store
    connector._close(store)


def test_published_coordinates_use_cell_ids_and_role(opened_store):
    assert list(embeddings(opened_store)) == ["X_umap"]
    coordinates = embedding_coordinates(opened_store, embedding(opened_store))
    assert list(coordinates.columns) == ["umap_1", "umap_2"]
    assert coordinates.index.name == "ids"
    assert coordinates.index.tolist() == [f"cell{i}" for i in range(6)]
    np.testing.assert_allclose(coordinates.to_numpy(), UMAP)


def test_coordinates_require_a_cell_selection(opened_store, monkeypatch):
    ref = embedding(opened_store)
    status = opened_store.inspect_artifact(ref)
    monkeypatch.setattr(
        opened_store,
        "inspect_artifact",
        lambda ref: SimpleNamespace(inputs={}, parameters=status.parameters),
    )
    with pytest.raises(ValueError, match="has no cell-selection input"):
        embedding_coordinates(opened_store, ref)


@pytest.mark.parametrize("shape", [(2, 2), (6,)])
def test_coordinates_require_matching_selection_rows(opened_store, monkeypatch, shape):
    ref = embedding(opened_store)
    monkeypatch.setattr(
        opened_store,
        "load_artifact",
        lambda ref: {"values": zarr.array(np.zeros(shape, dtype=np.float32))},
    )
    with pytest.raises(ValueError, match="does not match its cell selection"):
        embedding_coordinates(opened_store, ref)


@pytest.mark.parametrize(
    "ref",
    [
        "X_umap",
        ArtifactRef(scope="datastore", kind="cell_selection", artifact_id="c" * 64),
    ],
)
def test_coordinates_require_an_explicit_embedding_reference(ref):
    with pytest.raises(TypeError, match="embedding ArtifactRef"):
        embedding_coordinates(None, ref)


def test_reopened_mount_preserves_frozen_coordinate_alignment(cytebase_build, tmp_path):
    from scarf import DataStore, mount_datastore
    from scarf.cytebase.connector import _close
    from scarf.embeddings.imported import write_imported_embedding
    from scarf.storage.artifacts import fingerprint_array

    target = tmp_path / "analysis.zarr"
    store = mount_datastore(
        str(cytebase_build.store), at=str(target), min_features_per_cell=-1
    )
    mask = np.array([False, True, False, True, False, True])
    coordinates = UMAP[mask]
    try:
        store.cells.insert("subset", mask)
        selection = store.snapshot_cell_selection("subset")
        ref = write_imported_embedding(
            store.zw,
            assay="RNA",
            dimreduc_key="X_umap_subset",
            role="umap",
            coordinates=coordinates,
            source_digest=bytes.fromhex(cytebase_build.sha256),
            payload_fingerprints={"values": fingerprint_array(coordinates)},
            source_cell_ids=store.cells.fetch_all("ids")[mask],
            cell_selection=selection,
        )
        store.cells.update_key(~mask, "I")
    finally:
        _close(store)

    reopened = DataStore(str(target), zarr_mode="r", min_features_per_cell=-1)
    try:
        assert embedding(reopened, "X_umap_subset") == ref
        result = embedding_coordinates(reopened, ref)
        assert result.index.tolist() == ["cell1", "cell3", "cell5"]
        np.testing.assert_allclose(result.to_numpy(), coordinates)
        np.testing.assert_array_equal(reopened.cells.fetch_all("I"), ~mask)
    finally:
        _close(reopened)
