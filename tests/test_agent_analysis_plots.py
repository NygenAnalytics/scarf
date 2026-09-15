"""Bounded, immutable cluster displays for agent results and reports."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent import _plots
from scarf.storage.refs import ArtifactRef


class VirtualArray:
    """A large immutable array that refuses unbounded or whole-column reads."""

    def __init__(self, rows: int, *, coordinates: bool = False) -> None:
        self.shape = (rows, 2) if coordinates else (rows,)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(float if coordinates else int)
        self.coordinates = coordinates
        self.read_sizes: list[int] = []

    def __getitem__(self, selection: slice) -> np.ndarray:
        assert isinstance(selection, slice)
        assert selection.start is not None and selection.stop is not None
        assert selection.stop - selection.start <= _plots.DISPLAY_BLOCK_ROWS
        rows = np.arange(selection.start, min(selection.stop, self.shape[0]))
        self.read_sizes.append(len(rows))
        if self.coordinates:
            return np.column_stack((rows, rows % 11)).astype(float)
        return np.where(rows == self.shape[0] - 1, 7, rows % 3)


def display_store(
    monkeypatch: pytest.MonkeyPatch, n: int = 621_200
) -> tuple[Any, dict[str, ArtifactRef], dict[str, VirtualArray]]:
    refs = {
        "cell_selection": ArtifactRef("datastore", "cell_selection", "1" * 64),
        "graph": ArtifactRef("assay", "connectivity_map", "2" * 64, "RNA2"),
        "clusters": ArtifactRef("assay", "cluster_labels", "3" * 64, "RNA2"),
        "umap": ArtifactRef("assay", "embedding", "4" * 64, "RNA2"),
    }
    arrays = {
        "clusters": VirtualArray(n),
        "umap": VirtualArray(n, coordinates=True),
    }
    store = SimpleNamespace(zw=object(), workspace="selected-workspace")
    store.load_artifact = lambda ref: {
        "values": arrays[next(name for name in arrays if refs[name] == ref)]
    }
    store.inspect_artifact = lambda ref: SimpleNamespace(
        complete=True,
        operation="run_umap" if ref == refs["umap"] else "run_leiden_clustering",
        inputs={
            "cell_selection": refs["cell_selection"].to_dict(),
            "graph": refs["graph"].to_dict(),
        },
    )
    monkeypatch.setattr(_plots, "as_zarr_array", lambda value, **_: value)
    monkeypatch.setattr(
        _plots, "graph_cell_selection", lambda root, ref: refs["cell_selection"]
    )

    def validate(root: Any, ref: ArtifactRef, **kwargs: Any) -> Any:
        assert root is store.zw and ref == refs["cell_selection"]
        assert kwargs == {
            "kind": "cell_selection",
            "scope": "datastore",
            "assay": None,
            "table_path": "cellData",
        }
        return SimpleNamespace(selected_count=n)

    monkeypatch.setattr(_plots, "validate_stored_selection_integrity", validate)
    return store, refs, arrays


def test_large_final_map_is_bounded_cluster_aware_and_preserves_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, refs, arrays = display_store(monkeypatch)
    plot = _plots.plot_final_umap(store, **refs, show=False)
    try:
        assert plot.provenance.n_cells == 50_000
        assert plot.provenance.extras["input_n_cells"] == 621_200
        assert plot.provenance.extras["workspace"] == "selected-workspace"
        assert plot.provenance.extras["layout"] == refs["umap"].to_dict()
        counts = plot.tables["cluster_counts"].set_index("cluster")["cells"].to_dict()
        assert sum(counts.values()) == 621_200 and counts["7"] == 1
        axes = plot.axes["clusters"]
        assert sum(len(artist.get_offsets()) for artist in axes.collections) == 50_000
        from matplotlib.colors import to_rgb

        palette = plot.scales[0].palette
        for artist in axes.collections:
            positions = np.asarray(artist.get_offsets())[:, 0].astype(int)
            expected = np.where(positions == 621_199, 7, positions % 3)
            np.testing.assert_allclose(
                artist.get_facecolors()[:, :3],
                np.asarray([to_rgb(palette[str(label)]) for label in expected]),
            )
        assert 621_199 in positions
        assert all(max(array.read_sizes) <= 100_000 for array in arrays.values())
        sidecar = plot.save_provenance(tmp_path / "map.json")
        assert (
            json.loads(sidecar.read_text())["provenance"]["extras"]["cluster_counts"]
            == counts
        )
    finally:
        plot.close()


def test_sampling_is_reproducible_across_read_boundaries_and_keeps_rare_cells() -> None:
    values = np.asarray(["common"] * 999 + ["rare"])
    counts = {"common": 999, "rare": 1}
    first = _plots._sample_cluster_rows(
        values, counts, maximum=50, seed=5, block_rows=73
    )
    second = _plots._sample_cluster_rows(
        values, counts, maximum=50, seed=5, block_rows=221
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0][-1] == 999 and list(first[1]).count("rare") == 1
    third = _plots._sample_cluster_rows(values, counts, maximum=50, seed=7)
    assert not np.array_equal(first[0], third[0])


@pytest.mark.parametrize("mismatch", ["selection", "graph", "producer", "shape"])
def test_final_map_rejects_incompatible_saved_artifacts(
    monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    store, refs, arrays = display_store(monkeypatch, n=12)
    inspect = store.inspect_artifact

    def changed(ref: ArtifactRef) -> Any:
        status = inspect(ref)
        if ref == refs["umap"]:
            if mismatch == "selection":
                status.inputs["cell_selection"] = ArtifactRef(
                    "datastore", "cell_selection", "a" * 64
                ).to_dict()
            elif mismatch == "graph":
                status.inputs["graph"] = ArtifactRef(
                    "assay", "connectivity_map", "a" * 64, "RNA2"
                ).to_dict()
            elif mismatch == "producer":
                status.operation = "import_dimreduc"
        return status

    store.inspect_artifact = changed
    if mismatch == "shape":
        arrays["umap"].shape = (13, 2)
    with pytest.raises(ValueError):
        _plots.plot_final_umap(store, **refs, show=False)
    assert not any(array.read_sizes for array in arrays.values())


def test_small_map_draws_every_frozen_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    store, refs, _ = display_store(monkeypatch, n=12)
    plot = _plots.plot_final_umap(store, **refs, show=False)
    try:
        assert plot.provenance.n_cells == 12
        offsets = np.concatenate(
            [artist.get_offsets()[:, 0] for artist in plot.axes["clusters"].collections]
        )
        np.testing.assert_array_equal(np.sort(offsets), np.arange(12))
    finally:
        plot.close()
