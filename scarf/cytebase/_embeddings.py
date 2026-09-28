"""Resolve imported embeddings from an open Scarf DataStore."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from scarf import DataStore
    from scarf.storage.artifacts import ArtifactRef


def embeddings(ds: "DataStore", *, assay: str = "RNA") -> dict[str, "ArtifactRef"]:
    """Map imported source keys to complete embedding artifacts in ``assay``.

    Discovery uses the DataStore's artifact provenance, so it also works on a
    local mount without a catalog connection. Multiple imports of the same key
    are ambiguous and require choosing an explicit artifact reference.
    """
    found: dict[str, ArtifactRef] = {}
    for ref in ds.list_artifacts(
        kind="embedding",
        from_assay=assay,
        operation="import_dimreduc",
        complete_only=True,
    ):
        key = (ds.inspect_artifact(ref).parameters or {}).get("dimreduc_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"Imported embedding {ref!r} has no source embedding key")
        if key in found:
            raise ValueError(
                f"Imported embedding {key!r} is ambiguous in assay {assay!r}; "
                "use ds.list_artifacts() and choose an explicit ArtifactRef"
            )
        found[key] = ref
    return found


def embedding(
    ds: "DataStore", key: str = "X_umap", *, assay: str = "RNA"
) -> "ArtifactRef":
    """Return the unique imported embedding for ``key``, or report missing data."""
    refs = ds.list_artifacts(
        kind="embedding",
        from_assay=assay,
        operation="import_dimreduc",
        complete_only=True,
        parameters={"dimreduc_key": key},
    )
    if len(refs) == 1:
        return refs[0]
    if refs:
        raise ValueError(
            f"Imported embedding {key!r} is ambiguous in assay {assay!r}; "
            "use ds.list_artifacts() and choose an explicit ArtifactRef"
        )
    available = set()
    for ref in ds.list_artifacts(
        kind="embedding",
        from_assay=assay,
        operation="import_dimreduc",
        complete_only=True,
    ):
        source_key = (ds.inspect_artifact(ref).parameters or {}).get("dimreduc_key")
        if isinstance(source_key, str) and source_key:
            available.add(source_key)
    names = ", ".join(sorted(available)) or "none"
    raise KeyError(
        f"Embedding {key!r} was not imported for assay {assay!r}; available: {names}"
    )


def embedding_coordinates(ds: "DataStore", ref: "ArtifactRef") -> "pd.DataFrame":
    """Read coordinates indexed by the embedding's frozen cell IDs.

    ``ref`` must identify an embedding artifact. Its saved cell selection
    determines row alignment, independently of the DataStore's current filter.
    """
    import numpy as np
    import pandas as pd

    from scarf.storage.artifacts import ArtifactRef
    from scarf.storage.selections import read_stored_selection_indices
    from scarf.storage.types import as_zarr_array

    if not isinstance(ref, ArtifactRef) or ref.kind != "embedding":
        raise TypeError("ref must be an embedding ArtifactRef")
    status = ds.inspect_artifact(ref)
    selection = (status.inputs or {}).get("cell_selection")
    if selection is None:
        raise ValueError(f"Embedding {ref!r} has no cell-selection input")
    cell_idx = read_stored_selection_indices(
        ds.zw,
        ArtifactRef.from_dict(selection),
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    values = as_zarr_array(ds.load_artifact(ref)["values"], name="values")
    if values.ndim != 2 or values.shape[0] != len(cell_idx):
        raise ValueError(f"Embedding {ref!r} does not match its cell selection")
    parameters = status.parameters or {}
    role = parameters.get("role") or str(
        parameters.get("dimreduc_key") or "embedding"
    ).removeprefix("X_")
    return pd.DataFrame(
        np.asarray(values[:]),
        index=pd.Index(ds.cells.fetch_all("ids")[cell_idx], name="ids"),
        columns=[f"{role}_{i + 1}" for i in range(values.shape[1])],
    )
