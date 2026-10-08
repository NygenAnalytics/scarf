"""Operation revisions: the registry, sparse provenance, and stale reporting."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.storage.operation_revisions as operation_revisions
from scarf.storage.artifact_writer import artifact_transaction, plan_artifact
from scarf.storage.artifacts import (
    ARTIFACT_KINDS,
    ArtifactRef,
    artifact_path,
    canonical_bytes,
    inspect_artifact,
    list_artifacts,
    make_provenance,
    require_complete_artifact,
)
from scarf.storage.lineage import ArtifactLineage
from scarf.storage.operation_revisions import (
    OPERATION_REVISIONS,
    OperationRevision,
    _validate_registry,
    applicable_revisions,
    effective_revision,
)
from scarf.utils.logging import logger

_PCA_PARAMETERS = {"dims": 11, "feat_scaling": True}
_PCA_INPUTS = {"normalized": {"fixture": "normalized"}}
_FIXED = "PCA centers features before projecting them."


type _Install = Callable[[Mapping[str, tuple[OperationRevision, ...]]], None]


@pytest.fixture
def revisions(monkeypatch: pytest.MonkeyPatch) -> _Install:
    """Isolate a test from the shipped registry and return its installer.

    The test starts with every operation at revision 1, so released revisions
    never change what it observes. Calling the installer replaces the
    registry with exactly the given revisions.
    """

    def install(entries: Mapping[str, tuple[OperationRevision, ...]]) -> None:
        _validate_registry(entries)
        monkeypatch.setattr(
            operation_revisions,
            "OPERATION_REVISIONS",
            MappingProxyType(dict(entries)),
        )

    install({})
    return install


def _whole_revision(change: str = _FIXED) -> OperationRevision:
    return OperationRevision(revision=2, release="1.0.0", change=change, applies=None)


def _root() -> zarr.Group:
    return zarr.open_group(store=MemoryStore(), mode="w")


def _write_data(group: zarr.Group) -> None:
    """Write the reduction payload."""
    group.create_array("data", data=np.arange(4, dtype=np.float32))


def _write(
    root: zarr.Group,
    *,
    operation: str = "run_pca",
    kind: str = "reduction",
    parameters: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> ArtifactRef:
    """Write a new complete artifact with the provenance this release records."""
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind=kind,
        operation=operation,
        parameters=_PCA_PARAMETERS if parameters is None else parameters,
        inputs=_PCA_INPUTS if inputs is None else inputs,
        execution_options={},
        invalidate_cache=True,
    )
    with artifact_transaction(root, planned) as group:
        _write_data(group)
    return planned.ref


def _plan(root: zarr.Group, **overrides: Any):
    arguments: dict[str, Any] = {
        "scope": "assay",
        "assay": "RNA",
        "kind": "reduction",
        "operation": "run_pca",
        "parameters": _PCA_PARAMETERS,
        "inputs": _PCA_INPUTS,
        "execution_options": {},
    }
    arguments.update(overrides)
    return plan_artifact(root, **arguments)


@contextmanager
def _messages(level: str) -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level=level,
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _write_record(root: zarr.Group, provenance: dict[str, Any], *, complete: bool):
    ref = ArtifactRef(
        scope="assay", assay="RNA", kind="reduction", artifact_id="c" * 64
    )
    group = root.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": ref.kind,
            "provenance": provenance,
            "execution_options": {},
            "complete": complete,
        }
    )
    return ref


def test_registry_validation_rejects_misnumbered_revisions() -> None:
    entry = OperationRevision(3, "1.0.0", "Change.", None)
    with pytest.raises(ValueError, match="numbered 2, 3"):
        _validate_registry({"run_pca": (entry,)})


def test_effective_revision_is_the_highest_applicable_revision(
    revisions: _Install,
) -> None:
    calls: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []

    def exact(kind: str, parameters: Mapping[str, Any], inputs: Mapping[str, Any]):
        calls.append((kind, parameters, inputs))
        return parameters.get("solver") == "exact"

    def fast(_kind: str, parameters: Mapping[str, Any], _inputs: Mapping[str, Any]):
        return parameters.get("solver") == "fast"

    scoped = (
        OperationRevision(2, "1.0.0", "Exact solves converge further.", exact),
        OperationRevision(3, "1.1.0", "Fast solves use more iterations.", fast),
    )
    revisions({"run_lsi": scoped, "run_pca": (_whole_revision(),)})

    inputs = {"normalized": {"fixture": "normalized"}}
    assert effective_revision("run_lsi", "reduction", {"solver": "exact"}, inputs) == 2
    assert effective_revision("run_lsi", "reduction", {"solver": "fast"}, inputs) == 3
    assert effective_revision("run_lsi", "reduction", {"solver": "other"}, inputs) == 1
    assert applicable_revisions("run_lsi", "reduction", {"solver": "fast"}, inputs) == (
        scoped[1],
    )
    assert calls[0] == ("reduction", {"solver": "exact"}, inputs)
    assert effective_revision("run_pca", "reduction", {}, {}) == 2
    assert effective_revision("run_umap", "embedding", {}, {}) == 1
    assert effective_revision("not_registered", "reduction", {}, {}) == 1


def test_released_predicates_are_total_over_sparse_provenance() -> None:
    """A predicate must answer for records that lack the keys it reads.

    Artifacts written before a parameter existed record no value for it, so
    every released predicate returns a bool for empty parameters and inputs.
    """
    for operation, revisions in OPERATION_REVISIONS.items():
        for entry in revisions:
            if entry.applies is None:
                continue
            for kind in sorted(ARTIFACT_KINDS):
                assert isinstance(entry.applies(kind, {}, {}), bool), (
                    operation,
                    entry.revision,
                    kind,
                )


def test_provenance_records_revision_one_by_omission() -> None:
    legacy = {
        "operation": "run_pca",
        "parameters": {"dims": 11, "feat_scaling": True},
        "inputs": {"normalized": {"fixture": "normalized"}},
    }
    first = make_provenance(
        operation="run_pca",
        parameters={"dims": np.int64(11), "feat_scaling": True},
        inputs=_PCA_INPUTS,
    )
    assert first == legacy
    assert canonical_bytes(first) == canonical_bytes(legacy)
    assert make_provenance(
        operation="run_pca",
        parameters=_PCA_PARAMETERS,
        inputs=_PCA_INPUTS,
        revision=2,
    ) == {**legacy, "revision": 2}
    for invalid in (0, -1, True, 2.0, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            make_provenance(
                operation="run_pca",
                parameters=_PCA_PARAMETERS,
                inputs=_PCA_INPUTS,
                revision=invalid,  # type: ignore[arg-type]
            )


# Revision 1 is recorded by omission, so an explicit 1 is invalid too.
@pytest.mark.parametrize("recorded", ["2", 1, True, 0])
def test_an_invalid_recorded_revision_is_not_current(recorded: object) -> None:
    root = _root()
    provenance = {
        **make_provenance(operation="run_pca", parameters={}, inputs={}),
        "revision": recorded,
    }
    status = inspect_artifact(root, _write_record(root, provenance, complete=True))
    assert status.complete
    assert status.revision is None
    assert not status.is_current
    assert status.superseded_by == ()


def test_status_reports_recorded_and_current_revisions(
    revisions: _Install,
) -> None:
    root = _root()
    legacy = _write(root)
    status = inspect_artifact(root, legacy)
    assert (status.revision, status.current_revision, status.is_current) == (1, 1, True)
    assert status.superseded_by == ()

    revision = _whole_revision()
    revisions({"run_pca": (revision,)})
    status = inspect_artifact(root, legacy)
    assert (status.revision, status.current_revision, status.is_current) == (
        1,
        2,
        False,
    )
    assert status.superseded_by == (revision,)

    current = inspect_artifact(root, _write(root))
    assert (current.revision, current.current_revision, current.is_current) == (
        2,
        2,
        True,
    )
    assert current.provenance is not None
    assert current.provenance["revision"] == 2

    missing = inspect_artifact(
        root,
        ArtifactRef(scope="assay", assay="RNA", kind="reduction", artifact_id="d" * 64),
    )
    assert (missing.revision, missing.current_revision, missing.is_current) == (
        None,
        None,
        False,
    )


def test_superseded_artifact_is_recomputed_with_one_log_line(
    revisions: _Install,
) -> None:
    root = _root()
    # The newer of two earlier results is reused, and named once superseded.
    older = _write(root)
    root[artifact_path(older)].attrs["created_at_ns"] = 1
    legacy = _write(root)
    assert _plan(root).ref == legacy

    revision = _whole_revision()
    revisions({"run_pca": (revision,)})
    with _messages("INFO") as messages:
        planned = _plan(root)
    assert not planned.reused
    assert planned.provenance["revision"] == 2
    expected = (
        f"Recomputing run_pca: artifact {legacy.artifact_id[:12]} is revision 1, "
        f"current 2: {_FIXED}"
    )
    assert [message for message in messages if "Recomputing" in message] == [expected]

    with artifact_transaction(root, planned) as group:
        _write_data(group)
    with _messages("INFO") as messages:
        again = _plan(root)
    assert again.reused
    assert again.ref == planned.ref
    assert not [message for message in messages if "Recomputing" in message]

    # The superseded artifact stays listable, loadable, and traceable.
    assert set(
        list_artifacts(root, scope="assay", assay="RNA", operation="run_pca")
    ) == {
        older,
        legacy,
        planned.ref,
    }
    assert require_complete_artifact(root, legacy).complete
    assert not inspect_artifact(root, legacy).is_current
    assert inspect_artifact(root, planned.ref).is_current


def test_scoped_revision_supersedes_only_the_artifacts_it_applies_to(
    revisions: _Install,
) -> None:
    root = _root()
    exact = _write(root, parameters={**_PCA_PARAMETERS, "solver": "exact"})
    fast = _write(root, parameters={**_PCA_PARAMETERS, "solver": "fast"})
    scoped = OperationRevision(
        2,
        "1.0.0",
        "Fast PCA solves use more power iterations.",
        lambda _kind, parameters, _inputs: parameters.get("solver") == "fast",
    )
    revisions({"run_pca": (scoped,)})
    with _messages("INFO") as messages:
        reused = _plan(root, parameters={**_PCA_PARAMETERS, "solver": "exact"})
        recomputed = _plan(root, parameters={**_PCA_PARAMETERS, "solver": "fast"})
    assert reused.reused and reused.ref == exact
    assert "revision" not in reused.provenance
    assert not recomputed.reused
    (line,) = [message for message in messages if "Recomputing" in message]
    assert line.startswith(f"Recomputing run_pca: artifact {fast.artifact_id[:12]} ")
    assert inspect_artifact(root, exact).is_current
    assert not inspect_artifact(root, fast).is_current


def test_artifact_from_a_newer_revision_is_not_reused(
    revisions: _Install,
) -> None:
    root = _root()
    revisions(
        {
            "run_pca": (
                _whole_revision(),
                OperationRevision(3, "1.1.0", "PCA whitening is exact.", None),
            )
        },
    )
    newer = _write(root)
    assert inspect_artifact(root, newer).revision == 3
    # This release only knows revision 2.
    revisions({"run_pca": (_whole_revision(),)})
    status = inspect_artifact(root, newer)
    assert (status.revision, status.current_revision) == (3, 2)
    assert not status.is_current
    assert status.superseded_by == ()
    assert not _plan(root).reused
    markdown = ArtifactLineage.from_store(root, newer).to_markdown()
    assert "- Revision: `3, newer than revision 2 of this Scarf release`" in markdown


def test_explicit_stale_inputs_stay_usable_and_traceable(
    revisions: _Install,
) -> None:
    root = _root()
    reduction = _write(root)
    revisions({"run_pca": (_whole_revision(),)})
    with _messages("INFO") as messages:
        index = _write(
            root,
            operation="build_ann_index",
            kind="ann_index",
            parameters={"ann_metric": "l2"},
            inputs={"coordinates": reduction},
        )
    assert not [
        message
        for message in messages
        if "Recomputing" in message or str(reduction.artifact_id[:12]) in message
    ]
    assert inspect_artifact(root, index).input_ref("coordinates") == reduction
    assert inspect_artifact(root, index).is_current

    lineage = ArtifactLineage.from_store(root, {"index": index})
    mermaid = lineage.to_mermaid()
    assert "status: stale (revision 1, current 2)" in mermaid
    assert mermaid.count("status: stale (") == 1
    markdown = lineage.to_markdown()
    reduction_section = markdown.split(f"/ reduction / {reduction.artifact_id[:12]}")[1]
    assert "- Status: `stale`" in reduction_section
    assert f"- Revision: `1, current 2: {_FIXED}`" in reduction_section
    # The index records its current revision; only its input is stale.
    index_section = markdown.split(f"/ ann_index / {index.artifact_id[:12]}")[1]
    index_section = index_section.split("####")[0]
    assert "- Status: `complete`" in index_section
    assert "- Revision:" not in index_section
    # A current artifact shows the revision it records.
    current = ArtifactLineage.from_store(root, _write(root))
    assert "- Revision: `2`" in current.to_markdown()
    assert "stale" not in current.to_mermaid()
