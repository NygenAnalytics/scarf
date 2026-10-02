"""Storage contract of frozen reference labels and label-transfer artifacts.

These tests build small in-memory stores, so every validation branch of the
writer and loader runs without a mapped datastore.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.mapping.label_transfer as label_transfer
from scarf.mapping.label_transfer import (
    ReferenceDistancePercentiles,
    encode_reference_labels,
    load_label_transfer,
    plan_label_transfer,
    plan_reference_labels,
    read_reference_labels,
    transfer_label_block,
    write_label_transfer,
    write_reference_labels,
)
from scarf.mapping.models import LabelTransferResult
from scarf.storage.artifact_writer import finish_artifact, plan_artifact, start_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    ExternalArtifactRef,
    artifact_group,
    inspect_artifact,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import resolve_generated_selection_artifact

_PERCENTILES = ReferenceDistancePercentiles(
    distances=np.array([0.0, 10.0]),
    percentiles=np.array([0.0, 1.0]),
)
_MAPPING_REFERENCE = ExternalArtifactRef(
    "reference-dataset",
    ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="mapping_reference",
        artifact_id="a" * 64,
    ),
)
# Three query cells with two neighbors each: the first is labelled A, the
# second abstains on a tie, and the third falls below the vote threshold.
_NEIGHBOR_CODES = np.array([[0, 0], [0, 1], [1, 0]])
_DISTANCES = np.array([[1.0, 2.0], [1.0, 1.0], [1.0, 1.5]])
_UNINFORMATIVE = np.zeros(3, dtype=bool)


def _complete_artifact(
    root: zarr.Group,
    *,
    scope: str,
    kind: str,
    operation: str,
    parameters: dict,
    inputs: dict,
    assay: str | None = None,
) -> ArtifactRef:
    planned = plan_artifact(
        root,
        scope=scope,
        assay=assay,
        kind=kind,
        operation=operation,
        parameters=parameters,
        inputs=inputs,
        execution_options={},
    )
    finish_artifact(start_artifact(root, planned), planned)
    return planned.ref


def _store(
    *,
    n_cells: int = 3,
    n_neighbors: int = 2,
    frozen_parameters: dict | None = None,
    frozen_inputs: dict | None = None,
) -> SimpleNamespace:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cell_data = root.create_group("cellData")
    row_ids = np.asarray([f"cell-{index}" for index in range(n_cells)])
    cell_data.create_array("ids", data=row_ids)
    selection = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.ones(n_cells, dtype=bool),
        row_ids=row_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="manual",
    )[0]
    projection = _complete_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="projection",
        operation="map_query",
        parameters={"save_k": n_neighbors},
        inputs={"cell_selection": selection},
    )
    frozen = _complete_artifact(
        root,
        scope="datastore",
        kind="reference_labels",
        operation="freeze_reference_labels",
        parameters=(
            {"source_column": "cell_type"}
            if frozen_parameters is None
            else frozen_parameters
        ),
        inputs=(
            {"mapping_reference": _MAPPING_REFERENCE, "labels_fingerprint": "f" * 64}
            if frozen_inputs is None
            else frozen_inputs
        ),
    )
    return SimpleNamespace(
        root=root,
        selection=selection,
        projection=projection,
        frozen=frozen,
    )


def _plan(store: SimpleNamespace, **options):
    return plan_label_transfer(
        store.root,
        projection=store.projection,
        cell_selection=store.selection,
        reference_labels=store.frozen,
        categories=np.asarray(["A", "B"]),
        n_cells=options.pop("n_cells", 3),
        n_neighbors=options.pop("n_neighbors", 2),
        threshold_fraction=options.pop("threshold_fraction", 0.7),
        max_distance=options.pop("max_distance", None),
        **options,
    )


def _block(threshold_fraction: float = 0.7, max_distance: float | None = None):
    return transfer_label_block(
        _NEIGHBOR_CODES,
        _DISTANCES,
        _UNINFORMATIVE,
        threshold_fraction=threshold_fraction,
        max_distance=max_distance,
        distance_percentiles=_PERCENTILES,
    )


def _transfer(store: SimpleNamespace) -> ArtifactRef:
    return write_label_transfer(
        store.root,
        _plan(store),
        [(0, _block())],
        chunk_rows=2,
    )


def _rewrite(root: zarr.Group, ref: ArtifactRef, name: str, values) -> None:
    """Change one payload array and record its new digest, as a writer would."""
    group = artifact_group(root, ref)
    group[name][...] = values
    digests = dict(group.attrs["array_digests"])
    digests[name] = label_transfer._array_digest(name, np.asarray(group[name][:]))
    group.attrs["array_digests"] = digests
    group.attrs["payload_fingerprint"] = label_transfer._payload_fingerprint(digests)


def _set_provenance(
    root: zarr.Group,
    ref: ArtifactRef,
    *,
    operation: str | None = None,
    parameters: dict | None = None,
    inputs: dict | None = None,
) -> None:
    group = artifact_group(root, ref)
    provenance = dict(group.attrs["provenance"])
    if operation is not None:
        provenance["operation"] = operation
    if parameters is not None:
        provenance["parameters"] = parameters
    if inputs is not None:
        provenance["inputs"] = inputs
    group.attrs["provenance"] = provenance


def test_in_memory_transfer_round_trips() -> None:
    store = _store()
    ref = _transfer(store)

    loaded = load_label_transfer(store.root, ref, load_votes=True)

    assert loaded.labels.tolist() == ["A", None, None]
    assert loaded.evidence["abstentionReason"].tolist() == [
        None,
        "tied_vote",
        "below_threshold",
    ]
    assert loaded.evidence["candidateLabel"].tolist() == ["A", None, "B"]
    assert loaded.reference_label_source == "cell_type"
    assert loaded.n_cells == 3
    assert "abstained=2" in repr(loaded)
    np.testing.assert_array_equal(loaded.cell_idx, [0, 1, 2])


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"threshold_fraction": 0.7, "max_distance": None}, "parameters do not match"),
        (
            {"threshold_fraction": 0.7, "max_distance": None, "algorithm_version": 2},
            "another algorithm version",
        ),
        (
            {"threshold_fraction": True, "max_distance": None, "algorithm_version": 1},
            "threshold_fraction is malformed",
        ),
        (
            {"threshold_fraction": 1, "max_distance": None, "algorithm_version": 1},
            "threshold_fraction is malformed",
        ),
        (
            {"threshold_fraction": 1.5, "max_distance": None, "algorithm_version": 1},
            "threshold_fraction is malformed",
        ),
        (
            {"threshold_fraction": 0.7, "max_distance": -1.0, "algorithm_version": 1},
            "max_distance is malformed",
        ),
        (
            {"threshold_fraction": 0.7, "max_distance": False, "algorithm_version": 1},
            "max_distance is malformed",
        ),
    ],
)
def test_loader_rejects_malformed_parameters(parameters, message) -> None:
    store = _store()
    ref = _transfer(store)
    _set_provenance(store.root, ref, parameters=parameters)

    with pytest.raises(ValueError, match=message):
        load_label_transfer(store.root, ref)


def test_loader_rejects_foreign_missing_and_relabelled_artifacts() -> None:
    store = _store()
    ref = _transfer(store)
    missing = ArtifactRef(
        scope="assay", assay="RNA", kind="label_transfer", artifact_id="0" * 64
    )

    with pytest.raises(ValueError, match="label_transfer artifact"):
        load_label_transfer(store.root, store.projection)
    with pytest.raises(TypeError, match="load_votes must be a boolean"):
        load_label_transfer(store.root, ref, load_votes="yes")
    with pytest.raises(ValueError, match="missing or incomplete"):
        load_label_transfer(store.root, missing)
    _set_provenance(store.root, ref, operation="relabel_cells")
    with pytest.raises(ValueError, match="another operation"):
        load_label_transfer(store.root, ref)


def test_loader_rejects_inputs_outside_the_contract() -> None:
    store = _store()
    ref = _transfer(store)
    inputs = dict(inspect_artifact(store.root, ref).inputs)
    without_projection = {key: inputs[key] for key in inputs if key != "projection"}

    _set_provenance(store.root, ref, inputs=without_projection)
    with pytest.raises(ValueError, match="inputs do not match"):
        load_label_transfer(store.root, ref)
    _set_provenance(
        store.root, ref, inputs={**inputs, "projection": store.selection.to_dict()}
    )
    with pytest.raises(ValueError, match="wrong kind or scope"):
        load_label_transfer(store.root, ref)


def _other_selection(store: SimpleNamespace) -> ArtifactRef:
    return resolve_generated_selection_artifact(
        store.root,
        scope="datastore",
        kind="cell_selection",
        values=np.array([True, True, False]),
        row_ids=np.asarray(store.root["cellData/ids"][:]),
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="other",
    )[0]


def _point_at_a_smaller_selection(store: SimpleNamespace) -> None:
    # The transfer and its projection agree, but on fewer cells than its rows.
    smaller = _other_selection(store).to_dict()
    _set_provenance(store.root, store.projection, inputs={"cell_selection": smaller})
    transfer = store.transfer
    inputs = dict(inspect_artifact(store.root, transfer).inputs)
    _set_provenance(store.root, transfer, inputs={**inputs, "cell_selection": smaller})


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda store: artifact_group(store.root, store.projection).attrs.update(
                {"complete": False}
            ),
            "projection is missing or incomplete",
        ),
        (
            lambda store: _set_provenance(
                store.root, store.projection, operation="map_elsewhere"
            ),
            "projection has another operation",
        ),
        (
            lambda store: _set_provenance(
                store.root,
                store.projection,
                inputs={"cell_selection": _other_selection(store).to_dict()},
            ),
            "use different cells",
        ),
        (
            lambda store: _set_provenance(
                store.root, store.projection, parameters={"save_k": 5}
            ),
            "do not match the projection neighbors",
        ),
        (
            lambda store: artifact_group(store.root, store.frozen).attrs.update(
                {"complete": False}
            ),
            "frozen reference labels are missing",
        ),
        (
            _point_at_a_smaller_selection,
            "rows do not match its query cell selection",
        ),
        (
            lambda store: _set_provenance(
                store.root, store.frozen, operation="snapshot_labels"
            ),
            "Frozen reference labels have another operation",
        ),
        (
            lambda store: _set_provenance(store.root, store.frozen, parameters={}),
            "freeze_reference_labels contract",
        ),
        (
            lambda store: _set_provenance(
                store.root, store.frozen, parameters={"source_column": ""}
            ),
            "freeze_reference_labels contract",
        ),
        (
            lambda store: _set_provenance(
                store.root,
                store.frozen,
                parameters={},
                inputs={
                    "mapping_reference": _MAPPING_REFERENCE.to_dict(),
                    "labels_fingerprint": "f" * 64,
                    "source_labels": "cluster_labels",
                },
            ),
            "freeze_reference_labels contract",
        ),
    ],
)
def test_loader_requires_matching_projection_and_frozen_labels(tamper, message) -> None:
    store = _store()
    ref = _transfer(store)
    store.transfer = ref
    tamper(store)

    with pytest.raises(ValueError, match=message):
        load_label_transfer(store.root, ref)


def _with_group(change):
    def tamper(root: zarr.Group, ref: ArtifactRef) -> None:
        change(artifact_group(root, ref))

    return tamper


def _replace_array(name: str, values: np.ndarray):
    def change(group: zarr.Group) -> None:
        del group[name]
        group.create_array(name, data=values)

    return _with_group(change)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            _with_group(lambda group: group["vote_fraction"].__setitem__(0, 0.1)),
            "differ from their recorded digests: vote_fraction",
        ),
        (
            _with_group(lambda group: group.attrs.update({"array_digests": "x"})),
            "array digests are malformed",
        ),
        (
            _with_group(
                lambda group: group.attrs.update(
                    {"array_digests": {**group.attrs["array_digests"], "labels": 1}}
                )
            ),
            "array digests are malformed",
        ),
        (
            _with_group(lambda group: group.attrs.update({"payload_fingerprint": "x"})),
            "does not match its array digests",
        ),
        (
            _with_group(lambda group: group.attrs.update({"note": "edited"})),
            "attributes do not match",
        ),
        (
            _with_group(lambda group: group.create_array("extra", data=np.zeros(3))),
            "arrays do not match",
        ),
        (
            _with_group(lambda group: group.create_group("nested")),
            "unexpected groups",
        ),
        (
            _replace_array("vote_fraction", np.zeros(3, dtype=np.float32)),
            "wrong shape or value type",
        ),
        (
            _replace_array("categories", np.asarray([], dtype="<U1")),
            "classes must be a non-empty vector",
        ),
        (
            _replace_array("labels", np.asarray(["A", "B"])),
            "one row per query cell",
        ),
        (
            _with_group(lambda group: group["labels"].attrs.pop("missing_mask")),
            "no linked missing-label mask",
        ),
        (
            _with_group(lambda group: group["vote_fraction"].attrs.update({"unit": 1})),
            "unexpected attributes",
        ),
    ],
)
def test_loader_rejects_a_damaged_payload(tamper, message) -> None:
    store = _store()
    ref = _transfer(store)
    tamper(store.root, ref)

    with pytest.raises(ValueError, match=message):
        load_label_transfer(store.root, ref)


@pytest.mark.parametrize(
    ("name", "row", "value", "load_votes", "message"),
    [
        ("abstention_reason", 0, 9, False, "unknown abstention reason"),
        ("abstention_reason", 0, 4, False, "abstentions do not match their reasons"),
        ("candidate_codes", 0, 7, False, "candidates name an unknown reference"),
        ("candidate_codes", 1, 0, False, "candidates do not match their reasons"),
        ("labels", 0, "B", False, "labels do not match their candidates"),
        ("vote_class_codes", 0, [5, 5], True, "votes name an unknown reference"),
        ("vote_class_codes", 0, [1, 1], True, "candidates do not match their votes"),
    ],
)
def test_loader_rejects_internally_inconsistent_payloads(
    name, row, value, load_votes, message
) -> None:
    store = _store()
    ref = _transfer(store)
    values = np.asarray(artifact_group(store.root, ref)[name][:])
    values[row] = value
    # The digests are rewritten too, so only the consistency checks object.
    _rewrite(store.root, ref, name, values)

    with pytest.raises(ValueError, match=message):
        load_label_transfer(store.root, ref, load_votes=load_votes)


def test_writer_refuses_reused_gapped_overlong_and_partial_blocks() -> None:
    store = _store()
    ref = _transfer(store)
    reused = _plan(store)
    assert reused.reused and reused.ref == ref
    with pytest.raises(ValueError, match="reused label transfer is loaded"):
        write_label_transfer(store.root, reused, [], chunk_rows=2)

    block = _block()
    for blocks, message in (
        ([(1, block)], "must be contiguous"),
        ([(0, block), (3, block)], "must fit the query cells"),
    ):
        fresh = _store()
        plan = _plan(fresh)
        with pytest.raises(RuntimeError, match=message):
            write_label_transfer(fresh.root, plan, blocks, chunk_rows=2)
        # A failed write leaves no incomplete artifact behind.
        assert not inspect_artifact(fresh.root, plan.ref).exists

    partial = _store()
    plan = _plan(partial)
    first_rows = transfer_label_block(
        _NEIGHBOR_CODES[:2],
        _DISTANCES[:2],
        _UNINFORMATIVE[:2],
        threshold_fraction=0.7,
        max_distance=None,
        distance_percentiles=_PERCENTILES,
    )
    with pytest.raises(RuntimeError, match="did not cover every projected"):
        write_label_transfer(partial.root, plan, [(0, first_rows)], chunk_rows=2)


def test_planner_checks_the_cell_count_and_reuses_only_matching_classes() -> None:
    store = _store()
    _transfer(store)

    with pytest.raises(ValueError, match="has changed size"):
        _plan(store, n_cells=4)
    other_classes = plan_label_transfer(
        store.root,
        projection=store.projection,
        cell_selection=store.selection,
        reference_labels=store.frozen,
        categories=np.asarray(["A", "C"]),
        n_cells=3,
        n_neighbors=2,
        threshold_fraction=0.7,
        max_distance=None,
    )
    assert not other_classes.reused


def test_vote_block_rejects_misaligned_inputs() -> None:
    def block(codes, distances, uninformative):
        return transfer_label_block(
            codes,
            distances,
            uninformative,
            threshold_fraction=0.5,
            max_distance=None,
            distance_percentiles=_PERCENTILES,
        )

    with pytest.raises(ValueError, match="matching matrices"):
        block(np.array([0, 1]), np.array([1.0, 2.0]), np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="matching matrices"):
        block(_NEIGHBOR_CODES, _DISTANCES[:, :1], _UNINFORMATIVE)
    with pytest.raises(ValueError, match="one value per query cell"):
        block(_NEIGHBOR_CODES, _DISTANCES, np.zeros(2, dtype=bool))


def test_reference_label_encoding_keeps_value_types_and_rejects_ambiguity() -> None:
    floats, _ = encode_reference_labels(np.array([1.5, 2.5]), np.ones(2, dtype=bool))
    booleans, _ = encode_reference_labels(
        np.array([True, False]), np.ones(2, dtype=bool)
    )
    assert floats.dtype == np.float64 and floats.tolist() == [1.5, 2.5]
    assert booleans.dtype == bool and booleans.tolist() == [True, False]

    with pytest.raises(ValueError, match="no usable label"):
        encode_reference_labels(np.array(["a", "b"]), np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="same text"):
        encode_reference_labels(
            np.array([1, "1"], dtype=object), np.ones(2, dtype=bool)
        )

    misaligned = SimpleNamespace(
        selected_cell_count=2,
        _fetch_cell_labels=lambda _column: (np.array(["a"]), np.array([True])),
    )
    with pytest.raises(ValueError, match="one value per selected reference cell"):
        read_reference_labels(misaligned, "cell_type")


def test_label_transfer_result_rejects_misaligned_arrays() -> None:
    from dataclasses import replace

    store = _store()
    loaded = load_label_transfer(store.root, _transfer(store), load_votes=True)

    with pytest.raises(ValueError, match="cell_idx must have one row"):
        replace(loaded, cell_idx=np.arange(2))
    with pytest.raises(ValueError, match="Vote arrays must have one row"):
        replace(
            loaded,
            vote_class_codes=np.zeros((3, 2), dtype=np.int64),
            vote_class_fractions=np.zeros((3, 3)),
        )
    assert isinstance(loaded, LabelTransferResult)


def test_loader_passes_artifact_resolution_errors_through() -> None:
    store = _store()
    ref = _transfer(store)
    inputs = dict(inspect_artifact(store.root, ref).inputs)
    _set_provenance(
        store.root,
        ref,
        inputs={**inputs, "projection": {"type": "artifact", "scope": "assay"}},
    )

    with pytest.raises(ArtifactResolutionError, match="malformed 'projection'"):
        load_label_transfer(store.root, ref)


def test_layout_checks_rows_neighbors_and_digest_coverage() -> None:
    store = _store()
    ref = _transfer(store)

    group = artifact_group(store.root, ref)
    with pytest.raises(ValueError, match="do not match the projected query cells"):
        label_transfer._transfer_payload_arrays(group, n_cells=4)
    with pytest.raises(ValueError, match="do not match the projection neighbors"):
        label_transfer._transfer_payload_arrays(group, n_neighbors=3)
    # A plan for other neighbor counts never reuses this payload.
    assert not _plan(store, n_neighbors=3).reused
    with pytest.raises(ValueError, match="do not cover its payload arrays"):
        label_transfer._payload_fingerprint({})


def test_frozen_reference_labels_are_reused_only_when_readable(monkeypatch) -> None:
    store = _store()
    reference = SimpleNamespace(
        selected_cell_count=3,
        _fetch_cell_labels=lambda _column: (
            np.array(["A", "B", "A"], dtype=object),
            np.ones(3, dtype=bool),
        ),
        external_ref=_MAPPING_REFERENCE,
        dataset_fingerprint="reference-dataset",
        assay_name="RNA",
    )
    first = plan_reference_labels(store.root, reference, "cell_type")
    write_reference_labels(store.root, first)
    assert plan_reference_labels(store.root, reference, "cell_type").ref == first.ref

    def unreadable(*_args, **_kwargs):
        raise ValueError("the stored copy cannot be read")

    monkeypatch.setattr(label_transfer, "fingerprint_stored_arrays", unreadable)
    assert not plan_reference_labels(store.root, reference, "cell_type").reused
