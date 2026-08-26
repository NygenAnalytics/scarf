import numpy as np
import pytest

from profiling.stages import (
    validate_cluster_source_identity,
    validate_experiment_branches,
)


def _ids() -> np.ndarray:
    return np.array(["c0", "c1", "c2", "c3"], dtype=object)


def _active() -> np.ndarray:
    return np.array([True, True, True, False])


def _labels() -> np.ndarray:
    return np.array(["a", "b", "a", "a"], dtype=object)


def test_cluster_import_accepts_matching_identity() -> None:
    groups = validate_cluster_source_identity(
        sourceIds=_ids(),
        targetIds=_ids(),
        sourceActive=_active(),
        targetActive=_active(),
        labels=_labels(),
    )
    assert groups == ["a", "b"]


def test_cluster_import_rejects_reordered_ids() -> None:
    reordered = np.array(["c1", "c0", "c2", "c3"], dtype=object)
    with pytest.raises(ValueError, match="identical in order"):
        validate_cluster_source_identity(
            sourceIds=reordered,
            targetIds=_ids(),
            sourceActive=_active(),
            targetActive=_active(),
            labels=_labels(),
        )


def test_cluster_import_rejects_missing_labels() -> None:
    labels = _labels()
    labels[1] = ""
    with pytest.raises(ValueError, match="missing labels"):
        validate_cluster_source_identity(
            sourceIds=_ids(),
            targetIds=_ids(),
            sourceActive=_active(),
            targetActive=_active(),
            labels=labels,
        )


def test_cluster_import_rejects_duplicate_ids() -> None:
    ids = np.array(["c0", "c0", "c2", "c3"], dtype=object)
    with pytest.raises(ValueError, match="not unique"):
        validate_cluster_source_identity(
            sourceIds=ids,
            targetIds=ids,
            sourceActive=_active(),
            targetActive=_active(),
            labels=_labels(),
        )


def test_cluster_import_rejects_mismatched_masks() -> None:
    with pytest.raises(ValueError, match="active-cell mask"):
        validate_cluster_source_identity(
            sourceIds=_ids(),
            targetIds=_ids(),
            sourceActive=_active(),
            targetActive=np.array([True, True, False, False]),
            labels=_labels(),
        )


def test_cluster_import_rejects_one_group() -> None:
    labels = np.array(["a", "a", "a", "a"], dtype=object)
    with pytest.raises(ValueError, match="at least two groups"):
        validate_cluster_source_identity(
            sourceIds=_ids(),
            targetIds=_ids(),
            sourceActive=_active(),
            targetActive=_active(),
            labels=labels,
        )


def test_validate_experiment_checks_pca_and_marker_branches_separately() -> None:
    validate_experiment_branches(
        pcaComplete=True,
        importedColumnPresent=True,
        markerComplete=True,
    )
    with pytest.raises(ValueError, match="PCA branch"):
        validate_experiment_branches(
            pcaComplete=False,
            importedColumnPresent=True,
            markerComplete=True,
        )
    with pytest.raises(ValueError, match="imported cluster column"):
        validate_experiment_branches(
            pcaComplete=True,
            importedColumnPresent=False,
            markerComplete=True,
        )
    with pytest.raises(ValueError, match="marker_table"):
        validate_experiment_branches(
            pcaComplete=True,
            importedColumnPresent=True,
            markerComplete=False,
        )
