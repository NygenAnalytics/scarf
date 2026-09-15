from tests.agent_examples import example
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix

from scarf.agent.parameter_tuning import diagnostics as diagnostics_module
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from scarf.agent.parameter_tuning.diagnostics import (
    _cross_unit_support,
    _subsample_partition_stability,
    score_advisory_doublets,
)
from scarf.storage.refs import ArtifactRef


def test_cross_unit_support_requires_replication_per_cluster() -> None:
    labels = np.asarray([1, 1, 2, 2])
    units = np.asarray(["a", "b", "a", "a"])

    assert _cross_unit_support(labels, units) == 0.5
    assert _cross_unit_support(labels, np.asarray(["a"] * 4)) is None


def test_subsample_partition_stability_reclusters_induced_graph() -> None:
    community = csr_matrix(np.ones((8, 8)) - np.eye(8))
    graph = block_diag((community, community), format="csr")
    labels = np.asarray([1] * 8 + [2] * 8)

    assert _subsample_partition_stability(graph, labels, 0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="does not align"):
        _subsample_partition_stability(graph, labels[:-1], 0.5)


def test_advisory_doublets_record_duplicate_feature_limitation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="a" * 64,
    )
    clusters = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        artifact_id="b" * 64,
    )
    monkeypatch.setattr(
        diagnostics_module,
        "resolve_native_doublet_inputs",
        lambda *_args: (clusters, graph),
    )

    class Features:
        def fetch_all(self, column: str) -> np.ndarray:
            assert column == "ids"
            return np.asarray(["A", "A", "B", "C", "C"])

    class Store:
        def get_assay(self, assay: str) -> SimpleNamespace:
            assert assay == "RNA"
            return SimpleNamespace(feats=Features())

        def run_doublet_detection(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("duplicate identifiers must stop doublet mapping")

    evidence = score_advisory_doublets(
        Store(),
        example(ParameterCandidateEvaluation),
        [example(ParameterCandidateEvaluation)],
        assay="RNA",
        feature_selection=ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="feature_selection",
            artifact_id="c" * 64,
        ),
        capture_column="library_id",
    )

    assert evidence.scores == ()
    assert evidence.cell_selections == ()
    assert evidence.native_graph == graph
    assert evidence.native_clusters == clusters
    assert len(evidence.limitations) == 1
    assert "2 feature identifiers are duplicated" in evidence.limitations[0]
