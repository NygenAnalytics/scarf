from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf.features.enrichment.net import PreparedNetwork, prepare_network
from scarf.features.markers import regression
from scarf.metrics.concordance import label_concordance_score
from scarf.metrics.lisi import (
    _lisi_knn_summary,
    _neighbor_probabilities,
    _simpson_from_probabilities,
    compute_lisi,
    compute_simpson,
    lisi_batch_mixing_score,
)


def _prepared_network(**overrides: object) -> PreparedNetwork:
    values: dict[str, object] = {
        "source_names": np.array(["A"]),
        "source_sizes": np.array([1], dtype=np.int64),
        "matched_feature_index": np.array([3], dtype=np.int64),
        "edge_source_index": np.array([0], dtype=np.int64),
        "edge_feature_index": np.array([3], dtype=np.int64),
        "edge_weight": np.array([1.0]),
        "network_digest": "digest",
    }
    values.update(overrides)
    return PreparedNetwork(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_names": np.array([["A"]])}, "one-dimensional"),
        ({"source_names": np.array([], dtype=str)}, "at least one source"),
        ({"source_sizes": np.array([1, 1])}, "names and sizes"),
        ({"edge_weight": np.array([1.0, 2.0])}, "edge arrays"),
        (
            {
                "source_sizes": np.array([0], dtype=np.int64),
                "matched_feature_index": np.array([], dtype=np.int64),
                "edge_source_index": np.array([], dtype=np.int64),
                "edge_feature_index": np.array([], dtype=np.int64),
                "edge_weight": np.array([], dtype=np.float64),
            },
            "matched edge",
        ),
        ({"source_sizes": np.array([0])}, "sizes must be positive"),
        (
            {
                "source_names": np.array(["A", "A"]),
                "source_sizes": np.array([1, 1]),
                "edge_source_index": np.array([0, 1]),
                "edge_feature_index": np.array([3, 4]),
                "edge_weight": np.array([1.0, 1.0]),
                "matched_feature_index": np.array([3, 4]),
            },
            "names must be unique",
        ),
        ({"source_names": np.array([""])}, "non-empty strings"),
        (
            {
                "source_names": np.array(["B", "A"]),
                "source_sizes": np.array([1, 1]),
                "edge_source_index": np.array([0, 1]),
                "edge_feature_index": np.array([3, 4]),
                "edge_weight": np.array([1.0, 1.0]),
                "matched_feature_index": np.array([3, 4]),
            },
            "must be sorted",
        ),
        ({"source_sizes": np.array([1.0])}, "integer dtypes"),
        ({"matched_feature_index": np.array([-1])}, "sorted and unique"),
        ({"edge_source_index": np.array([1])}, "source indices"),
        ({"edge_feature_index": np.array([4])}, "feature indices"),
        ({"source_sizes": np.array([2])}, "sizes do not match"),
        (
            {
                "source_sizes": np.array([2]),
                "matched_feature_index": np.array([3, 4]),
                "edge_source_index": np.array([0, 0]),
                "edge_feature_index": np.array([4, 3]),
                "edge_weight": np.array([1.0, 1.0]),
            },
            "canonically sorted",
        ),
        ({"edge_weight": np.array([np.inf])}, "weights must be finite"),
        ({"network_digest": ""}, "non-empty string"),
    ],
)
def test_prepared_network_rejects_each_corrupt_internal_contract(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepared_network(**overrides)


def _prepare(
    net: object,
    *,
    names: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    tmin: object = 1,
    weighted: object = False,
) -> PreparedNetwork:
    return prepare_network(
        net,  # type: ignore[arg-type]
        active_feature_names=np.array(["GeneA"]) if names is None else names,
        active_feature_index=np.array([0]) if indices is None else indices,
        tmin=tmin,  # type: ignore[arg-type]
        weighted=weighted,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: _prepare([]), TypeError, "pandas DataFrame"),
        (
            lambda: _prepare(
                pd.DataFrame({"source": ["Set"], "target": ["GeneA"]}),
                tmin=True,
            ),
            ValueError,
            "integer greater",
        ),
        (
            lambda: _prepare(
                pd.DataFrame({"source": ["Set"], "target": ["GeneA"]}),
                weighted=1,
            ),
            TypeError,
            "boolean",
        ),
        (
            lambda: _prepare(pd.DataFrame({"source": ["Set"]})),
            ValueError,
            "contain 'source' and 'target'",
        ),
        (
            lambda: _prepare(
                pd.DataFrame(
                    [["Set", "Other", "GeneA"]],
                    columns=["source", "source", "target"],
                )
            ),
            ValueError,
            "unique names",
        ),
        (
            lambda: _prepare(
                pd.DataFrame({"source": ["Set"], "target": ["GeneA"]}),
                names=np.array([["GeneA"]]),
            ),
            ValueError,
            "one-dimensional",
        ),
        (
            lambda: _prepare(
                pd.DataFrame({"source": ["Set"], "target": ["GeneA"]}),
                names=np.array([], dtype=str),
                indices=np.array([], dtype=np.int64),
            ),
            ValueError,
            "no active features",
        ),
        (
            lambda: _prepare(
                pd.DataFrame({"source": ["Set"], "target": ["GeneA"]}),
                names=np.array(["GeneA", "GeneB"]),
            ),
            ValueError,
            "must be aligned",
        ),
        (
            lambda: _prepare(
                pd.DataFrame({"source": ["Set"], "target": ["GeneA"]}),
                indices=np.array([0.0]),
            ),
            ValueError,
            "integer dtype",
        ),
        (
            lambda: _prepare(
                pd.DataFrame({"source": ["Set"], "target": ["GeneA"]}),
                indices=np.array([-1]),
            ),
            ValueError,
            "non-negative and unique",
        ),
        (
            lambda: _prepare(pd.DataFrame({"source": [None], "target": ["GeneA"]})),
            ValueError,
            "must not be missing",
        ),
        (
            lambda: _prepare(pd.DataFrame({"source": ["  "], "target": ["GeneA"]})),
            ValueError,
            "must be non-empty",
        ),
        (
            lambda: _prepare(
                pd.DataFrame(
                    {"source": ["Set"], "target": ["GeneA"], "weight": ["bad"]}
                ),
                weighted=True,
            ),
            ValueError,
            "must be numeric",
        ),
        (
            lambda: _prepare(
                pd.DataFrame(
                    {"source": ["Set"], "target": ["GeneA"], "weight": [np.inf]}
                ),
                weighted=True,
            ),
            ValueError,
            "must be finite",
        ),
        (
            lambda: _prepare(
                pd.DataFrame({"source": ["Set"], "target": ["GeneA"], "weight": [0.0]}),
                weighted=True,
            ),
            ValueError,
            "no non-zero edges",
        ),
        (
            lambda: _prepare(pd.DataFrame({"source": ["Set"], "target": ["Missing"]})),
            ValueError,
            "no targets overlapping",
        ),
    ],
)
def test_prepare_network_rejects_invalid_user_inputs(
    call: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        call()


def test_python_regression_kernel_reports_nonfinite_sentinel_and_clipped_r() -> None:
    data = np.array(
        [
            [1.0, 0.0, 2.0, 1.0, 4.0],
            [np.nan, 0.0, 2.0, 2.0, 3.0],
            [2.0, 0.0, 2.0, 3.0, 2.0],
            [3.0, 0.0, 2.0, 4.0, 1.0],
        ]
    )
    r_values, status = regression._regression_r_batch.py_func(
        data,
        np.array([-1.5, -0.5, 0.5, 1.5]),
        1e-300,
        2,
        np.finfo(float).eps,
    )

    np.testing.assert_array_equal(
        status,
        [
            regression._REG_NONFINITE,
            regression._REG_SENTINEL,
            regression._REG_SENTINEL,
            regression._REG_OK,
            regression._REG_OK,
        ],
    )
    np.testing.assert_array_equal(r_values, [0.0, 0.0, 0.0, 1.0, -1.0])

    r_values, status = regression._regression_r_batch.py_func(
        np.arange(1.0, 5.0).reshape(-1, 1),
        np.array([-1.5, -0.5, 0.5, 1.5]),
        0.0,
        2,
        np.finfo(float).eps,
    )
    np.testing.assert_array_equal(r_values, [0.0])
    np.testing.assert_array_equal(status, [regression._REG_SENTINEL])


def test_two_cell_regression_rejects_nonfinite_feature_with_its_label() -> None:
    with pytest.raises(ValueError, match="bad.*non-finite"):
        regression._regression_batch_results(
            np.array([[1.0], [np.nan]]),
            np.array([-0.5, 0.5]),
            0.25,
            np.array([0.0, 1.0]),
            1,
            np.array(["bad"]),
        )


def test_lisi_low_level_validators_reject_invalid_arrays() -> None:
    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        _neighbor_probabilities(np.array([]), 1.0, 1e-5)

    probabilities = np.full((2, 2), 0.5)
    labels = np.array([0, 1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="matching shapes"):
        _simpson_from_probabilities(probabilities, np.array([[0], [1]]), labels, 2)
    with pytest.raises(TypeError, match="integers"):
        _simpson_from_probabilities(
            probabilities, np.array([[0.0, 1.0], [1.0, 2.0]]), labels, 2
        )
    with pytest.raises(IndexError, match="outside"):
        _simpson_from_probabilities(
            probabilities, np.array([[0, 3], [1, 2]]), labels, 2
        )
    with pytest.raises(FloatingPointError, match="finite Simpson"):
        _simpson_from_probabilities(
            np.zeros((2, 2)), np.array([[0, 1], [1, 2]]), labels, 2
        )


def test_compute_lisi_rejects_misaligned_graph_and_metadata() -> None:
    metadata = pd.DataFrame({"batch": ["a", "b"]})
    with pytest.raises(ValueError, match="two-dimensional"):
        compute_lisi(np.array([1.0]), np.array([[0, 1]]), metadata, ["batch"])
    with pytest.raises(ValueError, match="matching shapes"):
        compute_lisi(
            np.ones((2, 3)), np.ones((2, 2), dtype=np.int64), metadata, ["batch"]
        )
    with pytest.raises(ValueError, match="Metadata rows"):
        compute_lisi(
            np.ones((3, 3)),
            np.zeros((3, 3), dtype=np.int64),
            metadata,
            ["batch"],
        )


def test_lisi_summary_rejects_invalid_knn_shapes_and_neighbor_count() -> None:
    kwargs = {
        "labels": np.array(["a", "b"]),
        "perplexity": None,
        "scale": True,
        "invert": False,
        "label_name": "Batch",
    }
    with pytest.raises(ValueError, match="two-dimensional"):
        _lisi_knn_summary(np.ones(2), np.ones((2, 3)), **kwargs)
    with pytest.raises(ValueError, match="matching shapes"):
        _lisi_knn_summary(np.ones((2, 3)), np.ones((2, 4)), **kwargs)
    with pytest.raises(ValueError, match="at least three neighbors"):
        _lisi_knn_summary(np.ones((2, 2)), np.ones((2, 2)), **kwargs)


def test_compute_simpson_rejects_invalid_shapes_and_missing_labels() -> None:
    labels = pd.Categorical(["a", "b"])
    with pytest.raises(ValueError, match="two-dimensional"):
        compute_simpson(np.ones(2), np.ones((2, 2)), labels, 1.0)
    with pytest.raises(ValueError, match="matching shapes"):
        compute_simpson(np.ones((2, 2)), np.ones((2, 3)), labels, 1.0)
    with pytest.raises(ValueError, match="missing values"):
        compute_simpson(
            np.ones((3, 2)),
            np.zeros((3, 2), dtype=np.int64),
            pd.Categorical(["a", None]),
            1.0,
        )


@pytest.mark.parametrize(
    ("scores", "labels", "message"),
    [
        (np.ones((2, 1)), ["a", "b"], "aligned vectors"),
        (np.array([1.0, np.inf]), ["a", "b"], "finite values"),
        (np.ones(2), ["a", None], "missing values"),
        (np.ones(2), ["a", "a"], "at least two batches"),
    ],
)
def test_batch_mixing_score_rejects_invalid_inputs(
    scores: np.ndarray, labels: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        lisi_batch_mixing_score(scores, labels)


def test_label_concordance_rejects_invalid_vectors_and_metric() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        label_concordance_score([np.ones((1, 2)), np.ones(2)])
    with pytest.raises(ValueError, match="matching lengths"):
        label_concordance_score([np.ones(2), np.ones(3)])
    with pytest.raises(ValueError, match="missing values"):
        label_concordance_score([np.array(["a", None]), np.array(["a", "b"])])
    with pytest.raises(ValueError, match="not one of"):
        label_concordance_score([np.ones(2), np.ones(2)], metric="bad")  # type: ignore[arg-type]
