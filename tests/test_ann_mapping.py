import numpy as np
import pytest

from scarf.matrix import ChunkedArray
from scarf.neighbors.stream import AnnStream


def _ann_stream() -> AnnStream:
    data = ChunkedArray.from_numpy(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
        block_size=2,
    )
    return AnnStream(
        data=data,
        k=1,
        n_cluster=2,
        reduction_method="pca",
        dims=2,
        loadings=np.eye(2),
        use_for_pca=np.ones(3, dtype=bool),
        mu=np.array([0.5, 0.5]),
        sigma=np.array([0.5, 0.5]),
        ann_metric="l2",
        ann_efc=10,
        ann_ef=10,
        ann_m=8,
        nthreads=1,
        ann_parallel=False,
        rand_state=0,
        do_kmeans_fit=False,
        disable_scaling=False,
        ann_idx=None,
        lsi_skip_first=True,
        lsi_params={},
        harmonize=False,
    )


def test_transform_query_matches_existing_reference_reducer():
    ann = _ann_stream()
    query = np.array([[1.5, 0.0], [0.0, 1.5]], dtype=np.float64)

    assert not hasattr(ann, "annPath")
    assert not hasattr(ann, "_embedding_bytes")
    assert not hasattr(ann, "embeddings")
    np.testing.assert_array_equal(ann.transform_query(query), ann.reducer(query))


def test_transform_query_validates_feature_shape_and_finite_results():
    ann = _ann_stream()

    with pytest.raises(ValueError, match="features"):
        ann.transform_query(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        ann.transform_query(np.array([[np.nan, 0.0]]))
