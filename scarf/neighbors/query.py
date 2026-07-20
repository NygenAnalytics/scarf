from collections.abc import Iterable, Iterator
from typing import cast

import numpy as np

from .stream import AnnStream


def self_query_blocks(
    ann_stream: AnnStream,
    transformed_blocks: Iterable[np.ndarray],
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, int]]:
    """Query ordered embedding blocks against their source ANN index."""
    start = 0
    for block in transformed_blocks:
        end = start + block.shape[0]
        indices, distances, missed = cast(
            tuple[np.ndarray, np.ndarray, int],
            ann_stream.transform_ann(
                block,
                k=ann_stream.k,
                self_indices=np.arange(start, end),
            ),
        )
        yield start, end, indices, distances, missed
        start = end
