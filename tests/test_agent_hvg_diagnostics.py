import numpy as np
import pytest

from scarf.agent.parameter_tuning.hvg import (
    HvgGroupVariability,
    aggregate_hvg_rankings,
    effective_hvg_candidate_counts,
)


def test_hvg_candidate_counts_are_capped_and_unique() -> None:
    assert effective_hvg_candidate_counts(2500) == (1000, 2000, 2500)
    assert effective_hvg_candidate_counts(8, (3, 5, 10, 10)) == (3, 5, 8)
    with pytest.raises(ValueError, match="greater than 0"):
        effective_hvg_candidate_counts(0)


def test_batch_aware_hvg_ranking_keeps_nested_registered_candidates() -> None:
    corrected = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
    eligible = np.asarray([True, True, True, True, False])
    groups = [
        HvgGroupVariability(
            group_id="str:a",
            cell_count=10,
            corrected_variance=np.asarray([5.0, 1.0, 4.0, 3.0, 2.0]),
            detected_features=np.ones(5, dtype=bool),
        ),
        HvgGroupVariability(
            group_id="str:b",
            cell_count=10,
            corrected_variance=np.asarray([1.0, 5.0, 4.0, 3.0, 2.0]),
            detected_features=np.ones(5, dtype=bool),
        ),
    ]

    ranking = aggregate_hvg_rankings(
        corrected,
        eligible,
        groups,
        valid_group_count=2,
        candidate_targets=(2, 3),
    )

    assert ranking.ranking_mode == "batchAware"
    assert ranking.recurrence.tolist() == [1, 1, 2, 2, 0]
    narrow = ranking.candidate_mask(2)
    broad = ranking.candidate_mask(3)
    assert np.all(~narrow | broad)
    assert int(narrow.sum()) == 2
    assert int(broad.sum()) == 3
