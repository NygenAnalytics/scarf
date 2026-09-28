import numpy as np

from scarf.agent.parameter_tuning.hvg import (
    HvgGroupVariability,
    aggregate_hvg_rankings,
)


def test_batch_aware_hvg_ranking_prefers_recurrent_features() -> None:
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
        candidate_targets=(3,),
    )

    # Features 2 and 3 rank highly in both groups; ineligible feature 4 is absent.
    assert ranking.ranking.tolist()[:2] == [2, 3]
    assert sorted(ranking.ranking.tolist()) == [0, 1, 2, 3]
