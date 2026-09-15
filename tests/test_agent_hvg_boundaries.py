"""Feature ranking respects frozen eligibility and measured group support."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent.parameter_tuning.hvg import (
    HvgGroupVariability,
    aggregate_hvg_rankings,
    effective_hvg_candidate_counts,
    rank_core_hvgs,
)
from scarf.storage.refs import ArtifactRef


def _ranking_store() -> tuple[
    Any, ArtifactRef, ArtifactRef, dict[str, np.ndarray], list[np.ndarray]
]:
    eligible = ArtifactRef("assay", "feature_selection", "a" * 64, "RNA")
    statistics = ArtifactRef("assay", "feature_selection", "b" * 64, "RNA")
    data = {
        "mask": np.asarray([True, False, True, True, True]),
        "variance": np.asarray([1.0, 100, 2, 2, 0]),
    }
    saved = []

    def load(ref: ArtifactRef) -> dict[str, np.ndarray]:
        return (
            {"values": data["mask"]}
            if ref == eligible
            else {"corrected_variance": data["variance"]}
        )

    def save(
        *, from_assay: str, mask: np.ndarray, invalidate_cache: bool
    ) -> ArtifactRef:
        assert from_assay == "RNA" and not invalidate_cache
        saved.append(mask.copy())
        return eligible

    return (
        SimpleNamespace(load_artifact=load, set_feature_selection=save),
        eligible,
        statistics,
        data,
        saved,
    )


def test_ranked_feature_intervention_preserves_eligibility_and_tie_order() -> None:
    store, eligible, statistics, _, saved = _ranking_store()
    rank_core_hvgs(store, eligible=eligible, statistics=statistics, top_n=3)
    np.testing.assert_array_equal(saved[-1], [True, False, True, True, False])
    rank_core_hvgs(
        store,
        eligible=eligible,
        statistics=statistics,
        top_n=3,
        ranking=np.asarray([4, 3, 2, 1, 0]),
    )
    np.testing.assert_array_equal(saved[-1], [False, False, True, True, True])
    assert len(saved) == 2


@pytest.mark.parametrize(
    "damage",
    [
        "unaligned",
        "nonfinite",
        "duplicate",
        "matrix",
        "negativeIndex",
        "largeIndex",
        "incompleteUniverse",
        "tooFewRequested",
        "booleanCount",
        "tooFewEligible",
    ],
)
def test_ranked_feature_intervention_rejects_incomplete_evidence_before_persistence(
    damage: str,
) -> None:
    store, eligible, statistics, data, saved = _ranking_store()
    ranking = None
    top_n = 3
    if damage == "unaligned":
        data["variance"] = data["variance"][:-1]
    elif damage == "nonfinite":
        data["variance"][0] = np.nan
    elif damage == "duplicate":
        ranking = np.asarray([0, 0, 2, 3, 4])
    elif damage == "matrix":
        ranking = np.asarray([[0, 2, 3, 4]])
    elif damage == "negativeIndex":
        ranking = np.asarray([-1, 0, 2, 3, 4])
    elif damage == "largeIndex":
        ranking = np.asarray([0, 2, 3, 4, 5])
    elif damage == "incompleteUniverse":
        ranking = np.asarray([0, 2, 3])
    elif damage == "tooFewRequested":
        top_n = 2
    elif damage == "booleanCount":
        top_n = True
    else:
        data["mask"][:] = [True, False, True, False, False]
    with pytest.raises(ValueError):
        rank_core_hvgs(
            store,
            eligible=eligible,
            statistics=statistics,
            top_n=top_n,
            ranking=ranking,
        )
    assert saved == []


@pytest.mark.parametrize(
    "count,targets",
    [(True, (1,)), (0, (1,)), (10, "123"), (10, (True,)), (10, (0,)), (10, ())],
)
def test_hvg_search_requires_valid_registered_counts(count: Any, targets: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        effective_hvg_candidate_counts(count, targets)


def test_one_supported_group_is_explicit_global_evidence_and_masks_remain_nested() -> (
    None
):
    def unavailable_groups() -> Any:
        raise AssertionError(
            "An unsupported batch ranking must not consume group summaries"
        )
        yield

    ranking = aggregate_hvg_rankings(
        np.asarray([3.0, 9, 2, 1]),
        np.asarray([True, False, True, True]),
        unavailable_groups(),
        valid_group_count=1,
        candidate_targets=(2, 3, 5),
    )
    assert ranking.ranking_mode == "global"
    assert ranking.eligible_feature_count == 3
    assert ranking.candidate_counts == (2, 3)
    np.testing.assert_array_equal(ranking.candidate_mask(2), [True, False, True, False])
    assert np.all(ranking.candidate_mask(2) <= ranking.candidate_mask(3))
    with pytest.raises(ValueError, match="registered counts"):
        ranking.candidate_mask(1)


@pytest.mark.parametrize(
    "damage",
    [
        "globalShape",
        "globalNonfinite",
        "globalNegative",
        "groupCountType",
        "negativeGroups",
        "excessGroups",
        "missingGroups",
        "emptyGroupId",
        "cellCountType",
        "emptyGroup",
        "groupShape",
        "groupNonfinite",
        "groupNegative",
        "undetectedGroup",
    ],
)
def test_batch_aware_ranking_cannot_claim_unavailable_group_evidence(
    damage: str,
) -> None:
    corrected = np.asarray([3.0, 2, 1])
    eligible = np.ones(3, dtype=bool)
    first = HvgGroupVariability("batch_a", 100, corrected.copy(), eligible.copy())
    second = replace(first, group_id="batch_b")
    groups = [first, second]
    declared: Any = 2
    if damage == "globalShape":
        corrected = corrected[:2]
    elif damage == "globalNonfinite":
        corrected[0] = np.nan
    elif damage == "globalNegative":
        corrected[0] = -1
    elif damage == "groupCountType":
        declared = True
    elif damage == "negativeGroups":
        declared = -1
    elif damage == "excessGroups":
        groups.append(replace(first, group_id="batch_c"))
    elif damage == "missingGroups":
        groups.pop()
    elif damage == "emptyGroupId":
        groups[0] = replace(first, group_id="")
    elif damage == "cellCountType":
        groups[0] = replace(first, cell_count=True)
    elif damage == "emptyGroup":
        groups[0] = replace(first, cell_count=0)
    elif damage == "groupShape":
        groups[0] = replace(first, detected_features=np.ones(2, dtype=bool))
    elif damage == "groupNonfinite":
        groups[0] = replace(first, corrected_variance=np.asarray([np.nan, 2, 1]))
    elif damage == "groupNegative":
        groups[0] = replace(first, corrected_variance=np.asarray([-1.0, 2, 1]))
    else:
        groups[0] = replace(first, detected_features=np.zeros(3, dtype=bool))
    with pytest.raises((TypeError, ValueError)):
        aggregate_hvg_rankings(corrected, eligible, groups, valid_group_count=declared)
