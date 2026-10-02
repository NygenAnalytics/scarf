"""Scientific and operational requests fail closed before a run is created."""

from typing import Any

import pytest
from pydantic import ValidationError

from scarf.agent.models import AnalysisConfig, RuntimeConfig, Study


@pytest.mark.parametrize("field", ["context", "objective"])
def test_study_rejects_whitespace_in_required_scientific_context(field: str) -> None:
    values = {"context": "Supplied observations", "objective": "Discover populations"}
    values[field] = " \t\n "
    with pytest.raises(ValidationError, match="Study text must not be blank"):
        Study.model_validate(values)


@pytest.mark.parametrize("resolutions", [(), (0.25, 0.5, 0.75, 1.0, 1.25)])
def test_scientific_search_requires_one_to_four_resolutions(
    resolutions: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError, match="one to four Leiden resolutions"):
        AnalysisConfig(resolutions=resolutions)


@pytest.mark.parametrize(
    "resolutions",
    [(0.5, 0.5), (0.0,), (-0.5,), (float("nan"),), (float("inf"),)],
)
def test_scientific_search_rejects_duplicate_or_invalid_resolutions(
    resolutions: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError, match="distinct positive finite values"):
        AnalysisConfig(resolutions=resolutions)


@pytest.mark.parametrize(
    "values",
    [
        {"qcPolicy": "manual"},
        {"qcPolicy": "retain", "qcBounds": {"RNA_nCounts": (10, None)}},
        {"qcPolicy": "gentleMad5", "qcBounds": {"RNA_nCounts": (10, None)}},
    ],
)
def test_manual_thresholds_cannot_be_ignored_or_invented_by_other_qc_policies(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="supplied only for manual QC"):
        AnalysisConfig.model_validate(values)


@pytest.mark.parametrize(
    "bounds",
    [
        {"": (1.0, 2.0)},
        {"RNA_nCounts": (float("nan"), None)},
        {"RNA_nCounts": (None, float("inf"))},
    ],
)
def test_manual_qc_requires_named_finite_bounds(
    bounds: dict[str, tuple[float | None, float | None]],
) -> None:
    with pytest.raises(ValidationError, match="names and finite values"):
        AnalysisConfig(qcPolicy="manual", qcBounds=bounds)


def test_manual_qc_accepts_asymmetric_bounds_but_rejects_inverted_ranges() -> None:
    bounds = {"RNA_nCounts": (100.0, None), "RNA_percentMito": (None, 15.0)}
    request = AnalysisConfig(qcPolicy="manual", qcBounds=bounds)
    assert request.qcBounds == bounds
    with pytest.raises(ValidationError, match="lower bound must not exceed upper"):
        AnalysisConfig(qcPolicy="manual", qcBounds={"RNA_nCounts": (100.0, 50.0)})


@pytest.mark.parametrize(
    "values",
    [
        {"maxCandidates": 6},
        {"maxCandidates": 0},
        {"maxFinalists": 3},
        {"maxFinalists": 0},
        {"hvgCount": 1},
        {"pcaDims": 1},
        {"neighborsK": 1},
        {"randomSeed": -1},
    ],
)
def test_scientific_request_enforces_supported_execution_limits(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError) as error:
        AnalysisConfig.model_validate(values)
    assert error.value.errors()[0]["loc"] == (next(iter(values)),)


@pytest.mark.parametrize(
    "settings",
    [
        {"API-Key": "not-a-real-key"},
        {"extra_body": {"authorization": "not-a-real-token"}},
        {"metadata": [{"password": "not-a-real-password"}]},
        {"metadata": ({"cookie": "not-a-real-cookie"},)},
        {"extra_headers": {"custom": "value"}},
    ],
)
def test_runtime_rejects_credentials_even_in_nested_saved_settings(
    settings: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Configure credentials on the provider"):
        RuntimeConfig(modelSettings=settings)


@pytest.mark.parametrize("setting", ["max_tokens", "max_output_tokens", "timeout"])
def test_runtime_requires_explicit_output_and_deadline_overrides(setting: str) -> None:
    with pytest.raises(ValidationError, match="explicit runtime output-token"):
        RuntimeConfig(modelSettings={setting: 256})


def test_runtime_preserves_safe_model_configuration_and_explicit_overrides() -> None:
    settings = {
        "temperature": 0.2,
        "metadata": [{"purpose": "provisional annotation"}],
        "extra_body": {"sampling": {"top_p": 0.8}, "stop": ("END",)},
    }
    runtime = RuntimeConfig(
        modelSettings=settings, maxOutputTokens=512, decisionTimeout=45.0
    )
    assert runtime.modelSettings == settings
    assert runtime.model_dump(exclude_unset=True) == {
        "modelSettings": settings,
        "maxOutputTokens": 512,
        "decisionTimeout": 45.0,
    }
    assert RuntimeConfig().model_dump(exclude_unset=True) == {}


@pytest.mark.parametrize(
    "values",
    [
        {"maxRequests": 0},
        {"maxRequestsPerDecision": 4},
        {"maxPromptBytes": 1023},
        {"maxOutputTokens": 127},
        {"decisionTimeout": 0},
        {"decisionTimeout": float("inf")},
        {"nthreads": 0},
    ],
)
def test_runtime_rejects_invalid_provider_and_resource_limits(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError) as error:
        RuntimeConfig.model_validate(values)
    assert error.value.errors()[0]["loc"] == (next(iter(values)),)


def test_requests_reject_unknown_fields_and_reassignment_of_frozen_policy() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AnalysisConfig.model_validate({"autoRepair": True})
    request = AnalysisConfig()
    with pytest.raises(ValidationError, match="Instance is frozen"):
        request.scoreDoublets = True
    assert request.scoreDoublets is False


def test_lenient_native_probe_policy_is_frozen_and_old_budgets_remain_readable() -> (
    None
):
    config = AnalysisConfig()
    assert config.interactionMode == "lenient"
    assert config.maxCandidates == 5
    assert AnalysisConfig(interactionMode="strict", maxCandidates=4).maxCandidates == 4
    for old_budget in (1, 2, 3):
        assert (
            AnalysisConfig.model_validate({"maxCandidates": old_budget}).maxCandidates
            == old_budget
        )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.interactionMode = "strict"


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_new_runs_refuse_old_search_budgets_before_creating_records(
    budget: int, tmp_path: Any
) -> None:
    from scarf.agent import analyze_rna

    destination = tmp_path / "analysis"
    with pytest.raises(ValueError, match="four native probes"):
        analyze_rna(
            tmp_path,
            run_dir=destination,
            model="unused-model",
            study=Study(context="Observed cells", objective="Describe populations"),
            config=AnalysisConfig(maxCandidates=budget),
        )
    assert not destination.exists()
