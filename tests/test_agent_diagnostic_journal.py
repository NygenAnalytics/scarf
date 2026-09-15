"""Observed diagnostic calls and reuse remain honest across failed attempts."""

import pytest

from scarf.agent.orchestrator import journal
from scarf.agent.parameter_tuning.execution import diagnostic_call, diagnostic_reuse
from tests.agent_journal_store import memory_journal


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_diagnostic_attempt_preserves_actual_calls_and_partial_work(
    interrupted: bool,
) -> None:
    store, prefix, request = memory_journal("analysis")
    error = (
        KeyboardInterrupt("Interrupted diagnostic")
        if interrupted
        else ValueError("Capture evidence failed")
    )
    calls = []

    def score(capture: str) -> str:
        calls.append(capture)
        if capture == "bad":
            raise error
        return "the-same-core-artifact"

    with pytest.raises(type(error)) as caught:
        with journal.diagnostic_attempt(
            store, prefix, request.workflowRunId, {"cells": "frozen"}
        ) as counts:
            assert (
                diagnostic_call("core.doubletDetection", score, "a")
                == "the-same-core-artifact"
            )
            assert (
                diagnostic_call("core.doubletDetection", score, "a")
                == "the-same-core-artifact"
            )
            diagnostic_reuse("metric.batch", "cacheHits")
            diagnostic_call("core.doubletDetection", score, "bad")
    assert caught.value is error
    assert calls == ["a", "a", "bad"]
    assert counts["core.doubletDetection"]["attempted"] == 3
    assert counts["core.doubletDetection"]["completed"] == 2
    assert counts["core.doubletDetection"]["failed"] == 1
    state = journal.analysis_snapshot(store, request.workflowRunId)
    attempt = state["diagnosticAttempts"][0]
    assert attempt["finished"]["status"] == ("interrupted" if interrupted else "failed")
    assert attempt["finished"]["operations"] == counts
    assert "not observable" in attempt["finished"]["interpretation"]
    assert "unknown operation counts" in attempt["finished"]["interpretation"]
    assert attempt["started"]["recordedAtNs"] <= attempt["finished"]["recordedAtNs"]

    with journal.diagnostic_attempt(
        store, prefix, request.workflowRunId, {"cells": "frozen"}
    ):
        diagnostic_reuse("diagnostic.augmentedEvaluation", "restored")
    resumed = journal.analysis_snapshot(store, request.workflowRunId)
    assert resumed["diagnosticAttempts"][0] == attempt
    assert len(resumed["diagnosticAttempts"]) == 2
    restored = resumed["diagnosticAttempts"][1]["finished"]["operations"]
    assert "core.doubletDetection" not in restored
    assert restored["diagnostic.augmentedEvaluation"]["restored"] == 1


def test_unfinished_and_legacy_attempts_do_not_manufacture_zero_computation() -> None:
    store, prefix, request = memory_journal("analysis")
    assert (
        journal.analysis_snapshot(store, request.workflowRunId)["diagnosticAttempts"]
        == []
    )
    journal.save_checkpoint(
        store,
        prefix,
        request.workflowRunId,
        "parameter_tuning/diagnostic_attempts/interrupted/started",
        {"cells": "frozen"},
        {"recordedAtNs": 1},
    )
    attempt = journal.analysis_snapshot(store, request.workflowRunId)[
        "diagnosticAttempts"
    ][0]
    assert "finished" not in attempt
    assert "operations" not in attempt["started"]
    journal.save_checkpoint(
        store,
        prefix,
        request.workflowRunId,
        "parameter_tuning/diagnostic_attempts/interrupted/unrecognized",
        {"cells": "frozen"},
        {},
    )
    with pytest.raises(ValueError, match="Unknown diagnostic attempt"):
        journal.analysis_snapshot(store, request.workflowRunId)


@pytest.mark.parametrize("missing_start", [False, True])
def test_diagnostic_finish_requires_the_exact_original_start(
    missing_start: bool,
) -> None:
    store, prefix, request = memory_journal("analysis")
    key = "parameter_tuning/diagnostic_attempts/attempt"
    if not missing_start:
        journal.save_checkpoint(
            store,
            prefix,
            request.workflowRunId,
            key + "/started",
            {"cells": "original"},
            {"recordedAtNs": 1},
        )
    journal.save_checkpoint(
        store,
        prefix,
        request.workflowRunId,
        key + "/finished",
        {"cells": "different"},
        {"recordedAtNs": 2, "status": "completed", "operations": {}},
    )
    with pytest.raises(
        ValueError, match="matching start" if missing_start else "inputs changed"
    ):
        journal.analysis_snapshot(store, request.workflowRunId)


def test_primary_evidence_restore_and_review_replay_count_no_new_diagnostic_calls(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    from scarf.agent.parameter_tuning.execution import diagnostic_work
    from scarf.agent.orchestrator import rna_tuning
    from tests.test_agent_rna_evidence_mode import make_run, assess

    request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, object())
    monkeypatch.setattr(rna_tuning, "run_agent_sync", assess)
    with diagnostic_work() as first:
        run.review("full", 0, selected, {})
    with diagnostic_work() as replay:
        run.review("full", 0, selected, {})
    assert "diagnostic.reviewEvidence" not in first
    assert replay["diagnostic.reviewEvidence"]["restored"] == 1
    assert all(row["attempted"] == 0 for row in replay.values())


@pytest.mark.parametrize(
    "failure",
    [
        None,
        ValueError("Scientific evidence failed"),
        KeyboardInterrupt("Interrupted diagnostic"),
    ],
)
def test_diagnostic_persistence_failure_cannot_replace_the_original_error(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException | None
) -> None:
    store, prefix, request = memory_journal("analysis")
    original_save = journal.save_checkpoint

    def save(store, prefix, workflow, key, inputs, outputs):
        if key.endswith("/finished"):
            raise OSError("Store is full")
        return original_save(store, prefix, workflow, key, inputs, outputs)

    monkeypatch.setattr(journal, "save_checkpoint", save)
    with pytest.raises(type(failure) if failure is not None else OSError) as caught:
        with journal.diagnostic_attempt(
            store, prefix, request.workflowRunId, {"cells": "frozen"}
        ):
            if failure is not None:
                raise failure
    if failure is not None:
        assert caught.value is failure
        assert "Store is full" in " ".join(failure.__notes__)
    else:
        assert str(caught.value) == "Store is full"
    attempt = journal.analysis_snapshot(store, request.workflowRunId)[
        "diagnosticAttempts"
    ][0]
    assert "started" in attempt and "finished" not in attempt


from tests.test_agent_rna_adaptive import checkpoints as memory_checkpoints  # noqa: E402, F401
