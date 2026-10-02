"""Crash boundaries, integrity, and local writer ownership."""

import json
import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scarf.agent.records import (
    RecordError,
    RunLockedError,
    RunRecords,
    digest,
    process_alive,
    process_identity,
    procedure_identity,
    run_lock,
    source_lock,
)


def test_reopen_preserves_manifest_evidence_and_order(tmp_path: Path) -> None:
    manifest = {"request": {"assay": "RNA"}, "seed": 44}
    records = RunRecords.create(tmp_path / "run", manifest)
    manifest["seed"] = 9
    reference = records.write_evidence(
        "context", {"missing": None, "labels": ["α", "β"]}
    )
    records.append("prepared", evidence=reference)
    records.append("completed", pipelineLabel="fixed-final")

    reopened = RunRecords(tmp_path / "run")
    assert reopened.manifest["seed"] == 44
    changed = reopened.manifest
    changed["seed"] = 8
    assert reopened.manifest["seed"] == 44
    assert reopened.read_json(reference)["labels"] == ["α", "β"]
    assert [event["kind"] for event in reopened.events()] == ["prepared", "completed"]
    completed = reopened.latest("completed")
    assert completed is not None
    assert completed["pipelineLabel"] == "fixed-final"
    assert reopened.latest("absent") is None


def test_creation_never_reuses_a_directory(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    with pytest.raises(RecordError, match="already exists"):
        RunRecords.create(directory, {})
    assert not (directory / "run.json").exists()
    invalid = tmp_path / "invalid"
    with pytest.raises(RecordError, match="finite"):
        RunRecords.create(invalid, {"value": float("nan")})
    assert not invalid.exists()


def test_immutable_writes_are_idempotent_but_never_replace(tmp_path: Path) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    path = records.write_json("calls/one.json", {"output": [1, 2]})
    records.write_json(path, {"output": [1, 2]})
    with pytest.raises(RecordError, match="Immutable"):
        records.write_json(path, {"output": [2, 1]})
    assert records.read_json(path) == {"output": [1, 2]}
    with pytest.raises(RecordError, match="manifest"):
        records.write_json("run.json", {"replacement": True}, immutable=False)
    with pytest.raises(RecordError, match="append"):
        records.write_json("events/000001.json", {})


@pytest.mark.parametrize("relative", ["../outside.json", "/tmp/outside.json", ""])
def test_record_paths_cannot_escape(tmp_path: Path, relative: str) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    with pytest.raises(RecordError, match="relative"):
        records.write_json(relative, {})


def test_symlink_evidence_cannot_escape_or_alias(tmp_path: Path) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    outside = tmp_path / "outside"
    outside.mkdir()
    (records.path / "evidence").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RecordError, match="symlink"):
        records.write_evidence("example", {})
    assert not list(outside.iterdir())


@pytest.mark.parametrize("fault", ["gap", "edited", "truncated", "duplicate"])
def test_inconsistent_events_stop_recovery(tmp_path: Path, fault: str) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    records.append("started")
    records.append("completed", result="valid")
    first = records.path / "events/000001.json"
    last = records.path / "events/000002.json"
    if fault == "gap":
        first.unlink()
    elif fault == "edited":
        event = json.loads(last.read_text())
        event["result"] = "altered"
        last.write_text(json.dumps(event))
    elif fault == "truncated":
        last.write_text('{"sequence":')
    else:
        last.write_text('{"sequence":2,"sequence":2}')
    with pytest.raises(RecordError):
        RunRecords(records.path)


def test_abandoned_atomic_write_is_not_a_committed_event(tmp_path: Path) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    records.append("started")
    (records.path / "events/.tmp-interrupted").write_text('{"unfinished":')
    reopened = RunRecords(records.path)
    assert len(reopened.events()) == 1
    assert reopened.append("continued")["sequence"] == 2


def test_replace_failure_preserves_previous_report_and_removes_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    records.write_json("report.json", {"state": "old"}, immutable=False)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        records.write_json("report.json", {"state": "new"}, immutable=False)
    assert records.read_json("report.json") == {"state": "old"}
    assert not list(records.path.glob(".tmp-*"))


def test_failed_file_sync_never_publishes_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = RunRecords.create(tmp_path / "run", {})

    def fail_sync(descriptor: int) -> None:
        raise OSError("simulated full disk")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="full disk"):
        records.append("neverCommitted")
    assert records.events() == []
    assert not list((records.path / "events").glob(".tmp-*"))


def test_locks_reject_another_writer_and_survive_release(tmp_path: Path) -> None:
    path = tmp_path / "run"
    with run_lock(path):
        assert not path.exists()
        with pytest.raises(RunLockedError):
            with run_lock(path):
                pytest.fail("Second writer entered")
    with run_lock(path):
        RunRecords.create(path, {})
    assert (tmp_path / ".run.scarf-agent-run.lock").exists()


def test_source_aliases_share_adjacent_lock(tmp_path: Path) -> None:
    source = tmp_path / "mount"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with source_lock(source):
        with pytest.raises(RunLockedError):
            with source_lock(alias):
                pytest.fail("Aliased source acquired twice")
    assert not list(source.iterdir())
    assert (tmp_path / ".mount.scarf-agent.lock").exists()


def test_process_exit_releases_writer_without_deleting_lock(tmp_path: Path) -> None:
    path = tmp_path / "run"
    code = (
        "import sys; from scarf.agent.records import run_lock\n"
        "with run_lock(sys.argv[1]):\n"
        " print('locked',flush=True)\n"
        " sys.stdin.read()\n"
    )
    with subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as worker:
        assert worker.stdout is not None
        assert worker.stdout.readline().strip() == "locked"
        with pytest.raises(RunLockedError):
            with run_lock(path):
                pytest.fail("Entered while the other process owns the lock")
        worker.terminate()
        worker.wait(timeout=10)
    with run_lock(path):
        assert not path.exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux process identity"
)
def test_process_death_and_pid_reuse_are_detected(tmp_path: Path) -> None:
    code = (
        "import json,sys; from scarf.agent.records import process_identity; "
        "print(json.dumps(process_identity()),flush=True); sys.stdin.read()"
    )
    with subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as worker:
        assert worker.stdout is not None
        identity = json.loads(worker.stdout.readline())
        assert process_alive(identity)
        reused = {**identity, "startTicks": str(int(identity["startTicks"]) + 1)}
        assert not process_alive(reused)
        assert worker.stdin is not None
        worker.stdin.close()
        worker.wait(timeout=10)
        assert not process_alive(identity)


def test_unknown_process_identity_is_conservative() -> None:
    assert process_alive({})
    assert process_alive({"pid": -1, "platform": sys.platform})
    assert process_alive({"pid": os.getpid(), "platform": "other"})
    assert process_alive(process_identity())


def test_hashes_are_stable_and_reject_nonfinite_values() -> None:
    assert digest({"a": 1, "b": [2]}) == digest({"b": [2], "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})
    with pytest.raises(RecordError):
        digest({"infinite": float("inf")})
    assert len(procedure_identity()) == 64


def test_json_numeric_overflow_is_not_accepted_on_read(tmp_path: Path) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    (records.path / "overflow.json").write_text('{"value": 1e999}')
    with pytest.raises(RecordError, match="finite"):
        records.read_json("overflow.json")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_numbers_are_not_recovery_inputs(
    tmp_path: Path, value: str
) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    (records.path / "evidence.json").write_text(f'{{"value": {value}}}')
    with pytest.raises(RecordError, match="Nonfinite JSON constant"):
        records.read_json("evidence.json")


def test_manifest_requires_an_object_before_creation_and_on_reopen(
    tmp_path: Path,
) -> None:
    with pytest.raises(RecordError, match="JSON object"):
        RunRecords.create(tmp_path / "invalid", [])  # type: ignore[arg-type]
    assert not (tmp_path / "invalid").exists()
    records = RunRecords.create(tmp_path / "run", {})
    (records.path / "run.json").write_text("[]")
    with pytest.raises(RecordError, match="JSON object"):
        RunRecords(records.path)


def test_missing_events_directory_prevents_recovery(tmp_path: Path) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    (records.path / "events").rmdir()
    with pytest.raises(RecordError, match="missing its events directory"):
        RunRecords(records.path)


def test_event_must_be_an_object(tmp_path: Path) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    (records.path / "events/000001.json").write_text("[]")
    with pytest.raises(RecordError, match="Event must be an object"):
        records.events()


@pytest.mark.parametrize("timestamp", [None, "2026-10-01T10:00:00", "invalid", 12])
def test_valid_event_hash_does_not_admit_invalid_timestamp(
    tmp_path: Path, timestamp: object
) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    event = records.append("started")
    del event["recordHash"]
    if timestamp is None:
        del event["timestamp"]
    else:
        event["timestamp"] = timestamp
    event["recordHash"] = digest(event)
    (records.path / "events/000001.json").write_text(json.dumps(event))
    with pytest.raises(RecordError, match="Invalid event timestamp"):
        records.events()


@pytest.mark.parametrize("kind", ["", "  ", None, 1])
def test_invalid_event_kind_never_creates_a_record(
    tmp_path: Path, kind: object
) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    with pytest.raises(RecordError, match="nonempty string"):
        records.append(kind)  # type: ignore[arg-type]
    assert records.events() == []


@pytest.mark.parametrize(
    "field", ["sequence", "timestamp", "previousHash", "recordHash"]
)
def test_event_payload_cannot_replace_journal_envelope(
    tmp_path: Path, field: str
) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    with pytest.raises(RecordError, match="envelope"):
        records.append("started", **{field: "replacement"})
    assert records.events() == []


@pytest.mark.parametrize("identical", [False, True])
def test_atomic_publication_handles_a_competing_immutable_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identical: bool
) -> None:
    records = RunRecords.create(tmp_path / "run", {})
    real_link = os.link

    def competing_link(source: Path, target: Path) -> None:
        target.write_bytes(source.read_bytes() if identical else b'{"winner": true}')
        real_link(source, target)

    monkeypatch.setattr(os, "link", competing_link)
    if identical:
        records.write_json("evidence/selection.json", {"selected": "native"})
        assert records.read_json("evidence/selection.json") == {"selected": "native"}
    else:
        with pytest.raises(RecordError, match="Concurrent record creation"):
            records.write_json("evidence/selection.json", {"selected": "native"})
        assert records.read_json("evidence/selection.json") == {"winner": True}
    assert not list((records.path / "evidence").glob(".tmp-*"))


def test_writer_lock_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    (tmp_path / ".run.scarf-agent-run.lock").symlink_to(target)
    with pytest.raises(RecordError, match="must not be a symlink"):
        with run_lock(tmp_path / "run"):
            pytest.fail("Symlink lock admitted a writer")
    assert target.read_text() == "preserve"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock errors")
def test_lock_io_error_is_not_misreported_as_another_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    def broken_lock(descriptor: int, flags: int) -> None:
        raise OSError(errno.EIO, "filesystem cannot lock")

    monkeypatch.setattr(fcntl, "flock", broken_lock)
    with pytest.raises(OSError) as caught:
        with run_lock(tmp_path / "run"):
            pytest.fail("Broken filesystem admitted a writer")
    assert caught.value.errno == errno.EIO
    assert not isinstance(caught.value, RunLockedError)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux process identity"
)
def test_previous_boot_does_not_block_recovery() -> None:
    identity = process_identity()
    assert not process_alive({**identity, "bootId": "different-boot"})


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux procfs access")
@pytest.mark.parametrize("failure", [PermissionError("denied"), "malformed"])
def test_unreadable_procfs_is_conservative(
    monkeypatch: pytest.MonkeyPatch, failure: object
) -> None:
    identity = process_identity()
    read_text = Path.read_text

    def inaccessible_procfs(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path(f"/proc/{os.getpid()}/stat"):
            if isinstance(failure, BaseException):
                raise failure
            return str(failure)
        return read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", inaccessible_procfs)
    assert process_identity() == {"pid": os.getpid(), "platform": sys.platform}
    assert process_alive(identity)


@pytest.mark.skipif(os.name == "nt", reason="Windows must never probe with kill")
@pytest.mark.parametrize("probe", [None, ProcessLookupError(), PermissionError()])
def test_pid_only_recovery_uses_conservative_existence_probe(
    monkeypatch: pytest.MonkeyPatch, probe: BaseException | None
) -> None:
    inspected = []

    def check_process(pid: int, signal: int) -> None:
        inspected.append((pid, signal))
        if probe is not None:
            raise probe

    monkeypatch.setattr(os, "kill", check_process)
    assert process_alive({"pid": 12345, "platform": sys.platform}) == (
        not isinstance(probe, ProcessLookupError)
    )
    assert inspected == [(12345, 0)]
