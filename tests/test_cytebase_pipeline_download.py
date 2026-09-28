"""Offline tests for downloading one CELLxGENE H5AD with aria2c."""

import hashlib
import io
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from scarf.cytebase.pipeline import download
from tests.fixtures_cytebase import SOURCE_URL

pytestmark = pytest.mark.usefixtures("cytebase_offline")

ARIA2C = "/opt/aria2/bin/aria2c"
OBJECT_URL = "https://objects.example.org/bucket/source.h5ad"
PAYLOAD = b"0123456789"
SIZE = len(PAYLOAD)
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
ETAG = '"v1"'
RANGE_HEADERS = {"Content-Range": f"bytes 0-0/{SIZE}", "ETag": ETAG}


def _option(command: list[str], name: str) -> str:
    (value,) = [arg.split("=", 1)[1] for arg in command if arg.startswith(f"--{name}=")]
    return value


def _partial(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part")


def _leftovers(destination: Path) -> list[str]:
    return sorted(path.name for path in destination.parent.iterdir())


class Recorder:
    """A progress callback that keeps every event."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, stage: str, **values: Any) -> None:
        self.events.append((stage, values))

    def stages(self) -> list[tuple[str, str | None, int | None]]:
        return [
            (stage, values.get("phase"), values.get("completed"))
            for stage, values in self.events
        ]


class FakeProcess:
    """One aria2c run that writes its payload on start and replays its output."""

    def __init__(
        self, aria2: "FakeAria2", command: list[str], options: dict[str, Any]
    ) -> None:
        self.aria2 = aria2
        self.command = command
        self.options = options
        self.calls: list[tuple[Any, ...]] = []
        self.returncode: int | None = None
        partial = Path(_option(command, "dir")) / _option(command, "out")
        partial.write_bytes(aria2.payload)
        if aria2.control:
            partial.with_name(f"{partial.name}.aria2").write_bytes(b"control")
        self.stdout = io.StringIO("".join(f"{line}\n" for line in aria2.lines))

    def poll(self) -> int | None:
        self.calls.append(("poll",))
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        if timeout is not None and self.aria2.ignore_terminate:
            raise subprocess.TimeoutExpired(self.command, timeout)
        if self.returncode is None:
            self.returncode = self.aria2.returncode
        return self.returncode

    def terminate(self) -> None:
        self.calls.append(("terminate",))
        if not self.aria2.ignore_terminate:
            self.returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.calls.append(("kill",))
        self.returncode = -signal.SIGKILL


class FakeAria2:
    """Stands in for ``shutil.which`` and ``subprocess.Popen`` for aria2c."""

    def __init__(self) -> None:
        self.installed = True
        self.lines: list[str] = []
        self.payload = PAYLOAD
        self.control = False
        self.returncode = 0
        self.ignore_terminate = False
        self.lookups: list[str] = []
        self.processes: list[FakeProcess] = []

    def which(self, name: str) -> str | None:
        self.lookups.append(name)
        return ARIA2C if self.installed else None

    def popen(self, command: list[str], **options: Any) -> FakeProcess:
        process = FakeProcess(self, command, options)
        self.processes.append(process)
        return process


class FakeProbe:
    """Replaces ``_probe``; answers are (size, ETag) pairs or callables returning one."""

    def __init__(self) -> None:
        self.answers: list[Any] = [(SIZE, ETAG), (SIZE, ETAG)]
        self.clients: list[httpx.Client] = []

    def __call__(self, client: httpx.Client, url: str) -> tuple[int, str]:
        assert url == SOURCE_URL
        self.clients.append(client)
        answer = self.answers.pop(0)
        return answer() if callable(answer) else answer


@pytest.fixture
def aria2(monkeypatch) -> FakeAria2:
    fake = FakeAria2()
    monkeypatch.setattr(download.shutil, "which", fake.which)
    monkeypatch.setattr(download.subprocess, "Popen", fake.popen)
    return fake


@pytest.fixture
def probe(monkeypatch) -> FakeProbe:
    fake = FakeProbe()
    monkeypatch.setattr(download, "_probe", fake)
    return fake


@pytest.fixture
def destination(tmp_path) -> Path:
    """A download target whose directory ``download_h5ad`` creates."""
    return tmp_path / "work" / "source.h5ad"


def _ranged(
    status: int = 206, headers: dict[str, str] | None = None, body: bytes = b"x"
) -> httpx.Response:
    """A streamed reply to ``Range: bytes=0-0``; the defaults form a valid probe."""
    return httpx.Response(
        status,
        headers=RANGE_HEADERS if headers is None else headers,
        stream=httpx.ByteStream(body),
    )


def _probe_once(response: httpx.Response) -> tuple[int, str]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=0-0"
        return response

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return download._probe(client, SOURCE_URL)


def _serve(monkeypatch, handler) -> list[dict[str, Any]]:
    """Send the download's HTTP client to ``handler``; return the client options."""
    real_client = httpx.Client
    options: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> httpx.Client:
        options.append(kwargs)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(download.httpx, "Client", client)
    return options


def test_download_connections_default_to_two():
    assert download.download_connections() == 2


@pytest.mark.parametrize(("value", "expected"), [("1", 1), ("4", 4), (" 3 ", 3)])
def test_download_connections_read_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("CYTEBASE_DOWNLOAD_CONNECTIONS", value)
    assert download.download_connections() == expected


@pytest.mark.parametrize(
    ("value", "unparsable"),
    [
        ("0", False),
        ("5", False),
        ("-1", False),
        ("two", True),
        ("", True),
        ("2.5", True),
    ],
)
def test_download_connections_reject_other_values(monkeypatch, value, unparsable):
    monkeypatch.setenv("CYTEBASE_DOWNLOAD_CONNECTIONS", value)
    with pytest.raises(ValueError, match="must be 1 to 4") as excinfo:
        download.download_connections()
    assert isinstance(excinfo.value.__cause__, ValueError) is unparsable


@pytest.mark.parametrize(
    "headers",
    [RANGE_HEADERS, RANGE_HEADERS | {"Content-Encoding": "Identity"}],
    ids=["implicit-identity", "explicit-identity"],
)
def test_probe_reads_one_byte_for_the_size_and_strong_etag(headers):
    assert _probe_once(_ranged(headers=headers)) == (SIZE, ETAG)


# Each case edits the valid probe reply; a None value removes that header.
PROBE_REJECTIONS = {
    "range-ignored": (200, {"Content-Range": None}, PAYLOAD, "byte range"),
    "not-partial-content": (200, {}, b"x", "byte range"),
    "no-content-range": (206, {"Content-Range": None}, b"x", "byte range"),
    "empty-source": (206, {"Content-Range": "bytes 0-0/0"}, b"x", "byte range"),
    "unknown-size": (206, {"Content-Range": "bytes 0-0/*"}, b"x", "byte range"),
    "other-range": (206, {"Content-Range": "bytes 0-1/10"}, b"x", "byte range"),
    "no-etag": (206, {"ETag": None}, b"x", "strong ETag"),
    "weak-etag": (206, {"ETag": 'W/"v1"'}, b"x", "strong ETag"),
    "unquoted-etag": (206, {"ETag": "v1"}, b"x", "strong ETag"),
    "empty-etag": (206, {"ETag": '""'}, b"x", "strong ETag"),
    "compressed": (206, {"Content-Encoding": "gzip"}, b"x", "uncompressed"),
    "extra-bytes": (206, {}, b"xy", "more bytes than requested"),
}


@pytest.mark.parametrize(
    ("status", "changes", "body", "message"),
    list(PROBE_REJECTIONS.values()),
    ids=list(PROBE_REJECTIONS),
)
def test_probe_rejects_replies_that_cannot_anchor_a_download(
    status, changes, body, message
):
    headers = {
        name: value
        for name, value in (RANGE_HEADERS | changes).items()
        if value is not None
    }
    with pytest.raises(ValueError, match=message):
        _probe_once(_ranged(status, headers, body))


def test_probe_reports_a_missing_byte_as_a_retryable_transport_error():
    with pytest.raises(httpx.TransportError, match="Incomplete"):
        _probe_once(_ranged(body=b""))


def test_probe_raises_http_errors_for_retry_to_classify():
    with pytest.raises(httpx.HTTPStatusError, match="404"):
        _probe_once(_ranged(404, {}, b""))


def test_download_requires_aria2c(aria2, destination):
    aria2.installed = False
    with pytest.raises(RuntimeError, match="aria2c is required"):
        download.download_h5ad(SOURCE_URL, destination)
    assert aria2.lookups == ["aria2c"]
    assert not destination.parent.exists()


def test_download_checks_the_connection_limit_before_any_work(
    aria2, probe, monkeypatch, destination
):
    monkeypatch.setenv("CYTEBASE_DOWNLOAD_CONNECTIONS", "8")
    with pytest.raises(ValueError, match="must be 1 to 4"):
        download.download_h5ad(SOURCE_URL, destination)
    assert not destination.parent.exists()
    assert probe.clients == []


@pytest.mark.parametrize("suffix", ["", ".part", ".part.aria2"])
def test_download_refuses_existing_destination_or_resume_files(
    aria2, probe, destination, suffix
):
    existing = destination.with_name(f"{destination.name}{suffix}")
    existing.parent.mkdir()
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="must be absent"):
        download.download_h5ad(SOURCE_URL, destination)
    assert existing.read_bytes() == b"keep"
    assert _leftovers(destination) == [existing.name]
    assert probe.clients == []
    assert aria2.processes == []


def test_download_treats_a_dangling_partial_link_as_existing(aria2, probe, destination):
    partial = _partial(destination)
    partial.parent.mkdir()
    partial.symlink_to(destination.parent / "gone")
    with pytest.raises(FileExistsError, match="must be absent"):
        download.download_h5ad(SOURCE_URL, destination)
    assert partial.is_symlink()
    assert probe.clients == []


def test_download_hashes_and_links_the_finished_file(aria2, probe, destination):
    aria2.lines = [
        "09/25 12:00:00 [NOTICE] Downloading 1 item(s)",
        "[#2089b0 0B/0B CN:1 DL:0B]",
        "[#2089b0 4B/10B(40%) CN:2 DL:100B]",
        "",
        "[#2089b0 10B/10B(100%) CN:1]",
        "09/25 12:00:01 [NOTICE] Download complete",
    ]
    progress = Recorder()
    timings = {"queuedSeconds": 1.5}

    result = download.download_h5ad(
        SOURCE_URL, destination, SIZE, progress=progress, timings=timings
    )

    assert result == (SIZE, SHA256)
    assert destination.read_bytes() == PAYLOAD
    assert destination.stat().st_nlink == 1
    assert _leftovers(destination) == ["source.h5ad"]
    assert timings["queuedSeconds"] == 1.5
    assert timings["downloadTransferSeconds"] >= 0
    assert timings["downloadHashSeconds"] >= 0
    first, second = probe.clients
    assert first is second
    assert first.is_closed
    # The zero-byte line precedes the source headers and is not reported.
    assert progress.stages() == [
        ("downloading", "probe", 0),
        ("downloading", "transfer", 0),
        ("downloading", "transfer", 4),
        ("downloading", "transfer", SIZE),
        ("downloading", "source_check", SIZE),
        ("hashing_source", "checksum", 0),
        ("hashing_source", "checksum", SIZE),
        ("downloaded", "complete", SIZE),
    ]
    assert progress.events[0][1]["total"] == SIZE
    assert progress.events[1][1]["message"] == (
        "Starting aria2 with up to 2 source connections"
    )
    assert progress.events[2] == (
        "downloading",
        {
            "completed": 4,
            "total": SIZE,
            "downloadedBytes": 4,
            "activeConnections": 2,
            "downloadSpeedBytesPerSecond": 100,
            "unit": "bytes",
            "phase": "transfer",
            "message": "aria2 downloading the local H5AD",
        },
    )
    assert progress.events[3][1]["activeConnections"] == 1
    assert progress.events[3][1]["downloadSpeedBytesPerSecond"] == 0
    assert progress.events[-1] == (
        "downloaded",
        {
            "completed": SIZE,
            "total": SIZE,
            "downloadedBytes": SIZE,
            "activeConnections": 0,
            "unit": "bytes",
            "phase": "complete",
            "message": "Local H5AD downloaded, size checked and SHA-256 recorded",
        },
    )


def test_download_runs_one_aria2c_bound_to_the_probed_etag(
    aria2, probe, monkeypatch, tmp_path
):
    monkeypatch.setenv("CYTEBASE_DOWNLOAD_CONNECTIONS", "3")
    monkeypatch.setenv("SCARF_TEST_MARKER", "inherited")
    monkeypatch.chdir(tmp_path)

    assert download.download_h5ad(SOURCE_URL, Path("work/source.h5ad")) == (
        SIZE,
        SHA256,
    )

    (process,) = aria2.processes
    command = process.command
    assert command[0] == ARIA2C
    assert command[-2:] == ["--", SOURCE_URL]
    assert {
        "--split=3",
        "--max-connection-per-server=3",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--http-accept-gzip=false",
        "--header=Accept-Encoding: identity",
        '--header=If-Match: "v1"',
        f"--stop-with-process={os.getpid()}",
        f"--dir={(tmp_path / 'work').resolve()}",
        "--out=source.h5ad.part",
    } <= set(command)
    env = process.options.pop("env")
    assert env["LC_ALL"] == "C"
    assert env["SCARF_TEST_MARKER"] == "inherited"
    assert process.options == {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    assert process.calls == [("wait", None), ("poll",)]
    assert process.stdout.closed
    assert (tmp_path / "work" / "source.h5ad").read_bytes() == PAYLOAD


def test_download_reports_hashing_each_second_and_at_the_end(
    aria2, probe, monkeypatch, destination
):
    monkeypatch.setattr(download, "_BUFFER_BYTES", 3)
    # Transfer start and end, hash start, one read per 3-byte block, hash end.
    clock = [100.0, 104.0, 110.0, 110.5, 111.0, 111.5, 111.6, 113.25]
    monkeypatch.setattr(download, "monotonic", lambda: clock.pop(0))
    progress = Recorder()
    timings: dict[str, float] = {}

    assert download.download_h5ad(
        SOURCE_URL, destination, progress=progress, timings=timings
    ) == (SIZE, SHA256)

    hashed = [
        values["completed"]
        for stage, values in progress.events
        if stage == "hashing_source"
    ]
    assert hashed == [0, 6, SIZE]
    assert timings == {"downloadTransferSeconds": 4.0, "downloadHashSeconds": 3.25}
    assert progress.events[0] == (
        "downloading",
        {
            "completed": 0,
            "total": None,
            "unit": "bytes",
            "phase": "probe",
            "message": "Checking source identity and byte-range support",
        },
    )


def test_download_rejects_an_unexpected_size_before_aria2(aria2, probe, destination):
    with pytest.raises(ValueError, match=f"is {SIZE} bytes, expected {SIZE + 1}"):
        download.download_h5ad(SOURCE_URL, destination, SIZE + 1)
    assert _leftovers(destination) == []
    assert aria2.processes == []


def test_download_leaves_a_partial_claimed_by_another_process(
    aria2, probe, destination
):
    partial = _partial(destination)

    def claim_partial() -> tuple[int, str]:
        partial.write_bytes(b"other")
        return SIZE, ETAG

    probe.answers = [claim_partial]
    with pytest.raises(FileExistsError, match="File exists"):
        download.download_h5ad(SOURCE_URL, destination)
    assert partial.read_bytes() == b"other"
    assert aria2.processes == []


def test_download_retries_a_busy_source_with_the_real_probe(
    aria2, monkeypatch, recorded_sleeps, destination
):
    replies = [_ranged(503, {}, b""), _ranged(), _ranged()]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return replies.pop(0)

    options = _serve(monkeypatch, handler)
    progress = Recorder()

    assert download.download_h5ad(SOURCE_URL, destination, SIZE, progress=progress) == (
        SIZE,
        SHA256,
    )

    assert recorded_sleeps == [2.0]
    assert progress.events[1] == (
        "retrying_transfer",
        {"message": "Retry 1/3 after 2 seconds"},
    )
    assert [
        (str(request.url), request.headers["Range"], request.headers["Accept-Encoding"])
        for request in requests
    ] == [(SOURCE_URL, "bytes=0-0", "identity")] * 3
    assert options == [
        {
            "timeout": httpx.Timeout(60, connect=30),
            "follow_redirects": True,
            "headers": {"Accept-Encoding": "identity"},
        }
    ]
    assert '--header=If-Match: "v1"' in aria2.processes[0].command


def test_download_probe_follows_source_redirects(aria2, monkeypatch, destination):
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["Range"]))
        if request.url == SOURCE_URL:
            return httpx.Response(302, headers={"Location": OBJECT_URL})
        return _ranged()

    _serve(monkeypatch, handler)

    assert download.download_h5ad(SOURCE_URL, destination) == (SIZE, SHA256)
    assert seen == [(SOURCE_URL, "bytes=0-0"), (OBJECT_URL, "bytes=0-0")] * 2
    # aria2 resolves the redirect itself, so it keeps the stable source URL.
    assert aria2.processes[0].command[-1] == SOURCE_URL


def test_download_refuses_a_source_without_a_strong_etag(
    aria2, monkeypatch, destination
):
    weak = RANGE_HEADERS | {"ETag": 'W/"v1"'}
    _serve(monkeypatch, lambda request: _ranged(headers=weak))
    with pytest.raises(ValueError, match="strong ETag"):
        download.download_h5ad(SOURCE_URL, destination)
    assert aria2.processes == []
    assert _leftovers(destination) == []


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("[#2089b0 4B/11B(36%) CN:2 DL:5B]", id="other-total"),
        pytest.param("[#2089b0 11B/10B CN:2]", id="beyond-total"),
        pytest.param("[#2089b0 5B/0B CN:1]", id="bytes-without-total"),
    ],
)
def test_download_terminates_aria2_reporting_another_size(
    aria2, probe, destination, line
):
    aria2.lines = ["[#2089b0 4B/10B(40%) CN:2 DL:100B]", line]
    timings: dict[str, float] = {}
    with pytest.raises(ValueError, match="unexpected source size"):
        download.download_h5ad(SOURCE_URL, destination, timings=timings)
    (process,) = aria2.processes
    assert process.calls == [("poll",), ("terminate",), ("wait", 10)]
    assert process.stdout.closed
    assert _leftovers(destination) == []
    assert list(timings) == ["downloadTransferSeconds"]
    assert len(probe.clients) == 1


def test_download_kills_aria2_that_ignores_terminate(aria2, probe, destination):
    aria2.lines = ["[#2089b0 4B/11B CN:2]"]
    aria2.ignore_terminate = True
    with pytest.raises(ValueError, match="unexpected source size"):
        download.download_h5ad(SOURCE_URL, destination)
    (process,) = aria2.processes
    assert process.calls == [
        ("poll",),
        ("terminate",),
        ("wait", 10),
        ("kill",),
        ("wait", None),
    ]
    assert process.stdout.closed
    assert _leftovers(destination) == []


def test_download_reports_aria2_failures_with_its_output(aria2, probe, destination):
    aria2.returncode = 22
    aria2.lines = [
        "[NOTICE] Downloading 1 item(s)",
        "[#2089b0 4B/10B(40%) CN:2 DL:100B]",
        "   ",
        "  errorCode=22 Precondition failed  ",
    ]
    with pytest.raises(RuntimeError, match="exit code 22") as excinfo:
        download.download_h5ad(SOURCE_URL, destination)
    assert str(excinfo.value) == (
        "aria2c failed with exit code 22: "
        "[NOTICE] Downloading 1 item(s) | errorCode=22 Precondition failed"
    )
    (process,) = aria2.processes
    assert process.calls == [("wait", None), ("poll",)]
    assert _leftovers(destination) == []
    assert len(probe.clients) == 1


def test_aria2_failure_message_keeps_the_last_twelve_lines(aria2, probe, destination):
    aria2.returncode = 1
    aria2.lines = [f"line {index}" for index in range(13)] + ["x" * 600]
    with pytest.raises(RuntimeError, match="exit code 1") as excinfo:
        download.download_h5ad(SOURCE_URL, destination)
    kept = [f"line {index}" for index in range(2, 13)] + ["x" * 500]
    assert str(excinfo.value) == "aria2c failed with exit code 1: " + " | ".join(kept)


@pytest.mark.parametrize(
    ("payload", "control"),
    [
        pytest.param(PAYLOAD, True, id="control-file-left"),
        pytest.param(PAYLOAD[:-1], False, id="short-file"),
    ],
)
def test_download_rejects_an_unfinished_aria2_result(
    aria2, probe, destination, payload, control
):
    aria2.payload = payload
    aria2.control = control
    timings: dict[str, float] = {}
    with pytest.raises(ValueError, match="did not finish the expected H5AD"):
        download.download_h5ad(SOURCE_URL, destination, timings=timings)
    assert _leftovers(destination) == []
    assert len(probe.clients) == 1
    assert list(timings) == ["downloadTransferSeconds"]


@pytest.mark.parametrize(
    "later",
    [
        pytest.param((SIZE, '"v2"'), id="new-etag"),
        pytest.param((SIZE + 1, ETAG), id="new-size"),
    ],
)
def test_download_rejects_a_source_that_changed_during_the_transfer(
    aria2, probe, destination, later
):
    probe.answers = [(SIZE, ETAG), later]
    progress = Recorder()
    timings: dict[str, float] = {}
    with pytest.raises(ValueError, match="changed during download"):
        download.download_h5ad(
            SOURCE_URL, destination, progress=progress, timings=timings
        )
    assert progress.stages()[-1] == ("downloading", "source_check", SIZE)
    assert list(timings) == ["downloadTransferSeconds"]
    assert _leftovers(destination) == []


def test_download_rejects_a_partial_that_changes_before_hashing(
    aria2, probe, destination
):
    def append_byte() -> tuple[int, str]:
        with _partial(destination).open("ab") as partial:
            partial.write(b"!")
        return SIZE, ETAG

    probe.answers = [(SIZE, ETAG), append_byte]
    timings: dict[str, float] = {}
    with pytest.raises(ValueError, match="size changed while hashing"):
        download.download_h5ad(SOURCE_URL, destination, timings=timings)
    assert set(timings) == {"downloadTransferSeconds", "downloadHashSeconds"}
    assert _leftovers(destination) == []


def test_download_never_replaces_a_destination_created_meanwhile(
    aria2, probe, destination
):
    def claim_destination() -> tuple[int, str]:
        destination.write_bytes(b"other")
        return SIZE, ETAG

    probe.answers = [(SIZE, ETAG), claim_destination]
    with pytest.raises(FileExistsError, match="File exists"):
        download.download_h5ad(SOURCE_URL, destination)
    assert destination.read_bytes() == b"other"
    assert _leftovers(destination) == ["source.h5ad"]
