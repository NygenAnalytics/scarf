"""Runtime reachability tests for scarf.agent.check_runtime."""

import json
import os
from typing import Any

import pytest

from scarf.agent import check_runtime, load_env
from scarf.agent import runtime as runtime_module


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_load_env_reads_file_without_overriding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OLLAMA_BASE_URL=https://from-file.example/v1\n"
        "OLLAMA_MODEL=from-file\n"
        "OLLAMA_API_KEY=from-file-key\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "already-set")

    loaded = load_env(env_file)
    assert loaded == env_file
    assert os.environ["OLLAMA_BASE_URL"] == "https://from-file.example/v1"
    assert os.environ["OLLAMA_MODEL"] == "from-file"
    assert os.environ["OLLAMA_API_KEY"] == "already-set"


def test_check_runtime_ok_when_model_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float = 0.0):
        return _FakeResponse({"data": [{"id": "test-model"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    check_runtime(baseUrl="http://example.test/v1", model="test-model")


def test_check_runtime_sends_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake_urlopen(request: Any, timeout: float = 0.0):
        seen["authorization"] = request.get_header("Authorization")
        return _FakeResponse({"data": [{"id": "test-model"}]})

    monkeypatch.setenv("OLLAMA_API_KEY", "secret-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    check_runtime(baseUrl="http://example.test/v1", model="test-model")
    assert seen["authorization"] == "Bearer secret-key"


def test_check_runtime_uses_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float = 0.0):
        return _FakeResponse({"data": [{"id": "env-model"}]})

    monkeypatch.setattr(runtime_module, "load_env", lambda path=None: None)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    check_runtime()


def test_check_runtime_fails_when_endpoint_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    def fake_urlopen(request: object, timeout: float = 0.0):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="endpoint unreachable"):
        check_runtime(baseUrl="http://example.test/v1", model="test-model")


def test_check_runtime_fails_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float = 0.0):
        return _FakeResponse({"data": [{"id": "other-model"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Model 'test-model' not found"):
        check_runtime(baseUrl="http://example.test/v1", model="test-model")


@pytest.mark.integration
def test_check_runtime_live_ollama_smoke() -> None:
    import urllib.error
    import urllib.request

    load_env()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
    try:
        urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=2.0)
    except (urllib.error.URLError, TimeoutError):
        pytest.skip("OpenAI-compatible endpoint is not reachable")

    try:
        check_runtime(baseUrl=base_url, model=model)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider

    from scarf.agent import EvidenceItem, decide

    decision = decide(
        model=OllamaModel(
            model,
            provider=OllamaProvider(base_url=base_url),
        ),
        question="Which matrix looks like raw counts?",
        evidence=[
            EvidenceItem(
                id="matrix:X",
                label="X",
                summary="float, mostly non-integer",
            ),
            EvidenceItem(
                id="matrix:raw/X",
                label="raw/X",
                summary="integer-like",
            ),
        ],
    )
    assert decision.selectedId in {"matrix:X", "matrix:raw/X"}
    assert decision.selectedId in decision.evidenceIds
