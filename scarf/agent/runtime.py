"""Runtime checks and local env loading for LLM endpoints."""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env(path: Path | None = None) -> Path | None:
    """Load scarf/agent/.env into os.environ without overriding existing vars.

    Returns the path loaded, or None if no file was found.
    """
    env_path = path or _ENV_PATH
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


def _unreachable_message(baseUrl: str, model: str) -> str:
    return (
        f"OpenAI-compatible endpoint unreachable at {baseUrl!r} "
        f"(requested model {model!r})."
    )


def _missing_model_message(baseUrl: str, model: str, available: set[str]) -> str:
    listed = ", ".join(sorted(available)) if available else "(none)"
    return f"Model {model!r} not found at {baseUrl!r}. Available models: {listed}."


def _missing_config_message() -> str:
    example = _ENV_PATH.with_name(".env.example")
    return (
        "baseUrl and model are required. Pass them as arguments or set "
        "OLLAMA_BASE_URL and OLLAMA_MODEL in the environment or in "
        f"{_ENV_PATH} (see {example})."
    )


def _request_json(
    url: str,
    *,
    timeout: float,
    api_key: str | None,
) -> Any:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _listed_model_ids(payload: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(payload, dict):
        return ids
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.add(item["id"])
    models = payload.get("models")
    if isinstance(models, list):
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str):
                ids.add(name)
    return ids


def check_runtime(
    *,
    baseUrl: str | None = None,
    model: str | None = None,
    timeout: float = 5.0,
) -> None:
    """Fail fast if the OpenAI-compatible endpoint or model is unavailable.

    Loads `scarf/agent/.env` first (without overriding existing env vars).
    Uses `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_API_KEY` when args are omitted.
    """
    load_env()
    resolved_base = baseUrl or os.environ.get("OLLAMA_BASE_URL")
    resolved_model = model or os.environ.get("OLLAMA_MODEL")
    if not resolved_base or not resolved_model:
        raise ValueError(_missing_config_message())

    api_key = os.environ.get("OLLAMA_API_KEY") or None
    models_url = f"{resolved_base.rstrip('/')}/models"
    try:
        payload = _request_json(models_url, timeout=timeout, api_key=api_key)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(_unreachable_message(resolved_base, resolved_model)) from exc

    available = _listed_model_ids(payload)
    if resolved_model not in available:
        raise RuntimeError(
            _missing_model_message(resolved_base, resolved_model, available)
        )
