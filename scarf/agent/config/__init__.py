"""Configuration shared by the four Scarf domain agents."""

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator

from ..types import AgentDataModel

__all__ = [
    "AgentRunConfig",
    "get_model_settings",
    "get_usage_limits",
    "CONFIG",
]


class Config:
    """Shared configuration for the four Scarf domain agents."""

    # BiologicalInterpretation
    _MAX_CLUSTERS: int = 20
    _MAX_CONDITIONS: int = 30
    _MAX_MARKERS: int = 25
    # CharacterizeFeatures
    _MAX_EXOGENOUS: int = 25
    _CONTEXT_LIMIT: int = 1200
    _AUTO_DOWNLOAD_SPECIES: frozenset[str] = frozenset({"homo_sapiens", "mus_musculus"})
    # DataEnrichment
    _MAX_FEATURE_QUERIES: int = 50
    # ParameterTuning
    _MAX_CANDIDATES_OFFERED: int = 25
    _CANDIDATE_ID: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,63}$")
    _RANDOM_SEED: int = 4444
    _PCA_RANDOM_SEED: int = 4466
    # CharacterizeCovariates
    _CATEGORICAL_MAX_LEVELS: int = 50
    _EMBEDDING_TOKENS: tuple[str, ...] = (
        "umap",
        "pca",
        "tsne",
        "scvi",
        "latent",
        "phate",
        "forceatlas",
        "diffmap",
        "diffusionmap",
        "diffusion",
    )
    _DOMAINS: frozenset[str] = frozenset(
        {"biological", "technical", "design", "ignore", "unknown"}
    )  # Only these domains reach the design table, so only they are worth collapsing.
    _ANALYSED: frozenset[str] = frozenset({"biological", "technical", "design"})
    _KINDS: frozenset[str] = frozenset({"categorical", "continuous"})
    _RESERVED_COLUMNS = frozenset({"I", "ids", "names"})

    _SHORT_EMBEDDING_PARTS: frozenset[str] = frozenset({"fa", "dm", "pc"})
    _INDEXED_NAME: re.Pattern[str] = re.compile(r"(?P<stem>.+?)[-_]?(?P<index>\d+)")
    _ONTOLOGY_SUFFIX: str = "_ontology_term_id"
    _SAMPLE_LEVELS: int = 8
    _ASSOCIATION_FLOOR: float = 0.1
    _DROP_REASONS: dict[str, str] = {
        "dropAssayStat": "Scarf assay statistic column",
        "dropProvenance": "analysis-linked column",
        "dropEmbedding": "embedding-style column",
        "dropConstant": "single-level column",
    }


class AgentRunConfig(AgentDataModel):
    """Bound one agent run without selecting a scientific workflow."""

    requestLimit: int = 6
    toolCallLimit: int = 4
    inputTokenLimit: int | None = None
    outputTokenLimit: int | None = 32768
    totalTokenLimit: int | None = None
    timeoutSeconds: float = 600.0
    retries: int = 2
    temperature: float = 0.0
    seed: int = 4444
    sequentialTools: bool = True
    thinkingOffProfile: Literal[
        "auto",
        "unified",
        "ollama",
        "chatTemplate",
        "thinkingBody",
        "reasoningBody",
    ] = "auto"
    extraModelSettings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("requestLimit", "toolCallLimit")
    @classmethod
    def validate_positive_limit(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("agent limits must be positive integers")
        return value

    @field_validator("inputTokenLimit", "outputTokenLimit", "totalTokenLimit")
    @classmethod
    def validate_token_limit(cls, value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or value < 1):
            raise ValueError("token limits must be positive integers or None")
        return value

    @field_validator("retries")
    @classmethod
    def validate_retries(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("retries must be a non-negative integer")
        return value

    @field_validator("timeoutSeconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeoutSeconds must be positive")
        return float(value)

    def with_limits(
        self,
        *,
        request_limit: int,
        tool_call_limit: int,
        output_token_limit: int,
        timeout_seconds: float,
    ) -> "AgentRunConfig":
        """Return a copy clamped to the supplied execution maxima."""
        values = self.model_dump()
        values.update(
            {
                "requestLimit": min(self.requestLimit, request_limit),
                "toolCallLimit": min(self.toolCallLimit, tool_call_limit),
                "outputTokenLimit": (
                    output_token_limit
                    if self.outputTokenLimit is None
                    else min(self.outputTokenLimit, output_token_limit)
                ),
                "timeoutSeconds": min(self.timeoutSeconds, timeout_seconds),
            }
        )
        return type(self).model_validate(values)

    @classmethod
    def get_example(cls) -> "AgentRunConfig":
        return cls(requestLimit=6, toolCallLimit=4, outputTokenLimit=2048)


def get_model_settings(
    config: AgentRunConfig | None = None,
    *,
    model: Any = None,
) -> Any:
    """Return settings that disable thinking across supported request shapes."""
    from pydantic_ai.settings import ModelSettings

    run_config = config or AgentRunConfig()
    extra_body = {
        "thinking": {"type": "disabled"},
        "reasoning_effort": "none",
        "chat_template_kwargs": {"thinking": False},  # for together-ai
        "reasoning": {"enabled": False},  # for openrouter
    }
    settings = ModelSettings(
        temperature=run_config.temperature,
        seed=run_config.seed,
        timeout=run_config.timeoutSeconds,
        parallel_tool_calls=not run_config.sequentialTools,
        thinking=False,
        extra_body=extra_body,
    )
    if run_config.outputTokenLimit is not None:
        settings["max_tokens"] = run_config.outputTokenLimit

    resolved: dict[str, Any] = dict(settings)
    base_url = str(getattr(model, "base_url", ""))
    hostname = (urlparse(base_url).hostname or "").casefold()
    if hostname == "baseten.co" or hostname.endswith(".baseten.co"):
        # Pydantic AI maps this provider-specific setting to the top-level
        # ``reasoning_effort`` field accepted by Baseten's OpenAI endpoint.
        resolved["openai_reasoning_effort"] = "none"
    resolved.update(run_config.extraModelSettings)
    return resolved


def get_usage_limits(config: AgentRunConfig | None = None) -> Any:
    """Translate the public run configuration into Pydantic-AI limits."""
    from pydantic_ai import UsageLimits

    run_config = config or AgentRunConfig()
    return UsageLimits(
        request_limit=run_config.requestLimit,
        tool_calls_limit=run_config.toolCallLimit,
        input_tokens_limit=run_config.inputTokenLimit,
        output_tokens_limit=run_config.outputTokenLimit,
        total_tokens_limit=run_config.totalTokenLimit,
    )


CONFIG = Config()
