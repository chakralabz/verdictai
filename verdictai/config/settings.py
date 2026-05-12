"""Singleton settings loaded from YAML and environment variables."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from verdictai.ingestion.chunker.docling.options import DoclingChunkerConfig
from verdictai.ingestion.embeddings.config import EmbeddingConfig
from verdictai.ingestion.parser.docling.docling_parser_config import (
    DoclingParserConfig,
)
from verdictai.ingestion.store.config import StoreSettings

DEFAULT_CONFIG_FILE = "config.yaml"
CONFIG_FILE_ENV_VAR = "VERDICTAI_CONFIG_FILE"
_ENV_REFERENCE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ParserSettings(BaseSettings):
    """Parser configuration namespace.

    Attributes:
        docling: Runtime settings for the Docling parser.
    """

    model_config = SettingsConfigDict(extra="forbid")

    docling: DoclingParserConfig = Field(default_factory=DoclingParserConfig)


class ChunkerSettings(BaseSettings):
    """Chunker configuration namespace.

    Attributes:
        docling: Runtime settings for Docling-backed chunkers.
    """

    model_config = SettingsConfigDict(extra="forbid")

    docling: DoclingChunkerConfig = Field(default_factory=DoclingChunkerConfig)


class VerdictAISettings(BaseSettings):
    """Root VerdictAI application settings.

    Attributes:
        parser: Parser package configuration.
        chunker: Chunker package configuration.
        embeddings: Embedding provider configuration.
        store: Store package configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VERDICTAI_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    parser: ParserSettings = Field(default_factory=ParserSettings)
    chunker: ChunkerSettings = Field(default_factory=ChunkerSettings)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    store: StoreSettings = Field(default_factory=StoreSettings)


@lru_cache(maxsize=1)
def get_settings(config_file: str | Path | None = None) -> VerdictAISettings:
    """Return the process-wide VerdictAI settings singleton.

    Args:
        config_file: Optional YAML path. When omitted, `VERDICTAI_CONFIG_FILE`
            is consulted before falling back to `config.yaml` in the current
            working directory.

    Returns:
        The validated application settings instance.

    Notes:
        `.env` is loaded before YAML interpolation so values such as
        `${OPENAI_API_KEY}` can be referenced inside `config.yaml`. The singleton
        is cached because application configuration should be stable for the
        process lifetime.
    """

    load_dotenv(override=False)
    yaml_data = _load_yaml_config(_resolve_config_file(config_file))
    return VerdictAISettings(**yaml_data)


def reload_settings(config_file: str | Path | None = None) -> VerdictAISettings:
    """Clear the singleton cache and load settings again.

    Args:
        config_file: Optional YAML path passed to `get_settings`.

    Returns:
        A freshly loaded settings instance.
    """

    get_settings.cache_clear()
    return get_settings(config_file)


def _resolve_config_file(config_file: str | Path | None) -> Path | None:
    """Resolve the YAML settings path when one exists."""

    raw_path = config_file or os.getenv(CONFIG_FILE_ENV_VAR) or DEFAULT_CONFIG_FILE
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.exists():
        return path
    return None


def _load_yaml_config(config_file: Path | None) -> dict[str, Any]:
    """Load and expand a YAML settings file.

    Args:
        config_file: Resolved YAML path, or None when no file is configured.

    Returns:
        Expanded mapping suitable for `VerdictAISettings` validation.

    Raises:
        ValueError: If the YAML root is not a mapping.
    """

    if config_file is None:
        return {}

    raw_data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_data, dict):
        raise ValueError(f"VerdictAI config root must be a mapping: {config_file}")
    return _expand_env_references(raw_data)


def _expand_env_references(value: Any) -> Any:
    """Recursively expand `${ENV_VAR}` references in YAML values."""

    if isinstance(value, dict):
        return {key: _expand_env_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_references(item) for item in value]
    if isinstance(value, str):
        return _ENV_REFERENCE_PATTERN.sub(_replace_env_reference, value)
    return value


def _replace_env_reference(match: re.Match[str]) -> str:
    """Return the environment value for one regex match."""

    env_var = match.group(1)
    fallback = match.group(2)
    value = os.getenv(env_var)
    if value is not None:
        return value
    if fallback is not None:
        return fallback
    return match.group(0)
