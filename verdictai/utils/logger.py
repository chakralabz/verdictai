"""Application logging helpers.

The project uses one shared logging setup so modules can request a logger
without duplicating formatter or handler configuration. Development logs are
human-readable, while production logs are structured JSON for ingestion by log
aggregators.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Final

_LOGGER_NAMESPACE: Final[str] = "verdictai"
_PRODUCTION_ENVIRONMENTS: Final[frozenset[str]] = frozenset(
    {"prod", "production", "staging"}
)


class _JsonFormatter(logging.Formatter):
    """Render log records as compact JSON objects.

    Notes:
        - The output schema is intentionally small and stable so downstream log
          pipelines (e.g., ingestion, alerting) can treat it as structured data.
        - Exceptions are rendered using the base formatter's exception
          formatting.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record into JSON.

        Args:
            record: Log record to format.

        Returns:
            A compact JSON object as a single line string.
        """
        message = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            message["exception"] = self.formatException(record.exc_info)

        return json.dumps(message, ensure_ascii=False)


def _is_production_environment() -> bool:
    """Return whether the process should emit production logs.

    The environment is determined by the `VERDICTAI_ENV` or `ENV` environment
    variable (falling back to `development`).

    Returns:
        True when the current environment matches a production-like value.
    """

    environment = os.getenv("VERDICTAI_ENV") or os.getenv("ENV") or "development"
    return environment.strip().lower() in _PRODUCTION_ENVIRONMENTS


def configure_logging(*, force: bool = False) -> None:
    """Configure the shared project logger once per process.

    Args:
        force: When True, clears and re-installs handlers even if the project
            logger has already been configured.
    """

    project_logger = logging.getLogger(_LOGGER_NAMESPACE)
    if project_logger.handlers and not force:
        return

    # 1. Create a single stream handler shared by the entire application.
    handler = logging.StreamHandler()
    if _is_production_environment():
        # 2.A Production defaults: JSON logs + INFO level for ingestion systems.
        formatter: logging.Formatter = _JsonFormatter()
        level = logging.INFO
    else:
        # 2.B Development defaults: human-readable logs + DEBUG level.
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        level = logging.DEBUG

    handler.setFormatter(formatter)
    project_logger.handlers.clear()
    project_logger.addHandler(handler)
    project_logger.setLevel(level)
    project_logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured application logger.

    Args:
        name: Logger name. When omitted, returns the project root logger.
            When provided, it is namespaced under `verdictai.` unless it is
            already within the `verdictai` namespace.

    Returns:
        A `logging.Logger` instance with handlers/formatters configured.
    """

    configure_logging()
    if not name:
        return logging.getLogger(_LOGGER_NAMESPACE)

    # 1. Preserve already-namespaced loggers to avoid double-prefixing.
    if name == _LOGGER_NAMESPACE or name.startswith(f"{_LOGGER_NAMESPACE}."):
        return logging.getLogger(name)

    # 2. Namespace everything else under the project root.
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{name}")
