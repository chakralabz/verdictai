"""Exception classes backed by centralized VerdictAI error dictionaries."""

from __future__ import annotations

from typing import Any

from .definitions import ErrorDefinition


class VerdictAIError(Exception):
    """Base exception for application errors with stable error codes.

    Args:
        definition: Error dictionary containing `error_code` and `error_message`.
        message: Optional explicit message override.
        **context: Values used to format the dictionary message template.

    Attributes:
        error_code: Stable machine-readable error code.
        error_message: Rendered human-readable message.
        context: Formatting context attached to the error.
    """

    def __init__(
        self,
        definition: ErrorDefinition | str,
        message: str | None = None,
        **context: Any,
    ) -> None:
        """Create a coded VerdictAI exception."""

        if isinstance(definition, str):
            self.error_code = "UNKNOWN"
            template = definition
        else:
            self.error_code = definition["error_code"]
            template = definition["error_message"]

        self.context = context
        self.error_message = message or _format_error_message(template, context)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        """Return the rendered error message prefixed with the stable code."""

        return f"[{self.error_code}] {self.error_message}"

    def to_dict(self) -> dict[str, Any]:
        """Return the exception as a serializable error payload.

        Returns:
            Error payload containing the stable code, rendered message, and any
            formatting context provided at construction time.
        """

        payload: dict[str, Any] = {
            "error_code": self.error_code,
            "error_message": self.error_message,
        }
        if self.context:
            payload["context"] = self.context
        return payload


class VerdictAIRuntimeError(VerdictAIError, RuntimeError):
    """Runtime error raised by VerdictAI components."""


class VerdictAIValueError(VerdictAIError, ValueError):
    """Value error raised when caller-provided data is invalid."""


class VerdictAIFileNotFoundError(VerdictAIError, FileNotFoundError):
    """File-not-found error raised for missing VerdictAI inputs."""


class DoclingParserError(VerdictAIRuntimeError):
    """Raised when Docling parsing fails or returns an invalid payload."""


class DoclingChunkerError(VerdictAIRuntimeError):
    """Raised when Docling chunking cannot be completed."""


class EmbeddingError(VerdictAIRuntimeError):
    """Raised when embedding generation cannot be completed."""


def _format_error_message(template: str, context: dict[str, Any]) -> str:
    """Format an error template while tolerating partial context.

    Args:
        template: Message template from an error definition dictionary.
        context: Values used by `str.format`.

    Returns:
        A rendered message. If required context is missing, returns the original
        template so error construction never hides the original failure.
    """

    try:
        return template.format(**context)
    except KeyError:
        return template
