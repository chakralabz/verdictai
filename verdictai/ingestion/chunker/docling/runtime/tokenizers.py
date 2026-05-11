"""Tokenizer builders for token-aware Docling chunkers."""

from __future__ import annotations

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from verdictai.ingestion.chunker.docling.options import DoclingTokenizerConfig
from verdictai.utils.errors import (
    CHUNKER_OPENAI_TOKENIZER_DEPENDENCY_MISSING,
    CHUNKER_UNSUPPORTED_TOKENIZER_PROVIDER,
    DoclingChunkerError,
)


def build_docling_tokenizer(
    *,
    config: DoclingTokenizerConfig,
) -> object:
    """Build a Docling tokenizer aligned with the selected embedding model.

    Args:
        config: Tokenizer configuration.

    Returns:
        A tokenizer instance compatible with Docling chunkers.

    Raises:
        DoclingChunkerError: If the tokenizer provider is unsupported or missing
            optional dependencies.
    """

    if config.provider == "huggingface":
        return HuggingFaceTokenizer.from_pretrained(
            model_name=config.model_name,
            max_tokens=config.max_tokens,
        )

    if config.provider == "openai":
        try:
            import tiktoken
            from docling_core.transforms.chunker.tokenizer.openai import (
                OpenAITokenizer,
            )
        except ImportError as exc:  # pragma: no cover - depends on env
            raise DoclingChunkerError(
                CHUNKER_OPENAI_TOKENIZER_DEPENDENCY_MISSING
            ) from exc

        return OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model(config.model_name),
            max_tokens=config.max_tokens,
        )

    raise DoclingChunkerError(
        CHUNKER_UNSUPPORTED_TOKENIZER_PROVIDER,
        provider=config.provider,
    )
