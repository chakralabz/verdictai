import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from verdictai.ingestion.parser.docling.constants import (
    DOCLING_ARTIFACTS_ENV_VAR,
    MODEL_CACHE_ENV_VARS,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class DoclingParserConfig:
    """Runtime configuration for the Docling-backed parser.

    Attributes:
        pipeline: Pipeline selection. `auto` chooses between `standard` and `vlm`
            based on the input file extension.
        do_ocr: Whether to run OCR.
        do_table_structure: Whether to extract table structure.
        ocr_engine: OCR backend identifier understood by Docling.
        vlm_model: VLM preset name (or alias) used when `pipeline="vlm"`.
        force_backend_text: When True, prefers backend-provided text extraction.
        generate_page_images: When True, requests page images from the pipeline.
        generate_picture_images: When True, requests picture crops from the pipeline.
        enable_remote_services: When True, allows remote services if Docling uses them.
        suppress_external_logs: When True, mutes noisy third-party runtime loggers.
        suppress_progress_bars: When True, disables Hugging Face and tqdm progress UI.
        suppress_runtime_warnings: When True, ignores known notebook-only warnings such
            as missing `ipywidgets` progress integrations.
        include_docling_document: Whether parse metadata should include the serialized
            Docling document needed by Docling-native chunkers. Disable this when a
            caller only needs `ParsedBlock` records and wants a smaller payload.
        stream_page_progress: When True, streams page by page progress indication to user.
        model_cache_dir: Optional directory to use for model caching. When unset,
            `resolve_model_cache_dir()` falls back to common environment variables.
        artifact_dir: Optional directory to use for Docling artifact storage. When
            unset, `resolve_artifact_dir()` falls back to `DOCLING_ARTIFACTS_PATH`
            and then to the resolved model cache directory.
    """

    pipeline: Literal["auto", "standard", "vlm"] = "auto"
    do_ocr: bool = True
    do_table_structure: bool = True
    ocr_engine: Literal[
        "easyocr",
        "rapidocr",
        "tesseract",
        "tesseract_cli",
        "ocrmac",
    ] = "rapidocr"
    vlm_model: str = "granite_docling"
    force_backend_text: bool = True
    generate_page_images: bool = False
    generate_picture_images: bool = False
    enable_remote_services: bool = False
    suppress_external_logs: bool = True
    suppress_progress_bars: bool = True
    suppress_runtime_warnings: bool = True
    include_docling_document: bool = True
    stream_page_progress: bool = False
    model_cache_dir: Path | None = None
    artifact_dir: Path | None = None

    def resolve_model_cache_dir(self) -> Path | None:
        """Resolve the model cache directory.

        The config value takes precedence. When it is unset, the first populated
        environment variable in `MODEL_CACHE_ENV_VARS` is used.

        Returns:
            The resolved model-cache directory, or None when no configuration is
            available.
        """

        if self.model_cache_dir is not None:
            return self.model_cache_dir.expanduser().resolve()

        # 1. Prefer env vars set by container/runtime over hardcoding a path.
        for env_var in MODEL_CACHE_ENV_VARS:
            raw_value = os.getenv(env_var)
            if raw_value:
                return Path(raw_value).expanduser().resolve()

        return None

    def resolve_artifact_dir(self) -> Path | None:
        """Resolve the Docling artifacts directory.

        The explicit config value takes precedence. When it is unset, the parser
        uses `DOCLING_ARTIFACTS_PATH` from the runtime environment. As a final
        fallback, it reuses the resolved model-cache directory so local runtimes
        can still share one pre-populated models location.

        Returns:
            The resolved artifacts directory, or None when no configuration is
            available.
        """

        if self.artifact_dir is not None:
            return self.artifact_dir.expanduser().resolve()

        artifacts_env_value = os.getenv(DOCLING_ARTIFACTS_ENV_VAR)
        if artifacts_env_value:
            return Path(artifacts_env_value).expanduser().resolve()

        return self.resolve_model_cache_dir()
