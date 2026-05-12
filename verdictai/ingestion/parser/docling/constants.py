"""Constants for the Docling parser package."""

from __future__ import annotations

from typing import Final

DOCLING_ARTIFACTS_ENV_VAR: Final[str] = "DOCLING_ARTIFACTS_PATH"
MODEL_CACHE_ENV_VARS: Final[tuple[str, ...]] = (
    "DOCLING_MODEL_DIR",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "TORCH_HOME",
)
QUIET_RUNTIME_ENV_VARS: Final[dict[str, str]] = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
}

DEFAULT_SECTION: Final[str] = "Document"
HEADING_LABELS: Final[set[str]] = {
    "TITLE",
    "DOCUMENT_TITLE",
    "SECTION_HEADER",
    "HEADER",
}
LIST_LABELS: Final[set[str]] = {
    "LIST_ITEM",
    "ORDERED_LIST_ITEM",
    "UNORDERED_LIST_ITEM",
}
TABLE_LABELS: Final[set[str]] = {"TABLE", "TABLE_CELL"}
FIGURE_LABELS: Final[set[str]] = {"PICTURE", "FIGURE", "IMAGE"}
IMAGE_SUFFIXES: Final[set[str]] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}
NOISY_DOCLING_LOGGERS: Final[tuple[str, ...]] = (
    "RapidOCR",
    "rapidocr",
    "docling",
    "easyocr",
    "filelock",
    "huggingface_hub",
    "onnxruntime",
    "transformers",
)
OCR_OPTION_CLASS_BY_ENGINE: Final[dict[str, str]] = {
    "easyocr": "EasyOcrOptions",
    "rapidocr": "RapidOcrOptions",
    "tesseract": "TesseractOcrOptions",
    "tesseract_cli": "TesseractCliOcrOptions",
    "ocrmac": "OcrMacOptions",
}
VLM_PRESET_ALIASES: Final[dict[str, str]] = {
    "GRANITE_DOCLING": "GRANITEDOCLING_TRANSFORMERS",
    "GRANITEDOCLING": "GRANITEDOCLING_TRANSFORMERS",
    "SMOLDOCLING": "SMOLDOCLING_TRANSFORMERS",
    "SMOL_DOCLING": "SMOLDOCLING_TRANSFORMERS",
}
