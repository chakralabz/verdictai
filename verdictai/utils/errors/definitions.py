"""Central error definitions for VerdictAI.

Each error is defined as a plain dictionary so new failures can be added without
creating new classes or spreading message strings across the codebase.
"""

from __future__ import annotations

from typing import Final, TypedDict


class ErrorDefinition(TypedDict):
    """Dictionary payload used to construct VerdictAI exceptions.

    Attributes:
        error_code: Stable, machine-readable error identifier.
        error_message: Human-readable message template.
    """

    error_code: str
    error_message: str


PARSER_PROGRESS_REPORT_MISSING: Final[ErrorDefinition] = {
    "error_code": "PARSER_001",
    "error_message": "Docling progress stream completed without a report.",
}
PARSER_DOCLING_EMPTY_DOCUMENT: Final[ErrorDefinition] = {
    "error_code": "PARSER_002",
    "error_message": "Docling returned no document for {source_name}.",
}
PARSER_DOCLING_CONVERSION_FAILED: Final[ErrorDefinition] = {
    "error_code": "PARSER_003",
    "error_message": "Docling failed to parse {source_name}: {reason}",
}
PARSER_DOCLING_NO_USABLE_BLOCKS: Final[ErrorDefinition] = {
    "error_code": "PARSER_004",
    "error_message": "Docling produced no usable blocks for {source_name}.",
}
PARSER_DOCLING_NOT_INSTALLED: Final[ErrorDefinition] = {
    "error_code": "PARSER_005",
    "error_message": (
        "Docling is not installed in the current runtime. Install the project "
        "dependencies before using DoclingParser."
    ),
}
PARSER_RAPIDOCR_NOT_INSTALLED: Final[ErrorDefinition] = {
    "error_code": "PARSER_006",
    "error_message": (
        "RapidOCR is not installed in the current runtime. Install the project "
        "dependencies before inspecting figures."
    ),
}
PARSER_UNSUPPORTED_VLM_PRESET: Final[ErrorDefinition] = {
    "error_code": "PARSER_007",
    "error_message": "Unsupported Docling VLM preset '{preset_name}'.",
}
PARSER_SOURCE_NOT_FOUND: Final[ErrorDefinition] = {
    "error_code": "PARSER_008",
    "error_message": "Document does not exist: {source_path}",
}
PARSER_SOURCE_NOT_FILE: Final[ErrorDefinition] = {
    "error_code": "PARSER_009",
    "error_message": "Document path is not a file: {source_path}",
}

CHUNKER_DOCLING_DEPENDENCIES_UNAVAILABLE: Final[ErrorDefinition] = {
    "error_code": "CHUNKER_001",
    "error_message": (
        "Docling chunking dependencies are unavailable. Install the project "
        "dependencies and the required docling-core chunking extras."
    ),
}
CHUNKER_DOCLING_METADATA_MISSING: Final[ErrorDefinition] = {
    "error_code": "CHUNKER_002",
    "error_message": (
        "The parse result does not include `docling_document` metadata. Chunking "
        "requires the direct output of `DoclingParser.parse_with_report()`."
    ),
}
CHUNKER_DOCLING_METADATA_INVALID: Final[ErrorDefinition] = {
    "error_code": "CHUNKER_003",
    "error_message": (
        "The serialized `docling_document` payload could not be restored."
    ),
}
CHUNKER_OPENAI_TOKENIZER_DEPENDENCY_MISSING: Final[ErrorDefinition] = {
    "error_code": "CHUNKER_004",
    "error_message": "OpenAI tokenizer support requires the `tiktoken` package.",
}
CHUNKER_UNSUPPORTED_TOKENIZER_PROVIDER: Final[ErrorDefinition] = {
    "error_code": "CHUNKER_005",
    "error_message": "Unsupported tokenizer provider: {provider}",
}


ERROR_DEFINITIONS: Final[dict[str, ErrorDefinition]] = {
    definition["error_code"]: definition
    for definition in (
        PARSER_PROGRESS_REPORT_MISSING,
        PARSER_DOCLING_EMPTY_DOCUMENT,
        PARSER_DOCLING_CONVERSION_FAILED,
        PARSER_DOCLING_NO_USABLE_BLOCKS,
        PARSER_DOCLING_NOT_INSTALLED,
        PARSER_RAPIDOCR_NOT_INSTALLED,
        PARSER_UNSUPPORTED_VLM_PRESET,
        PARSER_SOURCE_NOT_FOUND,
        PARSER_SOURCE_NOT_FILE,
        CHUNKER_DOCLING_DEPENDENCIES_UNAVAILABLE,
        CHUNKER_DOCLING_METADATA_MISSING,
        CHUNKER_DOCLING_METADATA_INVALID,
        CHUNKER_OPENAI_TOKENIZER_DEPENDENCY_MISSING,
        CHUNKER_UNSUPPORTED_TOKENIZER_PROVIDER,
    )
}
