"""Docling parser implementation.

This module adapts Docling output to the ingestion parser protocol used by the
rest of the application. The implementation is intentionally defensive:

- Docling is imported lazily so importing the package does not force heavy
  runtime dependencies in unrelated call paths.
- Model-cache environment variables are set explicitly so the process can reuse
  model artifacts downloaded during application startup.
- Docling's rich document tree is normalized into deterministic `ParsedBlock`
  objects with stable IDs and provenance metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import threading
import warnings
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, contextmanager, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

from verdictai.ingestion.parser.docling.docling_parser_config import (
    MODEL_CACHE_ENV_VARS,
    QUIET_RUNTIME_ENV_VARS,
    DoclingParserConfig,
)
from verdictai.ingestion.parser.document_parser import DocumentParserProtocol
from verdictai.ingestion.parser.schemas import (
    DocumentFigureInspection,
    DocumentParseProgress,
    DocumentParseResult,
    ParsedBlock,
)
from verdictai.ingestion.parser.types import (
    ExtractedProvenance,
    JsonValue,
    PipelineMetadata,
    ProgressStage,
    ProvenanceEntry,
)
from verdictai.utils import get_logger

logger = get_logger(__name__)

_DEFAULT_SECTION = "Document"
_HEADING_LABELS = {"TITLE", "DOCUMENT_TITLE", "SECTION_HEADER", "HEADER"}
_LIST_LABELS = {"LIST_ITEM", "ORDERED_LIST_ITEM", "UNORDERED_LIST_ITEM"}
_TABLE_LABELS = {"TABLE", "TABLE_CELL"}
_FIGURE_LABELS = {"PICTURE", "FIGURE", "IMAGE"}
_NOISY_DOCLING_LOGGERS = (
    "RapidOCR",
    "rapidocr",
    "docling",
    "easyocr",
    "filelock",
    "huggingface_hub",
    "onnxruntime",
    "transformers",
)

ProgressCallback = Callable[[DocumentParseProgress], None]


class DoclingParserError(RuntimeError):
    """Raised when Docling parsing fails or returns an invalid payload."""


class DoclingParser(DocumentParserProtocol):
    """Parse documents into normalized `ParsedBlock` records using Docling.

    Notes:
        This class exposes API for synchronous parsing, async
        parsing, progress streaming, and Docling-specific figure inspection
        without requiring callers to reach into private helper methods.
    """

    def __init__(
        self,
        config: DoclingParserConfig | None = None,
        *,
        log: logging.Logger | None = None,
    ) -> None:
        """Create a Docling-backed parser.

        Args:
            config: Parser configuration. When omitted, defaults are used.
            log: Optional logger. When omitted, uses the module logger.
        """
        self.config = config or DoclingParserConfig()
        self._logger = log or logger

    def parse_document(self, path: str | Path) -> list[ParsedBlock]:
        """Parse a document and return canonical blocks.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            Parsed blocks in reading order.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If `path` exists but is not a file.
            DoclingParserError: If Docling fails or returns an invalid payload.
        """

        return self.parse_with_report(path).blocks

    def parse(self, path: str | Path) -> DocumentParseResult:
        """Parse a document and return the full structured report.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            A `DocumentParseResult` containing normalized blocks and metadata.
        """

        return self.parse_with_report(path)

    def parse_blocks(self, path: str | Path) -> list[ParsedBlock]:
        """Parse a document and return only the emitted blocks.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            Parsed blocks in reading order.
        """

        return self.parse_document(path)

    def parse_with_report(self, path: str | Path) -> DocumentParseResult:
        """Parse a document and return blocks plus structured metadata.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            A `DocumentParseResult` containing normalized blocks and metadata.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If `path` exists but is not a file.
            DoclingParserError: If Docling fails or returns an invalid payload.
        """

        return self._parse_with_report_internal(path)

    async def parse_async(self, path: str | Path) -> DocumentParseResult:
        """Parse a document asynchronously via a worker thread.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            A `DocumentParseResult` containing normalized blocks and metadata.
        """

        return await asyncio.to_thread(self.parse_with_report, path)

    async def parse_with_progress(
        self,
        path: str | Path,
    ) -> AsyncIterator[DocumentParseProgress]:
        """Yield structured progress events while parsing a document.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Yields:
            `DocumentParseProgress` events, ending with a `completed` event whose
            `report` field contains the final parse result.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If `path` exists but is not a file.
            DoclingParserError: If Docling fails or returns an invalid payload.

        Notes:
            The underlying Docling conversion remains synchronous, so this method
            runs the parse in a dedicated worker thread and forwards coarse-grained
            stage updates back onto the caller's event loop without scheduling an
            additional background asyncio task from inside the parser.
            Failures are raised after the background worker exits rather than being
            emitted as a terminal progress event.
        """

        queue: asyncio.Queue[DocumentParseProgress | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        error: Exception | None = None

        def on_progress(event: DocumentParseProgress) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run_parse() -> None:
            nonlocal error
            try:
                self._parse_with_report_internal(path, progress_callback=on_progress)
            except Exception as exc:  # pragma: no cover - exercised via await below
                error = exc
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        worker = threading.Thread(
            target=run_parse,
            name=f"{self.__class__.__name__}-progress",
            daemon=True,
        )
        worker.start()
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            await asyncio.to_thread(worker.join)

        if error is not None:
            raise error

    async def parse_with_progress_report(
        self,
        path: str | Path,
    ) -> tuple[DocumentParseResult, list[DocumentParseProgress]]:
        """Collect progress events and return the final parse report.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            A tuple containing the final parse result and the ordered progress
            events that produced it.

        Raises:
            RuntimeError: If the progress stream does not end with a final report.
        """

        events = [event async for event in self.parse_with_progress(path)]
        if not events or events[-1].report is None:
            raise RuntimeError("Docling progress stream completed without a report.")
        return events[-1].report, events

    def inspect_figures(self, path: str | Path) -> list[DocumentFigureInspection]:
        """Inspect figure items with an optional secondary OCR pass.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            Figure inspection records in document order.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If `path` exists but is not a file.
            DoclingParserError: If Docling conversion fails or RapidOCR is unavailable.
        """

        source_path = self._resolve_source_path(path)
        self._validate_source_path(source_path)
        self._configure_model_environment()
        with self._quiet_docling_runtime():
            api = self._import_docling()
            conversion_result, _pipeline_metadata = self._convert_document(
                api=api,
                source_path=source_path,
            )
            document = getattr(conversion_result, "document", None)
            if document is None:
                raise DoclingParserError(
                    f"Docling returned no document for {source_path.name}."
                )
            ocr_engine = self._build_rapidocr_engine()

        inspections: list[DocumentFigureInspection] = []
        for item, level in self._iterate_items(document):
            label = self._extract_label(item)
            if self._normalize_block_type(label=label) != "figure":
                continue

            provenance = self._extract_provenance(item)
            image = self._safe_call(item, "get_image", document)

            ocr_text = ""
            ocr_scores: list[float] = []
            if image is not None:
                with self._quiet_docling_runtime():
                    ocr_output = ocr_engine(image)
                if ocr_output.txts:
                    ocr_text = " ".join(ocr_output.txts)
                if ocr_output.scores:
                    ocr_scores = [float(score) for score in ocr_output.scores]

            inspections.append(
                DocumentFigureInspection(
                    page=provenance.page_number,
                    docling_label=label,
                    has_image=image is not None,
                    figure_ocr_text=ocr_text,
                    figure_ocr_scores=ocr_scores,
                    docling_is_ocr=provenance.is_ocr,
                    docling_confidence=provenance.average_confidence,
                    provenance=provenance.entries,
                    bbox=provenance.bbox,
                    tree_level=level,
                )
            )

        return inspections

    def _parse_with_report_internal(
        self,
        path: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> DocumentParseResult:
        """Execute the shared parsing flow used by both public entrypoints.

        Args:
            path: Filesystem path (string or `Path`) to the source document.
            progress_callback: Optional callback invoked after each major parsing
                stage with a `DocumentParseProgress` payload.

        Returns:
            A `DocumentParseResult` containing normalized blocks and metadata.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If `path` exists but is not a file.
            DoclingParserError: If Docling fails, returns no document, or yields
                no usable content.

        Notes:
            This method is the single source of truth for stage ordering so the
            sync `parse_with_report()` and async `parse_with_progress()` APIs stay
            behaviorally aligned.
        """

        # 1. Validate inputs early with explicit error types.
        source_path = self._resolve_source_path(path)
        self._emit_progress(
            progress_callback,
            stage="validating_input",
            message=f"Validating input path for {source_path.name}.",
            percent=5,
            metadata={"source_path": str(source_path)},
        )
        self._validate_source_path(source_path)

        # 2. Configure model cache environment before importing Docling so any
        #    backend library that reads env vars at import-time sees them.
        self._configure_model_environment()
        self._emit_progress(
            progress_callback,
            stage="configuring_environment",
            message="Configured Docling model and artifact environment.",
            percent=15,
            metadata={
                "model_cache_dir": self._stringify_path(
                    self.config.resolve_model_cache_dir()
                ),
                "artifacts_dir": self._stringify_path(
                    self.config.resolve_artifact_dir()
                ),
            },
        )

        self._emit_progress(
            progress_callback,
            stage="importing_backend",
            message="Importing Docling runtime.",
            percent=25,
        )
        with self._quiet_docling_runtime():
            api = self._import_docling()
            converter, pipeline_metadata = self._build_converter(api, source_path)

        self._emit_progress(
            progress_callback,
            stage="building_converter",
            message="Prepared Docling converter for this document.",
            percent=35,
            metadata={"pipeline": cast(JsonValue, pipeline_metadata)},
        )
        try:
            # 3. Convert the document into Docling's internal representation.
            self._emit_progress(
                progress_callback,
                stage="converting",
                message="Running Docling conversion.",
                percent=45,
                metadata={"source_name": source_path.name},
            )
            with self._quiet_docling_runtime():
                conversion_result = converter.convert(source_path)
        except Exception as exc:  # pragma: no cover - depends on docling runtime
            raise DoclingParserError(
                f"Docling failed to parse {source_path.name}: {exc}"
            ) from exc

        self._emit_progress(
            progress_callback,
            stage="conversion_complete",
            message="Docling conversion finished; normalizing output.",
            percent=70,
            metadata={
                "conversion_status": getattr(conversion_result, "status", None),
            },
        )
        document = getattr(conversion_result, "document", None)
        if document is None:
            raise DoclingParserError(
                f"Docling returned no document for {source_path.name}."
            )

        # 4. Normalize the Docling tree into stable `ParsedBlock` records.
        self._emit_progress(
            progress_callback,
            stage="normalizing_blocks",
            message="Building canonical parsed blocks from Docling output.",
            percent=82,
        )
        blocks = self._build_blocks(document=document, source_path=source_path)
        if not blocks:
            # 4.A Fallback to a whole-document export when item iteration yields
            #     nothing (some backends can return an empty tree but still support
            #     exporting a text/markdown representation).
            fallback_block = self._build_fallback_block(
                document=document,
                source_path=source_path,
            )
            if fallback_block is None:
                raise DoclingParserError(
                    f"Docling produced no usable blocks for {source_path.name}."
                )
            blocks = [fallback_block]

        # 5. Assemble metadata for observability and downstream debugging.
        self._emit_progress(
            progress_callback,
            stage="building_metadata",
            message="Assembling parse metadata and summary statistics.",
            percent=92,
            metadata={"block_count": len(blocks)},
        )
        metadata = self._build_result_metadata(
            document=document,
            source_path=source_path,
            blocks=blocks,
            pipeline_metadata=pipeline_metadata,
            conversion_result=conversion_result,
        )
        report = DocumentParseResult(blocks=blocks, metadata=metadata)
        self._emit_progress(
            progress_callback,
            stage="completed",
            message="Docling parsing completed successfully.",
            percent=100,
            metadata={
                "block_count": len(blocks),
                "page_count": metadata.get("page_count"),
            },
            report=report,
        )
        return report

    def _resolve_source_path(self, path: str | Path) -> Path:
        """Resolve a source document path.

        Args:
            path: Filesystem path (string or `Path`) to the source document.

        Returns:
            Fully resolved filesystem path for the document.
        """

        return Path(path).expanduser().resolve()

    def _validate_source_path(self, source_path: Path) -> None:
        """Validate that the resolved path exists and points to a file.

        Args:
            source_path: Resolved filesystem path.

        Raises:
            FileNotFoundError: If `source_path` does not exist.
            ValueError: If `source_path` exists but is not a file.
        """

        if not source_path.exists():
            raise FileNotFoundError(f"Document does not exist: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"Document path is not a file: {source_path}")

    def _configure_model_environment(self) -> None:
        """Configure model cache environment variables.

        This method sets default values (via `os.environ.setdefault`) for the
        env vars in `MODEL_CACHE_ENV_VARS` so the process can reuse predownloaded
        model artifacts across runs.
        """

        model_cache_dir = self.config.resolve_model_cache_dir()
        if model_cache_dir is None:
            return

        model_cache_dir.mkdir(parents=True, exist_ok=True)
        model_cache_dir_str = str(model_cache_dir)
        for env_var in MODEL_CACHE_ENV_VARS:
            os.environ.setdefault(env_var, model_cache_dir_str)
        if self.config.suppress_progress_bars:
            for env_var, value in QUIET_RUNTIME_ENV_VARS.items():
                os.environ.setdefault(env_var, value)

    def _import_docling(self) -> SimpleNamespace:
        """Import docling lazily and expose the needed API surface.

        Returns:
            A `SimpleNamespace` containing the docling classes and constants used
            by this parser.

        Raises:
            DoclingParserError: If docling is unavailable in the current runtime.
        """

        try:
            from docling.datamodel import vlm_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                EasyOcrOptions,
                OcrMacOptions,
                PdfPipelineOptions,
                RapidOcrOptions,
                TesseractCliOcrOptions,
                TesseractOcrOptions,
                VlmPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.pipeline.vlm_pipeline import VlmPipeline
        except ImportError as exc:  # pragma: no cover - depends on env
            raise DoclingParserError(
                "Docling is not installed in the current runtime. "
                "Install the project dependencies before using DoclingParser."
            ) from exc

        return SimpleNamespace(
            DocumentConverter=DocumentConverter,
            EasyOcrOptions=EasyOcrOptions,
            InputFormat=InputFormat,
            OcrMacOptions=OcrMacOptions,
            PdfFormatOption=PdfFormatOption,
            PdfPipelineOptions=PdfPipelineOptions,
            RapidOcrOptions=RapidOcrOptions,
            TesseractCliOcrOptions=TesseractCliOcrOptions,
            TesseractOcrOptions=TesseractOcrOptions,
            VlmPipeline=VlmPipeline,
            VlmPipelineOptions=VlmPipelineOptions,
            vlm_model_specs=vlm_model_specs,
        )

    def _build_converter(
        self,
        api: SimpleNamespace,
        source_path: Path,
    ) -> tuple[Any, PipelineMetadata]:
        """Build a docling converter configured for the selected pipeline.

        Args:
            api: Docling API namespace from `_import_docling()`.
            source_path: Path to the source document, used for auto pipeline selection.

        Returns:
            A tuple of:
              1) The configured Docling converter instance.
              2) A JSON-serializable metadata dict describing the chosen pipeline.
        """

        format_options: dict[Any, Any] = {}
        pipeline = self._select_pipeline(source_path)
        pipeline_metadata: PipelineMetadata = {
            "pipeline": pipeline,
            "ocr_enabled": self.config.do_ocr,
            "table_structure_enabled": self.config.do_table_structure,
            "ocr_engine": self.config.ocr_engine if self.config.do_ocr else None,
            "vlm_model": None,
            "force_backend_text": False,
            "picture_images_enabled": self.config.generate_picture_images,
            "remote_services_enabled": False,
            "model_cache_dir": self._stringify_path(
                self.config.resolve_model_cache_dir()
            ),
            "artifacts_dir": self._stringify_path(self.config.resolve_artifact_dir()),
        }

        if pipeline == "standard":
            format_options[api.InputFormat.PDF] = api.PdfFormatOption(
                pipeline_options=self._build_standard_pipeline_options(api)
            )
        else:
            format_options[api.InputFormat.PDF] = api.PdfFormatOption(
                pipeline_cls=api.VlmPipeline,
                pipeline_options=self._build_vlm_pipeline_options(api),
            )
            pipeline_metadata["vlm_model"] = self.config.vlm_model
            pipeline_metadata["force_backend_text"] = self.config.force_backend_text
            pipeline_metadata["remote_services_enabled"] = (
                self.config.enable_remote_services
            )

        converter = api.DocumentConverter(
            format_options=format_options if format_options else None
        )
        return converter, pipeline_metadata

    def _convert_document(
        self,
        *,
        api: SimpleNamespace,
        source_path: Path,
    ) -> tuple[Any, PipelineMetadata]:
        """Build a converter and run one Docling conversion.

        Args:
            api: Docling API namespace from `_import_docling()`.
            source_path: Resolved source document path.

        Returns:
            A tuple containing the raw Docling conversion result and the pipeline
            metadata used to produce it.
        """

        converter, pipeline_metadata = self._build_converter(api, source_path)
        conversion_result = converter.convert(source_path)
        return conversion_result, pipeline_metadata

    def _build_rapidocr_engine(self) -> Any:
        """Create a RapidOCR engine configured for the project's model cache.

        Returns:
            A configured `RapidOCR` callable.

        Raises:
            DoclingParserError: If RapidOCR is unavailable in the current runtime.
        """

        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on env
            raise DoclingParserError(
                "RapidOCR is not installed in the current runtime. "
                "Install the project dependencies before inspecting figures."
            ) from exc

        artifacts_root = self.config.resolve_artifact_dir()
        rapidocr_model_root = None
        if artifacts_root is not None:
            candidate = artifacts_root / "RapidOcr/onnx/PP-OCRv4"
            if candidate.exists():
                rapidocr_model_root = candidate

        ocr_params: dict[str, str] = {}
        if rapidocr_model_root is not None:
            ocr_params.update(
                {
                    "Global.model_root_dir": str(rapidocr_model_root),
                    "Det.model_path": str(
                        rapidocr_model_root / "det/ch_PP-OCRv4_det_mobile.onnx"
                    ),
                    "Cls.model_path": str(
                        rapidocr_model_root / "cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx"
                    ),
                    "Rec.model_path": str(
                        rapidocr_model_root / "rec/ch_PP-OCRv4_rec_mobile.onnx"
                    ),
                }
            )

        return RapidOCR(params=ocr_params or None)

    def _build_standard_pipeline_options(self, api: SimpleNamespace) -> Any:
        """Create Docling `PdfPipelineOptions` for the standard pipeline.

        Args:
            api: Docling API namespace from `_import_docling()`.

        Returns:
            A Docling `PdfPipelineOptions` instance.
        """

        kwargs: dict[str, Any] = {
            "do_ocr": self.config.do_ocr,
            "do_table_structure": self.config.do_table_structure,
            "generate_picture_images": self.config.generate_picture_images,
        }
        artifacts_dir = self.config.resolve_artifact_dir()
        if artifacts_dir is not None:
            kwargs["artifacts_path"] = str(artifacts_dir)
        ocr_options = self._build_ocr_options(api)
        if ocr_options is not None:
            kwargs["ocr_options"] = ocr_options
        return api.PdfPipelineOptions(**kwargs)

    def _build_vlm_pipeline_options(self, api: SimpleNamespace) -> Any:
        """Create Docling `VlmPipelineOptions` for the VLM pipeline.

        Args:
            api: Docling API namespace from `_import_docling()`.

        Returns:
            A Docling `VlmPipelineOptions` instance.

        Raises:
            DoclingParserError: If the configured VLM preset is unsupported.
        """

        preset_name = self._resolve_vlm_preset_name(self.config.vlm_model)
        try:
            vlm_options = getattr(api.vlm_model_specs, preset_name)
        except AttributeError as exc:
            raise DoclingParserError(
                f"Unsupported Docling VLM preset '{self.config.vlm_model}'."
            ) from exc

        return api.VlmPipelineOptions(
            vlm_options=vlm_options,
            force_backend_text=self.config.force_backend_text,
            generate_page_images=self.config.generate_page_images,
            generate_picture_images=self.config.generate_picture_images,
            enable_remote_services=self.config.enable_remote_services,
        )

    def _build_ocr_options(self, api: SimpleNamespace) -> Any | None:
        """Instantiate the configured OCR backend options.

        Args:
            api: Docling API namespace from `_import_docling()`.

        Returns:
            OCR options instance understood by Docling, or None when OCR is disabled.
        """

        if not self.config.do_ocr:
            return None

        option_class_name = {
            "easyocr": "EasyOcrOptions",
            "rapidocr": "RapidOcrOptions",
            "tesseract": "TesseractOcrOptions",
            "tesseract_cli": "TesseractCliOcrOptions",
            "ocrmac": "OcrMacOptions",
        }[self.config.ocr_engine]

        option_class = getattr(api, option_class_name)
        return option_class()

    def _select_pipeline(self, source_path: Path) -> Literal["standard", "vlm"]:
        """Choose the Docling pipeline for the current source file.

        Args:
            source_path: Input document path.

        Returns:
            `"standard"` for the default PDF pipeline, or `"vlm"` for the VLM pipeline.
        """

        if self.config.pipeline == "standard":
            return "standard"
        if self.config.pipeline == "vlm":
            return "vlm"

        image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        return "vlm" if source_path.suffix.lower() in image_suffixes else "standard"

    def _build_blocks(
        self,
        *,
        document: Any,
        source_path: Path,
    ) -> list[ParsedBlock]:
        """Normalize Docling items into canonical ingestion blocks.

        Args:
            document: Docling document object returned by the converter.
            source_path: Resolved filesystem path for stable IDs/metadata.

        Returns:
            List of `ParsedBlock` records in reading order.
        """

        blocks: list[ParsedBlock] = []
        heading_stack: list[str] = []
        page_order: defaultdict[int, int] = defaultdict(int)

        for item, level in self._iterate_items(document):
            # 1. Extract the best representations Docling exposes for this item.
            label = self._extract_label(item)
            block_type = self._normalize_block_type(label=label)
            text = self._extract_text(item=item, document=document)
            markdown = self._extract_markdown(item=item, document=document)

            if not text and not markdown and block_type not in {"table", "figure"}:
                continue

            # 2. Extract provenance and compute a stable order within each page.
            provenance = self._extract_provenance(item)
            page_number = provenance.page_number or 1
            order_in_page = page_order[page_number]
            page_order[page_number] += 1

            # 3. Maintain section context from headings while walking the tree.
            heading_level = self._resolve_heading_level(
                block_type=block_type,
                label=label,
                level=level,
            )
            if block_type == "heading":
                heading_text = text or markdown or "Untitled section"
                self._update_heading_stack(
                    heading_stack=heading_stack,
                    heading=heading_text,
                    heading_level=heading_level or 1,
                )
                section = heading_text
                parent_section_path = heading_stack[:-1]
            else:
                section = heading_stack[-1] if heading_stack else _DEFAULT_SECTION
                parent_section_path = list(heading_stack)

            # 4. Construct a deterministic block id suitable for storage and joins.
            block_id = self._build_block_id(
                source_path=source_path,
                page=page_number,
                order_in_page=order_in_page,
                block_type=block_type,
                text=text,
                section=section,
            )

            block_metadata: dict[str, JsonValue] = {
                "docling_label": label,
                "docling_tree_level": level,
                "docling_item_type": type(item).__name__,
            }
            if provenance.entries:
                block_metadata["provenance"] = cast(JsonValue, provenance.entries)

            blocks.append(
                ParsedBlock(
                    doc_id=self._document_id(source_path),
                    source_path=str(source_path),
                    source_name=source_path.name,
                    doc_type=source_path.suffix.lower().lstrip(".") or None,
                    block_id=block_id,
                    page=page_number,
                    section=section,
                    heading_level=heading_level,
                    block_type=block_type,
                    text=text,
                    markdown=markdown,
                    bbox=provenance.bbox,
                    order_in_page=order_in_page,
                    is_ocr=provenance.is_ocr,
                    confidence=provenance.average_confidence,
                    parent_section_path=parent_section_path,
                    parser_used="docling",
                    ocr_used=self.config.ocr_engine if provenance.is_ocr else None,
                    parse_confidence=provenance.average_confidence,
                    metadata=block_metadata,
                )
            )

        return blocks

    def _build_fallback_block(
        self,
        *,
        document: Any,
        source_path: Path,
    ) -> ParsedBlock | None:
        """Create a single whole-document block when item iteration yields nothing.

        Args:
            document: Docling document object returned by the converter.
            source_path: Resolved filesystem path for stable IDs/metadata.

        Returns:
            A single `ParsedBlock` representing the entire document content, or
            None if Docling cannot export content.
        """

        markdown = self._safe_call(document, "export_to_markdown")
        text = self._safe_call(document, "export_to_text")
        if not isinstance(markdown, str):
            markdown = None
        if not isinstance(text, str):
            text = None
        if not text and not markdown:
            return None

        content = (text or markdown or "").strip()
        if not content:
            return None

        return ParsedBlock(
            doc_id=self._document_id(source_path),
            source_path=str(source_path),
            source_name=source_path.name,
            doc_type=source_path.suffix.lower().lstrip(".") or None,
            block_id=self._build_block_id(
                source_path=source_path,
                page=1,
                order_in_page=0,
                block_type="document",
                text=content,
                section=_DEFAULT_SECTION,
            ),
            page=1,
            section=_DEFAULT_SECTION,
            block_type="document",
            text=content,
            markdown=markdown,
            order_in_page=0,
            parent_section_path=[],
            parser_used="docling",
            metadata={"fallback": "whole-document-export"},
        )

    def _build_result_metadata(
        self,
        *,
        document: Any,
        source_path: Path,
        blocks: list[ParsedBlock],
        pipeline_metadata: PipelineMetadata,
        conversion_result: Any,
    ) -> dict[str, JsonValue]:
        """Assemble top-level parse metadata for observability and debugging.

        Args:
            document: Docling document object returned by the converter.
            source_path: Resolved filesystem path for the original input.
            blocks: Emitted `ParsedBlock` list.
            pipeline_metadata: Pipeline selection and configuration summary.
            conversion_result: Raw converter output, used for status/warnings/errors.

        Returns:
            JSON-serializable metadata dictionary.
        """

        pages = self._extract_page_count(document=document, blocks=blocks)
        headings = sum(1 for block in blocks if block.block_type == "heading")
        tables = sum(1 for block in blocks if block.block_type == "table")
        figures = sum(1 for block in blocks if block.block_type == "figure")

        metadata: dict[str, JsonValue] = {
            "parser_used": "docling",
            "source_path": str(source_path),
            "source_name": source_path.name,
            "doc_type": source_path.suffix.lower().lstrip(".") or None,
            "document_id": self._document_id(source_path),
            "page_count": pages,
            "block_count": len(blocks),
            "heading_count": headings,
            "table_count": tables,
            "figure_count": figures,
            "pipeline": cast(JsonValue, pipeline_metadata),
            "conversion_status": getattr(conversion_result, "status", None),
            "warnings": cast(
                JsonValue,
                self._coerce_to_string_list(getattr(conversion_result, "warnings", None)),
            ),
            "errors": cast(
                JsonValue,
                self._coerce_to_string_list(getattr(conversion_result, "errors", None)),
            ),
        }

        document_name = getattr(document, "name", None)
        if document_name:
            metadata["docling_document_name"] = document_name

        return metadata

    def _iterate_items(self, document: Any) -> list[tuple[Any, int]]:
        """Return Docling items in reading order.

        Args:
            document: Docling document object.

        Returns:
            List of `(item, level)` tuples, where `level` is the Docling tree depth.
            Returns an empty list if the document does not expose an iterator.
        """

        iterator = getattr(document, "iterate_items", None)
        if not callable(iterator):
            return []

        try:
            items = list(iterator())
        except TypeError:
            items = list(iterator(document))

        normalized: list[tuple[Any, int]] = []
        for entry in items:
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and isinstance(entry[1], int)
            ):
                normalized.append((entry[0], entry[1]))
        return normalized

    def _extract_text(self, *, item: Any, document: Any) -> str:
        """Extract the best plain-text representation of a Docling item.

        Args:
            item: Docling item from `document.iterate_items()`.
            document: Docling document object (used for some caption APIs).

        Returns:
            Best-effort plain text. Empty string when no text is available.
        """

        for attribute in ("text", "orig"):
            value = getattr(item, attribute, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if self._normalize_block_type(label=self._extract_label(item)) == "figure":
            caption = self._safe_call(item, "caption_text", document)
            if isinstance(caption, str) and caption.strip():
                return caption.strip()

        exported = self._safe_call(item, "export_to_text")
        if isinstance(exported, str) and exported.strip():
            return exported.strip()

        markdown = self._safe_call(item, "export_to_markdown", document)
        if isinstance(markdown, str):
            return markdown.strip()

        return ""

    def _extract_markdown(self, *, item: Any, document: Any) -> str | None:
        """Extract a markdown representation for a Docling item.

        Args:
            item: Docling item from `document.iterate_items()`.
            document: Docling document object used by some export APIs.

        Returns:
            Markdown string when available; otherwise None.
        """

        markdown = self._safe_call(item, "export_to_markdown", document)
        if not isinstance(markdown, str):
            markdown = self._safe_call(item, "export_to_markdown")
        if isinstance(markdown, str):
            normalized = markdown.strip()
            return normalized or None
        return None

    def _extract_provenance(self, item: Any) -> ExtractedProvenance:
        """Extract page/bbox/confidence data from Docling provenance.

        Args:
            item: Docling item from `document.iterate_items()`.

        Returns:
            Structured provenance fields derived from Docling metadata.
        """

        prov_entries = getattr(item, "prov", None) or []
        normalized_provenance: list[ProvenanceEntry] = []
        page_number: int | None = None
        bbox: tuple[float, float, float, float] | None = None
        confidence_values: list[float] = []
        is_ocr = False

        for prov in prov_entries:
            # 1. Pull out best-effort fields from Docling provenance objects.
            page_no = getattr(prov, "page_no", None)
            bbox_value = getattr(prov, "bbox", None)
            charspan = getattr(prov, "charspan", None)
            confidence = getattr(prov, "confidence", None)
            source = getattr(prov, "source", None)

            if page_number is None and isinstance(page_no, int) and page_no > 0:
                page_number = page_no

            if bbox is None:
                bbox = self._normalize_bbox(bbox_value)

            if isinstance(confidence, (float, int)):
                confidence_values.append(float(confidence))

            source_name = self._stringify_source(source)
            if source_name and "ocr" in source_name.lower():
                is_ocr = True

            normalized_provenance.append(
                {
                    "page_no": page_no,
                    "bbox": list(bbox) if bbox else None,
                    "charspan": self._coerce_charspan(charspan),
                    "confidence": float(confidence)
                    if isinstance(confidence, (float, int))
                    else None,
                    "source": source_name,
                }
            )

        # 2. Derive aggregate confidence across provenance entries.
        average_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        )
        return ExtractedProvenance(
            page_number=page_number,
            bbox=bbox,
            average_confidence=average_confidence,
            is_ocr=is_ocr,
            entries=normalized_provenance,
        )

    def _normalize_block_type(self, *, label: str | None) -> str:
        """Map Docling item labels onto canonical ingestion block types."""

        if label in _HEADING_LABELS:
            return "heading"
        if label in _LIST_LABELS:
            return "list_item"
        if label in _TABLE_LABELS:
            return "table"
        if label in _FIGURE_LABELS:
            return "figure"
        if label == "CODE":
            return "code"
        if label == "FORMULA":
            return "formula"
        if label == "CAPTION":
            return "caption"
        return "paragraph"

    def _extract_label(self, item: Any) -> str | None:
        """Return the Docling item label name when present."""

        label = getattr(item, "label", None)
        if label is None:
            return None
        name = getattr(label, "name", None)
        if isinstance(name, str):
            return name
        if isinstance(label, str):
            return label
        return None

    def _resolve_heading_level(
        self,
        *,
        block_type: str,
        label: str | None,
        level: int,
    ) -> int | None:
        """Compute a normalized heading depth in the supported 1..6 range."""

        if block_type != "heading" and label not in _HEADING_LABELS:
            return None
        return max(1, min(level or 1, 6))

    def _update_heading_stack(
        self,
        *,
        heading_stack: list[str],
        heading: str,
        heading_level: int,
    ) -> None:
        """Maintain heading ancestry while iterating the Docling tree."""

        # 1. Normalize text to avoid unstable section labels caused by whitespace.
        clean_heading = " ".join(heading.split()) or "Untitled section"
        if len(heading_stack) >= heading_level:
            del heading_stack[heading_level - 1 :]
        # 2. Backfill missing ancestors so `heading_stack` depth always matches
        #    `heading_level - 1` before appending the current heading.
        while len(heading_stack) < heading_level - 1:
            heading_stack.append(_DEFAULT_SECTION)
        heading_stack.append(clean_heading)

    def _extract_page_count(self, *, document: Any, blocks: list[ParsedBlock]) -> int:
        """Determine total page count from the Docling document or emitted blocks.

        Args:
            document: Docling document object.
            blocks: Emitted blocks used as a fallback page-count signal.

        Returns:
            Total page count (0 when unknown).
        """

        pages = getattr(document, "pages", None)
        if pages is not None:
            try:
                return len(pages)
            except TypeError:
                pass

        if blocks:
            return max(block.page for block in blocks)
        return 0

    def _resolve_vlm_preset_name(self, value: str) -> str:
        """Translate ergonomic VLM names to Docling preset constants.

        Args:
            value: User-facing VLM identifier.

        Returns:
            Docling preset constant name.
        """

        normalized = value.strip().upper()
        alias_map = {
            "GRANITE_DOCLING": "GRANITEDOCLING_TRANSFORMERS",
            "GRANITEDOCLING": "GRANITEDOCLING_TRANSFORMERS",
            "SMOLDOCLING": "SMOLDOCLING_TRANSFORMERS",
            "SMOL_DOCLING": "SMOLDOCLING_TRANSFORMERS",
        }
        return alias_map.get(normalized, normalized)

    def _document_id(self, source_path: Path) -> str:
        """Build a stable document identifier.

        Args:
            source_path: Resolved source path.

        Returns:
            SHA1 hex digest derived from the path string.
        """

        digest = hashlib.sha1(str(source_path).encode("utf-8"), usedforsecurity=False)
        return digest.hexdigest()

    def _build_block_id(
        self,
        *,
        source_path: Path,
        page: int,
        order_in_page: int,
        block_type: str,
        text: str,
        section: str,
    ) -> str:
        """Build a deterministic content block identifier.

        Args:
            source_path: Resolved source path.
            page: One-based page number.
            order_in_page: Stable order index within the page (0-based).
            block_type: Normalized block type.
            text: Block text (used as part of the fingerprint).
            section: Section label (used as part of the fingerprint).

        Returns:
            SHA1 hex digest for the block fingerprint.
        """

        text_preview = " ".join(text.split())[:160]
        fingerprint = "|".join(
            [
                str(source_path),
                str(page),
                str(order_in_page),
                block_type,
                section,
                text_preview,
            ]
        )
        return hashlib.sha1(
            fingerprint.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()

    def _normalize_bbox(
        self,
        bbox: Any,
    ) -> tuple[float, float, float, float] | None:
        """Normalize various Docling bbox shapes into a fixed 4-tuple.

        Args:
            bbox: Docling bbox payload which may be tuple/list-like or an object
                with `l/t/r/b` attributes.

        Returns:
            `(left, top, right, bottom)` when parseable, otherwise None.
        """

        if bbox is None:
            return None

        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            try:
                left, top, right, bottom = (float(value) for value in bbox)
                return (left, top, right, bottom)
            except (TypeError, ValueError):
                return None

        coordinates: list[Any] = []
        for attribute in ("l", "t", "r", "b"):
            value = getattr(bbox, attribute, None)
            if value is None:
                coordinates = []
                break
            coordinates.append(value)

        if len(coordinates) == 4:
            try:
                left, top, right, bottom = (float(value) for value in coordinates)
                return (left, top, right, bottom)
            except (TypeError, ValueError):
                return None

        return None

    def _stringify_source(self, value: Any) -> str | None:
        """Convert a provenance source enum or object to a readable string.

        Args:
            value: Docling provenance source value.

        Returns:
            String identifier or None.
        """

        if value is None:
            return None
        name = getattr(value, "name", None)
        if isinstance(name, str):
            return name
        value_as_str = str(value)
        return value_as_str if value_as_str else None

    def _coerce_charspan(self, value: Any) -> list[int] | None:
        """Normalize character-span metadata when present.

        Args:
            value: Charspan payload.

        Returns:
            Two-integer list `[start, end]` when valid, else None.
        """

        if isinstance(value, (tuple, list)) and len(value) == 2:
            try:
                return [int(value[0]), int(value[1])]
            except (TypeError, ValueError):
                return None
        return None

    def _coerce_to_string_list(self, value: Any) -> list[str]:
        """Normalize Docling warning/error collections to strings.

        Args:
            value: Warning/error payload from Docling.

        Returns:
            List of stringified items (empty list when value is None).
        """

        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        return [str(value)]

    def _stringify_path(self, value: Path | None) -> str | None:
        """Convert a path to string when present.

        Args:
            value: Path value.

        Returns:
            Path as string, or None.
        """

        return str(value) if value is not None else None

    def _safe_call(self, target: Any, method_name: str, *args: Any) -> Any | None:
        """Invoke a Docling method defensively.

        Args:
            target: Object to call the method on.
            method_name: Method name to look up on `target`.
            *args: Positional args passed to the method.

        Returns:
            Method result on success, otherwise None.
        """

        method = getattr(target, method_name, None)
        if not callable(method):
            return None

        try:
            return method(*args)
        except Exception:  # pragma: no cover - depends on docling runtime
            self._logger.debug(
                "Docling method %s failed for %s",
                method_name,
                type(target).__name__,
                exc_info=True,
            )
            return None

    def _emit_progress(
        self,
        callback: ProgressCallback | None,
        *,
        stage: ProgressStage,
        message: str,
        percent: int,
        metadata: dict[str, JsonValue] | None = None,
        report: DocumentParseResult | None = None,
    ) -> None:
        """Build and dispatch one progress event to the provided callback.

        Args:
            callback: Progress consumer. When None, emission is skipped.
            stage: Stable lifecycle stage identifier.
            message: Human-readable description of the stage.
            percent: Approximate completion percentage for the stage.
            metadata: Optional JSON-serializable payload for observability.
            report: Final parse result attached to the terminal `completed` event.
        """

        if callback is None:
            return

        callback(
            DocumentParseProgress(
                stage=stage,
                message=message,
                percent=percent,
                metadata=metadata or {},
                report=report,
            )
        )

    @contextmanager
    def _suppress_docling_logs(self) -> Any:
        """Temporarily raise third-party Docling logger levels to `ERROR`.

        Yields:
            Control to the wrapped Docling operation.

        Notes:
            This only targets known noisy third-party logger namespaces so
            `verdictai` application logs remain unchanged.
        """

        loggers = [logging.getLogger(name) for name in _NOISY_DOCLING_LOGGERS]
        previous_levels = {logger.name: logger.level for logger in loggers}
        previous_disabled = {logger.name: logger.disabled for logger in loggers}

        try:
            for noisy_logger in loggers:
                noisy_logger.disabled = False
                noisy_logger.setLevel(logging.ERROR)
            yield
        finally:
            for noisy_logger in loggers:
                noisy_logger.disabled = previous_disabled[noisy_logger.name]
                noisy_logger.setLevel(previous_levels[noisy_logger.name])

    @contextmanager
    def _temporary_environment(self, values: dict[str, str]) -> Any:
        """Temporarily set environment variables for a noisy runtime section.

        Args:
            values: Environment variables to set for the context duration.

        Yields:
            Control to the wrapped runtime section.
        """

        previous_values = {key: os.environ.get(key) for key in values}
        try:
            for key, value in values.items():
                os.environ[key] = value
            yield
        finally:
            for key, previous_value in previous_values.items():
                if previous_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous_value

    @contextmanager
    def _quiet_docling_runtime(self) -> Any:
        """Suppress third-party logs, warnings, and progress UI around Docling.

        Yields:
            Control to a Docling or OCR runtime section.

        Notes:
            The context intentionally silences known non-actionable notebook noise
            such as `tqdm` widget warnings, Hugging Face progress bars, and noisy
            OCR backend logger output while preserving parser exceptions.
        """

        with ExitStack() as stack:
            if self.config.suppress_external_logs:
                stack.enter_context(self._suppress_docling_logs())
            if self.config.suppress_progress_bars:
                stack.enter_context(self._temporary_environment(QUIET_RUNTIME_ENV_VARS))
                stack.enter_context(redirect_stderr(io.StringIO()))
            if self.config.suppress_runtime_warnings:
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("ignore")
                try:
                    from tqdm import TqdmWarning
                except ImportError:  # pragma: no cover - depends on env
                    pass
                else:
                    warnings.filterwarnings("ignore", category=TqdmWarning)
                warnings.filterwarnings("ignore", message=".*IProgress not found.*")
            yield
