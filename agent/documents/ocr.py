"""OCR for scanned PDF pages and images.

RapidOCR runs PaddleOCR models on the ONNX runtime the embeddings already use, and pypdfium2
renders PDF pages; both install from pip, so there is no system Tesseract to manage. The engine
loads lazily on first use (~1.5 s) and is shared, so OCR costs nothing until a scan arrives.
"""

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RENDER_DPI = 200
MIN_CONFIDENCE = 0.5


class OCRUnavailableError(RuntimeError):
    """The OCR dependencies aren't installed."""


class OCREngine:
    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()

    def _load(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise OCRUnavailableError(
                    "OCR needs the rapidocr_onnxruntime and pypdfium2 packages"
                ) from exc
            logger.info("Loading OCR models")
            self._engine = RapidOCR()
        return self._engine

    def read(self, image: np.ndarray) -> str:
        """Text lines in reading order, dropping low-confidence fragments."""
        # One lock: the engine isn't documented as thread-safe, and OCR is CPU-bound anyway.
        with self._lock:
            result, _ = self._load()(image)
        lines = [text for _, text, score in (result or []) if float(score) >= MIN_CONFIDENCE]
        return "\n".join(lines)

    def read_pdf_page(self, path: Path, index: int) -> str:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise OCRUnavailableError("OCR needs the pypdfium2 package") from exc
        document = pdfium.PdfDocument(str(path))
        try:
            bitmap = document[index].render(scale=RENDER_DPI / 72)
            image = np.asarray(bitmap.to_pil().convert("RGB"))
        finally:
            document.close()
        return self.read(image)

    def read_image(self, path: Path) -> str:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as picture:
                image = np.asarray(picture.convert("RGB"))
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"not a readable image: {exc}") from exc
        return self.read(image)


engine = OCREngine()
