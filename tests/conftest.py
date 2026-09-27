from pathlib import Path

import pytest

from eval.pdf_writer import write_pdf


@pytest.fixture
def make_pdf(tmp_path):
    def _make(name: str, pages: list[str]) -> Path:
        return write_pdf(tmp_path / name, pages)

    return _make


class FakeOCR:
    """Stands in for RapidOCR: returns canned text and records what it was asked to read."""

    def __init__(self):
        self.text = ""
        self.pages: list[int] = []
        self.images: list[str] = []

    def read_pdf_page(self, path, index):
        self.pages.append(index)
        return self.text

    def read_image(self, path):
        self.images.append(Path(path).name)
        return self.text


@pytest.fixture(autouse=True)
def fake_ocr(request, monkeypatch):
    """Unit tests never load the OCR models; tests marked `real_ocr` use the real engine."""
    if request.node.get_closest_marker("real_ocr"):
        return None
    fake = FakeOCR()
    monkeypatch.setattr("agent.documents.loader.ocr_engine", fake)
    return fake
