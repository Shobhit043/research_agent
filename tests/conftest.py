from pathlib import Path

import pytest

from eval.pdf_writer import write_pdf


@pytest.fixture
def make_pdf(tmp_path):
    def _make(name: str, pages: list[str]) -> Path:
        return write_pdf(tmp_path / name, pages)

    return _make
