"""Builds the PDF in the evaluation corpus. The companies, products and figures are fictional,
so the model can't answer from memory: every correct answer has to come from retrieval.

    python -m eval.build_corpus
"""

from pathlib import Path

from eval.pdf_writer import write_pdf

CORPUS = Path(__file__).parent / "corpus"

NORTHWIND_PAGES = [
    """Northwind Robotics - Annual Report 2025
Letter from the CEO, Maria Okafor

2025 was our strongest year yet. Revenue reached 612 million USD,
up 23 percent from 498 million USD in 2024. Net income was 41 million USD.
We shipped 3,900 autonomous picking robots to 210 customers.""",
    """Business segments

Warehouse automation contributed 58 percent of revenue and grew 31 percent.
Agricultural drones contributed 27 percent of revenue.
Service contracts contributed the remaining 15 percent.
Our largest customer accounted for 9 percent of revenue.""",
    """People and investment

Headcount was 2,140 employees at year end, up from 1,780 a year earlier.
Research and development spending was 94 million USD, about 15 percent of revenue.
A new manufacturing plant in Monterrey, Mexico will open in the third quarter of 2026.""",
    """Risks and outlook

Our main supply chain risk is dependence on a single lidar supplier, Veltrix,
which provides about 70 percent of our sensors.
Other risks include import tariffs and a shortage of skilled technicians.
For 2026 we expect revenue between 700 and 740 million USD.""",
]


def main() -> None:
    CORPUS.mkdir(exist_ok=True)
    path = write_pdf(CORPUS / "northwind_annual_report_2025.pdf", NORTHWIND_PAGES)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
