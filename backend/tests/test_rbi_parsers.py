from datetime import date
from pathlib import Path

import pytest

from app.ingestion.encoding import decode_html
from app.ingestion.sources import rbi

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return decode_html((FIXTURES / name).read_bytes())


def test_split_title_extracts_version_date() -> None:
    title, updated = rbi.split_title("X Directions, 2025 (Updated as on October 09, 2026) ")
    assert title == "X Directions, 2025"
    assert updated == date(2026, 10, 9)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("November 28, 2025", date(2025, 11, 28)),
        ("Nov 28, 2025", date(2025, 11, 28)),
        ("Sept. 5, 2024", None),  # unknown abbreviation: refuse rather than guess
        ("not a date", None),
    ],
)
def test_parse_date(text: str, expected: date | None) -> None:
    assert rbi.parse_date(text) == expected


def test_parse_listing_tracks_category_and_issue_date() -> None:
    items = rbi.parse_listing(_load("rbi_listing.html"))

    assert [i.source_id for i in items] == ["12931", "12932", "11322"]  # duplicate dropped
    first = items[0]
    assert first.title == (
        "Reserve Bank of India (Non-Banking Financial Companies – Miscellaneous) Directions, 2025"
    )
    assert first.updated_as_on == date(2026, 2, 26)
    assert first.category == "Non-Banking Financial Companies"
    assert first.issued_on == date(2025, 11, 28)
    assert items[1].updated_as_on is None
    assert items[2].category == "Commercial Banks"
    assert items[2].issued_on == date(2018, 7, 3)
    assert items[2].url == rbi.detail_url("11322")


def test_parse_detail_metadata() -> None:
    page = rbi.parse_detail(_load("rbi_detail.html"))

    assert page.title.endswith("Statutory Liquidity Ratio) Directions, 2025")
    assert page.updated_as_on == date(2026, 10, 9)
    assert page.issued_on == date(2025, 11, 28)
    assert page.ref_numbers == ["RBI/DOR/2025-26/150", "DOR.RET.REC.69/12-01-001/2025-26"]
    assert page.annex_links == ["https://rbidocs.rbi.org.in/rdocs/content/pdfs/150MD_A1.pdf"]


def test_parse_detail_blocks_keep_structure_and_skip_noise() -> None:
    blocks = rbi.parse_detail(_load("rbi_detail.html")).blocks

    assert [(b.kind, b.text[:20]) for b in blocks] == [
        ("paragraph", "In exercise of the p"),
        ("heading", "Chapter I - Prelimin"),
        ("heading", "A. Short Title and C"),
        ("paragraph", "1. These Directions "),
        ("paragraph", "2. These Directions "),  # &nbsp; normalised
        ("table", "Item | Ratio\nCRR | 3"),  # cell <p> not duplicated as a paragraph
        ("paragraph", "(i) Treasury Bills o"),  # <p> inside <li> captured once
    ]
    assert all("Table of Contents" not in b.text for b in blocks)
    assert all("Footer" not in b.text for b in blocks)


def test_parse_detail_fails_loudly_on_unknown_layout() -> None:
    with pytest.raises(rbi.PageLayoutChanged):
        rbi.parse_detail("<html><body><p>redesigned site</p></body></html>")


def test_parse_versions_sorted_and_case_insensitive() -> None:
    versions = rbi.parse_versions(_load("rbi_versions.html"))
    assert versions == [
        rbi.VersionEntry("6", date(2025, 11, 28)),
        rbi.VersionEntry("7", date(2025, 12, 11)),
        rbi.VersionEntry("119", date(2026, 1, 22)),
    ]


def test_decode_html_keeps_utf8_and_repairs_cp1252_bytes() -> None:
    assert decode_html("Banks – CRR".encode()) == "Banks – CRR"
    assert decode_html(b"Banks \x96 CRR") == "Banks – CRR"
