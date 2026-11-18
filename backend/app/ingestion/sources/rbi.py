"""Parsers for RBI Master Direction pages.

Three page types:
  listing   /Scripts/BS_ViewMasterDirections.aspx      every Master Direction, grouped
                                                        by category and issue date
  detail    /Scripts/BS_ViewMasDirections.aspx?id=N    full text of the current version
  versions  /Scripts/NotificationPreVersion.aspx?id=N  dates of every published version

Parsers are pure functions (HTML in, dataclasses out) so they are fast to test
against saved fixtures and independent of the network.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal
from urllib.parse import parse_qs, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

BASE_URL = "https://www.rbi.org.in/Scripts/"
LISTING_URL = urljoin(BASE_URL, "BS_ViewMasterDirections.aspx")

_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")
_UPDATED_RE = re.compile(
    r"\(?\s*Updated\s+as\s+on\s+([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})\s*\)?", re.I
)
_REF_RE = re.compile(r"^(RBI/[\w./-]+|[A-Z]{2,}[\w.()/-]*\d[\w.()/-]*)$")


def detail_url(source_id: str) -> str:
    return urljoin(BASE_URL, f"BS_ViewMasDirections.aspx?id={source_id}")


def versions_url(source_id: str) -> str:
    return urljoin(BASE_URL, f"NotificationPreVersion.aspx?id={source_id}")


def parse_date(text: str) -> date | None:
    cleaned = re.sub(r"\s+", " ", text.replace(".", "")).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def split_title(raw_title: str) -> tuple[str, date | None]:
    """Split off the version date.

    'X Directions, 2025 (Updated as on October 09, 2026)' -> ('X Directions, 2025', 2026-10-09)
    """
    title = _clean(raw_title)
    m = _UPDATED_RE.search(title)
    if not m:
        return title, None
    return _clean(title[: m.start()]), parse_date(m.group(1))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# --------------------------------------------------------------------------- listing


@dataclass(frozen=True)
class ListingItem:
    source_id: str
    title: str
    updated_as_on: date | None
    category: str | None
    issued_on: date | None
    url: str


def parse_listing(html: str) -> list[ListingItem]:
    items: list[ListingItem] = []
    seen: set[str] = set()
    category: str | None = None
    issued_on: date | None = None

    for row in _soup(html).find_all("tr"):
        header = row.find("td", class_="tableheader")
        if header is not None:
            text = _clean(header.get_text(" "))
            parsed = parse_date(text)
            if parsed:
                issued_on = parsed
            elif text:
                category, issued_on = text, None
            continue

        link = row.find("a", href=re.compile(r"BS_ViewMasDirections\.aspx\?id=\d+", re.I))
        if not isinstance(link, Tag):
            continue
        source_id = parse_qs(urlsplit(str(link["href"]).strip()).query)["id"][0]
        if source_id in seen:
            continue
        seen.add(source_id)
        title, updated = split_title(link.get_text(" "))
        items.append(
            ListingItem(source_id, title, updated, category, issued_on, detail_url(source_id))
        )
    return items


# --------------------------------------------------------------------------- detail

BlockKind = Literal["heading", "paragraph", "table"]


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str


@dataclass(frozen=True)
class DetailPage:
    title: str
    updated_as_on: date | None
    ref_numbers: list[str]
    issued_on: date | None
    annex_links: list[str]
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


class PageLayoutChanged(ValueError):
    """The page no longer matches the expected structure. Fail loudly, never guess."""


def _content_cell(soup: BeautifulSoup) -> Tag:
    for table in soup.find_all("table", class_="td"):
        if table.find("p", class_="head"):
            return table
    raise PageLayoutChanged("content table with a title paragraph not found")


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_detail(html: str) -> DetailPage:
    cell = _content_cell(_soup(html))

    title_p = cell.find("p", class_="head")
    if not isinstance(title_p, Tag):
        raise PageLayoutChanged("title paragraph not found")
    title, updated = split_title(title_p.get_text(" "))

    # The paragraphs above the title hold the reference numbers and issue date.
    preamble = [p for p in title_p.find_all_previous("p") if _inside(p, cell)]
    preamble.reverse()  # find_all_previous walks backwards; restore document order
    ref_numbers: list[str] = []
    issued_on: date | None = None
    for p in preamble:
        if issued_on is None:
            issued_on = parse_date(_clean(p.get_text(" ")))
        for part in p.get_text("\n").split("\n"):
            if _REF_RE.match(part := _clean(part)):
                ref_numbers.append(part)

    annex_links = sorted(
        {
            urljoin(BASE_URL, str(a["href"]).strip())
            for a in cell.find_all("a", href=True)
            if str(a["href"]).lower().endswith(".pdf")
        }
    )

    blocks: list[Block] = []
    for el in title_p.find_all_next(["p", "table", "li"]):
        if not isinstance(el, Tag) or not _inside(el, cell):
            continue
        # Only top-level elements of the content cell; anything inside a data
        # table is captured as part of that table's block.
        if el.find_parent("table") is not cell:
            continue
        if el.name == "table":
            if "table of contents" in el.get_text(" ").lower()[:200]:
                continue
            if text := _table_text(el):
                blocks.append(Block("table", text))
            continue
        if el.name == "p" and el.find_parent("li") is not None:
            continue  # the enclosing <li> already holds this text
        text = _clean(el.get_text(" "))
        if not text:
            continue
        kind: BlockKind = "heading" if "head" in (el.get("class") or []) else "paragraph"
        blocks.append(Block(kind, text))

    return DetailPage(title, updated, ref_numbers, issued_on, annex_links, blocks)


def _inside(el: Tag | None, ancestor: Tag) -> bool:
    return el is not None and any(p is ancestor for p in el.parents)


# --------------------------------------------------------------------------- versions


@dataclass(frozen=True)
class VersionEntry:
    hist_id: str
    updated_as_on: date


def parse_versions(html: str) -> list[VersionEntry]:
    entries: dict[str, VersionEntry] = {}
    for a in _soup(html).find_all(
        "a", href=re.compile(r"NotificationPreVersion\.aspx\?.*Histid=", re.I)
    ):
        query = parse_qs(urlsplit(str(a["href"])).query)
        hist = next((v[0] for k, v in query.items() if k.lower() == "histid"), None)
        m = _UPDATED_RE.search(a.get_text(" "))
        d = parse_date(m.group(1)) if m else None
        if hist and d:
            entries[hist] = VersionEntry(hist, d)
    return sorted(entries.values(), key=lambda e: e.updated_as_on)
