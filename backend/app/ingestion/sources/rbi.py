"""Parsers for RBI Master Direction pages.

Three page types:
  listing   /Scripts/BS_ViewMasterDirections.aspx      every Master Direction, grouped
                                                        by category and issue date
  detail    /Scripts/BS_ViewMasDirections.aspx?id=N    full text of the current version
  versions  /Scripts/NotificationPreVersion.aspx?id=N  dates of every published version

Parsers are pure functions (HTML in, dataclasses out) so they are fast to test
against saved fixtures and independent of the network.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from urllib.parse import parse_qs, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

from app.ingestion.document import AmendmentAction, Block, BlockKind, Footnote

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


@dataclass(frozen=True)
class DetailPage:
    title: str
    updated_as_on: date | None
    ref_numbers: list[str]
    issued_on: date | None
    annex_links: list[str]
    # sha256 of the page's normalised visible text. Independent of how we split it
    # into blocks, so improving the parser never looks like a regulatory change.
    fingerprint: str
    blocks: list[Block] = field(default_factory=list)
    footnotes: dict[str, Footnote] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


# Footnotes come in many shapes, e.g.
#   "Inserted with effect from July 1, 2026, vide <circular> dated June 30, 2026."
#   "Substituted w.e.f. February 13, 2026, vide <circular>"
#   "Deleted vide the <circular>"            (no date)
#   "Amended with effect from April 01, 2026, pursuant to the <circular>"
#   "Modified w.e.f January 01, 2026."       (no source)
#   "Ibid."                                   (same as previous footnote)
# so each part is found separately instead of with one brittle pattern.
_DATE_PATTERN = r"[A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}"
_ACTION_RE = re.compile(
    r"^(Inserted|Added|Substituted|Modified|Amended|Revised|Deleted|Omitted)\b", re.I
)
_EFFECTIVE_RE = re.compile(
    rf"(?:with\s+effect\s+from|w\.?\s*e\.?\s*f\.?(?:\s+from)?)\s*,?\s*({_DATE_PATTERN})", re.I
)
_SOURCE_RE = re.compile(
    rf"\b(?:vide|consequent\s+to|pursuant\s+to)\s+(?:the\s+)?(.+?)"
    rf"(?:,?\s+dated\s+{_DATE_PATTERN})?\s*[.,]?\s*$",
    re.I | re.S,
)
_ACTIONS: dict[str, AmendmentAction] = {
    "inserted": "inserted",
    "added": "inserted",
    "substituted": "substituted",
    "modified": "amended",
    "amended": "amended",
    "revised": "amended",
    "deleted": "deleted",
    "omitted": "deleted",
}


def parse_footnote(
    fid: str, text: str, url: str | None, previous: Footnote | None = None
) -> Footnote:
    if text.lower().startswith("ibid") and previous is not None:
        # "Ibid." = same source as the previous footnote
        return Footnote(
            fid,
            text,
            previous.action,
            previous.effective_from,
            previous.amending_title,
            previous.amending_url,
        )
    action = _ACTION_RE.match(text)
    if not action:
        return Footnote(fid, text, "note", None, None, url)
    effective = _EFFECTIVE_RE.search(text)
    source = _SOURCE_RE.search(text)
    return Footnote(
        id=fid,
        text=text,
        action=_ACTIONS[action.group(1).lower()],
        effective_from=parse_date(effective.group(1)) if effective else None,
        amending_title=_clean(source.group(1)).rstrip(" ,.") if source else None,
        amending_url=url,
    )


def _marker_links(el: Tag) -> list[Tag]:
    return [
        a
        for sup in el.find_all("sup")
        for a in sup.find_all("a", href=True)
        if str(a["href"]).startswith("#")
    ]


def _footnote_refs(el: Tag) -> tuple[str, ...]:
    refs = [str(a["href"])[1:] for a in _marker_links(el)]
    return tuple(dict.fromkeys(refs))  # dedupe, keep order


def _is_footnote_paragraph(p: Tag) -> bool:
    # Seen both as <p class="footnote"> and <p><span class="footnote">.
    return "footnote" in (p.get("class") or []) or p.find(class_="footnote") is not None


def _text_without_markers(el: Tag) -> str:
    """Element text minus <sup> footnote numbers, which otherwise leak into sentences."""
    parts = [str(s) for s in el.find_all(string=True) if s.find_parent("sup") is None]
    return _clean(" ".join(parts))


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
        cells = [_text_without_markers(c) for c in tr.find_all(["td", "th"])]
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

    fingerprint = hashlib.sha256(_clean(cell.get_text(" ")).encode()).hexdigest()

    footnotes: dict[str, Footnote] = {}
    blocks: list[Block] = []
    for el in title_p.find_all_next(["p", "table", "li"]):
        if not isinstance(el, Tag) or not _inside(el, cell):
            continue
        # Only top-level elements of the content cell; anything inside a data
        # table is captured as part of that table's block.
        if el.find_parent("table") is not cell:
            continue
        classes: list[str] = list(el.get("class") or [])
        if el.name == "p" and _is_footnote_paragraph(el):
            anchor = el.find("a", id=True)
            if isinstance(anchor, Tag):
                fid = str(anchor["id"])
                link = next(
                    (
                        urljoin(BASE_URL, str(a["href"]))
                        for a in el.find_all("a", href=True)
                        if not str(a["href"]).startswith("#")
                    ),
                    None,
                )
                previous = next(reversed(footnotes.values()), None)
                footnotes[fid] = parse_footnote(fid, _text_without_markers(el), link, previous)
            continue
        if el.name == "table":
            if "table of contents" in el.get_text(" ").lower()[:200]:
                continue
            if text := _table_text(el):
                blocks.append(Block("table", text, _footnote_refs(el)))
            continue
        if el.name == "p" and el.find_parent("li") is not None:
            continue  # the enclosing <li> already holds this text
        text = _text_without_markers(el)
        if not text:
            continue
        kind: BlockKind = "heading" if "head" in classes else "paragraph"
        blocks.append(Block(kind, text, _footnote_refs(el)))

    # Fallback: each inline marker repeats its footnote text in a tooltip title.
    for a in _marker_links(cell):
        fid = str(a["href"])[1:]
        sup = a.find_parent("sup")
        tooltip = _clean(str(sup.get("title", ""))) if sup is not None else ""
        if fid not in footnotes and tooltip:
            footnotes[fid] = parse_footnote(fid, tooltip, None)

    return DetailPage(
        title=title,
        updated_as_on=updated,
        ref_numbers=ref_numbers,
        issued_on=issued_on,
        annex_links=annex_links,
        fingerprint=fingerprint,
        blocks=blocks,
        footnotes=footnotes,
    )


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
