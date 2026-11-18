"""Retrieval golden dataset.

Labels are *stable identifiers* (document source key + paragraph refs + a text snippet),
never chunk UUIDs, so the dataset survives re-chunking and re-embedding.
"""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["en", "hinglish", "hi"]


class GoldItem(BaseModel):
    id: str
    question: str
    language: Language
    doc_source_key: str
    doc_title: str
    para_refs: list[str] = Field(min_length=1)
    text_snippet: str  # start of the gold chunk text; used when refs are ambiguous
    as_of: date | None = None
    origin: str  # e.g. "synthetic:openai/gpt-oss-120b"
    reviewed: bool = False  # set to true only once a HUMAN has checked the item
    # Review against the gold clause (who did it is recorded, so an AI review is never
    # mistaken for a human one).
    review: Literal["ok", "fixed"] | None = None
    reviewed_by: str | None = None  # e.g. "ai:claude" or a person's name
    review_note: str | None = None
    original_question: str | None = None  # set when the review rewrote the question


def load(path: Path) -> list[GoldItem]:
    return [
        GoldItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save(path: Path, items: list[GoldItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")
