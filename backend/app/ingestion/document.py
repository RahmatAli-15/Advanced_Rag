"""Source-neutral document model.

Every source parser (RBI today; SEBI, GST later) turns its pages into these types,
so chunking, embedding and retrieval never depend on a particular website's HTML.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

BlockKind = Literal["heading", "paragraph", "table"]
AmendmentAction = Literal["inserted", "substituted", "amended", "deleted", "note"]


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str
    footnote_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Footnote:
    """A footnote. Most record an amendment ('Inserted w.e.f. <date> vide <circular>');
    the rest (action='note') are plain explanatory notes."""

    id: str
    text: str
    action: AmendmentAction
    effective_from: date | None
    amending_title: str | None
    amending_url: str | None
