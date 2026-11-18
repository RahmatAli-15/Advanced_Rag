"""Generate a synthetic retrieval test set from the corpus.

Protocol (documented so the numbers can be trusted, and reproduced):
1. Sample clauses from versions currently in force: text chunks of 60-400 tokens that
   have paragraph refs, at most `per_document` per document, with a fixed seed.
2. For each clause an LLM writes the question a compliance officer would ask, plus
   Hinglish and Hindi versions, or declares the clause unsuitable (boilerplate).
3. Questions that copy 5+ consecutive words from the clause are rejected: copied
   wording makes keyword search look better than it is on real questions.
4. Items are marked `reviewed: false` until a human checks them.
"""

import logging
import random
import re
import uuid
from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, ContentStatus, Document, DocumentVersion
from app.eval.dataset import GoldItem
from app.llm.gateway import LLMGateway, complete_json
from app.llm.types import LLMRequest, Message

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write evaluation questions for a search engine over Reserve Bank of \
India regulations for NBFCs.

You are given ONE clause. Write the question a compliance officer at an NBFC would \
naturally type to find this clause.

Rules:
- The clause must be the best answer to the question.
- Use your own words. Never copy 5 or more consecutive words from the clause.
- Do not mention paragraph numbers or the document title.
- Ask about the substance (an obligation, limit, deadline, definition, condition).
- Also give the same question in Hinglish (Hindi-English mix in Roman script, as people \
in India type it) and in Hindi (Devanagari script).
- If the clause is boilerplate (titles, commencement, repeal lists, signatures) or has \
no askable substance, set "usable" to false.

Reply with JSON only:
{"usable": true, "question_en": "...", "question_hinglish": "...", "question_hi": "..."}"""


class GeneratedQuestions(BaseModel):
    usable: bool
    question_en: str = ""
    question_hinglish: str = ""
    question_hi: str = ""


@dataclass(frozen=True)
class SampledClause:
    source_key: str
    doc_title: str
    para_refs: list[str]
    context: str
    text: str


def copies_clause(question: str, clause: str, n: int = 5) -> bool:
    """True if the question contains n consecutive words that also appear in the clause."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    clause_norm = " " + " ".join(re.findall(r"[a-z0-9]+", clause.lower())) + " "
    return any(f" {' '.join(words[i : i + n])} " in clause_norm for i in range(len(words) - n + 1))


async def sample_clauses(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    n: int,
    per_document: int,
    seed: int,
) -> list[SampledClause]:
    stmt = (
        select(Document.source_key, Document.title, Chunk.para_refs, Chunk.context, Chunk.text)
        .join(DocumentVersion, DocumentVersion.document_id == Document.id)
        .join(Chunk, Chunk.version_id == DocumentVersion.id)
        .where(
            DocumentVersion.content_status == ContentStatus.CAPTURED,
            DocumentVersion.valid_to.is_(None),
            Chunk.kind == "text",
            Chunk.token_count.between(60, 400),
        )
        .order_by(Document.source_key, Chunk.ordinal)
    )
    async with session_factory() as s:
        rows = [r for r in (await s.execute(stmt)).all() if r.para_refs]

    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    by_doc: dict[str, list[SampledClause]] = {}
    for r in rows:
        by_doc.setdefault(r.source_key, []).append(
            SampledClause(r.source_key, r.title, list(r.para_refs), r.context, r.text)
        )
    picked: list[SampledClause] = []
    for key in sorted(by_doc):
        candidates = by_doc[key]
        picked.extend(rng.sample(candidates, min(per_document, len(candidates))))
    rng.shuffle(picked)
    return picked[:n]


@dataclass
class GenerateStats:
    sampled: int = 0
    unusable: int = 0
    rejected_copying: int = 0
    failed: int = 0
    items: int = 0


async def generate(
    session_factory: async_sessionmaker[AsyncSession],
    gateway: LLMGateway,
    model_label: str,
    *,
    n: int = 60,
    per_document: int = 2,
    seed: int = 42,
) -> tuple[list[GoldItem], GenerateStats]:
    stats = GenerateStats()
    clauses = await sample_clauses(session_factory, n=n, per_document=per_document, seed=seed)
    stats.sampled = len(clauses)
    items: list[GoldItem] = []
    for clause in clauses:
        request = LLMRequest(
            [
                Message("system", SYSTEM_PROMPT),
                Message("user", f"Clause ({clause.context.splitlines()[0]}):\n\n{clause.text}"),
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        try:
            out = await complete_json(gateway, request, GeneratedQuestions.model_validate_json)
        except Exception as e:  # one bad clause must not stop the run
            stats.failed += 1
            log.error("generation failed for %s %s: %s", clause.source_key, clause.para_refs, e)
            continue
        if not out.usable or not out.question_en.strip():
            stats.unusable += 1
            continue
        if copies_clause(out.question_en, clause.text):
            stats.rejected_copying += 1
            continue
        group = uuid.uuid4().hex[:8]
        for language, question in (
            ("en", out.question_en),
            ("hinglish", out.question_hinglish),
            ("hi", out.question_hi),
        ):
            if question.strip():
                items.append(
                    GoldItem(
                        id=f"{group}-{language}",
                        question=question.strip(),
                        language=language,
                        doc_source_key=clause.source_key,
                        doc_title=clause.doc_title,
                        para_refs=clause.para_refs,
                        text_snippet=clause.text[:120],
                        origin=f"synthetic:{model_label}",
                    )
                )
    stats.items = len(items)
    return items, stats
