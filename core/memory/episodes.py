"""Episodic memory storage: what the user said, when, in which session.

Two backends share one interface: InMemoryEpisodeStore (dev, tests, evals) and
PostgresEpisodeStore (pgvector + full-text search).
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field

from core.memory.embeddings import cosine


class Episode(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    text: str
    # The assistant's reply, kept for display; retrieval only looks at `text`.
    reply: str | None = None
    # Said in a private turn: only ever shown to local models.
    private: bool = False
    importance: float = 0.5
    embedding: list[float] | None = Field(default=None, exclude=True)
    embedding_model: str | None = None


class Hit(BaseModel):
    episode: Episode
    similarity: float  # cosine similarity, 0 when the episode has no comparable vector
    keyword: float  # keyword match strength normalised to 0..1


class SearchFilter(BaseModel):
    exclude_session: uuid.UUID | None = None
    include_private: bool = False


class EpisodeStore(Protocol):
    async def add(self, episode: Episode) -> None: ...

    async def search(
        self,
        query: str,
        embedding: list[float] | None,
        embedding_model: str | None,
        flt: SearchFilter,
        limit: int,
    ) -> list[Hit]: ...

    async def recent(self, limit: int) -> list[Episode]: ...

    async def delete(self, episode_id: uuid.UUID) -> bool: ...


_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "then", "there", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your", "i'm", "im", "it's", "what's", "whats", "about", "am"]
)


def keyword_terms(text: str) -> list[str]:
    terms = []
    for word in re.findall(r"[a-z0-9']+", text.lower()):
        word = word.strip("'").removesuffix("'s")
        if not word or word in _STOPWORDS:
            continue
        # Crude stemming so "name"/"named"/"names" and "job"/"jobs" match.
        for suffix in ("ing", "ed", "s"):
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        if len(word) > 3 and word.endswith("e"):
            word = word[:-1]
        terms.append(word)
    return terms


class InMemoryEpisodeStore:
    def __init__(self) -> None:
        self._episodes: dict[uuid.UUID, Episode] = {}

    async def add(self, episode: Episode) -> None:
        self._episodes[episode.id] = episode.model_copy(deep=True)

    async def search(
        self,
        query: str,
        embedding: list[float] | None,
        embedding_model: str | None,
        flt: SearchFilter,
        limit: int,
    ) -> list[Hit]:
        pool = [
            e
            for e in self._episodes.values()
            if e.session_id != flt.exclude_session and (flt.include_private or not e.private)
        ]
        if not pool:
            return []
        keyword = _bm25(keyword_terms(query), pool)
        hits = []
        for e in pool:
            sim = 0.0
            if embedding is not None and e.embedding is not None and e.embedding_model == embedding_model:
                sim = cosine(embedding, e.embedding)
            hits.append(Hit(episode=e, similarity=sim, keyword=keyword.get(e.id, 0.0)))
        # Candidates are the best by meaning plus the best by exact words (names,
        # numbers), so a strong keyword match isn't cut off by the similarity sort.
        by_sim = sorted(hits, key=lambda h: h.similarity, reverse=True)[:limit]
        by_kw = [h for h in sorted(hits, key=lambda h: h.keyword, reverse=True)[:limit] if h.keyword]
        merged: dict[uuid.UUID, Hit] = {}
        for h in by_sim + by_kw:
            merged.setdefault(h.episode.id, h)
        return list(merged.values())

    async def recent(self, limit: int) -> list[Episode]:
        return sorted(self._episodes.values(), key=lambda e: e.occurred_at, reverse=True)[:limit]

    async def delete(self, episode_id: uuid.UUID) -> bool:
        return self._episodes.pop(episode_id, None) is not None


def _bm25(terms: list[str], docs: list[Episode], k1: float = 1.2, b: float = 0.75) -> dict[uuid.UUID, float]:
    if not terms:
        return {}
    tokenized = {d.id: keyword_terms(d.text) for d in docs}
    avg_len = sum(len(t) for t in tokenized.values()) / len(docs) or 1.0
    df = Counter(term for toks in tokenized.values() for term in set(toks))
    n = len(docs)
    scores: dict[uuid.UUID, float] = {}
    for doc_id, toks in tokenized.items():
        tf = Counter(toks)
        score = 0.0
        for term in set(terms):
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(toks) / avg_len))
        if score:
            scores[doc_id] = score
    top = max(scores.values(), default=0.0)
    return {k: v / top for k, v in scores.items()} if top else {}
