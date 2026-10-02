"""Episodic memory: remember what the user said, recall it when it's relevant.

Write path (after each turn): skip pleasantries -> embed the user's message -> store.
Read path (before each turn): embed the question -> vector + keyword search over
other sessions -> keep hits that clear the similarity bar -> format as context.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from core.memory.embeddings import Embedder, EmbeddingUnavailable
from core.memory.episodes import Episode, EpisodeStore, SearchFilter

_PLEASANTRIES = frozenset(
    ["hi", "hello", "hey", "yo", "thanks", "thank", "you", "thx", "ok", "okay", "k", "cool", "great", "nice", "sure", "yes", "no", "yep", "nope", "bye", "goodbye", "good", "morning", "night", "evening", "afternoon", "lol", "haha", "hmm", "got", "it", "alright", "fine"]
)


def is_trivial(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return not words or all(w in _PLEASANTRIES for w in words)


@dataclass
class RecalledMemory:
    episode: Episode
    score: float


@dataclass
class Recall:
    context: str | None = None
    memories: list[RecalledMemory] = field(default_factory=list)
    warning: str | None = None


def _local_now() -> datetime:
    return datetime.now().astimezone()


class EpisodicMemory:
    def __init__(
        self,
        store: EpisodeStore,
        embedder: Embedder,
        *,
        max_items: int = 6,
        keyword_weight: float = 0.15,
        keyword_slack: float = 0.08,
        clock: Callable[[], datetime] = _local_now,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.max_items = max_items
        # A keyword match adds to the score and lowers the similarity bar a little:
        # embeddings are weak on names and numbers, keywords are strong on them.
        self.keyword_weight = keyword_weight
        self.keyword_slack = keyword_slack
        self.clock = clock

    async def recall(self, session_id: uuid.UUID, user_text: str, *, include_private: bool) -> Recall:
        warning = None
        try:
            query_vec: list[float] | None = (await self.embedder.embed([user_text], "query"))[0]
        except EmbeddingUnavailable as e:
            query_vec, warning = None, f"memory search is keyword-only: {e}"

        hits = await self.store.search(
            user_text,
            query_vec,
            self.embedder.model_id,
            SearchFilter(exclude_session=session_id, include_private=include_private),
            limit=self.max_items * 4,
        )
        bar = self.embedder.min_similarity
        kept: list[RecalledMemory] = []
        for h in hits:
            if query_vec is None:
                relevant = h.keyword >= 0.5
            else:
                relevant = h.similarity >= bar or (h.keyword > 0 and h.similarity >= bar - self.keyword_slack)
            if relevant:
                kept.append(RecalledMemory(h.episode, h.similarity + self.keyword_weight * h.keyword))
        kept.sort(key=lambda m: m.score, reverse=True)
        kept = kept[: self.max_items]
        return Recall(context=self._format(kept), memories=kept, warning=warning)

    async def record_turn(
        self, session_id: uuid.UUID, user_text: str, reply_text: str, *, private: bool
    ) -> Episode | None:
        if is_trivial(user_text):
            return None
        embedding, model = None, None
        try:
            embedding = (await self.embedder.embed([user_text], "document"))[0]
            model = self.embedder.model_id
        except EmbeddingUnavailable:
            # Still worth keeping: keyword search finds it, and vectors can be backfilled.
            pass
        episode = Episode(
            session_id=session_id,
            text=user_text,
            reply=reply_text or None,
            private=private,
            embedding=embedding,
            embedding_model=model,
        )
        await self.store.add(episode)
        return episode

    def _format(self, memories: list[RecalledMemory]) -> str:
        now = self.clock()
        # Labelled as background: small models otherwise read the date out in every answer.
        lines = [f"(Background, mention only if relevant) Now: {now:%A, %d %B %Y, %H:%M}."]
        if memories:
            lines.append(
                "Things the user told you in earlier conversations (oldest first; if two "
                "conflict, the newer one is current):"
            )
            for m in sorted(memories, key=lambda m: m.episode.occurred_at):
                when = m.episode.occurred_at.astimezone(now.tzinfo)
                lines.append(f"- [{when:%Y-%m-%d %H:%M}] User: {m.episode.text}")
        else:
            lines.append("No earlier conversations seem relevant to this message.")
        return "\n".join(lines)
