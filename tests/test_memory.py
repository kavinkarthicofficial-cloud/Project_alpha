from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from core.memory.embeddings import EmbeddingUnavailable, HashEmbedder
from core.memory.engine import EpisodicMemory, is_trivial
from core.memory.episodes import InMemoryEpisodeStore, keyword_terms

OLD_SESSION, NEW_SESSION = uuid.uuid4(), uuid.uuid4()


class DownEmbedder(HashEmbedder):
    async def embed(self, texts, kind):
        raise EmbeddingUnavailable("ollama is not running")


def memory(embedder=None) -> EpisodicMemory:
    fixed = datetime(2026, 10, 2, 9, 30, tzinfo=UTC)
    return EpisodicMemory(InMemoryEpisodeStore(), embedder or HashEmbedder(), clock=lambda: fixed)


def test_keyword_terms_match_word_forms():
    assert keyword_terms("we named him Biscuit") == ["nam", "biscuit"]
    assert keyword_terms("What's my dog's name?") == ["dog", "nam"]
    assert keyword_terms("switched jobs") == ["switch", "job"]


def test_pleasantries_are_not_remembered():
    assert is_trivial("Thanks!") and is_trivial("ok cool") and is_trivial("  ")
    assert not is_trivial("I'm vegetarian")


async def test_recall_finds_what_was_said_in_another_session():
    mem = memory()
    await mem.record_turn(OLD_SESSION, "My dog is called Biscuit", "Lovely name!", private=False)
    await mem.record_turn(OLD_SESSION, "I work at Zoho as an engineer", "Nice.", private=False)

    recall = await mem.recall(NEW_SESSION, "what is my dog called", include_private=False)

    assert recall.memories[0].episode.text == "My dog is called Biscuit"
    assert "Friday, 02 October 2026" in recall.context and "Background" in recall.context
    assert "User: My dog is called Biscuit" in recall.context


async def test_recall_skips_the_current_session_and_irrelevant_memories():
    mem = memory()
    await mem.record_turn(NEW_SESSION, "My dog is called Biscuit", "", private=False)
    await mem.record_turn(OLD_SESSION, "I enjoy hiking on weekends", "", private=False)

    recall = await mem.recall(NEW_SESSION, "what is my dog called", include_private=False)

    assert recall.memories == []
    assert "No earlier conversations seem relevant" in recall.context


async def test_private_memories_only_reach_private_turns():
    mem = memory()
    await mem.record_turn(OLD_SESSION, "My bank is HDFC", "", private=True)

    public = await mem.recall(NEW_SESSION, "which bank is mine", include_private=False)
    private = await mem.recall(NEW_SESSION, "which bank is mine", include_private=True)

    assert public.memories == [] and len(private.memories) == 1


async def test_context_lists_memories_oldest_first():
    mem = memory()
    store = mem.store
    await mem.record_turn(OLD_SESSION, "I work at Infosys", "", private=False)
    await mem.record_turn(OLD_SESSION, "I now work at Zoho", "", private=False)
    # make the first one clearly older
    first = (await store.recent(10))[-1]
    first.occurred_at -= timedelta(days=30)
    await store.add(first)

    recall = await mem.recall(NEW_SESSION, "where do I work", include_private=False)

    assert recall.context.index("Infosys") < recall.context.index("Zoho")


async def test_embedder_down_falls_back_to_keywords():
    mem = memory(DownEmbedder())
    stored = await mem.record_turn(OLD_SESSION, "My dog is called Biscuit", "", private=False)
    assert stored is not None and stored.embedding is None

    recall = await mem.recall(NEW_SESSION, "Biscuit the dog", include_private=False)

    assert "keyword-only" in recall.warning
    assert [m.episode.text for m in recall.memories] == ["My dog is called Biscuit"]


async def test_delete_forgets():
    mem = memory()
    ep = await mem.record_turn(OLD_SESSION, "My dog is called Biscuit", "", private=False)
    assert await mem.store.delete(ep.id)
    assert (await mem.recall(NEW_SESSION, "dog called", include_private=False)).memories == []
