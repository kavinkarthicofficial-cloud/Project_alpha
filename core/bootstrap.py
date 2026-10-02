from __future__ import annotations

from dataclasses import dataclass

import yaml

from core.agent.orchestrator import Memory, NoMemory, Orchestrator
from core.config import Settings
from core.llm.registry import ModelRegistry, ModelRouter
from core.memory.embeddings import build_embedder
from core.memory.engine import EpisodicMemory
from core.memory.episodes import EpisodeStore, InMemoryEpisodeStore
from core.storage.base import ConversationStore, InMemoryStore
from core.tools.builtin import default_tools


@dataclass
class Stores:
    conversations: ConversationStore
    episodes: EpisodeStore


def build_stores(settings: Settings, *, in_memory: bool = False) -> Stores:
    if settings.database_url and not in_memory:
        from sqlalchemy.ext.asyncio import create_async_engine

        from core.memory.postgres_store import PostgresEpisodeStore
        from core.storage.postgres import PostgresStore

        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        return Stores(PostgresStore(engine), PostgresEpisodeStore(engine))
    return Stores(InMemoryStore(), InMemoryEpisodeStore())


def build_memory(settings: Settings, episodes: EpisodeStore) -> Memory:
    config = yaml.safe_load(settings.models_config.read_text())
    if not config.get("embedding"):
        return NoMemory()
    return EpisodicMemory(episodes, build_embedder(config["embedding"], settings.ollama_base_url))


@dataclass
class Services:
    orchestrator: Orchestrator
    episodes: EpisodeStore


def build_services(
    settings: Settings, *, in_memory: bool = False, memory_enabled: bool = True
) -> Services:
    """in_memory: ignore DATABASE_URL (evals use a fresh, isolated store per case)."""
    registry = ModelRegistry.from_yaml(settings.models_config, settings.ollama_base_url)
    stores = build_stores(settings, in_memory=in_memory)
    orchestrator = Orchestrator(
        stores.conversations,
        ModelRouter(registry),
        default_tools(),
        assistant_name=settings.assistant_name,
        memory=build_memory(settings, stores.episodes) if memory_enabled else NoMemory(),
        max_steps=settings.max_agent_steps,
    )
    return Services(orchestrator, stores.episodes)


def build_orchestrator(settings: Settings, **kw: bool) -> Orchestrator:
    return build_services(settings, **kw).orchestrator
