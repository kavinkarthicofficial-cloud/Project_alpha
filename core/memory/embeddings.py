"""Text embeddings for memory retrieval.

The embedding model is the one part of the system that must NOT change casually:
vectors from different models can't be compared. Every stored vector records which
model produced it, and searches only compare vectors from the same model.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Literal, Protocol

import httpx

Kind = Literal["query", "document"]


class EmbeddingUnavailable(Exception):
    pass


class Embedder(Protocol):
    model_id: str  # stored next to every vector, e.g. "ollama:nomic-embed-text"
    min_similarity: float

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        min_similarity: float = 0.58,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.model = model
        self.model_id = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.min_similarity = min_similarity
        self.prefixes = {"query": query_prefix, "document": document_prefix}

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        prefix = self.prefixes[kind]
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": [prefix + t for t in texts]},
                )
        except httpx.TransportError as e:
            raise EmbeddingUnavailable(f"cannot reach Ollama: {e}") from e
        if resp.status_code != 200:
            raise EmbeddingUnavailable(f"ollama embed {resp.status_code}: {resp.text[:200]}")
        return resp.json()["embeddings"]


class OpenAIEmbedder:
    def __init__(self, model: str, min_similarity: float = 0.3) -> None:
        self.model = model
        self.model_id = f"openai:{model}"
        self.min_similarity = min_similarity
        self._client: Any = None

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        import openai

        try:
            if self._client is None:
                self._client = openai.AsyncOpenAI()
            resp = await self._client.embeddings.create(model=self.model, input=texts)
        except openai.OpenAIError as e:
            raise EmbeddingUnavailable(str(e)) from e
        return [d.embedding for d in resp.data]


class HashEmbedder:
    """Deterministic bag-of-words vectors. For tests only: similarity = word overlap."""

    def __init__(self, dims: int = 256, min_similarity: float = 0.2) -> None:
        self.dims = dims
        self.model_id = f"hash:{dims}"
        self.min_similarity = min_similarity

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
            out.append(vec)
        return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm if norm else 0.0


def build_embedder(cfg: dict[str, Any], ollama_base_url: str | None = None) -> Embedder:
    provider = cfg["provider"]
    if provider == "ollama":
        return OllamaEmbedder(
            cfg["model"],
            base_url=ollama_base_url or cfg.get("base_url", "http://localhost:11434"),
            min_similarity=cfg.get("min_similarity", 0.58),
            query_prefix=cfg.get("query_prefix", ""),
            document_prefix=cfg.get("document_prefix", ""),
        )
    if provider == "openai":
        return OpenAIEmbedder(cfg["model"], min_similarity=cfg.get("min_similarity", 0.3))
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embedding provider {provider!r}")
