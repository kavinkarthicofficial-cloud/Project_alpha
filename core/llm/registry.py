from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.llm.anthropic_provider import AnthropicProvider
from core.llm.ollama_provider import OllamaProvider
from core.llm.openai_provider import OpenAIProvider
from core.llm.types import LLMProvider, ModelProfile


class UnknownModel(KeyError):
    pass


class ModelRegistry:
    def __init__(
        self,
        profiles: dict[str, ModelProfile],
        providers: dict[str, LLMProvider],
        routes: dict[str, str],
        fallback_order: list[str],
    ) -> None:
        for alias in [*routes.values(), *fallback_order]:
            if alias not in profiles:
                raise ValueError(f"models.yaml references unknown model alias {alias!r}")
        for p in profiles.values():
            if p.provider not in providers:
                raise ValueError(f"model {p.alias!r} uses unknown provider {p.provider!r}")
        self.profiles = profiles
        self.providers = providers
        self.routes = routes
        self.fallback_order = fallback_order

    @classmethod
    def from_yaml(cls, path: Path, ollama_base_url: str | None = None) -> ModelRegistry:
        data: dict[str, Any] = yaml.safe_load(path.read_text())
        provider_cfg = data.get("providers", {})
        providers: dict[str, LLMProvider] = {}
        if "anthropic" in provider_cfg:
            providers["anthropic"] = AnthropicProvider()
        if "openai" in provider_cfg:
            providers["openai"] = OpenAIProvider()
        if "ollama" in provider_cfg:
            base = ollama_base_url or (provider_cfg["ollama"] or {}).get(
                "base_url", "http://localhost:11434"
            )
            providers["ollama"] = OllamaProvider(base_url=base)
        profiles = {
            alias: ModelProfile(alias=alias, **cfg) for alias, cfg in data["models"].items()
        }
        return cls(profiles, providers, data.get("routes", {}), data.get("fallback_order", []))

    def get(self, alias: str) -> ModelProfile:
        try:
            return self.profiles[alias]
        except KeyError:
            raise UnknownModel(alias) from None

    def provider_for(self, profile: ModelProfile) -> LLMProvider:
        return self.providers[profile.provider]


class ModelRouter:
    """Decides which model serves a request, and what to try next if it is unavailable."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def candidates(
        self, task: str, preferred: str | None = None, private: bool = False
    ) -> list[ModelProfile]:
        reg = self.registry
        if preferred is not None:
            primary = reg.get(preferred)
            if private and not primary.local:
                primary = reg.get(reg.routes["private"])
        elif private:
            primary = reg.get(reg.routes["private"])
        else:
            primary = reg.get(reg.routes.get(task, reg.routes["chat"]))

        ordered = [primary]
        for alias in reg.fallback_order:
            p = reg.get(alias)
            if p not in ordered and (p.local or not private):
                ordered.append(p)
        return ordered
