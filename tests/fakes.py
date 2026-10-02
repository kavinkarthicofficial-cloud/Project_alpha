from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.agent.orchestrator import Orchestrator
from core.llm.registry import ModelRegistry, ModelRouter
from core.llm.types import (
    Completed,
    Message,
    ModelProfile,
    StopReason,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    ToolSpec,
)
from core.storage.base import InMemoryStore
from core.tools.builtin import default_tools


@dataclass
class Step:
    text: str = ""
    calls: list[ToolCallBlock] = field(default_factory=list)
    stop: StopReason | None = None
    raise_before: Exception | None = None
    raise_after_text: Exception | None = None


class FakeProvider:
    """Plays back scripted steps and records what each call received."""

    def __init__(self, name: str, steps: list[Step] | None = None) -> None:
        self.name = name
        self.steps = list(steps or [])
        self.calls: list[dict[str, Any]] = []

    async def stream(self, profile: ModelProfile, messages: list[Message], tools: list[ToolSpec] | None = None):
        self.calls.append({"alias": profile.alias, "messages": messages, "tools": tools})
        step = self.steps.pop(0)
        if step.raise_before:
            raise step.raise_before
        if step.text:
            yield TextDelta(step.text)
        if step.raise_after_text:
            raise step.raise_after_text
        content: list[TextBlock | ToolCallBlock] = []
        if step.text:
            content.append(TextBlock(text=step.text))
        content.extend(step.calls)
        stop = step.stop or ("tool_use" if step.calls else "end")
        yield Completed(
            message=Message(role="assistant", content=content, model_alias=profile.alias),
            stop_reason=stop,
            provider_model=f"{self.name}-model",
        )


def profile(alias: str, provider: str, **kw: Any) -> ModelProfile:
    defaults = {"model": f"{alias}-v1", "context_window": 100_000, "max_output_tokens": 1000}
    return ModelProfile(alias=alias, provider=provider, **{**defaults, **kw})


def make_orchestrator(
    cloud: FakeProvider, local: FakeProvider, **orch_kw: Any
) -> Orchestrator:
    profiles = {
        "big": profile("big", cloud.name),
        "backup": profile("backup", cloud.name),
        "tiny": profile("tiny", local.name, local=True, supports_tools=False),
    }
    registry = ModelRegistry(
        profiles,
        {cloud.name: cloud, local.name: local},
        routes={"chat": "big", "private": "tiny"},
        fallback_order=["big", "backup", "tiny"],
    )
    return Orchestrator(InMemoryStore(), ModelRouter(registry), default_tools(), **orch_kw)
