from __future__ import annotations

from core.llm.registry import ModelRouter
from core.llm.types import Completed, Message, ModelProfile, ProviderUnavailable


async def complete(
    router: ModelRouter, task: str, messages: list[Message], *, preferred: str | None = None
) -> tuple[str, ModelProfile]:
    """One-shot, tool-free completion for background work (grading, later extraction),
    with the same model routing and fallback as chat."""
    last_error: ProviderUnavailable | None = None
    for profile in router.candidates(task, preferred):
        provider = router.registry.provider_for(profile)
        try:
            async for ev in provider.stream(profile, messages, None):
                if isinstance(ev, Completed):
                    return ev.message.text_content(), profile
        except ProviderUnavailable as e:
            last_error = e
    raise ProviderUnavailable(f"no model available for {task!r}: {last_error}")
