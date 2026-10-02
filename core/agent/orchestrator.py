"""The agent loop: one user message in, a stream of events out.

Per turn: append the user message -> recall relevant memories -> pick a model -> call
it -> run any tool calls -> repeat until the model answers -> record the turn in memory.
Every step is appended to the store, so the next turn (on any model) sees the full
conversation.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from core.agent.context import fit_to_budget
from core.agent.prompts import build_system_prompt
from core.llm.registry import ModelRouter
from core.llm.types import (
    Completed,
    InvalidToolInput,
    Message,
    ModelProfile,
    ProviderError,
    ProviderUnavailable,
    TextDelta,
    ToolResultBlock,
    Usage,
)
from core.memory.engine import Recall
from core.storage.base import ConversationStore
from core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_JSON_RETRIES = 2


@dataclass
class Notice:
    text: str


@dataclass
class ToolCallEvent:
    name: str
    arguments: dict


@dataclass
class ToolResultEvent:
    name: str
    content: str
    is_error: bool


@dataclass
class TurnDone:
    model_alias: str
    provider_model: str
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


@dataclass
class TurnError:
    message: str


@dataclass
class MemoryRecalled:
    memories: list[dict]


AgentEvent = (
    TextDelta | Notice | MemoryRecalled | ToolCallEvent | ToolResultEvent | TurnDone | TurnError
)


class Memory(Protocol):
    async def recall(
        self, session_id: uuid.UUID, user_text: str, *, include_private: bool
    ) -> Recall: ...

    async def record_turn(
        self, session_id: uuid.UUID, user_text: str, reply_text: str, *, private: bool
    ) -> object: ...


class NoMemory:
    async def recall(self, session_id: uuid.UUID, user_text: str, *, include_private: bool) -> Recall:
        return Recall()

    async def record_turn(
        self, session_id: uuid.UUID, user_text: str, reply_text: str, *, private: bool
    ) -> None:
        return None


class Orchestrator:
    def __init__(
        self,
        store: ConversationStore,
        router: ModelRouter,
        tools: ToolRegistry,
        *,
        assistant_name: str = "Assistant",
        memory: Memory | None = None,
        max_steps: int = 8,
    ) -> None:
        self.store = store
        self.router = router
        self.tools = tools
        self.assistant_name = assistant_name
        self.memory = memory or NoMemory()
        self.max_steps = max_steps

    async def _append(self, session_id: uuid.UUID, history: list[Message], msg: Message, **kw) -> None:
        await self.store.append(session_id, msg, **kw)
        history.append(msg)

    async def run_turn(
        self,
        session_id: uuid.UUID,
        user_text: str,
        *,
        model: str | None = None,
        private: bool = False,
        incognito: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Run one user turn.

        private: only local models see this turn, and what it stores is only ever
            shown to local models.
        incognito: nothing from this turn is written to long-term memory.
        """
        session = await self.store.get_session(session_id)
        candidates = self.router.candidates("chat", model or session.model_alias, private)

        history = await self.store.list_messages(session_id)
        if not history:
            await self._append(
                session_id, history, Message.text("system", build_system_prompt(self.assistant_name))
            )
        await self._append(session_id, history, Message.text("user", user_text))
        try:
            recall = await self.memory.recall(session_id, user_text, include_private=private)
        except Exception as e:  # memory trouble must not block the conversation
            logger.exception("memory recall failed")
            recall = Recall(warning=f"memory unavailable this turn ({type(e).__name__})")
        if recall.warning:
            yield Notice(recall.warning)
        if recall.memories:
            yield MemoryRecalled(
                [
                    {
                        "id": str(m.episode.id),
                        "text": m.episode.text,
                        "occurred_at": m.episode.occurred_at.isoformat(),
                        "score": round(m.score, 3),
                    }
                    for m in recall.memories
                ]
            )
        if recall.context:
            await self._append(session_id, history, Message.text("system", recall.context))

        total = Usage()
        active: ModelProfile | None = None
        for _ in range(self.max_steps):
            # Once a model has answered, the rest of this turn stays on it.
            pool = [active] if active else candidates
            completed: Completed | None = None
            for i, profile in enumerate(pool):
                streamed = False
                try:
                    async for ev in self._call(profile, history):
                        if isinstance(ev, Completed):
                            completed = ev
                        else:
                            streamed = streamed or isinstance(ev, TextDelta)
                            yield ev
                    active = profile
                    break
                except ProviderUnavailable as e:
                    # After partial output, switching models would repeat the answer.
                    if not streamed and i + 1 < len(pool):
                        yield Notice(f"{profile.alias} unavailable ({e}); trying {pool[i + 1].alias}.")
                        continue
                    if streamed:
                        yield TurnError(f"{profile.alias} failed mid-response: {e}")
                    else:
                        yield TurnError(f"No model available. Last error from {profile.alias}: {e}")
                    return
                except ProviderError as e:
                    yield TurnError(f"{profile.alias}: {e}")
                    return
            assert completed is not None and active is not None

            reply = completed.message
            reply.model_alias = active.alias
            await self._append(
                session_id, history, reply, usage=completed.usage, provider_model=completed.provider_model
            )
            total.input_tokens += completed.usage.input_tokens
            total.output_tokens += completed.usage.output_tokens
            total.cache_read_tokens += completed.usage.cache_read_tokens

            calls = reply.tool_calls()
            if calls and completed.stop_reason in ("max_tokens", "refusal"):
                # Never run a truncated or refused tool call, but answer it so the
                # history stays valid for every provider.
                await self._append(
                    session_id,
                    history,
                    Message(
                        role="user",
                        content=[
                            ToolResultBlock(
                                tool_call_id=c.id,
                                tool_name=c.name,
                                content="Not executed: the response ended before the call was complete.",
                                is_error=True,
                            )
                            for c in calls
                        ],
                    ),
                )
                yield TurnError(f"Response stopped early ({completed.stop_reason}).")
                return
            if not calls:
                if not incognito:
                    try:
                        await self.memory.record_turn(
                            session_id, user_text, reply.text_content(), private=private
                        )
                    except Exception as e:  # the answer is already delivered; just report it
                        logger.exception("memory write failed")
                        yield Notice(f"Couldn't save this to memory ({type(e).__name__}).")
                yield TurnDone(active.alias, completed.provider_model, completed.stop_reason, total)
                return

            results: list[ToolResultBlock] = []
            for call in calls:
                yield ToolCallEvent(call.name, call.arguments)
                result = await self.tools.execute(call)
                yield ToolResultEvent(call.name, result.content, result.is_error)
                results.append(result)
            await self._append(session_id, history, Message(role="user", content=results))

        yield TurnError(f"Stopped after {self.max_steps} steps without a final answer.")

    async def _call(
        self, profile: ModelProfile, history: list[Message]
    ) -> AsyncIterator[TextDelta | Notice | Completed]:
        provider = self.router.registry.provider_for(profile)
        tools = self.tools.specs() if profile.supports_tools and len(self.tools) else None
        messages = fit_to_budget(history, profile)
        for attempt in range(MAX_TOOL_JSON_RETRIES + 1):
            try:
                async for ev in provider.stream(profile, messages, tools):
                    yield ev
                return
            except InvalidToolInput:
                if attempt == MAX_TOOL_JSON_RETRIES:
                    raise
                yield Notice("Model produced a malformed tool call; retrying.")
