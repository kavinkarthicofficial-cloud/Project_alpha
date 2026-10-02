from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import openai

from core.llm.types import (
    Completed,
    Message,
    ModelProfile,
    ProviderError,
    ProviderUnavailable,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)

_STOP_REASONS: dict[str | None, StopReason] = {
    "stop": "end",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


# Gemini's documented placeholder for function calls it didn't produce itself
# (e.g. after switching models mid-conversation).
GEMINI_SKIP_SIGNATURE = {"google": {"thought_signature": "skip_thought_signature_validator"}}


def _tool_call_extra(msg: Message, call_id: str, profile: ModelProfile | None) -> dict | None:
    """Provider-specific data a tool call must carry when replayed, such as Gemini's
    thought signature. Replayed only to the model that produced it."""
    if profile is None:
        return None
    native = (msg.provider_state or {}).get("openai") or {}
    if native.get("model") == profile.model and call_id in native.get("tool_call_extra", {}):
        return native["tool_call_extra"][call_id]
    if profile.options.get("thought_signatures"):
        return GEMINI_SKIP_SIGNATURE
    return None


def to_openai(messages: list[Message], profile: ModelProfile | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            out.append({"role": "system", "content": msg.text_content()})
        elif msg.role == "user":
            # Tool results must directly follow the assistant turn that requested them.
            for b in msg.content:
                if isinstance(b, ToolResultBlock):
                    out.append({"role": "tool", "tool_call_id": b.tool_call_id, "content": b.content})
            text = msg.text_content()
            if text:
                out.append({"role": "user", "content": text})
        else:
            entry: dict[str, Any] = {"role": "assistant", "content": msg.text_content() or None}
            calls = msg.tool_calls()
            if calls:
                entry["tool_calls"] = []
                for c in calls:
                    call: dict[str, Any] = {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    extra = _tool_call_extra(msg, c.id, profile)
                    if extra:
                        call["extra_content"] = extra
                    entry["tool_calls"].append(call)
            out.append(entry)
    return out


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}
    return args if isinstance(args, dict) else {"__invalid_json__": raw}


class OpenAIProvider:
    """OpenAI, and any service with an OpenAI-compatible API (Gemini, Groq, OpenRouter,
    GitHub Models, ...): set `base_url` and `api_key_env` in the model's options."""

    name = "openai"

    def __init__(self) -> None:
        self._clients: dict[tuple[str | None, str | None], openai.AsyncOpenAI] = {}

    def _get_client(self, profile: ModelProfile) -> openai.AsyncOpenAI:
        base_url = profile.options.get("base_url")
        key_env = profile.options.get("api_key_env")
        api_key = None
        if key_env:
            api_key = os.environ.get(key_env)
            if not api_key:
                raise ProviderUnavailable(f"{key_env} is not set (add it to .env)")
        key = (base_url, key_env)
        if key not in self._clients:
            # None falls back to the SDK defaults: OPENAI_API_KEY and api.openai.com.
            self._clients[key] = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._clients[key]

    async def stream(
        self,
        profile: ModelProfile,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        req: dict[str, Any] = {
            "model": profile.model,
            "messages": to_openai(messages, profile),
            "max_completion_tokens": profile.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            req["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        if "reasoning_effort" in profile.options:
            req["reasoning_effort"] = profile.options["reasoning_effort"]

        text_parts: list[str] = []
        calls: dict[int | str, dict[str, Any]] = {}
        finish: str | None = None
        usage = Usage()
        provider_model = profile.model
        try:
            stream = await self._get_client(profile).chat.completions.create(**req)
            async for chunk in stream:
                provider_model = chunk.model or provider_model
                if chunk.usage:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    text_parts.append(delta.content)
                    yield TextDelta(delta.content)
                for tc in delta.tool_calls or []:
                    # OpenAI streams each call in pieces keyed by `index`; Gemini sends
                    # whole calls with no index, so fall back to the call id.
                    if tc.index is not None:
                        slot: int | str = tc.index
                    elif tc.id:
                        slot = f"id:{tc.id}"
                    else:
                        slot = next(reversed(calls), 0)
                    acc = calls.setdefault(slot, {"id": "", "name": "", "arguments": "", "extra": None})
                    extra = (tc.model_extra or {}).get("extra_content")
                    if extra:
                        acc["extra"] = extra
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    finish = choice.finish_reason
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            raise ProviderUnavailable(f"{type(e).__name__}: {e}") from e
        except openai.APIStatusError as e:
            raise ProviderError(f"{e.status_code}: {e.message}") from e
        except openai.OpenAIError as e:
            # Raised before any request, e.g. when OPENAI_API_KEY is not set.
            raise ProviderUnavailable(str(e)) from e

        blocks: list[TextBlock | ToolCallBlock] = []
        if text_parts:
            blocks.append(TextBlock(text="".join(text_parts)))
        extras: dict[str, Any] = {}
        for c in calls.values():  # insertion order = the order the calls arrived
            kwargs: dict[str, Any] = {"name": c["name"], "arguments": _parse_arguments(c["arguments"])}
            if c["id"]:
                kwargs["id"] = c["id"]
            block = ToolCallBlock(**kwargs)
            if c["extra"]:
                extras[block.id] = c["extra"]
            blocks.append(block)

        state = {"openai": {"model": profile.model, "tool_call_extra": extras}} if extras else None
        yield Completed(
            message=Message(
                role="assistant", content=blocks, model_alias=profile.alias, provider_state=state
            ),
            stop_reason=_STOP_REASONS.get(finish, "other"),
            usage=usage,
            provider_model=provider_model,
        )
