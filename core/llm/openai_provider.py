from __future__ import annotations

import json
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


def to_openai(messages: list[Message]) -> list[dict[str, Any]]:
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
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in calls
                ]
            out.append(entry)
    return out


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}
    return args if isinstance(args, dict) else {"__invalid_json__": raw}


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI()
        return self._client

    async def stream(
        self,
        profile: ModelProfile,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        req: dict[str, Any] = {
            "model": profile.model,
            "messages": to_openai(messages),
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
        calls: dict[int, dict[str, str]] = {}
        finish: str | None = None
        usage = Usage()
        provider_model = profile.model
        try:
            stream = await self._get_client().chat.completions.create(**req)
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
                    acc = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
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
        for _, c in sorted(calls.items()):
            kwargs: dict[str, Any] = {"name": c["name"], "arguments": _parse_arguments(c["arguments"])}
            if c["id"]:
                kwargs["id"] = c["id"]
            blocks.append(ToolCallBlock(**kwargs))

        yield Completed(
            message=Message(role="assistant", content=blocks, model_alias=profile.alias),
            stop_reason=_STOP_REASONS.get(finish, "other"),
            usage=usage,
            provider_model=provider_model,
        )
