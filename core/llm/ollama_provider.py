from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.llm.types import (
    Completed,
    Message,
    ModelProfile,
    ProviderError,
    ProviderUnavailable,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
    Usage,
)


def to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            out.append({"role": "system", "content": msg.text_content()})
        elif msg.role == "user":
            for b in msg.content:
                if isinstance(b, ToolResultBlock):
                    out.append({"role": "tool", "tool_name": b.tool_name, "content": b.content})
            text = msg.text_content()
            if text:
                out.append({"role": "user", "content": text})
        else:
            entry: dict[str, Any] = {"role": "assistant", "content": msg.text_content()}
            calls = msg.tool_calls()
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                ]
            out.append(entry)
    return out


class OllamaProvider:
    """Local models through Ollama's native /api/chat endpoint. Nothing leaves the machine."""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def stream(
        self,
        profile: ModelProfile,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        req: dict[str, Any] = {
            "model": profile.model,
            "messages": to_ollama(messages),
            "stream": True,
            "options": {"num_ctx": profile.context_window, **profile.options.get("ollama", {})},
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

        text_parts: list[str] = []
        calls: list[ToolCallBlock] = []
        done: dict[str, Any] = {}
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout) as client,
                client.stream("POST", f"{self.base_url}/api/chat", json=req) as resp,
            ):
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    if resp.status_code == 404:
                        raise ProviderUnavailable(f"model not available locally: {body}")
                    raise ProviderError(f"ollama {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise ProviderError(f"ollama: {data['error']}")
                    msg = data.get("message") or {}
                    if msg.get("content"):
                        text_parts.append(msg["content"])
                        yield TextDelta(msg["content"])
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        args = fn.get("arguments")
                        if not isinstance(args, dict):
                            args = {"__invalid_json__": repr(args)}
                        calls.append(ToolCallBlock(name=fn.get("name", ""), arguments=args))
                    if data.get("done"):
                        done = data
        except httpx.TransportError as e:
            raise ProviderUnavailable(f"cannot reach Ollama at {self.base_url}: {e}") from e

        blocks: list[TextBlock | ToolCallBlock] = []
        if text_parts:
            blocks.append(TextBlock(text="".join(text_parts)))
        blocks.extend(calls)
        if calls:
            stop = "tool_use"
        elif done.get("done_reason") == "length":
            stop = "max_tokens"
        else:
            stop = "end"
        yield Completed(
            message=Message(role="assistant", content=blocks, model_alias=profile.alias),
            stop_reason=stop,
            usage=Usage(
                input_tokens=done.get("prompt_eval_count", 0),
                output_tokens=done.get("eval_count", 0),
            ),
            provider_model=done.get("model", profile.model),
        )
