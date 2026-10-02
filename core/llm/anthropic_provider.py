from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic

from core.llm.types import (
    Completed,
    InvalidToolInput,
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

FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS: dict[str | None, StopReason] = {
    "end_turn": "end",
    "stop_sequence": "end",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


def _wrap_context(text: str) -> dict[str, Any]:
    return {"type": "text", "text": f"<context>\n{text}\n</context>"}


def to_anthropic(
    messages: list[Message], profile: ModelProfile
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert neutral messages to (system, messages) for the Messages API.

    The conversion is deterministic for a given model, so the request prefix is
    byte-identical from turn to turn (prompt caching and thinking-block replay
    both depend on that).
    """
    system_parts: list[str] = []
    i = 0
    while i < len(messages) and messages[i].role == "system":
        system_parts.append(messages[i].text_content())
        i += 1

    out: list[dict[str, Any]] = []

    def append_user_blocks(blocks: list[dict[str, Any]]) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": "user", "content": blocks})

    for msg in messages[i:]:
        if msg.role == "system":
            text = msg.text_content()
            if profile.supports_mid_system and out and out[-1]["role"] == "user":
                out.append({"role": "system", "content": text})
            else:
                append_user_blocks([_wrap_context(text)])
        elif msg.role == "user":
            blocks: list[dict[str, Any]] = []
            for b in msg.content:
                if isinstance(b, ToolResultBlock):
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b.tool_call_id,
                            "content": b.content,
                            "is_error": b.is_error,
                        }
                    )
                elif isinstance(b, TextBlock) and b.text:
                    blocks.append({"type": "text", "text": b.text})
            append_user_blocks(blocks)
        else:
            native = (msg.provider_state or {}).get("anthropic")
            if native and native.get("model") == profile.model:
                content = native["content"]
            else:
                content = []
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text:
                        content.append({"type": "text", "text": b.text})
                    elif isinstance(b, ToolCallBlock):
                        content.append(
                            {"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments}
                        )
            out.append({"role": "assistant", "content": content})

    system = "\n\n".join(p for p in system_parts if p) or None
    return system, out


def from_anthropic(final: Any, profile: ModelProfile) -> Message:
    blocks: list[TextBlock | ToolCallBlock] = []
    for b in final.content:
        if b.type == "text":
            blocks.append(TextBlock(text=b.text))
        elif b.type == "tool_use":
            args = b.input if isinstance(b.input, dict) else {"__invalid_json__": repr(b.input)}
            blocks.append(ToolCallBlock(id=b.id, name=b.name, arguments=args))
    native = [b.model_dump(mode="json", exclude_none=True) for b in final.content]
    return Message(
        role="assistant",
        content=blocks,
        model_alias=profile.alias,
        provider_state={"anthropic": {"model": profile.model, "content": native}},
    )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            client = anthropic.AsyncAnthropic()
            # Without this check the SDK raises a bare TypeError at request time.
            if client.api_key is None and client.auth_token is None and client.credentials is None:
                raise ProviderUnavailable(
                    "no Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)"
                )
            self._client = client
        return self._client

    def _build_request(
        self, profile: ModelProfile, messages: list[Message], tools: list[ToolSpec] | None
    ) -> dict[str, Any]:
        system, msgs = to_anthropic(messages, profile)
        req: dict[str, Any] = {
            "model": profile.model,
            "max_tokens": profile.max_output_tokens,
            "messages": msgs,
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            req["system"] = system
        if tools:
            req["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "eager_input_streaming": True,
                }
                for t in tools
            ]
        opts = profile.options
        if "effort" in opts:
            req["output_config"] = {"effort": opts["effort"]}
        if "thinking" in opts:
            req["thinking"] = opts["thinking"]
        return req

    async def stream(
        self,
        profile: ModelProfile,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        req = self._build_request(profile, messages, tools)
        try:
            client = self._get_client()
            if profile.options.get("server_fallback"):
                # On a safety-classifier refusal, the API retries on the model
                # Anthropic recommends for that refusal category.
                cm = client.beta.messages.stream(**req, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                cm = client.messages.stream(**req)
            async with cm as stream:
                async for event in stream:
                    if event.type == "text" and event.text:
                        yield TextDelta(event.text)
                final = await stream.get_final_message()
        except ValueError as e:
            # Tool-call JSON the SDK could not parse at all (eager input streaming).
            raise InvalidToolInput(str(e)) from e
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
            anthropic.CredentialsError,
        ) as e:
            raise ProviderUnavailable(f"{type(e).__name__}: {e}") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"{e.status_code}: {e.message}") from e

        usage = final.usage
        yield Completed(
            message=from_anthropic(final, profile),
            stop_reason=_STOP_REASONS.get(final.stop_reason, "other"),
            usage=Usage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
            ),
            provider_model=final.model,
        )
