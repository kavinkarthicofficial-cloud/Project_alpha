from __future__ import annotations

from core.llm.anthropic_provider import to_anthropic
from core.llm.ollama_provider import to_ollama
from core.llm.openai_provider import GEMINI_SKIP_SIGNATURE, to_openai
from core.llm.types import Message, ToolCallBlock, ToolResultBlock
from tests.fakes import profile

CALL = ToolCallBlock(id="call_1", name="get_current_time", arguments={"timezone": "UTC"})


def conversation(model_alias: str = "other") -> list[Message]:
    return [
        Message.text("system", "You are helpful."),
        Message.text("user", "What time is it?"),
        Message.text("system", "User lives in Pune."),
        Message(role="assistant", content=[CALL], model_alias=model_alias),
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="call_1", tool_name="get_current_time", content="12:00")],
        ),
    ]


def test_anthropic_native_mid_conversation_system():
    system, msgs = to_anthropic(conversation(), profile("opus", "anthropic", supports_mid_system=True))
    assert system == "You are helpful."
    assert [m["role"] for m in msgs] == ["user", "system", "assistant", "user"]
    assert msgs[2]["content"][0] == {
        "type": "tool_use", "id": "call_1", "name": "get_current_time", "input": {"timezone": "UTC"}
    }
    assert msgs[3]["content"][0]["type"] == "tool_result"


def test_anthropic_merges_context_into_user_turn_when_unsupported():
    _, msgs = to_anthropic(conversation(), profile("haiku", "anthropic", supports_mid_system=False))
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "<context>" in msgs[0]["content"][1]["text"]


def test_anthropic_replays_native_content_only_for_the_same_model():
    native = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "Hi"}]
    msgs = [
        Message.text("user", "hello"),
        Message.text("assistant", "Hi").model_copy(
            update={"provider_state": {"anthropic": {"model": "opus-v1", "content": native}}}
        ),
    ]
    _, same = to_anthropic(msgs, profile("opus", "anthropic"))
    _, other = to_anthropic(msgs, profile("haiku", "anthropic"))
    assert same[1]["content"] == native
    assert other[1]["content"] == [{"type": "text", "text": "Hi"}]


def test_anthropic_conversion_is_deterministic():
    p = profile("opus", "anthropic", supports_mid_system=True)
    assert to_anthropic(conversation(), p) == to_anthropic(conversation(), p)


def test_openai_tool_results_follow_assistant_call():
    msgs = to_openai(conversation())
    assert [m["role"] for m in msgs] == ["system", "user", "system", "assistant", "tool"]
    assert msgs[3]["tool_calls"][0]["function"]["arguments"] == '{"timezone": "UTC"}'
    assert msgs[4] == {"role": "tool", "tool_call_id": "call_1", "content": "12:00"}


def test_ollama_conversion():
    msgs = to_ollama(conversation())
    assert [m["role"] for m in msgs] == ["system", "user", "system", "assistant", "tool"]
    assert msgs[3]["tool_calls"][0]["function"]["arguments"] == {"timezone": "UTC"}
    assert msgs[4]["tool_name"] == "get_current_time"


async def test_anthropic_without_credentials_is_unavailable(monkeypatch):
    import pytest

    from core.llm.anthropic_provider import AnthropicProvider
    from core.llm.types import ProviderUnavailable

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")
    monkeypatch.setenv("HOME", "/nonexistent")
    with pytest.raises(ProviderUnavailable, match="(?i)credentials"):
        async for _ in AnthropicProvider().stream(profile("opus", "anthropic"), [Message.text("user", "hi")]):
            pass


def test_gemini_thought_signature_is_replayed_to_the_same_model_only():
    gemini = profile("gemini", "openai", model="gemini-flash-latest", options={"thought_signatures": True})
    lite = profile("lite", "openai", model="gemini-flash-lite-latest", options={"thought_signatures": True})
    plain = profile("gpt", "openai", model="gpt-5")
    signed = {"google": {"thought_signature": "abc"}}
    msgs = [
        Message.text("user", "time?"),
        Message(
            role="assistant",
            content=[CALL],
            provider_state={"openai": {"model": "gemini-flash-latest", "tool_call_extra": {"call_1": signed}}},
        ),
    ]

    assert to_openai(msgs, gemini)[1]["tool_calls"][0]["extra_content"] == signed
    assert to_openai(msgs, lite)[1]["tool_calls"][0]["extra_content"] == GEMINI_SKIP_SIGNATURE
    assert "extra_content" not in to_openai(msgs, plain)[1]["tool_calls"][0]


async def test_openai_compatible_model_without_its_key_is_unavailable(monkeypatch):
    import pytest

    from core.llm.openai_provider import OpenAIProvider
    from core.llm.types import ProviderUnavailable

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    gemini = profile("gemini", "openai", options={"base_url": "https://example.invalid/", "api_key_env": "GEMINI_API_KEY"})
    with pytest.raises(ProviderUnavailable, match="GEMINI_API_KEY"):
        async for _ in OpenAIProvider().stream(gemini, [Message.text("user", "hi")]):
            pass
