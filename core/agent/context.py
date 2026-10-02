"""Fit a conversation into a model's context window.

Memory and history are the same for every model; only how much of it fits differs.
Small local models get the most recent turns, large models get everything.
"""

from __future__ import annotations

from core.llm.types import Message, ModelProfile, TextBlock

# Room for the provider's own formatting overhead and tool definitions.
SAFETY_MARGIN_TOKENS = 1024


def estimate_tokens(msg: Message) -> int:
    # ~4 characters per token is close enough for budgeting; exact counts are per-provider.
    return len(msg.model_dump_json(include={"content"})) // 4 + 4


def _starts_turn(msg: Message) -> bool:
    return msg.role == "user" and any(isinstance(b, TextBlock) for b in msg.content)


def fit_to_budget(messages: list[Message], profile: ModelProfile) -> list[Message]:
    budget = profile.context_window - profile.max_output_tokens - SAFETY_MARGIN_TOKENS
    head_len = 0
    while head_len < len(messages) and messages[head_len].role == "system":
        head_len += 1
    head, body = messages[:head_len], messages[head_len:]

    total = sum(estimate_tokens(m) for m in messages)
    if total <= budget:
        return list(messages)

    # Drop whole turns from the front, so a tool call is never separated from its result.
    turn_starts = [i for i, m in enumerate(body) if _starts_turn(m)]
    for start in turn_starts[1:]:
        dropped = sum(estimate_tokens(m) for m in body[:start])
        if total - dropped <= budget:
            return head + body[start:]
    # Even the latest turn alone is over budget; send it anyway and let the provider decide.
    return head + body[turn_starts[-1]:] if turn_starts else list(messages)
