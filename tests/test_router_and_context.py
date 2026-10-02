from __future__ import annotations

import pytest

from core.agent.context import fit_to_budget
from core.config import ROOT
from core.llm.registry import ModelRegistry, UnknownModel
from core.llm.types import Message
from tests.fakes import FakeProvider, make_orchestrator, profile


def router():
    return make_orchestrator(FakeProvider("cloud"), FakeProvider("local")).router


def aliases(profiles):
    return [p.alias for p in profiles]


def test_default_route_then_fallbacks_without_duplicates():
    assert aliases(router().candidates("chat")) == ["big", "backup", "tiny"]


def test_preferred_model_goes_first():
    assert aliases(router().candidates("chat", preferred="backup")) == ["backup", "big", "tiny"]


def test_private_only_uses_local_models():
    assert aliases(router().candidates("chat", preferred="big", private=True)) == ["tiny"]


def test_unknown_model():
    with pytest.raises(UnknownModel):
        router().candidates("chat", preferred="nope")


def test_shipped_models_yaml_is_valid():
    reg = ModelRegistry.from_yaml(ROOT / "config" / "models.yaml")
    assert reg.get(reg.routes["chat"]).provider == "anthropic"
    assert all(reg.get(a).local for a in [reg.routes["private"]])


def test_fit_to_budget_drops_oldest_whole_turns():
    small = profile("tiny", "x", context_window=1400, max_output_tokens=100)
    msgs = [Message.text("system", "sys")]
    for i in range(10):
        msgs += [Message.text("user", f"question {i} " + "x" * 200), Message.text("assistant", "a" * 200)]

    fitted = fit_to_budget(msgs, small)

    assert fitted[0].role == "system"
    assert fitted[1].role == "user" and len(fitted) < len(msgs)
    assert fitted[-1] is msgs[-1]


def test_fit_to_budget_keeps_everything_when_it_fits():
    msgs = [Message.text("system", "sys"), Message.text("user", "hi")]
    assert fit_to_budget(msgs, profile("big", "x")) == msgs
