from __future__ import annotations

from core.agent.orchestrator import MemoryRecalled, Notice, ToolResultEvent, TurnDone, TurnError
from core.llm.types import ProviderUnavailable, TextDelta, ToolCallBlock, ToolResultBlock
from core.memory.engine import Recall, RecalledMemory
from core.memory.episodes import Episode
from tests.fakes import FakeProvider, Step, make_orchestrator


async def collect(orch, session_id, text, **kw):
    return [ev async for ev in orch.run_turn(session_id, text, **kw)]


async def test_simple_turn_is_persisted():
    cloud, local = FakeProvider("cloud", [Step(text="Hi!")]), FakeProvider("local")
    orch = make_orchestrator(cloud, local, assistant_name="Nova")
    s = await orch.store.create_session()

    events = await collect(orch, s.id, "hello")

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hi!"]
    assert isinstance(events[-1], TurnDone) and events[-1].model_alias == "big"
    history = await orch.store.list_messages(s.id)
    assert [m.role for m in history] == ["system", "user", "assistant"]
    assert "Nova" in history[0].text_content()
    assert history[2].model_alias == "big"


async def test_tool_loop_runs_tool_and_feeds_result_back():
    call = ToolCallBlock(name="get_current_time", arguments={})
    cloud = FakeProvider("cloud", [Step(calls=[call]), Step(text="It's Friday.")])
    orch = make_orchestrator(cloud, FakeProvider("local"))
    s = await orch.store.create_session()

    events = await collect(orch, s.id, "what day is it?")

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1 and not results[0].is_error and "weekday" in results[0].content
    second_call_msgs = cloud.calls[1]["messages"]
    tool_msg = second_call_msgs[-1]
    assert isinstance(tool_msg.content[0], ToolResultBlock)
    assert tool_msg.content[0].tool_call_id == call.id


async def test_invalid_tool_arguments_return_error_result():
    bad = ToolCallBlock(name="get_current_time", arguments={"timezone": 5})
    cloud = FakeProvider("cloud", [Step(calls=[bad]), Step(text="Sorry.")])
    orch = make_orchestrator(cloud, FakeProvider("local"))
    s = await orch.store.create_session()

    events = await collect(orch, s.id, "time?")

    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.is_error and "Invalid input" in result.content


async def test_falls_back_when_model_unavailable():
    cloud = FakeProvider(
        "cloud",
        [Step(raise_before=ProviderUnavailable("no key")), Step(text="from backup")],
    )
    orch = make_orchestrator(cloud, FakeProvider("local"))
    s = await orch.store.create_session()

    events = await collect(orch, s.id, "hi")

    assert any(isinstance(e, Notice) and "trying backup" in e.text for e in events)
    assert events[-1].model_alias == "backup"
    assert [c["alias"] for c in cloud.calls] == ["big", "backup"]


async def test_no_fallback_after_partial_output():
    cloud = FakeProvider("cloud", [Step(text="Hal", raise_after_text=ProviderUnavailable("reset"))])
    orch = make_orchestrator(cloud, FakeProvider("local"))
    s = await orch.store.create_session()

    events = await collect(orch, s.id, "hi")

    assert isinstance(events[-1], TurnError) and "mid-response" in events[-1].message
    assert len(cloud.calls) == 1


async def test_switching_models_keeps_the_conversation():
    cloud = FakeProvider("cloud", [Step(text="Noted, Biscuit.")])
    local = FakeProvider("local", [Step(text="Your dog is Biscuit.")])
    orch = make_orchestrator(cloud, local)
    s = await orch.store.create_session()

    await collect(orch, s.id, "My dog is called Biscuit.")
    await orch.store.set_model(s.id, "tiny")
    events = await collect(orch, s.id, "What's my dog called?")

    assert events[-1].model_alias == "tiny"
    seen = [m.text_content() for m in local.calls[0]["messages"]]
    assert "My dog is called Biscuit." in seen and "Noted, Biscuit." in seen
    # The local model has no tool support, so it gets no tools.
    assert local.calls[0]["tools"] is None


async def test_private_turn_stays_local():
    cloud, local = FakeProvider("cloud"), FakeProvider("local", [Step(text="ok")])
    orch = make_orchestrator(cloud, local)
    s = await orch.store.create_session(model_alias="big")

    events = await collect(orch, s.id, "my bank PIN hint is...", private=True)

    assert events[-1].model_alias == "tiny"
    assert cloud.calls == []


async def test_truncated_tool_call_is_answered_not_run():
    call = ToolCallBlock(name="get_current_time", arguments={})
    cloud = FakeProvider("cloud", [Step(calls=[call], stop="max_tokens")])
    orch = make_orchestrator(cloud, FakeProvider("local"))
    s = await orch.store.create_session()

    events = await collect(orch, s.id, "time?")

    assert isinstance(events[-1], TurnError)
    assert not any(isinstance(e, ToolResultEvent) for e in events)
    last = (await orch.store.list_messages(s.id))[-1]
    assert last.content[0].tool_call_id == call.id and last.content[0].is_error


class FakeMemory:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.recorded: list[tuple[str, str, bool]] = []

    async def recall(self, session_id, user_text, *, include_private):
        if self.fail:
            raise RuntimeError("db down")
        episode = Episode(text="My dog is called Biscuit")
        return Recall(context="User's dog: Biscuit", memories=[RecalledMemory(episode, 0.9)])

    async def record_turn(self, session_id, user_text, reply_text, *, private):
        self.recorded.append((user_text, reply_text, private))


async def test_memory_is_recalled_before_and_recorded_after_the_turn():
    cloud = FakeProvider("cloud", [Step(text="It's Biscuit.")])
    mem = FakeMemory()
    orch = make_orchestrator(cloud, FakeProvider("local"), memory=mem)
    s = await orch.store.create_session()

    events = await collect(orch, s.id, "what's my dog called?")

    recalled = next(e for e in events if isinstance(e, MemoryRecalled))
    assert recalled.memories[0]["text"] == "My dog is called Biscuit"
    sent = cloud.calls[0]["messages"]
    assert [m.role for m in sent] == ["system", "user", "system"]
    assert sent[-1].text_content() == "User's dog: Biscuit"
    assert mem.recorded == [("what's my dog called?", "It's Biscuit.", False)]


async def test_incognito_turn_is_not_recorded():
    mem = FakeMemory()
    orch = make_orchestrator(FakeProvider("cloud", [Step(text="ok")]), FakeProvider("local"), memory=mem)
    s = await orch.store.create_session()

    await collect(orch, s.id, "my secret plan", incognito=True)

    assert mem.recorded == []


async def test_memory_failure_does_not_block_the_answer():
    orch = make_orchestrator(
        FakeProvider("cloud", [Step(text="ok")]), FakeProvider("local"), memory=FakeMemory(fail=True)
    )
    s = await orch.store.create_session()

    events = await collect(orch, s.id, "hi")

    assert any(isinstance(e, Notice) and "memory unavailable" in e.text for e in events)
    assert isinstance(events[-1], TurnDone)
