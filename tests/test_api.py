from __future__ import annotations

import httpx
import pytest

from apps.api.main import app
from core.memory.embeddings import HashEmbedder
from core.memory.engine import EpisodicMemory
from core.memory.episodes import InMemoryEpisodeStore
from tests.fakes import FakeProvider, Step, make_orchestrator


@pytest.fixture
def client():
    cloud = FakeProvider("cloud", [Step(text="Hello there"), Step(text="Second")])
    episodes = InMemoryEpisodeStore()
    app.state.episodes = episodes
    app.state.orchestrator = make_orchestrator(
        cloud, FakeProvider("local"), memory=EpisodicMemory(episodes, HashEmbedder())
    )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_chat_streams_events(client):
    async with client:
        session = (await client.post("/sessions", json={})).json()
        resp = await client.post(f"/sessions/{session['id']}/chat", json={"message": "hi"})
        assert resp.status_code == 200
        body = resp.text
        assert "event: text" in body and "Hello there" in body
        assert "event: done" in body

        history = (await client.get(f"/sessions/{session['id']}/messages")).json()
        assert [m["role"] for m in history] == ["system", "user", "system", "assistant"]


async def test_switch_session_model(client):
    async with client:
        session = (await client.post("/sessions", json={"model": "big"})).json()
        resp = await client.patch(f"/sessions/{session['id']}", json={"model": "tiny"})
        assert resp.json()["model_alias"] == "tiny"


async def test_unknown_model_and_session(client):
    async with client:
        assert (await client.post("/sessions", json={"model": "nope"})).status_code == 400
        missing = "00000000-0000-0000-0000-000000000000"
        resp = await client.post(f"/sessions/{missing}/chat", json={"message": "hi"})
        assert resp.status_code == 404


async def test_unexpected_failure_still_ends_with_error_event(client):
    app.state.orchestrator.router.registry.providers["cloud"].steps = [Step(raise_before=TypeError("boom"))]
    async with client:
        session = (await client.post("/sessions", json={})).json()
        resp = await client.post(f"/sessions/{session['id']}/chat", json={"message": "hi"})
        assert "event: error" in resp.text and "TypeError" in resp.text


async def test_memories_can_be_listed_and_forgotten(client):
    async with client:
        session = (await client.post("/sessions", json={})).json()
        await client.post(f"/sessions/{session['id']}/chat", json={"message": "My dog is Biscuit"})
        await client.post(
            f"/sessions/{session['id']}/chat", json={"message": "My PIN is 1234", "incognito": True}
        )

        memories = (await client.get("/memories")).json()
        assert [m["text"] for m in memories] == ["My dog is Biscuit"]
        assert "embedding" not in memories[0]

        assert (await client.delete(f"/memories/{memories[0]['id']}")).status_code == 204
        assert (await client.get("/memories")).json() == []
        assert (await client.delete(f"/memories/{memories[0]['id']}")).status_code == 404


async def test_web_page_is_served(client):
    async with client:
        resp = await client.get("/")
        assert resp.status_code == 200 and "<title>" in resp.text


async def test_config_exposes_name_and_wake_word(client, monkeypatch):
    monkeypatch.setenv("ASSISTANT_NAME", "Nova")
    from core.config import get_settings

    get_settings.cache_clear()
    try:
        async with client:
            assert (await client.get("/config")).json() == {"assistant_name": "Nova", "wake_word": "Nova"}
            assert (await client.get("/static/voice.js")).status_code == 200
    finally:
        get_settings.cache_clear()
