from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.agent.orchestrator import (
    AgentEvent,
    MemoryRecalled,
    Notice,
    Orchestrator,
    ToolCallEvent,
    ToolResultEvent,
    TurnDone,
    TurnError,
)
from core.bootstrap import build_services
from core.config import get_settings
from core.llm.registry import UnknownModel
from core.llm.types import Message, TextDelta
from core.memory.episodes import Episode, EpisodeStore
from core.storage.base import SessionNotFound

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "orchestrator"):
        services = build_services(get_settings())
        app.state.orchestrator = services.orchestrator
        app.state.episodes = services.episodes
    yield


app = FastAPI(title="Project Alpha", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


def get_episodes(request: Request) -> EpisodeStore:
    return request.app.state.episodes


Orch = Annotated[Orchestrator, Depends(get_orchestrator)]
Episodes = Annotated[EpisodeStore, Depends(get_episodes)]


class CreateSession(BaseModel):
    model: str | None = None


class UpdateSession(BaseModel):
    model: str | None


class ChatRequest(BaseModel):
    message: str
    # Use a different model for this one message without changing the session default.
    model: str | None = None
    # Keep this message on local models only.
    private: bool = False
    # Don't remember anything from this message.
    incognito: bool = False


def _check_model(orch: Orchestrator, alias: str | None) -> None:
    if alias is not None:
        try:
            orch.router.registry.get(alias)
        except UnknownModel:
            raise HTTPException(400, f"Unknown model {alias!r}") from None


async def _session_or_404(orch: Orchestrator, session_id: uuid.UUID):
    try:
        return await orch.store.get_session(session_id)
    except SessionNotFound:
        raise HTTPException(404, "Session not found") from None


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/config")
async def ui_config() -> dict[str, str]:
    settings = get_settings()
    return {
        "assistant_name": settings.assistant_name,
        "wake_word": settings.wake_word or settings.assistant_name,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def list_models(orch: Orch) -> dict[str, Any]:
    reg = orch.router.registry
    return {
        "models": [
            p.model_dump(include={"alias", "provider", "model", "local", "supports_tools", "context_window"})
            for p in reg.profiles.values()
        ],
        "routes": reg.routes,
        "fallback_order": reg.fallback_order,
    }


@app.post("/sessions", status_code=201)
async def create_session(body: CreateSession, orch: Orch):
    _check_model(orch, body.model)
    return await orch.store.create_session(body.model)


@app.patch("/sessions/{session_id}")
async def update_session(session_id: uuid.UUID, body: UpdateSession, orch: Orch):
    _check_model(orch, body.model)
    await _session_or_404(orch, session_id)
    return await orch.store.set_model(session_id, body.model)


@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: uuid.UUID, orch: Orch) -> list[Message]:
    await _session_or_404(orch, session_id)
    return await orch.store.list_messages(session_id)


@app.get("/memories")
async def list_memories(episodes: Episodes, limit: int = 50) -> list[Episode]:
    """What the assistant remembers, newest first."""
    return await episodes.recent(min(limit, 500))


@app.delete("/memories/{episode_id}", status_code=204)
async def forget(episode_id: uuid.UUID, episodes: Episodes) -> Response:
    if not await episodes.delete(episode_id):
        raise HTTPException(404, "Memory not found")
    return Response(status_code=204)


_EVENT_NAMES = {
    TextDelta: "text",
    Notice: "notice",
    MemoryRecalled: "memory",
    ToolCallEvent: "tool_call",
    ToolResultEvent: "tool_result",
    TurnDone: "done",
    TurnError: "error",
}


def _sse(event: AgentEvent) -> str:
    data = dataclasses.asdict(event)
    if isinstance(event, TurnDone):
        data["usage"] = event.usage.model_dump()
    return f"event: {_EVENT_NAMES[type(event)]}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/sessions/{session_id}/chat")
async def chat(session_id: uuid.UUID, body: ChatRequest, orch: Orch) -> StreamingResponse:
    """Streams Server-Sent Events: memory, text, notice, tool_call, tool_result, then done or error."""
    _check_model(orch, body.model)
    await _session_or_404(orch, session_id)

    async def events() -> AsyncIterator[str]:
        try:
            async for ev in orch.run_turn(
                session_id,
                body.message,
                model=body.model,
                private=body.private,
                incognito=body.incognito,
            ):
                yield _sse(ev)
        except Exception as e:  # the client must always get a final event
            logger.exception("chat turn failed")
            yield _sse(TurnError(f"Internal error: {type(e).__name__}"))

    return StreamingResponse(events(), media_type="text/event-stream")
