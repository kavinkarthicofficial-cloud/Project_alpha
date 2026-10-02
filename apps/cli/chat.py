"""Terminal chat for trying the assistant without the web app.

    python -m apps.cli.chat [--model llama3]

Commands:
  /model <alias>   switch model (/model alone returns to the default)
  /models          list models
  /private on|off  keep messages on local models only
  /incognito on|off  don't remember anything
  /memories        what the assistant remembers
  /forget <id>     delete one memory (id prefix from /memories)
  /quit
"""

from __future__ import annotations

import argparse
import asyncio

from core.agent.orchestrator import (
    MemoryRecalled,
    Notice,
    ToolCallEvent,
    ToolResultEvent,
    TurnDone,
    TurnError,
)
from core.bootstrap import build_services
from core.config import get_settings
from core.llm.registry import UnknownModel
from core.llm.types import TextDelta

DIM, RESET = "\033[2m", "\033[0m"


async def main(model: str | None) -> None:
    settings = get_settings()
    services = build_services(settings)
    orch, episodes = services.orchestrator, services.episodes
    registry = orch.router.registry
    session = await orch.store.create_session(model)
    private = incognito = False
    print(f"{settings.assistant_name} — session {session.id}")
    print(f"{DIM}model: {model or registry.routes['chat'] + ' (default)'} · /models to list · /quit to exit{RESET}")

    while True:
        try:
            line = (await asyncio.to_thread(input, "\nyou › ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/models":
            for p in registry.profiles.values():
                tags = ", ".join(t for t, on in [("local", p.local), ("tools", p.supports_tools)] if on)
                print(f"  {p.alias:8} {p.provider}:{p.model} {DIM}{tags}{RESET}")
            continue
        if line.startswith("/model"):
            alias = line.removeprefix("/model").strip() or None
            try:
                if alias:
                    registry.get(alias)
            except UnknownModel:
                print(f"unknown model {alias!r}; try /models")
                continue
            await orch.store.set_model(session.id, alias)
            print(f"{DIM}now using {alias or registry.routes['chat'] + ' (default)'}{RESET}")
            continue
        if line.startswith("/private"):
            private = line.endswith("on")
            print(f"{DIM}private mode {'on: local models only' if private else 'off'}{RESET}")
            continue
        if line.startswith("/incognito"):
            incognito = line.endswith("on")
            print(f"{DIM}incognito {'on: nothing will be remembered' if incognito else 'off'}{RESET}")
            continue
        if line == "/memories":
            items = await episodes.recent(50)
            if not items:
                print(f"{DIM}nothing remembered yet{RESET}")
            for e in items:
                print(f"  {DIM}{str(e.id)[:8]} {e.occurred_at.astimezone():%Y-%m-%d %H:%M}{RESET} {e.text}")
            continue
        if line.startswith("/forget"):
            prefix = line.removeprefix("/forget").strip()
            matches = [e for e in await episodes.recent(500) if str(e.id).startswith(prefix)] if prefix else []
            if len(matches) != 1:
                print("give the id prefix of exactly one memory (see /memories)")
                continue
            await episodes.delete(matches[0].id)
            print(f"{DIM}forgot: {matches[0].text}{RESET}")
            continue

        print()
        async for ev in orch.run_turn(session.id, line, private=private, incognito=incognito):
            match ev:
                case MemoryRecalled(memories=ms):
                    print(f"{DIM}[remembered: {' · '.join(m['text'] for m in ms)}]{RESET}")
                case TextDelta(text=t):
                    print(t, end="", flush=True)
                case Notice(text=t):
                    print(f"{DIM}[{t}]{RESET}")
                case ToolCallEvent(name=n, arguments=a):
                    print(f"{DIM}[tool {n} {a}]{RESET}")
                case ToolResultEvent(content=c, is_error=err):
                    print(f"{DIM}[{'error' if err else 'result'}: {c[:200]}]{RESET}")
                case TurnDone(model_alias=m, usage=u):
                    print(f"\n{DIM}— {m} · {u.input_tokens} in / {u.output_tokens} out{RESET}")
                case TurnError(message=msg):
                    print(f"\n[error] {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model alias from config/models.yaml")
    asyncio.run(main(parser.parse_args().model))
