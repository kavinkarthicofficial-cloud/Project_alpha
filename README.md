# Project Alpha

A personal assistant with long-term memory. Working title: the name is set by
`ASSISTANT_NAME` until we pick one.

**Status: Phase 1 (episodic memory).** The assistant remembers what you tell it and
recalls it in later conversations, with any model. Fact extraction with dates and
conflict resolution is Phase 2.

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp .env.example .env            # add ANTHROPIC_API_KEY / OPENAI_API_KEY if you have them
ollama pull nomic-embed-text    # local embedding model for memory search (~270 MB)

# Terminal chat (no database needed; conversations live in memory)
.venv/bin/python -m apps.cli.chat                 # default model (opus)
.venv/bin/python -m apps.cli.chat --model llama3  # local, via Ollama
```

In the chat: `/models` lists models, `/model llama3` switches mid-conversation,
`/model` returns to the default, `/private on` keeps messages on local models only,
`/incognito on` stops anything being remembered, `/memories` and `/forget <id>`
show and delete what it remembers.

Without `DATABASE_URL`, memories last until the process exits. Set up Postgres
(below) to keep them.

### Web app and API

```bash
.venv/bin/uvicorn apps.api.main:app --reload    # then open http://localhost:8000
```

The page has streaming chat, a model picker, Private and Incognito toggles, the
memories each answer used, and a panel to review and forget memories.

| Method | Path | |
|---|---|---|
| GET | `/models` | configured models, routes, fallback order |
| POST | `/sessions` | `{"model": "opus"}` (optional) |
| PATCH | `/sessions/{id}` | `{"model": "llama3"}` switches the session's model |
| POST | `/sessions/{id}/chat` | `{"message": "...", "model"?: "...", "private"?: false, "incognito"?: false}` → Server-Sent Events: `memory`, `text`, `notice`, `tool_call`, `tool_result`, `done` / `error` |
| GET | `/sessions/{id}/messages` | full history, with which model wrote each turn |
| GET | `/memories` | what the assistant remembers, newest first |
| DELETE | `/memories/{id}` | forget one memory |

### Postgres (keeps conversations and memories across restarts)

```bash
docker compose up -d              # Postgres 16 + pgvector on port 5433
# set DATABASE_URL in .env (see .env.example), then:
.venv/bin/alembic upgrade head
```

## How memory works (Phase 1)

[core/memory/engine.py](core/memory/engine.py)

- **Saving:** after each turn, what the user said is stored as an *episode* with its
  embedding (pleasantries like "thanks" are skipped). Incognito turns aren't stored.
  Private turns are stored but only ever shown to local models.
- **Recalling:** before each turn, the message is embedded and matched against
  episodes from *other* sessions (the current one is already in context), by meaning
  (vector similarity) and by exact words (keyword search, which catches names and
  numbers). Hits above the similarity bar (`min_similarity` in models.yaml) are
  added to the conversation as a context message: oldest first, with dates, so the
  model can tell a newer fact from an outdated one. The current date is always
  included.
- **Failure-tolerant:** if the embedding model is down, search falls back to
  keywords, and episodes are saved without vectors. If memory fails entirely, the
  turn still runs.

## Switching models

All models are declared in [config/models.yaml](config/models.yaml): which provider,
context window, tool support, and whether a model may write to memory.
`routes` picks the model per task (chat, extraction, private); `fallback_order` is
what gets tried when a model is unavailable (no key, network, rate limit, outage).

The conversation is stored in one provider-neutral format
([core/llm/types.py](core/llm/types.py)), and each adapter in `core/llm/` converts it
to its provider's format. So switching from Claude to a local model mid-conversation
keeps everything. Things that make this safe:

- **History is append-only.** The session's system prompt never changes, and per-turn
  context (memory, from Phase 1) is appended as a new message instead of edited in.
  Provider prompt caches and Claude's thinking blocks depend on that.
- **Native replay.** A Claude turn is replayed byte-for-byte to the model that wrote
  it, and rebuilt from the neutral format for any other model.
- **Per-model budgets.** [core/agent/context.py](core/agent/context.py) drops the oldest
  whole turns when a small model's context window would overflow.
- **No double answers.** Fallback to another model only happens before any text was
  streamed.
- **Tool input is validated** against its JSON schema before running, whichever model
  produced it.

## Evals

```bash
.venv/bin/python -m evals.run --model llama3 --judge llama3              # with memory
.venv/bin/python -m evals.run --model llama3 --judge llama3 --no-memory  # baseline
.venv/bin/python -m evals.run --model llama3 --same-session              # upper bound
```

30 cases in [evals/datasets/memory_v0.jsonl](evals/datasets/memory_v0.jsonl): recall,
knowledge updates, temporal reasoning, preferences, multi-hop and abstention. Facts
are told in earlier sessions and asked about in a new one. An LLM judge compares
each answer with the case's reference answer (`--grader substring` is the free,
noisier alternative). Use a strong judge (`--judge opus`) when a key is available;
small local judges make mistakes too.

## Tests

```bash
.venv/bin/pytest            # no network or API keys needed
.venv/bin/ruff check .
```

## Layout

```
apps/api/        FastAPI server (SSE streaming) + web page (static/index.html)
apps/cli/        terminal chat
core/llm/        neutral types, provider adapters (Anthropic, OpenAI, Ollama), registry + router
core/agent/      orchestrator (agent loop), context budgeting, system prompt
core/tools/      tool registry with permission levels, built-in tools
core/memory/     episodic memory: embeddings, episode stores (in-memory, Postgres), engine
core/storage/    conversation store: in-memory and Postgres
config/          models.yaml
evals/           memory eval dataset + runner
migrations/      Alembic
```
