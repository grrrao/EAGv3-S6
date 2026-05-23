# Session 6 — Agentic Architecture (EAG V3)

A from-scratch agentic system built around four cognitive roles — Memory, Perception, Decision, and Action — each with typed Pydantic v2 boundaries. All LLM calls are routed through **LLM Gateway V3** (`http://localhost:8101`); no provider SDK is called directly.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│                  agent6.py                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐ │
│  │ Memory   │  │Perception │  │ Decision │ │
│  │ (read /  │→ │ (goal     │→ │ (tool or │ │
│  │  write)  │  │  list)    │  │  answer) │ │
│  └──────────┘  └───────────┘  └──────────┘ │
│                                    │        │
│                               ┌────▼─────┐  │
│                               │  Action  │  │
│                               │ (MCP via │  │
│                               │  stdio)  │  │
│                               └──────────┘  │
└─────────────────────────────────────────────┘
```

### The four roles

| Role | File | Responsibility |
|------|------|----------------|
| **Memory** | `memory.py` | Keyword-search store (`state/memory.json`). Classifies incoming items (fact / preference / tool_outcome / scratchpad) via LLM, then retrieves top-K matches by token overlap. No vector DB. |
| **Perception** | `perception.py` | Maintains a typed goal list across iterations. Decomposes the query into bounded goals on the first call, marks goals done based on run history (not memory), and flags which artifact (if any) the next goal needs attached. |
| **Decision** | `decision.py` | Given the current goal, memory hits, attached artifact bytes, and recent history, emits exactly one of: a final answer (plain text) or a single MCP tool call. |
| **Action** | `action.py` | Executes the tool call against the MCP server. Results ≥ 4 KB are stored as content-addressable artifacts (`art:<sha256[:16]>`) so they don't bloat the prompt on every iteration. |

### Key design choices

- **Pydantic v2 on every boundary** — `Goal`, `Observation`, `ToolCall`, `DecisionOutput`, `MemoryItem` are all validated models.
- **Content-addressable artifact store** — large tool results (web pages, search results) are stored under `state/artifacts/` and referenced by `art:` handles. The most recent artifact is auto-attached to the next open goal that needs it.
- **Sticky-done goals** — once a goal is marked done by Perception, it stays done for the lifetime of the run.
- **Non-fatal degradation** — `mem.remember()`, `mem.record_outcome()`, and `perception.observe()` all fail gracefully; the run continues on empty or stale state rather than crashing.
- **Artifact guard** — if Decision tries to pass an `art:` handle as a tool argument, Action intercepts it and returns an error before the MCP call, then retries Decision with the bytes attached.

---

## LLM Gateway V3

All LLM calls go through the gateway's `/v1/chat` endpoint. The agent never imports a provider SDK.

**Routing** — each call carries `auto_route="perception"|"memory"|"decision"`. The gateway runs a lightweight router LLM to classify the request as TINY / LARGE and selects a provider from the tier-ordered failover list:

| Tier | Provider order |
|------|---------------|
| TINY | github → openrouter → groq → cerebras → gemini |
| LARGE | gemini → groq → cerebras → github → openrouter |

**Retry** — `gateway.py` wraps every call with up to 6 retries on 429 / 5xx. Backoff is the minimum `(Xs left)` value across all providers in the 503 body, capped at 90 s.

---

## MCP Tools (9 total)

Served by `mcp_session.py` over stdio transport:

| Tool | Purpose |
|------|---------|
| `fetch_url` | Fetch and scrape a URL (Crawl4AI) |
| `web_search` | DuckDuckGo / Tavily search |
| `remember_fact` | Persist a fact to memory |
| `read_file` | Read a local file |
| + 5 more | (list, write, etc.) |

---

## Running

```bash
# Prerequisites: LLM Gateway V3 running on port 8101
cd llm_gatewayV3 && ./run.sh

# Single query
uv run python run_query.py A       # Wikipedia fetch
uv run python run_query.py B       # Tokyo weather + activities
uv run python run_query.py C1      # Remember mom's birthday
uv run python run_query.py C2      # Recall mom's birthday
uv run python run_query.py D       # Search + read 3 pages + synthesise

# All five queries sequentially (15 s cooldown between each)
uv run python run_query.py ALL
```

---

## Validation

```bash
# Phase 1 — schema / memory / keyword checks
uv run python scripts/phase1_check.py

# Phase 2 — live gateway + full perception + decision round-trip
uv run python scripts/phase2_check.py

# Capture Proof-of-Performance JSON (system prompts + validated outputs)
uv run python scripts/capture_pop.py
# → writes runs/pop.json
```

---

## Project Structure

```
session6/
├── agent6.py          # Main run loop (orchestrator)
├── perception.py      # Goal list maintenance (Perception role)
├── decision.py        # Tool-or-answer selection (Decision role)
├── action.py          # MCP tool execution + artifact store writes (Action role)
├── memory.py          # Keyword memory read/write (Memory role)
├── gateway.py         # LLM Gateway V3 client wrapper + retry logic
├── artifacts.py       # Content-addressable blob store
├── mcp_session.py     # MCP stdio session setup
├── schemas.py         # Pydantic v2 models for all role boundaries
├── constants.py       # Tunable limits (MAX_ITERATIONS, temperatures, etc.)
├── run_query.py       # CLI entry point (A / B / C1 / C2 / D / ALL)
├── scripts/
│   ├── phase1_check.py   # Offline validation
│   ├── phase2_check.py   # Live round-trip validation
│   └── capture_pop.py    # Proof-of-Performance capture
├── runs/
│   └── pop.json          # Captured system prompts + validated role outputs
└── state/
    ├── memory.json        # Persistent keyword memory
    └── artifacts/         # Content-addressable tool-result blobs
```
