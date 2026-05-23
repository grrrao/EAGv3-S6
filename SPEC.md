# EAG V3 Session 6 — Agentic Architecture: Build Spec

This spec drives a Claude Code implementation of the Session 6 agent. The agent
replaces the monolithic `agent5.py` loop with four typed cognitive roles —
**Memory, Perception, Decision, Action** — communicating through Pydantic
models, on top of two pre-built substrates: an MCP server (`mcp_server.py`) and
`llm_gatewayV3`.

This is a **clean rewrite from scratch**, not a refactor of agent5.

---

## 0. Ground truth — what already exists, do not modify

These two are pre-built. The agent calls them; the agent does not change them.

### `mcp_server.py` (stdio transport)
Exposes nine tools:
`web_search`, `fetch_url`, `get_time`, `currency_convert`, `read_file`,
`list_dir`, `create_file`, `update_file`, `edit_file`.

- File tools are sandboxed under `./sandbox/`.
- `web_search` is hard-capped at 5 results (Tavily → DDG fallback).
- `fetch_url` returns `{status, content_type, length_bytes, text}` — clean
  markdown from crawl4ai.
- Connect via FastMCP stdio client (`mcp.client.stdio.stdio_client`,
  `mcp.ClientSession`). Tool list comes from `session.list_tools()`.

### `llm_gatewayV3` (HTTP on `http://localhost:8101`)
- Use `client.LLM` from the gateway package. Do **not** import provider SDKs
  directly.
- `llm.chat(...)` returns a dict with keys: `text`, `tool_calls`, `parsed`,
  `stop_reason`, `input_tokens`, `output_tokens`, `latency_ms`,
  `router_decision`, `reasoning_applied`, etc.
- For structured output, pass `response_format={"type":"json_schema","schema": SomeModel.model_json_schema()}` and read `result["parsed"]` (already a dict).
- For tool use, pass `tools=<list>`, `tool_choice="auto"` and read
  `result["tool_calls"]` (canonical list).
- `auto_route` accepts `"perception" | "memory" | "decision"`.
- Explicit `provider="g"` (Gemini) wins over `auto_route`.

**The gateway is the only LLM substrate. No `openai`/`google-generativeai`/etc.
imports anywhere in the agent code.**

---

## 1. Project layout

```
session6/
├── agent6.py            # the loop
├── schemas.py           # all Pydantic models
├── memory.py            # Memory service (read, write, persistence)
├── perception.py        # Perception role
├── decision.py          # Decision role
├── action.py            # Action role
├── artifacts.py         # Content-addressable artifact store
├── gateway.py           # Thin wrapper exposing LLM() with a `ensure_gateway()` health check
├── mcp_session.py       # async context manager wrapping stdio_client + ClientSession
├── run_query.py         # CLI: python run_query.py A|B|C|D
├── state/               # runtime, gitignored
│   ├── memory.json
│   └── artifacts/
│       ├── <id>.bin
│       └── <id>.json
├── pyproject.toml       # uv-managed
├── .gitignore           # excludes state/, .env, __pycache__, *.pyc
└── README.md            # Phase 5
```

Dependency manager: **uv**. No manual venv activation.
Required deps: `pydantic>=2`, `httpx`, `mcp[cli]`, plus whatever the gateway
client and `mcp_server.py` need (already in their requirements).

---

## 2. Locked constants

```python
ARTIFACT_THRESHOLD_BYTES = 4096          # 4 KB — bytes above this go to artifacts
MAX_ITERATIONS = 15                      # per query, hard cap
MEMORY_TOP_K = 8                         # default for memory.read
SYNTHESIS_KEYWORDS = {                   # for force-attach safety net (Query D)
    "synthesise", "synthesize", "extract", "list", "compare", "decide",
}
PERCEPTION_TEMPERATURE = 1.0             # Gemini 3 loops at low temp on structured output
DECISION_TEMPERATURE = 0.7               # gateway default
```

---

## 3. Hard rules (apply across every phase)

1. **Pydantic v2 on every boundary.** Every role's input and output is a model
   in `schemas.py`. No free-form dicts crossing role boundaries. The `history`
   list inside the loop is `list[dict]`, that is the *only* exception, and each
   dict mirrors one of the typed shapes.
2. **No regex on LLM reasoning output.** Structured outputs via
   `response_format` replace string parsing. String inspection of *tool
   arguments* (e.g. checking if a path starts with `art:`) is fine — that is
   not parsing LLM reasoning.
3. **Gateway is the only LLM substrate.** No direct provider SDK calls.
4. **No third-party agentic frameworks.** No LangGraph, LangChain, CrewAI.
5. **Perception and the Memory classifier write both pin to Gemini** via
   `provider="g"`. Decision uses `auto_route="decision"`. The keyword
   `memory.read` makes no LLM call.
6. **`state/` must be safe to delete** between attempts. The agent must
   recreate any directories it needs.
7. **MCP dispatch uses real stdio + `ClientSession.call_tool`.** No
   hand-rolled tool dispatch.
8. **Tools the gateway sees in Decision come from
   `session.list_tools()`** — converted to the gateway's tool format. Don't
   hardcode the tool list.

---

# Phase 1 — Schemas + Memory + Artifacts

Goal: produce three modules that compile, persist, and round-trip cleanly with
no LLM calls except `memory.remember`. End-of-phase check is a small script
that exercises remember/read/record_outcome/put/get.

### `schemas.py`

Exactly these models, Pydantic v2:

```python
class MemoryItem(BaseModel):
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str          # one short human-readable line
    value: dict              # structured payload
    artifact_id: str | None = None
    source: str
    run_id: str
    goal_id: str | None = None
    confidence: float = 1.0
    created_at: datetime

class Artifact(BaseModel):
    id: str                  # "art:<sha256-prefix>"
    content_type: str
    size_bytes: int
    source: str
    descriptor: str
    created_at: datetime

class Goal(BaseModel):
    id: str
    text: str
    done: bool = False
    attach_artifact_id: str | None = None

class Observation(BaseModel):
    goals: list[Goal]

    @property
    def all_done(self) -> bool: ...
    def next_unfinished(self) -> Goal | None: ...

class ToolCall(BaseModel):
    name: str
    arguments: dict

class DecisionOutput(BaseModel):
    answer: str | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        # exactly one of answer / tool_call must be set
        ...

    @property
    def is_answer(self) -> bool: ...
```

Plus a `PerceptionRaw` model for what the LLM emits (without `Goal.id`,
because goal identity is positional — the loop assigns ids). Shape:

```python
class _PerceivedGoal(BaseModel):
    text: str
    done: bool = False
    attach_artifact_index: int | None = None   # index into MEMORY HITS, NOT a handle string

class PerceptionRaw(BaseModel):
    goals: list[_PerceivedGoal]
```

Same for Memory's classifier output:

```python
class _ClassifiedItem(BaseModel):
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict
    confidence: float = 1.0
```

### `artifacts.py`

A `ArtifactStore` class backed by `state/artifacts/`:

```python
class ArtifactStore:
    def put(self, blob: bytes, *, content_type: str, source: str,
            descriptor: str) -> str: ...
    def get_bytes(self, artifact_id: str) -> bytes: ...
    def get_meta(self, artifact_id: str) -> Artifact: ...
    def exists(self, artifact_id: str) -> bool: ...
```

- `id = "art:" + sha256(blob).hexdigest()[:16]`. Content-addressable, so
  identical fetches dedupe by overwriting safely.
- Files: `state/artifacts/<id_no_prefix>.bin` and `.json`.
- `put` is idempotent: if the file exists, do not overwrite the metadata
  `created_at` (but it is fine to no-op the write).
- No eviction.

### `memory.py`

A `Memory` class backed by `state/memory.json`:

```python
class Memory:
    def __init__(self, path: Path = Path("state/memory.json")): ...

    # READS — no LLM
    def read(self, query: str, history: list[dict],
             kinds: list[str] | None = None,
             top_k: int = MEMORY_TOP_K) -> list[MemoryItem]: ...
    def filter(self, *, kinds: list[str] | None = None,
               goal_id: str | None = None,
               recent: int | None = None) -> list[MemoryItem]: ...

    # WRITES
    def remember(self, raw_text: str, *, source: str, run_id: str,
                 goal_id: str | None = None) -> MemoryItem | None:
        # one LLM call via gateway, provider="g", response_format=_ClassifiedItem
        # If classification yields a scratchpad item that is empty/noisy,
        # the implementation MAY return None (e.g. confidence < 0.3) — but
        # only after recording the attempt nowhere. Default to persisting.
        ...

    def record_outcome(self, *, tool_call: ToolCall, result_text: str,
                       artifact_id: str | None, run_id: str,
                       goal_id: str | None) -> MemoryItem:
        # no LLM. kind="tool_outcome".
        # keywords = tokenize(tool_call.name + " " + " ".join(str(v) for v in tool_call.arguments.values()))
        ...
```

**Read algorithm (no LLM):**

```
tokens = lowercase_tokens(query) - STOPWORDS
for item in items (optionally filtered by `kinds`):
    item_tokens = set(item.keywords) | lowercase_tokens(item.descriptor)
    score = len(tokens & item_tokens) / max(1, len(tokens))
    # tiebreak: more recent items first
return top_k by (score desc, created_at desc), score > 0 only
```

Stopwords: a small fixed set — `{"the","a","an","and","or","of","to","for",
"in","on","at","is","are","what","when","who","how","my","me","i","you","your"}`.

**Persistence:** load lazily on first call; write the whole file after every
mutation. JSON-encode `datetime` as ISO strings.

**Remember LLM call:** `provider="g"`, `response_format={"type":"json_schema",
"schema": _ClassifiedItem.model_json_schema()}`, `temperature=0`. System prompt
roughly:

> You classify a single user statement into one of four memory kinds: fact
> (durable observed truth), preference (user-stated or inferred preference),
> tool_outcome (record of a tool call — almost never used for raw user text),
> scratchpad (working note for this run only). Extract keywords (lowercased,
> deduped) and a structured `value` dict. Return only JSON conforming to the
> schema.

### Phase 1 sanity script

Create `scripts/phase1_check.py` (gitignored). It:

1. Cleans `state/`.
2. `mem.remember("Mom's birthday is 15 May 2026", source="user_query", run_id="r1")`
   → asserts a fact item is written with keywords containing `mom`, `birthday`,
   `may`, `2026` (case-insensitive).
3. `mem.read("when is mom's birthday")` → asserts at least one hit, top hit
   is the fact above.
4. Puts a 50 KB blob into the artifact store, reads it back by handle, asserts
   bytes equal. Asserts `exists(handle)` is True and `exists("art:doesnotexist")` is False.
5. `mem.record_outcome(...)` for a fake `fetch_url` call with an artifact id →
   asserts the tool_outcome item carries the artifact_id and that
   `mem.read("fetch_url")` finds it.

When all five pass, Phase 1 is done.

---

# Phase 2 — Perception + the Action layer

These two are independent of each other but both feed Phase 3, so they go
together.

### `perception.py`

```python
def observe(query: str, hits: list[MemoryItem], history: list[dict],
            prior_goals: list[Goal], run_id: str) -> Observation: ...
```

**Inputs the LLM sees (built by the function):**

- The user query.
- A `MEMORY HITS` section, numbered, each line including any `artifact_id` and
  marked with its index `i` so the model can reference it as
  `attach_artifact_index: i`.
- A compact history transcript (one line per event; the
  `result_descriptor` field for actions, `text` for answers).
- The prior goals section: each prior goal with its `text` and `done` flag,
  in order.

**The four obligations (put these verbatim in the system prompt):**

1. If the prior goal list is empty, decompose the query into one or more
   bounded goals, each a short imperative statement.
2. For each prior goal, examine the run history. Mark the goal `done: true`
   the moment the history contains an action that satisfies it. Once done,
   the goal remains done in every subsequent iteration.
3. For the first unfinished goal in the list, decide whether it needs raw
   bytes from a previously fetched artifact. If yes, set
   `attach_artifact_index` to the integer index of the artifact in
   `MEMORY HITS`.
4. Preserve goal order. Do not reorder, do not insert in the middle, do not
   drop a goal.

**LLM call:**
- `provider="g"`, `temperature=1.0`,
  `response_format={"type":"json_schema","schema": PerceptionRaw.model_json_schema()}`.
- Parse `result["parsed"]` into `PerceptionRaw`.

**Position-based identity (the loop's job, not the LLM's):**

```
new_goals = []
for i, pg in enumerate(parsed.goals):
    # map by position to prior goals
    prior_id = prior_goals[i].id if i < len(prior_goals) else f"g{run_id}_{i}"
    # sticky-done: once True, stays True
    done = pg.done or (i < len(prior_goals) and prior_goals[i].done)
    # resolve artifact_index → handle via MEMORY HITS
    attach = None
    if pg.attach_artifact_index is not None:
        attach = hits[pg.attach_artifact_index].artifact_id if 0 <= pg.attach_artifact_index < len(hits) else None
    new_goals.append(Goal(id=prior_id, text=pg.text, done=done, attach_artifact_id=attach))
```

**Force-attach safety net (Query D):** after the LLM output is parsed and
goals are reconstructed, if the first unfinished goal's `text` contains any
`SYNTHESIS_KEYWORDS` (lowercase compare) and `attach_artifact_id is None`,
attach the most recent artifact found in `hits` (by `created_at` on the
artifact metadata, or order-in-hits as fallback).

### `action.py`

```python
async def execute(session: ClientSession, tool_call: ToolCall,
                  artifacts: ArtifactStore) -> tuple[str, str | None]:
    # 1. art: guard
    for k, v in tool_call.arguments.items():
        if isinstance(v, str) and v.startswith("art:"):
            return (f"ERROR: argument '{k}' is an artifact handle, not a path/url. "
                    f"Artifact bytes are attached to your prompt when needed."), None

    # 2. dispatch
    result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)

    # 3. collapse content blocks
    text = "".join(block.text for block in result.content if hasattr(block, "text"))

    # 4. threshold
    blob = text.encode("utf-8")
    if len(blob) >= ARTIFACT_THRESHOLD_BYTES:
        art_id = artifacts.put(blob,
                               content_type="text/plain",
                               source=f"tool:{tool_call.name}",
                               descriptor=f"{tool_call.name} -> {text[:80]}")
        preview = text[:200].replace("\n", " ")
        return f"[artifact {art_id}, {len(blob)} bytes] preview: {preview}", art_id
    return text, None
```

### Phase 2 sanity script

`scripts/phase2_check.py`:

1. Mock `MemoryItem` list with one artifact-bearing item and one without.
   Call `perception.observe` with a fresh query
   ("Fetch the Shannon Wikipedia page and tell me his birth date").
   Assert at least 2 goals, none done, none with attach on the first call.
2. Simulate one iteration of history with a `fetch_url` outcome. Re-run
   `perception.observe` — assert the fetch goal is now `done`, and the
   extraction goal carries `attach_artifact_id` pointing at the
   artifact in the hits.
3. Spin up the MCP server via stdio, call `action.execute` with a real
   `get_time` ToolCall — assert descriptor is short, no artifact created.
4. Call `action.execute` with `fetch_url` on a real URL — assert an artifact
   is created and the descriptor begins with `[artifact art:`.
5. Call `action.execute` with `read_file({"path":"art:abc"})` — assert the
   error branch fires, no MCP dispatch happens, returns `(error_string, None)`.

When all five pass, Phase 2 is done.

---

# Phase 3 — Decision + the Agent Loop

### `decision.py`

```python
def next_step(goal: Goal, hits: list[MemoryItem],
              attached: list[tuple[str, bytes]], history: list[dict],
              mcp_tools: list[dict]) -> DecisionOutput: ...
```

**Prompt sections built by the function:**

- System: the three substantive rules (below).
- The current `goal.text`.
- A `MEMORY HITS` section (short descriptors, no bytes).
- A compact recent history (last 6 events).
- An `ATTACHED ARTIFACTS:` section. For each `(art_id, bytes)`, decode UTF-8
  and embed up to 80,000 chars. Multiple artifacts get separators.
- The MCP tool list as the gateway's `tools` parameter.

**System prompt — three rules, verbatim:**

> 1. Respond with exactly one of two outputs: a final answer in plain text, OR
>    a single tool call. Never both. Never narrate before a tool call.
> 2. Strings beginning with `art:` are internal artifact handles. They are NOT
>    paths or URLs. Never pass `art:...` to `read_file`, `fetch_url`, or any
>    other tool. When a goal requires the bytes of an artifact, those bytes
>    appear in the ATTACHED ARTIFACTS section above. Read them there.
> 3. When the goal asks for an extraction, a list, a comparison, or a
>    selection, the answer must be substantive: at least three sentences or a
>    list of items. Do not return meta-answers like "the page has been
>    fetched, how would you like to proceed?".

**LLM call:**

```python
result = llm.chat(
    messages=[{"role":"user","content": user_prompt}],
    system=system_prompt,
    tools=mcp_tools,
    tool_choice="auto",
    auto_route="decision",
    temperature=DECISION_TEMPERATURE,
)
```

**Output mapping:**

```python
if result.get("tool_calls"):
    tc = result["tool_calls"][0]
    return DecisionOutput(tool_call=ToolCall(name=tc["name"], arguments=tc["input"]))
return DecisionOutput(answer=result["text"])
```

(The exact tool-call key may be `input` or `arguments` depending on the
gateway's canonical shape — implementer should print one call early in Phase 3
to confirm and adapt. The gateway returns a canonical list per the README.)

**MCP-tool-to-gateway-tool conversion** (utility used by the loop):
Convert each MCP tool entry from `session.list_tools()` to the gateway's
expected shape: `{"name": t.name, "description": t.description,
"input_schema": t.inputSchema}`.

### `agent6.py` — the loop

```python
async def run(query: str) -> str:
    ensure_gateway()                                  # raises with clear msg if 8101 dead
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    memory.remember(query, source="user_query", run_id=run_id)

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)         # list[dict] for gateway
        for it in range(1, MAX_ITERATIONS + 1):
            print(f"─── iter {it} ───")
            hits = memory.read(query, history)
            print(f"[memory.read] {len(hits)} hits")

            obs = perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals
            for g in obs.goals:
                marker = "done" if g.done else "open"
                line = f"[perception] [{marker}] {g.text}"
                if g.attach_artifact_id and not g.done:
                    line += f"\n             attach={g.attach_artifact_id}"
                print(line)

            if obs.all_done:
                print("[done] all goals satisfied")
                break

            goal = obs.next_unfinished()
            attached: list[tuple[str, bytes]] = []
            if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                attached.append((goal.attach_artifact_id,
                                 artifacts.get_bytes(goal.attach_artifact_id)))
                print(f"[attach] {goal.attach_artifact_id} "
                      f"({len(attached[0][1])} bytes)")

            out = decision.next_step(goal, hits, attached, history, mcp_tools)
            if out.is_answer:
                print(f"[decision] ANSWER: {out.answer[:200]}")
                history.append({"iter": it, "kind": "answer",
                                "goal_id": goal.id, "text": out.answer})
                continue

            print(f"[decision] TOOL_CALL: {out.tool_call.name}({json.dumps(out.tool_call.arguments)[:120]})")
            result_text, art_id = await action.execute(session, out.tool_call, artifacts)
            print(f"[action] → {result_text[:120]}")

            memory.record_outcome(tool_call=out.tool_call, result_text=result_text,
                                  artifact_id=art_id, run_id=run_id, goal_id=goal.id)
            history.append({"iter": it, "kind": "action", "goal_id": goal.id,
                            "tool": out.tool_call.name,
                            "arguments": out.tool_call.arguments,
                            "result_descriptor": result_text[:300],
                            "artifact_id": art_id})

    return final_answer_from(history)
```

`final_answer_from(history)` returns the last `kind=="answer"` event's `text`
(or a "no answer reached" string if the loop exited on `MAX_ITERATIONS`).

### `run_query.py`

```
python run_query.py A      # Shannon Wikipedia
python run_query.py B      # Tokyo activities
python run_query.py C1     # Mom's birthday — run 1 (records fact)
python run_query.py C2     # Mom's birthday — run 2 (recalls)
python run_query.py D      # asyncio research
```

The query texts are the literal strings from the session doc. C1 and C2 share
the same `state/`; C1 must not be re-run between C1 and C2 (the README will
say so).

### Phase 3 sanity check

Just run Query A end-to-end. Expected:

- 2–3 iterations.
- Iter 1: a `fetch_url` tool call. Artifact created.
- Iter 2: `perception` attaches the artifact to goal 2. `decision` returns
  an answer mentioning 1916 and 2001.

When Query A passes, Phase 3 is done.

---

# Phase 4 — Pass all four target queries

Apply the agent to B, C1+C2, D. Each must complete within `MAX_ITERATIONS = 15`
(the passing bar is "twice the expected count" per the doc, which is well
under 15 for all four).

For each query, record the **clean-state terminal output** to
`runs/<query>.log` for use in Phase 5. "Clean state" = `rm -rf state/` before
running, except between C1 and C2.

**Expected behaviour per query** (from the doc, used as acceptance criteria):

- **Query A** — Shannon Wikipedia. 2–3 iters. Final answer includes
  `April 30, 1916` and `February 24, 2001` and three contributions.
- **Query B** — Tokyo. 3–6 iters. Final answer names a specific activity
  appropriate to the fetched Saturday weather (rain → indoor; sun → either).
- **Query C1** — Mom's birthday. 3–4 iters. At least one `create_file` call
  under `sandbox/`. `state/memory.json` afterwards contains a fact with
  keywords matching mom/birthday/may/2026.
- **Query C2** — recall. 1–2 iters. The fact is found via `memory.read`
  before any tool call; the final answer states the date directly. (It is
  acceptable for Decision to also call `list_dir` once before answering.)
- **Query D** — asyncio synthesis. 5–7 iters. Three `fetch_url` calls,
  three artifacts. Final answer is a numbered list of common advice.

If a query exceeds 15 iterations or fails the acceptance check: investigate
the role boundary that broke (usually Perception goal mutation, sticky-done
not sticking, or Decision returning a meta-answer). Tune prompts, do not
add logic to the loop.

---

# Phase 5 — README, terminal capture, PoP, video

(Defer until Phases 1–4 are green.)

To produce:

1. `README.md` documenting setup (`uv sync`, `.env` keys for Tavily +
   gateway), how to run each query, and notes on `state/` cleanup.
2. The captured terminal output for each query, inlined in the README (or
   in `runs/`).
3. The "Perception and Decision Prompt and Validation JSON of PoP" —
   meaning the full system prompts used for Perception and Decision, plus a
   sample validated `Observation` JSON and a sample validated
   `DecisionOutput` JSON, captured from a real run.
4. The YouTube link for the video walkthrough.

---

# Open questions to confirm before Phase 1

These are the things I'd flag to Claude Code at start of Phase 1 if anything
about the substrates differs from what I read:

1. The gateway `client.LLM.chat()` returns `result["tool_calls"]` as a
   canonical list. Each entry's argument key should be `input` (per the
   gateway README example), but the implementer should print one tool call
   in Phase 3 to confirm and adapt the Decision output mapping if needed.
2. `result["parsed"]` is expected to be a dict already validated against the
   submitted schema. If it ever comes back `None` while `text` contains
   JSON, Memory.remember and Perception.observe should fall back to
   `json.loads(result["text"])` and re-validate via the Pydantic model.
3. The `provider="g"` shortcut for Gemini is documented in the gateway
   README. If a different shortcut is required, change in one place
   (`gateway.py`).

---

# Acceptance summary

The build is done when:

- [ ] Phase 1 sanity script (memory + artifacts) passes
- [ ] Phase 2 sanity script (perception + action) passes
- [ ] Phase 3: Query A completes in ≤ 3 iterations with correct facts
- [ ] Phase 4: Queries B, C1, C2, D all pass acceptance criteria, each in
      ≤ 15 iterations
- [ ] `state/` is .gitignored and the agent recreates anything it needs
- [ ] No `openai`/`google-generativeai`/etc. imports in agent code
- [ ] No regex on LLM reasoning output anywhere
- [ ] Pydantic v2 on every role boundary
- [ ] Phase 5 deliverables (README + captures + PoP + video) — last
