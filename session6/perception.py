from __future__ import annotations

import json

from constants import PERCEPTION_TEMPERATURE, SYNTHESIS_KEYWORDS
from gateway import llm_chat
from schemas import Goal, MemoryItem, Observation, PerceptionRaw


def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    # MEMORY HITS section — integer-indexed, artifact-flagged
    hits_lines = []
    for i, item in enumerate(hits):
        line = f"[{i}] ({item.kind}) {item.descriptor}"
        # Only show value for facts/preferences — tool_outcome values expose art: handles
        # which cause the LLM to treat artifact IDs as file paths.
        if item.value and item.kind in ("fact", "preference", "scratchpad"):
            line += f" value={json.dumps(item.value)[:120]}"
        if item.artifact_id:
            line += f" [artifact_index={i}, artifact_id={item.artifact_id}]"
        hits_lines.append(line)
    hits_section = "\n".join(hits_lines) if hits_lines else "(none)"

    # Compact history transcript
    hist_lines = []
    for event in history:
        if event["kind"] == "action":
            args_str = json.dumps(event.get("arguments", {}))[:80]
            hist_lines.append(
                f"iter {event['iter']}: TOOL {event['tool']}({args_str}) "
                f"→ {event['result_descriptor'][:120]}"
            )
        elif event["kind"] == "answer":
            hist_lines.append(f"iter {event['iter']}: ANSWER {event['text'][:120]}")
    hist_section = "\n".join(hist_lines) if hist_lines else "(none)"

    # Prior goals section
    goals_lines = []
    for g in prior_goals:
        status = "done" if g.done else "open"
        goals_lines.append(f"[{status}] {g.text}")
    goals_section = "\n".join(goals_lines) if goals_lines else "(none)"

    system = """You are the Perception role in an agentic system. Your job is to maintain and update the goal list.

Four obligations (apply all four, in order):
1. If the prior goal list is empty, decompose the query into one or more bounded goals, each a short imperative statement.
2. For each prior goal, examine the RUN HISTORY section ONLY. Mark the goal done: true the moment the history contains an action that directly satisfies it. MEMORY HITS are background knowledge, NOT evidence of completed actions — never mark a goal done based on memory hits alone. Once done, the goal remains done in every subsequent iteration — never flip a done goal back to open.
3. For the first unfinished goal in the list, decide whether it needs raw bytes from a previously fetched artifact. If yes, set attach_artifact_index to the integer index (shown as artifact_index=N in MEMORY HITS) of the relevant artifact entry.
4. Preserve goal order exactly. Do not reorder, do not insert goals in the middle, do not drop a goal."""

    user = f"""QUERY: {query}

MEMORY HITS (use integer index for attach_artifact_index):
{hits_section}

RUN HISTORY:
{hist_section}

PRIOR GOALS:
{goals_section}

Return JSON with a "goals" array. Each element: text (string), done (bool), attach_artifact_index (int — use the artifact_index shown in MEMORY HITS, or -1 if no attachment needed)."""

    result = llm_chat(
        messages=[{"role": "user", "content": user}],
        system=system,
        auto_route="perception",
        temperature=PERCEPTION_TEMPERATURE,
        max_tokens=2048,
    )

    # No response_format — parse JSON from plain text. All providers can serve
    # this call (no "structured" capability required), and the existing fallbacks
    # handle partial/wrapped JSON from any model.
    raw_text = (result.get("text") or "").strip()
    if not raw_text and not result.get("parsed"):
        # Provider returned an empty body (e.g. Cerebras rate-limited at model
        # level but still returned HTTP 200). Raise so agent6 catches it and
        # falls back to prior goals rather than silently producing empty goals.
        raise ValueError("perception LLM returned empty response")

    parsed = result.get("parsed")
    if parsed is None and raw_text:
        text = raw_text
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed = None
    if not parsed:
        parsed = {"goals": []}

    try:
        raw = PerceptionRaw.model_validate(parsed)
    except Exception:
        raw = PerceptionRaw(goals=[])

    # Map positions → prior goal IDs; apply sticky-done; resolve artifact indices
    new_goals: list[Goal] = []
    for i, pg in enumerate(raw.goals):
        prior_id = prior_goals[i].id if i < len(prior_goals) else f"g{run_id}_{i}"
        done = pg.done or (i < len(prior_goals) and prior_goals[i].done)
        attach = None
        if pg.attach_artifact_index >= 0:
            idx = pg.attach_artifact_index
            if idx < len(hits) and hits[idx].artifact_id:
                attach = hits[idx].artifact_id
        new_goals.append(Goal(id=prior_id, text=pg.text, done=done, attach_artifact_id=attach))

    obs = Observation(goals=new_goals)

    # Force-attach safety net: use only artifacts produced in THIS run (from history).
    # Never attach stale artifacts from previous query runs stored in memory.
    first_open = obs.next_unfinished()
    if first_open and first_open.attach_artifact_id is None:
        # Collect artifact IDs produced by this run's actions, most recent last
        current_run_arts = [
            e["artifact_id"] for e in history
            if e["kind"] == "action" and e.get("artifact_id")
        ]
        if current_run_arts:
            goal_lower = first_open.text.lower()
            has_synthesis_kw = any(kw in goal_lower for kw in SYNTHESIS_KEYWORDS)
            # Attach if the goal looks like synthesis, OR unconditionally if we've
            # already fetched something (avoids extra round-trip to re-fetch)
            if has_synthesis_kw or current_run_arts:
                best = current_run_arts[-1]  # most recent artifact from this run
                for g in obs.goals:
                    if g.id == first_open.id:
                        g.attach_artifact_id = best
                        break

    return obs
