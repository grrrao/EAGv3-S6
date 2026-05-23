from __future__ import annotations

import json
import uuid

import action
import decision
import perception
from artifacts import ArtifactStore
from constants import MAX_ITERATIONS
from gateway import ensure_gateway
from memory import Memory
from mcp_session import mcp_session
from schemas import Goal, Observation


def _load_tools(mcp_tool_list) -> list[dict]:
    """Convert MCP tool entries to gateway tool format."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
        }
        for t in mcp_tool_list
    ]


def _final_answer(history: list[dict]) -> str:
    for event in reversed(history):
        if event["kind"] == "answer":
            return event["text"]
    return "No answer reached within the iteration limit."


async def run(query: str) -> str:
    ensure_gateway()

    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    mem = Memory()
    arts = ArtifactStore()

    print(f"[run] id={run_id} query={query[:80]}")
    try:
        mem.remember(query, source="user_query", run_id=run_id)
    except Exception as e:
        # Non-fatal: storing the query in memory is best-effort.
        # The run can still answer from existing memory and tool calls.
        print(f"[memory] query classification failed (non-fatal): {e}")

    async with mcp_session() as session:
        raw_tools = (await session.list_tools()).tools
        mcp_tools = _load_tools(raw_tools)
        print(f"[tools] {len(mcp_tools)} tools loaded")

        for it in range(1, MAX_ITERATIONS + 1):
            print(f"\n─── iter {it} ───")

            hits = mem.read(query, history)
            print(f"[memory.read] {len(hits)} hits")

            try:
                obs = perception.observe(query, hits, history, prior_goals, run_id)
            except Exception as e:
                # Non-fatal: if LLM gateway is exhausted, reuse prior goals.
                # If there are no prior goals (first iteration), seed with the
                # raw query as a single goal so decision gets one chance to run.
                print(f"[perception] LLM call failed (non-fatal): {e}")
                if prior_goals:
                    obs = Observation(goals=list(prior_goals))
                else:
                    fallback = Goal(id=f"g{run_id}_0", text=query, done=False)
                    obs = Observation(goals=[fallback])
                    print(f"[perception] seeding with raw query as fallback goal")
            prior_goals = obs.goals

            for g in obs.goals:
                marker = "done" if g.done else "open"
                line = f"[perception] [{marker}] {g.text}"
                if g.attach_artifact_id and not g.done:
                    line += f"\n             attach={g.attach_artifact_id}"
                print(line)

            if obs.all_done:
                print("[done] all goals satisfied")
                # If all goals were satisfied by tool actions (no answer yet),
                # synthesize a final confirmation with no tools available.
                if not any(e["kind"] == "answer" for e in history):
                    hits = mem.read(query, history)
                    sum_goal = Goal(id="summary", text=query, done=False)
                    try:
                        out = decision.next_step(sum_goal, hits, [], history, [])
                        if out.is_answer:
                            history.append({
                                "iter": it,
                                "kind": "answer",
                                "goal_id": "summary",
                                "text": out.answer,
                            })
                    except Exception as e:
                        print(f"[decision] synthesis LLM call failed (non-fatal): {e}")
                break

            goal = obs.next_unfinished()
            if goal is None:
                break

            attached: list[tuple[str, bytes]] = []
            if goal.attach_artifact_id and arts.exists(goal.attach_artifact_id):
                blob = arts.get_bytes(goal.attach_artifact_id)
                attached.append((goal.attach_artifact_id, blob))
                print(f"[attach] {goal.attach_artifact_id} ({len(blob)} bytes)")

            try:
                out = decision.next_step(goal, hits, attached, history, mcp_tools)
            except Exception as e:
                print(f"[decision] LLM call failed (aborting run): {e}")
                break

            if out.is_answer:
                print(f"[decision] ANSWER: {out.answer[:200]}")
                history.append({
                    "iter": it,
                    "kind": "answer",
                    "goal_id": goal.id,
                    "text": out.answer,
                })
                # Optimistically mark answered goal done; exit early if all done
                for g in prior_goals:
                    if g.id == goal.id:
                        g.done = True
                if all(g.done for g in prior_goals) and prior_goals:
                    print("[done] all goals satisfied by answer")
                    break
                continue

            tc = out.tool_call
            print(f"[decision] TOOL_CALL: {tc.name}({json.dumps(tc.arguments)[:120]})")
            result_text, art_id = await action.execute(session, tc, arts)
            print(f"[action] → {result_text[:120]}")

            # art: guard fired — model tried read_file/fetch_url on an artifact handle.
            # Retry Decision in the same iteration (free retry, no iter count burned).
            if result_text.startswith("ERROR: argument") and "artifact handle" in result_text:
                print("[retry] art: guard triggered — retrying decision without tool call")
                try:
                    out2 = decision.next_step(goal, hits, attached, history, mcp_tools)
                except Exception as e:
                    print(f"[decision] art-guard retry LLM call failed (aborting run): {e}")
                    break
                if out2.is_answer:
                    print(f"[decision] ANSWER: {out2.answer[:200]}")
                    history.append({"iter": it, "kind": "answer",
                                    "goal_id": goal.id, "text": out2.answer})
                    for g in prior_goals:
                        if g.id == goal.id:
                            g.done = True
                    if all(g.done for g in prior_goals) and prior_goals:
                        print("[done] all goals satisfied by answer")
                        break
                    continue
                # If still a tool call, fall through and record both to history normally

            try:
                mem.record_outcome(
                    tool_call=tc,
                    result_text=result_text,
                    artifact_id=art_id,
                    run_id=run_id,
                    goal_id=goal.id,
                )
            except Exception as e:
                print(f"[memory] record_outcome failed (non-fatal): {e}")
            history.append({
                "iter": it,
                "kind": "action",
                "goal_id": goal.id,
                "tool": tc.name,
                "arguments": tc.arguments,
                "result_descriptor": result_text[:300],
                "artifact_id": art_id,
            })

    answer = _final_answer(history)
    print(f"\n[final] {answer[:300]}")
    return answer
