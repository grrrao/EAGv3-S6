"""Capture Perception and Decision prompts + validated JSON output for PoP.

Runs one iteration of a real query, captures the Observation and DecisionOutput
from the first perception/decision calls, and writes runs/pop.json containing:
  - perception_system_prompt  (intercepted at call time)
  - decision_system_prompt    (intercepted at call time)
  - sample_observation        (Observation.model_dump())
  - sample_decision           (DecisionOutput.model_dump())
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

STATE = Path("state")
RUNS = Path("runs")

QUERY = (
    "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his "
    "birth date, death date, and three key contributions to information theory."
)


async def capture() -> dict:
    import gateway as gw
    import decision
    import perception
    import action
    from artifacts import ArtifactStore
    from gateway import ensure_gateway
    from memory import Memory
    from mcp_session import mcp_session
    from schemas import Goal

    ensure_gateway()
    run_id = uuid.uuid4().hex[:8]
    mem = Memory()
    arts = ArtifactStore()

    captured: dict = {
        "perception_system_prompt": None,
        "decision_system_prompt": None,
        "sample_observation": None,
        "sample_decision": None,
    }

    # Patch llm_chat to intercept system prompts on the first relevant calls
    original_llm_chat = gw.llm_chat

    def intercepting_llm_chat(**kwargs):
        role = kwargs.get("auto_route", "")
        sys_prompt = kwargs.get("system", "")
        if role == "perception" and captured["perception_system_prompt"] is None:
            captured["perception_system_prompt"] = sys_prompt
        if role == "decision" and captured["decision_system_prompt"] is None:
            captured["decision_system_prompt"] = sys_prompt
        return original_llm_chat(**kwargs)

    gw.llm_chat = intercepting_llm_chat
    # Patch all modules that imported llm_chat directly
    import perception as _p
    import decision as _d
    _p.llm_chat = intercepting_llm_chat
    _d.llm_chat = intercepting_llm_chat

    mem.remember(QUERY, source="user_query", run_id=run_id)

    async with mcp_session() as session:
        raw_tools = (await session.list_tools()).tools
        mcp_tools = [
            {"name": t.name, "description": t.description or "",
             "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {}}
            for t in raw_tools
        ]

        prior_goals: list[Goal] = []
        history: list[dict] = []

        for it in range(1, 5):
            hits = mem.read(QUERY, history)
            obs = perception.observe(QUERY, hits, history, prior_goals, run_id)
            prior_goals = obs.goals

            if captured["sample_observation"] is None:
                captured["sample_observation"] = json.loads(obs.model_dump_json())

            if obs.all_done:
                break

            goal = obs.next_unfinished()
            if goal is None:
                break

            attached = []
            if goal.attach_artifact_id and arts.exists(goal.attach_artifact_id):
                blob = arts.get_bytes(goal.attach_artifact_id)
                attached.append((goal.attach_artifact_id, blob))

            out = decision.next_step(goal, hits, attached, history, mcp_tools)

            if captured["sample_decision"] is None:
                captured["sample_decision"] = json.loads(out.model_dump_json())

            if out.is_answer:
                history.append({"iter": it, "kind": "answer",
                                 "goal_id": goal.id, "text": out.answer})
                for g in prior_goals:
                    if g.id == goal.id:
                        g.done = True
                break

            result_text, art_id = await action.execute(session, out.tool_call, arts)
            mem.record_outcome(tool_call=out.tool_call, result_text=result_text,
                               artifact_id=art_id, run_id=run_id, goal_id=goal.id)
            history.append({"iter": it, "kind": "action", "goal_id": goal.id,
                             "tool": out.tool_call.name,
                             "arguments": out.tool_call.arguments,
                             "result_descriptor": result_text[:300],
                             "artifact_id": art_id})

    gw.llm_chat = original_llm_chat
    return captured


def main() -> None:
    if STATE.exists():
        shutil.rmtree(STATE)
    STATE.mkdir()
    RUNS.mkdir(exist_ok=True)

    pop = asyncio.run(capture())
    out_path = RUNS / "pop.json"
    out_path.write_text(json.dumps(pop, indent=2, default=str), encoding="utf-8")
    print(f"[pop] written to {out_path}")
    print(f"[pop] observation goals: {len((pop['sample_observation'] or {}).get('goals', []))}")
    dec = pop.get("sample_decision") or {}
    print(f"[pop] decision is_answer={dec.get('answer') is not None}, "
          f"tool_call={dec.get('tool_call') is not None}")
    print("[pop] perception_system_prompt captured:", pop["perception_system_prompt"] is not None)
    print("[pop] decision_system_prompt captured:", pop["decision_system_prompt"] is not None)


if __name__ == "__main__":
    main()
