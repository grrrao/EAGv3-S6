from __future__ import annotations

import json

from constants import DECISION_TEMPERATURE
from gateway import llm_chat
from schemas import DecisionOutput, Goal, MemoryItem, ToolCall


def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    system = (
        "1. Respond with exactly one of two outputs: a final answer in plain text, OR "
        "a single tool call. Never both. Never narrate before a tool call.\n"
        "2. Strings beginning with art: are internal artifact handles. They are NOT "
        "paths or URLs. Never pass art:... to read_file, fetch_url, or any other tool. "
        "When a goal requires the bytes of an artifact, those bytes appear in the "
        "ATTACHED ARTIFACTS section. Read them there.\n"
        "3. When the goal asks for an extraction, a list, a comparison, or a selection, "
        "the answer must be substantive: at least three sentences or a list of items. "
        "Do not return meta-answers like \"the page has been fetched, how would you like to proceed?\"."
    )

    # Memory hits (short descriptors only)
    hits_lines = []
    for item in hits:
        line = f"- ({item.kind}) {item.descriptor}"
        # Show structured value only for facts/preferences — NOT tool_outcome, which
        # contains art: handles that tempt the model to call read_file(art:...).
        if item.value and item.kind in ("fact", "preference", "scratchpad"):
            line += f" value={json.dumps(item.value)[:120]}"
        if item.artifact_id:
            line += f" [artifact={item.artifact_id}]"
        hits_lines.append(line)
    hits_section = "\n".join(hits_lines) if hits_lines else "(none)"

    # Recent history (last 6 events)
    recent = history[-6:]
    hist_lines = []
    for event in recent:
        if event["kind"] == "action":
            args_str = json.dumps(event.get("arguments", {}))[:80]
            hist_lines.append(
                f"TOOL {event['tool']}({args_str}) → {event['result_descriptor'][:150]}"
            )
        elif event["kind"] == "answer":
            hist_lines.append(f"ANSWER {event['text'][:150]}")
    hist_section = "\n".join(hist_lines) if hist_lines else "(none)"

    # Attached artifact bytes
    artifact_parts = []
    for art_id, blob in attached:
        content = blob.decode("utf-8", errors="replace")[:6000]
        artifact_parts.append(
            f"=== ARTIFACT {art_id} ===\n{content}\n=== END {art_id} ==="
        )
    attached_section = "\n\n".join(artifact_parts)

    parts = [
        f"CURRENT GOAL: {goal.text}",
        f"\nMEMORY HITS:\n{hits_section}",
        f"\nRECENT HISTORY:\n{hist_section}",
    ]
    if attached_section:
        parts.append(f"\nATTACHED ARTIFACTS:\n{attached_section}")
    parts.append("\nEither answer the goal directly in plain text, OR call a tool.")

    user_prompt = "\n".join(parts)

    result = llm_chat(
        messages=[{"role": "user", "content": user_prompt}],
        system=system,
        tools=mcp_tools,
        tool_choice="auto",
        auto_route="decision",
        temperature=DECISION_TEMPERATURE,
        max_tokens=2048,
    )

    if result.get("tool_calls"):
        tc = result["tool_calls"][0]
        # Gateway canonical shape uses "arguments" (confirmed from schemas.py ToolCall)
        args = tc.get("arguments") or tc.get("input") or {}
        return DecisionOutput(tool_call=ToolCall(name=tc["name"], arguments=args))

    return DecisionOutput(answer=result.get("text") or "")
