"""Phase 2 sanity check: perception + action."""
from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

STATE = Path("state")


def clean_state() -> None:
    if STATE.exists():
        shutil.rmtree(STATE)
    STATE.mkdir()
    print("[setup] state/ cleaned")


def make_item(kind, descriptor, artifact_id=None):
    from schemas import MemoryItem
    return MemoryItem(
        id=uuid.uuid4().hex[:8],
        kind=kind,
        keywords=descriptor.lower().split()[:5],
        descriptor=descriptor,
        value={},
        artifact_id=artifact_id,
        source="test",
        run_id="r_test",
        created_at=datetime.now(timezone.utc),
    )


def test_perception_first_call() -> None:
    import perception
    query = "Fetch the Shannon Wikipedia page and tell me his birth date"
    hits = [
        make_item("fact", "Shannon biographical info", artifact_id=None),
        make_item("fact", "Information theory history", artifact_id=None),
    ]
    run_id = uuid.uuid4().hex[:8]
    obs = perception.observe(query, hits, [], [], run_id)
    assert len(obs.goals) >= 2, f"Expected ≥2 goals, got {len(obs.goals)}"
    assert not any(g.done for g in obs.goals), "No goals should be done on first call"
    assert obs.goals[0].attach_artifact_id is None, "No artifact to attach on first call"
    print(f"[1] perception first call OK — {len(obs.goals)} goals, all open")
    return obs


def test_perception_with_history(first_obs) -> None:
    import perception
    from artifacts import ArtifactStore
    query = "Fetch the Shannon Wikipedia page and tell me his birth date"
    run_id = uuid.uuid4().hex[:8]

    # Create a real artifact
    store = ArtifactStore()
    art_id = store.put(
        b"Wikipedia content about Claude Shannon " * 200,
        content_type="text/plain",
        source="tool:fetch_url",
        descriptor="fetch_url -> Wikipedia Shannon page",
    )

    hits = [
        make_item("tool_outcome", "fetch_url Wikipedia Shannon", artifact_id=art_id),
    ]
    history = [{
        "iter": 1,
        "kind": "action",
        "goal_id": first_obs.goals[0].id,
        "tool": "fetch_url",
        "arguments": {"url": "https://en.wikipedia.org/wiki/Claude_Shannon"},
        "result_descriptor": f"[artifact {art_id}, 7400 bytes] preview: Wikipedia...",
        "artifact_id": art_id,
    }]

    obs = perception.observe(query, hits, history, first_obs.goals, run_id)
    fetch_goal = obs.goals[0]
    assert fetch_goal.done, f"Fetch goal should be done after fetch action"
    extract_goal = obs.next_unfinished()
    assert extract_goal is not None, "Should still have an unfinished goal"
    assert extract_goal.attach_artifact_id == art_id, (
        f"Extract goal should attach artifact {art_id}, got {extract_goal.attach_artifact_id}"
    )
    print(f"[2] perception with history OK — fetch=done, extract attaches art_id={art_id[:20]}...")


async def test_action_get_time() -> None:
    import action
    from mcp_session import mcp_session
    from schemas import ToolCall

    from artifacts import ArtifactStore
    store = ArtifactStore()
    tc = ToolCall(name="get_time", arguments={"timezone": "UTC"})
    async with mcp_session() as session:
        text, art_id = await action.execute(session, tc, store)
    assert art_id is None, f"get_time should not create artifact, got {art_id}"
    assert "UTC" in text or "utc" in text.lower() or len(text) < 4096, f"Unexpected result: {text[:100]}"
    print(f"[3] action get_time OK — no artifact, result: {text[:60]}")


async def test_action_fetch_url() -> None:
    import action
    from mcp_session import mcp_session
    from schemas import ToolCall

    from artifacts import ArtifactStore
    store = ArtifactStore()
    tc = ToolCall(name="fetch_url", arguments={"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"})
    async with mcp_session() as session:
        text, art_id = await action.execute(session, tc, store)
    assert art_id is not None, "fetch_url should create an artifact for a real URL"
    assert text.startswith("[artifact art:"), f"Expected artifact descriptor, got: {text[:80]}"
    print(f"[4] action fetch_url OK — artifact={art_id}")


async def test_action_art_guard() -> None:
    import action
    from mcp_session import mcp_session
    from schemas import ToolCall

    from artifacts import ArtifactStore
    store = ArtifactStore()
    tc = ToolCall(name="read_file", arguments={"path": "art:abc123"})
    async with mcp_session() as session:
        text, art_id = await action.execute(session, tc, store)
    assert art_id is None
    assert "ERROR" in text and "artifact handle" in text
    print(f"[5] action art: guard OK — error returned, no dispatch")


async def run_all() -> None:
    clean_state()
    first_obs = test_perception_first_call()
    test_perception_with_history(first_obs)
    await test_action_get_time()
    await test_action_fetch_url()
    await test_action_art_guard()
    print("\n✓ All Phase 2 checks passed")


if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}", file=sys.stderr)
        sys.exit(1)
