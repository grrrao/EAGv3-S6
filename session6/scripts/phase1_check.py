"""Phase 1 sanity check: memory + artifacts, no loop."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Run from session6/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

STATE = Path("state")


def clean_state() -> None:
    if STATE.exists():
        shutil.rmtree(STATE)
    STATE.mkdir()
    print("[setup] state/ cleaned")


def test_remember_fact() -> None:
    from memory import Memory
    mem = Memory()
    item = mem.remember(
        "Mom's birthday is 15 May 2026",
        source="user_query",
        run_id="r1",
    )
    assert item is not None, "remember() returned None"
    assert item.kind == "fact", f"Expected fact, got {item.kind}"
    kw_lower = {k.lower() for k in item.keywords}
    all_kw_text = " ".join(kw_lower)
    # LLM may return "mom" or "mom's"→"moms"; accept either
    assert any(kw in kw_lower for kw in ("mom", "moms")), f"mom keyword missing from {kw_lower}"
    for expected in ("birthday", "may", "2026"):
        # Accept exact keyword or embedded in a compound keyword (e.g. "15may2026")
        assert expected in kw_lower or expected in all_kw_text, f"Keyword '{expected}' missing from {kw_lower}"
    print(f"[1] remember fact OK — keywords={item.keywords}")


def test_read_fact() -> None:
    from memory import Memory
    mem = Memory()
    hits = mem.read("when is mom's birthday", [])
    assert hits, "read() returned no hits"
    assert hits[0].kind == "fact"
    print(f"[2] read fact OK — top hit: {hits[0].descriptor!r}")


def test_artifact_round_trip() -> None:
    from artifacts import ArtifactStore
    store = ArtifactStore()
    blob = b"x" * 51200  # 50 KB
    art_id = store.put(blob, content_type="application/octet-stream", source="test", descriptor="50KB blob")
    assert art_id.startswith("art:"), f"Bad art_id: {art_id}"
    assert store.exists(art_id), "exists() returned False after put"
    assert store.get_bytes(art_id) == blob, "get_bytes mismatch"
    assert not store.exists("art:doesnotexist"), "exists() should be False for unknown id"
    print(f"[3] artifact round-trip OK — id={art_id}")


def test_record_outcome() -> None:
    from artifacts import ArtifactStore
    from memory import Memory
    from schemas import ToolCall

    store = ArtifactStore()
    blob = b"fetched content " * 300
    art_id = store.put(blob, content_type="text/plain", source="tool:fetch_url", descriptor="test fetch")

    mem = Memory()
    tc = ToolCall(name="fetch_url", arguments={"url": "https://example.com"})
    outcome = mem.record_outcome(
        tool_call=tc,
        result_text=f"[artifact {art_id}, {len(blob)} bytes]",
        artifact_id=art_id,
        run_id="r1",
        goal_id="g_test",
    )
    assert outcome.kind == "tool_outcome"
    assert outcome.artifact_id == art_id

    hits = mem.read("fetch_url", [])
    assert any(h.artifact_id == art_id for h in hits), "tool_outcome not found via read"
    print(f"[4] record_outcome OK — artifact_id propagated, findable via read")


def test_idempotent_put() -> None:
    from artifacts import ArtifactStore
    store = ArtifactStore()
    blob = b"idempotent test blob"
    id1 = store.put(blob, content_type="text/plain", source="test", descriptor="first")
    id2 = store.put(blob, content_type="text/plain", source="test", descriptor="second")
    assert id1 == id2, f"Idempotent put returned different ids: {id1} vs {id2}"
    meta = store.get_meta(id1)
    assert meta.descriptor == "first", "metadata was overwritten on second put"
    print(f"[5] idempotent put OK — id={id1}, descriptor preserved")


if __name__ == "__main__":
    clean_state()
    try:
        test_remember_fact()
        test_read_fact()
        test_artifact_round_trip()
        test_record_outcome()
        test_idempotent_put()
        print("\n✓ All Phase 1 checks passed")
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}", file=sys.stderr)
        sys.exit(1)
