from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from constants import MEMORY_TOP_K
from gateway import gemini_schema, llm_chat
from schemas import MemoryItem, ToolCall, _ClassifiedItem

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for",
    "in", "on", "at", "is", "are", "what", "when", "who",
    "how", "my", "me", "i", "you", "your",
}


def _tokenize(text: str) -> set[str]:
    tokens = (re.sub(r"[^a-z0-9]", "", w) for w in text.lower().split())
    return {t for t in tokens if t and t not in _STOPWORDS}


def _normalize_keywords(kws: list[str]) -> list[str]:
    """Strip punctuation from LLM-generated keywords and drop stopwords/empties."""
    seen: dict[str, None] = {}
    for kw in kws:
        normalized = re.sub(r"[^a-z0-9]", "", kw.lower())
        if normalized and normalized not in _STOPWORDS:
            seen[normalized] = None
    return list(seen)


class Memory:
    def __init__(self, path: Path = Path("state/memory.json")):
        self._path = path
        self._items: list[MemoryItem] | None = None

    def _load(self) -> list[MemoryItem]:
        if self._items is not None:
            return self._items
        if not self._path.exists():
            self._items = []
            return self._items
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._items = [MemoryItem.model_validate(item) for item in raw]
        return self._items

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [json.loads(item.model_dump_json()) for item in (self._items or [])]
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read(
        self,
        query: str,
        history: list[dict],
        kinds: list[str] | None = None,
        top_k: int = MEMORY_TOP_K,
    ) -> list[MemoryItem]:
        items = self._load()
        q_tokens = _tokenize(query)
        scored: list[tuple[float, datetime, MemoryItem]] = []
        for item in items:
            if kinds and item.kind not in kinds:
                continue
            item_tokens = set(item.keywords) | _tokenize(item.descriptor)
            score = len(q_tokens & item_tokens) / max(1, len(q_tokens))
            if score > 0:
                scored.append((score, item.created_at, item))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item for _, _, item in scored[:top_k]]

    def filter(
        self,
        *,
        kinds: list[str] | None = None,
        goal_id: str | None = None,
        recent: int | None = None,
    ) -> list[MemoryItem]:
        items = self._load()
        result = [
            item for item in items
            if (not kinds or item.kind in kinds)
            and (goal_id is None or item.goal_id == goal_id)
        ]
        result.sort(key=lambda x: x.created_at, reverse=True)
        return result[:recent] if recent else result

    def remember(
        self,
        raw_text: str,
        *,
        source: str,
        run_id: str,
        goal_id: str | None = None,
    ) -> MemoryItem | None:
        system = (
            "You classify a single user statement into one of four memory kinds: "
            "fact (durable observed truth), preference (user-stated or inferred preference), "
            "tool_outcome (record of a tool call — almost never used for raw user text), "
            "scratchpad (working note for this run only). "
            "Extract keywords (lowercased, deduped) and a structured value dict. "
            "Return only JSON conforming to the schema."
        )
        result = llm_chat(
            messages=[{"role": "user", "content": raw_text}],
            system=system,
            auto_route="memory",
            response_format={
                "type": "json_schema",
                "schema": gemini_schema(_ClassifiedItem),
            },
            temperature=0,
            max_tokens=1024,
        )
        parsed = result.get("parsed")
        if parsed is None and result.get("text"):
            try:
                parsed = json.loads(result["text"])
            except (json.JSONDecodeError, ValueError):
                return None
        if not parsed:
            return None
        try:
            classified = _ClassifiedItem.model_validate(parsed)
        except Exception:
            return None
        if classified.confidence < 0.3 and classified.kind == "scratchpad":
            return None
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind=classified.kind,
            keywords=_normalize_keywords(classified.keywords),
            descriptor=classified.descriptor,
            value=classified.value,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
            confidence=classified.confidence,
            created_at=datetime.now(timezone.utc),
        )
        self._load().append(item)
        self._save()
        return item

    def record_outcome(
        self,
        *,
        tool_call: ToolCall,
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str | None,
    ) -> MemoryItem:
        kw_text = tool_call.name + " " + " ".join(str(v) for v in tool_call.arguments.values())
        keywords = list(_tokenize(kw_text))
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind="tool_outcome",
            keywords=keywords,
            descriptor=f"{tool_call.name} → {result_text[:80]}",
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result": result_text[:500],
            },
            artifact_id=artifact_id,
            source=f"tool:{tool_call.name}",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
            created_at=datetime.now(timezone.utc),
        )
        self._load().append(item)
        self._save()
        return item
