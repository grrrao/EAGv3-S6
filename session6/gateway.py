from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any

import httpx

_GATEWAY_URL = "http://localhost:8101"
_GW_PATH = Path(__file__).parent.parent / "llm_gatewayV3"

# Load gateway client without polluting sys.path (avoids shadowing local schemas.py)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_gw_client", _GW_PATH / "client.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LLM = _mod.LLM


def ensure_gateway() -> None:
    try:
        r = httpx.get(f"{_GATEWAY_URL}/v1/status", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"LLM Gateway V3 is not running at {_GATEWAY_URL}.\n"
            f"Start it with:  cd llm_gatewayV3 && ./run.sh\n"
            f"Error: {e}"
        ) from e


def get_llm() -> LLM:
    return LLM(base_url=_GATEWAY_URL)


_MAX_SLEEP = 90  # never sleep longer than this between retries


def _backoff_seconds(body: str) -> int | None:
    """Parse all '(Xs left)' values from a gateway 503 body.

    Takes the MINIMUM across all providers (shortest path to any provider
    becoming available) and caps at _MAX_SLEEP so a single provider's long
    RPD blackout doesn't stall the whole agent for hours.
    Returns None if no match is found.
    """
    import re
    matches = re.findall(r"\((\d+)s left\)", body)
    if not matches:
        return None
    min_wait = min(int(m) for m in matches)
    return min(min_wait + 2, _MAX_SLEEP)


def llm_chat(**kwargs: Any) -> dict:
    """Call llm.chat() with sensible defaults and up to 6 retries on transient errors.

    max_tokens defaults to 1024 — keeps est_tokens low enough that Groq/Cerebras/
    GitHub can serve as fallbacks. Callers that need more (decision, perception)
    pass max_tokens explicitly.

    When the gateway returns a 503 with '(Xs left)' backoff markers we sleep the
    minimum across all providers, capped at _MAX_SLEEP, so a single provider's
    hour-long RPD blackout doesn't stall the agent.
    """
    kwargs.setdefault("max_tokens", 1024)
    llm = get_llm()
    max_attempts = 6
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.chat(**kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (429, 500, 502, 503):
                raise
            if attempt == max_attempts:
                raise
            body = e.response.text
            wait = _backoff_seconds(body) or min(5 * attempt, _MAX_SLEEP)
            print(f"[gateway] attempt {attempt} got {e.response.status_code} "
                  f"(backoff {wait}s): {body[:120]}")
            last_err = e
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def gemini_schema(model_class) -> dict:
    """Return a Gemini-compatible JSON schema from a Pydantic model.

    Gemini's proto backend rejects anyOf, type arrays, and several other
    JSON Schema constructs that Pydantic v2 emits for Optional / union fields.
    This cleans the schema before it reaches the gateway.
    """
    schema = model_class.model_json_schema()
    return _clean(schema)


def _clean(node):
    """Recursively clean a JSON Schema for Gemini compatibility.

    Only fixes what the gateway's own cleaner misses: anyOf nullable unions
    and type arrays. Keeps $defs/$ref intact so the gateway can inline them.
    """
    if isinstance(node, list):
        return [_clean(x) for x in node]
    if not isinstance(node, dict):
        return node

    # anyOf with exactly one non-null branch → flatten to that branch
    if "anyOf" in node:
        non_null = [b for b in node["anyOf"] if b.get("type") != "null" and b != {"type": "null"}]
        rest = {k: v for k, v in node.items() if k != "anyOf"}
        base = non_null[0] if non_null else {"type": "string"}
        return _clean({**base, **rest})

    # type: ["X", "null"] → type: "X"
    if isinstance(node.get("type"), list):
        types = [t for t in node["type"] if t != "null"]
        node = {**node, "type": types[0] if types else "string"}

    return {k: _clean(v) for k, v in node.items()}
