"""CLI entry point: python run_query.py A|B|C1|C2|D|ALL"""
from __future__ import annotations

import asyncio
import sys

# Flush stdout immediately so per-iteration prints appear even when piped
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

QUERIES: dict[str, str] = {
    "A": (
        "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his "
        "birth date, death date, and three key contributions to information theory."
    ),
    "B": (
        "Find 3 family-friendly things to do in Tokyo this weekend. "
        "Check Saturday's weather forecast there and tell me which one is most appropriate."
    ),
    "C1": (
        "My mom's birthday is 15 May 2026. Remember that and give me a calendar "
        "reminder for two weeks before and on the day."
    ),
    "C2": "When is mom's birthday?",
    "D": (
        "Search for 'Python asyncio best practices', read the top 3 results, "
        "and give me a short numbered list of the advice they agree on."
    ),
}

ALL_KEYS = ["A", "B", "C1", "C2", "D"]


async def run_all() -> None:
    import asyncio
    from agent6 import run
    results: dict[str, str] = {}
    for i, key in enumerate(ALL_KEYS):
        query = QUERIES[key]
        print(f"\n{'#'*60}")
        print(f"### Query {key} ###")
        print(query)
        print(f"{'#'*60}\n")
        answer = await run(query)
        results[key] = answer
        print(f"\n{'='*60}")
        print(f"FINAL ANSWER [{key}]:\n{answer}")
        print(f"{'='*60}\n")
        # Brief cooldown between queries to let provider rate-limit windows
        # partially reset (Gemini RPM resets each minute; GitHub backoffs expire).
        if i < len(ALL_KEYS) - 1:
            print(f"[cooldown] waiting 15s before next query...")
            await asyncio.sleep(15)

    print(f"\n{'#'*60}")
    print("### ALL QUERIES COMPLETE ###")
    for key in ALL_KEYS:
        print(f"\n[{key}] {results[key][:120]}{'...' if len(results[key]) > 120 else ''}")


def main() -> None:
    key = sys.argv[1].upper() if len(sys.argv) > 1 else ""

    if key == "ALL":
        asyncio.run(run_all())
        return

    if key not in QUERIES:
        print(f"Usage: python run_query.py A|B|C1|C2|D|ALL")
        print(f"Valid keys: {', '.join(QUERIES)}, ALL")
        sys.exit(1)

    query = QUERIES[key]
    print(f"=== Query {key} ===")
    print(query)
    print()

    from agent6 import run
    answer = asyncio.run(run(query))
    print(f"\n{'='*60}")
    print(f"FINAL ANSWER:\n{answer}")


if __name__ == "__main__":
    main()
