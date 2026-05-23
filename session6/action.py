from __future__ import annotations

from mcp import ClientSession

from artifacts import ArtifactStore
from constants import ARTIFACT_THRESHOLD_BYTES
from schemas import ToolCall


async def execute(
    session: ClientSession,
    tool_call: ToolCall,
    artifacts: ArtifactStore,
) -> tuple[str, str | None]:
    # art: guard — artifact handles are not valid tool arguments
    for k, v in tool_call.arguments.items():
        if isinstance(v, str) and v.startswith("art:"):
            return (
                f"ERROR: argument '{k}' is an artifact handle, not a path/url. "
                "Artifact bytes are attached to your prompt when needed."
            ), None

    result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)

    text = "".join(
        block.text for block in result.content if hasattr(block, "text")
    )

    blob = text.encode("utf-8")
    if len(blob) >= ARTIFACT_THRESHOLD_BYTES:
        art_id = artifacts.put(
            blob,
            content_type="text/plain",
            source=f"tool:{tool_call.name}",
            descriptor=f"{tool_call.name} -> {text[:80]}",
        )
        preview = text[:200].replace("\n", " ")
        return f"[artifact {art_id}, {len(blob)} bytes] preview: {preview}", art_id

    return text, None
