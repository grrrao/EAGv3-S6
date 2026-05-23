from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_SERVER_PATH = str(Path(__file__).parent.parent / "mcp_server.py")
_PYTHON = sys.executable


@asynccontextmanager
async def mcp_session():
    params = StdioServerParameters(command=_PYTHON, args=[_SERVER_PATH])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
