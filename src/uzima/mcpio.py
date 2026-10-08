"""MCP client plumbing shared by the scripted planner and the eval runner.

Opens two stdio MCP servers:
  * uzima-triage  - ours (uzima.server)
  * time          - borrowed: the official `mcp-server-time` reference server
Every call made through `Toolbox.call` is printed and, for the borrowed
server, written to the audit log here (our own server audits itself).
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import console, store

TIMEZONE = os.environ.get("UZIMA_TZ", "Africa/Nairobi")


def server_params(run_id: str, actor: str = "mcp-client") -> dict[str, StdioServerParameters]:
    env = {**os.environ, "UZIMA_RUN_ID": run_id, "UZIMA_ACTOR": actor, "UZIMA_DB": str(store.db_path()), "PYTHONUTF8": "1"}
    return {
        "uzima-triage": StdioServerParameters(command=sys.executable, args=["-m", "uzima.server"], env=env),
        "time": StdioServerParameters(command=sys.executable,
                                      args=["-m", "mcp_server_time", "--local-timezone", TIMEZONE], env=env),
    }


def result_text(res: Any) -> str:
    parts = []
    for block in getattr(res, "content", None) or []:
        t = getattr(block, "text", None)
        if t is not None:
            parts.append(t)
    return "\n".join(parts)


def parse(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": text}


class Toolbox:
    def __init__(self, run_id: str, verbose: bool = True, actor: str = "scripted-planner"):
        self.run_id = run_id
        self.actor = actor
        self.verbose = verbose
        self.sessions: dict[str, ClientSession] = {}
        self.unavailable: dict[str, str] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "Toolbox":
        for name, params in server_params(self.run_id, self.actor).items():
            try:
                read, write = await self._stack.enter_async_context(stdio_client(params, errlog=self._errlog()))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.sessions[name] = session
            except Exception as e:  # recover: borrowed server missing -> fall back, logged
                self.unavailable[name] = f"{type(e).__name__}: {e}"
                if name == "uzima-triage":
                    raise
        return self

    def _errlog(self):
        f = open(store.db_path().parent / "uzima-mcp-stderr.log", "a", encoding="utf-8")
        self._stack.callback(f.close)
        return f

    async def __aexit__(self, *exc) -> None:
        try:
            await self._stack.aclose()
        except Exception:
            pass

    async def call(self, server: str, tool: str, args: dict[str, Any], who: str = "planner") -> dict[str, Any]:
        if self.verbose:
            console.tool_call(server, tool, args, who)
        if server not in self.sessions:
            out = {"ok": False, "error": f"SERVER_UNAVAILABLE {server}: {self.unavailable.get(server)}"}
        else:
            with store.Timer() as t:
                try:
                    res = await self.sessions[server].call_tool(tool, args)
                    out = parse(result_text(res))
                    if getattr(res, "isError", False) or getattr(res, "is_error", False):
                        out = {"ok": False, "error": result_text(res)}
                except Exception as e:
                    out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if server != "uzima-triage":
                store.audit(who, tool, args, out, ok=out.get("ok", True) is not False, server=server,
                            latency_ms=t.ms, run_id=self.run_id)
        if self.verbose:
            console.tool_result(out, ok=not (isinstance(out, dict) and out.get("ok") is False))
        return out

    async def now(self, who: str = "planner") -> str:
        """Current local time from the borrowed time server, with a logged fallback."""
        out = await self.call("time", "get_current_time", {"timezone": TIMEZONE}, who)
        if isinstance(out, dict) and out.get("datetime"):
            return out["datetime"]
        fallback = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        store.audit("system", "time_fallback", {"reason": out}, {"used": fallback}, ok=False,
                    server="uzima-orchestrator", run_id=self.run_id)
        if self.verbose:
            console.warn(f"time server unavailable - fell back to local clock {fallback} (logged)")
        return fallback
