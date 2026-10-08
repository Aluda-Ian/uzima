"""Terminal output so the demo shows every tool call on screen."""

from __future__ import annotations

import json
import os
import sys

_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
if os.name == "nt":  # enable ANSI on Windows 10+ terminals
    os.system("")

CODES = {"RED": "41;97", "ORANGE": "48;5;208;30", "YELLOW": "43;30", "GREEN": "42;30",
         "dim": "2", "bold": "1", "cyan": "36", "magenta": "35", "red": "31", "green": "32", "yellow": "33"}


def c(text: str, code: str) -> str:
    if not _COLOUR:
        return text
    return f"\033[{CODES.get(code, code)}m{text}\033[0m"


def badge(colour: str | None) -> str:
    return c(f" {colour or 'NONE'} ", colour or "dim")


def short(obj, n: int = 220) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 3] + "..."


def tool_call(server: str, name: str, args: dict, who: str = "agent") -> None:
    print(c(f"  -> [{who}] {server}.{name}", "cyan"), c(short(args, 260), "dim"))


def tool_result(result, ok: bool = True) -> None:
    print(c("     <- ", "green" if ok else "red") + c(short(result, 300), "dim"))


def step(text: str) -> None:
    print(c(f"\n== {text}", "bold"))


def info(text: str) -> None:
    print(c(f"   {text}", "magenta"))


def warn(text: str) -> None:
    print(c(f"   ! {text}", "yellow"))
