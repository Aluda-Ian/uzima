"""uzima command line.

    uzima demo                 full agent run on 3 synthetic patients + a re-check (needs Ollama)
    uzima demo --planner scripted   same flow with the no-model baseline planner
    uzima intake --patient SYN-002  one intake through the agent
    uzima recheck --encounter E0003 --patient SYN-003
    uzima queue                the live queue
    uzima log [--run RUN_ID]   the audit log (every tool call, gate decision, fallback)
    uzima eval --planner scripted|agent --runs 3
    uzima check                environment check (Python deps, time server, Ollama + model)
    uzima serve                run the uzima-triage MCP server on stdio for any MCP client
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from . import console, gate, store
from .data_access import patients_by_ref

DEMO_PATIENTS = ["SYN-002", "SYN-003", "SYN-005"]


def _run_id(prefix: str) -> str:
    rid = f"{prefix}-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
    os.environ["UZIMA_RUN_ID"] = rid
    return rid


def _approver(args) -> gate.Approver:
    if getattr(args, "simulate_clinician", False):
        console.warn("SIMULATED clinician: every proposal auto-approved as 'SIMULATED-EVAL-CLINICIAN'. "
                     "Smoke-test only - not a real approval.")
        return gate.SimulatedApprover()
    return gate.ConsoleApprover(clinician_name=getattr(args, "clinician", None))


def _load_card(args) -> dict:
    if getattr(args, "file", None):
        return json.loads(Path(args.file).read_text(encoding="utf-8"))
    try:
        return patients_by_ref()[args.patient]
    except KeyError:
        sys.exit(f"Unknown synthetic patient {args.patient}. Try one of: {', '.join(patients_by_ref())}")


def _ollama_ok(model: str) -> tuple[bool, str]:
    import httpx
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=3).json()
    except Exception as e:
        return False, f"Ollama not reachable at {host} ({type(e).__name__}). Install from https://ollama.com and start it."
    names = [m.get("name", "") for m in tags.get("models", [])]
    if not any(n == model or n.split(":")[0] == model or n == f"{model}:latest" for n in names):
        return False, f"Model '{model}' not pulled. Run: ollama pull {model}   (have: {names or 'none'})"
    return True, f"Ollama OK at {host}, model {model} available"


async def _intake(planner: str, card: dict, approver, run_id: str, model: str) -> dict:
    console.step(f"INTAKE {card.get('patient_ref')} - planner={planner}"
                 + (f" model={model}" if planner == "agent" else ""))
    print(f"   complaint: {card['complaint']!r}")
    if planner == "agent":
        from . import agent
        return await agent.run("intake", card, approver, run_id, model=model)
    from . import scripted
    from .mcpio import Toolbox
    async with Toolbox(run_id) as tb:
        return await scripted.run_intake(tb, card, approver)


async def _recheck(planner: str, eid: str, recheck: dict, approver, run_id: str, model: str) -> dict:
    console.step(f"RE-CHECK {eid} - planner={planner}")
    print(f"   nurse re-check: {json.dumps(recheck, ensure_ascii=False)}")
    if planner == "agent":
        from . import agent
        return await agent.run("recheck", recheck, approver, run_id, model=model, encounter_id=eid)
    from . import scripted
    from .mcpio import Toolbox
    async with Toolbox(run_id) as tb:
        return await scripted.run_recheck(tb, eid, recheck, approver)


def _print_summary(s: dict) -> None:
    print(console.c("   RESULT:", "bold"), f"status={s.get('status')} encounter={s.get('encounter_id')} "
          f"colour={console.badge(s.get('colour'))} department={s.get('department')}")
    if s.get("agent_summary"):
        print("   agent says:", s["agent_summary"].replace("\n", "\n              "))
    if s.get("usage"):
        u = s["usage"]
        print(console.c(f"   cost: {u['llm_calls']} local LLM calls, {u['input_tokens']} in / {u['output_tokens']} out "
                        f"tokens, {s.get('steps')} steps, {s.get('nudges')} verifier nudges - $0 API spend", "dim"))


def cmd_queue(_args=None) -> None:
    from .server import queue_snapshot
    q = queue_snapshot()
    console.step(f"QUEUE @ {q['now']}  (db: {store.db_path()})")
    if not q["queue"]:
        print("   (empty)")
    for r in q["queue"]:
        flag = console.c(" RE-CHECK FLAG", "yellow") if r["flagged_for_review"] else ""
        over = console.c(" OVERDUE", "red") if r["overdue"] else ""
        print(f"   {console.badge(r['colour'])} {r['encounter_id']} {r['patient_ref']:<8} "
              f"waited {r['minutes_waited']:>3} / target {r['target_minutes']} min  -> {r['department']}{flag}{over}")
    if q["pending_clinician"]:
        print("   pending clinician:", q["pending_clinician"])


def cmd_log(args) -> None:
    rows = store.audit_rows(args.run, args.limit)
    for r in rows:
        mark = console.c("ok ", "green") if r["ok"] else console.c("ERR", "red")
        print(f"{r['ts']} {mark} {r['run_id'] or '-':<22} {r['actor']:<34} {r['server'] or '':<20} {r['tool']:<26} "
              f"in={console.short(r['input_json'], 90)} out={console.short(r['output_json'], 110)} "
              f"{r['latency_ms'] or ''}ms")
    print(f"\n{len(rows)} rows from {store.db_path()}")


def cmd_check(args) -> int:
    ok = True
    print(f"python {sys.version.split()[0]} ({sys.executable})")
    for mod in ("mcp", "mcp_server_time", "langgraph", "langchain_mcp_adapters", "langchain_ollama", "yaml"):
        try:
            __import__(mod)
            print(f"  [ok] import {mod}")
        except Exception as e:
            ok = False
            print(f"  [!!] import {mod}: {e}")
    good, msg = _ollama_ok(args.model)
    print(("  [ok] " if good else "  [!!] ") + msg)
    print(f"db: {store.db_path()}")
    return 0 if ok and good else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uzima", description="Uzima triage-desk agent (SATS, clinician-gated)")
    ap.add_argument("--db", help="SQLite file (default ./uzima.db or $UZIMA_DB)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, planner=True):
        if planner:
            p.add_argument("--planner", choices=["agent", "scripted"], default="agent")
        p.add_argument("--model", default=os.environ.get("UZIMA_MODEL", "qwen2.5:7b"),
                       help="Ollama model tag (open-weights)")
        p.add_argument("--clinician", help="name of the clinician at the gate (asked if omitted)")
        p.add_argument("--simulate-clinician", action="store_true",
                       help="auto-approve as SIMULATED-EVAL-CLINICIAN (smoke tests only)")

    p = sub.add_parser("demo", help="3 synthetic patients + a deterioration re-check")
    common(p)
    p.add_argument("--keep", action="store_true", help="do not reset the demo database first")
    p = sub.add_parser("intake", help="one intake")
    common(p)
    p.add_argument("--patient", default="SYN-002")
    p.add_argument("--file", help="JSON intake card instead of a synthetic patient")
    p = sub.add_parser("recheck", help="re-check a waiting patient")
    common(p)
    p.add_argument("--encounter", required=True)
    p.add_argument("--patient", help="use this synthetic patient's recheck block")
    p.add_argument("--file", help="JSON with re-check vitals / patient_reported_change")
    sub.add_parser("queue")
    p = sub.add_parser("log")
    p.add_argument("--run")
    p.add_argument("--limit", type=int, default=200)
    p = sub.add_parser("eval")
    p.add_argument("--planner", choices=["agent", "scripted"], default="scripted")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--tasks", help="comma list e.g. T01,T04")
    p.add_argument("--model", default=os.environ.get("UZIMA_MODEL", "qwen2.5:7b"))
    p.add_argument("--verbose", action="store_true", help="print every tool call")
    p.add_argument("--out", default="evals/results")
    p = sub.add_parser("check")
    p.add_argument("--model", default=os.environ.get("UZIMA_MODEL", "qwen2.5:7b"))
    sub.add_parser("serve", help="run the uzima-triage MCP server on stdio")

    args = ap.parse_args(argv)
    if args.db:
        os.environ["UZIMA_DB"] = str(Path(args.db).resolve())

    if args.cmd == "serve":
        from .server import main as serve
        serve()
        return 0
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "queue":
        cmd_queue()
        return 0
    if args.cmd == "log":
        cmd_log(args)
        return 0
    if args.cmd == "eval":
        if args.planner == "agent":
            good, msg = _ollama_ok(args.model)
            if not good:
                sys.exit(msg)
        from . import evals
        return evals.main(args.planner, args.runs, args.model, args.tasks.split(",") if args.tasks else None,
                          args.verbose, args.out)

    if args.planner == "agent":
        good, msg = _ollama_ok(args.model)
        if not good:
            sys.exit(f"{msg}\nOr run the no-model baseline: uzima {args.cmd} --planner scripted")
    approver = _approver(args)

    if args.cmd == "intake":
        rid = _run_id("intake")
        _print_summary(asyncio.run(_intake(args.planner, _load_card(args), approver, rid, args.model)))
        print(console.c(f"   audit: uzima log --run {rid}", "dim"))
        return 0
    if args.cmd == "recheck":
        rc = json.loads(Path(args.file).read_text(encoding="utf-8")) if args.file else \
            patients_by_ref()[args.patient or "SYN-003"]["recheck"]
        rid = _run_id("recheck")
        _print_summary(asyncio.run(_recheck(args.planner, args.encounter, rc, approver, rid, args.model)))
        print(console.c(f"   audit: uzima log --run {rid}", "dim"))
        return 0
    if args.cmd == "demo":
        if not args.db and not os.environ.get("UZIMA_DB"):
            os.environ["UZIMA_DB"] = str(Path("uzima-demo.db").resolve())
        if not args.keep and store.db_path().exists():
            store.db_path().unlink()
        print(console.c("UZIMA - triage desk demo. Synthetic patients only. The agent scores and sequences; "
                        "a named clinician confirms every change.", "bold"))
        eids = {}
        for ref in DEMO_PATIENTS:
            rid = _run_id(f"demo-{ref}")
            s = asyncio.run(_intake(args.planner, patients_by_ref()[ref], approver, rid, args.model))
            _print_summary(s)
            eids[ref] = s.get("encounter_id")
        cmd_queue()
        if eids.get("SYN-003"):
            rid = _run_id("demo-recheck")
            s = asyncio.run(_recheck(args.planner, eids["SYN-003"], patients_by_ref()["SYN-003"]["recheck"],
                                     approver, rid, args.model))
            _print_summary(s)
        cmd_queue()
        print(console.c("\n   Full audit trail: uzima log --db uzima-demo.db", "dim"))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
