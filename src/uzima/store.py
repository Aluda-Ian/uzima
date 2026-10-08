"""SQLite store: encounters, proposals, clinician approvals, the queue, and
the audit log. Shared by the MCP server (agent side) and the human gate
(clinician side). The ONLY code path that writes an approval is
`record_clinician_decision`, which is called by the human gate — it is
deliberately not exposed as an MCP tool, so the agent cannot approve itself.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS encounters (
  id TEXT PRIMARY KEY, patient_ref TEXT NOT NULL, age_years REAL, pregnant INTEGER,
  complaint TEXT NOT NULL, nurse_notes TEXT, vitals_json TEXT NOT NULL,
  arrival_time TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rechecks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, encounter_id TEXT NOT NULL, vitals_json TEXT NOT NULL,
  patient_reported_change TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
  id TEXT PRIMARY KEY, encounter_id TEXT NOT NULL, kind TEXT NOT NULL,
  payload_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, run_id TEXT
);
CREATE TABLE IF NOT EXISTS approvals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, proposal_id TEXT NOT NULL, clinician_name TEXT NOT NULL,
  clinician_role TEXT NOT NULL, decision TEXT NOT NULL, final_value TEXT, note TEXT, ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queue (
  encounter_id TEXT PRIMARY KEY, colour TEXT, department TEXT, arrival_time TEXT NOT NULL,
  priority_proposal TEXT, route_proposal TEXT, flagged_for_review INTEGER DEFAULT 0, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT, actor TEXT NOT NULL,
  server TEXT, tool TEXT NOT NULL, input_json TEXT, output_json TEXT, ok INTEGER, latency_ms REAL
);
"""

GATED_KINDS = ("priority", "deterioration_flag", "referral")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    return Path(os.environ.get("UZIMA_DB", "uzima.db")).resolve()


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(path or db_path()), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _next_id(conn: sqlite3.Connection, table: str, prefix: str) -> str:
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] + 1
    return f"{prefix}{n:04d}"


# ---------------------------------------------------------------- audit

def audit(actor: str, tool: str, inputs: Any, outputs: Any, *, ok: bool = True, server: str | None = None,
          latency_ms: float | None = None, run_id: str | None = None) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO audit (ts, run_id, actor, server, tool, input_json, output_json, ok, latency_ms)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (utcnow(), run_id or os.environ.get("UZIMA_RUN_ID"), actor, server, tool,
             json.dumps(inputs, default=str), json.dumps(outputs, default=str), int(ok), latency_ms))


def audit_rows(run_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    with connect() as c:
        if run_id:
            rows = c.execute("SELECT * FROM audit WHERE run_id=? ORDER BY id LIMIT ?", (run_id, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()[::-1]
    return [dict(r) for r in rows]


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = round((time.perf_counter() - self.t0) * 1000, 1)


# ---------------------------------------------------------------- encounters

def create_encounter(data: dict[str, Any]) -> str:
    with connect() as c:
        eid = _next_id(c, "encounters", "E")
        c.execute(
            "INSERT INTO encounters VALUES (?,?,?,?,?,?,?,?,?)",
            (eid, data["patient_ref"], data.get("age_years"), None if data.get("pregnant") is None else int(bool(data["pregnant"])),
             data["complaint"], data.get("nurse_notes", ""), json.dumps(data["vitals"]),
             data["arrival_time"], utcnow()))
    return eid


def get_encounter(eid: str) -> dict[str, Any] | None:
    with connect() as c:
        r = c.execute("SELECT * FROM encounters WHERE id=?", (eid,)).fetchone()
        if not r:
            return None
        enc = dict(r)
        enc["vitals"] = json.loads(enc.pop("vitals_json"))
        enc["pregnant"] = None if enc["pregnant"] is None else bool(enc["pregnant"])
        enc["proposals"] = [proposal_view(p) for p in
                            c.execute("SELECT * FROM proposals WHERE encounter_id=? ORDER BY id", (eid,)).fetchall()]
        q = c.execute("SELECT * FROM queue WHERE encounter_id=?", (eid,)).fetchone()
        enc["queue_entry"] = dict(q) if q else None
        enc["rechecks"] = [dict(x) for x in c.execute(
            "SELECT id, vitals_json, patient_reported_change, created_at FROM rechecks WHERE encounter_id=?", (eid,))]
    return enc


def add_recheck(eid: str, vitals: dict[str, Any], change: str) -> None:
    with connect() as c:
        c.execute("INSERT INTO rechecks (encounter_id, vitals_json, patient_reported_change, created_at) VALUES (?,?,?,?)",
                  (eid, json.dumps(vitals), change, utcnow()))


# ---------------------------------------------------------------- proposals

def proposal_view(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    d = dict(row)
    d["payload"] = json.loads(d.pop("payload_json"))
    return d


def create_proposal(eid: str, kind: str, payload: dict[str, Any]) -> str:
    assert kind in GATED_KINDS
    with connect() as c:
        pid = _next_id(c, "proposals", "P")
        c.execute("INSERT INTO proposals VALUES (?,?,?,?,?,?,?)",
                  (pid, eid, kind, json.dumps(payload), "pending_clinician", utcnow(), os.environ.get("UZIMA_RUN_ID")))
    return pid


def get_proposal(pid: str) -> dict[str, Any] | None:
    with connect() as c:
        r = c.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
        if not r:
            return None
        p = proposal_view(r)
        a = c.execute("SELECT * FROM approvals WHERE proposal_id=? ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
        p["approval"] = dict(a) if a else None
    return p


def latest_committed_colour(eid: str) -> str | None:
    with connect() as c:
        r = c.execute("SELECT colour FROM queue WHERE encounter_id=?", (eid,)).fetchone()
    return r["colour"] if r else None


# ---------------------------------------------------------------- human gate (NOT an MCP tool)

def record_clinician_decision(pid: str, clinician_name: str, clinician_role: str, decision: str,
                              final_value: str | None = None, note: str = "") -> dict[str, Any]:
    """Called only from the human gate. decision in approve|override|reject."""
    if not clinician_name or not clinician_name.strip():
        raise ValueError("A named clinician is required to decide on a proposal.")
    if decision not in ("approve", "override", "reject"):
        raise ValueError("decision must be approve, override or reject")
    p = get_proposal(pid)
    if not p:
        raise ValueError(f"unknown proposal {pid}")
    if p["status"] != "pending_clinician":
        raise ValueError(f"proposal {pid} is already {p['status']}")
    if decision == "approve":
        final_value = proposed_value(p)
    if decision == "override" and not final_value:
        raise ValueError("override needs the clinician's own value")
    if decision == "override" and not note.strip():
        raise ValueError("override needs a reason (note)")
    status = {"approve": "approved", "override": "overridden", "reject": "rejected"}[decision]
    with connect() as c:
        c.execute("INSERT INTO approvals (proposal_id, clinician_name, clinician_role, decision, final_value, note, ts)"
                  " VALUES (?,?,?,?,?,?,?)", (pid, clinician_name.strip(), clinician_role.strip() or "nurse",
                                              decision, final_value, note, utcnow()))
        c.execute("UPDATE proposals SET status=? WHERE id=?", (status, pid))
    result = {"proposal_id": pid, "status": status, "clinician": clinician_name, "final_value": final_value}
    audit(f"clinician:{clinician_name}", "human_gate.decide",
          {"proposal_id": pid, "decision": decision, "final_value": final_value, "note": note}, result,
          server="uzima-gate")
    return result


def proposed_value(p: dict[str, Any]) -> str | None:
    pl = p["payload"]
    if p["kind"] in ("priority", "deterioration_flag"):
        return pl.get("proposed_colour")
    if p["kind"] == "referral":
        return pl.get("department")
    return None


# ---------------------------------------------------------------- commit (agent-callable, gated)

def commit(pid: str) -> dict[str, Any]:
    p = get_proposal(pid)
    if not p:
        return {"ok": False, "error": f"UNKNOWN_PROPOSAL {pid}"}
    if p["status"] == "committed":
        return {"ok": False, "error": f"ALREADY_COMMITTED {pid}"}
    if p["status"] == "pending_clinician":
        return {"ok": False, "error": "BLOCKED_AWAITING_CLINICIAN",
                "detail": "No named clinician has approved this proposal. Queue unchanged. "
                          "Wait for the clinician gate, then retry."}
    if p["status"] == "rejected":
        return {"ok": False, "error": "REJECTED_BY_CLINICIAN", "clinician": p["approval"]["clinician_name"],
                "note": p["approval"]["note"], "detail": "Queue unchanged."}
    a = p["approval"]
    if not a or not a["clinician_name"]:
        return {"ok": False, "error": "BLOCKED_NO_NAMED_CLINICIAN"}
    eid = p["encounter_id"]
    enc = get_encounter(eid)
    value = a["final_value"]
    with connect() as c:
        row = c.execute("SELECT * FROM queue WHERE encounter_id=?", (eid,)).fetchone()
        if row is None:
            c.execute("INSERT INTO queue (encounter_id, arrival_time, updated_at) VALUES (?,?,?)",
                      (eid, enc["arrival_time"], utcnow()))
        if p["kind"] == "priority":
            c.execute("UPDATE queue SET colour=?, priority_proposal=?, updated_at=? WHERE encounter_id=?",
                      (value, pid, utcnow(), eid))
        elif p["kind"] == "deterioration_flag":
            c.execute("UPDATE queue SET colour=?, priority_proposal=?, flagged_for_review=1, updated_at=?"
                      " WHERE encounter_id=?", (value, pid, utcnow(), eid))
        elif p["kind"] == "referral":
            c.execute("UPDATE queue SET department=?, route_proposal=?, updated_at=? WHERE encounter_id=?",
                      (value, pid, utcnow(), eid))
        c.execute("UPDATE proposals SET status='committed' WHERE id=?", (pid,))
    return {"ok": True, "proposal_id": pid, "encounter_id": eid, "kind": p["kind"], "applied_value": value,
            "approved_by": f"{a['clinician_name']} ({a['clinician_role']})", "decision": a["decision"]}


def queue_rows() -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute(
            "SELECT q.*, e.patient_ref, e.complaint FROM queue q JOIN encounters e ON e.id=q.encounter_id").fetchall()
    return [dict(r) for r in rows]


def pending_proposals() -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM proposals WHERE status='pending_clinician' ORDER BY id").fetchall()
    return [proposal_view(r) for r in rows]
