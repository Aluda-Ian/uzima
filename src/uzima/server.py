"""uzima-triage MCP server.

Eight tools. Five act (register, score, recheck, route, commit); three read.
Every call is written to the audit table with inputs, outputs, timestamp,
latency and the run id. Nothing here moves a patient in the queue unless a
named clinician has already approved the proposal through the human gate.

Run standalone (e.g. for Claude Desktop, Goose, or any MCP client):
    uzima serve          # or: python -m uzima.server
"""

from __future__ import annotations

import functools
import os
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except Exception:  # mcp 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as _Server  # type: ignore

from . import sats, store

INSTRUCTIONS = (
    "Uzima triage-desk tools built on the South African Triage Scale (adult). These tools SCORE and "
    "SEQUENCE; they never diagnose or prescribe. Every proposal must be approved by a named clinician "
    "at the gate before commit_to_queue will apply it."
)

mcp = _Server("uzima-triage", instructions=INSTRUCTIONS)


def audited(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with store.Timer() as t:
            try:
                out = fn(*args, **kwargs)
                ok = not (isinstance(out, dict) and out.get("ok") is False)
            except Exception as e:  # surface as data so the agent can recover
                out, ok = {"ok": False, "error": f"{type(e).__name__}: {e}"}, False
        safe_kwargs = {k: (v.model_dump() if isinstance(v, BaseModel) else
                           [x.model_dump() if isinstance(x, BaseModel) else x for x in v] if isinstance(v, list) else v)
                       for k, v in kwargs.items()}
        store.audit(os.environ.get("UZIMA_ACTOR", "mcp-client"), fn.__name__, safe_kwargs, out, ok=ok, server="uzima-triage", latency_ms=t.ms)
        return out
    return wrapper


def _claims(discriminators: Optional[list[dict[str, str]]]) -> list[dict[str, str]]:
    out = []
    for d in discriminators or []:
        out.append(d.model_dump() if isinstance(d, BaseModel) else dict(d))
    return out


def _vitals(**kw) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


# ------------------------------------------------------------------ action tools

@mcp.tool()
@audited
def register_arrival(
    patient_ref: str,
    complaint: str,
    age_years: Optional[float] = None,
    mobility: Optional[str] = None,
    rr: Optional[int] = None,
    hr: Optional[int] = None,
    sbp: Optional[int] = None,
    temp_c: Optional[float] = None,
    avpu: Optional[str] = None,
    trauma: Optional[bool] = None,
    spo2: Optional[int] = None,
    glucose_mmol: Optional[float] = None,
    pregnant: Optional[bool] = None,
    nurse_notes: str = "",
    arrival_time: Optional[str] = None,
) -> dict:
    """Open a triage encounter from the desk intake card. Records the presenting complaint and the vitals
    the nurse measured. mobility: walking|with_help|stretcher. avpu: A|C|V|P|U (C = new confusion).
    arrival_time: ISO-8601 with offset (get it from the time server). Returns encounter_id."""
    if not complaint or not complaint.strip():
        return {"ok": False, "error": "MISSING_COMPLAINT"}
    arrival = arrival_time or datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        datetime.fromisoformat(arrival)
    except ValueError:
        return {"ok": False, "error": f"BAD_ARRIVAL_TIME '{arrival}' - use ISO-8601 e.g. 2026-10-08T09:15:00+03:00"}
    vit = _vitals(mobility=mobility, rr=rr, hr=hr, sbp=sbp, temp_c=temp_c, avpu=avpu, trauma=trauma,
                  spo2=spo2, glucose_mmol=glucose_mmol)
    eid = store.create_encounter({"patient_ref": patient_ref, "age_years": age_years, "pregnant": pregnant,
                                  "complaint": complaint, "nurse_notes": nurse_notes, "vitals": vit,
                                  "arrival_time": arrival})
    _, _, missing = sats.score_tews(vit)
    return {"ok": True, "encounter_id": eid, "arrival_time": arrival, "vitals_recorded": vit,
            "missing_tews_fields": missing}


@mcp.tool()
@audited
def score_priority(encounter_id: str, discriminators: Optional[list[dict[str, str]]] = None) -> dict:
    """Score an encounter on the SATS adult chart (TEWS from vitals + clinical discriminators) and raise a
    PRIORITY PROPOSAL for the clinician gate. discriminators is a list like
    [{"code": "chest_pain", "evidence": "<exact words copied from the complaint>"}]; codes come from lookup_scale,
    and a claim whose evidence is not a verbatim quote is rejected. Does NOT change the queue."""
    enc = store.get_encounter(encounter_id)
    if not enc:
        return {"ok": False, "error": f"UNKNOWN_ENCOUNTER {encounter_id}"}
    res = sats.score(enc, _claims(discriminators)).as_dict()
    if not res["in_scope"]:
        return {"ok": True, "encounter_id": encounter_id, "requires_approval": False, "handover": "MANUAL_TRIAGE",
                **res}
    pid = store.create_proposal(encounter_id, "priority", res)
    return {"ok": True, "encounter_id": encounter_id, "proposal_id": pid, "kind": "priority",
            "requires_approval": True, "status": "pending_clinician", **res}


@mcp.tool()
@audited
def record_recheck(
    encounter_id: str,
    mobility: Optional[str] = None,
    rr: Optional[int] = None,
    hr: Optional[int] = None,
    sbp: Optional[int] = None,
    temp_c: Optional[float] = None,
    avpu: Optional[str] = None,
    trauma: Optional[bool] = None,
    spo2: Optional[int] = None,
    glucose_mmol: Optional[float] = None,
    patient_reported_change: str = "",
    discriminators: Optional[list[dict[str, str]]] = None,
) -> dict:
    """Deterioration watch. Record re-check vitals (fields not given are carried over from intake) and/or a
    patient-reported change while waiting. discriminators: [{"code": ..., "evidence": <verbatim quote from
    patient_reported_change>}]. If the re-scored colour is higher than the committed colour, or TEWS
    rose by 2 or more, raise a DETERIORATION FLAG PROPOSAL for a second look. Never acts on its own."""
    enc = store.get_encounter(encounter_id)
    if not enc:
        return {"ok": False, "error": f"UNKNOWN_ENCOUNTER {encounter_id}"}
    new = dict(enc["vitals"])
    new.update(_vitals(mobility=mobility, rr=rr, hr=hr, sbp=sbp, temp_c=temp_c, avpu=avpu, trauma=trauma,
                       spo2=spo2, glucose_mmol=glucose_mmol))
    store.add_recheck(encounter_id, new, patient_reported_change)
    before = sats.score(enc, [])
    probe = dict(enc, vitals=new, complaint=enc["complaint"] + " || RECHECK: " + patient_reported_change)
    after = sats.score(probe, _claims(discriminators))
    if not after.in_scope:
        return {"ok": True, "flag_raised": False, "requires_approval": False, "warnings": after.warnings}
    current = store.latest_committed_colour(encounter_id) or before.colour
    escalated = sats.colour_rank(after.colour) > sats.colour_rank(current)
    tews_jump = (after.tews or 0) - (before.tews or 0)
    changes = [{"field": k, "before": enc["vitals"].get(k), "after": new.get(k)}
               for k in sorted(set(new) | set(enc["vitals"])) if enc["vitals"].get(k) != new.get(k)]
    summary = {"current_colour": current, "rescored_colour": after.colour, "tews_before": before.tews,
               "tews_after": after.tews, "vital_changes": changes}
    if not (escalated or tews_jump >= 2):
        return {"ok": True, "encounter_id": encounter_id, "flag_raised": False, "requires_approval": False,
                **summary, "note": "No escalation on SATS. No flag raised."}
    payload = after.as_dict()
    payload.update(summary)
    payload["reason"] = ("colour escalated" if escalated else f"TEWS rose by {tews_jump}") + " on re-check"
    if not escalated:
        payload["proposed_colour"] = current  # flag for a second look without changing colour
    pid = store.create_proposal(encounter_id, "deterioration_flag", payload)
    return {"ok": True, "encounter_id": encounter_id, "flag_raised": True, "proposal_id": pid,
            "kind": "deterioration_flag", "requires_approval": True, "status": "pending_clinician", **payload}


@mcp.tool()
@audited
def suggest_route(encounter_id: str) -> dict:
    """Propose which department queue this patient should join, from the facility routing table, using the
    COMMITTED triage colour. Raises a REFERRAL PROPOSAL for the clinician gate. Does not move anyone."""
    enc = store.get_encounter(encounter_id)
    if not enc:
        return {"ok": False, "error": f"UNKNOWN_ENCOUNTER {encounter_id}"}
    colour = store.latest_committed_colour(encounter_id)
    if not colour:
        return {"ok": False, "error": "NO_COMMITTED_PRIORITY",
                "detail": "Score and commit an approved priority before routing."}
    codes = []
    for p in enc["proposals"]:
        if p["kind"] in ("priority", "deterioration_flag") and p["status"] == "committed":
            codes += [c["rule_id"].split("SATS.DISC.")[1] for c in p["payload"].get("citations", [])
                      if c["rule_id"].startswith("SATS.DISC.")]
    r = sats.route(enc, colour, codes)
    payload = {**r, "colour_used": colour, "citations": [{"rule_id": r["rule_id"], "source_field": "routing.yaml",
                                                          "value": r["matched_on"], "detail": r["department"]}]}
    pid = store.create_proposal(encounter_id, "referral", payload)
    return {"ok": True, "encounter_id": encounter_id, "proposal_id": pid, "kind": "referral",
            "requires_approval": True, "status": "pending_clinician", **payload}


@mcp.tool()
@audited
def commit_to_queue(proposal_id: str) -> dict:
    """Apply a clinician-approved proposal (priority, deterioration flag, or referral) to the live queue.
    Refuses with BLOCKED_AWAITING_CLINICIAN if no named clinician has approved it. Uses the clinician's
    final value (which may be an override), not the agent's proposal."""
    return store.commit(proposal_id)


# ------------------------------------------------------------------ read tools

@mcp.tool()
@audited
def get_queue(now: Optional[str] = None) -> dict:
    """The live triage queue, ordered by committed colour then arrival time, with minutes waited versus the
    SATS target time. `now` is an ISO-8601 time (use the time server); defaults to the server clock."""
    return queue_snapshot(now)


def queue_snapshot(now: Optional[str] = None) -> dict:
    t_now = datetime.fromisoformat(now) if now else datetime.now(timezone.utc)
    rows = []
    for r in store.queue_rows():
        waited = (t_now - datetime.fromisoformat(r["arrival_time"])).total_seconds() / 60
        tgt = sats.target_minutes(r["colour"]) if r["colour"] else None
        rows.append({"encounter_id": r["encounter_id"], "patient_ref": r["patient_ref"], "colour": r["colour"],
                     "department": r["department"], "minutes_waited": round(waited),
                     "target_minutes": tgt, "overdue": tgt is not None and waited > tgt,
                     "flagged_for_review": bool(r["flagged_for_review"])})
    rows.sort(key=lambda x: (-sats.colour_rank(x["colour"]) if x["colour"] else 0, -x["minutes_waited"]))
    pend = [{"proposal_id": p["id"], "encounter_id": p["encounter_id"], "kind": p["kind"]}
            for p in store.pending_proposals()]
    return {"ok": True, "now": t_now.isoformat(), "queue": rows, "pending_clinician": pend,
            "needs_recheck": [r["encounter_id"] for r in rows if r["overdue"]]}


@mcp.tool()
@audited
def get_encounter(encounter_id: str) -> dict:
    """Full record for one encounter: intake, vitals, re-checks, every proposal with its status and clinician
    decision, and the current queue entry. Use it to verify your work."""
    enc = store.get_encounter(encounter_id)
    if not enc:
        return {"ok": False, "error": f"UNKNOWN_ENCOUNTER {encounter_id}"}
    for p in enc["proposals"]:
        full = store.get_proposal(p["id"])
        p["approval"] = full["approval"] if full else None
        p["payload"] = {k: p["payload"].get(k) for k in
                        ("proposed_colour", "department", "tews", "deciding_rules", "warnings") if k in p["payload"]}
    return {"ok": True, **enc}


@mcp.tool()
@audited
def lookup_scale(query: str = "") -> dict:
    """Search the SATS adult discriminator list (codes you may pass to score_priority / record_recheck).
    Empty query returns every code. Also returns the TEWS bands and target times."""
    scale = sats.load_scale()
    q = sats.normalise(query)
    hits = [{"code": d["code"], "colour": d["colour"], "label": d["label"]} for d in scale["discriminators"]
            if not q or q in d["code"].replace("_", " ") or q in d["label"].lower()
            or any(q in w.replace("_", " ") for w in d["code"].split("_"))]
    return {"ok": True, "scale": scale["scale_name"], "source": scale["source"],
            "verification_status": scale["verification_status"], "discriminators": hits,
            "tews_bands": scale["tews"]["bands"],
            "targets_minutes": {k: v["target_minutes"] for k, v in scale["colours"].items()}}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
