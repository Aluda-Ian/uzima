"""Scripted baseline planner.

A fixed plan over the SAME MCP tools the LLM agent uses, with naive keyword
matching for discriminators. It exists for three reasons:
  1. a no-model path so a stranger can run the full pipeline in one command;
  2. a baseline the LLM agent is evaluated against in EVALS.md;
  3. what the agent hands over to if it cannot complete safely.
It still goes through the clinician gate for every irreversible step.
"""

from __future__ import annotations

from typing import Any

from . import console, gate, sats
from .mcpio import Toolbox

S = "uzima-triage"
INTAKE_FIELDS = ("patient_ref", "complaint", "age_years", "mobility", "rr", "hr", "sbp", "temp_c", "avpu",
                 "trauma", "spo2", "glucose_mmol", "pregnant", "nurse_notes")
VITAL_FIELDS = ("mobility", "rr", "hr", "sbp", "temp_c", "avpu", "trauma", "spo2", "glucose_mmol")


async def gate_and_commit(tb: Toolbox, out: dict[str, Any], approver: gate.Approver, who: str) -> dict[str, Any]:
    pid = out["proposal_id"]
    decision = approver({"id": pid, "kind": out["kind"], "encounter_id": out["encounter_id"], "payload": out})
    res = gate.apply(pid, decision)
    console.info(f"gate: {res['status']} by {res['clinician']} -> {res['final_value']}")
    return await tb.call(S, "commit_to_queue", {"proposal_id": pid}, who)


async def run_intake(tb: Toolbox, intake: dict[str, Any], approver: gate.Approver,
                     who: str = "scripted") -> dict[str, Any]:
    summary: dict[str, Any] = {"planner": who}
    arrival = await tb.now(who)
    args = {k: intake[k] for k in INTAKE_FIELDS if intake.get(k) is not None}
    reg = await tb.call(S, "register_arrival", {**args, "arrival_time": arrival}, who)
    if not reg.get("ok"):
        return {**summary, "status": "FAILED_REGISTER", "error": reg}
    eid = reg["encounter_id"]
    summary["encounter_id"] = eid

    claims = sats.keyword_claims(intake["complaint"] + " " + intake.get("nurse_notes", ""))
    scored = await tb.call(S, "score_priority", {"encounter_id": eid, "discriminators": claims}, who)
    if not scored.get("ok"):
        return {**summary, "status": "FAILED_SCORE", "error": scored}
    if not scored.get("requires_approval"):
        summary.update(status="HANDED_TO_NURSE", reason=scored.get("warnings"))
        return summary
    committed = await gate_and_commit(tb, scored, approver, who)
    if not committed.get("ok"):
        summary.update(status="PRIORITY_NOT_COMMITTED", detail=committed)
        return summary

    routed = await tb.call(S, "suggest_route", {"encounter_id": eid}, who)
    if routed.get("ok"):
        await gate_and_commit(tb, routed, approver, who)

    final = await tb.call(S, "get_encounter", {"encounter_id": eid}, who)
    q = final.get("queue_entry") or {}
    ok = q.get("colour") == committed.get("applied_value")
    summary.update(status="DONE" if ok else "VERIFY_MISMATCH", colour=q.get("colour"),
                   department=q.get("department"), verified=ok)
    return summary


async def run_recheck(tb: Toolbox, encounter_id: str, recheck: dict[str, Any], approver: gate.Approver,
                      who: str = "scripted") -> dict[str, Any]:
    args = {k: recheck[k] for k in VITAL_FIELDS if recheck.get(k) is not None}
    change = recheck.get("patient_reported_change", "")
    claims = sats.keyword_claims(change) if change else []
    # evidence must be found in "complaint || RECHECK: change" - keyword claims quote the change text
    out = await tb.call(S, "record_recheck", {"encounter_id": encounter_id, **args,
                                              "patient_reported_change": change, "discriminators": claims}, who)
    if not out.get("ok"):
        return {"status": "FAILED_RECHECK", "error": out}
    if not out.get("flag_raised"):
        return {"status": "NO_FLAG", "encounter_id": encounter_id, "rescored": out.get("rescored_colour")}
    committed = await gate_and_commit(tb, out, approver, who)
    if not committed.get("ok"):
        return {"status": "FLAG_NOT_COMMITTED", "encounter_id": encounter_id, "detail": committed}
    routed = await tb.call(S, "suggest_route", {"encounter_id": encounter_id}, who)  # colour changed -> re-route
    if routed.get("ok"):
        await gate_and_commit(tb, routed, approver, who)
    q = (await tb.call(S, "get_encounter", {"encounter_id": encounter_id}, who)).get("queue_entry") or {}
    return {"status": "FLAG_COMMITTED", "encounter_id": encounter_id, "colour": q.get("colour"),
            "department": q.get("department")}
