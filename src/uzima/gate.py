"""The clinician gate.

A proposal (priority, deterioration flag, referral) reaches the queue only
after a NAMED clinician decides on it here. The agent cannot call this: it
is not an MCP tool, it runs in the orchestrator process on the nurse's
terminal, and it writes straight to the store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from . import console, sats, store


@dataclass
class Decision:
    decision: str  # approve | override | reject
    clinician_name: str
    clinician_role: str = "triage nurse"
    final_value: str | None = None
    note: str = ""


Approver = Callable[[dict[str, Any]], Decision]


def render_proposal(p: dict[str, Any]) -> None:
    pl = p.get("payload", p)
    kind = p.get("kind")
    console.step(f"CLINICIAN GATE - {kind} proposal {p.get('id') or p.get('proposal_id')} "
                 f"for encounter {p.get('encounter_id')}")
    if kind in ("priority", "deterioration_flag"):
        print(f"   Proposed: {console.badge(pl.get('proposed_colour'))}  "
              f"TEWS={pl.get('tews')} ({pl.get('tews_colour')})  target: see within {pl.get('target_minutes')} min")
        if kind == "deterioration_flag":
            print(f"   Reason: {pl.get('reason')}  (was {pl.get('current_colour')}, TEWS "
                  f"{pl.get('tews_before')} -> {pl.get('tews_after')})")
            for ch in pl.get("vital_changes", []):
                print(f"     {ch['field']}: {ch['before']} -> {ch['after']}")
        print("   Deciding rule(s): " + ", ".join(pl.get("deciding_rules", [])))
        print("   Sources:")
        for cit in pl.get("citations", []):
            mark = cit.get("colour") or (f"+{cit['points']}" if cit.get("points") is not None else "")
            print(f"     [{cit['rule_id']}] {cit['source_field']}={cit['value']!r}  ({cit['detail']}) {mark}")
    else:
        print(f"   Proposed department: {console.c(str(pl.get('department')), 'bold')}  "
              f"(rule {pl.get('rule_id')}, matched on {pl.get('matched_on')}, colour {pl.get('colour_used')})")
    for w in pl.get("warnings", []) or []:
        console.warn(w)
    for r in pl.get("rejected_claims", []) or []:
        console.warn(f"rejected agent claim: {r}")


class ConsoleApprover:
    """Interactive: the nurse types her name once, then decides on each proposal."""

    def __init__(self, clinician_name: str | None = None, role: str = "triage nurse"):
        self.name = clinician_name or os.environ.get("UZIMA_CLINICIAN")
        self.role = role

    def __call__(self, proposal: dict[str, Any]) -> Decision:
        render_proposal(proposal)
        while not self.name:
            self.name = input("   Your name (clinician confirming): ").strip()
        kind = proposal.get("kind")
        while True:
            ans = input(f"   {self.name}: [a]pprove / [o]verride / [r]eject ? ").strip().lower()[:1]
            if ans == "a":
                return Decision("approve", self.name, self.role)
            if ans == "r":
                note = input("   Reason for rejecting: ").strip() or "rejected at desk"
                return Decision("reject", self.name, self.role, note=note)
            if ans == "o":
                if kind == "referral":
                    val = input("   Department to use: ").strip()
                else:
                    val = input("   Colour to use (RED/ORANGE/YELLOW/GREEN): ").strip().upper()
                    if val not in sats.load_scale()["colours"]:
                        print("   not a SATS colour")
                        continue
                note = ""
                while not note:
                    note = input("   Reason for override (required): ").strip()
                return Decision("override", self.name, self.role, final_value=val, note=note)


class SimulatedApprover:
    """FOR EVALS ONLY. Approves exactly what was proposed, under a name that
    makes it impossible to mistake for a real clinician in the audit log.
    Optionally rejects or overrides by proposal kind, to test those paths."""

    NAME = "SIMULATED-EVAL-CLINICIAN"

    def __init__(self, script: dict[str, Decision] | None = None):
        self.script = script or {}
        self.seen: list[dict[str, Any]] = []

    def __call__(self, proposal: dict[str, Any]) -> Decision:
        self.seen.append(proposal)
        if proposal.get("kind") in self.script:
            return self.script[proposal["kind"]]
        return Decision("approve", self.NAME, "simulated (eval harness)", note="auto-approve for eval")


def apply(proposal_id: str, d: Decision) -> dict[str, Any]:
    return store.record_clinician_decision(proposal_id, d.clinician_name, d.clinician_role, d.decision,
                                           d.final_value, d.note)
