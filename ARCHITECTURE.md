# Architecture

```
 nurse at triage desk (terminal)
        |  intake card / re-check                      ^ CLINICIAN GATE: approve / override(+reason) / reject
        v                                              |  (LangGraph interrupt; writes approvals table directly)
 +--------------------------- LangGraph (uzima/agent.py) -------------------------------+
 |  agent (Ollama, qwen2.5:7b) -> tools -> [proposal?] -> clinician_gate -> agent ...    |
 |        \-> no tool calls -> verify (get_encounter; nudge x2 or hand over) -> output_guard |
 +----------------------|------------------------------------|--------------------------+
          stdio MCP      v                                     v   stdio MCP
 +-------------------------------+                 +-------------------------------+
 | uzima-triage (OURS)           |                 | mcp-server-time (BORROWED)    |
 | register_arrival   (act)      |                 | get_current_time(Africa/Nairobi)
 | score_priority     (act, proposal)              +-------------------------------+
 | record_recheck     (act, proposal)
 | suggest_route      (act, proposal)
 | commit_to_queue    (act, GATED)
 | get_queue / get_encounter / lookup_scale (read)
 +---------------|---------------+
                 v
   SQLite: encounters, rechecks, proposals, approvals, queue, audit
   SATS adult chart + routing table: versioned YAML with a rule id on every line
```

## Agent shape

The agent is a single LangGraph state machine with five nodes. It plans in the model: from the intake card, the model decides which tools to call, picks discriminator codes, and quotes their evidence. Everything around the model is deterministic.

* **tools** runs each MCP call and logs it. A tool error comes back to the model as `{"ok": false, "error": ...}` so it can correct its arguments; a crash would end the run. If a small model prints a tool call as JSON text instead of making the call, this node recovers it into a real call and logs the recovery.
* **clinician_gate** fires after any tool returns `requires_approval`. It pauses the graph with `interrupt()` until a named human decides. It handles one proposal per pass, so no side effect runs twice when the graph resumes.
* **verify** runs when the model stops calling tools. It reads the record back with `get_encounter`. If a step is missing (no score, approved but not committed, not routed), it sends the model a concrete instruction. After two nudges it stops and hands the case to the nurse rather than guessing.
* **output_guard** blocks a final message that reads like a diagnosis, a dose, or a treatment, and logs the block.

The **scripted planner** (`scripted.py`) calls the same tools in a fixed order and uses naive keyword matching for discriminators. It serves as the baseline in the evals and as the no-model path for the demo.

## Built versus borrowed, and why

| Server | Built or borrowed | Why |
|---|---|---|
| `uzima-triage` | **Built** | No existing server does SATS scoring with per-rule citations, verbatim-evidence checks, or a commit that refuses without a named clinician. The tool boundaries (score, recheck, route, commit, queue) fit any triage scale, so swapping SATS for the Manchester Triage System means changing the YAML file, not the tools. |
| `mcp-server-time` | **Borrowed** (official MCP reference server) | Wait time against the SATS target is only as good as the timestamp behind it. IANA timezone handling has edge cases (tzdata on Windows, DST), and a maintained reference server already gets them right. Writing our own would add code for judges to trust and no new capability. If the server is missing, the orchestrator falls back to the local clock and logs the fallback. |

## Where safety lives (not in the prompt)

* **Gate:** `commit_to_queue` checks the `approvals` table on the server side. Approvals are written only by `store.record_clinician_decision`, which is not exposed as an MCP tool. A decision with an empty name is refused, and an override requires a reason.
* **Sourcing:** `score_priority` rejects any discriminator whose code is not in the scale or whose evidence is not a verbatim substring of the complaint or notes. Every TEWS point cites the vital sign and the band it fell in.
* **Scope:** under 12 years returns `MANUAL_TRIAGE`. Missing vitals produce an `INCOMPLETE_VITALS` warning that the nurse sees at the gate.
* **Data:** an Ollama model on localhost, SQLite on local disk, synthetic patients only.

## Cost per run

The model runs locally, so API spend is $0. Each agent run records its local LLM calls, input and output tokens, steps, and verifier nudges in the `run_summary` audit row. `uzima demo` prints them.
