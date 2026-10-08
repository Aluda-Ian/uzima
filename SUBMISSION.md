# Uzima: submission description (about 300 words)

**Track:** Health & Wellbeing, The Triage Desk. **Sub-theme:** Symptom-based priority scoring (with deterioration watch and referral routing on the same gate).

**Facility and workflow.** Uzima is built for the triage desk of a public hospital outpatient department, such as a Kenyan county referral hospital. One or two nurses take vitals and case cards for everyone who walks in, so a patient with chest pain can wait in the same line as a prescription refill. Uzima sits beside the nurse's existing step. She enters the complaint and the vitals she has already measured, and within seconds she sees a proposed SATS colour.

**What it does.** A LangGraph agent running an open-weights model (Qwen 2.5 7B through Ollama, entirely on the local machine) calls our MCP server, `uzima-triage`. The agent registers the arrival and stamps the time using the official `mcp-server-time` server. It reads the complaint in English or Kiswahili and picks SATS discriminators, quoting the patient's exact words for each one. It then scores the Triage Early Warning Score from the vitals. Every point and every colour cites its source: a vital sign and the band it fell in, or a verbatim quote. The server rejects claims it cannot find in the text.

**The nurse decides.** Each priority, referral, or deterioration flag is a proposal. The graph pauses at a clinician gate, and the named nurse approves it, overrides it with a reason, or rejects it. The queue changes only after that, and only to her value. While a patient waits, a re-check that scores worse raises a flag for a second look; it never moves the patient on its own. Every tool call and every decision is logged with inputs, outputs, timestamps, and the clinician's name.

**Honesty.** The test set includes cases built to fail: negation, Kiswahili, a misleading keyword in the nurse notes, prompt injection, missing vitals, and a child. The scripted baseline fails 3 of 15 tasks, and the agent is measured on the same suite. The SATS chart was transcribed for this prototype and has not been clinically verified.
