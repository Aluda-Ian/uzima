"""Fast unit tests for the scoring engine and the clinician gate (no model, no MCP)."""

import os
import tempfile

import pytest

from uzima import gate, sats, store


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("UZIMA_DB", os.path.join(d, "t.db"))


def enc(**vitals):
    base = dict(mobility="walking", rr=16, hr=80, sbp=130, temp_c=36.8, avpu="A", trauma=False)
    base.update(vitals)
    return {"age_years": 40, "complaint": "test complaint", "vitals": base}


def test_normal_vitals_green():
    r = sats.score(enc())
    assert r.colour == "GREEN" and r.tews == 1  # rr 16 -> 1 point


@pytest.mark.parametrize("hr,pts", [(40, 2), (41, 1), (50, 1), (51, 0), (100, 0), (101, 1), (110, 1), (111, 2),
                                    (129, 2), (130, 3)])
def test_hr_band_edges(hr, pts):
    total, cites, _ = sats.score_tews({"hr": hr})
    assert total == pts


def test_tews_band_colours():
    assert [sats.tews_to_colour(x) for x in (0, 2, 3, 4, 5, 6, 7, 12)] == \
        ["GREEN", "GREEN", "YELLOW", "YELLOW", "ORANGE", "ORANGE", "RED", "RED"]


def test_discriminator_needs_verbatim_evidence():
    e = enc()
    e["complaint"] = "Chest pain since morning"
    r = sats.score(e, [{"code": "chest_pain", "evidence": "chest pain"}])
    assert r.colour == "ORANGE" and "SATS.DISC.chest_pain" in r.deciding_rules
    r2 = sats.score(e, [{"code": "chest_pain", "evidence": "crushing chest pain"}])
    assert r2.colour == "GREEN" and r2.rejected_claims


def test_missing_vitals_warned():
    r = sats.score({"age_years": 50, "complaint": "x", "vitals": {"rr": 16}})
    assert "hr" in r.missing_fields and any("INCOMPLETE_VITALS" in w for w in r.warnings)


def test_child_out_of_scope():
    r = sats.score({"age_years": 6, "complaint": "x", "vitals": {}})
    assert not r.in_scope and r.colour is None


def test_hypoglycaemia_red():
    assert sats.score(enc(glucose_mmol=2.1)).colour == "RED"


def test_every_point_is_cited():
    r = sats.score(enc(hr=125, rr=24))
    assert sum(c.points or 0 for c in r.citations) == r.tews


def _proposal():
    eid = store.create_encounter({"patient_ref": "T", "complaint": "x", "vitals": {}, "arrival_time": store.utcnow()})
    return store.create_proposal(eid, "priority", {"proposed_colour": "YELLOW", "citations": [{"rule_id": "x"}]})


def test_commit_blocked_until_named_approval():
    pid = _proposal()
    assert store.commit(pid)["error"] == "BLOCKED_AWAITING_CLINICIAN"
    with pytest.raises(ValueError):
        gate.apply(pid, gate.Decision("approve", "  "))
    gate.apply(pid, gate.Decision("approve", "Nurse Wanjiru"))
    out = store.commit(pid)
    assert out["ok"] and out["applied_value"] == "YELLOW"
    assert store.commit(pid)["error"].startswith("ALREADY_COMMITTED")


def test_override_needs_reason_and_wins():
    pid = _proposal()
    with pytest.raises(ValueError):
        gate.apply(pid, gate.Decision("override", "Nurse Wanjiru", final_value="RED"))
    gate.apply(pid, gate.Decision("override", "Nurse Wanjiru", final_value="RED", note="looks unwell"))
    assert store.commit(pid)["applied_value"] == "RED"


def test_rejected_never_commits():
    pid = _proposal()
    gate.apply(pid, gate.Decision("reject", "Nurse Wanjiru", note="wrong patient"))
    assert store.commit(pid)["error"] == "REJECTED_BY_CLINICIAN"
