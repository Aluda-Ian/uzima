"""SATS adult scoring engine.

Pure functions, no I/O beyond loading the YAML scale. Every point and every
colour it produces carries a citation: which input field, what value, which
rule id in `data/sats_adult.yaml`. Nothing here diagnoses anything — it maps
recorded vitals and quoted complaint text onto a published triage scale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

VITAL_FIELDS = ("mobility", "rr", "hr", "sbp", "temp_c", "avpu", "trauma")


@lru_cache(maxsize=1)
def load_scale() -> dict[str, Any]:
    text = resources.files("uzima.data").joinpath("sats_adult.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


@lru_cache(maxsize=1)
def load_routing() -> dict[str, Any]:
    text = resources.files("uzima.data").joinpath("routing.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def colour_rank(colour: str) -> int:
    return load_scale()["colours"][colour]["rank"]


def target_minutes(colour: str) -> int:
    return load_scale()["colours"][colour]["target_minutes"]


def max_colour(colours: list[str]) -> str:
    return max(colours, key=colour_rank) if colours else "GREEN"


def discriminator_index() -> dict[str, dict[str, Any]]:
    return {d["code"]: d for d in load_scale()["discriminators"]}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@dataclass
class Citation:
    rule_id: str
    source_field: str
    value: Any
    detail: str
    points: int | None = None
    colour: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {
            "rule_id": self.rule_id,
            "source_field": self.source_field,
            "value": self.value,
            "detail": self.detail,
        }
        if self.points is not None:
            d["points"] = self.points
        if self.colour is not None:
            d["colour"] = self.colour
        return d


@dataclass
class ScoreResult:
    in_scope: bool
    colour: str | None
    tews: int | None
    tews_colour: str | None
    citations: list[Citation] = field(default_factory=list)
    deciding_rules: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    rejected_claims: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        scale = load_scale()
        out: dict[str, Any] = {
            "in_scope": self.in_scope,
            "proposed_colour": self.colour,
            "tews": self.tews,
            "tews_colour": self.tews_colour,
            "deciding_rules": self.deciding_rules,
            "citations": [c.as_dict() for c in self.citations],
            "missing_fields": self.missing_fields,
            "rejected_claims": self.rejected_claims,
            "warnings": self.warnings,
            "scale": {"id": scale["scale_id"], "source": scale["source"],
                      "verification_status": scale["verification_status"]},
        }
        if self.colour:
            out["target_minutes"] = target_minutes(self.colour)
            out["colour_label"] = scale["colours"][self.colour]["label"]
        return out


def _in_range(x: float, lo: float | None, hi: float | None) -> bool:
    return (lo is None or x >= lo) and (hi is None or x < hi)


def score_tews(vitals: dict[str, Any]) -> tuple[int, list[Citation], list[str]]:
    """Return (tews_total, citations, missing_fields)."""
    tews_cfg = load_scale()["tews"]
    total = 0
    cites: list[Citation] = []
    missing: list[str] = []

    for key, cfg in tews_cfg["numeric"].items():
        val = vitals.get(key)
        if val is None or val == "":
            missing.append(key)
            continue
        x = float(val)
        for r in cfg["ranges"]:
            if _in_range(x, r["lo"], r["hi"]):
                total += r["points"]
                cites.append(Citation(
                    rule_id=f"TEWS.{key}.{r['text']}", source_field=f"vitals.{key}", value=val,
                    detail=f"{cfg['label']} {val} falls in band '{r['text']}' -> {r['points']} pt",
                    points=r["points"]))
                break

    for key, cfg in tews_cfg["categorical"].items():
        val = vitals.get(key)
        if key == "trauma":
            val = "yes" if val in (True, "yes", "true", 1) else ("no" if val in (False, "no", "false", 0) else None)
        if isinstance(val, str) and key == "avpu":
            val = val.strip().upper()[:1]
        if isinstance(val, str) and key == "mobility":
            val = val.strip().lower().replace(" ", "_")
        if val is None or val not in cfg["values"]:
            missing.append(key)
            continue
        pts = cfg["values"][val]
        total += pts
        shown = cfg.get("text", {}).get(val, val)
        cites.append(Citation(
            rule_id=f"TEWS.{key}.{val}", source_field=f"vitals.{key}", value=val,
            detail=f"{cfg['label']} '{shown}' -> {pts} pt", points=pts))

    return total, cites, missing


def tews_to_colour(tews: int) -> str:
    for band in load_scale()["tews"]["bands"]:
        if _in_range(tews, band["lo"], band["hi"]):
            return band["colour"]
    return "RED"


def apply_measured_rules(vitals: dict[str, Any], *, use_local_protocol: bool = True) -> list[Citation]:
    out = []
    for rule in load_scale().get("measured_rules", []):
        if rule.get("local_protocol") and not use_local_protocol:
            continue
        val = vitals.get(rule["field"])
        if val is None:
            continue
        hit = (rule["op"] == "lt" and float(val) < rule["value"]) or (rule["op"] == "gt" and float(val) > rule["value"])
        if hit:
            out.append(Citation(rule_id=rule["id"], source_field=f"vitals.{rule['field']}", value=val,
                                detail=rule["label"], colour=rule["colour"]))
    return out


def validate_claims(claims: list[dict[str, str]], source_texts: dict[str, str]) -> tuple[list[Citation], list[dict[str, Any]]]:
    """Accept a discriminator claim only if its code exists in the scale AND its
    evidence is a verbatim quote from the intake text. Unsourced flags are
    rejected — the challenge rule is that an unsourced urgency flag is worse
    than none."""
    index = discriminator_index()
    accepted: list[Citation] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    norm_sources = {k: normalise(v) for k, v in source_texts.items() if v}
    for claim in claims or []:
        code = (claim.get("code") or "").strip()
        evidence = (claim.get("evidence") or "").strip()
        if code not in index:
            rejected.append({"code": code, "evidence": evidence, "reason": "UNKNOWN_CODE (use lookup_scale)"})
            continue
        if len(normalise(evidence)) < 3:
            rejected.append({"code": code, "evidence": evidence, "reason": "NO_EVIDENCE_QUOTE"})
            continue
        where = next((k for k, v in norm_sources.items() if normalise(evidence) in v), None)
        if where is None:
            rejected.append({"code": code, "evidence": evidence,
                             "reason": "EVIDENCE_NOT_FOUND_VERBATIM in complaint/notes"})
            continue
        if code in seen:
            continue
        seen.add(code)
        d = index[code]
        accepted.append(Citation(rule_id=f"SATS.DISC.{code}", source_field=where, value=evidence,
                                 detail=d["label"], colour=d["colour"]))
    return accepted, rejected


def score(encounter: dict[str, Any], claims: list[dict[str, str]] | None = None,
          *, use_local_protocol: bool = True) -> ScoreResult:
    scale = load_scale()
    age = encounter.get("age_years")
    if age is not None and float(age) < scale["min_age_years"]:
        return ScoreResult(
            in_scope=False, colour=None, tews=None, tews_colour=None,
            warnings=[f"OUT_OF_SCOPE: age {age} < {scale['min_age_years']}. Uzima only implements the adult "
                      "SATS chart. Triage this patient manually with the paediatric chart."])

    vitals = encounter.get("vitals", {})
    tews, cites, missing = score_tews(vitals)
    tews_col = tews_to_colour(tews)
    measured = apply_measured_rules(vitals, use_local_protocol=use_local_protocol)
    accepted, rejected = validate_claims(
        claims or [], {"complaint": encounter.get("complaint", ""), "nurse_notes": encounter.get("nurse_notes", "")})

    candidates: list[tuple[str, str]] = [(tews_col, f"TEWS={tews}")]
    candidates += [(c.colour, c.rule_id) for c in measured + accepted]
    final = max_colour([c for c, _ in candidates])
    deciding = [rid for c, rid in candidates if c == final]

    warnings = []
    if missing:
        warnings.append("INCOMPLETE_VITALS: " + ", ".join(missing) +
                        " not recorded - TEWS may be UNDER-estimated. Measure before confirming.")
    if rejected:
        warnings.append(f"{len(rejected)} discriminator claim(s) rejected as unsourced/unknown - not used.")

    return ScoreResult(in_scope=True, colour=final, tews=tews, tews_colour=tews_col,
                       citations=cites + measured + accepted, deciding_rules=deciding,
                       missing_fields=missing, rejected_claims=rejected, warnings=warnings)


def keyword_claims(text: str) -> list[dict[str, str]]:
    """Naive baseline used by the scripted planner: plain substring match on
    English keywords. Known weaknesses (kept on purpose, see EVALS.md): no
    negation handling, English only."""
    t = normalise(text)
    claims = []
    for d in load_scale()["discriminators"]:
        for group in d.get("keywords", []):
            if all(term in t for term in group):
                # quote the first matched term as evidence (verbatim from text)
                claims.append({"code": d["code"], "evidence": group[0]})
                break
    return claims


def route(encounter: dict[str, Any], colour: str | None, discriminator_codes: list[str]) -> dict[str, Any]:
    vitals = encounter.get("vitals", {})
    facts = {
        "colour": colour,
        "pregnant": bool(encounter.get("pregnant")),
        "trauma": vitals.get("trauma") in (True, "yes", "true", 1),
        "discriminators": discriminator_codes,
    }
    cfg = load_routing()
    for rule in cfg["rules"]:
        w = rule.get("when", {})
        if "colour_in" in w and colour not in w["colour_in"]:
            continue
        if "pregnant" in w and facts["pregnant"] != w["pregnant"]:
            continue
        if "trauma" in w and facts["trauma"] != w["trauma"]:
            continue
        if "discriminator_in" in w and not set(w["discriminator_in"]) & set(discriminator_codes):
            continue
        used = {k: facts[{"colour_in": "colour", "discriminator_in": "discriminators"}.get(k, k)] for k in w}
        return {"department": rule["department"], "rule_id": rule["id"], "matched_on": used,
                "facility_table": cfg["facility"]}
    raise RuntimeError("routing table has no default rule")
