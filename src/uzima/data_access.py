from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=1)
def patients_by_ref() -> dict[str, dict[str, Any]]:
    text = resources.files("uzima.data").joinpath("synthetic_patients.json").read_text(encoding="utf-8")
    return {p["patient_ref"]: p for p in json.loads(text)["patients"]}
