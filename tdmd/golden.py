from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class GoldenSeries:
    case: str
    steps: list[int]
    E: list[float]
    T: list[float]
    P: list[float]


def to_dict(series: list[GoldenSeries]) -> dict[str, Any]:
    return {"series": [s.__dict__ for s in series]}


def from_dict(d: dict[str, Any]) -> list[GoldenSeries]:
    out = []
    for s in d.get("series", []):
        out.append(
            GoldenSeries(
                case=s["case"],
                steps=[int(x) for x in s["steps"]],
                E=[float(x) for x in s["E"]],
                T=[float(x) for x in s["T"]],
                P=[float(x) for x in s["P"]],
            )
        )
    return out


def save(path: str, series: list[GoldenSeries]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(series), f, indent=2)


def load(path: str) -> list[GoldenSeries]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return from_dict(d)
