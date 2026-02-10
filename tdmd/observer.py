from __future__ import annotations

import inspect
from typing import Any

import numpy as np

# Keep legacy observer contract detection stable:
# observer(step, r, v) or observer(step, r, v, box)
_OBSERVER_MIN_ARGS_WITH_BOX = 4


def observer_accepts_box(observer: Any) -> bool:
    if observer is None:
        return False
    try:
        sig = inspect.signature(observer)
        params = list(sig.parameters.values())
        return (
            any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            or len(params) >= _OBSERVER_MIN_ARGS_WITH_BOX
        )
    except (TypeError, ValueError):
        return False


def emit_observer(
    observer: Any,
    *,
    accepts_box: bool,
    step: int,
    r: np.ndarray,
    v: np.ndarray,
    box: float,
) -> None:
    if observer is None:
        return
    if accepts_box:
        observer(int(step), r, v, float(box))
    else:
        observer(int(step), r, v)

