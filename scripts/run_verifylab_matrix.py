from __future__ import annotations

from tdmd.verifylab.presets import DEFAULT_THRESHOLDS, PRESETS, preset_names
from tdmd.verifylab.runner import apply_envelope_gate, build_arg_parser, main

__all__ = [
    "DEFAULT_THRESHOLDS",
    "PRESETS",
    "preset_names",
    "apply_envelope_gate",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
