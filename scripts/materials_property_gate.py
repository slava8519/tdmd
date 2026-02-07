from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_materials_pack_module():
    path = ROOT / "scripts" / "materials_parity_pack.py"
    spec = importlib.util.spec_from_file_location("materials_parity_pack_mod", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _filter_fixture(fixture: dict, *, case_prefix: str) -> dict:
    out = dict(fixture)
    cases = list(fixture.get("cases", []))
    if str(case_prefix).strip():
        pfx = str(case_prefix).strip()
        cases = [c for c in cases if str(c.get("name", "")).startswith(pfx)]
    out["cases"] = cases
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Materials property-level strict gate")
    ap.add_argument("--fixture", default="examples/interop/materials_parity_suite_v2.json")
    ap.add_argument("--config", default="examples/td_1d_morse.yaml")
    ap.add_argument("--case-prefix", default="", help="Optional case-name prefix filter")
    ap.add_argument("--out", default="results/materials_property_gate_summary.json")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.is_absolute():
        fixture_path = ROOT / fixture_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    with fixture_path.open("r", encoding="utf-8") as f:
        fixture = json.load(f)
    fixture_f = _filter_fixture(fixture, case_prefix=str(args.case_prefix))
    if not list(fixture_f.get("cases", [])):
        raise SystemExit("materials_property_gate: no cases after filtering")

    mod = _load_materials_pack_module()
    with tempfile.TemporaryDirectory(prefix="tdmd_material_prop_gate_") as td:
        tmp_fixture = Path(td) / "fixture.json"
        with tmp_fixture.open("w", encoding="utf-8") as f:
            json.dump(fixture_f, f, indent=2)
        summary = mod.run_suite(fixture_path=str(tmp_fixture), cfg_path=str(args.config))

    by_property = dict(summary.get("by_property", {}) or {})
    prop_ok = True
    for grp, st in by_property.items():
        if int(st.get("fail", 0)) != 0:
            prop_ok = False
            break
    ok_all = bool(summary.get("ok_all", False) and prop_ok)

    out = {
        "fixture": str(args.fixture),
        "config": str(args.config),
        "case_prefix": str(args.case_prefix),
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ok_all": bool(ok_all),
        "total": int(summary.get("total", 0)),
        "ok": int(summary.get("ok", 0)),
        "fail": int(summary.get("fail", 0)),
        "by_property": by_property,
        "cases": list(summary.get("cases", [])),
        "thresholds": dict(summary.get("thresholds", {}) or {}),
        "property_thresholds": dict(summary.get("property_thresholds", {}) or {}),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(
        f"[materials-property-gate] fixture={args.fixture} case_prefix={args.case_prefix or '<all>'} "
        f"total={out['total']} ok={out['ok']} fail={out['fail']} ok_all={out['ok_all']}",
        flush=True,
    )
    for grp, st in sorted(by_property.items()):
        print(
            f"  - {grp}: ok={int(st.get('ok', 0))} fail={int(st.get('fail', 0))} "
            f"max_abs_diff={float(st.get('max_abs_diff', 0.0)):.3e}",
            flush=True,
        )

    if args.strict and not bool(out["ok_all"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
