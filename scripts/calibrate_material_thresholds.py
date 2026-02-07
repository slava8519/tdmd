from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime


def _abs_max(values: list[float]) -> float:
    if not values:
        return 1.0
    return max(1.0, max(abs(float(v)) for v in values))


def _collect_scales(fixture: dict) -> dict[str, float]:
    eos_vals: list[float] = []
    thermo_vals: list[float] = []
    transport_vals: list[float] = []
    verify_vals: list[float] = []

    for case in list(fixture.get("cases", [])):
        exp = dict(case.get("expected", {}) or {})
        init = dict(exp.get("initial", {}) or {})
        fin = dict(exp.get("serial_final", {}) or {})
        sync = dict(exp.get("sync_verify", {}) or {})

        eos_vals.extend(
            [
                float(init.get("pe", 0.0)),
                float(init.get("virial", 0.0)),
                float(init.get("P", 0.0)),
                float(fin.get("P", 0.0)),
            ]
        )
        thermo_vals.extend(
            [
                float(init.get("T", 0.0)),
                float(fin.get("E", 0.0)),
                float(fin.get("KE", 0.0)),
                float(fin.get("PE", 0.0)),
                float(fin.get("T", 0.0)),
                float(fin.get("P", 0.0)),
            ]
        )
        pos = list(fin.get("positions", []) or [])
        vel = list(fin.get("velocities", []) or [])
        for row in pos:
            transport_vals.extend(float(x) for x in list(row))
        for row in vel:
            transport_vals.extend(float(x) for x in list(row))
        verify_vals.extend(
            [
                float(sync.get("max_dr", 0.0)),
                float(sync.get("max_dv", 0.0)),
                float(sync.get("max_dE", 0.0)),
                float(sync.get("max_dT", 0.0)),
                float(sync.get("max_dP", 0.0)),
                float(sync.get("final_dr", 0.0)),
                float(sync.get("final_dv", 0.0)),
                float(sync.get("final_dE", 0.0)),
                float(sync.get("final_dT", 0.0)),
                float(sync.get("final_dP", 0.0)),
                float(sync.get("rms_dr", 0.0)),
                float(sync.get("rms_dv", 0.0)),
                float(sync.get("rms_dE", 0.0)),
                float(sync.get("rms_dT", 0.0)),
                float(sync.get("rms_dP", 0.0)),
            ]
        )

    return {
        "eos": _abs_max(eos_vals),
        "thermo": _abs_max(thermo_vals),
        "transport": _abs_max(transport_vals),
        "verify": _abs_max(verify_vals),
    }


def _recommend_thresholds(*, scales: dict[str, float], floor: float, rel: dict[str, float]) -> dict[str, float]:
    eos_tol = max(float(floor), float(rel["eos"]) * float(scales["eos"]))
    thermo_tol = max(float(floor), float(rel["thermo"]) * float(scales["thermo"]))
    transport_tol = max(float(floor), float(rel["transport"]) * float(scales["transport"]))
    verify_tol = max(float(floor), float(rel["verify"]) * float(scales["verify"]))

    return {
        "force_abs": transport_tol,
        "energy_abs": thermo_tol,
        "virial_abs": eos_tol,
        "temp_abs": thermo_tol,
        "press_abs": eos_tol,
        "traj_r_abs": transport_tol,
        "traj_v_abs": transport_tol,
        "verify_abs": verify_tol,
        "eos_energy_abs": eos_tol,
        "eos_virial_abs": eos_tol,
        "eos_press_abs": eos_tol,
        "thermo_energy_abs": thermo_tol,
        "thermo_temp_abs": thermo_tol,
        "thermo_press_abs": thermo_tol,
        "transport_r_abs": transport_tol,
        "transport_v_abs": transport_tol,
        "transport_verify_abs": verify_tol,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate materials threshold policy from parity fixture")
    ap.add_argument("--fixture", default="examples/interop/materials_parity_suite_v2.json")
    ap.add_argument("--out-json", default="golden/material_threshold_policy_v2.json")
    ap.add_argument("--out-md", default="results/material_threshold_policy_v2.md")
    ap.add_argument("--floor", type=float, default=1e-10)
    ap.add_argument("--rel-eos", type=float, default=1e-12)
    ap.add_argument("--rel-thermo", type=float, default=1e-12)
    ap.add_argument("--rel-transport", type=float, default=1e-12)
    ap.add_argument("--rel-verify", type=float, default=1e-12)
    args = ap.parse_args()

    fixture_path = Path(args.fixture)
    with fixture_path.open("r", encoding="utf-8") as f:
        fixture = json.load(f)

    scales = _collect_scales(fixture)
    rel = {
        "eos": float(args.rel_eos),
        "thermo": float(args.rel_thermo),
        "transport": float(args.rel_transport),
        "verify": float(args.rel_verify),
    }
    rec = _recommend_thresholds(scales=scales, floor=float(args.floor), rel=rel)

    out = {
        "policy_version": 2,
        "fixture": str(args.fixture),
        "suite_version": int(fixture.get("suite_version", 0)),
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scales": scales,
        "relative_factors": rel,
        "floor": float(args.floor),
        "thresholds": rec,
        "notes": [
            "CPU path is reference semantics.",
            "Thresholds are derived from fixture value scale with absolute floor.",
            "Transport and verify classes are intentionally conservative in v2 smoke.",
        ],
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Materials Threshold Calibration v2\n\n")
        f.write(f"- fixture: `{args.fixture}`\n")
        f.write(f"- suite_version: `{int(fixture.get('suite_version', 0))}`\n")
        f.write(f"- generated_utc: `{out['generated_utc']}`\n")
        f.write(f"- floor: `{float(args.floor):.3e}`\n")
        f.write("\n## Scales\n\n")
        f.write(json.dumps(scales, indent=2))
        f.write("\n\n## Relative Factors\n\n")
        f.write(json.dumps(rel, indent=2))
        f.write("\n\n## Recommended Thresholds\n\n")
        f.write(json.dumps(rec, indent=2))
        f.write("\n")

    print(f"[materials-threshold-calibrate] fixture={args.fixture}")
    print(f"[materials-threshold-calibrate] out_json={out_json}")
    print(f"[materials-threshold-calibrate] out_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
