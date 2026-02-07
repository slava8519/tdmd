from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdmd.config import load_config


def _load_materials_pack_module():
    path = ROOT / "scripts" / "materials_parity_pack.py"
    spec = importlib.util.spec_from_file_location("materials_parity_pack_mod", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate materials parity fixture expected values")
    ap.add_argument(
        "--template",
        default="examples/interop/materials_parity_suite_v2_template.json",
        help="Template JSON with cases/thresholds",
    )
    ap.add_argument("--config", default="examples/td_1d_morse.yaml", help="TD config for sync verify path")
    ap.add_argument(
        "--out",
        default="examples/interop/materials_parity_suite_v2.json",
        help="Output fixture JSON path",
    )
    args = ap.parse_args()

    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = ROOT / template_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    with template_path.open("r", encoding="utf-8") as f:
        fixture = json.load(f)

    cfg = load_config(str(args.config))
    mod = _load_materials_pack_module()

    cases_out = []
    for case in list(fixture.get("cases", [])):
        actual = mod._compute_case_actual(case, cfg)
        c2 = dict(case)
        c2["expected"] = actual
        cases_out.append(c2)

    out_obj = dict(fixture)
    out_obj["cases"] = cases_out
    prov = dict(out_obj.get("provenance", {}) or {})
    prov["generated_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    prov["generator"] = "scripts/generate_materials_parity_fixture.py"
    prov["config"] = str(args.config)
    # Keep fixture provenance portable across machines/repositories.
    prov["cwd"] = "<repo_root>"
    out_obj["provenance"] = prov

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[materials-fixture-gen] template={template_path}")
    print(f"[materials-fixture-gen] out={out_path}")
    print(f"[materials-fixture-gen] cases={len(cases_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
