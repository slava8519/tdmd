from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .atoms import normalize_atom_types, normalize_mass
from .celllist import forces_on_targets_celllist
from .constants import GEOM_EPSILON
from .wavefront_1d import (
    WAVEFRONT_1D_CONTRACT_VERSION,
    describe_wavefront_1d_zones,
)
from .zones import Zone, ZoneLayout1DCells, assign_atoms_to_zones, zones_overlapping_range_pbc

WAVEFRONT_REFERENCE_CONTRACT_VERSION = "pr_sw03_v1"

WAVEFRONT_REFERENCE_KIND_PAIR_SYNC_1D = "sync_1d_pair_sequential"
WAVEFRONT_REFERENCE_KIND_MANY_BODY_TARGET_LOCAL = "sequential_1d_many_body_target_local"
WAVEFRONT_REFERENCE_KIND_SHADOW_WAVE_BATCH = "shadow_wave_batch_candidate_local"


def _zone_order(*, zones_total: int, traversal: str) -> list[int]:
    if str(traversal).strip().lower() == "backward":
        return list(range(int(zones_total) - 1, -1, -1))
    return list(range(int(zones_total)))


def _build_zones(
    *,
    r: np.ndarray,
    box: float,
    cutoff: float,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
) -> list[Zone]:
    zones = ZoneLayout1DCells(
        box=float(box),
        cell_size=float(cell_size),
        zones_total=int(zones_total),
        zone_cells_w=int(zone_cells_w),
        zone_cells_s=int(zone_cells_s),
        min_zone_width=float(cutoff),
        strict_min_width=True,
    ).build()
    assign_atoms_to_zones(np.asarray(r, dtype=np.float64), zones, float(box))
    return zones


def _candidate_ids_for_zone(
    *, zone: Zone, zones: list[Zone], box: float, cutoff: float
) -> tuple[np.ndarray, list[int]]:
    z0p = float(zone.z0) - float(cutoff)
    z1p = float(zone.z1) + float(cutoff)
    support_zone_ids = zones_overlapping_range_pbc(z0p, z1p, float(box), zones)
    candidate_ids = [
        np.asarray(zones[int(zid)].atom_ids, dtype=np.int32)
        for zid in support_zone_ids
        if np.asarray(zones[int(zid)].atom_ids).size
    ]
    if candidate_ids:
        return np.concatenate(candidate_ids), [int(zid) for zid in support_zone_ids]
    return np.empty((0,), dtype=np.int32), [int(zid) for zid in support_zone_ids]


def _pair_target_forces(
    *,
    r: np.ndarray,
    box: float,
    potential: object,
    cutoff: float,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    atom_types: np.ndarray,
) -> np.ndarray:
    if target_ids.size == 0 or candidate_ids.size == 0:
        return np.zeros((int(target_ids.size), 3), dtype=np.float64)
    return np.asarray(
        forces_on_targets_celllist(
            np.asarray(r, dtype=np.float64),
            float(box),
            potential,
            float(cutoff),
            np.asarray(target_ids, dtype=np.int32),
            np.asarray(candidate_ids, dtype=np.int32),
            rc=max(float(cutoff), GEOM_EPSILON),
            atom_types=np.asarray(atom_types, dtype=np.int32),
        ),
        dtype=np.float64,
    )


def _many_body_target_forces(
    *,
    r: np.ndarray,
    box: float,
    potential: object,
    cutoff: float,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    atom_types: np.ndarray,
) -> np.ndarray:
    if target_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if hasattr(potential, "forces_on_targets"):
        try:
            out = potential.forces_on_targets(
                r=np.asarray(r, dtype=np.float64),
                box=float(box),
                cutoff=float(cutoff),
                rc=max(float(cutoff), GEOM_EPSILON),
                atom_types=np.asarray(atom_types, dtype=np.int32),
                target_ids=np.asarray(target_ids, dtype=np.int32),
                candidate_ids=np.asarray(candidate_ids, dtype=np.int32),
            )
        except TypeError:
            out = potential.forces_on_targets(
                r=np.asarray(r, dtype=np.float64),
                box=float(box),
                cutoff=float(cutoff),
                atom_types=np.asarray(atom_types, dtype=np.int32),
                target_ids=np.asarray(target_ids, dtype=np.int32),
                candidate_ids=np.asarray(candidate_ids, dtype=np.int32),
            )
        return np.asarray(out, dtype=np.float64)
    full_force, _pe, _virial = potential.forces_energy_virial(
        np.asarray(r, dtype=np.float64),
        float(box),
        float(cutoff),
        np.asarray(atom_types, dtype=np.int32),
    )
    return np.asarray(full_force, dtype=np.float64)[np.asarray(target_ids, dtype=np.int32)]


def _zone_target_forces(
    *,
    r: np.ndarray,
    box: float,
    potential: object,
    cutoff: float,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    atom_types: np.ndarray,
) -> tuple[np.ndarray, str]:
    if hasattr(potential, "forces_energy_virial"):
        return (
            _many_body_target_forces(
                r=r,
                box=box,
                potential=potential,
                cutoff=cutoff,
                target_ids=target_ids,
                candidate_ids=candidate_ids,
                atom_types=atom_types,
            ),
            "many_body_target_local",
        )
    return (
        _pair_target_forces(
            r=r,
            box=box,
            potential=potential,
            cutoff=cutoff,
            target_ids=target_ids,
            candidate_ids=candidate_ids,
            atom_types=atom_types,
        ),
        "pair_candidate_local",
    )


def _reference_force_field(
    *,
    r: np.ndarray,
    zones: list[Zone],
    order: list[int],
    box: float,
    potential: object,
    cutoff: float,
    atom_types: np.ndarray,
) -> tuple[np.ndarray, str]:
    out = np.zeros_like(np.asarray(r, dtype=np.float64))
    reference_force_kind = WAVEFRONT_REFERENCE_KIND_PAIR_SYNC_1D
    if hasattr(potential, "forces_energy_virial"):
        reference_force_kind = WAVEFRONT_REFERENCE_KIND_MANY_BODY_TARGET_LOCAL
    for zid in order:
        zone = zones[int(zid)]
        target_ids = np.asarray(zone.atom_ids, dtype=np.int32)
        if target_ids.size == 0:
            continue
        candidate_ids, _support_zone_ids = _candidate_ids_for_zone(
            zone=zone,
            zones=zones,
            box=float(box),
            cutoff=float(cutoff),
        )
        zone_force, _force_kind = _zone_target_forces(
            r=r,
            box=box,
            potential=potential,
            cutoff=cutoff,
            target_ids=target_ids,
            candidate_ids=candidate_ids,
            atom_types=atom_types,
        )
        out[target_ids] = zone_force
    return out, reference_force_kind


def _shadow_wave_force_field(
    *,
    r: np.ndarray,
    zones: list[Zone],
    order: list[int],
    box: float,
    potential: object,
    cutoff: float,
    atom_types: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    out = np.zeros_like(np.asarray(r, dtype=np.float64))
    wavefront = describe_wavefront_1d_zones(
        zones=zones,
        box=float(box),
        cutoff=float(cutoff),
        traversal_order=list(order),
    )
    admissibility_violations: list[str] = []
    per_zone = dict(wavefront.get("per_zone", {}) or {})
    for wave in list(wavefront.get("candidate_waves", [])):
        zone_ids = [int(zid) for zid in list(dict(wave).get("zone_ids", []))]
        for left, right in combinations(zone_ids, 2):
            deps_left = set(
                int(v) for v in list(dict(per_zone.get(str(left), {})).get("support_zone_ids", []))
            )
            deps_right = set(
                int(v) for v in list(dict(per_zone.get(str(right), {})).get("support_zone_ids", []))
            )
            if int(right) in deps_left or int(left) in deps_right:
                admissibility_violations.append(
                    f"wave{int(dict(wave).get('wave_index', 0))}:{int(left)}<->{int(right)}"
                )
        for zid in zone_ids:
            zone = zones[int(zid)]
            target_ids = np.asarray(zone.atom_ids, dtype=np.int32)
            if target_ids.size == 0:
                continue
            candidate_ids, _support_zone_ids = _candidate_ids_for_zone(
                zone=zone,
                zones=zones,
                box=float(box),
                cutoff=float(cutoff),
            )
            zone_force, _force_kind = _zone_target_forces(
                r=r,
                box=box,
                potential=potential,
                cutoff=cutoff,
                target_ids=target_ids,
                candidate_ids=candidate_ids,
                atom_types=atom_types,
            )
            out[target_ids] = zone_force
    return out, {
        "wavefront": wavefront,
        "admissibility_ok": not admissibility_violations,
        "admissibility_violations": admissibility_violations,
        "force_kind": WAVEFRONT_REFERENCE_KIND_SHADOW_WAVE_BATCH,
    }


def _acceleration(
    force: np.ndarray,
    *,
    mass_scalar: float | None,
    mass_arr: np.ndarray | None,
) -> np.ndarray:
    if mass_arr is None:
        return np.asarray(force, dtype=np.float64) / float(mass_scalar)
    return np.asarray(force, dtype=np.float64) / np.asarray(mass_arr, dtype=np.float64)[:, None]


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return float(np.max(diff)) if diff.size else 0.0


def prove_wavefront_1d_reference_equivalence(
    *,
    r0: np.ndarray,
    v0: np.ndarray,
    mass: float | np.ndarray,
    box: float,
    potential: object,
    dt: float,
    cutoff: float,
    n_steps: int,
    atom_types: np.ndarray | None,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    traversal: str = "forward",
    atol: float = 1e-10,
    require_multi_zone_wave: bool = True,
) -> dict[str, Any]:
    r_ref = np.asarray(r0, dtype=np.float64).copy()
    v_ref = np.asarray(v0, dtype=np.float64).copy()
    r_shadow = np.asarray(r0, dtype=np.float64).copy()
    v_shadow = np.asarray(v0, dtype=np.float64).copy()

    mass_scalar, mass_arr, _inv_mass = normalize_mass(mass, n_atoms=r_ref.shape[0])
    atom_types_arr = normalize_atom_types(atom_types, n_atoms=r_ref.shape[0])
    order = _zone_order(zones_total=int(zones_total), traversal=str(traversal))

    rows: list[dict[str, Any]] = []
    admissibility_violations: list[str] = []
    max_force_diff = 0.0
    max_position_diff = 0.0
    max_velocity_diff = 0.0
    max_wave_size = 0
    multi_zone_wave_seen = False
    reference_force_kind = ""
    shadow_force_kind = ""

    for step in range(1, int(n_steps) + 1):
        zones_pre = _build_zones(
            r=r_ref,
            box=float(box),
            cutoff=float(cutoff),
            cell_size=float(cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(zone_cells_w),
            zone_cells_s=int(zone_cells_s),
        )
        f0_ref, reference_force_kind = _reference_force_field(
            r=r_ref,
            zones=zones_pre,
            order=order,
            box=float(box),
            potential=potential,
            cutoff=float(cutoff),
            atom_types=atom_types_arr,
        )
        f0_shadow, shadow_pre = _shadow_wave_force_field(
            r=r_shadow,
            zones=zones_pre,
            order=order,
            box=float(box),
            potential=potential,
            cutoff=float(cutoff),
            atom_types=atom_types_arr,
        )
        shadow_force_kind = str(shadow_pre.get("force_kind", ""))
        admissibility_violations.extend(
            str(item) for item in list(shadow_pre.get("admissibility_violations", []))
        )
        pre_wavefront = dict(shadow_pre.get("wavefront", {}) or {})
        pre_wave_size = int(pre_wavefront.get("wave_size_max", 0) or 0)
        max_wave_size = max(max_wave_size, pre_wave_size)
        multi_zone_wave_seen = bool(multi_zone_wave_seen or pre_wave_size > 1)
        force_diff_pre = _max_abs(f0_ref, f0_shadow)
        max_force_diff = max(max_force_diff, force_diff_pre)

        v_half_ref = v_ref + 0.5 * float(dt) * _acceleration(
            f0_ref, mass_scalar=mass_scalar, mass_arr=mass_arr
        )
        v_half_shadow = v_shadow + 0.5 * float(dt) * _acceleration(
            f0_shadow, mass_scalar=mass_scalar, mass_arr=mass_arr
        )
        r_ref = np.mod(r_ref + float(dt) * v_half_ref, float(box))
        r_shadow = np.mod(r_shadow + float(dt) * v_half_shadow, float(box))
        pos_diff = _max_abs(r_ref, r_shadow)
        max_position_diff = max(max_position_diff, pos_diff)

        zones_post = _build_zones(
            r=r_ref,
            box=float(box),
            cutoff=float(cutoff),
            cell_size=float(cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(zone_cells_w),
            zone_cells_s=int(zone_cells_s),
        )
        f1_ref, _reference_force_kind = _reference_force_field(
            r=r_ref,
            zones=zones_post,
            order=order,
            box=float(box),
            potential=potential,
            cutoff=float(cutoff),
            atom_types=atom_types_arr,
        )
        f1_shadow, shadow_post = _shadow_wave_force_field(
            r=r_shadow,
            zones=zones_post,
            order=order,
            box=float(box),
            potential=potential,
            cutoff=float(cutoff),
            atom_types=atom_types_arr,
        )
        admissibility_violations.extend(
            str(item) for item in list(shadow_post.get("admissibility_violations", []))
        )
        post_wavefront = dict(shadow_post.get("wavefront", {}) or {})
        post_wave_size = int(post_wavefront.get("wave_size_max", 0) or 0)
        max_wave_size = max(max_wave_size, post_wave_size)
        multi_zone_wave_seen = bool(multi_zone_wave_seen or post_wave_size > 1)
        force_diff_post = _max_abs(f1_ref, f1_shadow)
        max_force_diff = max(max_force_diff, force_diff_post)

        v_ref = v_half_ref + 0.5 * float(dt) * _acceleration(
            f1_ref, mass_scalar=mass_scalar, mass_arr=mass_arr
        )
        v_shadow = v_half_shadow + 0.5 * float(dt) * _acceleration(
            f1_shadow, mass_scalar=mass_scalar, mass_arr=mass_arr
        )
        vel_diff = _max_abs(v_ref, v_shadow)
        max_velocity_diff = max(max_velocity_diff, vel_diff)

        rows.append(
            {
                "step": int(step),
                "force_max_abs_pre": float(force_diff_pre),
                "force_max_abs_post": float(force_diff_post),
                "position_max_abs": float(pos_diff),
                "velocity_max_abs": float(vel_diff),
                "pre_first_wave_size": int(pre_wavefront.get("first_wave_size", 0) or 0),
                "pre_wave_size_max": int(pre_wave_size),
                "pre_wave_count": int(pre_wavefront.get("wave_count", 0) or 0),
                "post_first_wave_size": int(post_wavefront.get("first_wave_size", 0) or 0),
                "post_wave_size_max": int(post_wave_size),
                "post_wave_count": int(post_wavefront.get("wave_count", 0) or 0),
                "pre_admissibility_ok": bool(shadow_pre.get("admissibility_ok", False)),
                "post_admissibility_ok": bool(shadow_post.get("admissibility_ok", False)),
            }
        )

    violations: list[str] = []
    if float(max_force_diff) > float(atol):
        violations.append(f"force_max_abs>{float(atol):.3e}")
    if float(max_position_diff) > float(atol):
        violations.append(f"position_max_abs>{float(atol):.3e}")
    if float(max_velocity_diff) > float(atol):
        violations.append(f"velocity_max_abs>{float(atol):.3e}")
    if admissibility_violations:
        violations.append("wave_admissibility")
    if bool(require_multi_zone_wave) and not bool(multi_zone_wave_seen):
        violations.append("no_multi_zone_wave")

    return {
        "contract_version": WAVEFRONT_REFERENCE_CONTRACT_VERSION,
        "wavefront_contract_version": WAVEFRONT_1D_CONTRACT_VERSION,
        "n_steps": int(n_steps),
        "box": float(box),
        "cutoff": float(cutoff),
        "cell_size": float(cell_size),
        "zones_total": int(zones_total),
        "zone_cells_w": int(zone_cells_w),
        "zone_cells_s": int(zone_cells_s),
        "traversal": str(traversal),
        "reference_force_kind": str(reference_force_kind),
        "shadow_force_kind": str(shadow_force_kind),
        "many_body": bool(hasattr(potential, "forces_energy_virial")),
        "max_force_max_abs": float(max_force_diff),
        "max_position_max_abs": float(max_position_diff),
        "max_velocity_max_abs": float(max_velocity_diff),
        "max_wave_size": int(max_wave_size),
        "multi_zone_wave_seen": bool(multi_zone_wave_seen),
        "atol": float(atol),
        "require_multi_zone_wave": bool(require_multi_zone_wave),
        "admissibility_violations": sorted(set(str(item) for item in admissibility_violations)),
        "rows": rows,
        "ok_all": not violations,
        "violations": violations,
    }
