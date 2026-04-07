from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from .zones import Zone, ZoneLayout1DCells, zones_overlapping_range_pbc

WAVEFRONT_1D_CONTRACT_VERSION = "pr_sw01_v1"

WAVEFRONT_REASON_SUPPORT_OVERLAP = "support_overlap"
WAVEFRONT_REASON_EMPTY_ZONE = "empty_zone"

WAVEFRONT_FALLBACK_RUNTIME_SEQUENTIAL_ONLY = "runtime_sequential_only"
WAVEFRONT_FALLBACK_LAYOUT_INVALID = "layout_invalid"
WAVEFRONT_FALLBACK_DEFERRED_SUPPORT_OVERLAP = "deferred_support_overlap"
WAVEFRONT_FALLBACK_NO_MULTI_ZONE_WAVE = "no_multi_zone_wave"
WAVEFRONT_FALLBACK_EMPTY_ZONES_PRESENT = "empty_zones_present"


def _active_zone(zone: Zone) -> bool:
    return bool(int(zone.n_cells) > 0 and float(zone.z1) > float(zone.z0))


def _support_zone_ids_for_zone(
    *, zone: Zone, zones: Sequence[Zone], box: float, cutoff: float
) -> list[int]:
    support = zones_overlapping_range_pbc(
        float(zone.z0) - float(cutoff), float(zone.z1) + float(cutoff), float(box), list(zones)
    )
    return sorted(int(zid) for zid in support if int(zid) != int(zone.zid))


def _normalize_traversal_order(
    *, traversal_order: Iterable[int] | None, active_zone_ids: set[int]
) -> list[int]:
    if traversal_order is None:
        return sorted(int(zid) for zid in active_zone_ids)
    out: list[int] = []
    seen: set[int] = set()
    for value in traversal_order:
        zid = int(value)
        if zid in active_zone_ids and zid not in seen:
            out.append(zid)
            seen.add(zid)
    for zid in sorted(active_zone_ids):
        if zid not in seen:
            out.append(zid)
    return out


def _pair_block_reasons(
    *, zid: int, admitted_zid: int, support_zone_ids: dict[int, list[int]]
) -> list[str]:
    reasons: list[str] = []
    deps_a = set(int(v) for v in support_zone_ids.get(int(zid), []))
    deps_b = set(int(v) for v in support_zone_ids.get(int(admitted_zid), []))
    if int(admitted_zid) in deps_a or int(zid) in deps_b:
        reasons.append(WAVEFRONT_REASON_SUPPORT_OVERLAP)
    return reasons


def describe_wavefront_1d_zones(
    *,
    zones: Sequence[Zone],
    box: float,
    cutoff: float,
    traversal_order: Iterable[int] | None = None,
) -> dict[str, Any]:
    zones_list = [zone for zone in zones]
    all_zone_ids = [int(zone.zid) for zone in zones_list]
    active_zone_ids = {int(zone.zid) for zone in zones_list if _active_zone(zone)}
    inactive_zone_ids = sorted(zid for zid in all_zone_ids if zid not in active_zone_ids)
    order = _normalize_traversal_order(
        traversal_order=traversal_order, active_zone_ids=active_zone_ids
    )

    support_zone_ids = {
        int(zone.zid): _support_zone_ids_for_zone(
            zone=zone, zones=zones_list, box=float(box), cutoff=float(cutoff)
        )
        for zone in zones_list
        if int(zone.zid) in active_zone_ids
    }
    per_zone: dict[str, dict[str, Any]] = {}
    for zone in zones_list:
        zid = int(zone.zid)
        width = float(zone.z1) - float(zone.z0)
        support_wraps = bool(
            (float(zone.z0) - float(cutoff)) < 0.0 or (float(zone.z1) + float(cutoff)) > float(box)
        )
        per_zone[str(zid)] = {
            "zid": zid,
            "active": bool(zid in active_zone_ids),
            "width": float(width),
            "support_zone_ids": list(support_zone_ids.get(zid, [])),
            "support_wraps_pbc": bool(support_wraps),
            "wave_index": None,
            "deferred_from_first_wave": False,
            "first_wave_blocked_by": [],
            "first_wave_block_reasons": (
                [WAVEFRONT_REASON_EMPTY_ZONE] if zid in inactive_zone_ids else []
            ),
        }

    candidate_waves: list[list[int]] = []
    pending = list(order)
    first_wave_deferred: dict[int, dict[str, Any]] = {}
    while pending:
        current_wave: list[int] = []
        next_pending: list[int] = []
        for zid in pending:
            blocked_by: list[int] = []
            reasons: list[str] = []
            for admitted_zid in current_wave:
                pair_reasons = _pair_block_reasons(
                    zid=int(zid),
                    admitted_zid=int(admitted_zid),
                    support_zone_ids=support_zone_ids,
                )
                if pair_reasons:
                    blocked_by.append(int(admitted_zid))
                    reasons.extend(pair_reasons)
            if reasons:
                next_pending.append(int(zid))
                if not candidate_waves:
                    first_wave_deferred[int(zid)] = {
                        "blocked_by": blocked_by,
                        "reasons": sorted(set(str(reason) for reason in reasons)),
                    }
                continue
            current_wave.append(int(zid))
        if not current_wave:
            # Safety valve: preserve deterministic progress even if future predicates change.
            current_wave.append(int(pending[0]))
            next_pending = [int(zid) for zid in pending[1:]]
        wave_index = len(candidate_waves)
        candidate_waves.append(list(current_wave))
        for zid in current_wave:
            per_zone[str(int(zid))]["wave_index"] = int(wave_index)
        pending = list(next_pending)

    for zid, blocked in first_wave_deferred.items():
        per_zone[str(int(zid))]["deferred_from_first_wave"] = True
        per_zone[str(int(zid))]["first_wave_blocked_by"] = [int(v) for v in blocked["blocked_by"]]
        per_zone[str(int(zid))]["first_wave_block_reasons"] = list(blocked["reasons"])

    first_wave_zone_ids = list(candidate_waves[0]) if candidate_waves else []
    deferred_reason_counts: Counter[str] = Counter()
    for blocked in first_wave_deferred.values():
        for reason in blocked["reasons"]:
            deferred_reason_counts[str(reason)] += 1
    multi_zone_wave = any(len(wave) > 1 for wave in candidate_waves)
    fallback_reasons = [WAVEFRONT_FALLBACK_RUNTIME_SEQUENTIAL_ONLY]
    if first_wave_deferred:
        fallback_reasons.append(WAVEFRONT_FALLBACK_DEFERRED_SUPPORT_OVERLAP)
    if not multi_zone_wave:
        fallback_reasons.append(WAVEFRONT_FALLBACK_NO_MULTI_ZONE_WAVE)
    if inactive_zone_ids:
        fallback_reasons.append(WAVEFRONT_FALLBACK_EMPTY_ZONES_PRESENT)

    return {
        "contract_version": WAVEFRONT_1D_CONTRACT_VERSION,
        "layout_valid": True,
        "box": float(box),
        "cutoff": float(cutoff),
        "zones_total": int(len(zones_list)),
        "active_zones_total": int(len(active_zone_ids)),
        "inactive_zones_total": int(len(inactive_zone_ids)),
        "inactive_zone_ids": inactive_zone_ids,
        "traversal_order": [int(zid) for zid in order],
        "candidate_waves": [
            {
                "wave_index": int(idx),
                "zone_ids": [int(zid) for zid in wave],
                "size": int(len(wave)),
            }
            for idx, wave in enumerate(candidate_waves)
        ],
        "first_wave_zone_ids": [int(zid) for zid in first_wave_zone_ids],
        "first_wave_size": int(len(first_wave_zone_ids)),
        "wave_count": int(len(candidate_waves)),
        "wave_size_max": int(max((len(wave) for wave in candidate_waves), default=0)),
        "deferred_zones_total": int(len(first_wave_deferred)),
        "deferred_zone_ids": sorted(int(zid) for zid in first_wave_deferred.keys()),
        "deferred_reason_counts": {
            str(reason): int(count) for reason, count in sorted(deferred_reason_counts.items())
        },
        "fallback_to_sequential_reasons": list(
            dict.fromkeys(str(reason) for reason in fallback_reasons)
        ),
        "per_zone": per_zone,
    }


def describe_wavefront_1d_layout(
    *,
    box: float,
    cutoff: float,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    traversal_order: Iterable[int] | None = None,
) -> dict[str, Any]:
    try:
        zones = ZoneLayout1DCells(
            box=float(box),
            cell_size=float(cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(zone_cells_w),
            zone_cells_s=int(zone_cells_s),
            min_zone_width=float(cutoff),
            strict_min_width=True,
        ).build()
    except Exception as exc:
        return {
            "contract_version": WAVEFRONT_1D_CONTRACT_VERSION,
            "layout_valid": False,
            "box": float(box),
            "cutoff": float(cutoff),
            "zones_total": int(zones_total),
            "active_zones_total": 0,
            "inactive_zones_total": int(zones_total),
            "inactive_zone_ids": list(range(max(0, int(zones_total)))),
            "traversal_order": [],
            "candidate_waves": [],
            "first_wave_zone_ids": [],
            "first_wave_size": 0,
            "wave_count": 0,
            "wave_size_max": 0,
            "deferred_zones_total": int(zones_total),
            "deferred_zone_ids": list(range(max(0, int(zones_total)))),
            "deferred_reason_counts": {
                WAVEFRONT_FALLBACK_LAYOUT_INVALID: int(max(0, int(zones_total)))
            },
            "fallback_to_sequential_reasons": [
                WAVEFRONT_FALLBACK_LAYOUT_INVALID,
                WAVEFRONT_FALLBACK_RUNTIME_SEQUENTIAL_ONLY,
            ],
            "layout_error": str(exc),
            "per_zone": {},
        }
    return describe_wavefront_1d_zones(
        zones=zones,
        box=float(box),
        cutoff=float(cutoff),
        traversal_order=traversal_order,
    )
