from __future__ import annotations

from tdmd.wavefront_1d import (
    WAVEFRONT_FALLBACK_LAYOUT_INVALID,
    WAVEFRONT_FALLBACK_RUNTIME_SEQUENTIAL_ONLY,
    WAVEFRONT_REASON_SUPPORT_OVERLAP,
    WAVEFRONT_1D_CONTRACT_VERSION,
    describe_wavefront_1d_layout,
)


def test_wavefront_layout_groups_independent_slabs_into_candidate_waves():
    desc = describe_wavefront_1d_layout(
        box=40.0,
        cutoff=4.0,
        cell_size=10.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
    )

    assert bool(desc.get("layout_valid")) is True
    assert str(desc.get("contract_version", "")) == WAVEFRONT_1D_CONTRACT_VERSION
    assert list(desc.get("first_wave_zone_ids", [])) == [0, 2]
    assert [wave["zone_ids"] for wave in desc.get("candidate_waves", [])] == [[0, 2], [1, 3]]
    assert int(desc.get("first_wave_size", 0)) == 2
    assert int(desc.get("wave_size_max", 0)) == 2
    assert int(desc.get("deferred_zones_total", 0)) == 2
    assert dict(desc.get("deferred_reason_counts", {})).get(WAVEFRONT_REASON_SUPPORT_OVERLAP) == 2
    assert WAVEFRONT_FALLBACK_RUNTIME_SEQUENTIAL_ONLY in list(
        desc.get("fallback_to_sequential_reasons", [])
    )


def test_wavefront_layout_captures_pbc_support_overlap():
    desc = describe_wavefront_1d_layout(
        box=40.0,
        cutoff=4.0,
        cell_size=10.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
    )

    per_zone = dict(desc.get("per_zone", {}))
    assert list(dict(per_zone.get("0", {})).get("support_zone_ids", [])) == [1, 3]
    assert list(dict(per_zone.get("3", {})).get("support_zone_ids", [])) == [0, 2]
    assert list(dict(per_zone.get("3", {})).get("first_wave_blocked_by", [])) == [0, 2]


def test_wavefront_layout_reports_invalid_geometry():
    desc = describe_wavefront_1d_layout(
        box=6.0,
        cutoff=3.0,
        cell_size=1.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
    )

    assert bool(desc.get("layout_valid")) is False
    assert str(desc.get("contract_version", "")) == WAVEFRONT_1D_CONTRACT_VERSION
    assert WAVEFRONT_FALLBACK_LAYOUT_INVALID in list(desc.get("fallback_to_sequential_reasons", []))
    assert WAVEFRONT_FALLBACK_RUNTIME_SEQUENTIAL_ONLY in list(
        desc.get("fallback_to_sequential_reasons", [])
    )
    assert dict(desc.get("deferred_reason_counts", {})).get(WAVEFRONT_FALLBACK_LAYOUT_INVALID) == 4
