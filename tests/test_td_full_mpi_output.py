from __future__ import annotations

from tdmd.td_full_mpi import _output_due_impl


def test_output_due_impl_tracks_telemetry_cadence():
    due = _output_due_impl(
        True,
        out_traj_every=4,
        out_metrics_every=6,
        out_telemetry_every=5,
        step=10,
    )
    assert due == (True, False, False, True)


def test_output_due_impl_tracks_any_output_channel():
    due = _output_due_impl(
        True,
        out_traj_every=4,
        out_metrics_every=6,
        out_telemetry_every=5,
        step=12,
    )
    assert due == (True, True, True, False)
