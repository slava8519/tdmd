# PR-0 Data Contract (Golden Path)

This document defines the **GPU-ready data contract** that PR-0 should introduce while still being CPU-only.

Goal:
> Make GPU integration a backend swap, not a refactor of data layout.

## Single door principle
All compute backends (CPU/GPU) must receive a **materialized, contiguous SoA** view:
- `ZoneView` (work atoms)
- `HaloView` (halo atoms)

Implementation location:
- `tdmd/zone_views.py`

## Required dtypes/layout
- positions: `float64`, C-contiguous, shape `(n,)`
- `type`: `int32`, C-contiguous, shape `(n,)`
- `id`: `int32`, C-contiguous, shape `(n,)`

## Minimal structs
- `ZoneView`: n, x,y,z,type,id
- `HaloView`: n, x,y,z,type,id (read-only)
- `ForceParams`: cutoff, box(Lx,Ly,Lz), pbc, precision, potential dict
- `ComputeRequest`: need_energy, need_virial, deterministic
- `ComputeResult`: fx,fy,fz (+ optional energy/virial6)

## PR-0 acceptance (data-level)
- TD semantics unchanged.
- CPUBackend uses existing CPU force path and matches prior outputs.
- A unit test verifies CPUBackend equivalence using ZoneView/HaloView.
