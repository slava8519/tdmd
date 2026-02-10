# GPU Backend API Contract

This document defines the **minimal, strict contract** for introducing a GPU compute backend into TDMD‑TD **without changing TD semantics**.

Principle:
> GPU is a **zone compute backend**. The TD automaton (deps, liveness, scheduling) remains on CPU.

This contract exists to ensure:
- scientific reproducibility (CPU reference remains the ground truth),
- isolation of GPU from TD scheduling semantics,
- deterministic and testable data movement between CPU↔GPU.

---

## 0. Scope and non-goals

### Scope
- The backend computes **forces** (and optionally energy/virial) for a *single work zone* given:
  - work-zone atoms,
  - required halo atoms,
  - simulation parameters (cutoff, box, potential config),
  - timestep does **not** affect force evaluation itself, but may be used for diagnostics.

### Non-goals
- GPU backend must **not**:
  - change TD automaton state machine,
  - decide which zone is in `W`,
  - perform MPI communication,
  - mutate global particle ownership or deps tables.

---

## 1. Terminology

- **Work zone (W)**: the single zone allowed to run compute in the formal core.
- **Halo atoms**: atoms required to compute interactions for the work zone but owned by other zones/ranks.
- **SoA**: structure-of-arrays layout (separate arrays for x,y,z etc).

---

## 2. Backend interface (conceptual)

The code should introduce an interface (Protocol/ABC) similar to:

```python
class ZoneComputeBackend(Protocol):
    def compute_zone(
        self,
        work: ZoneView,
        halos: HaloView,
        params: ForceParams,
        *,
        request: ComputeRequest,
    ) -> ComputeResult:
        ...
```

### 2.1 Inputs

#### `ZoneView` (work zone)
Required fields:
- `n`: number of atoms
- positions:
  - `x: float32/float64[n]`
  - `y: float32/float64[n]`
  - `z: float32/float64[n]`
- `type: int32[n]` (optional if single-species)
- `id: int32[n]` (stable IDs for diagnostics)
- optionally velocities are NOT required for pure force evaluation.

#### `HaloView` (halo atoms)
Same shape as ZoneView but for halo atoms:
- `n_halo`
- `x,y,z,type,id`

Important:
- halo atoms are **read-only** for compute backend.

#### `ForceParams`
- `cutoff: float`
- `box: float` or `{Lx,Ly,Lz}` + periodic flags
- potential parameters (LJ, tabulated, etc)
- numerical mode: `precision = "f32"|"f64"`

#### `ComputeRequest`
- flags for optional outputs:
  - `need_energy: bool`
  - `need_virial: bool`
- deterministic mode:
  - `deterministic: bool` (forces stable across runs, may cost perf)

---

## 3. Outputs

### `ComputeResult`
Required:
- `fx, fy, fz`: arrays for **work-zone atoms only**, shape `[n]`

Optional:
- `energy`: float (zone contribution)
- `virial`: 3x3 or 6‑component symmetric representation

Diagnostics (optional but recommended):
- `backend_diag: dict[str, Any]`:
  - bytes moved, kernel time, occupancy proxy, etc

Important:
- Backend must **not** output forces for halo atoms (halo forces are handled by the owner side per TD rules).
- Energy/virial accounting must be documented for double-count rules (see §7).

---

## 4. Ownership and mutation rules (hard constraints)

### 4.1 CPU owns truth
- The canonical particle state lives on CPU (host).
- GPU backend is a *pure function* on provided buffers.

### 4.2 What GPU may mutate
- GPU may write:
  - forces for work-zone atoms,
  - optional per-atom temporary scratch (internal),
  - optional reduction accumulators (energy/virial).

GPU must never mutate:
- positions/velocities (unless an explicit future “integrator backend” is introduced),
- halo buffers (read-only),
- TD automaton state.

---

## 5. Memory & layout requirements (first implementation)

### 5.1 First GPU milestone: simple SoA, contiguous
For PR‑2 (single-zone correctness), choose the simplest robust layout:
- SoA arrays contiguous for work and halo,
- no neighbor list on GPU initially (use brute-force within small test sizes or precomputed pairs).

Later performance work may introduce:
- cell lists / neighbor lists per zone,
- GPU‑resident caches,
- pinned host memory.

### 5.2 Alignment and dtypes
- `float64` is preferred for scientific parity at first.
- If using `float32`, verify v2 tolerances must be widened explicitly and documented.

---

## 6. Determinism & reductions

GPU reductions are a common source of non-reproducibility.

Requirements:
- In `deterministic=True`, the backend should use:
  - fixed reduction order (e.g., block‑serial reduction),
  - stable atom ordering by `id`,
  - avoid atomics for global sums when feasible.

Acceptance criterion for PR‑2:
- verify v2: dr/dv/E/T/P within tolerances vs CPU baseline, across repeated runs.

---

## 7. Energy / virial conventions

You must explicitly state the accounting policy:

### 7.1 Pairwise potentials
Common options:
- **Half-energy per pair**: when computing interactions i<j, accumulate 0.5*U per atom.
- **Work-zone-only accounting**: accumulate only contributions where i is in work zone and j is in (work+halo). This is acceptable but must match CPU reference.

Rule:
- GPU backend must match the CPU backend’s convention exactly.

### 7.2 Many-body potentials (future)
EAM/MEAM requires different staging (density pass). Do not mix into PR‑2.

---

## 8. Integration points in TDMD‑TD

The backend must be invoked only at the compute stage of the **single work zone**:
- TD automaton selects `work_zid` (CPU),
- host prepares `ZoneView` + `HaloView`,
- backend computes forces (+ optional reductions),
- host applies forces/integrator step as today.

This preserves:
- `W<=1` formal core,
- liveness proofs (A4b),
- WFG diagnostics meaning.

---

## 9. Verification requirements (must pass before enabling GPU by default)

Minimum for PR‑2:
1. Unit test: CPUBackend equivalence (no GPU).
2. GPU vs CPU regression (small case):
   - same initial state,
   - fixed seed,
   - compare:
     - dr/dv norms,
     - energy/temperature/pressure (if computed),
     - invariants (docs/INVARIANTS.md).

Minimum for PR‑3:
- verifylab matrix includes GPU mode and prints:
  - speedup,
  - wfgC_rate / p100,
  - Pareto frontier for GPU vs CPU.

---

## 10. Implementation choices: recommended first stack

For a first correctness-oriented GPU implementation:
- Prefer minimal Python integration:
  - `numba.cuda` **or** `cupy.RawKernel`
- Avoid full C++/CUDA extension until correctness and contracts are stable.

Later, for performance:
- C++/CUDA extension + custom neighbor lists.

---

## 11. Failure modes checklist (what to instrument)

Add diagnostics counters (host-side) for:
- bytes H2D/D2H per step,
- kernel launch count,
- kernel time (optional),
- mismatch counters in verify mode,
- max abs force diff CPU vs GPU for debug runs.

---

## 12. “Do not break TD” checklist (pre-merge gate)

Before merging any GPU PR, ensure:
- TD automaton semantics unchanged (state transitions, deps, A4b).
- `formal_core` mode still enforces `W<=1`.
- WFG diagnostics still functional.
- verify v2 green on CPU-only.
- GPU path gated and tested.



---

## PR-0 data contract (recommended)
For PR-0, adopt the golden GPU-ready SoA contract in `tdmd/zone_views.py`.
See `docs/PR0_DATA_CONTRACT.md`.
