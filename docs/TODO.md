# Task Board (Unified CPU + GPU, PR-sized)

> Orchestrator maintains this file.
> Strategy focus: TD-correct MD for metals and alloys, with CPU reference first and GPU as refinement.
> Completed phases A–H and maintenance tracks MB/ZA/ML are archived in `docs/archive/COMPLETED_PHASES.md`.
> Next active maintenance theme: large-run single-GPU TD scaling via `1D` slab-wavefront
> execution, with explicit independence/wavefront contracts before any runtime batching.

## Planned Maintenance - Single-GPU TD Wavefront
- [ ] PR-SW01 Single-GPU wavefront contract for `1D` slabs.
  - DoD: define the formal notion of a wave of mutually independent slab zones on one GPU,
    including halo/dependency constraints, no-hidden-barrier rules, and observability fields for
    wave size / deferred zones / fallback-to-sequential reasons.
  - Delivery note: this is a contract/invariant step first, not a performance-only shortcut.
- [ ] PR-SW02 Representative large-run evidence pack for slab-wavefront viability.
  - DoD: keep `al_crack_100k_compare_gpu` and the `z=1..12` TD sweep as first-class operator
    evidence, plus at least one `EAM/eam-alloy` control benchmark showing where single-GPU TD
    currently loses scaling as `z` grows.
  - Evidence must distinguish:
    - TD vs space at equal `z`,
    - TD absolute runtime vs `z`,
    - orchestration overhead vs force-kernel time.
- [ ] PR-SW03 CPU/reference wave-batch scheduling proof harness.
  - DoD: a deterministic shadow scheduler can group formally independent `1D` slabs into waves
    without changing TD-observable results; verification proves equivalence against the current
    sequential slab order on representative CPU cases.
- [ ] PR-SW04 CUDA fused multi-zone wave execution.
  - DoD: one GPU wave may contain several independent slabs; pair/table/EAM target-local work is
    issued as a fused batch instead of zone-by-zone launches, with no new implicit global barrier.
  - Focus first on slab-local neighbor reuse and neighbor-only slab exchange, not on generic 3D
    block batching.
- [ ] PR-SW05 Wavefront profiling, strict acceptance, and zoning-policy integration.
  - DoD: representative single-GPU large-run artifacts show whether wavefront execution actually
    improves TD absolute runtime vs current sequential `1D` slabs; `td_autozoning_advisor_gpu`
    may consume the corrected wavefront cost model but must remain recommendation-only.
  - Decision rule: if representative wavefront execution does not beat the best current
    sequential-TD wall-clock for at least one large-run workload, close the track as a no-go and
    keep `z ~ G` style guidance instead of forcing higher zone counts.

## Mandatory Rules (all PRs)
- [ ] Preserve TD semantics: no bypass/merge of F/D/P/W/S, no implicit global barriers.
- [ ] Preserve invariants or add a new invariant + checks (pytest and/or VerifyLab metric/counter).
- [ ] GPU changes are refinement only: CPU path remains formal reference.
- [ ] Do not mark task complete if verification is non-strict only.
- [ ] Minimum quality gates for completion:
  - [ ] `.venv/bin/python -m pytest -q`
  - [ ] `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
  - [ ] `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
  - [ ] if PR touches ensemble behavior: `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset nvt_smoke --strict`
  - [ ] if PR touches ensemble behavior: `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset npt_smoke --strict`
  - [ ] if PR touches GPU: `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
  - [ ] if PR touches GPU backend policy: keep hardware-strict no-fallback rule active (`gpu_smoke_hw` must reject CPU fallback).
- [ ] Required docs updates when behavior changes:
  - [ ] `docs/SPEC_TD_AUTOMATON.md`
  - [ ] `docs/INVARIANTS.md`
  - [ ] `docs/THEORY_VS_CODE.md`
  - [ ] `docs/VERIFYLAB.md`
  - [ ] if PR touches visualization/output contract: `docs/VISUALIZATION.md`
  - [ ] if PR touches GPU: `docs/GPU_BACKEND.md` and `docs/VERIFYLAB_GPU.md`

## Critical Order (implementation sequence)
1. Stabilize verification gates and task/runtime contract (PR-01, PR-02).
2. Enable multi-component CPU pair model (PR-03, PR-04).
3. Add CPU materials physics reference (PR-05, PR-06).
4. Integrate materials physics into TD paths (PR-07, PR-08).
5. Expand interop + materials verification (PR-09, PR-10).
6. Optimize time-blocking/performance (PR-11).
7. GPU track only after CPU reference is valid: PR-G01 -> ... -> PR-G10.
8. CUDA execution cycle: PR-C01 -> ... -> PR-C08.
