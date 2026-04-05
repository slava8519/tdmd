# TDMD — Corrected Refactoring Roadmap

> This document replaces the earlier external analysis with a repo-aligned plan.
> It is intended for engineering execution, not for generic cleanup.
> Source of truth remains `AGENTS.md`, `docs/TODO.md`, `docs/CUDA_EXECUTION_PLAN.md`,
> `docs/MODE_CONTRACTS.md`, and the strict VerifyLab gates.

## 1. Status Rebaseline

This roadmap starts from the actual current repository state, not from an older snapshot.

- The CUDA execution cycle `PR-C01..PR-C08` is complete.
- The active GPU stack is `CuPy RawKernel`, not `numba-cuda`.
- The current operator decision is to stay on `CuPy RawKernel` unless fresh profiler evidence
  justifies a Plan B.
- The active maintenance priority is:
  1. `PR-MB01` many-body TD force-scope contract
  2. `PR-MB02` CPU-reference target-local `EAM/eam-alloy` TD path
  3. `PR-MB03` GPU refinement of that corrected many-body path
  4. `PR-ZA01` resource-aware TD auto-zoning advisor
  5. `PR-ML01..PR-ML02` ML-potential groundwork on the same CPU-reference-first contract
- `AGENTS.md` is an active governing contract and must not be demoted to an optional prompt file.

## 2. Non-Negotiables

- Preserve TD semantics: no bypass or merge of `F/D/P/W/S`.
- Preserve the formal core invariant `W<=1`.
- Do not introduce implicit global barriers or synchronous assumptions.
- CPU remains the formal reference semantics.
- GPU remains a refinement path only.
- No false green: diagnostic runs do not count as acceptance.
- For hardware-strict GPU validation, CPU fallback is failure.

## 3. What From the External Analysis Is Useful

The following observations are valid and worth acting on:

- `tdmd/forces_gpu.py` is still structurally large and mixes multiple responsibilities.
- `tdmd/td_full_mpi.py` remains a high-risk monolith and should eventually be decomposed.
- `scripts/run_verifylab_matrix.py` is large enough to justify library extraction.
- `tdmd/td_local.py` still contains duplicated execution-path logic.
- Packaging and CI are not fully consolidated:
  - `pytest` is still in runtime dependencies,
  - CI still installs from `requirements.txt`,
  - `requirements*.txt` duplicate `pyproject.toml`.
- Typing coverage is still selective and can be expanded in a controlled way.

## 4. What Must Be Rejected or Reframed

The following parts of the earlier plan should not be executed as written:

- Do not replace or remove `AGENTS.md`.
  If `CLAUDE.md` is added later, it should be a thin pointer layer, not a new source of truth.
- Do not start with broad documentation reshuffling just because AI prompt files look noisy.
  Navigation cleanup is acceptable; governance replacement is not.
- Do not use weak checkpoints like `pytest + smoke_ci + interop_smoke` for GPU/materials/MPI
  refactors. Scope-specific strict lanes remain mandatory.
- Do not do repo-wide `ruff check --fix .` expansion as an early step.
  That would create large mechanical churn across semantically sensitive modules.
- Do not prioritize structural MPI/GPU refactors ahead of the many-body locality problem.
  Current profiling shows that repeated full-system many-body evaluation is the real ceiling.
- Do not delete remote branches as part of the main refactoring roadmap.
  Branch hygiene is optional repository maintenance, not a project-critical engineering step.

## 5. Engineering Principle for Refactoring

Refactoring is allowed only when it satisfies at least one of these:

1. It directly enables `PR-MB01..PR-MB03`.
2. It reduces verification or maintenance risk in a module that is already on the critical path.
3. It prepares a clean contract boundary for future ML-potential work.

If a refactor only improves aesthetics, naming, or file layout without supporting one of those
goals, it is lower priority than the current many-body track.

## 6. Corrected Execution Order

### Track A — Low-Risk Infrastructure Cleanup

Purpose: remove real packaging and CI drift without touching TD semantics.

#### A1. Consolidate packaging contract
- Move `pytest` from runtime dependencies to the `dev` extra.
- Decide whether `requirements.txt` and `requirements-dev.txt` stay as generated artifacts or are
  removed entirely.
- Keep one source of truth for install dependencies.

#### A2. Fix CI install path
- Switch CI from `pip install -r requirements.txt` to `pip install -e '.[dev]'`.
- Add pip caching if it does not obscure failure diagnosis.
- Keep PR validation enabled; do not optimize by reducing safety coverage first.

#### A3. Add safe artifact hygiene
- Add a non-destructive cleanup target for benchmark/VerifyLab output.
- Do not remove operator evidence by default.
- Cleanup should target generated run directories only, not arbitrary repo content.

Acceptance for Track A:
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`

### Track B — VerifyLab and Benchmark Refactor

Purpose: make verification infrastructure more modular before deeper runtime refactors.

#### B1. Extract preset definitions from `scripts/run_verifylab_matrix.py`
- Move preset data and validation logic into `tdmd/verifylab/presets.py`.
- Keep script CLI behavior stable.

#### B2. Extract reusable matrix runner
- Move expansion and execution logic into `tdmd/verifylab/runner.py`.
- Keep the script as a thin CLI entrypoint.

#### B3. Keep benchmark scripts operator-facing and explicit
- Preserve existing heavy/manual benchmarks:
  - `eam_decomp_perf_smoke`
  - `eam_decomp_perf_gpu_heavy`
  - `eam_decomp_zone_sweep_gpu`
  - `eam_td_breakdown_gpu`
  - `gpu_perf_smoke`
  - `gpu-profile`
- Do not turn observability benchmarks into false acceptance gates.

Acceptance for Track B:
- Base gates from Track A
- If benchmark wiring changes:
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`
  - `.venv/bin/python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v1.json --config examples/td_1d_morse.yaml --strict`
  - `.venv/bin/python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v2.json --config examples/td_1d_morse.yaml --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_smoke --strict`

### Track C — Critical Path: Many-Body TD Locality

Purpose: address the actual current performance and design bottleneck before large structural cleanup.

#### C1. `PR-MB01` many-body TD force-scope contract
- Make the distinction explicit between:
  - full-system many-body evaluation,
  - target-local many-body evaluation,
  - observability-only wrappers and breakdown metrics.
- Freeze the large-run `eam_td_breakdown_gpu` artifact as a baseline for this track.

#### C2. `PR-MB02` CPU-reference target-local `EAM/eam-alloy` TD path
- Remove repeated dependence on full-system `forces_full(ctx.r)[ids0]` inside async TD half-steps.
- Preserve TD scheduling semantics.
- Keep CPU path as the reference proof of correctness.

#### C3. `PR-MB03` GPU refinement on the corrected locality model
- Refine the GPU path only after the CPU-reference target-local model is correct.
- Re-run breakdown benchmarks to confirm that `forces_full` no longer dominates the same way.

Acceptance for Track C:
- Base gates from Track A
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset longrun_envelope_ci --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`
- `.venv/bin/python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v1.json --config examples/td_1d_morse.yaml --strict`
- `.venv/bin/python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v2.json --config examples/td_1d_morse.yaml --strict`
- If GPU path changes:
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke_hw --strict`

### Track D — Structural Refactors After Many-Body Correction

Purpose: decompose large modules only after the main runtime contract is corrected.

#### D1. `tdmd/forces_gpu.py` split by responsibility
- Candidate target shape:
  - `tdmd/gpu/kernels.py`
  - `tdmd/gpu/state.py`
  - `tdmd/gpu/neighbors.py`
  - `tdmd/gpu/pair_forces.py`
  - `tdmd/gpu/eam_forces.py`
  - `tdmd/forces_gpu.py` as facade
- Keep public API stable while moving implementation detail.
- Cache refactor should be lifetime- and invalidation-aware, not just “wrap globals in a class”.

#### D2. `tdmd/td_local.py` cleanup
- Reduce duplicated execution-path scaffolding.
- Keep semantics of `sync_mode`, async path, and dispatch invariants explicit.
- Treat `run_td_local_legacy()` as an adapter problem, not as a pure deletion candidate.

#### D3. `tdmd/td_full_mpi.py` decomposition
- Extract wire/lifecycle/overlap/halo helpers gradually.
- Do not change transport semantics while moving code.
- Keep MPI overlap and cuda-aware evidence intact.

#### D4. Typed islands in critical modules
- Add `Protocol`-based contracts only where they match real runtime variation.
- For potentials, prefer capability-based contracts that can support pair, many-body, and future
  ML inference paths.

Acceptance for Track D:
- Base gates from Track A
- Plus scope-specific lanes:
  - GPU refactor: all GPU strict lanes and `eam_decomp_perf_smoke`
  - MPI refactor: `mpi_overlap_smoke`, `mpi_overlap_cudaaware_smoke`, `cluster_scale_smoke`,
    `cluster_stability_smoke`, `mpi_transport_matrix_smoke`
  - `td_local` async/liveness-sensitive refactor: `longrun_envelope_ci`

### Track E — Auto-Zoning and ML-Ready Contracts

Purpose: build future-facing abstractions only on top of the corrected many-body model.

#### E1. `PR-ZA01` resource-aware TD auto-zoning advisor
- Detect available CPU/GPU/MPI resources.
- Enumerate geometry-valid TD and space layouts.
- Emit a recommendation from benchmark evidence.
- Keep it advisory first; do not silently change runtime scheduling policy.

#### E2. `PR-ML01` ML-potential runtime contract
- Define descriptor/cutoff/neighbor/inference contract.
- CPU reference first.
- No hidden synchronization or implicit barriers.

#### E3. `PR-ML02` ML-potential VerifyLab and interop groundwork
- Strict CPU acceptance for at least one ML-potential family before any GPU refinement.
- Reuse the same many-body locality model, not a separate shortcut path.

Acceptance for Track E:
- Base gates from Track A
- Materials or ML-specific strict fixtures as introduced
- GPU strict lanes only when GPU ML refinement exists

## 7. Recommended Immediate Sequence

The recommended next sequence is:

1. Track A1-A2: packaging and CI consolidation
2. Track B1-B2: VerifyLab extraction
3. Track C1: `PR-MB01` many-body TD force-scope contract
4. Track C2: `PR-MB02` CPU-reference target-local many-body TD path
5. Track C3: `PR-MB03` GPU refinement
6. Track E1: advisory auto-zoning
7. Track E2-E3: ML-potential groundwork
8. Track D: broader structural decomposition only where still justified after C/E progress

This order intentionally puts correctness and future ML-ready contracts ahead of broad cleanup.

## 8. Review Checklist for Any Refactoring PR

- Is the change directly enabling many-body locality, VerifyLab modularity, or ML-readiness?
- Does it preserve `F/D/P/W/S`, `W<=1`, and no-hidden-barrier semantics?
- Are the strict gates chosen for the touched scope, not just the minimum smoke pair?
- Does the PR keep CPU as the reference semantics?
- If GPU is touched, is hardware-strict no-fallback evidence still available?
- If docs are changed, do they align with `AGENTS.md` and `docs/TODO.md`?
- If a cache or abstraction is introduced, does it define lifetime, invalidation, and observability?

## 9. Anti-Goals

- Do not replace governance with AI-tool-specific prompt files.
- Do not do cosmetic monolith splitting before fixing the many-body bottleneck.
- Do not introduce repo-wide mechanical churn under the banner of “cleanup”.
- Do not weaken strict verification to make refactors easier.
- Do not present diagnostic benchmark results as acceptance evidence.
- Do not design future ML support around pair-potential assumptions.

## 10. Deliverable Shape

Every step in this roadmap should land as a small PR-sized slice with:

- explicit scope,
- touched files listed,
- strict verification evidence,
- contract/invariant impact statement,
- risks/assumptions,
- next logical follow-up.

That is the standard required by the current repository governance.
