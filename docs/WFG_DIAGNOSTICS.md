# WFG Diagnostics (Local, Sampled)

After A4b provides liveness in runtime, WFG(t) is primarily a *diagnostic* tool.

## Goals
- No global synchronization.
- Rank-local approximation and sampling.
- Used to measure contention patterns and transient cycles.

## API
- `TDAutomaton1W.wfg_local_sample() -> dict[zid, list[donor_zid]]`
- `TDAutomaton1W.wfg_local_cycle() -> cycle | None`
- `TDAutomaton1W.record_wfg_sample()` updates diag:
  - wfg_samples, wfg_cycles, wfg_max_outdeg, wfg_last_cycle

## Metrics
- **wfgS**: number of local WFG samples collected.
- **wfgC**: number of local samples in which a cycle was detected (rank-local).
- **wfgO**: max out-degree in sampled local WFG (proxy for "how blocked" zones are).
- **cycle_rate ≈ wfgC/wfgS**: empirical proxy of how often cyclic wait patterns appear during sampling.

Notes:
- These are *local* diagnostics; nonzero values are expected under load and do not imply global deadlock when A4b is enabled.
- Use trends (growth with scale, sensitivity to overlap parameters) as scientific evidence of contention.

## Normalizations
- **wfgC_rate**: `wfgC / wfgS` (fraction of samples with a local cycle).
- **wfgC_per_100_steps**: `100 * wfgC / steps` (cycles per 100 simulation steps; coarse proxy).

Recommended usage:
- Scheduling quality: track `wfgC_rate`.
- End-to-end impact: compare `speedup` vs `wfgC_per_100_steps`.
