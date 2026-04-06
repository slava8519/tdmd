from __future__ import annotations

import argparse
import json
import os
from statistics import median

import numpy as np

from tdmd.backend import resolve_backend

_RAWKERNEL_NAME = "tdmd_perf_axpy_loop"


def _rawkernel_src(loop_iters: int) -> str:
    iters = int(max(1, loop_iters))
    return f"""
extern "C" __global__
void tdmd_perf_axpy_loop(const double* a, const double* b, double* out, const int n) {{
    const int i = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;
    double x = a[i];
    const double y = b[i];
    #pragma unroll 8
    for (int k = 0; k < {iters}; ++k) {{
        x = x * 1.0000001 + y * 0.999999;
    }}
    out[i] = x;
}}
"""


def _measure_ms(cp, repeats: int, fn) -> float:
    vals: list[float] = []
    for _ in range(max(1, int(repeats))):
        ev0 = cp.cuda.Event()
        ev1 = cp.cuda.Event()
        ev0.record()
        fn()
        ev1.record()
        ev1.synchronize()
        vals.append(float(cp.cuda.get_elapsed_time(ev0, ev1)))
    return float(median(vals)) if vals else 0.0


def _make_kernel(cp, loop_iters: int):
    return cp.RawKernel(_rawkernel_src(loop_iters), _RAWKERNEL_NAME)


def _calibrate_kernel(
    cp,
    *,
    d_a,
    d_b,
    d_out,
    n_vals: int,
    threads: int,
    base_loop_iters: int,
    max_loop_iters: int,
    min_kernel_ms: float,
    repeats: int,
):
    loop_iters = int(max(1, base_loop_iters))
    loop_iters_max = int(max(loop_iters, max_loop_iters))
    probe_ms = 0.0
    probe_count = 0

    while True:
        kernel = _make_kernel(cp, loop_iters)
        blocks = (int(n_vals) + int(threads) - 1) // int(threads)

        def _kernel(_kernel=kernel, _blocks=blocks):
            _kernel(
                (_blocks,),
                (threads,),
                (d_a, d_b, d_out, np.int32(n_vals)),
            )

        _kernel()
        cp.cuda.Device().synchronize()
        probe_ms = _measure_ms(cp, max(1, min(int(repeats), 5)), _kernel)
        probe_count += 1
        if probe_ms >= float(min_kernel_ms) or loop_iters >= loop_iters_max:
            return kernel, int(loop_iters), float(probe_ms), int(probe_count)
        loop_iters = min(loop_iters_max, loop_iters * 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="CUDA perf smoke: H2D/D2H and kernel timing")
    ap.add_argument("--n-atoms", type=int, default=65536)
    ap.add_argument("--delta-count", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-json", default="results/gpu_perf_smoke.summary.json")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-delta-over-full", type=float, default=0.65)
    ap.add_argument("--max-transfer-over-kernel", type=float, default=4.0)
    ap.add_argument("--base-kernel-loop-iters", type=int, default=64)
    ap.add_argument("--max-kernel-loop-iters", type=int, default=4096)
    ap.add_argument("--min-kernel-ms", type=float, default=0.1)
    args = ap.parse_args()

    calibration = {
        "base_kernel_loop_iters": int(max(1, args.base_kernel_loop_iters)),
        "kernel_loop_iters": int(max(1, args.base_kernel_loop_iters)),
        "max_kernel_loop_iters": int(
            max(max(1, args.base_kernel_loop_iters), args.max_kernel_loop_iters)
        ),
        "min_kernel_ms": float(max(0.0, args.min_kernel_ms)),
        "probe_kernel_ms": 0.0,
        "probe_count": 0,
        "applied": False,
        "target_met": False,
    }

    backend = resolve_backend("cuda")
    if backend.device != "cuda":
        out = {
            "total": 1,
            "ok": 0,
            "fail": 1,
            "ok_all": False,
            "reason": "cuda_unavailable",
            "backend": {
                "requested_device": "cuda",
                "effective_device": str(backend.device),
                "reason": str(backend.reason),
            },
            "calibration": calibration,
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return 2 if args.strict else 0

    cp = backend.xp
    n_atoms = int(max(1024, args.n_atoms))
    n_vals = int(n_atoms * 3)
    delta_count = int(max(1, min(args.delta_count, n_atoms)))

    rng = np.random.default_rng(int(args.seed))
    r_host = rng.normal(size=(n_atoms, 3)).astype(np.float64)
    t_host = rng.integers(low=1, high=4, size=(n_atoms,), dtype=np.int32)
    upd_ids = np.sort(rng.choice(n_atoms, size=delta_count, replace=False).astype(np.int32))
    r_delta_host = np.asarray(r_host[upd_ids], dtype=np.float64)
    t_delta_host = np.asarray(t_host[upd_ids], dtype=np.int32)

    d_r = cp.asarray(r_host)
    d_t = cp.asarray(t_host)
    d_idx = cp.asarray(upd_ids)

    d_a = cp.asarray(r_host.reshape(-1))
    d_b = cp.asarray((0.5 * r_host + 0.1).reshape(-1))
    d_out = cp.zeros_like(d_a)
    threads = 256
    kernel, loop_iters, probe_ms, probe_count = _calibrate_kernel(
        cp,
        d_a=d_a,
        d_b=d_b,
        d_out=d_out,
        n_vals=n_vals,
        threads=threads,
        base_loop_iters=int(args.base_kernel_loop_iters),
        max_loop_iters=int(args.max_kernel_loop_iters),
        min_kernel_ms=float(args.min_kernel_ms),
        repeats=int(args.repeats),
    )
    calibration.update(
        {
            "kernel_loop_iters": int(loop_iters),
            "probe_kernel_ms": float(probe_ms),
            "probe_count": int(probe_count),
            "applied": bool(int(loop_iters) != int(calibration["base_kernel_loop_iters"])),
            "target_met": bool(float(probe_ms) >= float(calibration["min_kernel_ms"])),
        }
    )
    blocks = (n_vals + threads - 1) // threads
    host_out = np.empty_like(r_host.reshape(-1))

    def _h2d_full():
        _dr = cp.asarray(r_host)
        _dt = cp.asarray(t_host)
        _ = _dr, _dt

    def _h2d_delta():
        d_r[d_idx] = cp.asarray(r_delta_host)
        d_t[d_idx] = cp.asarray(t_delta_host)

    def _kernel():
        kernel(
            (blocks,),
            (threads,),
            (d_a, d_b, d_out, np.int32(n_vals)),
        )

    def _d2h():
        d_out.get(out=host_out)
        _x = host_out
        _ = _x

    # Warm-up to amortize first-use overhead for both transfers and compute.
    _h2d_full()
    _h2d_delta()
    _kernel()
    _d2h()
    cp.cuda.Device().synchronize()

    h2d_full_ms = _measure_ms(cp, args.repeats, _h2d_full)
    h2d_delta_ms = _measure_ms(cp, args.repeats, _h2d_delta)
    kernel_ms = _measure_ms(cp, args.repeats, _kernel)
    d2h_ms = _measure_ms(cp, args.repeats, _d2h)

    delta_over_full = float(h2d_delta_ms / max(h2d_full_ms, 1e-12))
    transfer_over_kernel = float((h2d_delta_ms + d2h_ms) / max(kernel_ms, 1e-12))
    ok = bool(
        delta_over_full <= float(args.max_delta_over_full)
        and transfer_over_kernel <= float(args.max_transfer_over_kernel)
    )

    summary = {
        "n": 1,
        "total": 1,
        "ok": int(ok),
        "fail": int(not ok),
        "ok_all": bool(ok),
        "worst": {
            "h2d_full_ms": float(h2d_full_ms),
            "h2d_delta_ms": float(h2d_delta_ms),
            "kernel_ms": float(kernel_ms),
            "d2h_ms": float(d2h_ms),
            "delta_over_full": float(delta_over_full),
            "transfer_over_kernel": float(transfer_over_kernel),
            "kernel_loop_iters": int(loop_iters),
        },
        "by_case": {
            "gpu_perf_smoke": {
                "total": 1,
                "ok": int(ok),
                "fail": int(not ok),
                "worst": {
                    "h2d_full_ms": float(h2d_full_ms),
                    "h2d_delta_ms": float(h2d_delta_ms),
                    "kernel_ms": float(kernel_ms),
                    "d2h_ms": float(d2h_ms),
                    "delta_over_full": float(delta_over_full),
                    "transfer_over_kernel": float(transfer_over_kernel),
                    "kernel_loop_iters": int(loop_iters),
                },
            }
        },
        "rows": [
            {
                "case": "gpu_perf_smoke",
                "ok": bool(ok),
                "h2d_full_ms": float(h2d_full_ms),
                "h2d_delta_ms": float(h2d_delta_ms),
                "kernel_ms": float(kernel_ms),
                "d2h_ms": float(d2h_ms),
                "delta_over_full": float(delta_over_full),
                "transfer_over_kernel": float(transfer_over_kernel),
                "kernel_loop_iters": int(loop_iters),
            }
        ],
        "thresholds": {
            "max_delta_over_full": float(args.max_delta_over_full),
            "max_transfer_over_kernel": float(args.max_transfer_over_kernel),
        },
        "calibration": calibration,
        "backend": {
            "requested_device": "cuda",
            "effective_device": str(backend.device),
            "reason": str(backend.reason),
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return 0 if ok or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
