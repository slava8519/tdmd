from __future__ import annotations

import argparse
import json
import os
from statistics import median

import numpy as np

from tdmd.backend import resolve_backend


_RAWKERNEL_NAME = "tdmd_perf_axpy_loop"
_RAWKERNEL_SRC = r"""
extern "C" __global__
void tdmd_perf_axpy_loop(const double* a, const double* b, double* out, const int n) {
    const int i = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= n) return;
    double x = a[i];
    const double y = b[i];
    #pragma unroll 8
    for (int k = 0; k < 64; ++k) {
        x = x * 1.0000001 + y * 0.999999;
    }
    out[i] = x;
}
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
    args = ap.parse_args()

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

    d_r = cp.asarray(r_host)
    d_t = cp.asarray(t_host)
    d_idx = cp.asarray(upd_ids)

    d_a = cp.asarray(r_host.reshape(-1))
    d_b = cp.asarray((0.5 * r_host + 0.1).reshape(-1))
    d_out = cp.zeros_like(d_a)
    kernel = cp.RawKernel(_RAWKERNEL_SRC, _RAWKERNEL_NAME)
    threads = 256
    blocks = (n_vals + threads - 1) // threads

    def _h2d_full():
        _dr = cp.asarray(r_host)
        _dt = cp.asarray(t_host)
        _ = _dr, _dt

    def _h2d_delta():
        d_r[d_idx] = cp.asarray(r_host[upd_ids])
        d_t[d_idx] = cp.asarray(t_host[upd_ids])

    def _kernel():
        kernel(
            (blocks,),
            (threads,),
            (d_a, d_b, d_out, np.int32(n_vals)),
        )

    def _d2h():
        _x = cp.asnumpy(d_out)
        _ = _x

    # Warm-up to amortize JIT/first-use overhead.
    _kernel()
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
            }
        ],
        "thresholds": {
            "max_delta_over_full": float(args.max_delta_over_full),
            "max_transfer_over_kernel": float(args.max_transfer_over_kernel),
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return 0 if ok or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
