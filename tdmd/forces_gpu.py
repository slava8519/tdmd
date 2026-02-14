from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import ComputeBackend, resolve_backend
from .celllist import build_cell_list
from .constants import NUMERICAL_ZERO
from .potentials import EAMAlloyPotential, LennardJones, Morse, TablePotential


def supports_pair_gpu(potential) -> bool:
    return isinstance(potential, (LennardJones, Morse, TablePotential, EAMAlloyPotential))


@dataclass(frozen=True)
class NeighborListDiagnostics:
    attempts: int = 0
    overflow_retries: int = 0
    final_max_neighbors: int = 0
    max_neighbors_used: int = 0
    avg_neighbors: float = 0.0
    overflowed: bool = False


_LAST_NEIGHBOR_DIAGNOSTICS = NeighborListDiagnostics()


_NEIGHBOR_RAWKERNEL_NAME = "tdmd_build_neighbor_list"
_NEIGHBOR_RAWKERNEL_SRC = r"""
extern "C" __global__
void tdmd_build_neighbor_list(
    const double* r,
    const int* target_ids,
    const int* target_cells_xyz,
    const int* cell_atoms_flat,
    const int* cell_starts,
    const int n_targets,
    const int ncell,
    const double box,
    const double cutoff2,
    const int max_neighbors,
    int* out_ids,
    int* out_counts,
    int* overflow_flag
) {
    const int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= n_targets) return;

    const int i = target_ids[tid];
    const int cx = target_cells_xyz[3 * tid + 0];
    const int cy = target_cells_xyz[3 * tid + 1];
    const int cz = target_cells_xyz[3 * tid + 2];
    int count = 0;

    for (int dx = -1; dx <= 1; ++dx) {
        const int nx = (cx + dx + ncell) % ncell;
        for (int dy = -1; dy <= 1; ++dy) {
            const int ny = (cy + dy + ncell) % ncell;
            for (int dz = -1; dz <= 1; ++dz) {
                const int nz = (cz + dz + ncell) % ncell;
                const int lin = (nx * ncell + ny) * ncell + nz;
                const int start = cell_starts[lin];
                const int end = cell_starts[lin + 1];
                for (int p = start; p < end; ++p) {
                    const int j = cell_atoms_flat[p];
                    if (j == i) continue;

                    double rx = r[3 * i + 0] - r[3 * j + 0];
                    double ry = r[3 * i + 1] - r[3 * j + 1];
                    double rz = r[3 * i + 2] - r[3 * j + 2];
                    rx -= box * nearbyint(rx / box);
                    ry -= box * nearbyint(ry / box);
                    rz -= box * nearbyint(rz / box);
                    const double r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 > 0.0 && r2 < cutoff2) {
                        if (count < max_neighbors) {
                            out_ids[tid * max_neighbors + count] = j;
                        } else {
                            atomicExch(overflow_flag, 1);
                        }
                        count++;
                    }
                }
            }
        }
    }
    out_counts[tid] = (count < max_neighbors) ? count : max_neighbors;
}
"""

_LJ_FORCE_RAWKERNEL_NAME = "tdmd_lj_force_from_neighbors"
_LJ_FORCE_RAWKERNEL_SRC = r"""
extern "C" __global__
void tdmd_lj_force_from_neighbors(
    const double* r,
    const int* target_ids,
    const int* neighbor_ids,
    const int n_targets,
    const int max_neighbors,
    const double box,
    const double cutoff2,
    const int* atom_types,
    const double* eps_mat,
    const double* sig_mat,
    const int mat_stride,
    double* out_f
) {
    const int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= n_targets) return;
    const int i = target_ids[tid];
    const int ti = atom_types[i];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    for (int k = 0; k < max_neighbors; ++k) {
        const int j = neighbor_ids[tid * max_neighbors + k];
        if (j < 0) continue;
        double rx = r[3 * i + 0] - r[3 * j + 0];
        double ry = r[3 * i + 1] - r[3 * j + 1];
        double rz = r[3 * i + 2] - r[3 * j + 2];
        rx -= box * nearbyint(rx / box);
        ry -= box * nearbyint(ry / box);
        rz -= box * nearbyint(rz / box);
        const double r2 = rx * rx + ry * ry + rz * rz;
        if (!(r2 > 0.0 && r2 < cutoff2)) continue;
        const int tj = atom_types[j];
        const int pidx = ti * mat_stride + tj;
        const double eps = eps_mat[pidx];
        const double sig = sig_mat[pidx];
        const double inv_r2 = 1.0 / r2;
        const double sr2 = (sig * sig) * inv_r2;
        const double sr6 = sr2 * sr2 * sr2;
        const double sr12 = sr6 * sr6;
        const double coef = 24.0 * eps * (2.0 * sr12 - sr6) * inv_r2;
        fx += coef * rx;
        fy += coef * ry;
        fz += coef * rz;
    }
    out_f[3 * tid + 0] = fx;
    out_f[3 * tid + 1] = fy;
    out_f[3 * tid + 2] = fz;
}
"""

_MORSE_FORCE_RAWKERNEL_NAME = "tdmd_morse_force_from_neighbors"
_MORSE_FORCE_RAWKERNEL_SRC = r"""
extern "C" __global__
void tdmd_morse_force_from_neighbors(
    const double* r,
    const int* target_ids,
    const int* neighbor_ids,
    const int n_targets,
    const int max_neighbors,
    const double box,
    const double cutoff2,
    const int* atom_types,
    const double* d_mat,
    const double* a_mat,
    const double* r0_mat,
    const int mat_stride,
    double* out_f
) {
    const int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= n_targets) return;
    const int i = target_ids[tid];
    const int ti = atom_types[i];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    for (int k = 0; k < max_neighbors; ++k) {
        const int j = neighbor_ids[tid * max_neighbors + k];
        if (j < 0) continue;
        double rx = r[3 * i + 0] - r[3 * j + 0];
        double ry = r[3 * i + 1] - r[3 * j + 1];
        double rz = r[3 * i + 2] - r[3 * j + 2];
        rx -= box * nearbyint(rx / box);
        ry -= box * nearbyint(ry / box);
        rz -= box * nearbyint(rz / box);
        const double r2 = rx * rx + ry * ry + rz * rz;
        if (!(r2 > 0.0 && r2 < cutoff2)) continue;
        const int tj = atom_types[j];
        const int pidx = ti * mat_stride + tj;
        const double D = d_mat[pidx];
        const double a = a_mat[pidx];
        const double r0 = r0_mat[pidx];
        const double rr = sqrt(r2 + 1.0e-30);
        const double x = rr - r0;
        const double exp1 = exp(-a * x);
        const double dUdr = 2.0 * D * a * (1.0 - exp1) * exp1;
        const double coef = dUdr / (rr + 1.0e-30);
        fx += coef * rx;
        fy += coef * ry;
        fz += coef * rz;
    }
    out_f[3 * tid + 0] = fx;
    out_f[3 * tid + 1] = fy;
    out_f[3 * tid + 2] = fz;
}
"""

_TABLE_FORCE_RAWKERNEL_NAME = "tdmd_table_force_from_neighbors"
_TABLE_FORCE_RAWKERNEL_SRC = r"""
extern "C" __global__
void tdmd_table_force_from_neighbors(
    const double* r,
    const int* target_ids,
    const int* neighbor_ids,
    const int n_targets,
    const int max_neighbors,
    const double box,
    const double cutoff2,
    const double* r_grid,
    const double* f_grid,
    const int n_grid,
    double* out_f
) {
    const int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= n_targets) return;
    const int i = target_ids[tid];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    for (int k = 0; k < max_neighbors; ++k) {
        const int j = neighbor_ids[tid * max_neighbors + k];
        if (j < 0) continue;
        double rx = r[3 * i + 0] - r[3 * j + 0];
        double ry = r[3 * i + 1] - r[3 * j + 1];
        double rz = r[3 * i + 2] - r[3 * j + 2];
        rx -= box * nearbyint(rx / box);
        ry -= box * nearbyint(ry / box);
        rz -= box * nearbyint(rz / box);
        const double r2 = rx * rx + ry * ry + rz * rz;
        if (!(r2 > 0.0 && r2 < cutoff2)) continue;
        const double rr = sqrt(r2 + 1.0e-30);
        if (rr < r_grid[0] || rr > r_grid[n_grid - 1]) continue;
        int lo = 0;
        int hi = n_grid - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (r_grid[mid] <= rr) lo = mid;
            else hi = mid;
        }
        const double x0 = r_grid[lo];
        const double x1 = r_grid[lo + 1];
        const double y0 = f_grid[lo];
        const double y1 = f_grid[lo + 1];
        const double t = (rr - x0) / (x1 - x0);
        const double fmag = y0 + t * (y1 - y0);
        const double coef = fmag / (rr + 1.0e-30);
        fx += coef * rx;
        fy += coef * ry;
        fz += coef * rz;
    }
    out_f[3 * tid + 0] = fx;
    out_f[3 * tid + 1] = fy;
    out_f[3 * tid + 2] = fz;
}
"""


def _interp_with_grad_cp(cp, x, y, q):
    qq = q
    n = int(x.size)
    idx = cp.searchsorted(x, qq, side="right") - 1
    idx = cp.clip(idx, 0, n - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    dx = x1 - x0
    t = (qq - x0) / dx
    val = y0 + t * (y1 - y0)
    grad = (y1 - y0) / dx
    m = (qq >= x[0]) & (qq <= x[-1])
    val = cp.where(m, val, 0.0)
    grad = cp.where(m, grad, 0.0)
    return val, grad


def _interp_with_grad_table_cp(cp, x, table, table_idx, q):
    """Piecewise-linear interpolation for row-indexed tables on device."""
    qq = q
    n = int(x.size)
    idx = cp.searchsorted(x, qq, side="right") - 1
    idx = cp.clip(idx, 0, n - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = table[table_idx, idx]
    y1 = table[table_idx, idx + 1]
    dx = x1 - x0
    t = (qq - x0) / dx
    val = y0 + t * (y1 - y0)
    grad = (y1 - y0) / dx
    m = (qq >= x[0]) & (qq <= x[-1])
    val = cp.where(m, val, 0.0)
    grad = cp.where(m, grad, 0.0)
    return val, grad


def _pair_mats_lj(cp, pot: LennardJones, types_i, types_j):
    ni = int(types_i.size)
    nj = int(types_j.size)
    eps = cp.full((ni, nj), float(pot.epsilon), dtype=cp.float64)
    sig = cp.full((ni, nj), float(pot.sigma), dtype=cp.float64)
    if pot.pair_coeffs:
        ti = types_i[:, None]
        tj = types_j[None, :]
        for (a, b), prm in pot.pair_coeffs.items():
            mask = ((ti == int(a)) & (tj == int(b))) | ((ti == int(b)) & (tj == int(a)))
            if bool(cp.any(mask)):
                eps = cp.where(mask, float(prm["epsilon"]), eps)
                sig = cp.where(mask, float(prm["sigma"]), sig)
    return eps, sig


def _pair_mats_morse(cp, pot: Morse, types_i, types_j):
    ni = int(types_i.size)
    nj = int(types_j.size)
    d = cp.full((ni, nj), float(pot.D_e), dtype=cp.float64)
    a = cp.full((ni, nj), float(pot.a), dtype=cp.float64)
    r0 = cp.full((ni, nj), float(pot.r0), dtype=cp.float64)
    if pot.pair_coeffs:
        ti = types_i[:, None]
        tj = types_j[None, :]
        for (x, y), prm in pot.pair_coeffs.items():
            mask = ((ti == int(x)) & (tj == int(y))) | ((ti == int(y)) & (tj == int(x)))
            if bool(cp.any(mask)):
                d = cp.where(mask, float(prm["D_e"]), d)
                a = cp.where(mask, float(prm["a"]), a)
                r0 = cp.where(mask, float(prm["r0"]), r0)
    return d, a, r0


def _flatten_cell_atoms(cl) -> tuple[np.ndarray, np.ndarray]:
    """Flatten cell buckets into CSR-like buffers for device-side neighbor walk."""
    ncell = int(cl.ncell)
    n_cells = ncell * ncell * ncell
    starts = np.zeros((n_cells + 1,), dtype=np.int32)
    flat: list[int] = []

    for lin in range(n_cells):
        x = lin // (ncell * ncell)
        rem = lin % (ncell * ncell)
        y = rem // ncell
        z = rem % ncell
        arr = cl.cell_atoms.get((x, y, z))
        if arr is not None and arr.size:
            flat.extend(int(v) for v in arr.tolist())
        starts[lin + 1] = len(flat)
    flat_arr = np.asarray(flat, dtype=np.int32) if flat else np.zeros((0,), dtype=np.int32)
    return flat_arr, starts


def _neighbor_rawkernel(cp):
    return cp.RawKernel(_NEIGHBOR_RAWKERNEL_SRC, _NEIGHBOR_RAWKERNEL_NAME)


def _lj_force_rawkernel(cp):
    return cp.RawKernel(_LJ_FORCE_RAWKERNEL_SRC, _LJ_FORCE_RAWKERNEL_NAME)


def _morse_force_rawkernel(cp):
    return cp.RawKernel(_MORSE_FORCE_RAWKERNEL_SRC, _MORSE_FORCE_RAWKERNEL_NAME)


def _table_force_rawkernel(cp):
    return cp.RawKernel(_TABLE_FORCE_RAWKERNEL_SRC, _TABLE_FORCE_RAWKERNEL_NAME)


def _pair_matrix_lj_dense(pot: LennardJones, atom_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_type = int(np.max(np.asarray(atom_types, dtype=np.int32))) if atom_types.size else 1
    if pot.pair_coeffs:
        for (a, b) in pot.pair_coeffs.keys():
            max_type = max(max_type, int(a), int(b))
    stride = max(2, max_type + 1)
    eps = np.full((stride, stride), float(pot.epsilon), dtype=np.float64)
    sig = np.full((stride, stride), float(pot.sigma), dtype=np.float64)
    if pot.pair_coeffs:
        for (a, b), prm in pot.pair_coeffs.items():
            ai, bi = int(a), int(b)
            eps[ai, bi] = float(prm["epsilon"])
            eps[bi, ai] = float(prm["epsilon"])
            sig[ai, bi] = float(prm["sigma"])
            sig[bi, ai] = float(prm["sigma"])
    return eps, sig


def _pair_matrix_morse_dense(
    pot: Morse, atom_types: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_type = int(np.max(np.asarray(atom_types, dtype=np.int32))) if atom_types.size else 1
    if pot.pair_coeffs:
        for (a, b) in pot.pair_coeffs.keys():
            max_type = max(max_type, int(a), int(b))
    stride = max(2, max_type + 1)
    d = np.full((stride, stride), float(pot.D_e), dtype=np.float64)
    a = np.full((stride, stride), float(pot.a), dtype=np.float64)
    r0 = np.full((stride, stride), float(pot.r0), dtype=np.float64)
    if pot.pair_coeffs:
        for (x, y), prm in pot.pair_coeffs.items():
            xi, yi = int(x), int(y)
            d[xi, yi] = float(prm["D_e"])
            d[yi, xi] = float(prm["D_e"])
            a[xi, yi] = float(prm["a"])
            a[yi, xi] = float(prm["a"])
            r0[xi, yi] = float(prm["r0"])
            r0[yi, xi] = float(prm["r0"])
    return d, a, r0


def _set_last_neighbor_diagnostics(diag: NeighborListDiagnostics) -> None:
    global _LAST_NEIGHBOR_DIAGNOSTICS
    _LAST_NEIGHBOR_DIAGNOSTICS = diag


def get_last_neighbor_list_diagnostics() -> NeighborListDiagnostics:
    return _LAST_NEIGHBOR_DIAGNOSTICS


def _build_neighbor_list_once(
    *,
    cp,
    r: np.ndarray,
    box: float,
    cutoff: float,
    target_ids: np.ndarray,
    cl,
    cell_atoms_flat: np.ndarray,
    cell_starts: np.ndarray,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    tids = np.asarray(target_ids, dtype=np.int32)
    d_r = cp.asarray(np.asarray(r, dtype=np.float64))
    d_tids = cp.asarray(tids)
    d_tcells = cp.asarray(np.asarray(cl.idx[tids], dtype=np.int32))
    d_flat = cp.asarray(cell_atoms_flat)
    d_starts = cp.asarray(cell_starts)

    max_nei = int(max(1, max_neighbors))
    d_out_ids = cp.full((tids.size, max_nei), -1, dtype=cp.int32)
    d_out_counts = cp.zeros((tids.size,), dtype=cp.int32)
    d_overflow = cp.zeros((1,), dtype=cp.int32)

    kernel = _neighbor_rawkernel(cp)
    threads = 128
    blocks = (int(tids.size) + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            d_r.reshape(-1),
            d_tids,
            d_tcells.reshape(-1),
            d_flat,
            d_starts,
            np.int32(tids.size),
            np.int32(int(cl.ncell)),
            np.float64(float(box)),
            np.float64(float(cutoff) * float(cutoff)),
            np.int32(max_nei),
            d_out_ids.reshape(-1),
            d_out_counts,
            d_overflow,
        ),
    )
    overflowed = int(cp.asnumpy(d_overflow)[0]) != 0
    return cp.asnumpy(d_out_ids), cp.asnumpy(d_out_counts), overflowed


def build_neighbor_list_celllist_backend(
    *,
    r: np.ndarray,
    box: float,
    cutoff: float,
    rc: float,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    atom_types: np.ndarray | None = None,
    backend: ComputeBackend | None = None,
    device: str = "cpu",
    max_neighbors: int = 256,
    max_retries: int = 6,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build per-target neighbor list on CUDA from cell-list buckets."""
    if backend is None:
        backend = resolve_backend(device)
    if backend.device != "cuda":
        return None
    cp = backend.xp

    tids = np.asarray(target_ids, dtype=np.int32)
    cids = np.asarray(candidate_ids, dtype=np.int32)
    if tids.size == 0 or cids.size == 0:
        _set_last_neighbor_diagnostics(NeighborListDiagnostics())
        return np.full((tids.size, 0), -1, dtype=np.int32), np.zeros((tids.size,), dtype=np.int32)

    cl = build_cell_list(r, cids, float(box), rc=float(rc))
    cell_atoms_flat, cell_starts = _flatten_cell_atoms(cl)
    max_nei = int(max(1, max_neighbors))
    upper_bound = int(max(1, cids.size))
    retries = int(max(0, max_retries))

    attempts = 0
    overflow_retries = 0
    last_ids: np.ndarray | None = None
    last_counts: np.ndarray | None = None
    last_overflow = True

    for _ in range(retries + 1):
        attempts += 1
        out_ids, out_counts, overflowed = _build_neighbor_list_once(
            cp=cp,
            r=r,
            box=box,
            cutoff=cutoff,
            target_ids=tids,
            cl=cl,
            cell_atoms_flat=cell_atoms_flat,
            cell_starts=cell_starts,
            max_neighbors=max_nei,
        )
        last_ids, last_counts, last_overflow = out_ids, out_counts, overflowed
        if not overflowed:
            break
        overflow_retries += 1
        if max_nei >= upper_bound:
            break
        max_nei = min(upper_bound, max_nei * 2)

    if last_counts is None or last_ids is None:
        _set_last_neighbor_diagnostics(
            NeighborListDiagnostics(
                attempts=attempts,
                overflow_retries=overflow_retries,
                final_max_neighbors=max_nei,
                overflowed=True,
            )
        )
        return None

    max_used = int(last_counts.max()) if last_counts.size else 0
    avg_used = float(last_counts.mean()) if last_counts.size else 0.0
    _set_last_neighbor_diagnostics(
        NeighborListDiagnostics(
            attempts=attempts,
            overflow_retries=overflow_retries,
            final_max_neighbors=max_nei,
            max_neighbors_used=max_used,
            avg_neighbors=avg_used,
            overflowed=bool(last_overflow),
        )
    )
    if last_overflow:
        return None
    return last_ids, last_counts


def _forces_from_neighbor_matrix_cp(
    *,
    cp,
    r: np.ndarray,
    box: float,
    cutoff: float,
    potential,
    target_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    atom_types: np.ndarray,
) -> np.ndarray:
    tids = np.asarray(target_ids, dtype=np.int32)
    if tids.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if neighbor_ids.size == 0:
        return np.zeros((tids.size, 3), dtype=np.float64)

    d_tids = cp.asarray(tids)
    d_nei = cp.asarray(np.asarray(neighbor_ids, dtype=np.int32))
    valid = d_nei >= 0
    js_safe = cp.where(valid, d_nei, 0)
    max_neighbors = int(neighbor_ids.shape[1]) if neighbor_ids.ndim == 2 else 0
    d_types = cp.asarray(np.asarray(atom_types, dtype=np.int32))
    cutoff2 = float(cutoff) * float(cutoff)

    if isinstance(potential, LennardJones):
        eps, sig = _pair_matrix_lj_dense(potential, np.asarray(atom_types, dtype=np.int32))
        d_eps = cp.asarray(eps)
        d_sig = cp.asarray(sig)
        d_out = cp.zeros((tids.size, 3), dtype=cp.float64)
        kernel = _lj_force_rawkernel(cp)
        threads = 128
        blocks = (int(tids.size) + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                cp.asarray(np.asarray(r, dtype=np.float64)).reshape(-1),
                d_tids,
                d_nei.reshape(-1),
                np.int32(tids.size),
                np.int32(max_neighbors),
                np.float64(float(box)),
                np.float64(cutoff2),
                d_types,
                d_eps.reshape(-1),
                d_sig.reshape(-1),
                np.int32(int(eps.shape[0])),
                d_out.reshape(-1),
            ),
        )
        return cp.asnumpy(d_out)

    if isinstance(potential, Morse):
        d_mat, a_mat, r0_mat = _pair_matrix_morse_dense(
            potential, np.asarray(atom_types, dtype=np.int32)
        )
        d_d = cp.asarray(d_mat)
        d_a = cp.asarray(a_mat)
        d_r0 = cp.asarray(r0_mat)
        d_out = cp.zeros((tids.size, 3), dtype=cp.float64)
        kernel = _morse_force_rawkernel(cp)
        threads = 128
        blocks = (int(tids.size) + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                cp.asarray(np.asarray(r, dtype=np.float64)).reshape(-1),
                d_tids,
                d_nei.reshape(-1),
                np.int32(tids.size),
                np.int32(max_neighbors),
                np.float64(float(box)),
                np.float64(cutoff2),
                d_types,
                d_d.reshape(-1),
                d_a.reshape(-1),
                d_r0.reshape(-1),
                np.int32(int(d_mat.shape[0])),
                d_out.reshape(-1),
            ),
        )
        return cp.asnumpy(d_out)

    if isinstance(potential, TablePotential):
        d_out = cp.zeros((tids.size, 3), dtype=cp.float64)
        d_r_grid = cp.asarray(np.asarray(potential.r_grid, dtype=np.float64))
        d_f_grid = cp.asarray(np.asarray(potential.f_grid, dtype=np.float64))
        kernel = _table_force_rawkernel(cp)
        threads = 128
        blocks = (int(tids.size) + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                cp.asarray(np.asarray(r, dtype=np.float64)).reshape(-1),
                d_tids,
                d_nei.reshape(-1),
                np.int32(tids.size),
                np.int32(max_neighbors),
                np.float64(float(box)),
                np.float64(cutoff2),
                d_r_grid,
                d_f_grid,
                np.int32(int(d_r_grid.size)),
                d_out.reshape(-1),
            ),
        )
        return cp.asnumpy(d_out)

    rr_t = cp.asarray(np.asarray(r[tids], dtype=np.float64))
    rr_j = cp.asarray(np.asarray(r, dtype=np.float64))[js_safe]
    dr = rr_t[:, None, :] - rr_j
    dr = dr - float(box) * cp.rint(dr / float(box))
    r2 = cp.sum(dr * dr, axis=2)
    mask = valid & (r2 > 0.0) & (r2 < cutoff2)

    return np.zeros((tids.size, 3), dtype=np.float64)


def forces_on_targets_pair_backend(
    *,
    r: np.ndarray,
    box: float,
    cutoff: float,
    potential,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    atom_types: np.ndarray,
    backend: ComputeBackend | None = None,
    device: str = "cpu",
) -> np.ndarray | None:
    if backend is None:
        backend = resolve_backend(device)
    if backend.device != "cuda":
        return None
    if not supports_pair_gpu(potential):
        return None

    cp = backend.xp
    tids = np.asarray(target_ids, dtype=np.int32)
    cids = np.asarray(candidate_ids, dtype=np.int32)
    if tids.size == 0 or cids.size == 0:
        return np.zeros((tids.size, 3), dtype=np.float64)

    rr_t = cp.asarray(np.asarray(r[tids], dtype=np.float64))
    rr_c = cp.asarray(np.asarray(r[cids], dtype=np.float64))
    dr = rr_t[:, None, :] - rr_c[None, :, :]
    dr = dr - float(box) * cp.rint(dr / float(box))
    r2 = cp.sum(dr * dr, axis=2)

    cutoff2 = float(cutoff) * float(cutoff)
    mask = (r2 > 0.0) & (r2 < cutoff2)

    types_i = cp.asarray(np.asarray(atom_types[tids], dtype=np.int32))
    types_j = cp.asarray(np.asarray(atom_types[cids], dtype=np.int32))

    if isinstance(potential, LennardJones):
        eps, sig = _pair_mats_lj(cp, potential, types_i, types_j)
        inv_r2 = cp.where(mask, 1.0 / r2, 0.0)
        sr2 = (sig * sig) * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        coef = cp.where(mask, 24.0 * eps * (2.0 * sr12 - sr6) * inv_r2, 0.0)
    elif isinstance(potential, Morse):
        d, a, r0 = _pair_mats_morse(cp, potential, types_i, types_j)
        rr = cp.sqrt(r2 + NUMERICAL_ZERO)
        x = rr - r0
        exp1 = cp.exp(-a * x)
        dUdr = 2.0 * d * a * (1.0 - exp1) * exp1
        coef = cp.where(mask, dUdr / (rr + NUMERICAL_ZERO), 0.0)
    elif isinstance(potential, TablePotential):
        rr = cp.sqrt(r2 + NUMERICAL_ZERO)
        r_grid = cp.asarray(np.asarray(potential.r_grid, dtype=np.float64))
        f_grid = cp.asarray(np.asarray(potential.f_grid, dtype=np.float64))
        force_mag = cp.interp(rr, r_grid, f_grid, left=0.0, right=0.0)
        coef = cp.where(mask, force_mag / (rr + NUMERICAL_ZERO), 0.0)
    elif isinstance(potential, EAMAlloyPotential):
        cand = np.unique(np.concatenate([cids, tids]).astype(np.int32))
        if cand.size == 0:
            return np.zeros((tids.size, 3), dtype=np.float64)
        pos_in_cand = {int(gid): int(i) for i, gid in enumerate(cand.tolist())}
        tgt_local = np.asarray([pos_in_cand[int(g)] for g in tids.tolist()], dtype=np.int32)

        rr_c = cp.asarray(np.asarray(r[cand], dtype=np.float64))
        elem_np = potential._types_to_elem_idx(np.asarray(atom_types[cand], dtype=np.int32))
        elem = cp.asarray(elem_np.astype(np.int32))
        n = int(cand.size)

        drc = rr_c[:, None, :] - rr_c[None, :, :]
        drc = drc - float(box) * cp.rint(drc / float(box))
        r2c = cp.sum(drc * drc, axis=2)
        rc = cp.sqrt(r2c + NUMERICAL_ZERO)
        m = (r2c > 0.0) & (r2c < cutoff2)

        grid_r = cp.asarray(np.asarray(potential.grid_r, dtype=np.float64))
        grid_rho = cp.asarray(np.asarray(potential.grid_rho, dtype=np.float64))
        density = cp.asarray(np.asarray(potential.density_table, dtype=np.float64))
        embed = cp.asarray(np.asarray(potential.embed_table, dtype=np.float64))
        phi = cp.asarray(np.asarray(potential.phi_table, dtype=np.float64))
        nelem = int(embed.shape[0])

        idx_j = cp.broadcast_to(elem[None, :], (n, n))
        idx_i = cp.broadcast_to(elem[:, None], (n, n))
        rho_j, drho_col = _interp_with_grad_table_cp(cp, grid_r, density, idx_j, rc)
        rho_i, drho_row = _interp_with_grad_table_cp(cp, grid_r, density, idx_i, rc)
        rho_j = cp.where(m, rho_j, 0.0)
        drho_col = cp.where(m, drho_col, 0.0)
        drho_row = cp.where(m, drho_row, 0.0)
        rho_acc = cp.sum(rho_j, axis=1)

        _F, dF = _interp_with_grad_table_cp(cp, grid_rho, embed, elem, rho_acc)

        phi_flat = phi.reshape(nelem * nelem, -1)
        phi_idx = idx_i * nelem + idx_j
        _phi, dphi = _interp_with_grad_table_cp(cp, grid_r, phi_flat, phi_idx, rc)
        dphi = cp.where(m, dphi, 0.0)

        dEdr = dphi + dF[:, None] * drho_col + dF[None, :] * drho_row
        coef_c = cp.where(m, -dEdr / (rc + NUMERICAL_ZERO), 0.0)
        f_c = cp.sum(coef_c[:, :, None] * drc, axis=1)
        return cp.asnumpy(f_c[cp.asarray(tgt_local, dtype=cp.int32)])
    else:
        return None

    ff = cp.sum(coef[:, :, None] * dr, axis=1)
    return cp.asnumpy(ff)


def forces_on_targets_celllist_backend(
    *,
    r: np.ndarray,
    box: float,
    cutoff: float,
    rc: float,
    potential,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    atom_types: np.ndarray,
    backend: ComputeBackend | None = None,
    device: str = "cpu",
) -> np.ndarray | None:
    """GPU force path with CPU-equivalent cell-list candidate pruning.

    Cell buckets and neighbor candidate sets are built identically to CPU cell-list,
    then each target-candidate interaction slice is evaluated on GPU.
    """
    if backend is None:
        backend = resolve_backend(device)
    if backend.device != "cuda":
        return None
    if not supports_pair_gpu(potential):
        return None
    if isinstance(potential, EAMAlloyPotential):
        return None

    tids = np.asarray(target_ids, dtype=np.int32)
    cids = np.asarray(candidate_ids, dtype=np.int32)
    if tids.size == 0 or cids.size == 0:
        return np.zeros((tids.size, 3), dtype=np.float64)

    max_neighbors = int(max(64, min(1024, cids.size)))
    built = build_neighbor_list_celllist_backend(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=rc,
        target_ids=tids,
        candidate_ids=cids,
        backend=backend,
        max_neighbors=max_neighbors,
    )
    if built is None:
        # Avoid fallback to the legacy Python-loop cell-list path.
        return forces_on_targets_pair_backend(
            r=r,
            box=box,
            cutoff=cutoff,
            potential=potential,
            target_ids=tids,
            candidate_ids=cids,
            atom_types=atom_types,
            backend=backend,
        )

    neighbor_ids, _counts = built
    cp = backend.xp
    return _forces_from_neighbor_matrix_cp(
        cp=cp,
        r=r,
        box=box,
        cutoff=cutoff,
        potential=potential,
        target_ids=tids,
        neighbor_ids=neighbor_ids,
        atom_types=atom_types,
    )
