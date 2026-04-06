from __future__ import annotations

import weakref
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


@dataclass(frozen=True)
class _DeviceNeighborList:
    d_ids: object
    d_counts: object


@dataclass
class _GPUStateCache:
    host_id: int = -1
    shape: tuple[int, int] = (0, 0)
    d_r: object | None = None
    d_atom_types: object | None = None
    dirty_mask: np.ndarray | None = None


@dataclass(frozen=True)
class DeviceStateSyncDiagnostics:
    last_synced_atoms: int = 0
    full_sync: bool = False
    used_dirty_tracking: bool = False


_GPU_STATE_CACHE = _GPUStateCache()
_LAST_DEVICE_STATE_SYNC_DIAGNOSTICS = DeviceStateSyncDiagnostics()


@dataclass(frozen=True)
class _DeviceTablePotentialState:
    d_r_grid: object
    d_f_grid: object


@dataclass(frozen=True)
class _DeviceEAMPotentialState:
    d_grid_rho: object
    d_grid_r: object
    d_embed: object
    d_density: object
    d_phi_flat: object
    nelem: int
    nrho: int
    nr: int


_GPU_POTENTIAL_CACHE: dict[int, tuple[weakref.ReferenceType[object], object]] = {}
_RAWKERNEL_CACHE: dict[tuple[int, int, str], object] = {}


def _set_last_device_state_sync_diagnostics(diag: DeviceStateSyncDiagnostics) -> None:
    global _LAST_DEVICE_STATE_SYNC_DIAGNOSTICS
    _LAST_DEVICE_STATE_SYNC_DIAGNOSTICS = diag


def get_last_device_state_sync_diagnostics() -> DeviceStateSyncDiagnostics:
    return _LAST_DEVICE_STATE_SYNC_DIAGNOSTICS


def reset_device_state_cache() -> None:
    global _GPU_STATE_CACHE
    _GPU_STATE_CACHE = _GPU_STATE_CACHE.__class__()
    _set_last_device_state_sync_diagnostics(DeviceStateSyncDiagnostics())


def reset_rawkernel_cache() -> None:
    _RAWKERNEL_CACHE.clear()


def reset_device_potential_cache() -> None:
    _GPU_POTENTIAL_CACHE.clear()


def _get_cached_potential_state(potential):
    key = int(id(potential))
    cached = _GPU_POTENTIAL_CACHE.get(key)
    if cached is None:
        return None
    pot_ref, state = cached
    if pot_ref() is potential:
        return state
    _GPU_POTENTIAL_CACHE.pop(key, None)
    return None


def _set_cached_potential_state(potential, state) -> None:
    _GPU_POTENTIAL_CACHE[int(id(potential))] = (weakref.ref(potential), state)


def mark_device_state_dirty(r: np.ndarray, ids: np.ndarray | None = None) -> None:
    global _GPU_STATE_CACHE

    rr = np.asarray(r, dtype=np.float64)
    if (
        _GPU_STATE_CACHE.d_r is None
        or _GPU_STATE_CACHE.d_atom_types is None
        or _GPU_STATE_CACHE.host_id != id(rr)
        or _GPU_STATE_CACHE.shape != rr.shape
    ):
        return
    if _GPU_STATE_CACHE.dirty_mask is None or _GPU_STATE_CACHE.dirty_mask.shape[0] != rr.shape[0]:
        _GPU_STATE_CACHE.dirty_mask = np.zeros((rr.shape[0],), dtype=bool)
    if ids is None:
        _GPU_STATE_CACHE.dirty_mask[:] = True
        return
    idx = np.asarray(ids, dtype=np.int32)
    if idx.size:
        _GPU_STATE_CACHE.dirty_mask[np.unique(idx)] = True


def _get_device_state(
    *,
    cp,
    r: np.ndarray,
    atom_types: np.ndarray,
    update_ids: np.ndarray | None = None,
    prefer_marked_dirty: bool = False,
):
    """Maintain persistent device arrays for positions and atom types."""
    global _GPU_STATE_CACHE

    rr = np.asarray(r, dtype=np.float64)
    at = np.asarray(atom_types, dtype=np.int32)
    reset = (
        _GPU_STATE_CACHE.d_r is None
        or _GPU_STATE_CACHE.d_atom_types is None
        or _GPU_STATE_CACHE.host_id != id(rr)
        or _GPU_STATE_CACHE.shape != rr.shape
    )
    if reset:
        _GPU_STATE_CACHE.host_id = id(rr)
        _GPU_STATE_CACHE.shape = rr.shape
        _GPU_STATE_CACHE.d_r = cp.asarray(rr)
        _GPU_STATE_CACHE.d_atom_types = cp.asarray(at)
        _GPU_STATE_CACHE.dirty_mask = np.zeros((rr.shape[0],), dtype=bool)
        _set_last_device_state_sync_diagnostics(
            DeviceStateSyncDiagnostics(
                last_synced_atoms=int(rr.shape[0]),
                full_sync=True,
                used_dirty_tracking=bool(prefer_marked_dirty),
            )
        )
    else:
        if _GPU_STATE_CACHE.dirty_mask is None or _GPU_STATE_CACHE.dirty_mask.shape[0] != rr.shape[0]:
            _GPU_STATE_CACHE.dirty_mask = np.zeros((rr.shape[0],), dtype=bool)
        if update_ids is None:
            if prefer_marked_dirty:
                sync_ids = np.flatnonzero(_GPU_STATE_CACHE.dirty_mask).astype(np.int32, copy=False)
            else:
                sync_ids = np.arange(rr.shape[0], dtype=np.int32)
        else:
            uniq = np.unique(np.asarray(update_ids, dtype=np.int32))
            if prefer_marked_dirty:
                sync_ids = uniq[_GPU_STATE_CACHE.dirty_mask[uniq]] if uniq.size else uniq
            else:
                sync_ids = uniq
        if sync_ids.size:
            d_idx = cp.asarray(sync_ids)
            _GPU_STATE_CACHE.d_r[d_idx] = cp.asarray(rr[sync_ids])
            _GPU_STATE_CACHE.d_atom_types[d_idx] = cp.asarray(at[sync_ids])
            _GPU_STATE_CACHE.dirty_mask[sync_ids] = False
        _set_last_device_state_sync_diagnostics(
            DeviceStateSyncDiagnostics(
                last_synced_atoms=int(sync_ids.size),
                full_sync=False,
                used_dirty_tracking=bool(prefer_marked_dirty),
            )
        )
    return _GPU_STATE_CACHE.d_r, _GPU_STATE_CACHE.d_atom_types


def _peek_device_state(*, r: np.ndarray):
    rr = np.asarray(r, dtype=np.float64)
    if (
        _GPU_STATE_CACHE.d_r is None
        or _GPU_STATE_CACHE.d_atom_types is None
        or _GPU_STATE_CACHE.host_id != id(rr)
        or _GPU_STATE_CACHE.shape != rr.shape
    ):
        return None, None
    return _GPU_STATE_CACHE.d_r, _GPU_STATE_CACHE.d_atom_types


def _get_device_potential_state(cp, potential):
    cached = _get_cached_potential_state(potential)
    if isinstance(potential, TablePotential):
        if isinstance(cached, _DeviceTablePotentialState):
            return cached
        state = _DeviceTablePotentialState(
            d_r_grid=cp.asarray(np.asarray(potential.r_grid, dtype=np.float64)),
            d_f_grid=cp.asarray(np.asarray(potential.f_grid, dtype=np.float64)),
        )
        _set_cached_potential_state(potential, state)
        return state
    if isinstance(potential, EAMAlloyPotential):
        if isinstance(cached, _DeviceEAMPotentialState):
            return cached
        nelem = int(potential.embed_table.shape[0])
        nrho = int(potential.grid_rho.size)
        nr = int(potential.grid_r.size)
        phi_flat = np.asarray(potential.phi_table, dtype=np.float64).reshape(nelem * nelem, -1)
        state = _DeviceEAMPotentialState(
            d_grid_rho=cp.asarray(np.asarray(potential.grid_rho, dtype=np.float64)),
            d_grid_r=cp.asarray(np.asarray(potential.grid_r, dtype=np.float64)),
            d_embed=cp.asarray(np.asarray(potential.embed_table, dtype=np.float64)),
            d_density=cp.asarray(np.asarray(potential.density_table, dtype=np.float64)),
            d_phi_flat=cp.asarray(phi_flat),
            nelem=nelem,
            nrho=nrho,
            nr=nr,
        )
        _set_cached_potential_state(potential, state)
        return state
    return None


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
    const int* neighbor_counts,
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
    const int count = neighbor_counts[tid];
    for (int k = 0; k < count; ++k) {
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
    const int* neighbor_counts,
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
    const int count = neighbor_counts[tid];
    for (int k = 0; k < count; ++k) {
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
    const int* neighbor_counts,
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
    const int count = neighbor_counts[tid];
    for (int k = 0; k < count; ++k) {
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

_EAM_DENSITY_RAWKERNEL_NAME = "tdmd_eam_density_from_neighbors"
_EAM_EMBED_DERIV_RAWKERNEL_NAME = "tdmd_eam_embed_deriv"
_EAM_FORCE_RAWKERNEL_NAME = "tdmd_eam_force_from_neighbors"
_EAM_RAWKERNEL_SRC = r"""
__device__ inline void tdmd_interp_value_grad(
    const double* x,
    const double* table_row,
    const int n,
    const double q,
    double* out_val,
    double* out_grad
) {
    if (n < 2 || q < x[0] || q > x[n - 1]) {
        *out_val = 0.0;
        *out_grad = 0.0;
        return;
    }
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (x[mid] <= q) lo = mid;
        else hi = mid;
    }
    const double x0 = x[lo];
    const double x1 = x[lo + 1];
    const double y0 = table_row[lo];
    const double y1 = table_row[lo + 1];
    const double dx = x1 - x0;
    if (dx == 0.0) {
        *out_val = y0;
        *out_grad = 0.0;
        return;
    }
    const double t = (q - x0) / dx;
    *out_val = y0 + t * (y1 - y0);
    *out_grad = (y1 - y0) / dx;
}

extern "C" __global__
void tdmd_eam_density_from_neighbors(
    const double* r,
    const int* target_ids,
    const int* neighbor_ids,
    const int* neighbor_counts,
    const int n_targets,
    const int max_neighbors,
    const double box,
    const double cutoff2,
    const int* elem_idx,
    const double* grid_r,
    const int nr,
    const double* density_table,
    double* rho_acc
) {
    const int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= n_targets) return;
    const int i = target_ids[tid];
    double rho = 0.0;
    const int count = neighbor_counts[tid];
    for (int k = 0; k < count; ++k) {
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
        const int aj = elem_idx[j];
        double rho_ij = 0.0;
        double drho = 0.0;
        tdmd_interp_value_grad(grid_r, density_table + aj * nr, nr, rr, &rho_ij, &drho);
        rho += rho_ij;
    }
    rho_acc[tid] = rho;
}

extern "C" __global__
void tdmd_eam_embed_deriv(
    const int* target_ids,
    const int n_targets,
    const int* elem_idx,
    const double* grid_rho,
    const int nrho,
    const double* embed_table,
    const double* rho_acc,
    double* out_dF
) {
    const int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= n_targets) return;
    const int i = target_ids[tid];
    const int ai = elem_idx[i];
    double F = 0.0;
    double dF = 0.0;
    tdmd_interp_value_grad(grid_rho, embed_table + ai * nrho, nrho, rho_acc[tid], &F, &dF);
    out_dF[tid] = dF;
}

extern "C" __global__
void tdmd_eam_force_from_neighbors(
    const double* r,
    const int* target_ids,
    const int* neighbor_ids,
    const int* neighbor_counts,
    const int n_targets,
    const int max_neighbors,
    const double box,
    const double cutoff2,
    const int* elem_idx,
    const int* gid_to_local,
    const double* dF,
    const double* grid_r,
    const int nr,
    const double* density_table,
    const double* phi_table_flat,
    const int nelem,
    double* out_f
) {
    const int tid = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid >= n_targets) return;
    const int i = target_ids[tid];
    const int ai = elem_idx[i];
    const double dFi = dF[tid];
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    const int count = neighbor_counts[tid];
    for (int k = 0; k < count; ++k) {
        const int j = neighbor_ids[tid * max_neighbors + k];
        if (j < 0) continue;
        const int lj = gid_to_local[j];
        if (lj < 0) continue;
        double rx = r[3 * i + 0] - r[3 * j + 0];
        double ry = r[3 * i + 1] - r[3 * j + 1];
        double rz = r[3 * i + 2] - r[3 * j + 2];
        rx -= box * nearbyint(rx / box);
        ry -= box * nearbyint(ry / box);
        rz -= box * nearbyint(rz / box);
        const double r2 = rx * rx + ry * ry + rz * rz;
        if (!(r2 > 0.0 && r2 < cutoff2)) continue;
        const double rr = sqrt(r2 + 1.0e-30);
        const int aj = elem_idx[j];
        const double dFj = dF[lj];

        double phi = 0.0;
        double dphi = 0.0;
        tdmd_interp_value_grad(
            grid_r,
            phi_table_flat + (ai * nelem + aj) * nr,
            nr,
            rr,
            &phi,
            &dphi
        );

        double rho_ij = 0.0;
        double drho_col = 0.0;
        tdmd_interp_value_grad(
            grid_r,
            density_table + aj * nr,
            nr,
            rr,
            &rho_ij,
            &drho_col
        );

        double rho_ji = 0.0;
        double drho_row = 0.0;
        tdmd_interp_value_grad(
            grid_r,
            density_table + ai * nr,
            nr,
            rr,
            &rho_ji,
            &drho_row
        );

        const double dEdr = dphi + dFi * drho_col + dFj * drho_row;
        const double coef = -dEdr / (rr + 1.0e-30);
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


def _rawkernel_cache_key(cp, name: str) -> tuple[int, int, str]:
    device_id = -1
    try:
        device_id = int(cp.cuda.Device().id)
    except Exception:
        device_id = -1
    return (id(cp), device_id, str(name))


def _cached_rawkernel(cp, src: str, name: str):
    key = _rawkernel_cache_key(cp, name)
    kernel = _RAWKERNEL_CACHE.get(key)
    if kernel is None:
        kernel = cp.RawKernel(src, name)
        _RAWKERNEL_CACHE[key] = kernel
    return kernel


def _neighbor_rawkernel(cp):
    return _cached_rawkernel(cp, _NEIGHBOR_RAWKERNEL_SRC, _NEIGHBOR_RAWKERNEL_NAME)


def _lj_force_rawkernel(cp):
    return _cached_rawkernel(cp, _LJ_FORCE_RAWKERNEL_SRC, _LJ_FORCE_RAWKERNEL_NAME)


def _morse_force_rawkernel(cp):
    return _cached_rawkernel(cp, _MORSE_FORCE_RAWKERNEL_SRC, _MORSE_FORCE_RAWKERNEL_NAME)


def _table_force_rawkernel(cp):
    return _cached_rawkernel(cp, _TABLE_FORCE_RAWKERNEL_SRC, _TABLE_FORCE_RAWKERNEL_NAME)


def _eam_density_rawkernel(cp):
    return _cached_rawkernel(cp, _EAM_RAWKERNEL_SRC, _EAM_DENSITY_RAWKERNEL_NAME)


def _eam_embed_deriv_rawkernel(cp):
    return _cached_rawkernel(cp, _EAM_RAWKERNEL_SRC, _EAM_EMBED_DERIV_RAWKERNEL_NAME)


def _eam_force_rawkernel(cp):
    return _cached_rawkernel(cp, _EAM_RAWKERNEL_SRC, _EAM_FORCE_RAWKERNEL_NAME)


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
    d_r,
    box: float,
    cutoff: float,
    target_ids: np.ndarray,
    cl,
    cell_atoms_flat: np.ndarray,
    cell_starts: np.ndarray,
    max_neighbors: int,
) -> tuple[_DeviceNeighborList, bool]:
    tids = np.asarray(target_ids, dtype=np.int32)
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
    return _DeviceNeighborList(d_ids=d_out_ids, d_counts=d_out_counts), overflowed


def _build_neighbor_list_celllist_backend_device(
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
    prefer_marked_dirty: bool = False,
) -> _DeviceNeighborList | None:
    """Build a device-resident per-target neighbor list from CPU-reference cell buckets."""
    if backend is None:
        backend = resolve_backend(device)
    if backend.device != "cuda":
        return None
    cp = backend.xp

    tids = np.asarray(target_ids, dtype=np.int32)
    cids = np.asarray(candidate_ids, dtype=np.int32)
    if tids.size == 0 or cids.size == 0:
        _set_last_neighbor_diagnostics(NeighborListDiagnostics())
        return _DeviceNeighborList(
            d_ids=cp.full((tids.size, 0), -1, dtype=cp.int32),
            d_counts=cp.zeros((tids.size,), dtype=cp.int32),
        )

    if atom_types is None:
        atom_types_arr = np.zeros((np.asarray(r).shape[0],), dtype=np.int32)
    else:
        atom_types_arr = np.asarray(atom_types, dtype=np.int32)
    update_ids = np.unique(np.concatenate((tids, cids)).astype(np.int32))
    d_r, _ = _get_device_state(
        cp=cp,
        r=r,
        atom_types=atom_types_arr,
        update_ids=update_ids,
        prefer_marked_dirty=bool(prefer_marked_dirty),
    )

    cl = build_cell_list(r, cids, float(box), rc=float(rc))
    cell_atoms_flat, cell_starts = _flatten_cell_atoms(cl)
    max_nei = int(max(1, max_neighbors))
    upper_bound = int(max(1, cids.size))
    retries = int(max(0, max_retries))

    attempts = 0
    overflow_retries = 0
    last_built: _DeviceNeighborList | None = None
    last_overflow = True

    for _ in range(retries + 1):
        attempts += 1
        built_once, overflowed = _build_neighbor_list_once(
            cp=cp,
            d_r=d_r,
            box=box,
            cutoff=cutoff,
            target_ids=tids,
            cl=cl,
            cell_atoms_flat=cell_atoms_flat,
            cell_starts=cell_starts,
            max_neighbors=max_nei,
        )
        last_built, last_overflow = built_once, overflowed
        if not overflowed:
            break
        overflow_retries += 1
        if max_nei >= upper_bound:
            break
        max_nei = min(upper_bound, max_nei * 2)

    if last_built is None:
        _set_last_neighbor_diagnostics(
            NeighborListDiagnostics(
                attempts=attempts,
                overflow_retries=overflow_retries,
                final_max_neighbors=max_nei,
                overflowed=True,
            )
        )
        return None

    last_counts = cp.asnumpy(last_built.d_counts)
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
    return last_built


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
    prefer_marked_dirty: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build per-target neighbor list on CUDA from cell-list buckets and materialize it on host."""
    if backend is None:
        backend = resolve_backend(device)
    built = _build_neighbor_list_celllist_backend_device(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=rc,
        target_ids=target_ids,
        candidate_ids=candidate_ids,
        atom_types=atom_types,
        backend=backend,
        max_neighbors=max_neighbors,
        max_retries=max_retries,
        prefer_marked_dirty=prefer_marked_dirty,
    )
    if built is None:
        return None
    cp = backend.xp
    return cp.asnumpy(built.d_ids), cp.asnumpy(built.d_counts)


def _forces_from_neighbor_matrix_cp(
    *,
    cp,
    r: np.ndarray,
    box: float,
    cutoff: float,
    potential,
    target_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    neighbor_counts: np.ndarray | object | None = None,
    atom_types: np.ndarray,
    d_r=None,
    d_types=None,
) -> np.ndarray:
    tids = np.asarray(target_ids, dtype=np.int32)
    if tids.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if neighbor_ids is None:
        return np.zeros((tids.size, 3), dtype=np.float64)
    if isinstance(neighbor_ids, cp.ndarray):
        d_nei = neighbor_ids.astype(cp.int32, copy=False)
    else:
        d_nei = cp.asarray(np.asarray(neighbor_ids, dtype=np.int32))
    if d_nei.size == 0:
        return np.zeros((tids.size, 3), dtype=np.float64)

    d_tids = cp.asarray(tids)
    valid = d_nei >= 0
    js_safe = cp.where(valid, d_nei, 0)
    if neighbor_counts is None:
        d_counts = cp.sum(valid, axis=1, dtype=cp.int32) if d_nei.ndim == 2 else cp.zeros((tids.size,), dtype=cp.int32)
    elif isinstance(neighbor_counts, cp.ndarray):
        d_counts = neighbor_counts.astype(cp.int32, copy=False)
    else:
        d_counts = cp.asarray(np.asarray(neighbor_counts, dtype=np.int32))
    max_neighbors = int(d_nei.shape[1]) if d_nei.ndim == 2 else 0
    if d_r is None or d_types is None:
        js = cp.asnumpy(d_nei[valid]).astype(np.int32, copy=False)
        update_ids = np.unique(np.concatenate((tids, js)).astype(np.int32))
        d_r, d_types = _get_device_state(
            cp=cp,
            r=r,
            atom_types=np.asarray(atom_types, dtype=np.int32),
            update_ids=update_ids,
        )
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
                d_r.reshape(-1),
                d_tids,
                d_nei.reshape(-1),
                d_counts,
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
                d_r.reshape(-1),
                d_tids,
                d_nei.reshape(-1),
                d_counts,
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
        pot_state = _get_device_potential_state(cp, potential)
        d_r_grid = pot_state.d_r_grid
        d_f_grid = pot_state.d_f_grid
        kernel = _table_force_rawkernel(cp)
        threads = 128
        blocks = (int(tids.size) + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                d_r.reshape(-1),
                d_tids,
                d_nei.reshape(-1),
                d_counts,
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

    rr_t = d_r[d_tids]
    rr_j = d_r[js_safe]
    dr = rr_t[:, None, :] - rr_j
    dr = dr - float(box) * cp.rint(dr / float(box))
    return np.zeros((tids.size, 3), dtype=np.float64)


def _forces_eam_on_candidate_union_cp(
    *,
    cp,
    r: np.ndarray,
    box: float,
    cutoff: float,
    potential: EAMAlloyPotential,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    atom_types: np.ndarray,
    backend: ComputeBackend,
    prefer_marked_dirty: bool = False,
) -> np.ndarray | None:
    tids = np.asarray(target_ids, dtype=np.int32)
    cids = np.asarray(candidate_ids, dtype=np.int32)
    if tids.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    union = np.unique(np.concatenate((tids, cids)).astype(np.int32))
    if union.size == 0:
        return np.zeros((tids.size, 3), dtype=np.float64)

    max_neighbors = int(max(64, min(1024, union.size)))
    built = _build_neighbor_list_celllist_backend_device(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=cutoff,
        target_ids=union,
        candidate_ids=union,
        atom_types=atom_types,
        backend=backend,
        max_neighbors=max_neighbors,
        prefer_marked_dirty=prefer_marked_dirty,
    )
    if built is None:
        return None

    d_r, _d_types_all = _peek_device_state(r=r)
    if d_r is None:
        d_r, _d_types_all = _get_device_state(
            cp=cp,
            r=r,
            atom_types=np.asarray(atom_types, dtype=np.int32),
            update_ids=union,
            prefer_marked_dirty=prefer_marked_dirty,
        )
    elem_idx = potential._types_to_elem_idx(np.asarray(atom_types, dtype=np.int32))
    gid_to_local = np.full((np.asarray(r).shape[0],), -1, dtype=np.int32)
    gid_to_local[union] = np.arange(union.size, dtype=np.int32)
    target_local = gid_to_local[tids]

    d_union = cp.asarray(union)
    d_elem = cp.asarray(elem_idx.astype(np.int32, copy=False))
    d_gid_to_local = cp.asarray(gid_to_local)
    d_rho = cp.zeros((union.size,), dtype=cp.float64)
    d_dF = cp.zeros((union.size,), dtype=cp.float64)
    d_out = cp.zeros((union.size, 3), dtype=cp.float64)

    pot_state = _get_device_potential_state(cp, potential)
    d_grid_r = pot_state.d_grid_r
    d_grid_rho = pot_state.d_grid_rho
    d_density = pot_state.d_density
    d_embed = pot_state.d_embed
    d_phi_flat = pot_state.d_phi_flat

    nelem = int(pot_state.nelem)
    nr = int(pot_state.nr)
    nrho = int(pot_state.nrho)
    cutoff2 = float(cutoff) * float(cutoff)
    threads = 128
    blocks = (int(union.size) + threads - 1) // threads

    _eam_density_rawkernel(cp)(
        (blocks,),
        (threads,),
        (
            d_r.reshape(-1),
            d_union,
            built.d_ids.reshape(-1),
            built.d_counts,
            np.int32(int(union.size)),
            np.int32(int(built.d_ids.shape[1])),
            np.float64(float(box)),
            np.float64(cutoff2),
            d_elem,
            d_grid_r,
            np.int32(nr),
            d_density.reshape(-1),
            d_rho,
        ),
    )

    _eam_embed_deriv_rawkernel(cp)(
        (blocks,),
        (threads,),
        (
            d_union,
            np.int32(int(union.size)),
            d_elem,
            d_grid_rho,
            np.int32(nrho),
            d_embed.reshape(-1),
            d_rho,
            d_dF,
        ),
    )

    _eam_force_rawkernel(cp)(
        (blocks,),
        (threads,),
        (
            d_r.reshape(-1),
            d_union,
            built.d_ids.reshape(-1),
            built.d_counts,
            np.int32(int(union.size)),
            np.int32(int(built.d_ids.shape[1])),
            np.float64(float(box)),
            np.float64(cutoff2),
            d_elem,
            d_gid_to_local,
            d_dF,
            d_grid_r,
            np.int32(nr),
            d_density.reshape(-1),
            d_phi_flat.reshape(-1),
            np.int32(nelem),
            d_out.reshape(-1),
        ),
    )

    return cp.asnumpy(d_out[cp.asarray(target_local, dtype=cp.int32)])


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
    prefer_marked_dirty: bool = False,
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

    if isinstance(potential, (LennardJones, Morse, TablePotential)):
        max_neighbors = int(max(64, min(1024, cids.size)))
        built = _build_neighbor_list_celllist_backend_device(
            r=r,
            box=box,
            cutoff=cutoff,
            rc=cutoff,
            target_ids=tids,
            candidate_ids=cids,
            atom_types=atom_types,
            backend=backend,
            max_neighbors=max_neighbors,
            prefer_marked_dirty=prefer_marked_dirty,
        )
        if built is not None:
            d_r, d_types_all = _peek_device_state(r=r)
            if d_r is None or d_types_all is None:
                d_r, d_types_all = _get_device_state(
                    cp=cp,
                    r=r,
                    atom_types=np.asarray(atom_types, dtype=np.int32),
                    update_ids=np.unique(np.concatenate((tids, cids)).astype(np.int32)),
                    prefer_marked_dirty=prefer_marked_dirty,
                )
            return _forces_from_neighbor_matrix_cp(
                cp=cp,
                r=r,
                box=box,
                cutoff=cutoff,
                potential=potential,
                target_ids=tids,
                neighbor_ids=built.d_ids,
                neighbor_counts=built.d_counts,
                atom_types=atom_types,
                d_r=d_r,
                d_types=d_types_all,
            )
        if isinstance(potential, TablePotential):
            return None

    if isinstance(potential, EAMAlloyPotential):
        return _forces_eam_on_candidate_union_cp(
            cp=cp,
            r=r,
            box=box,
            cutoff=cutoff,
            potential=potential,
            target_ids=tids,
            candidate_ids=cids,
            atom_types=atom_types,
            backend=backend,
            prefer_marked_dirty=prefer_marked_dirty,
        )

    update_ids = np.unique(np.concatenate((tids, cids)).astype(np.int32))
    d_r, d_types_all = _get_device_state(
        cp=cp,
        r=r,
        atom_types=np.asarray(atom_types, dtype=np.int32),
        update_ids=update_ids,
        prefer_marked_dirty=prefer_marked_dirty,
    )
    d_tids = cp.asarray(tids)
    d_cids = cp.asarray(cids)

    rr_t = d_r[d_tids]
    rr_c = d_r[d_cids]
    dr = rr_t[:, None, :] - rr_c[None, :, :]
    dr = dr - float(box) * cp.rint(dr / float(box))
    r2 = cp.sum(dr * dr, axis=2)

    cutoff2 = float(cutoff) * float(cutoff)
    mask = (r2 > 0.0) & (r2 < cutoff2)

    types_i = d_types_all[d_tids]
    types_j = d_types_all[d_cids]

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
    prefer_marked_dirty: bool = False,
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
    built = _build_neighbor_list_celllist_backend_device(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=rc,
        target_ids=tids,
        candidate_ids=cids,
        atom_types=atom_types,
        backend=backend,
        max_neighbors=max_neighbors,
        prefer_marked_dirty=prefer_marked_dirty,
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
            prefer_marked_dirty=prefer_marked_dirty,
        )

    neighbor_ids = built.d_ids
    neighbor_counts = built.d_counts
    cp = backend.xp
    d_r, d_types = _peek_device_state(r=r)
    if d_r is None or d_types is None:
        d_r, d_types = _get_device_state(
            cp=cp,
            r=r,
            atom_types=np.asarray(atom_types, dtype=np.int32),
            update_ids=np.unique(np.concatenate((tids, cids)).astype(np.int32)),
            prefer_marked_dirty=prefer_marked_dirty,
        )
    return _forces_from_neighbor_matrix_cp(
        cp=cp,
        r=r,
        box=box,
        cutoff=cutoff,
        potential=potential,
        target_ids=tids,
        neighbor_ids=neighbor_ids,
        neighbor_counts=neighbor_counts,
        atom_types=atom_types,
        d_r=d_r,
        d_types=d_types,
    )
