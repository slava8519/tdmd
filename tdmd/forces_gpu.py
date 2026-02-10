from __future__ import annotations

import numpy as np

from .backend import ComputeBackend, resolve_backend
from .celllist import build_cell_list
from .constants import NUMERICAL_ZERO
from .potentials import EAMAlloyPotential, LennardJones, Morse, TablePotential


def supports_pair_gpu(potential) -> bool:
    return isinstance(potential, (LennardJones, Morse, TablePotential, EAMAlloyPotential))


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

        rho_acc = cp.zeros((n,), dtype=cp.float64)
        drho_col = cp.zeros((n, n), dtype=cp.float64)
        drho_row = cp.zeros((n, n), dtype=cp.float64)
        for ej in range(nelem):
            rho_e, drho_e = _interp_with_grad_cp(cp, grid_r, density[ej], rc)
            mask_col = (elem[None, :] == int(ej)) & m
            mask_row = (elem[:, None] == int(ej)) & m
            rho_acc = rho_acc + cp.sum(cp.where(mask_col, rho_e, 0.0), axis=1)
            drho_col = cp.where(mask_col, drho_e, drho_col)
            drho_row = cp.where(mask_row, drho_e, drho_row)

        dF = cp.zeros((n,), dtype=cp.float64)
        for ei in range(nelem):
            _F, dF_e = _interp_with_grad_cp(cp, grid_rho, embed[ei], rho_acc)
            dF = cp.where(elem == int(ei), dF_e, dF)

        dphi = cp.zeros((n, n), dtype=cp.float64)
        for ei in range(nelem):
            for ej in range(nelem):
                _phi_ij, dphi_ij = _interp_with_grad_cp(cp, grid_r, phi[ei, ej], rc)
                mask_ij = (elem[:, None] == int(ei)) & (elem[None, :] == int(ej)) & m
                dphi = cp.where(mask_ij, dphi_ij, dphi)

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

    cl = build_cell_list(r, cids, float(box), rc=float(rc))
    out = np.zeros((tids.size, 3), dtype=np.float64)

    for ti, i in enumerate(tids.tolist()):
        ci = tuple(cl.idx[int(i)])
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cj = ((ci[0] + dx) % cl.ncell, (ci[1] + dy) % cl.ncell, (ci[2] + dz) % cl.ncell)
                    arr = cl.cell_atoms.get(cj)
                    if arr is not None and arr.size:
                        neigh.append(arr)
        if not neigh:
            continue
        js = np.concatenate(neigh).astype(np.int32)
        f1 = forces_on_targets_pair_backend(
            r=r,
            box=box,
            cutoff=cutoff,
            potential=potential,
            target_ids=np.array([int(i)], dtype=np.int32),
            candidate_ids=js,
            atom_types=atom_types,
            backend=backend,
        )
        if f1 is not None and f1.size:
            out[ti] = f1[0]
    return out
