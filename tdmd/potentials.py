from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import NUMERICAL_ZERO


def canonical_potential_kind(kind: str) -> str:
    k = str(kind).strip().lower()
    if k in ("eam_alloy", "eam/alloy"):
        return "eam/alloy"
    return k


_PAIR_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "lj": ("epsilon", "sigma"),
    "morse": ("D_e", "a", "r0"),
}


def _parse_pair_key(key: str) -> tuple[int, int]:
    txt = str(key).strip()
    for sep in ("-", ",", ":"):
        if sep in txt:
            parts = [p.strip() for p in txt.split(sep)]
            if len(parts) != 2:
                break
            try:
                i = int(parts[0])
                j = int(parts[1])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid pair_coeffs key '{key}'") from exc
            if i <= 0 or j <= 0:
                raise ValueError(f"pair_coeffs key '{key}' must use positive atom types")
            return (min(i, j), max(i, j))
    raise ValueError(f"invalid pair_coeffs key '{key}'; expected 'i-j' with positive ints")


def parse_pair_coeffs(kind: str, params: dict[str, Any]) -> dict[tuple[int, int], dict[str, float]]:
    kind = canonical_potential_kind(kind)
    required = _PAIR_PARAM_NAMES.get(kind, ())
    raw = params.get("pair_coeffs")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("potential.params.pair_coeffs must be a mapping")
    out: dict[tuple[int, int], dict[str, float]] = {}
    for k, v in raw.items():
        pair = _parse_pair_key(str(k))
        if not isinstance(v, dict):
            raise ValueError(f"pair_coeffs[{k!r}] must be a mapping")
        missing = [nm for nm in required if nm not in v]
        unknown = sorted(set(v.keys()) - set(required))
        if missing:
            raise ValueError(f"pair_coeffs[{k!r}] missing required params: {missing}")
        if unknown:
            raise ValueError(
                f"pair_coeffs[{k!r}] has unsupported params: {unknown}; allowed: {list(required)}"
            )
        vals = {nm: float(v[nm]) for nm in required}
        prev = out.get(pair)
        if prev is not None and any(float(prev[nm]) != float(vals[nm]) for nm in required):
            raise ValueError(f"duplicate pair_coeffs for {pair} with conflicting values")
        out[pair] = vals
    return out


def ensure_pair_coeffs_complete(
    pair_coeffs: dict[tuple[int, int], dict[str, float]],
    atom_types: np.ndarray,
) -> None:
    if not pair_coeffs:
        return
    types = sorted({int(t) for t in np.asarray(atom_types, dtype=np.int32).tolist()})
    missing: list[str] = []
    for i, ti in enumerate(types):
        for tj in types[i:]:
            if (ti, tj) not in pair_coeffs:
                missing.append(f"{ti}-{tj}")
    if missing:
        raise ValueError(
            "potential.params.pair_coeffs does not cover all type pairs in task: "
            f"missing {missing}"
        )


def _interp_with_grad(x: np.ndarray, y: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear interpolation and dy/dx."""
    qq = np.asarray(q, dtype=float)
    val = np.interp(qq, x, y, left=0.0, right=0.0)
    grad = np.zeros_like(qq, dtype=float)
    mask = (qq >= x[0]) & (qq <= x[-1])
    if np.any(mask):
        qm = qq[mask]
        idx = np.searchsorted(x, qm, side="right") - 1
        idx = np.clip(idx, 0, x.size - 2)
        dx = x[idx + 1] - x[idx]
        dy = y[idx + 1] - y[idx]
        grad[mask] = dy / dx
    return val, grad


def _read_setfl(
    path: str,
) -> tuple[list[str], float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Parse DYNAMO setfl/eam.alloy into (elements, drho, dr, cutoff, F, rho, phi)."""
    if not os.path.isfile(path):
        raise ValueError(f"EAM file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 6:
        raise ValueError(f"invalid setfl file (too short): {path}")

    hdr = lines[3].split()
    if len(hdr) < 2:
        raise ValueError(f"invalid setfl element header in {path}")
    try:
        nelem = int(hdr[0])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid nelem in setfl header: {hdr}") from exc
    elems = [str(x) for x in hdr[1 : 1 + nelem]]
    if len(elems) != nelem:
        raise ValueError(f"setfl header has nelem={nelem} but got elements={elems}")

    grid = lines[4].split()
    if len(grid) < 5:
        raise ValueError(f"invalid setfl grid line in {path}")
    try:
        nrho = int(grid[0])
        drho = float(grid[1])
        nr = int(grid[2])
        dr = float(grid[3])
        cutoff = float(grid[4])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid setfl grid values: {grid}") from exc
    if nrho < 2 or nr < 2 or drho <= 0.0 or dr <= 0.0 or cutoff <= 0.0:
        raise ValueError(f"invalid setfl grid parameters in {path}")

    line_idx = 5
    token_buf: list[str] = []

    def _next_data_line() -> str:
        nonlocal line_idx
        while line_idx < len(lines):
            txt = lines[line_idx].strip()
            line_idx += 1
            if not txt or txt.startswith("#"):
                continue
            return txt
        raise ValueError(f"unexpected end of file while parsing setfl: {path}")

    def _take_floats(n: int) -> np.ndarray:
        vals: list[float] = []
        while len(vals) < n:
            if not token_buf:
                token_buf.extend(_next_data_line().split())
            while token_buf and len(vals) < n:
                tok = token_buf.pop(0)
                try:
                    vals.append(float(tok))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"invalid numeric token in setfl {path}: {tok}") from exc
        return np.asarray(vals, dtype=float)

    F = np.zeros((nelem, nrho), dtype=float)
    rho = np.zeros((nelem, nr), dtype=float)
    for i in range(nelem):
        token_buf.clear()
        meta = _next_data_line().split()
        if len(meta) < 4:
            raise ValueError(f"invalid setfl element metadata line in {path}: {' '.join(meta)}")
        try:
            float(meta[0])
            float(meta[1])
            float(meta[2])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"invalid setfl element metadata numeric fields in {path}: {meta}"
            ) from exc
        F[i, :] = _take_floats(nrho)
        rho[i, :] = _take_floats(nr)

    phi = np.zeros((nelem, nelem, nr), dtype=float)
    r_grid = np.arange(nr, dtype=float) * dr
    for i in range(nelem):
        for j in range(i + 1):
            z = _take_floats(nr)  # z(r) = r * phi(r)
            p = np.zeros_like(z)
            if nr > 1:
                p[1:] = z[1:] / np.maximum(r_grid[1:], NUMERICAL_ZERO)
                p[0] = p[1]
            else:
                p[0] = 0.0
            phi[i, j, :] = p
            phi[j, i, :] = p
    return elems, float(drho), float(dr), float(cutoff), F, rho, phi


def _read_table_section(path: str, keyword: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.isfile(path):
        raise ValueError(f"table potential file not found: {path}")
    key = str(keyword).strip()
    if not key:
        raise ValueError("table potential keyword must be non-empty")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start = -1
    for i, raw in enumerate(lines):
        txt = raw.strip()
        if not txt or txt.startswith("#"):
            continue
        if txt.split()[0] == key:
            start = i
            break
    if start < 0:
        raise ValueError(f"table keyword '{key}' not found in file: {path}")

    i = start + 1
    n_expected: int | None = None
    while i < len(lines):
        txt = lines[i].strip()
        i += 1
        if not txt or txt.startswith("#"):
            continue
        toks = txt.split()
        if toks and toks[0].upper() == "N":
            if len(toks) < 2:
                raise ValueError(f"invalid table header after keyword '{key}' in {path}: {txt}")
            try:
                n_expected = int(toks[1])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid N value in table header: {txt}") from exc
        else:
            i -= 1
        break

    rows: list[tuple[float, float, float]] = []
    while i < len(lines):
        raw = lines[i]
        i += 1
        txt = raw.strip()
        if not txt or txt.startswith("#"):
            continue
        toks = txt.split()
        if len(toks) < 4:
            if rows:
                break
            raise ValueError(f"invalid table row in {path}: {txt}")
        try:
            _idx = int(toks[0])
            rr = float(toks[1])
            uu = float(toks[2])
            ff = float(toks[3])
        except (ValueError, TypeError, IndexError):
            if rows:
                break
            raise ValueError(f"invalid table row in {path}: {txt}")
        rows.append((rr, uu, ff))
        if n_expected is not None and len(rows) >= n_expected:
            break

    if not rows:
        raise ValueError(f"no data rows found for table keyword '{key}' in {path}")
    if n_expected is not None and len(rows) != n_expected:
        raise ValueError(
            f"table keyword '{key}' in {path}: expected {n_expected} rows, got {len(rows)}"
        )
    r = np.asarray([x[0] for x in rows], dtype=float)
    u = np.asarray([x[1] for x in rows], dtype=float)
    f = np.asarray([x[2] for x in rows], dtype=float)
    if np.any(r <= 0.0):
        raise ValueError(f"table keyword '{key}' in {path} has non-positive r values")
    if np.any(np.diff(r) <= 0.0):
        raise ValueError(f"table keyword '{key}' in {path} must have strictly increasing r grid")
    return r, u, f


@dataclass(frozen=True)
class LennardJones:
    epsilon: float = 1.0
    sigma: float = 1.0
    pair_coeffs: dict[tuple[int, int], dict[str, float]] | None = None

    def _coeff_arrays(
        self,
        n: int,
        type_i: np.ndarray | None,
        type_j: np.ndarray | None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        if type_i is None or type_j is None or not self.pair_coeffs:
            return float(self.epsilon), float(self.sigma)
        ti = np.asarray(type_i, dtype=np.int32)
        tj = np.asarray(type_j, dtype=np.int32)
        if ti.shape != tj.shape or ti.shape != (n,):
            raise ValueError("type_i/type_j must have shape (n_pairs,)")
        eps = np.full((n,), float(self.epsilon), dtype=float)
        sig = np.full((n,), float(self.sigma), dtype=float)
        for (a, b), prm in self.pair_coeffs.items():
            mask = ((ti == int(a)) & (tj == int(b))) | ((ti == int(b)) & (tj == int(a)))
            if np.any(mask):
                eps[mask] = float(prm["epsilon"])
                sig[mask] = float(prm["sigma"])
        return eps, sig

    def pair(
        self,
        r2: np.ndarray,
        cutoff2: float,
        type_i: np.ndarray | None = None,
        type_j: np.ndarray | None = None,
    ):
        mask = (r2 > 0.0) & (r2 < cutoff2)
        inv_r2 = np.zeros_like(r2, dtype=float)
        np.divide(1.0, r2, out=inv_r2, where=mask)
        eps, sig = self._coeff_arrays(int(r2.shape[0]), type_i, type_j)
        sig2 = sig * sig
        sr2 = sig2 * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        U = 4.0 * eps * (sr12 - sr6)
        coef = 24.0 * eps * (2.0 * sr12 - sr6) * inv_r2
        return coef, U


@dataclass(frozen=True)
class Morse:
    D_e: float = 0.29614
    a: float = 1.11892
    r0: float = 3.29692
    pair_coeffs: dict[tuple[int, int], dict[str, float]] | None = None

    def _coeff_arrays(
        self,
        n: int,
        type_i: np.ndarray | None,
        type_j: np.ndarray | None,
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        if type_i is None or type_j is None or not self.pair_coeffs:
            return float(self.D_e), float(self.a), float(self.r0)
        ti = np.asarray(type_i, dtype=np.int32)
        tj = np.asarray(type_j, dtype=np.int32)
        if ti.shape != tj.shape or ti.shape != (n,):
            raise ValueError("type_i/type_j must have shape (n_pairs,)")
        d = np.full((n,), float(self.D_e), dtype=float)
        aa = np.full((n,), float(self.a), dtype=float)
        r0 = np.full((n,), float(self.r0), dtype=float)
        for (x, y), prm in self.pair_coeffs.items():
            mask = ((ti == int(x)) & (tj == int(y))) | ((ti == int(y)) & (tj == int(x)))
            if np.any(mask):
                d[mask] = float(prm["D_e"])
                aa[mask] = float(prm["a"])
                r0[mask] = float(prm["r0"])
        return d, aa, r0

    def pair(
        self,
        r2: np.ndarray,
        cutoff2: float,
        type_i: np.ndarray | None = None,
        type_j: np.ndarray | None = None,
    ):
        r = np.sqrt(r2 + NUMERICAL_ZERO)
        mask = (r2 > 0.0) & (r2 < cutoff2)
        d, aa, r0 = self._coeff_arrays(int(r2.shape[0]), type_i, type_j)
        x = r - r0
        exp1 = np.exp(-aa * x)
        U = d * (1.0 - exp1) ** 2 - d
        dUdr = 2.0 * d * aa * (1.0 - exp1) * exp1
        coef = np.where(mask, dUdr / (r + NUMERICAL_ZERO), 0.0)
        U = np.where(mask, U, 0.0)
        return coef, U


@dataclass(frozen=True)
class TablePotential:
    r_grid: np.ndarray
    u_grid: np.ndarray
    f_grid: np.ndarray

    def pair(
        self,
        r2: np.ndarray,
        cutoff2: float,
        type_i: np.ndarray | None = None,
        type_j: np.ndarray | None = None,
    ):
        del type_i, type_j
        r = np.sqrt(r2 + NUMERICAL_ZERO)
        mask = (r2 > 0.0) & (r2 < cutoff2)
        U = np.zeros_like(r2, dtype=float)
        coef = np.zeros_like(r2, dtype=float)
        if np.any(mask):
            rr = r[mask]
            U[mask] = np.interp(rr, self.r_grid, self.u_grid, left=0.0, right=0.0)
            force_mag = np.interp(rr, self.r_grid, self.f_grid, left=0.0, right=0.0)
            coef[mask] = force_mag / (rr + NUMERICAL_ZERO)
        return coef, U


@dataclass(frozen=True)
class EAMAlloyPotential:
    elements: tuple[str, ...]
    grid_rho: np.ndarray
    grid_r: np.ndarray
    embed_table: np.ndarray  # [nelem, nrho]
    density_table: np.ndarray  # [nelem, nr]
    phi_table: np.ndarray  # [nelem, nelem, nr]

    def _types_to_elem_idx(self, atom_types: np.ndarray) -> np.ndarray:
        t = np.asarray(atom_types, dtype=np.int32)
        if t.ndim != 1:
            raise ValueError("atom_types must be 1D for EAM")
        idx = t - 1
        if np.any(idx < 0) or np.any(idx >= len(self.elements)):
            raise ValueError(
                f"EAM type->element mapping out of range for elements={list(self.elements)}; "
                f"seen types={sorted({int(x) for x in t.tolist()})}"
            )
        return idx.astype(np.int32)

    def _subset_state(
        self,
        r: np.ndarray,
        box: float,
        cutoff: float,
        atom_types: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[tuple[int, int, np.ndarray, float, int, int]],
    ]:
        cand = np.asarray(candidate_ids, dtype=np.int32)
        if cand.ndim != 1:
            raise ValueError("candidate_ids must be 1D")
        if cand.size == 0:
            return (
                cand,
                np.empty((0,), np.int32),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
                [],
            )
        cand = np.unique(cand)
        elem_all = self._types_to_elem_idx(atom_types)
        elem = elem_all[cand]
        rho_acc = np.zeros((cand.size,), dtype=float)
        pairs: list[tuple[int, int, np.ndarray, float, int, int]] = []
        for li in range(cand.size):
            i = int(cand[li])
            for lj in range(li + 1, cand.size):
                j = int(cand[lj])
                dr = r[i] - r[j]
                dr = dr - float(box) * np.round(dr / float(box))
                rij2 = float(np.dot(dr, dr))
                if rij2 <= 0.0:
                    continue
                rij = float(np.sqrt(rij2))
                if rij >= float(cutoff):
                    continue
                ai = int(elem[li])
                aj = int(elem[lj])
                rhoi, _ = _interp_with_grad(
                    self.grid_r, self.density_table[aj], np.array([rij], dtype=float)
                )
                rhoj, _ = _interp_with_grad(
                    self.grid_r, self.density_table[ai], np.array([rij], dtype=float)
                )
                rho_acc[li] += float(rhoi[0])
                rho_acc[lj] += float(rhoj[0])
                pairs.append((li, lj, dr, rij, ai, aj))

        dF = np.zeros((cand.size,), dtype=float)
        for li in range(cand.size):
            ai = int(elem[li])
            _Fi, dFi = _interp_with_grad(
                self.grid_rho, self.embed_table[ai], np.array([rho_acc[li]], dtype=float)
            )
            dF[li] = float(dFi[0])
        return cand, elem, rho_acc, dF, pairs

    def forces_energy_virial(
        self,
        r: np.ndarray,
        box: float,
        cutoff: float,
        atom_types: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        n = int(r.shape[0])
        candidate = np.arange(n, dtype=np.int32)
        cand, elem_idx, rho_acc, dF, pairs = self._subset_state(
            r, box, cutoff, atom_types, candidate
        )
        forces = np.zeros((n, 3), dtype=float)

        emb = np.zeros((cand.size,), dtype=float)
        for li in range(cand.size):
            ai = int(elem_idx[li])
            Fi, _dFi = _interp_with_grad(
                self.grid_rho, self.embed_table[ai], np.array([rho_acc[li]], dtype=float)
            )
            emb[li] = float(Fi[0])

        pair_energy = 0.0
        virial = 0.0
        for li, lj, dr, rij, ai, aj in pairs:
            phi, dphi = _interp_with_grad(
                self.grid_r, self.phi_table[ai, aj], np.array([rij], dtype=float)
            )
            rhoi, drhoi = _interp_with_grad(
                self.grid_r, self.density_table[aj], np.array([rij], dtype=float)
            )
            rhoj, drhoj = _interp_with_grad(
                self.grid_r, self.density_table[ai], np.array([rij], dtype=float)
            )
            del rhoi, rhoj
            dEdr = float(dphi[0]) + dF[li] * float(drhoi[0]) + dF[lj] * float(drhoj[0])
            fmag = -dEdr
            fij = (fmag / (rij + NUMERICAL_ZERO)) * dr
            i = int(cand[li])
            j = int(cand[lj])
            forces[i] += fij
            forces[j] -= fij
            pair_energy += float(phi[0])
            virial += float(np.dot(dr, fij))

        return forces, float(pair_energy + emb.sum()), float(virial)

    def energy_virial(
        self,
        r: np.ndarray,
        box: float,
        cutoff: float,
        atom_types: np.ndarray,
    ) -> tuple[float, float]:
        _f, pe, vir = self.forces_energy_virial(r, box, cutoff, atom_types)
        return pe, vir

    def forces_on_targets(
        self,
        r: np.ndarray,
        box: float,
        cutoff: float,
        atom_types: np.ndarray,
        target_ids: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        tgt = np.asarray(target_ids, dtype=np.int32)
        cand_in = np.asarray(candidate_ids, dtype=np.int32)
        if tgt.ndim != 1 or cand_in.ndim != 1:
            raise ValueError("target_ids/candidate_ids must be 1D")
        if tgt.size == 0:
            return np.zeros((0, 3), dtype=float)
        cand_union = np.unique(np.concatenate([cand_in, tgt]).astype(np.int32))
        cand, _elem, _rho, dF, pairs = self._subset_state(r, box, cutoff, atom_types, cand_union)
        if cand.size == 0:
            return np.zeros((tgt.size, 3), dtype=float)

        out = np.zeros((tgt.size, 3), dtype=float)
        pos_tgt = {int(gid): int(i) for i, gid in enumerate(tgt.tolist())}

        for li, lj, dr, rij, ai, aj in pairs:
            phi, dphi = _interp_with_grad(
                self.grid_r, self.phi_table[ai, aj], np.array([rij], dtype=float)
            )
            rhoi, drhoi = _interp_with_grad(
                self.grid_r, self.density_table[aj], np.array([rij], dtype=float)
            )
            rhoj, drhoj = _interp_with_grad(
                self.grid_r, self.density_table[ai], np.array([rij], dtype=float)
            )
            del phi, rhoi, rhoj
            dEdr = float(dphi[0]) + dF[li] * float(drhoi[0]) + dF[lj] * float(drhoj[0])
            fij = (-dEdr / (rij + NUMERICAL_ZERO)) * dr  # force on i from j
            gi = int(cand[li])
            gj = int(cand[lj])
            ti = pos_tgt.get(gi, -1)
            tj = pos_tgt.get(gj, -1)
            if ti >= 0:
                out[ti] += fij
            if tj >= 0:
                out[tj] -= fij
        return out


def make_potential(kind: str, params: dict):
    kind = canonical_potential_kind(kind)
    pair_coeffs = parse_pair_coeffs(kind, params)
    if kind == "lj":
        return LennardJones(
            epsilon=float(params.get("epsilon", 1.0)),
            sigma=float(params.get("sigma", 1.0)),
            pair_coeffs=pair_coeffs,
        )
    if kind == "morse":
        return Morse(
            D_e=float(params.get("D_e", 0.29614)),
            a=float(params.get("a", 1.11892)),
            r0=float(params.get("r0", 3.29692)),
            pair_coeffs=pair_coeffs,
        )
    if kind == "table":
        table_file = str(params.get("file", "")).strip()
        keyword = str(params.get("keyword", "")).strip()
        if not table_file or not keyword:
            raise ValueError("table potential requires params.file and params.keyword")
        r, u, f = _read_table_section(table_file, keyword)
        return TablePotential(r_grid=r, u_grid=u, f_grid=f)
    if kind == "eam/alloy":
        setfl_file = str(params.get("file", "")).strip()
        if not setfl_file:
            raise ValueError("eam/alloy potential requires params.file")
        elems_raw = params.get("elements")
        if not isinstance(elems_raw, (list, tuple)) or not elems_raw:
            raise ValueError("eam/alloy potential requires params.elements as non-empty list")
        elems_cfg = [str(x).strip() for x in elems_raw]
        if any(not x for x in elems_cfg):
            raise ValueError("eam/alloy params.elements must contain non-empty names")
        elems_file, drho, dr, _cutoff, F, rho, phi = _read_setfl(setfl_file)
        idx_map = {name: i for i, name in enumerate(elems_file)}
        try:
            sel = [idx_map[name] for name in elems_cfg]
        except KeyError as exc:
            raise ValueError(
                f"eam/alloy elements {elems_cfg} not found in file elements {elems_file}"
            ) from exc
        nrho = F.shape[1]
        nr = rho.shape[1]
        grid_rho = np.arange(nrho, dtype=float) * float(drho)
        grid_r = np.arange(nr, dtype=float) * float(dr)
        nsel = len(sel)
        F_sel = np.zeros((nsel, nrho), dtype=float)
        rho_sel = np.zeros((nsel, nr), dtype=float)
        phi_sel = np.zeros((nsel, nsel, nr), dtype=float)
        for a, ia in enumerate(sel):
            F_sel[a] = F[ia]
            rho_sel[a] = rho[ia]
            for b, ib in enumerate(sel):
                phi_sel[a, b] = phi[ia, ib]
        return EAMAlloyPotential(
            elements=tuple(elems_cfg),
            grid_rho=grid_rho,
            grid_r=grid_r,
            embed_table=F_sel,
            density_table=rho_sel,
            phi_table=phi_sel,
        )
    raise ValueError(kind)
