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
    if k in ("ml_reference", "ml/reference"):
        return "ml/reference"
    return k


ML_REFERENCE_CONTRACT_VERSION = "ml_ref_v1"


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
        rc: float | None = None,
    ) -> np.ndarray:
        del rc
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


def _as_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    raise ValueError(f"{key} must be a bool")


@dataclass(frozen=True)
class MLReferenceContract:
    version: str
    cutoff_radius: float
    cutoff_smoothing: str
    descriptor_family: str
    descriptor_width: float
    neighbor_mode: str
    requires_full_system_barrier: bool
    inference_family: str
    cpu_reference: bool
    target_local_supported: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "cutoff": {
                "radius": float(self.cutoff_radius),
                "smoothing": self.cutoff_smoothing,
            },
            "descriptor": {
                "family": self.descriptor_family,
                "width": float(self.descriptor_width),
            },
            "neighbor": {
                "mode": self.neighbor_mode,
                "requires_full_system_barrier": bool(self.requires_full_system_barrier),
            },
            "inference": {
                "family": self.inference_family,
                "cpu_reference": bool(self.cpu_reference),
                "target_local_supported": bool(self.target_local_supported),
            },
        }


def parse_ml_reference_params(
    params: dict[str, Any],
    *,
    max_atom_type: int | None = None,
) -> tuple[MLReferenceContract, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(params, dict):
        raise ValueError("ml/reference params must be a mapping")
    contract_raw = params.get("contract")
    if not isinstance(contract_raw, dict):
        raise ValueError("ml/reference requires params.contract mapping")
    version = str(contract_raw.get("version", "")).strip()
    if version != ML_REFERENCE_CONTRACT_VERSION:
        raise ValueError(f"ml/reference contract.version must be '{ML_REFERENCE_CONTRACT_VERSION}'")

    cutoff_raw = contract_raw.get("cutoff")
    if not isinstance(cutoff_raw, dict):
        raise ValueError("ml/reference contract.cutoff must be a mapping")
    cutoff_radius = float(cutoff_raw.get("radius", 0.0))
    cutoff_smoothing = str(cutoff_raw.get("smoothing", "")).strip().lower()
    if cutoff_radius <= 0.0:
        raise ValueError("ml/reference contract.cutoff.radius must be positive")
    if cutoff_smoothing != "cosine":
        raise ValueError("ml/reference contract.cutoff.smoothing must be 'cosine'")

    descriptor_raw = contract_raw.get("descriptor")
    if not isinstance(descriptor_raw, dict):
        raise ValueError("ml/reference contract.descriptor must be a mapping")
    descriptor_family = str(descriptor_raw.get("family", "")).strip().lower()
    descriptor_width = float(descriptor_raw.get("width", 0.0))
    if descriptor_family != "radial_density":
        raise ValueError("ml/reference contract.descriptor.family must be 'radial_density'")
    if descriptor_width <= 0.0:
        raise ValueError("ml/reference contract.descriptor.width must be positive")

    neighbor_raw = contract_raw.get("neighbor")
    if not isinstance(neighbor_raw, dict):
        raise ValueError("ml/reference contract.neighbor must be a mapping")
    neighbor_mode = str(neighbor_raw.get("mode", "")).strip().lower()
    requires_full_system_barrier = _as_bool(
        neighbor_raw.get("requires_full_system_barrier", False),
        key="ml/reference contract.neighbor.requires_full_system_barrier",
    )
    if neighbor_mode != "candidate_local":
        raise ValueError("ml/reference contract.neighbor.mode must be 'candidate_local'")
    if requires_full_system_barrier:
        raise ValueError(
            "ml/reference contract.neighbor.requires_full_system_barrier must be false"
        )

    inference_raw = contract_raw.get("inference")
    if not isinstance(inference_raw, dict):
        raise ValueError("ml/reference contract.inference must be a mapping")
    inference_family = str(inference_raw.get("family", "")).strip().lower()
    cpu_reference = _as_bool(
        inference_raw.get("cpu_reference", False),
        key="ml/reference contract.inference.cpu_reference",
    )
    target_local_supported = _as_bool(
        inference_raw.get("target_local_supported", True),
        key="ml/reference contract.inference.target_local_supported",
    )
    if inference_family != "quadratic_density":
        raise ValueError("ml/reference contract.inference.family must be 'quadratic_density'")
    if not cpu_reference:
        raise ValueError("ml/reference contract.inference.cpu_reference must be true")
    if not target_local_supported:
        raise ValueError("ml/reference contract.inference.target_local_supported must be true")

    species_raw = params.get("species")
    if not isinstance(species_raw, (list, tuple)) or not species_raw:
        raise ValueError("ml/reference requires params.species as non-empty list")
    bias: list[float] = []
    quadratic: list[float] = []
    neighbor_weight: list[float] = []
    for idx, item in enumerate(species_raw):
        if not isinstance(item, dict):
            raise ValueError(f"ml/reference params.species[{idx}] must be a mapping")
        try:
            bias_i = float(item.get("bias", 0.0))
            quadratic_i = float(item.get("quadratic", 0.0))
            neighbor_weight_i = float(item.get("neighbor_weight", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ml/reference params.species[{idx}] numeric fields are invalid"
            ) from exc
        if neighbor_weight_i <= 0.0:
            raise ValueError(f"ml/reference params.species[{idx}].neighbor_weight must be positive")
        bias.append(bias_i)
        quadratic.append(quadratic_i)
        neighbor_weight.append(neighbor_weight_i)

    if max_atom_type is not None and int(max_atom_type) > len(species_raw):
        raise ValueError(
            "ml/reference species list length "
            f"({len(species_raw)}) is smaller than max atom type ({int(max_atom_type)})"
        )

    contract = MLReferenceContract(
        version=version,
        cutoff_radius=float(cutoff_radius),
        cutoff_smoothing=cutoff_smoothing,
        descriptor_family=descriptor_family,
        descriptor_width=float(descriptor_width),
        neighbor_mode=neighbor_mode,
        requires_full_system_barrier=bool(requires_full_system_barrier),
        inference_family=inference_family,
        cpu_reference=bool(cpu_reference),
        target_local_supported=bool(target_local_supported),
    )
    return (
        contract,
        np.asarray(bias, dtype=float),
        np.asarray(quadratic, dtype=float),
        np.asarray(neighbor_weight, dtype=float),
    )


@dataclass(frozen=True)
class MLReferencePotential:
    contract: MLReferenceContract
    center_bias: np.ndarray
    center_quadratic: np.ndarray
    neighbor_weight: np.ndarray

    def _resolve_cutoff(self, cutoff: float) -> float:
        runtime_cutoff = float(cutoff)
        contract_cutoff = float(self.contract.cutoff_radius)
        if runtime_cutoff + NUMERICAL_ZERO < contract_cutoff:
            raise ValueError(
                "runtime cutoff is smaller than ml/reference contract cutoff: "
                f"runtime={runtime_cutoff} contract={contract_cutoff}"
            )
        return contract_cutoff

    def _types_to_species_idx(self, atom_types: np.ndarray) -> np.ndarray:
        t = np.asarray(atom_types, dtype=np.int32)
        if t.ndim != 1:
            raise ValueError("atom_types must be 1D for ml/reference")
        idx = t - 1
        if np.any(idx < 0) or np.any(idx >= self.center_bias.shape[0]):
            raise ValueError(
                "ml/reference type->species mapping out of range; "
                f"seen types={sorted({int(x) for x in t.tolist()})}"
            )
        return idx.astype(np.int32)

    def _feature_and_grad(self, rij: float, cutoff: float) -> tuple[float, float]:
        if rij <= 0.0 or rij >= float(cutoff):
            return 0.0, 0.0
        width = float(self.contract.descriptor_width)
        base = float(np.exp(-0.5 * (rij / width) ** 2))
        dbase = float(base * (-rij / (width * width)))
        switch = float(0.5 * (np.cos(np.pi * rij / cutoff) + 1.0))
        dswitch = float(-0.5 * np.pi / cutoff * np.sin(np.pi * rij / cutoff))
        feature = base * switch
        dfeature = dbase * switch + base * dswitch
        return float(feature), float(dfeature)

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
        list[tuple[int, int, np.ndarray, float, float, float]],
    ]:
        cand = np.asarray(candidate_ids, dtype=np.int32)
        if cand.ndim != 1:
            raise ValueError("candidate_ids must be 1D")
        if cand.size == 0:
            return cand, np.empty((0,), np.int32), np.empty((0,), dtype=float), []
        cand = np.unique(cand)
        species_idx = self._types_to_species_idx(atom_types)[cand]
        rho_acc = np.zeros((cand.size,), dtype=float)
        pairs: list[tuple[int, int, np.ndarray, float, float, float]] = []
        rc = self._resolve_cutoff(cutoff)
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
                if rij >= rc:
                    continue
                base_feat, base_grad = self._feature_and_grad(rij, rc)
                si = int(species_idx[li])
                sj = int(species_idx[lj])
                feat_i = float(self.neighbor_weight[sj]) * base_feat
                feat_j = float(self.neighbor_weight[si]) * base_feat
                grad_i = float(self.neighbor_weight[sj]) * base_grad
                grad_j = float(self.neighbor_weight[si]) * base_grad
                rho_acc[li] += feat_i
                rho_acc[lj] += feat_j
                pairs.append((li, lj, dr, rij, grad_i, grad_j))
        return cand, species_idx, rho_acc, pairs

    def _inference_state(
        self,
        rho_acc: np.ndarray,
        species_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        bias = self.center_bias[species_idx]
        quadratic = self.center_quadratic[species_idx]
        atom_energy = bias * rho_acc + 0.5 * quadratic * rho_acc * rho_acc
        dE_drho = bias + quadratic * rho_acc
        return atom_energy.astype(float), dE_drho.astype(float)

    def forces_energy_virial(
        self,
        r: np.ndarray,
        box: float,
        cutoff: float,
        atom_types: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        n = int(r.shape[0])
        cand, species_idx, rho_acc, pairs = self._subset_state(
            r, box, cutoff, atom_types, np.arange(n, dtype=np.int32)
        )
        forces = np.zeros((n, 3), dtype=float)
        atom_energy, dE_drho = self._inference_state(rho_acc, species_idx)
        virial = 0.0
        for li, lj, dr, rij, grad_i, grad_j in pairs:
            dEdr = float(dE_drho[li]) * grad_i + float(dE_drho[lj]) * grad_j
            fij = (-dEdr / (rij + NUMERICAL_ZERO)) * dr
            i = int(cand[li])
            j = int(cand[lj])
            forces[i] += fij
            forces[j] -= fij
            virial += float(np.dot(dr, fij))
        return forces, float(atom_energy.sum()), float(virial)

    def energy_virial(
        self,
        r: np.ndarray,
        box: float,
        cutoff: float,
        atom_types: np.ndarray,
    ) -> tuple[float, float]:
        _forces, pe, virial = self.forces_energy_virial(r, box, cutoff, atom_types)
        return pe, virial

    def forces_on_targets(
        self,
        r: np.ndarray,
        box: float,
        cutoff: float,
        atom_types: np.ndarray,
        target_ids: np.ndarray,
        candidate_ids: np.ndarray,
        rc: float | None = None,
    ) -> np.ndarray:
        del rc
        tgt = np.asarray(target_ids, dtype=np.int32)
        cand_in = np.asarray(candidate_ids, dtype=np.int32)
        if tgt.ndim != 1 or cand_in.ndim != 1:
            raise ValueError("target_ids/candidate_ids must be 1D")
        if tgt.size == 0:
            return np.zeros((0, 3), dtype=float)
        cand_union = np.unique(np.concatenate([cand_in, tgt]).astype(np.int32))
        cand, species_idx, rho_acc, pairs = self._subset_state(
            r, box, cutoff, atom_types, cand_union
        )
        if cand.size == 0:
            return np.zeros((tgt.size, 3), dtype=float)
        _atom_energy, dE_drho = self._inference_state(rho_acc, species_idx)
        out = np.zeros((tgt.size, 3), dtype=float)
        pos_tgt = {int(gid): int(i) for i, gid in enumerate(tgt.tolist())}
        for li, lj, dr, rij, grad_i, grad_j in pairs:
            dEdr = float(dE_drho[li]) * grad_i + float(dE_drho[lj]) * grad_j
            fij = (-dEdr / (rij + NUMERICAL_ZERO)) * dr
            gi = int(cand[li])
            gj = int(cand[lj])
            ti = pos_tgt.get(gi, -1)
            tj = pos_tgt.get(gj, -1)
            if ti >= 0:
                out[ti] += fij
            if tj >= 0:
                out[tj] -= fij
        return out


def describe_ml_reference_contract(potential: object) -> dict[str, object] | None:
    if not isinstance(potential, MLReferencePotential):
        return None
    return potential.contract.as_dict()


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
    if kind == "ml/reference":
        contract, bias, quadratic, neighbor_weight = parse_ml_reference_params(params)
        return MLReferencePotential(
            contract=contract,
            center_bias=bias,
            center_quadratic=quadratic,
            neighbor_weight=neighbor_weight,
        )
    raise ValueError(kind)
