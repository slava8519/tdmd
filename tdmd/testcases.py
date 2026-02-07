from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Case:
    name: str
    kind: str  # gas, fcc, bcc
    n_cells: int
    a: float
    temperature: float
    box: float | None = None  # if None computed
    seed: int = 123

def fcc_positions(n_cells: int, a: float) -> np.ndarray:
    # 4 atoms per cell
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ], dtype=float)
    pts = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cell = np.array([i,j,k], dtype=float)
                pts.append((cell[None,:] + basis) * a)
    return np.concatenate(pts, axis=0)

def bcc_positions(n_cells: int, a: float) -> np.ndarray:
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ], dtype=float)
    pts = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cell = np.array([i,j,k], dtype=float)
                pts.append((cell[None,:] + basis) * a)
    return np.concatenate(pts, axis=0)

def random_positions(n_atoms: int, box: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.random((n_atoms,3))*float(box)

def velocities_maxwell(n_atoms: int, T: float, mass: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed)+999)
    sigma = np.sqrt(float(T)/float(mass))
    v = rng.normal(0.0, sigma, size=(n_atoms,3))
    v -= v.mean(axis=0, keepdims=True)  # remove drift
    return v

def make_case_state(case: Case, mass: float) -> tuple[np.ndarray,np.ndarray,float]:
    if case.kind == "gas":
        box = float(case.box) if case.box is not None else float(case.n_cells)*float(case.a)
        n_atoms = int(case.n_cells)
        r = random_positions(n_atoms, box, case.seed)
        v = velocities_maxwell(n_atoms, case.temperature, mass, case.seed)
        return r, v, box
    if case.kind == "fcc":
        box = float(case.box) if case.box is not None else float(case.n_cells)*float(case.a)
        r = fcc_positions(case.n_cells, case.a)
        # small random jitter to avoid perfect symmetry in forces
        rng = np.random.default_rng(int(case.seed))
        r = (r + 1e-4*case.a*rng.normal(size=r.shape)) % box
        v = velocities_maxwell(r.shape[0], case.temperature, mass, case.seed)
        return r, v, box
    if case.kind == "bcc":
        box = float(case.box) if case.box is not None else float(case.n_cells)*float(case.a)
        r = bcc_positions(case.n_cells, case.a)
        rng = np.random.default_rng(int(case.seed))
        r = (r + 1e-4*case.a*rng.normal(size=r.shape)) % box
        v = velocities_maxwell(r.shape[0], case.temperature, mass, case.seed)
        return r, v, box
    raise ValueError(case.kind)

def default_cases() -> list[Case]:
    return [
        Case(name="gas_lowrho_lj", kind="gas", n_cells=256, a=1.0, temperature=1.0, box=30.0, seed=11),
        Case(name="fcc_solid", kind="fcc", n_cells=4, a=1.6, temperature=0.2, seed=22),
        Case(name="bcc_solid", kind="bcc", n_cells=5, a=1.5, temperature=0.2, seed=33),
    ]
