from __future__ import annotations

import numpy as np
import pytest

from tdmd.backend import resolve_backend
from tdmd.celllist import forces_on_targets_celllist
from tdmd.forces_gpu import (
    forces_on_targets_celllist_backend,
    forces_on_targets_pair_backend,
    supports_pair_gpu,
)
from tdmd.io.task import load_task, task_to_arrays
from tdmd.potentials import make_potential
from tdmd.serial import run_serial
from tdmd.td_local import run_td_local


def _sample_state(n: int = 24):
    rng = np.random.default_rng(123)
    box = 12.0
    r = rng.uniform(0.0, box, size=(n, 3)).astype(float)
    v = rng.normal(0.0, 0.1, size=(n, 3)).astype(float)
    atom_types = rng.integers(1, 3, size=(n,), dtype=np.int32)
    return r, v, box, atom_types


def test_supports_pair_gpu_flags():
    lj = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    morse = make_potential("morse", {"D_e": 0.4, "a": 1.2, "r0": 1.1})
    table = make_potential(
        "table", {"file": "examples/interop/table_zero.table", "keyword": "ZERO"}
    )
    assert supports_pair_gpu(lj)
    assert supports_pair_gpu(morse)
    assert supports_pair_gpu(table)


def test_gpu_pair_backend_returns_none_on_cpu_backend():
    r, _v, box, atom_types = _sample_state(12)
    ids = np.arange(r.shape[0], dtype=np.int32)
    lj = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    cpu_backend = resolve_backend("cpu")
    f = forces_on_targets_pair_backend(
        r=r,
        box=box,
        cutoff=2.5,
        potential=lj,
        target_ids=ids,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=cpu_backend,
    )
    assert f is None


def test_gpu_celllist_backend_returns_none_on_cpu_backend():
    r, _v, box, atom_types = _sample_state(12)
    ids = np.arange(r.shape[0], dtype=np.int32)
    lj = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    cpu_backend = resolve_backend("cpu")
    f = forces_on_targets_celllist_backend(
        r=r,
        box=box,
        cutoff=2.5,
        rc=2.5,
        potential=lj,
        target_ids=ids,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=cpu_backend,
    )
    assert f is None


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_pair_lj_matches_cpu_celllist():
    r, _v, box, atom_types = _sample_state(20)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::2]
    pot = make_potential(
        "lj",
        {
            "epsilon": 1.0,
            "sigma": 1.0,
            "pair_coeffs": {
                "1-1": {"epsilon": 0.8, "sigma": 1.0},
                "1-2": {"epsilon": 0.9, "sigma": 1.1},
                "2-2": {"epsilon": 1.1, "sigma": 0.95},
            },
        },
    )
    cutoff = 2.8
    f_cpu = forces_on_targets_celllist(
        r, box, pot, cutoff, target, ids, rc=cutoff, atom_types=atom_types
    )
    f_gpu = forces_on_targets_pair_backend(
        r=r,
        box=box,
        cutoff=cutoff,
        potential=pot,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=resolve_backend("cuda"),
    )
    assert f_gpu is not None
    assert np.allclose(f_gpu, f_cpu, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_pair_morse_matches_cpu_celllist():
    r, _v, box, atom_types = _sample_state(18)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[1::2]
    pot = make_potential(
        "morse",
        {
            "D_e": 0.4,
            "a": 1.2,
            "r0": 1.1,
            "pair_coeffs": {
                "1-1": {"D_e": 0.35, "a": 1.0, "r0": 1.0},
                "1-2": {"D_e": 0.45, "a": 1.3, "r0": 1.1},
                "2-2": {"D_e": 0.55, "a": 1.1, "r0": 1.2},
            },
        },
    )
    cutoff = 3.0
    f_cpu = forces_on_targets_celllist(
        r, box, pot, cutoff, target, ids, rc=cutoff, atom_types=atom_types
    )
    f_gpu = forces_on_targets_pair_backend(
        r=r,
        box=box,
        cutoff=cutoff,
        potential=pot,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=resolve_backend("cuda"),
    )
    assert f_gpu is not None
    assert np.allclose(f_gpu, f_cpu, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_celllist_lj_matches_cpu_celllist():
    r, _v, box, atom_types = _sample_state(22)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::3]
    pot = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    cutoff = 2.6
    f_cpu = forces_on_targets_celllist(
        r, box, pot, cutoff, target, ids, rc=cutoff, atom_types=atom_types
    )
    f_gpu = forces_on_targets_celllist_backend(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=cutoff,
        potential=pot,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=resolve_backend("cuda"),
    )
    assert f_gpu is not None
    assert np.allclose(f_gpu, f_cpu, atol=1e-9, rtol=1e-9)


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_celllist_table_matches_cpu_celllist():
    r, _v, box, atom_types = _sample_state(22)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::4]
    pot = make_potential("table", {"file": "examples/interop/table_zero.table", "keyword": "ZERO"})
    cutoff = 2.5
    f_cpu = forces_on_targets_celllist(
        r, box, pot, cutoff, target, ids, rc=cutoff, atom_types=atom_types
    )
    f_gpu = forces_on_targets_celllist_backend(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=cutoff,
        potential=pot,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=resolve_backend("cuda"),
    )
    assert f_gpu is not None
    assert np.allclose(f_gpu, f_cpu, atol=1e-9, rtol=1e-9)


def test_run_serial_cuda_request_preserves_results():
    r0, v0, box, atom_types = _sample_state(16)
    pot = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})

    r_cpu = r0.copy()
    v_cpu = v0.copy()
    run_serial(r_cpu, v_cpu, 1.0, box, pot, 0.002, 2.8, 3, atom_types=atom_types, device="cpu")

    r_dev = r0.copy()
    v_dev = v0.copy()
    auto = resolve_backend("auto")
    if auto.device != "cuda":
        with pytest.warns(RuntimeWarning, match="falling back to CPU"):
            run_serial(
                r_dev, v_dev, 1.0, box, pot, 0.002, 2.8, 3, atom_types=atom_types, device="cuda"
            )
    else:
        run_serial(r_dev, v_dev, 1.0, box, pot, 0.002, 2.8, 3, atom_types=atom_types, device="cuda")

    assert np.allclose(r_dev, r_cpu, atol=1e-8, rtol=1e-8)
    assert np.allclose(v_dev, v_cpu, atol=1e-8, rtol=1e-8)


def test_td_local_sync_cuda_request_preserves_results():
    r0, v0, box, atom_types = _sample_state(20)
    pot = make_potential("morse", {"D_e": 0.3, "a": 1.1, "r0": 1.0})

    r_cpu = r0.copy()
    v_cpu = v0.copy()
    run_td_local(
        r_cpu,
        v_cpu,
        1.0,
        box,
        pot,
        0.002,
        3.0,
        2,
        atom_types=atom_types,
        sync_mode=True,
        use_verlet=False,
        zones_total=4,
        cell_size=2.0,
        device="cpu",
    )

    r_dev = r0.copy()
    v_dev = v0.copy()
    auto = resolve_backend("auto")
    if auto.device != "cuda":
        with pytest.warns(RuntimeWarning, match="falling back to CPU"):
            run_td_local(
                r_dev,
                v_dev,
                1.0,
                box,
                pot,
                0.002,
                3.0,
                2,
                atom_types=atom_types,
                sync_mode=True,
                use_verlet=False,
                zones_total=4,
                cell_size=2.0,
                device="cuda",
            )
    else:
        run_td_local(
            r_dev,
            v_dev,
            1.0,
            box,
            pot,
            0.002,
            3.0,
            2,
            atom_types=atom_types,
            sync_mode=True,
            use_verlet=False,
            zones_total=4,
            cell_size=2.0,
            device="cuda",
        )

    assert np.allclose(r_dev, r_cpu, atol=1e-8, rtol=1e-8)
    assert np.allclose(v_dev, v_cpu, atol=1e-8, rtol=1e-8)


def test_td_local_async_cuda_request_preserves_results():
    r0, v0, box, atom_types = _sample_state(24)
    pot = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})

    r_cpu = r0.copy()
    v_cpu = v0.copy()
    run_td_local(
        r_cpu,
        v_cpu,
        1.0,
        box,
        pot,
        0.001,
        2.5,
        2,
        atom_types=atom_types,
        sync_mode=False,
        use_verlet=True,
        verlet_k_steps=5,
        zones_total=6,
        cell_size=2.0,
        device="cpu",
    )

    r_dev = r0.copy()
    v_dev = v0.copy()
    auto = resolve_backend("auto")
    if auto.device != "cuda":
        with pytest.warns(RuntimeWarning, match="falling back to CPU"):
            run_td_local(
                r_dev,
                v_dev,
                1.0,
                box,
                pot,
                0.001,
                2.5,
                2,
                atom_types=atom_types,
                sync_mode=False,
                use_verlet=True,
                verlet_k_steps=5,
                zones_total=6,
                cell_size=2.0,
                device="cuda",
            )
    else:
        run_td_local(
            r_dev,
            v_dev,
            1.0,
            box,
            pot,
            0.001,
            2.5,
            2,
            atom_types=atom_types,
            sync_mode=False,
            use_verlet=True,
            verlet_k_steps=5,
            zones_total=6,
            cell_size=2.0,
            device="cuda",
        )

    assert np.allclose(r_dev, r_cpu, atol=1e-8, rtol=1e-8)
    assert np.allclose(v_dev, v_cpu, atol=1e-8, rtol=1e-8)


def test_run_serial_eam_cuda_request_preserves_results():
    task = load_task("examples/interop/task_eam_al.yaml")
    arr = task_to_arrays(task)
    pot = make_potential(task.potential.kind, task.potential.params)
    r0 = arr.r.copy()
    v0 = arr.v.copy()
    mass = arr.masses.copy()
    atom_types = arr.atom_types.copy()

    r_cpu = r0.copy()
    v_cpu = v0.copy()
    run_serial(
        r_cpu,
        v_cpu,
        mass,
        float(task.box.x),
        pot,
        float(task.dt),
        float(task.cutoff),
        2,
        atom_types=atom_types,
        device="cpu",
    )

    r_dev = r0.copy()
    v_dev = v0.copy()
    auto = resolve_backend("auto")
    if auto.device != "cuda":
        with pytest.warns(RuntimeWarning, match="falling back to CPU"):
            run_serial(
                r_dev,
                v_dev,
                mass,
                float(task.box.x),
                pot,
                float(task.dt),
                float(task.cutoff),
                2,
                atom_types=atom_types,
                device="cuda",
            )
    else:
        run_serial(
            r_dev,
            v_dev,
            mass,
            float(task.box.x),
            pot,
            float(task.dt),
            float(task.cutoff),
            2,
            atom_types=atom_types,
            device="cuda",
        )

    assert np.allclose(r_dev, r_cpu, atol=1e-8, rtol=1e-8)
    assert np.allclose(v_dev, v_cpu, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_eam_targets_match_cpu_reference():
    task = load_task("examples/interop/task_eam_alloy_uniform_mass.yaml")
    arr = task_to_arrays(task)
    pot = make_potential(task.potential.kind, task.potential.params)
    ids = np.arange(arr.r.shape[0], dtype=np.int32)
    f_cpu = pot.forces_on_targets(
        r=arr.r,
        box=float(task.box.x),
        cutoff=float(task.cutoff),
        atom_types=arr.atom_types,
        target_ids=ids,
        candidate_ids=ids,
    )
    f_gpu = forces_on_targets_pair_backend(
        r=arr.r,
        box=float(task.box.x),
        cutoff=float(task.cutoff),
        potential=pot,
        target_ids=ids,
        candidate_ids=ids,
        atom_types=arr.atom_types,
        backend=resolve_backend("cuda"),
    )
    assert f_gpu is not None
    assert np.allclose(f_gpu, f_cpu, atol=1e-8, rtol=1e-8)
