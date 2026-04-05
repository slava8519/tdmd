from __future__ import annotations

import numpy as np
import pytest

from tdmd.backend import resolve_backend
from tdmd.celllist import forces_on_targets_celllist
from tdmd.forces_gpu import (
    _cached_rawkernel,
    build_neighbor_list_celllist_backend,
    forces_on_targets_celllist_backend,
    forces_on_targets_pair_backend,
    get_last_device_state_sync_diagnostics,
    get_last_neighbor_list_diagnostics,
    mark_device_state_dirty,
    reset_device_state_cache,
    reset_rawkernel_cache,
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


def test_cached_rawkernel_reuses_handle_for_same_backend_and_name():
    created: list[tuple[str, str]] = []

    class _FakeDevice:
        id = 7

    class _FakeCuda:
        @staticmethod
        def Device():
            return _FakeDevice()

    class _FakeCP:
        cuda = _FakeCuda()

        @staticmethod
        def RawKernel(src: str, name: str):
            created.append((src, name))
            return {"src": src, "name": name, "idx": len(created)}

    cp = _FakeCP()
    reset_rawkernel_cache()
    k0 = _cached_rawkernel(cp, "extern \"C\" __global__ void k() {}", "k")
    k1 = _cached_rawkernel(cp, "extern \"C\" __global__ void k() {}", "k")

    assert k0 is k1
    assert created == [('extern "C" __global__ void k() {}', "k")]


def test_cached_rawkernel_separates_entries_by_kernel_name():
    created: list[tuple[str, str]] = []

    class _FakeDevice:
        id = 3

    class _FakeCuda:
        @staticmethod
        def Device():
            return _FakeDevice()

    class _FakeCP:
        cuda = _FakeCuda()

        @staticmethod
        def RawKernel(src: str, name: str):
            created.append((src, name))
            return {"src": src, "name": name, "idx": len(created)}

    cp = _FakeCP()
    reset_rawkernel_cache()
    k0 = _cached_rawkernel(cp, "src0", "kernel0")
    k1 = _cached_rawkernel(cp, "src1", "kernel1")

    assert k0 is not k1
    assert created == [("src0", "kernel0"), ("src1", "kernel1")]


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_device_state_sync_skips_h2d_when_host_state_is_unchanged():
    r, _v, box, atom_types = _sample_state(18)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::2]
    pot = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    backend = resolve_backend("cuda")

    reset_device_state_cache()
    f0 = forces_on_targets_pair_backend(
        r=r,
        box=box,
        cutoff=2.8,
        potential=pot,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=backend,
        prefer_marked_dirty=True,
    )
    assert f0 is not None
    diag0 = get_last_device_state_sync_diagnostics()
    assert diag0.full_sync is True
    assert diag0.last_synced_atoms == r.shape[0]

    f1 = forces_on_targets_pair_backend(
        r=r,
        box=box,
        cutoff=2.8,
        potential=pot,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=backend,
        prefer_marked_dirty=True,
    )
    assert f1 is not None
    diag1 = get_last_device_state_sync_diagnostics()
    assert diag1.full_sync is False
    assert diag1.used_dirty_tracking is True
    assert diag1.last_synced_atoms == 0
    assert np.allclose(f1, f0, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_device_state_sync_updates_only_marked_atoms():
    r, _v, box, atom_types = _sample_state(20)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::2]
    pot = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    backend = resolve_backend("cuda")

    reset_device_state_cache()
    f0 = forces_on_targets_pair_backend(
        r=r,
        box=box,
        cutoff=2.8,
        potential=pot,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=backend,
        prefer_marked_dirty=True,
    )
    assert f0 is not None

    changed = np.asarray([0, 3, 7], dtype=np.int32)
    r[changed, 0] = (r[changed, 0] + 0.05) % box
    mark_device_state_dirty(r, changed)

    f1 = forces_on_targets_pair_backend(
        r=r,
        box=box,
        cutoff=2.8,
        potential=pot,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=backend,
        prefer_marked_dirty=True,
    )
    assert f1 is not None
    diag = get_last_device_state_sync_diagnostics()
    assert diag.full_sync is False
    assert diag.used_dirty_tracking is True
    assert diag.last_synced_atoms == changed.size
    assert not np.allclose(f1, f0, atol=1e-12, rtol=1e-12)


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
@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("lj", {"epsilon": 1.0, "sigma": 1.0}),
        ("morse", {"D_e": 0.4, "a": 1.2, "r0": 1.1}),
        ("table", {"file": "examples/interop/table_zero.table", "keyword": "ZERO"}),
    ],
)
def test_gpu_pair_backend_uses_neighbor_list_pipeline_for_pair_potentials(kind, params):
    r, _v, box, atom_types = _sample_state(20)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::2]
    pot = make_potential(kind, params)
    cutoff = 2.8

    empty = np.zeros((0,), dtype=np.int32)
    reset = build_neighbor_list_celllist_backend(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=cutoff,
        target_ids=empty,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=resolve_backend("cuda"),
    )
    assert reset is not None
    diag0 = get_last_neighbor_list_diagnostics()
    assert diag0.attempts == 0

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
    diag = get_last_neighbor_list_diagnostics()
    assert diag.attempts >= 1
    assert diag.max_neighbors_used > 0
    assert diag.overflowed is False


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_pair_table_matches_cpu_celllist(tmp_path):
    r, _v, box, atom_types = _sample_state(22)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[1::3]
    table_path = tmp_path / "pair_force.table"
    table_path.write_text(
        "\n".join(
            [
                "# Non-zero table for GPU direct-pair regression",
                "PAIRF",
                "N 5 R 0.8 2.8",
                "1 0.800000 0.000000 1.200000",
                "2 1.200000 0.000000 0.800000",
                "3 1.600000 0.000000 0.400000",
                "4 2.200000 0.000000 0.150000",
                "5 2.800000 0.000000 0.000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pot = make_potential("table", {"file": str(table_path), "keyword": "PAIRF"})
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


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_neighbor_list_kernel_parity_with_cpu_celllist():
    r, _v, box, atom_types = _sample_state(26)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::2]
    pot = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    cutoff = 2.7

    built = build_neighbor_list_celllist_backend(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=cutoff,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=resolve_backend("cuda"),
        max_neighbors=1024,
    )
    assert built is not None
    neighbor_ids, counts = built
    assert neighbor_ids.shape[0] == target.size
    assert counts.shape == (target.size,)

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
def test_gpu_neighbor_list_kernel_auto_retry_recovers_from_small_initial_buffer():
    r, _v, box, atom_types = _sample_state(40)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::2]
    cutoff = 3.5

    built = build_neighbor_list_celllist_backend(
        r=r,
        box=box,
        cutoff=cutoff,
        rc=cutoff,
        target_ids=target,
        candidate_ids=ids,
        atom_types=atom_types,
        backend=resolve_backend("cuda"),
        max_neighbors=1,
        max_retries=10,
    )
    assert built is not None
    _neighbor_ids, counts = built
    assert counts.size == target.size
    diag = get_last_neighbor_list_diagnostics()
    assert diag.overflow_retries > 0
    assert diag.final_max_neighbors > 1
    assert diag.overflowed is False


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_celllist_backend_prefers_neighbor_list_path_over_pair_fallback(monkeypatch):
    r, _v, box, atom_types = _sample_state(28)
    ids = np.arange(r.shape[0], dtype=np.int32)
    target = ids[::2]
    pot = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    cutoff = 2.8

    f_cpu = forces_on_targets_celllist(
        r, box, pot, cutoff, target, ids, rc=cutoff, atom_types=atom_types
    )

    def _forbid_pair_fallback(**_kwargs):
        raise AssertionError("pair fallback should not be used when neighbor list build succeeds")

    monkeypatch.setattr("tdmd.forces_gpu.forces_on_targets_pair_backend", _forbid_pair_fallback)

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


@pytest.mark.skipif(resolve_backend("auto").device != "cuda", reason="CUDA backend not available")
def test_gpu_eam_pair_backend_uses_neighbor_list_pipeline_without_dense_interp_helper(
    monkeypatch,
):
    task = load_task("examples/interop/task_eam_alloy_uniform_mass.yaml")
    arr = task_to_arrays(task)
    pot = make_potential(task.potential.kind, task.potential.params)
    ids = np.arange(arr.r.shape[0], dtype=np.int32)

    empty = np.zeros((0,), dtype=np.int32)
    reset = build_neighbor_list_celllist_backend(
        r=arr.r,
        box=float(task.box.x),
        cutoff=float(task.cutoff),
        rc=float(task.cutoff),
        target_ids=empty,
        candidate_ids=ids,
        atom_types=arr.atom_types,
        backend=resolve_backend("cuda"),
    )
    assert reset is not None
    diag0 = get_last_neighbor_list_diagnostics()
    assert diag0.attempts == 0

    def _forbid_dense_interp(*_args, **_kwargs):
        raise AssertionError("dense EAM interpolation helper should not be used")

    monkeypatch.setattr("tdmd.forces_gpu._interp_with_grad_table_cp", _forbid_dense_interp)

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
    diag = get_last_neighbor_list_diagnostics()
    assert diag.attempts >= 1
    assert diag.max_neighbors_used > 0
    assert diag.overflowed is False
