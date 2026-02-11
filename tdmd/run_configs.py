from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _unexpected_kwargs(scope: str, unknown: list[str]) -> TypeError:
    joined = ", ".join(unknown)
    return TypeError(f"{scope} got unexpected keyword arguments: {joined}")


def _missing_kwargs(scope: str, missing: list[str]) -> TypeError:
    joined = ", ".join(missing)
    return TypeError(f"{scope} missing required keyword arguments: {joined}")


class TDLocalRunConfig:
    def __init__(
        self,
        *,
        atom_types: np.ndarray | None = None,
        chaos_mode: bool = False,
        chaos_seed: int = 12345,
        chaos_delay_prob: float = 0.0,
        cell_size: float = 1.0,
        zones_total: int = 1,
        zone_cells_w: int = 1,
        zone_cells_s: int = 1,
        zone_cells_pattern: Any = None,
        traversal: str = "forward",
        buffer_k: float = 1.2,
        skin_from_buffer: bool = True,
        use_verlet: bool = True,
        verlet_k_steps: int = 20,
        decomposition: str = "1d",
        sync_mode: bool = False,
        zones_nx: int = 1,
        zones_ny: int = 1,
        zones_nz: int = 1,
        strict_min_zone_width: bool = False,
        ensemble_kind: str = "nve",
        thermostat: object | None = None,
        barostat: object | None = None,
        device: str = "cpu",
    ):
        self.atom_types = atom_types
        self.chaos_mode = chaos_mode
        self.chaos_seed = chaos_seed
        self.chaos_delay_prob = chaos_delay_prob
        self.cell_size = cell_size
        self.zones_total = zones_total
        self.zone_cells_w = zone_cells_w
        self.zone_cells_s = zone_cells_s
        self.zone_cells_pattern = zone_cells_pattern
        self.traversal = traversal
        self.buffer_k = buffer_k
        self.skin_from_buffer = skin_from_buffer
        self.use_verlet = use_verlet
        self.verlet_k_steps = verlet_k_steps
        self.decomposition = decomposition
        self.sync_mode = sync_mode
        self.zones_nx = zones_nx
        self.zones_ny = zones_ny
        self.zones_nz = zones_nz
        self.strict_min_zone_width = strict_min_zone_width
        self.ensemble_kind = ensemble_kind
        self.thermostat = thermostat
        self.barostat = barostat
        self.device = device

    @classmethod
    def from_legacy_kwargs(cls, kwargs: Mapping[str, Any]) -> "TDLocalRunConfig":
        allowed = {
            "atom_types",
            "chaos_mode",
            "chaos_seed",
            "chaos_delay_prob",
            "cell_size",
            "zones_total",
            "zone_cells_w",
            "zone_cells_s",
            "zone_cells_pattern",
            "traversal",
            "buffer_k",
            "skin_from_buffer",
            "use_verlet",
            "verlet_k_steps",
            "decomposition",
            "sync_mode",
            "zones_nx",
            "zones_ny",
            "zones_nz",
            "strict_min_zone_width",
            "ensemble_kind",
            "thermostat",
            "barostat",
            "device",
        }
        unknown = sorted(k for k in kwargs.keys() if k not in allowed)
        if unknown:
            raise _unexpected_kwargs("run_td_local", unknown)
        return cls(**dict(kwargs))


class TDFullMPIRunConfig:
    def __init__(
        self,
        *,
        thermo_every: int,
        cell_size: float,
        zones_total: int,
        zone_cells_w: int,
        zone_cells_s: int,
        zone_cells_pattern: Any,
        traversal: str,
        fast_sync: bool,
        strict_fast_sync: bool,
        startup_mode: str,
        warmup_steps: int,
        warmup_compute: bool,
        buffer_k: float,
        use_verlet: bool,
        verlet_k_steps: int,
        skin_from_buffer: bool,
        formal_core: bool = True,
        batch_size: int = 4,
        overlap_mode: str = "table_support",
        debug_invariants: bool = False,
        strict_min_zone_width: bool = False,
        enable_step_id: bool = True,
        max_step_lag: int = 1,
        table_max_age: int = 1,
        max_pending_delta_atoms: int = 200000,
        require_local_deps: bool = True,
        require_table_deps: bool = True,
        require_owner_deps: bool = False,
        require_owner_ver: bool = True,
        enable_req_holder: bool = True,
        holder_gossip: bool = True,
        deps_provider_mode: str = "dynamic",
        zones_nx: int = 1,
        zones_ny: int = 1,
        zones_nz: int = 1,
        owner_buffer: float = 0.0,
        cuda_aware_mpi: bool = False,
        comm_overlap_isend: bool = False,
        atom_types: np.ndarray | None = None,
        ensemble_kind: str = "nve",
        thermostat: object | None = None,
        barostat: object | None = None,
        device: str = "cpu",
        trace_enabled: bool = False,
        trace_path: str = "td_trace.csv",
    ):
        self.thermo_every = thermo_every
        self.cell_size = cell_size
        self.zones_total = zones_total
        self.zone_cells_w = zone_cells_w
        self.zone_cells_s = zone_cells_s
        self.zone_cells_pattern = zone_cells_pattern
        self.traversal = traversal
        self.fast_sync = fast_sync
        self.strict_fast_sync = strict_fast_sync
        self.startup_mode = startup_mode
        self.warmup_steps = warmup_steps
        self.warmup_compute = warmup_compute
        self.buffer_k = buffer_k
        self.use_verlet = use_verlet
        self.verlet_k_steps = verlet_k_steps
        self.skin_from_buffer = skin_from_buffer
        self.formal_core = formal_core
        self.batch_size = batch_size
        self.overlap_mode = overlap_mode
        self.debug_invariants = debug_invariants
        self.strict_min_zone_width = strict_min_zone_width
        self.enable_step_id = enable_step_id
        self.max_step_lag = max_step_lag
        self.table_max_age = table_max_age
        self.max_pending_delta_atoms = max_pending_delta_atoms
        self.require_local_deps = require_local_deps
        self.require_table_deps = require_table_deps
        self.require_owner_deps = require_owner_deps
        self.require_owner_ver = require_owner_ver
        self.enable_req_holder = enable_req_holder
        self.holder_gossip = holder_gossip
        self.deps_provider_mode = deps_provider_mode
        self.zones_nx = zones_nx
        self.zones_ny = zones_ny
        self.zones_nz = zones_nz
        self.owner_buffer = owner_buffer
        self.cuda_aware_mpi = cuda_aware_mpi
        self.comm_overlap_isend = comm_overlap_isend
        self.atom_types = atom_types
        self.ensemble_kind = ensemble_kind
        self.thermostat = thermostat
        self.barostat = barostat
        self.device = device
        self.trace_enabled = trace_enabled
        self.trace_path = trace_path

    @classmethod
    def from_legacy_kwargs(cls, kwargs: Mapping[str, Any]) -> "TDFullMPIRunConfig":
        allowed = {
            "thermo_every",
            "cell_size",
            "zones_total",
            "zone_cells_w",
            "zone_cells_s",
            "zone_cells_pattern",
            "traversal",
            "fast_sync",
            "strict_fast_sync",
            "startup_mode",
            "warmup_steps",
            "warmup_compute",
            "buffer_k",
            "use_verlet",
            "verlet_k_steps",
            "skin_from_buffer",
            "formal_core",
            "batch_size",
            "overlap_mode",
            "debug_invariants",
            "strict_min_zone_width",
            "enable_step_id",
            "max_step_lag",
            "table_max_age",
            "max_pending_delta_atoms",
            "require_local_deps",
            "require_table_deps",
            "require_owner_deps",
            "require_owner_ver",
            "enable_req_holder",
            "holder_gossip",
            "deps_provider_mode",
            "zones_nx",
            "zones_ny",
            "zones_nz",
            "owner_buffer",
            "cuda_aware_mpi",
            "comm_overlap_isend",
            "atom_types",
            "ensemble_kind",
            "thermostat",
            "barostat",
            "device",
            "trace_enabled",
            "trace_path",
        }
        unknown = sorted(k for k in kwargs.keys() if k not in allowed)
        if unknown:
            raise _unexpected_kwargs("run_td_full_mpi_1d", unknown)
        required = (
            "thermo_every",
            "cell_size",
            "zones_total",
            "zone_cells_w",
            "zone_cells_s",
            "zone_cells_pattern",
            "traversal",
            "fast_sync",
            "strict_fast_sync",
            "startup_mode",
            "warmup_steps",
            "warmup_compute",
            "buffer_k",
            "use_verlet",
            "verlet_k_steps",
            "skin_from_buffer",
        )
        missing = sorted(k for k in required if k not in kwargs)
        if missing:
            raise _missing_kwargs("run_td_full_mpi_1d", missing)
        return cls(**dict(kwargs))


class VerifyTaskRunConfig:
    def __init__(
        self,
        *,
        atom_types: np.ndarray | None = None,
        cell_size: float,
        zones_total: int,
        zone_cells_w: int,
        zone_cells_s: int,
        zone_cells_pattern: Any,
        traversal: str,
        buffer_k: float,
        skin_from_buffer: bool,
        use_verlet: bool,
        verlet_k_steps: int,
        steps: int = 50,
        observer_every: int = 5,
        tol_dr: float = 1e-6,
        tol_dv: float = 1e-6,
        tol_dE: float = 1e-5,
        tol_dT: float = 1e-5,
        tol_dP: float = 1e-4,
        decomposition: str = "1d",
        zones_nx: int = 1,
        zones_ny: int = 1,
        zones_nz: int = 1,
        sync_mode: bool = False,
        device: str = "cpu",
        strict_min_zone_width: bool = False,
        ensemble_kind: str = "nve",
        thermostat: object | None = None,
        barostat: object | None = None,
        chaos_mode: bool = False,
        chaos_seed: int = 12345,
        chaos_delay_prob: float = 0.0,
        case_name: str = "interop_task",
    ):
        self.atom_types = atom_types
        self.cell_size = cell_size
        self.zones_total = zones_total
        self.zone_cells_w = zone_cells_w
        self.zone_cells_s = zone_cells_s
        self.zone_cells_pattern = zone_cells_pattern
        self.traversal = traversal
        self.buffer_k = buffer_k
        self.skin_from_buffer = skin_from_buffer
        self.use_verlet = use_verlet
        self.verlet_k_steps = verlet_k_steps
        self.steps = steps
        self.observer_every = observer_every
        self.tol_dr = tol_dr
        self.tol_dv = tol_dv
        self.tol_dE = tol_dE
        self.tol_dT = tol_dT
        self.tol_dP = tol_dP
        self.decomposition = decomposition
        self.zones_nx = zones_nx
        self.zones_ny = zones_ny
        self.zones_nz = zones_nz
        self.sync_mode = sync_mode
        self.device = device
        self.strict_min_zone_width = strict_min_zone_width
        self.ensemble_kind = ensemble_kind
        self.thermostat = thermostat
        self.barostat = barostat
        self.chaos_mode = chaos_mode
        self.chaos_seed = chaos_seed
        self.chaos_delay_prob = chaos_delay_prob
        self.case_name = case_name

    @classmethod
    def from_legacy_kwargs(cls, kwargs: Mapping[str, Any]) -> "VerifyTaskRunConfig":
        allowed = {
            "atom_types",
            "cell_size",
            "zones_total",
            "zone_cells_w",
            "zone_cells_s",
            "zone_cells_pattern",
            "traversal",
            "buffer_k",
            "skin_from_buffer",
            "use_verlet",
            "verlet_k_steps",
            "steps",
            "observer_every",
            "tol_dr",
            "tol_dv",
            "tol_dE",
            "tol_dT",
            "tol_dP",
            "decomposition",
            "zones_nx",
            "zones_ny",
            "zones_nz",
            "sync_mode",
            "device",
            "strict_min_zone_width",
            "ensemble_kind",
            "thermostat",
            "barostat",
            "chaos_mode",
            "chaos_seed",
            "chaos_delay_prob",
            "case_name",
        }
        unknown = sorted(k for k in kwargs.keys() if k not in allowed)
        if unknown:
            raise _unexpected_kwargs("run_verify_task", unknown)
        required = (
            "cell_size",
            "zones_total",
            "zone_cells_w",
            "zone_cells_s",
            "zone_cells_pattern",
            "traversal",
            "buffer_k",
            "skin_from_buffer",
            "use_verlet",
            "verlet_k_steps",
        )
        missing = sorted(k for k in required if k not in kwargs)
        if missing:
            raise _missing_kwargs("run_verify_task", missing)
        return cls(**dict(kwargs))
