from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, List, Optional
import yaml

PotentialType = Literal["lj", "morse", "table", "eam/alloy", "eam_alloy"]
TraversalType = Literal["forward","backward","snake"]
StartupMode = Literal["rank0_all","scatter_zones"]
OverlapMode = Literal["table_support","geometric_rc"]

_ENSEMBLE_KINDS = {"nve", "nvt", "npt"}

@dataclass
class PotentialConfig:
    kind: PotentialType
    params: Dict[str, Any]

@dataclass
class SystemConfig:
    n_atoms: int
    mass: float
    box: float
    temperature: float
    seed: int = 1

@dataclass
class TDConfig:
    cell_size: float
    zones_total: int
    # v3.9: zone geometry
    decomposition: str = "1d"  # "1d" slabs (legacy) or "3d" blocks
    zones_nx: int = 1
    zones_ny: int = 1
    zones_nz: int = 1
    zone_cells_w: int = 1
    zone_cells_s: int = 1
    zone_cells_pattern: Optional[List[int]] = None
    traversal: TraversalType = "snake"
    fast_sync: bool = False
    strict_fast_sync: bool = False

    startup_mode: StartupMode = "scatter_zones"
    warmup_steps: int = 2
    warmup_compute: bool = True

    buffer_k: float = 1.2
    use_verlet: bool = True
    verlet_k_steps: int = 20
    skin_from_buffer: bool = True

    formal_core: bool = True
    batch_size: int = 4

    overlap_mode: OverlapMode = "table_support"
    debug_invariants: bool = False
    strict_min_zone_width: bool = False

    enable_step_id: bool = True

    # v1.8: time-lag safety controls (true TD)
    max_step_lag: int = 1      # max allowed lag between a work-zone and its dependencies
    table_max_age: int = 1     # max allowed age of interaction table in zone-time steps

    # v2.2: bounded pending-delta buffer (memory safety)
    max_pending_delta_atoms: int = 200000
    require_local_deps: bool = True  # v3.1 legacy alias (mapped to require_table_deps)
    require_table_deps: bool = True   # v3.2: readiness for table/forces deps (present locally: owned or shadow)
    require_owner_deps: bool = False  # readiness for ownership deps (optional; stronger, slower)
    require_owner_ver: bool = True    # v3.4: owner-deps require up-to-date holder version (not just known)
    enable_req_holder: bool = True    # v3.4: pull protocol for holder map
    holder_gossip: bool = True        # piggyback holder-map updates
    chaos_mode: bool = False       # v3.8: randomize TD-local scheduling and delta application
    chaos_seed: int = 12345        # v3.8: RNG seed for chaos mode
    chaos_delay_prob: float = 0.0  # v3.8: probability to defer an applicable delta/halo once
    deps_provider_mode: str = "dynamic"  # v4.1: dynamic|static_rr|static_3d
    owner_buffer: float = 0.0  # 0=>cutoff
    cuda_aware_mpi: bool = False
    comm_overlap_isend: bool = False

@dataclass
class RunConfig:
    dt: float
    n_steps: int
    thermo_every: int = 50
    cutoff: float = 8.0
    device: str = "auto"

@dataclass
class EnsembleControlConfig:
    kind: str
    params: Dict[str, Any]

@dataclass
class EnsembleConfig:
    kind: str = "nve"
    thermostat: Optional[EnsembleControlConfig] = None
    barostat: Optional[EnsembleControlConfig] = None

@dataclass
class Config:
    system: SystemConfig
    potential: PotentialConfig
    run: RunConfig
    td: TDConfig
    ensemble: EnsembleConfig


def _parse_ensemble_control(section: Any, key: str) -> EnsembleControlConfig:
    if not isinstance(section, dict):
        raise ValueError(f"{key} must be a mapping")
    extra = sorted(set(section.keys()) - {"kind", "params"})
    if extra:
        raise ValueError(f"{key} contains unsupported keys: {extra}")
    kind = str(section.get("kind", "")).strip().lower()
    if not kind:
        raise ValueError(f"{key}.kind must be a non-empty string")
    params = section.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"{key}.params must be a mapping")
    return EnsembleControlConfig(kind=kind, params=dict(params))


def _parse_ensemble_config(root: dict[str, Any]) -> EnsembleConfig:
    ens = root.get("ensemble", None)
    if ens is None:
        return EnsembleConfig(kind="nve", thermostat=None, barostat=None)
    if not isinstance(ens, dict):
        raise ValueError("ensemble must be a mapping")
    extra = sorted(set(ens.keys()) - {"kind", "thermostat", "barostat"})
    if extra:
        raise ValueError(f"ensemble contains unsupported keys: {extra}")

    kind = str(ens.get("kind", "")).strip().lower()
    if kind not in _ENSEMBLE_KINDS:
        raise ValueError(f"ensemble.kind must be one of {sorted(_ENSEMBLE_KINDS)}")

    thermostat = None
    barostat = None
    if ens.get("thermostat", None) is not None:
        thermostat = _parse_ensemble_control(ens.get("thermostat"), "ensemble.thermostat")
    if ens.get("barostat", None) is not None:
        barostat = _parse_ensemble_control(ens.get("barostat"), "ensemble.barostat")

    if kind == "nve":
        if thermostat is not None or barostat is not None:
            raise ValueError("ensemble.kind=nve must not include thermostat/barostat")
    elif kind == "nvt":
        if thermostat is None:
            raise ValueError("ensemble.kind=nvt requires ensemble.thermostat")
        if barostat is not None:
            raise ValueError("ensemble.kind=nvt must not include ensemble.barostat")
    elif kind == "npt":
        if thermostat is None:
            raise ValueError("ensemble.kind=npt requires ensemble.thermostat")
        if barostat is None:
            raise ValueError("ensemble.kind=npt requires ensemble.barostat")

    return EnsembleConfig(kind=kind, thermostat=thermostat, barostat=barostat)

def load_config(path: str) -> Config:
    with open(path,"r",encoding="utf-8") as f:
        d = yaml.safe_load(f)

    td = d["td"]
    pattern = td.get("zone_cells_pattern", None)
    if pattern is not None:
        pattern = [int(x) for x in pattern]

    batch_size = int(td.get("time_block_k", td.get("batch_size", 4)))
    if batch_size < 1:
        raise ValueError("td.time_block_k (or td.batch_size) must be >= 1")
    device = str(d.get("run", {}).get("device", "auto")).lower()
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("run.device must be one of: auto, cpu, cuda")
    ensemble = _parse_ensemble_config(d)

    return Config(
        system=SystemConfig(
            n_atoms=int(d["system"]["n_atoms"]),
            mass=float(d["system"]["mass"]),
            box=float(d["system"]["box"]),
            temperature=float(d["system"].get("temperature",300.0)),
            seed=int(d["system"].get("seed",1)),
        ),
        potential=PotentialConfig(
            kind=str(d["potential"]["kind"]).lower(),
            params=dict(d["potential"].get("params", {})),
        ),
        run=RunConfig(
            dt=float(d["run"]["dt"]),
            n_steps=int(d["run"]["n_steps"]),
            thermo_every=int(d["run"].get("thermo_every",50)),
            cutoff=float(d["run"].get("cutoff",8.0)),
            device=device,
        ),
        td=TDConfig(
            cell_size=float(td["cell_size"]),
            zones_total=int(td["zones_total"]),
            zone_cells_w=int(td.get("zone_cells_w",1)),
            zone_cells_s=int(td.get("zone_cells_s",1)),
            zone_cells_pattern=pattern,
            traversal=str(td.get("traversal","snake")).lower(),
            fast_sync=bool(td.get("fast_sync", False)),
            strict_fast_sync=bool(td.get("strict_fast_sync", False)),
            startup_mode=str(td.get("startup_mode","scatter_zones")).lower(),
            warmup_steps=int(td.get("warmup_steps",2)),
            warmup_compute=bool(td.get("warmup_compute",True)),
            buffer_k=float(td.get("buffer_k",1.2)),
            use_verlet=bool(td.get("use_verlet",True)),
            verlet_k_steps=int(td.get("verlet_k_steps",20)),
            skin_from_buffer=bool(td.get("skin_from_buffer",True)),
            formal_core=bool(td.get("formal_core", True)),
            batch_size=batch_size,
            overlap_mode=str(td.get("overlap_mode","table_support")).lower(),
            debug_invariants=bool(td.get("debug_invariants", False)),
            strict_min_zone_width=bool(td.get("strict_min_zone_width", False)),
            enable_step_id=bool(td.get("enable_step_id", True)),
            max_step_lag=int(td.get("max_step_lag", 1)),
            table_max_age=int(td.get("table_max_age", 1)),
            max_pending_delta_atoms=int(td.get("max_pending_delta_atoms", 200000)),
            require_local_deps=bool(td.get("require_local_deps", True)),
            require_table_deps=bool(td.get("require_table_deps", td.get("require_local_deps", True))),
            require_owner_deps=bool(td.get("require_owner_deps", False)),
            require_owner_ver=bool(td.get("require_owner_ver", True)),
            enable_req_holder=bool(td.get("enable_req_holder", True)),
            holder_gossip=bool(td.get("holder_gossip", True)),
            chaos_mode=bool(td.get("chaos_mode", False)),
            chaos_seed=int(td.get("chaos_seed", 12345)),
            chaos_delay_prob=float(td.get("chaos_delay_prob", 0.0)),
            deps_provider_mode=str(td.get("deps_provider_mode", "dynamic")),
            zones_nx=int(td.get("zones_nx", 1)),
            zones_ny=int(td.get("zones_ny", 1)),
            zones_nz=int(td.get("zones_nz", 1)),
            owner_buffer=float(td.get("owner_buffer", 0.0)),
            cuda_aware_mpi=bool(td.get("cuda_aware_mpi", False)),
            comm_overlap_isend=bool(td.get("comm_overlap_isend", False)),
        ),
        ensemble=ensemble,
    )
