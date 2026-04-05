from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ManyBodyEvaluationScope = Literal["full_system", "target_local"]
ManyBodyConsumptionScope = Literal["all_atoms", "target_ids"]


@dataclass(frozen=True)
class ManyBodyForceScope:
    runtime_kind: str
    evaluation_scope: ManyBodyEvaluationScope
    consumption_scope: ManyBodyConsumptionScope
    target_local_available: bool
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def is_many_body_potential(potential: object) -> bool:
    return hasattr(potential, "forces_energy_virial")


def supports_target_local_many_body(potential: object) -> bool:
    return is_many_body_potential(potential) and hasattr(potential, "forces_on_targets")


def td_local_many_body_force_scope(
    potential: object,
    *,
    sync_mode: bool,
    decomposition: str,
    device: str = "cpu",
) -> ManyBodyForceScope | None:
    if not is_many_body_potential(potential):
        return None

    target_local_available = supports_target_local_many_body(potential)
    decomp = str(decomposition).strip().lower()
    backend_device = str(device).strip().lower()
    if sync_mode:
        return ManyBodyForceScope(
            runtime_kind="td_local.sync_global",
            evaluation_scope="full_system",
            consumption_scope="all_atoms",
            target_local_available=target_local_available,
            rationale=(
                "sync many-body td_local dispatches through _TDLocalCtx.forces_full and consumes "
                "the full-system result directly"
            ),
        )
    if target_local_available:
        runtime_kind = "td_local.async_3d" if decomp == "3d" else "td_local.async_1d"
        if backend_device == "cuda":
            return ManyBodyForceScope(
                runtime_kind=runtime_kind,
                evaluation_scope="target_local",
                consumption_scope="target_ids",
                target_local_available=True,
                rationale=(
                    f"{runtime_kind} now tries CUDA target/candidate-local many-body dispatch first "
                    "and falls back to the existing full-system GPU path only if that refinement "
                    "is unavailable for a call"
                ),
            )
        return ManyBodyForceScope(
            runtime_kind=runtime_kind,
            evaluation_scope="target_local",
            consumption_scope="target_ids",
            target_local_available=True,
            rationale=(
                f"{runtime_kind} uses potential.forces_on_targets(...) on CPU for many-body target "
                "evaluation while keeping TD scheduling unchanged"
            ),
        )
    if decomp == "3d":
        return ManyBodyForceScope(
            runtime_kind="td_local.async_3d",
            evaluation_scope="full_system",
            consumption_scope="target_ids",
            target_local_available=target_local_available,
            rationale=(
                "async 3d many-body td_local currently evaluates _TDLocalCtx.forces_full(ctx.r) "
                "and slices the result to zone target ids"
            ),
        )
    return ManyBodyForceScope(
        runtime_kind="td_local.async_1d",
        evaluation_scope="full_system",
        consumption_scope="target_ids",
        target_local_available=target_local_available,
        rationale=(
            "async 1d many-body td_local currently evaluates _TDLocalCtx.forces_full(ctx.r) and "
            "slices the result to zone target ids"
        ),
    )
