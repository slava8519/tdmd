from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InteractionTableState:
    """Формальная «таблица взаимодействий» как объект состояния зоны.

    Внутри может быть bins/cell-list/Verlet; для TD-правила важны:
    - support set: candidate_ids (атомы, входящие в таблицу),
    - параметры rc и геометрия p-окрестности (для валидности),
    - шаг времени (time stamp), когда таблица построена.
    """

    impl: object
    candidate_ids: np.ndarray
    rc: float
    build_step: int
    z0p: float
    z1p: float
    # v4.3: optional 3D support geometry (AABB of receiver zone, for static_3d)
    lo: np.ndarray | None = None  # (3,)
    hi: np.ndarray | None = None  # (3,)

    def support_ids(self) -> np.ndarray:
        return self.candidate_ids

    def has_aabb(self) -> bool:
        return (self.lo is not None) and (self.hi is not None)
