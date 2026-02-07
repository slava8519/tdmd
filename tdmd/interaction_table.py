from __future__ import annotations
from typing import Protocol
import numpy as np

class InteractionTable(Protocol):
    """Абстракция «таблицы взаимодействий» зоны (в терминах диссертации).

    Требования формального ядра:
    - таблица строится на множестве кандидатов (atoms in P-neighborhood),
    - таблица должна иметь *support set* — атомы, входящие в таблицу.
      Это нужно для правила TD «не передавать атомы, входящие в таблицу следующей зоны».
    """
    candidate_ids: np.ndarray  # atoms that participate in the table (support)

    def support_ids(self) -> np.ndarray: ...
