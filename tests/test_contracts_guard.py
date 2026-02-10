import inspect
import re

import tdmd.td_full_mpi as td_full_mpi


def test_contract_w_leq_1_guard_present():
    """Regression tripwire against accidental removal of W<=1 enforcement markers."""
    src = inspect.getsource(td_full_mpi)
    patterns = [
        r"W<=1",
        r"violW",
        r"one\s*-?zone\s+.*W",
        r"\bmax_w\b",
    ]
    assert any(
        re.search(p, src) for p in patterns
    ), "Expected a W<=1 enforcement marker in td_full_mpi"


def test_contract_a4b_tie_break_present():
    """Regression tripwire: ensure A4b / tie-break logic markers still exist."""
    src = inspect.getsource(td_full_mpi)
    tokens = ["A4b", "tie", "priority", "owner_rank", "zone_id", "total order"]
    assert any(tok in src for tok in tokens), "Expected A4b tie-break markers in td_full_mpi"
