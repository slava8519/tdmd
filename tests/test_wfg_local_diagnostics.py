from __future__ import annotations

import numpy as np

from tdmd.td_automaton import TDAutomaton1W, ZoneRuntime, ZoneType
from tdmd.zone_bins_localz import PersistentZoneLocalZBinsCache


def _mk_zone(zid: int) -> ZoneRuntime:
    return ZoneRuntime(
        zid=zid, z0=0.0, z1=1.0, ztype=ZoneType.D, atom_ids=np.empty((0,), np.int32), n_cells=1
    )


def test_wfg_local_cycle_detect():
    zones = [_mk_zone(0), _mk_zone(1), _mk_zone(2)]
    cache = PersistentZoneLocalZBinsCache()
    autom = TDAutomaton1W(
        zones, box=10.0, cutoff=2.5, bins_cache=cache, traversal_order=[0, 1, 2], formal_core=True
    )

    ready = {0: False, 1: False, 2: True}
    autom.deps_table_func = lambda zid: [1] if zid == 0 else ([0] if zid == 1 else [])
    autom.deps_table_pred = lambda did: bool(ready[int(did)])

    g = autom.wfg_local_sample()
    assert g[0] == [1]
    assert g[1] == [0]
    cyc = autom.wfg_local_cycle()
    assert cyc is not None
    assert cyc[0] == cyc[-1]
