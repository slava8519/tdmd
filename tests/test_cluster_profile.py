from __future__ import annotations

from tdmd.cluster_profile import load_cluster_profile


def test_load_cluster_profile_smoke():
    p = load_cluster_profile("examples/cluster/cluster_profile_smoke.yaml")
    assert p.profile_version == 1
    assert p.name == "smoke_local_ci"
    assert p.runtime.allow_simulated_cluster is True
    assert p.runtime.prefer_simulated is True
    assert p.scaling.strong_ranks == [2, 4]
    assert p.stability.ranks == [2, 4]
    assert len(p.transport_matrix.entries) >= 2
