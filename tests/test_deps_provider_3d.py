from tdmd.deps_provider_3d import DepsProvider3DBlock


def _provider():
    return DepsProvider3DBlock(
        nx=2,
        ny=2,
        nz=2,
        box=(10.0, 10.0, 10.0),
        cutoff=0.1,
        mpi_size=4,
    )


def test_deps_provider3d_no_self_deps():
    p = _provider()
    for zid in range(p.nx * p.ny * p.nz):
        assert zid not in p.deps_table(zid)
        assert zid not in p.deps_owner(zid)


def test_deps_provider3d_sanity_small_grid():
    p = _provider()
    n = p.nx * p.ny * p.nz
    for zid in range(n):
        deps = p.deps_table(zid)
        assert len(deps) == len(set(deps))
        assert all(0 <= d < n for d in deps)
        assert len(deps) <= n - 1


def test_deps_provider3d_owner_rank_stability():
    p = _provider()
    for zid in range(p.nx * p.ny * p.nz):
        a = p.owner_rank(zid)
        b = p.owner_rank(zid)
        assert a == b
        assert 0 <= a < p.mpi_size
        assert a == zid % p.mpi_size
