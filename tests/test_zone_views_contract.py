import numpy as np
from tdmd.zone_views import build_zone_view, build_halo_view

def test_zone_views_contiguous_and_dtypes():
    x = np.arange(5, dtype=np.float32)
    y = np.arange(5, dtype=np.float32)
    z = np.arange(5, dtype=np.float32)
    t = np.arange(5, dtype=np.int64)
    i = np.arange(5, dtype=np.int64)

    zv = build_zone_view(x=x, y=y, z=z, type=t, id=i)
    assert zv.x.dtype == np.float64 and zv.x.flags["C_CONTIGUOUS"]
    assert zv.type.dtype == np.int32 and zv.type.flags["C_CONTIGUOUS"]

    hv = build_halo_view(x=x, y=y, z=z, type=t, id=i)
    assert hv.id.dtype == np.int32 and hv.id.flags["C_CONTIGUOUS"]
