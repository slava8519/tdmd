from __future__ import annotations

from tdmd.graph_utils import find_cycle, is_acyclic


def test_cycle_detection_simple():
    adj = {"a": ["b"], "b": ["c"], "c": ["a"]}
    cyc = find_cycle(adj)
    assert cyc is not None
    assert cyc[0] == cyc[-1]
    assert set(cyc[:-1]) == {"a", "b", "c"}
    assert not is_acyclic(adj)


def test_cycle_detection_acyclic():
    adj = {"a": ["b", "c"], "b": ["c"], "c": []}
    assert find_cycle(adj) is None
    assert is_acyclic(adj)


def test_cycle_detection_self_loop():
    adj = {"a": ["a"]}
    cyc = find_cycle(adj)
    assert cyc is not None
    assert cyc == ["a", "a"]
