from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def find_cycle(adj: dict[T, Iterable[T]]) -> list[T] | None:
    """Return one directed cycle as a list of nodes [v0, v1, ..., v0], or None if acyclic."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[T, int] = {}
    parent: dict[T, T] = {}

    def dfs(u: T) -> list[T] | None:
        color[u] = GRAY
        for v in adj.get(u, []):
            c = color.get(v, WHITE)
            if c == WHITE:
                parent[v] = u
                cyc = dfs(v)
                if cyc is not None:
                    return cyc
            elif c == GRAY:
                # found back-edge u->v; reconstruct cycle
                if u == v:
                    return [v, v]
                cycle = [v, u]
                cur = u
                while cur != v:
                    cur = parent[cur]
                    cycle.append(cur)
                cycle.reverse()
                cycle.append(cycle[0])
                return cycle
        color[u] = BLACK
        return None

    # ensure all nodes included
    nodes = set(adj.keys())
    for _u, vs in adj.items():
        for v in vs:
            nodes.add(v)

    for n in list(nodes):
        if color.get(n, WHITE) == WHITE:
            cyc = dfs(n)
            if cyc is not None:
                return cyc
    return None


def is_acyclic(adj: dict[T, Iterable[T]]) -> bool:
    return find_cycle(adj) is None
