"""
Topological sorting of a DAG
"""
from __future__ import annotations


from typing import (
    List,
    Dict,
    Set
)

Vertex = str
Graph = Dict[Vertex, Set[Vertex]]


def _sort(graph: Graph,
          vertex: Vertex,
          permanent: Set[Vertex],
          temporary: Set[Vertex],
          stack: List[Vertex]):
    if vertex in temporary:
        raise ValueError('Cycle in graph')
    temporary.add(vertex)
    for i in graph.get(vertex, []):
        if i not in permanent:
            _sort(graph, i, permanent, temporary, stack)
    permanent.add(vertex)
    stack.append(vertex)


def toposort(graph: Graph):
    """
    Perform the sorting and returns the element in order.

    >>> toposort({'a': ['b'], 'b': ['c']})
    ['c', 'b', 'a']

    >>> toposort({'a': ['b', 'c'], 'b': ['c']})
    ['c', 'b', 'a']

    >>> toposort({'a': ['b'], 'b': ['c', 'd']})
    ['c', 'd', 'b', 'a']

    >>> toposort({'a': ['b'], 'c': ['d']})
    ['b', 'a', 'd', 'c']

    >>> toposort({'a': ['b'], 'b': ['a']})
    Traceback (most recent call last):
      ...
    ValueError: Cycle in graph
    """
    permanent: Set[Vertex] = set()
    temporary: Set[Vertex] = set()
    stack: List[Vertex] = []
    for i in graph:
        if i not in permanent:
            _sort(graph, i, permanent, temporary, stack)
    return stack


if __name__ == "__main__":
    import doctest
    doctest.testmod()
