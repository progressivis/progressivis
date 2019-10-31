"""
Topological sorting of a DAG
"""

def _sort(graph, vertex, permanent, temporary, stack):
    if vertex in temporary:
        raise ValueError('Cycle in graph')
    temporary.add(vertex)
    for i in graph.get(vertex, []):
        if i not in permanent:
            _sort(graph, i, permanent, temporary, stack)
    permanent.add(vertex)
    stack.append(vertex)

def toposort(graph):
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
    permanent = set()
    temporary = set()
    stack = []
    for i in graph:
        if i not in permanent:
            _sort(graph, i, permanent, temporary, stack)
    return stack

if __name__ == "__main__":
    import doctest
    doctest.testmod()
