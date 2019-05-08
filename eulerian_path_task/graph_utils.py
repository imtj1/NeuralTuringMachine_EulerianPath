from operator import itemgetter

import networkx as nx

from collections import deque


def is_semieulerian(G):
    """Returns ``True`` if and only if ``G`` has an Eulerian path.

    An Eulerian path is a path that crosses every edge in G
    exactly once.

    Parameters
    ----------
    G: NetworkX Graph, DiGraph, MultiGraph or MultiDiGraph
        A directed or undirected Graph or MultiGraph.

    Returns
    -------
    True, False

    See Also
    --------
    is_eulerian()

    Examples
    --------
    >>> G = nx.DiGraph([(1,2),(2,3)])
    >>> nx.is_semieulerian(G)
    True
    """
    is_directed = G.is_directed()

    # Verify that graph is connected, short circuit
    if is_directed and not nx.is_weakly_connected(G):
        return False

    # is undirected
    if not is_directed and not nx.is_connected(G):
        return False

    # Not all vertex have even degree, check if exactly two vertex
    # have odd degrees.  If yes, then there is an Euler path. If not,
    # raise an error (no Euler path can be found)

    # if the odd condition is not meet, raise an error.
    start = _find_path_start(G)
    if not start:
        return False

    return True


def _find_path_start(G):
    """Returns a suitable starting vertex for Eulerian path

    The function also verifies that the graph contains a path. If no
    path exists, returns ``None''.

    """
    is_directed = G.is_directed()

    # list to check the odd degree condition
    check_odd = []

    if is_directed:
        degree = G.in_degree
        out_degree = G.out_degree
    else:
        degree = G.degree

    # Verify if an Euler path can be found. Complexity O(|V|)
    for v in G:
        deg = degree(v)
        # directed case
        if is_directed:
            outdeg = out_degree(v)
            if deg != outdeg:
                # if we have more than 2 odd nodes no Euler path exists
                if len(check_odd) > 2:
                    return False
                # is odd and we append it.
                check_odd.append(v)
        # undirected case
        else:
            if deg % 2 != 0:
                # if we have more than 2 odd nodes no Euler path exists
                if len(check_odd) > 2:
                    return False
                # is odd and we append it.
                check_odd.append(v)
    start = None
    if is_directed:

        first = check_odd[0]
        second = check_odd[1]
        if G.out_degree(first) == G.in_degree(first) + 1 and \
                G.in_degree(second) == G.out_degree(second) + 1:
            start = second
        elif G.out_degree(second) == G.in_degree(second) + 1 and \
                G.in_degree(first) == G.out_degree(first) + 1:
            start = first
        else:
            start = None

    else:
        if len(check_odd) > 0:
            start = check_odd[0]

    return start


def eulerian_path(G):
    """Return a generator of the edges of an Eulerian path in ``G``.

    Check if the graph ``G`` has an Eulerian path and return a
    generator for the edges. If no path is available, raise an error.

    Parameters
    ----------
    G: NetworkX Graph, DiGraph, MultiGraph or MultiDiGraph
        A directed or undirected Graph or MultiGraph.

    Returns
    -------
    edges: generator
        A generator that produces edges in the Eulerian path.

    Raises
    ------
    NetworkXError: If the graph does not have an Eulerian path.

    Examples
    --------
    >>> G = nx.Graph([('W', 'N'), ('N', 'E'), ('E', 'W'), ('W', 'S'), ('S', 'E')])
    >>> len(list(nx.eulerian_path(G)))
    5

    >>> G = nx.DiGraph([(1, 2), (2, 3)])
    >>> list(nx.eulerian_path(G))
    [(1, 2), (2, 3)]

    Notes
    -----
    Linear time algorithm, adapted from [1]_ and [3]_.
    Information about Euler paths in [2]_.
    Code for Eulerian circuit in [3]_.

    Important: In [1], Euler path is in reverse order,
    this implementation gives the path in correct order
    as in [3]_ for Eulerian_circuit. The distinction for
    directed graph is in using the in_degree neighbors, not the out
    ones. for undirected, it is using itemgetter(1) for get_vertex,
    which uses the correct vertex for this order. Also, every graph
    has an even number of odd vertices by the Handshaking Theorem [4]_.

    References
    ----------
    .. [1] http://www.graph-magics.com/articles/euler.php
    .. [2] http://en.wikipedia.org/wiki/Eulerian_path
    .. [3] https://github.com/networkx/networkx/blob/master/networkx/algorithms/euler.py
    .. [4] https://www.math.ku.edu/~jmartin/courses/math105-F11/Lectures/chapter5-part2.pdf

    """
    if not is_semieulerian(G):
        raise nx.NetworkXError("G does not have an Eulerian path.")

    g = G.__class__(G)  # copy graph structure (not attributes)
    is_directed = g.is_directed()

    if is_directed:
        degree = g.in_degree
        edges = g.in_edges
        get_vertex = itemgetter(0)
    else:
        degree = g.degree
        edges = g.edges
        get_vertex = itemgetter(1)

    # Begin algorithm:
    start = _find_path_start(g)
    if not start:
        raise nx.NetworkXError("G has no Eulerian path.")

    vertex_stack = deque([start])
    last_vertex = None

    while vertex_stack:

        current_vertex = vertex_stack[-1]
        # if no neighbors:
        if degree(current_vertex) == 0:
            # Special case, we cannot add a None vertex to the path.
            if last_vertex is not None:
                yield (last_vertex, current_vertex)
            last_vertex = current_vertex
            vertex_stack.pop()
        # we have neighbors, so add the vertex to the stack (2), take
        # any of its neighbors (1) remove the edge between selected
        # neighbor and that vertex, and set that neighbor as the
        # current vertex (4).
        else:
            edge = next(edges(current_vertex))
            vertex_stack.append(get_vertex(edge))
            g.remove_edge(*edge)


def findpath(graph):
    n = len(graph)
    numofadj = list()

    # Find out number of edges each vertex has
    for i in range(n):
        numofadj.append(sum(graph[i]))

        # Find out how many vertex has odd number edges
    startpoint = 0
    numofodd = 0
    for i in range(n - 1, -1, -1):
        if (numofadj[i] % 2 == 1):
            numofodd += 1
            startpoint = i

            # If number of vertex with odd number of edges
    # is greater than two return "No Solution".
    if (numofodd > 2):
        return []

        # If there is a path find the path
    # Initialize empty stack and path
    # take the starting current as discussed
    stack = list()
    path = list()
    cur = startpoint

    # Loop will run until there is element in the stack
    # or current edge has some neighbour.
    while (stack != [] or sum(graph[cur]) != 0):

        # If current node has not any neighbour
        # add it to path and pop stack
        # set new current to the popped element
        if (sum(graph[cur]) == 0):
            path.append(cur + 1)
            cur = stack.pop(-1)

            # If the current vertex has at least one
        # neighbour add the current vertex to stack,
        # remove the edge between them and set the
        # current to its neighbour.
        else:
            for i in range(n):
                if graph[cur][i] == 1:
                    stack.append(cur)
                    graph[cur][i] = 0
                    graph[i][cur] = 0
                    cur = i
                    break
    # print the path
    # for ele in path:
    #     print(ele, "-> ", end='')
    # print(cur + 1)
    path.append(cur + 1)
    return path
