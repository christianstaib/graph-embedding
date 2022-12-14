Terms and definitions
=====================

A graph is a tuple :math:`G = (V, E, A_V, A_E)`. :math:`V` is the set of vertices. :math:`E` is the set of edges. We don't differentiate between directed and undirected between graphs but expect that there is at most one edge between two vertices. :math:`A_V \colon `, :math:`A_E` are the attributes of the vertices and edges. 

For now we limit the allowed attributes to a list of numbers and strings.

A `path` in a graph is a list of vertices where two adjacent vertices in the list are also adjacent in the graph. We define the `length` of a path as the length of the list of vertices.

A `random walk` is a `path` that is build with a random selection of vertices. Random doesn't necessarily mean equal distributed.
