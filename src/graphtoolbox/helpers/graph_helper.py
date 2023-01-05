import networkx as nx


class GraphHelper():
    """
    This is a class comment
    """

    def __init__(self) -> None:
        pass

    def edges_to_nodes(
        self,
        graph:      nx.Graph
    ) -> nx.Graph:
        """
        Generates a new graph where each edge (u, v) is replaced with two edges
        (u, uv), (uv, v) and a new node uv. The attributes of the edge are
        copied to the new edge. This method may help if a graph embedding does
        not not support edge attributes.
        """
        new_graph = nx.Graph()
        max_node = max(list(graph.nodes))

        for node in nx.nodes(graph):
            new_graph.add_node(node)

            features = graph.nodes[node]
            new_graph.nodes[node].update(features)

        for u, v in nx.edges(graph):
            max_node += 1

            new_graph.add_node(max_node)
            new_graph.add_edge(u, max_node)
            new_graph.add_edge(max_node, v)

            features = graph.edges[u, v]
            new_graph.nodes[max_node].update(features)

        return new_graph
