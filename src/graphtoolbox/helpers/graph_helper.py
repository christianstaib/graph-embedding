import networkx as nx
import numpy as np


class GraphHelper():
    """
    This is a class comment
    """

    def __init__(self) -> None:
        pass

    def get_nx_graph(self, graph):
        """
        This is a method comment
        """
        G = nx.Graph()
        G.add_nodes_from(range(graph['num_nodes']))

        node_list = graph['node_feat']
        for node_number, features in enumerate(node_list):
            G.add_node(node_number, feature=features)

        edge_list = np.transpose(graph['edge_index'])
        for node_number, (u, v) in enumerate(edge_list):
            features = graph['edge_feat'][node_number]
            G.add_edge(u, v, feature=features)

        return G

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
