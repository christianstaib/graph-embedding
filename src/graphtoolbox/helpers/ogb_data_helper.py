import networkx as nx
import numpy as np


class OgbDataHelper:
    def __init__(self) -> None:
        pass

    def get_nx_graph(self, graph_dict: dict):
        """
        Constructs a NetworkX graph object from the given graph dictionary.

        Args:
            graph_dict: A dictionary representing a graph, with the following
            keys:
                - 'num_nodes': The number of nodes in the graph.
                - 'node_feat': A list of node features, with one feature vector
                  per node.
                - 'edge_index': An edge index array of shape (2, E), where E is
                  the number of edges.
                - 'edge_feat': A list of edge features, with one feature vector
                  per edge.

        Returns:
            A NetworkX graph object with nodes and edges corresponding to the
            input graph.
        """
        # Initialize a Graph object from NetworkX
        graph = nx.Graph()
        # Add nodes to the graph from 0 to `num_nodes - 1`
        graph.add_nodes_from(range(graph_dict['num_nodes']))

        # Iterate through the node feature list and add the features to each
        # node
        for node_number, features in enumerate(graph_dict['node_feat']):
            graph.add_node(node_number, feature=features)

        # Transpose the edge index array and iterate through it
        edge_list = np.transpose(graph_dict['edge_index'])
        for node_number, (from_node, to_node) in enumerate(edge_list):
            # Add an edge between the `from_node` and `to_node` with the
            # corresponding edge features
            features = graph_dict['edge_feat'][node_number]
            graph.add_edge(from_node, to_node, feature=features)

        # Return the final graph
        return graph

    def dummy_function(self):
        return 1
