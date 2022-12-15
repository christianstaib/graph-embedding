import networkx as nx
import numpy as np


class OgbDataHelper:
    def __init__(self) -> None:
        pass

    def get_nx_graph(self, graph):
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

    def dummy_function(self):
        return 1
