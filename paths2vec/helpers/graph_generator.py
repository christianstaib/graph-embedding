import numpy as np
import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm


class GraphGenerator:
    def __init__(self):
        pass

    def ogb_dataset_to_graphs(self, dataset):
        print("get graphs")
        graphs = []
        with Pool(4) as pool:
            for graph in tqdm(
                pool.imap(self.single_ogb_dict_to_graph, dataset),
                total=len(dataset),
            ):
                graphs.append(graph)

        return graphs

    def single_ogb_dict_to_graph(self, data):
        graph_dict, _ = data
        graph = nx.Graph()
        graph.add_nodes_from(range(graph_dict["num_nodes"]))

        for node_number, features in enumerate(graph_dict["node_feat"]):
            graph.add_node(node_number, feature=features)

        # change shape from (2, E) to (E, 2)
        edge_list = np.transpose(graph_dict["edge_index"])
        for node_number, (from_node, to_node) in enumerate(edge_list):
            features = graph_dict["edge_feat"][node_number]
            graph.add_edge(from_node, to_node, feature=features)

        return graph
