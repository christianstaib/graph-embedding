import networkx as nx
import numpy as np
from ogb.graphproppred import GraphPropPredDataset
from tqdm import tqdm


class OgbDataHelper:
    def __init__(self) -> None:
        pass

    def graph_dict_to_nx_graph(self, graph_dict: dict):
        """
        Constructs a NetworkX graph object from the given graph dictionary.

        Args:
            graph_dict: A dictionary representing a graph, with the following
                keys:
                    - 'num_nodes': The number of nodes in the graph.
                    - 'node_feat': A list of node features, with one feature vector per node.
                    - 'edge_index': An edge index array of shape (2, E), where E is the number of edges.
                    - 'edge_feat': A list of edge features, with one feature vector per edge.

        Returns:
            A NetworkX graph object with nodes and edges corresponding to the input graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(graph_dict['num_nodes']))

        for node_number, features in enumerate(graph_dict['node_feat']):
            graph.add_node(node_number, feature=features)

        # change shape from (2, E) to (E, 2)
        edge_list = np.transpose(graph_dict['edge_index'])
        for node_number, (from_node, to_node) in enumerate(edge_list):
            features = graph_dict['edge_feat'][node_number]
            graph.add_edge(from_node, to_node, feature=features)

        return graph

    def get_processed_dataset(self, dataset_name: str):
        """Processes a graph property prediction dataset.

        Args:   
            dataset_name (str): The name of the dataset to process.

        Returns:
            tuple: A tuple containing the following elements:
                - features (list): A list of NetworkX graphs representing the features of the dataset.
                - labels (list): A list of labels corresponding to the features.
                - split_idx (dict): A dictionary containing the train/test split indices for the dataset.
        """
        dataset = GraphPropPredDataset(name=dataset_name)
        split_idx = dataset.get_idx_split()

        features, labels = [], []
        for graph_dict, label in tqdm(dataset):
            features.append(self.graph_dict_to_nx_graph(graph_dict))
            labels.append(label)

        return features, labels, split_idx
