from multiprocessing import Pool
from tqdm import tqdm

import networkx as nx
from itertools import product
from random import sample, choice


class PathGenerator:
    def __init__(self, corpus_file, cpu_count, sample_size=None) -> None:
        self.corpus_file = corpus_file
        self.cpu_count = cpu_count
        self.sample_size = sample_size

    def paths_to_file(self, graphs, vertex_feature_idx, edge_feature_idx):
        self.vertex_feature_idx = vertex_feature_idx
        self.edge_feature_idx = edge_feature_idx
        with open(self.corpus_file, "w") as f_paths:
            with Pool(self.cpu_count) as pool:
                for paths in tqdm(
                    pool.imap(self.graph_to_paths, graphs),
                    total=len(graphs),
                ):
                    for path_str in paths:
                        f_paths.write(path_str + " EOS ")
                    f_paths.write("\n")

    def random_sample(self, input_list, num_samples=None):
        if num_samples == None:
            return input_list
        elif len(input_list) > num_samples:
            return sample(input_list, num_samples)
        else:
            return input_list

    def get_paths(self, graph):
        paths = []

        i = 0

        while True:
            source = choice(range(len(list(graph.nodes))))
            target = choice(range(len(list(graph.nodes))))
            if nx.has_path(graph, source, target):
                path = nx.shortest_path(graph, source=source, target=target)
                if len(path) >= 2 and path not in paths:
                    paths.append(path)
            i += 1
            if i >= self.sample_size:
                break

        return paths

    def substitute(self, path, graph):
        path_str = []
        for node_index, node_id in enumerate(path):
            node = graph.nodes[node_id]
            for i, feature in enumerate(node["feature"][self.vertex_feature_idx]):
                path_str.append(f"v{i}_{feature}")
            # path_str.append("EOW")

            if node_index != len(path) - 1:
                edge = graph.edges[(path[node_index], path[node_index + 1])]
                for i, feature in enumerate(edge["feature"][self.edge_feature_idx]):
                    path_str.append(f"e{i}_{feature}")
                # path_str.append("EOW")

        path_str = " ".join(path_str)
        return path_str

    def graph_to_paths(self, graph):
        shortest_paths = self.get_paths(graph)

        paths = []
        for shortest_path in shortest_paths:
            path_str = self.substitute(shortest_path, graph)
            paths.append(path_str)

        return paths
