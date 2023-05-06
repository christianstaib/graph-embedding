from ogb.graphproppred import Evaluator, GraphPropPredDataset

from multiprocessing import Pool

import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import random

cpu_count = 16
sample_size = 10000

vertex_feature_names = [
    "atomic_num",  # 0
    "chirality",  # 1
    "degree",  # 2
    "formal_charge",  # 3
    "num_h",  # 4
    "num_rad_e",  # 5
    "hybridization",  # 6
    "is_aromatic",  # 7
    "is_in_ring",  # 8
    "handmade_degree",  # 9
]

vertex_feat_to_include = range(10)
edge_feat_to_include = range(3)


def graph_dict_to_nx_graph(graph_dict: dict):
    graph = nx.Graph()
    graph.add_nodes_from(range(graph_dict["num_nodes"]))

    for node_number, features in enumerate(graph_dict["node_feat"]):
        graph.add_node(node_number, feature=features)

    # change shape from (2, E) to (E, 2)
    edge_list = np.transpose(graph_dict["edge_index"])
    for node_number, (from_node, to_node) in enumerate(edge_list):
        features = graph_dict["edge_feat"][node_number]
        graph.add_edge(from_node, to_node, feature=features)

    # append node degree as feature
    for node in graph.nodes:
        graph.nodes[node]["feature"] = np.append(
            graph.nodes[node]["feature"], graph.degree[node]
        )

    return graph


def get_random_samples(input_list, num_samples):
    if len(input_list) > num_samples:
        return random.sample(input_list, num_samples)
    else:
        return input_list


def get_shortest_paths(graph):
    shortest_paths = []

    shortest_path_dict = dict(nx.all_pairs_shortest_path(graph))

    for source in graph.nodes:
        for target in graph.nodes:
            if nx.has_path(graph, source, target):
                shortest_path = shortest_path_dict[source][target]
                if len(shortest_path) >= 2:
                    shortest_paths.append(shortest_path)

    return get_random_samples(shortest_paths, sample_size)


def substitute_nodes_with_features(shortest_path, graph):
    path_str = []
    for position_in_walk, node_id in enumerate(shortest_path):
        for j, feat in enumerate(vertex_feat_to_include):
            path_str.append(f'v{j}_{graph.nodes[node_id]["feature"][feat]}')
        path_str.append("EOW")

        if position_in_walk != len(shortest_path) - 1:
            for j, feat in enumerate(edge_feat_to_include):
                path_str.append(
                    f'e{j}_{graph.edges[(shortest_path[position_in_walk], shortest_path[position_in_walk+1])]["feature"][feat]}'
                )
            path_str.append("EOW")

    path_str = " ".join(path_str)
    return path_str


def process_dict_label_tuple(dict_label_tuple):
    graph_dict, _ = dict_label_tuple
    graph = graph_dict_to_nx_graph(graph_dict)
    shortest_paths = get_shortest_paths(graph)

    paths = []
    for shortest_path in shortest_paths:
        path_str = substitute_nodes_with_features(shortest_path, graph)
        paths.append(path_str)

    return paths


def write_output(results):
    start_line = 1
    line = 1

    with open("data/rows.cor", "w") as f_rows, open(
        "data/shortest_paths.cor", "w"
    ) as f_paths:
        for paths in results:
            for path_str in paths:
                f_paths.write(path_str + "\n")
                line += 1
            f_paths.write("\n")
            f_rows.write(str(start_line) + " " + str(line - 1) + "\n")
            start_line = line


def write_paths_to_file(dataset, mode):
    start_line = 1
    line = 1

    with open("data/rows.cor", mode) as f_rows, open(
        "data/shortest_paths.cor", mode
    ) as f_paths:
        with Pool(cpu_count) as pool:
            for paths in pool.imap(process_dict_label_tuple, dataset):
                for path_str in paths:
                    f_paths.write(path_str + " EOS ")
                    line += 1
                f_paths.write("\n")
                f_rows.write(str(start_line) + " " + str(line - 1) + "\n")
                start_line = line


if __name__ == "__main__":
    results = []

    for _ in tqdm(range(3)):
        dataset_name = "ogbg-molfreesolv"
        dataset = GraphPropPredDataset(name=dataset_name)
        split_idx = dataset.get_idx_split()

        write_paths_to_file(dataset, "w")

        model = Doc2Vec(
            corpus_file="data/shortest_paths.cor",
            window=2 * (len(vertex_feat_to_include) + len(edge_feat_to_include)),
            workers=cpu_count,
        )

        train_X = np.array([model.dv[x] for x in split_idx["train"]])
        train_y = np.array([dataset[x][1] for x in split_idx["train"]]).ravel()

        test_X = np.array([model.dv[x] for x in split_idx["test"]])
        test_y = np.array([dataset[x][1] for x in split_idx["test"]]).ravel()

        valid_X = np.array([model.dv[x] for x in split_idx["valid"]])
        valid_y = np.array([dataset[x][1] for x in split_idx["valid"]]).ravel()

        reg = RandomForestRegressor()
        reg.fit(train_X, train_y)

        predicted_valid_y = reg.predict(valid_X)

        evaluator = Evaluator(name=dataset_name)
        input_dict = {
            "y_true": valid_y.reshape((-1, 1)),
            "y_pred": predicted_valid_y.reshape((-1, 1)),
        }
        result_dict = evaluator.eval(input_dict)

        results.append(result_dict["rmse"])

    print(f"{dataset_name}: all pair shortest path: {sum(results) / len(results)}")
    print(results)
