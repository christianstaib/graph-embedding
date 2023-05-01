import networkx as nx
import numpy as np
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec
from ogb.graphproppred import GraphPropPredDataset, Evaluator
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import random


vertex_feat_to_include = range(9)
edge_feat_to_include = range(3)


def graph_dict_to_nx_graph(
    graph_dict: dict
):
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


def get_shortest_paths(graph):
    shortest_paths = []

    shortest_path_dict = dict(nx.all_pairs_shortest_path(graph))

    for source in graph.nodes:
        for target in graph.nodes:
            if nx.has_path(graph, source, target):
                shortest_path = shortest_path_dict[source][target]
                if len(shortest_path) >= 2:
                    shortest_paths.append(shortest_path)

    return shortest_paths


def substitute_nodes_with_features(shortest_path, graph):
    path_str = []
    for node_id in shortest_path:
        for j, feat in enumerate(vertex_feat_to_include):
            path_str.append(f'{j}_{graph.nodes[node_id]["feature"][feat]}')
        path_str.append('EOW')
    path_str = ' '.join(path_str)
    return path_str


def write_paths_to_file(dataset, dataset_name):
    start_line = 1
    line = 1
    with open(f'data/rows.cor', 'w') as f_rows, open(f'data/shortest_paths.cor', 'w') as f_paths:
        for i, (graph_dict, label) in enumerate(dataset):
            graph = graph_dict_to_nx_graph(graph_dict)
            shortest_paths = get_shortest_paths(graph)
            for shortest_path in shortest_paths:
                path_str = substitute_nodes_with_features(shortest_path, graph)
                f_paths.write(path_str + "\n")
                line += 1
            f_paths.write("\n")
            f_rows.write(str(start_line) + " " + str(line-1) + '\n')
            start_line = line


results = []

for _ in tqdm(range(1)):
    dataset_name = 'ogbg-molfreesolv'
    dataset = GraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()

    write_paths_to_file(dataset, dataset_name)

    model = Doc2Vec(corpus_file='data/shortest_paths.cor',
                    window=5*len(vertex_feat_to_include))

    train_X = np.array([model.dv[x] for x in split_idx['train']])
    train_y = np.array([dataset[x][1] for x in split_idx['train']]).ravel()

    test_X = np.array([model.dv[x] for x in split_idx['test']])
    test_y = np.array([dataset[x][1] for x in split_idx['test']]).ravel()

    valid_X = np.array([model.dv[x] for x in split_idx['valid']])
    valid_y = np.array([dataset[x][1] for x in split_idx['valid']]).ravel()

    train_X.shape, train_y.shape

    reg = RandomForestRegressor()
    reg.fit(train_X, train_y)

    predicted_valid_y = reg.predict(valid_X)

    evaluator = Evaluator(name=dataset_name)
    input_dict = {
        "y_true": valid_y.reshape((-1, 1)),
        "y_pred": predicted_valid_y.reshape((-1, 1))
    }
    result_dict = evaluator.eval(input_dict)

    results.append(result_dict["rmse"])

print('all pair shortest path:' + str(sum(results)/len(results)))
print(results)
