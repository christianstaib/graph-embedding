import networkx as nx
import numpy as np
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec
from ogb.graphproppred import GraphPropPredDataset, Evaluator
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import random


def graph_dict_to_nx_graph(
    graph_dict: dict
):
    graph = nx.Graph()
    graph.add_nodes_from(range(graph_dict['num_nodes']))

    for node_number, features in enumerate(graph_dict['node_feat']):
        graph.add_node(node_number, atomic_num=features[0])

    # change shape from (2, E) to (E, 2)
    edge_list = np.transpose(graph_dict['edge_index'])
    for node_number, (from_node, to_node) in enumerate(edge_list):
        features = graph_dict['edge_feat'][node_number]
        graph.add_edge(from_node, to_node, bond_type=features[0])

    subgraph_hashes = nx.weisfeiler_lehman_subgraph_hashes(graph, iterations=1, node_attr="atomic_num", edge_attr="bond_type")

    for node, list_hash in subgraph_hashes.items():
        graph.nodes[node]["atomic_num"] = list_hash[-1]

    return graph


def get_shortest_paths(graph):
    shortest_paths = []

    shortest_path_dict = dict(nx.all_pairs_shortest_path(graph))

    for _ in range(100):
        source = random.choice(list(graph.nodes))
        target = random.choice(list(graph.nodes))
        if nx.has_path(graph, source, target):
            shortest_path = shortest_path_dict[source][target]
            if len(shortest_path) >= 2:
                shortest_paths.append(shortest_path)

    return shortest_paths


def substitute_nodes_with_features(shortest_path, graph):
    feat_to_include = [0]
    path_str = []
    for node_id in shortest_path:
        for j, feat in enumerate(feat_to_include):
            path_str.append(f'{j}_{graph.nodes[node_id]["atomic_num"]}')
    path_str = ' '.join(path_str)
    return path_str


def write_paths_to_file(dataset):
    start_line = 1
    line = 1
    with open('rows.cor', 'w') as f_rows, open('shortest_paths.cor', 'w') as f_paths:
        for i, (graph_dict, label) in enumerate(tqdm(dataset)):
            graph = graph_dict_to_nx_graph(graph_dict)
            shortest_paths = get_shortest_paths(graph)
            for shortest_path in shortest_paths:
                path_str = substitute_nodes_with_features(shortest_path, graph)
                f_paths.write(path_str + " STOP ")
                line += 1
            f_paths.write("\n")
            f_rows.write(str(start_line) + " " + str(line-1) + '\n')
            start_line = line


dataset_name = 'ogbg-molfreesolv'
dataset = GraphPropPredDataset(name=dataset_name)
split_idx = dataset.get_idx_split()

write_paths_to_file(dataset)

model = Doc2Vec(corpus_file='shortest_paths.cor')

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
scatter_plot = sns.scatterplot(x=predicted_valid_y, y=valid_y)
scatter_plot.set(
    xlabel="Predicted Values",
    ylabel="Actual Values",
    title=dataset_name
)
scatter_plot.figure.savefig('test.png')

evaluator = Evaluator(name=dataset_name)
input_dict = {
    "y_true": valid_y.reshape((-1, 1)),
    "y_pred": predicted_valid_y.reshape((-1, 1))
}
result_dict = evaluator.eval(input_dict)

print(result_dict)