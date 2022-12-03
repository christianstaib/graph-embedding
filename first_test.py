from graph_embedding import SimpleEmbedder
import pandas as pd
import numpy as np
import networkx as nx

from ogb.graphproppred import GraphPropPredDataset

###

d_name = 'ogbg-molhiv'
dataset = GraphPropPredDataset(name=d_name)
split_idx = dataset.get_idx_split()

X = [data[0] for data in dataset]
y = [data[1] for data in dataset]

###

node_features = [
    'atomic_num_idx',
    'chirality_idx',
    'degree_idx',
    'formal_charge_idx',
    'num_h_idx',
    'number_radical_e_idx',
    'hybridization_idx',
    'is_aromatic_idx',
    'is_in_ring_idx'
]

edge_features = [
    'bond_type_idx',
    'bond_stereo_idx',
    'is_conjugated_idx'
]

###


def get_nx_graph(graph):
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

###


X = [get_nx_graph(x) for x in X]

###

train_X = np.array([X[index] for index in split_idx["train"]])
train_y = np.array([y[index] for index in split_idx["train"]])

valid_X = np.array([X[index] for index in split_idx["valid"]])
valid_y = np.array([y[index] for index in split_idx["valid"]])

test_X = np.array([X[index] for index in split_idx["test"]])
test_y = np.array([y[index] for index in split_idx["test"]])
