import os
import random
from multiprocessing import Pool

import numpy as np
from gensim.models.doc2vec import Doc2Vec
from imblearn.over_sampling import RandomOverSampler
from ogb.graphproppred import Evaluator, GraphPropPredDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

N = 1
MAX_WALK_LENGTH = 3
WORKERS = len(os.sched_getaffinity(0))


def get_neighborhood(graph_dict, node):
    neighborhood = {
        'nodes': [],
        'edges': []
    }

    from_nodes = graph_dict['edge_index'][0]
    to_nodes = graph_dict['edge_index'][1]

    for index, from_node in enumerate(from_nodes):
        if from_nodes[index] == node:
            neighborhood['nodes'].append(to_nodes[index])
            neighborhood['edges'].append(index)

    return neighborhood


def get_node_attributes(graph_dict, node):
    return [f'v{i}_{v}'for i, v in enumerate(graph_dict['node_feat'][node])]


def get_edge_attributes(graph_dict, edge):
    return [f'e{i}_{v}'for i, v in enumerate(graph_dict['edge_feat'][edge])]


def random_walk_evenly_distributed(graph_dict, how_often_visited):
    walk = {
        'vertices': [],
        'edges': []
    }

    # initialize with invalid value to make sure that the first vertex is not removed
    previous_vertex = -1

    # choose random start vertex
    least_visited_vertices = []
    for i, x in enumerate(how_often_visited):
        if x == min(how_often_visited):
            least_visited_vertices.append(i)
    vertex = random.choice(least_visited_vertices)
    how_often_visited[vertex] += 1

    # append the first vertex to the walk
    walk['vertices'].append(vertex)

    for _ in range(MAX_WALK_LENGTH):
        neighborhood = get_neighborhood(graph_dict, vertex)

        # remove previous vertex from neighborhood
        if previous_vertex in neighborhood['nodes']:
            index_to_remove = neighborhood['nodes'].index(previous_vertex)
            neighborhood['nodes'].pop(index_to_remove)
            neighborhood['edges'].pop(index_to_remove)
        previous_vertex = vertex

        # if there are no more neighbors, stop
        if len(neighborhood['nodes']) == 0:
            break

        # choose the next vertex to visit
        edge = random.choice(neighborhood['edges'])
        vertex = graph_dict['edge_index'][1][edge]
        how_often_visited[vertex] += 1

        # append the next vertex and edge to the walk
        walk['edges'].append(edge)
        walk['vertices'].append(vertex)

    return walk, how_often_visited


def walk_to_list_of_attributes(graph_dict, walk):
    list = []

    for i in range(len(walk['edges'])):
        list += get_node_attributes(graph_dict, walk['vertices'][i])
        list += get_edge_attributes(graph_dict, walk['edges'][i])

    list += get_node_attributes(graph_dict, walk['vertices'][-1])

    return list


def walk_to_words(graph_dict, walk):
    list_of_attributes = walk_to_list_of_attributes(graph_dict, walk)
    words = [attributes for attributes in list_of_attributes]
    return words


def generate_words(graph_dict):
    how_often_visited = [0] * graph_dict['num_nodes']

    this_graphs_words = []
    for _ in range(N):
        walk, how_often_visited = random_walk_evenly_distributed(graph_dict, how_often_visited)
        this_graphs_words.append(walk_to_words(graph_dict, walk))
    return this_graphs_words


print('get dataset')

dataset_name = 'ogbg-molhiv'
dataset = GraphPropPredDataset(name=dataset_name)
split_idx = dataset.get_idx_split()

graph_dicts = [graph_dict for graph_dict, _ in tqdm(dataset)]
labels = [label for _, label in tqdm(dataset)]


print('generate words.cor')
words = []
with Pool() as pool:
    for this_graphs_words in tqdm(pool.imap(generate_words, graph_dicts), total=len(graph_dicts)):
        words.append(this_graphs_words)

with open('words.cor', 'w') as test_file:
    for sub_words in words:
        for sub_sub_words in sub_words:
            test_file.write(' '.join(sub_sub_words) + '\n')

print('train Doc2Vec')
model = Doc2Vec(corpus_file='words.cor', workers=WORKERS, window=15)

print('get X y')
X = np.array([sum([model.dv[i+(j*N)]/N for i in range(N)]) for j in range(len(dataset))])
X.shape

X_train = np.array([X[i] for i in split_idx['train']])
y_train = np.array([labels[i][0] for i in split_idx['train']])

X_test = np.array([X[i] for i in split_idx['test']])
y_test = np.array([labels[i][0] for i in split_idx['test']])


print('resample')
ros = RandomOverSampler(random_state=0)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

pos_idx = [i for i, v in enumerate(y_train) if v == 1]
neg_idx = [i for i, v in enumerate(y_train) if v == 0]
print(len(pos_idx), len(neg_idx))

resampled_pos_idx = [i for i, v in enumerate(y_train_resampled) if v == 1]
resampled_neg_idx = [i for i, v in enumerate(y_train_resampled) if v == 0]
print(len(resampled_pos_idx), len(resampled_neg_idx))

print('train classifier')
clf = RandomForestClassifier(n_jobs=WORKERS)
clf.fit(X_train_resampled, y_train_resampled.ravel())

print('get metric')
y_test_predicted = clf.predict(X_test)
y_test.shape, y_test_predicted.shape
roc_auc_score(y_test, y_test_predicted)
evaluator = Evaluator(name=dataset_name)
input_dict = {"y_true": y_test.reshape(-1, 1), "y_pred": y_test_predicted.reshape(-1, 1)}
result_dict = evaluator.eval(input_dict)
print(result_dict)
