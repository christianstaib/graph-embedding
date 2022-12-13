# %%
import pandas as pd
import numpy as np
import networkx as nx
import random
import re

from tqdm import tqdm

from ogb.graphproppred import GraphPropPredDataset

# %%
d_name = 'ogbg-molpcba'
dataset = GraphPropPredDataset(name=d_name)
split_idx = dataset.get_idx_split()

X = [data[0] for data in dataset]
y = [data[1] for data in dataset]


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


def edges_to_nodes(
    graph: nx.Graph
) -> nx.Graph:
    '''
    Generates a new graph where each edge (u, v) is replaced with two edges (u, uv), (uv, v) and a new node uv.
    The attributes of the edge are copied to the new edge.
    This method may help if a graph embedding does not not support edge attributes.
    '''
    new_graph = nx.Graph()
    max_node = max(list(graph.nodes))

    # copy nodes
    for node in nx.nodes(graph):
        new_graph.add_node(node)

        # copy node attributes
        features = graph.nodes[node]
        new_graph.nodes[node].update(features)

    # add new nodes and their edges
    for u, v in nx.edges(graph):
        # new nodes are numbered starting from max(list(graph.nodes)) + 1
        max_node += 1

        # replace edge (u, v) with edges (u, max_node), (max_node, v) and node max_node
        new_graph.add_node(max_node)
        new_graph.add_edge(u, max_node)
        new_graph.add_edge(max_node, v)

        # copy edge attributes from (u, v) to node max_node
        features = graph.edges[u, v]
        new_graph.nodes[max_node].update(features)

    return new_graph


X = [get_nx_graph(x) for x in tqdm(X)]

# %%


def one_random_walk(graph, max_length):
    walk = []
    walk.append(random.choice(range(graph.number_of_nodes())))

    while len(walk) < max_length:
        neighborhood = list(graph.neighbors(walk[-1]))
        neighborhood = list(set(neighborhood) - set(walk))
        if neighborhood:
            walk.append(random.choice(neighborhood))
        else:
            break

    return walk

# %%


def random_walks(graph):
    num_walks = graph.number_of_nodes()
    walks = []

    for _ in range(num_walks):
        walks.append(one_random_walk(graph, 9999))

    return walks

# %%


def walks_to_string(walks):
    the_string = ' END '.join(
        [' '.join(['_' + str(num) + '_' for num in walk]) for walk in walks])

    return the_string


# %%
walks = random_walks(X[0])

walks

# %%


def get_replace_dict(graph):
    replace_dict = dict()

    for node in range(graph.number_of_nodes()):
        replace_dict['_' + str(node) + '_'] = ','.join([str(num)
                                                        for num in graph.nodes[node]['feature']])

    return replace_dict


# %%
get_replace_dict(X[0])

# %%
the_string = walks_to_string(walks)

print(the_string)

# %%


def get_paragraph(graph):
    walks = random_walks(graph)
    the_string = walks_to_string(walks)
    replace_dict = get_replace_dict(graph)

    pattern = '|'.join(sorted(re.escape(k) for k in replace_dict))

    the_better_string = re.sub(pattern, lambda m: replace_dict.get(
        m.group(0).upper()), the_string, flags=re.IGNORECASE)

    return (the_better_string)


# %%
def get_document(graphs):
    return '\n'.join([get_paragraph(graph) for graph in graphs])


# %%
document = get_document(X)

# %%
print(get_paragraph(X[0]))

# %%
# open text file
text_file = open(f"{d_name}.txt", "w")

# write string to file
n = text_file.write(document)

# close file
text_file.close()
