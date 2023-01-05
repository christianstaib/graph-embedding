from typing import List

import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.estimator import Estimator
from tqdm import tqdm

from graphtoolbox.random_walkers import WalksToStringHelper, RandomWalker

import logging


class RandomWalkEmbedder(Estimator):
    def __init__(self):
        pass

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        logging.info("Generating random walks")
        documents = self.generate_random_walks(graphs)

        logging.info("Fitting Doc2Vec model")
        self.model = Doc2Vec(
            documents
        )

        logging.info("Generating embeddings")
        self._embedding = [
            self.model.docvecs[str(i)]
            for i, _ in enumerate(tqdm(documents))
        ]

    def generate_random_walks(self, graphs):
        documents = []

        for i, graph in enumerate(tqdm(graphs)):
            document = self.generate_random_walk(graph)
            documents.append([i, document])

        return documents

    def generate_random_walk(self, graph):
        walks_to_string_helper = WalksToStringHelper()
        random_walker = RandomWalker()
        
        replace_dict = walks_to_string_helper.get_replace_dict(graph)
        walks = random_walker.random_walks(graph, 5*graph.number_of_nodes())
        document = walks_to_string_helper.get_documents(walks, replace_dict)

        return document

    def get_embedding(self) -> np.array:
        return np.array(self._embedding)
