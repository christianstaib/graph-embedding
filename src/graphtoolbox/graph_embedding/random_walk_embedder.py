from typing import List

import networkx as nx
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.estimator import Estimator

from graphtoolbox.random_walkers import WalksToStringHelper, RandomWalker


class RandomWalkEmbedder(Estimator):
    def __init__(self, attributed):
        self.attributed = attributed

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        walks_to_string_helper = WalksToStringHelper()
        random_walker = RandomWalker()

        documents = []

        for i, graph in enumerate(graphs):
            replace_dict = walks_to_string_helper.get_replace_dict(
                graph, self.attributed)
            walks = random_walker.random_walks(
                graph, 5*graph.number_of_nodes())
            document = walks_to_string_helper.get_document(walks, replace_dict)
            documents.append(document)

        documents = [
            TaggedDocument(words=doc, tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            documents,
            vector_size=250,
            epochs=100
        )

        self._embedding = [
            self.model.docvecs[str(i)]
            for i, _ in enumerate(documents)
        ]

    def get_embedding(self) -> np.array:
        return np.array(self._embedding)
