from gensim.models.doc2vec import Doc2Vec

from .epoch_logger import EpochLogger
from .path_generator import PathGenerator


class Paths2Vec:
    def __init__(self, cpu_count):
        self.cpu_count = cpu_count

    def fit(
        self,
        graphs,
        corpus_file,
        sample_size,
        window_in_nodes,
        vertex_feature_idx=[0],
        edge_feature_idx=[0],
    ):
        print("write paths to file")

        graph_to_path = PathGenerator(
            corpus_file=corpus_file, cpu_count=self.cpu_count, sample_size=sample_size
        )
        graph_to_path.paths_to_file(
            graphs=graphs,
            vertex_feature_idx=vertex_feature_idx,
            edge_feature_idx=edge_feature_idx,
        )

        print("get vectors")
        epoch_logger = EpochLogger(epochs=10)
        model = Doc2Vec(
            corpus_file=corpus_file,
            window=window_in_nodes * (len(vertex_feature_idx) + len(edge_feature_idx)),
            workers=self.cpu_count,
            callbacks=[epoch_logger],
        )

        return [model.dv[i] for i in range(len(graphs))]
