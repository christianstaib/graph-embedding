import numpy as np
from ogb.graphproppred import Evaluator, GraphPropPredDataset
from sklearn.impute import SimpleImputer

from helpers import GraphGenerator
from paths2vec import Paths2Vec
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor


def get_paths2vec_X(
    dataset_name,
    graphs,
    cpu_count,
    sample_size,
    window_in_nodes,
    vertex_feature_idx,
    edge_feature_idx,
):
    # generate vectors for graphs
    corpus_file = f"corpus_files/{dataset_name}_paths.cor"
    paths2vec = Paths2Vec(cpu_count=cpu_count)
    X = paths2vec.fit(
        graphs=graphs,
        corpus_file=corpus_file,
        sample_size=sample_size,
        window_in_nodes=window_in_nodes,
        vertex_feature_idx=vertex_feature_idx,
        edge_feature_idx=edge_feature_idx,
    )

    return X


def get_random_X(
    dataset_name,
    graphs,
    cpu_count,
    sample_size,
    window_in_nodes,
    vertex_feature_idx,
    edge_feature_idx,
):
    X = [np.random.normal(size=100) for _ in range(len(graphs))]
    return X


class PipelineEvaluator:
    def __init__(
        self,
        cpu_count,
        window_in_nodes,
        sample_size,
        vertex_feature_idx,
        edge_feature_idx,
    ):
        self.cpu_count = cpu_count
        self.window_in_nodes = window_in_nodes
        self.sample_size = sample_size
        self.vertex_feature_idx = vertex_feature_idx
        self.edge_feature_idx = edge_feature_idx
        pass

    def get_result_dicts(self, X_func, dataset_name, num_runs, max_elem):
        result_dicts = {}
        result_dicts["train"] = []
        result_dicts["test"] = []
        result_dicts["valid"] = []

        for i in range(num_runs):
            print(f"starting run {i + 1} of {num_runs}")

            dataset = GraphPropPredDataset(name=dataset_name)

            if dataset.task_type == "binary classification":
                estimator = make_pipeline(
                    StandardScaler(), MultiOutputClassifier(SVC())
                )
            elif dataset.task_type == "regression":
                estimator = make_pipeline(
                    StandardScaler(), MultiOutputRegressor(SVR(kernel="linear"))
                )

            if max_elem == None:
                frac = 1
            elif len(dataset) > max_elem:
                frac = 1 / len(dataset) * max_elem
            else:
                frac = 1

            # get subset
            used_new_idx = []
            split_idx = dataset.get_idx_split()
            for name, idx in split_idx.items():
                random_sublist = random.sample(list(idx), int(len(idx) * frac))
                split_idx[name] = random_sublist
                used_new_idx.extend(random_sublist)
            used_new_idx.sort()

            # convert ogb dicts to networkx graphs
            dict_calculator = GraphGenerator()
            sub_dataset = [dataset[i] for i in used_new_idx]
            graphs = dict_calculator.ogb_dataset_to_graphs(dataset=sub_dataset)

            X = X_func(
                dataset_name,
                graphs,
                self.cpu_count,
                self.sample_size,
                self.window_in_nodes,
                self.vertex_feature_idx,
                self.edge_feature_idx,
            )

            # split data
            data = dict()
            for name, idx_list in split_idx.items():
                data[name] = dict()
                data[name]["X"] = np.array(
                    [X[used_new_idx.index(idx)] for idx in idx_list]
                )
                data[name]["y"] = np.array([dataset[idx][1] for idx in idx_list])

            # fit
            imp = SimpleImputer(strategy="most_frequent")
            y = imp.fit_transform(data["train"]["y"])
            estimator.fit(data["train"]["X"], y)

            for subset_name in ["train", "test", "valid"]:
                # predict
                prediction = estimator.predict(data[subset_name]["X"])
                data[subset_name]["y_predicted"] = prediction
                # evaluate
                evaluator = Evaluator(name=dataset_name)
                input_dict = {
                    "y_true": data[subset_name]["y"],
                    "y_pred": data[subset_name]["y_predicted"],
                }
                result_dicts[subset_name].append(evaluator.eval(input_dict))

            # newline for space in log file
            print()

        return result_dicts

    def evaluate(self, dataset_name, num_runs, max_elem=None):
        methods = {"path2vec": get_paths2vec_X, "random": get_random_X}

        dataset = GraphPropPredDataset(name=dataset_name)

        final_dict = {}
        final_dict[dataset_name] = {}
        final_dict[dataset_name]["results"] = {}
        final_dict[dataset_name]["runs"] = num_runs
        final_dict[dataset_name]["max_elem"] = max_elem
        final_dict[dataset_name]["sample_size"] = self.sample_size
        final_dict[dataset_name]["windows_size"] = self.window_in_nodes
        metric = dataset.eval_metric
        final_dict[dataset_name]["metric"] = metric

        for methodname, method in methods.items():
            result_dict = self.get_result_dicts(
                method,
                dataset_name,
                num_runs,
                max_elem=max_elem,
            )

            # print results

            final_dict[dataset_name]["results"][methodname] = {}
            for result_dict_subset, dict_list in result_dict.items():
                results = [list(x.values())[0] for x in dict_list]

                final_dict[dataset_name]["results"][methodname][
                    result_dict_subset
                ] = results

        return final_dict
