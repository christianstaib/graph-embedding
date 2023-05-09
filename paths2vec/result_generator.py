import json

from helpers.pipeline_evaluator import PipelineEvaluator

cpu_count = 16
sample_size = 150
window_in_nodes = 3

datasets_runs = {
    "ogbg-molfreesolv": 10,
    "ogbg-molesol": 10,
    "ogbg-molsider": 10,
    "ogbg-molclintox": 10,
    "ogbg-molbace": 10,
    "ogbg-molbbbp": 10,
    "ogbg-mollipo": 10,
    "ogbg-moltox21": 10,
    "ogbg-moltoxcast": 10,
    "ogbg-molhiv": 3,
    "ogbg-molmuv": 3,
    "ogbg-molpcba": 1,
}

if __name__ == "__main__":
    pipeline_evaluator = PipelineEvaluator(
        cpu_count=cpu_count,
        window_in_nodes=window_in_nodes,
        sample_size=sample_size,
        vertex_feature_idx=range(9),
        edge_feature_idx=range(3),
    )

    for dataset_name, runs in datasets_runs.items():
        result_dict = pipeline_evaluator.evaluate(
            dataset_name=dataset_name, num_runs=runs
        )

        with open(f"results/{dataset_name}.json", "w") as fp:
            json.dump(result_dict, fp, indent=4)
