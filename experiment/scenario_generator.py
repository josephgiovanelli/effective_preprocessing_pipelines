from results_processors.utils.common import *
from collections import OrderedDict
import pandas as pd
import os
import copy
import warnings
warnings.filterwarnings("ignore")


base = OrderedDict([
    ("title", ""),
    ("setup", {
        "policy": "split",
    }),
    ("control", {
        "seed": 42
    })
])


def __write_scenario(path, scenario):
    try:
        # print("   -> {}".format(path))
        with open(path, "w") as f:
            for k, v in scenario.items():
                if isinstance(v, str):
                    f.write("{}: {}\n".format(k, v))
                else:
                    f.write("{}:\n".format(k))
                    for i, j in v.items():
                        f.write("  {}: {}\n".format(i, j))
    except Exception as e:
        print(e)


def get_filtered_datasets(toy):
    df = pd.read_csv("meta_features/simple-meta-features.csv")
    df = df.loc[df['did'].isin(list(dict.fromkeys(benchmark_suite + extended_benchmark_suite + [10, 20, 26])))]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    if toy:
        df = df.loc[df['NumberOfInstances'] <= 2000]
        df = df.loc[df['NumberOfFeatures'] <= 10]
        df = df.sort_values(by=['NumberOfInstances', 'NumberOfFeatures'])
        df = df[:10]
    df = df['did']
    return df.values.flatten().tolist()


args = parse_args()

scenario_path = "scenarios"
if args.toy_example == True:
    scenario_path = create_directory(scenario_path, "toy")
else:
    scenario_path = create_directory(scenario_path, "paper")
scenario_path = create_directory(scenario_path, args.experiment)

datasets = []
if args.experiment == "pipeline_construction" or args.experiment == "evaluation1" or args.experiment == "evaluation2_3":
    datasets = get_filtered_datasets(args.toy_example)
elif args.experiment == "pipeline_impact":
    datasets = pipeline_impact_suite


for dataset in datasets:
    for algorithm in algorithms:
        scenario = copy.deepcopy(base)
        scenario["title"] = "{} on dataset n. {} with Split policy".format(
            algorithm, dataset
        )
        if args.toy_example:
            runtime = 10
            step_pipeline = 5
        else:
            if args.experiment == "pipeline_impact":
                runtime = 130
                step_pipeline = 50
            else:
                runtime = 400
                step_pipeline = 400
        scenario["setup"]["runtime"] = runtime
        scenario["setup"]["dataset"] = dataset
        scenario["setup"]["algorithm"] = algorithm
        scenario["policy"] = {"step_pipeline": step_pipeline}

        algorithm_acronym = "".join(
            [c for c in algorithm if c.isupper()]).lower()

        if args.experiment == "evaluation2_3":
            experiment_steps = ['algorithm', 'pipeline_algorithm']
            for experiment_step in experiment_steps:
                if experiment_step == "algorithm":
                    step_pipeline = 0
                else:
                    if args.toy_example:
                        step_pipeline = 5
                    else:
                        step_pipeline = 75
                scenario["policy"]["step_pipeline"] = step_pipeline
                if experiment_step == "pipeline_algorithm":
                    if args.toy_example:
                        scenario["setup"]["runtime"] = 5
                    else:
                        scenario["setup"]["runtime"] = 200
                path = create_directory(scenario_path, experiment_step)
                path = os.path.join(path, "{}_{}.yaml".format(
                    algorithm_acronym, dataset))
                __write_scenario(path, scenario)
        else:
            if args.experiment == "evaluation1":
                if args.toy_example:
                    scenario["policy"]["step_pipeline"] = 5
                else:
                    scenario["policy"]["step_pipeline"] = 200
            path = os.path.join(scenario_path, "{}_{}.yaml".format(
                algorithm_acronym, dataset))
            __write_scenario(path, scenario)
