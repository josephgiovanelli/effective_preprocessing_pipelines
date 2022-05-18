import os
import copy
import re
import argparse

from collections import OrderedDict

import pandas as pd

from commons import large_comparison_classification_tasks, extended_benchmark_suite, preprocessing_impact_suite, benchmark_suite, algorithms
from results_processors.utils import create_directory

base = OrderedDict([
    ("title", ""),
    ("setup", {
        "policy": "split",
    }),
    ("control", {
        "seed": 42
    })
])

def parse_args():

    parser = argparse.ArgumentParser(
        description="""
            Automated Machine Learning Workflow creation and configuration
            """
    )

    parser.add_argument(
        "-exp",
        "--experiment",
        nargs="?",
        type=str,
        required=True,
        help="type of the experiments",
    )

    args = parser.parse_args()

    return args

def __write_scenario(path, scenario):
    try:
        print("   -> {}".format(path))
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

def get_filtered_datasets():
    df = pd.read_csv("results_processors/meta_features/simple-meta-features.csv")
    df = df.loc[df['did'].isin(list(dict.fromkeys(benchmark_suite + extended_benchmark_suite + [10, 20, 26])))]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

args = parse_args()

scenario_path = create_directory("./", "scenarios")
scenario_path = create_directory(scenario_path, args.experiment)

if args.experiment == "pipeline_construction" or args.experiment == "evaluation1":
    datasets = get_filtered_datasets() 
else:
    datasets = preprocessing_impact_suite

for dataset in datasets:
    for algorithm in algorithms:
        scenario = copy.deepcopy(base)
        scenario["title"] = "{} on dataset n. {} with Split policy".format(
            algorithm, dataset
        )
        if args.experiment == "pipeline_construction" or args.experiment == "evaluation1":
            runtime = 400 
        else:
            runtime = 130
        scenario["setup"]["runtime"] = runtime
        scenario["setup"]["dataset"] = dataset
        scenario["setup"]["algorithm"] = algorithm
        scenario["policy"] = {"step_pipeline": runtime}

        algorithm_acronym = "".join([c for c in algorithm if c.isupper()]).lower()
        if args.experiment == "pipeline_construction" or args.experiment == "evaluation1":
            if args.experiment == "evaluation1":
                scenario["policy"]["step_pipeline"] = 200
            path = os.path.join(scenario_path, "{}_{}.yaml".format(algorithm_acronym, dataset))
            __write_scenario(path, scenario)
        else:
            for experiment_step in ["algorithm", "algorithm_pipeline"]:
                step_pipeline = 0 if experiment_step == "algorithm" else 50
                scenario["policy"]["step_pipeline"] = step_pipeline
                path = create_directory(scenario_path, experiment_step)
                path = os.path.join(path, "{}_{}.yaml".format(algorithm_acronym, dataset))
                __write_scenario(path, scenario)
