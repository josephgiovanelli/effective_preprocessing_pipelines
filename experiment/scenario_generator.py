from utils.common import *
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

args = parse_args()


if args.toy_example == True:
    scenario_path = create_directory(SCENARIO_PATH, "toy")
else:
    scenario_path = create_directory(SCENARIO_PATH, "paper")
scenario_path = create_directory(scenario_path, args.experiment)

datasets = get_filtered_datasets(args.experiment, args.toy_example)


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
