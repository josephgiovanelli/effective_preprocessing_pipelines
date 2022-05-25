from __future__ import print_function

import argparse
import itertools
import os

from evaluation_utils import get_filtered_datasets, load_results_pipelines, load_results_auto, \
     get_winners_accuracy, save_comparison
from utils import create_directory


parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
parser.add_argument("-toy", "--toy-example", nargs="?", type=bool, required=False, default=False, help="wether it is a toy example or not")
args = parser.parse_args()


def main():
    # configure environment
    results_path = "results"
    plots_path = "plots"
    if args.toy_example:
        results_path = os.path.join(results_path, "toy")
        plots_path = os.path.join(plots_path, "toy")
        

    evaluation2_3_results_path = os.path.join(results_path, "evaluation2_3")
    evaluation2_3_pipeline_algorithm_results_path = os.path.join(evaluation2_3_results_path, "pipeline_algorithm")
    evaluation1_results_path = os.path.join(results_path, "evaluation1")
    plots_path = create_directory(plots_path, "evaluation2")
    new_results_path = create_directory(evaluation2_3_results_path, "summary")
    new_results_path = create_directory(new_results_path, "evaluation2")


    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets()]))]
    #print(filtered_data_sets)

    results_pipelines = load_results_pipelines(evaluation1_results_path, filtered_data_sets)
    results_pipelines = get_winners_accuracy(results_pipelines)
    results_auto = load_results_auto(evaluation2_3_pipeline_algorithm_results_path, filtered_data_sets)
    #print(results_pipelines)
    #print(results_auto)

    save_comparison(results_pipelines, results_auto, new_results_path, plots_path)


main()