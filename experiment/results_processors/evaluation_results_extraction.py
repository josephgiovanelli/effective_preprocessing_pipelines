from __future__ import print_function

import argparse
import os
import itertools

from evaluation_utils import get_filtered_datasets, load_results_pipelines, declare_winners, \
    summarize_winners, save_summary
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
    results_path = os.path.join(results_path, "evaluation1")
    plots_path = create_directory(plots_path, "evaluation1")

    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets()]))]
    #print(filtered_data_sets)

    results = load_results_pipelines(results_path, filtered_data_sets)
    #print(results)

    winners = declare_winners(results)
    #print(winners)

    summary = summarize_winners(winners)
    #print(summary)

    results_path = create_directory(results_path, "summary")
    save_summary(summary, results_path, plots_path)


main()