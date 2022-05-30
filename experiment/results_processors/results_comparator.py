from __future__ import print_function

import itertools
import os
import argparse

from evaluation2_results_extraction_utils import get_filtered_datasets, load_results, merge_results, \
    save_comparison, save_summary, plot_comparison
from utils import create_directory


parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
parser.add_argument("-toy", "--toy_example", action='store_true', default=False, help="wether it is a toy example or not")
args = parser.parse_args()


def main():
    # configure environment
    results_path = "results"
    plots_path = "plots"
    if args.toy_example:
        results_path = os.path.join(results_path, "toy")
        plots_path = os.path.join(plots_path, "toy")
    else:
        results_path = os.path.join(results_path, "paper")
        plots_path = os.path.join(plots_path, "paper")
    results_path = os.path.join(results_path, "evaluation2_3")
    input_auto = os.path.join(results_path, "pipeline_algorithm")
    input_algorithm = os.path.join(results_path, "algorithm")
    results_path = create_directory(results_path, "summary")
    results_path = create_directory(results_path, "evaluation3")
    plots_path = create_directory(plots_path, "evaluation3")

    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets(args.toy_example)]))]

    auto_results = load_results(input_auto, filtered_data_sets, algorithm_comparison = True)
    algorithm_results = load_results(input_algorithm, filtered_data_sets, algorithm_comparison = True)

    comparison, summary = merge_results(auto_results, algorithm_results, 'algorithm', filtered_data_sets)
    #print(comparison)
    save_comparison(comparison, results_path)
    save_summary(summary, results_path)
    plot_comparison(comparison, plots_path)

main()