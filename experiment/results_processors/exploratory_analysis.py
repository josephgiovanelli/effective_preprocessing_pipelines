from __future__ import print_function

import os

from utils.common import *
from utils.exploratory_analysis import get_paths, prototypes_impact_analysis, physical_pipelines_analysis, transformation_analysis


def main():
    args = parse_args()
    evaluation1_results_path, evaluation2_3_pipeline_algorithm_results_path, evaluation2_3_results_path, plots_path, new_results_path = get_paths(
        args.toy_example)
        
    prototypes_impact_analysis(
        evaluation1_results_path, evaluation2_3_pipeline_algorithm_results_path, evaluation2_3_results_path, plots_path, args.toy_example)
    transformation_analysis(
        evaluation2_3_pipeline_algorithm_results_path, new_results_path, plots_path)
    physical_pipelines_analysis(
        evaluation2_3_pipeline_algorithm_results_path, new_results_path, plots_path)

main()
