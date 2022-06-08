from __future__ import print_function

import os

from utils.common import *
from utils.exploratory_analysis import get_paths, prototypes_impact_analysis, \
    physical_pipelines_analysis, run_meta_learning, transformation_analysis, meta_learning_input_preparation


def main():
    args = parse_args()
    evaluation1_results_path, evaluation2_3_results_path, plots_path, new_results_path = get_paths(args.toy_example)
        
    prototypes_impact_analysis(evaluation1_results_path, evaluation2_3_results_path, plots_path, args.toy_example)
    transformation_analysis(evaluation2_3_results_path, new_results_path, plots_path)
    physical_pipelines_analysis(evaluation2_3_results_path, new_results_path, plots_path)
    meta_learning_input_preparation(new_results_path, evaluation2_3_results_path)
    run_meta_learning(args.toy_example)

main()
