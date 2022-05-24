from __future__ import print_function
import os
from pipeline_impact_utils import perform_algorithm_pipeline_analysis, save_analysis, load_results
from results_cooking_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix, chi2tests, save_chi2tests, join_result_with_simple_meta_features, \
    get_results, modify_class
from results_extraction_utils import create_possible_categories, get_filtered_datasets, \
    extract_results, save_results
from utils import create_directory

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-exp", "--experiment", nargs="?", type=str, required=True, help="type of the experiments")
    parser.add_argument("-toy", "--toy-example", nargs="?", type=bool, required=False, default=False, help="wether it is a toy example or not")
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    path = "results"
    result_path = "plots"
    if args.toy_example:
        path = os.path.join(path, "toy")
        result_path = create_directory(result_path, 'toy')
    input_path = os.path.join(path, "pipeline_impact")
    result_path = create_directory(result_path, 'pipeline_impact')
    input_pipeline = os.path.join(input_path, "algorithm_pipeline")
    print(input_pipeline, result_path)
    filtered_data_sets = get_filtered_datasets(experiment=args.experiment, toy=args.toy_example)
    print(filtered_data_sets)

    pipeline_algorithm_results = load_results(input_pipeline, filtered_data_sets)
    print(pipeline_algorithm_results)
    #algorithm_results = load_results(input_algorithm, filtered_data_sets)

    pipeline_algorithm_analysis = perform_algorithm_pipeline_analysis(pipeline_algorithm_results, args.toy_example)
    print(pipeline_algorithm_analysis)
    #algorithm_analysis = perform_algorithm_analysis(algorithm_results)

    result_path = create_directory(result_path, 'algorithm_pipeline')
    save_analysis(pipeline_algorithm_analysis, result_path, args.toy_example)
    #save_analysis(algorithm_analysis, create_directory(result_path, 'algorithm'))
    
    # print(pipeline_algorithm_analysis)
    # print(algorithm_analysis)

    # print(','.join(pipeline_algorithm_analysis.keys()))
    # for key, value in pipeline_algorithm_analysis.items():
    #     print(key, value)








main()