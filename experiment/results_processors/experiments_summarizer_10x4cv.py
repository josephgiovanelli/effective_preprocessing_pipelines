from __future__ import print_function

from results_cooking_utils import chi2test
from results_extraction_utils import create_possible_categories, get_filtered_datasets, extract_results_10x4cv, save_results_10x4cv
from utils import parse_args, create_directory
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
parser.add_argument("-exp", "--experiment", nargs="?", type=str, required=True, help="type of the experiments")
parser.add_argument("-mode", "--mode", nargs="?", type=str, required=False, help="algorithm or algorithm_pipeline")
parser.add_argument("-toy", "--toy-example", nargs="?", type=bool, required=False, default=False, help="wether it is a toy example or not")
args = parser.parse_args()


def main():
    # configure environment
    pipeline = args.mode.split('_')
    path = "results"
    if args.toy_example:
        path = os.path.join(path, "toy")
    input_path = os.path.join(path, "pipeline_construction", args.mode)
    result_path = create_directory(input_path, 'summary')
    result_path = create_directory(result_path, '10x4cv')
    categories = create_possible_categories(pipeline)
    filtered_data_sets = get_filtered_datasets(experiment=args.experiment, toy=args.toy_example)
    folds = 3 if args.toy_example else 4
    repeat = 10

    summaries = extract_results_10x4cv(input_path, filtered_data_sets, pipeline, categories, folds, repeat)

    save_results_10x4cv(create_directory(result_path, 'raw'), summaries)

    #if (not args.toy_example) or (args.toy_example and args.mode == 'features_rebalance' and args.mode == 'discretize_rebalance'):
    prob = 0.95
    a = 0.1 if categories['first_second'] == 'FN' else (0.9 if categories['first_second'] == 'DF' else 0.5)
    b = round(1 - a, 2)
    results = pd.DataFrame()
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    for batch in range(len(summaries)):
        total_train = summaries[batch]['train']['summary'][categories['first_second']] + summaries[batch]['train']['summary'][categories['second_first']]
        total_test = summaries[batch]['test']['summary'][categories['first_second']] + summaries[batch]['test']['summary'][categories['second_first']]
        train_frequencies = [summaries[batch]['train']['summary'][categories['first_second']], summaries[batch]['train']['summary'][categories['second_first']]]
        test_frequencies = [summaries[batch]['test']['summary'][categories['first_second']], summaries[batch]['test']['summary'][categories['second_first']]]
        critical, stat, statistic_test, alpha, p, p_value = chi2test(train_frequencies, test_frequencies, prob)
        critical_train, stat_train, statistic_test_train, alpha_train, p_train, p_value_train = chi2test(train_frequencies, [total_train * a, total_train * b], prob)
        critical_test, stat_test, statistic_test_test, alpha_test, p_test, p_value_test = chi2test(test_frequencies, [total_test * a, total_test * b], prob)
        results = results.append({'critical': critical, 'stat': stat, 'statistic_test': statistic_test, 'alpha': alpha, 'p': p, 'p_value': p_value}, ignore_index=True)
        results_train = results_train.append({'critical': critical_train, 'stat': stat_train, 'statistic_test': statistic_test_train, 'alpha': alpha_train, 'p': p_train, 'p_value': p_value_train}, ignore_index=True)
        results_test = results_test.append({'critical': critical_test, 'stat': stat_test, 'statistic_test': statistic_test_test, 'alpha': alpha_test, 'p': p_test, 'p_value': p_value_test}, ignore_index=True)
    results.to_csv(os.path.join(result_path, 'summary.csv'), index=False)
    results_train.to_csv(os.path.join(result_path, 'summary_train.csv'), index=False)
    results_test.to_csv(os.path.join(result_path, 'summary_test.csv'), index=False)
    
    results = pd.DataFrame()
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    for repetition in range(repeat):
        batch_results = pd.DataFrame()
        batch_train_results = pd.DataFrame()
        batch_test_results = pd.DataFrame()
        for fold in range(folds):
            batch = repetition * folds + fold
            total_train = summaries[batch]['train']['summary'][categories['first_second']] + summaries[batch]['train']['summary'][categories['second_first']]
            total_test = summaries[batch]['test']['summary'][categories['first_second']] + summaries[batch]['test']['summary'][categories['second_first']]
            train_frequencies = [summaries[batch]['train']['summary'][categories['first_second']], summaries[batch]['train']['summary'][categories['second_first']]]
            test_frequencies = [summaries[batch]['test']['summary'][categories['first_second']], summaries[batch]['test']['summary'][categories['second_first']]]
            critical, stat, statistic_test, alpha, p, p_value = chi2test(train_frequencies, test_frequencies, prob)
            critical_train, stat_train, statistic_test_train, alpha_train, p_train, p_value_train = chi2test(train_frequencies, [total_train * a, total_train * b], prob)
            critical_test, stat_test, statistic_test_test, alpha_test, p_test, p_value_test = chi2test(test_frequencies, [total_test * a, total_test * b], prob)
            batch_results = batch_results.append({'critical': critical, 'stat': stat, 'statistic_test': statistic_test, 'alpha': alpha, 'p': p, 'p_value': p_value}, ignore_index=True)
            batch_train_results = batch_train_results.append({'critical': critical_train, 'stat': stat_train, 'statistic_test': statistic_test_train, 'alpha': alpha_train, 'p': p_train, 'p_value': p_value_train}, ignore_index=True)
            batch_test_results = batch_test_results.append({'critical': critical_test, 'stat': stat_test, 'statistic_test': statistic_test_test, 'alpha': alpha_test, 'p': p_test, 'p_value': p_value_test}, ignore_index=True)
        batch_results = batch_results.mean().round(3)
        batch_train_results = batch_train_results.mean().round(3)
        batch_test_results = batch_test_results.mean().round(3)
        results = results.append(batch_results, ignore_index=True)
        results_train = results_train.append(batch_train_results, ignore_index=True)
        results_test = results_test.append(batch_test_results, ignore_index=True)
    results.to_csv(os.path.join(result_path, 'summary_with_mean_.csv'), index=False)
    results_train.to_csv(os.path.join(result_path, 'summary_train_with_mean_.csv'), index=False)
    results_test.to_csv(os.path.join(result_path, 'summary_test_with_mean_.csv'), index=False)


            

main()
