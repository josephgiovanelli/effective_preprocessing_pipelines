import logging
from statistics import mean

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.common import *

from os import listdir
from os.path import isfile, join


def load_results(input_path, filtered_datasets):
    result = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']
    for algorithm in algorithms:
        for dataset in filtered_datasets:
            acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
            acronym += '_' + str(dataset)
            if acronym in results:
                with open(os.path.join(input_path, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    best_accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                    best_config = data['context']['best_config']['iteration']
                    num_iterations = data['context']['iteration'] + 1
                    baseline_score = data['context']['baseline_score'] // 0.0001 / 100
                    history = data['context']['history']

                    result[acronym] = {}
                    result[acronym]['best_accuracy'] = best_accuracy
                    result[acronym]['best_config'] = best_config
                    result[acronym]['num_iterations'] = num_iterations
                    result[acronym]['baseline_score'] = baseline_score
                    result[acronym]['history'] = history

    return result


def find_pipeline_iterations(history):
    for iteration in history:
        if iteration['step'] == 'algorithm':
            pass
        else:
            return iteration['iteration']


def perform_algorithm_pipeline_analysis(results, toy):
    pipelines_iterations, algorithm_iterations = [], []

    for result in results:
        results[result]['algorithm_iterations'] = find_pipeline_iterations(
            results[result]['history'])
        results[result]['pipeline_iterations'] = results[result]['num_iterations'] - \
            results[result]['algorithm_iterations']
        algorithm_iterations.append(results[result]['algorithm_iterations'])
        pipelines_iterations.append(results[result]['pipeline_iterations'])
        # print(result, results[result]['pipeline_iterations'], results[result]['algorithm_iterations'])
    # print(min(pipelines_iterations), min(algorithm_iterations))

    scores = {}
    half_iteration = 5 if toy else 50
    for result in results:
        acronym = result.split('_')[0]
        if not(acronym in scores):
            scores[acronym] = {}
        scores[acronym][result] = []
        scores[acronym][result].append((0, results[result]['baseline_score']))
        max_score = results[result]['baseline_score']
        for i in range(1, half_iteration+1):
            if i <= results[result]['algorithm_iterations']:
                scores[acronym][result].append(
                    (i, results[result]['history'][i - 1]['max_history_score'] // 0.0001 / 100))
                max_score = results[result]['history'][i -
                                                       1]['max_history_score'] // 0.0001 / 100
            else:
                scores[acronym][result].append((i, max_score))
        for i in range(1, half_iteration+1):
            scores[acronym][result].append((half_iteration + i, results[result]['history']
                                           [results[result]['algorithm_iterations'] + i - 1]['max_history_score'] // 0.0001 / 100))

    return perform_analysis(results, scores, toy)


def perform_analysis(results, scores, toy):
    scores_to_kpi = {}
    outcome = {}

    max_iteration = 10 if toy else 100
    for result in results:
        for i in range(0, max_iteration+1):
            acronym = result.split('_')[0]
            if not (acronym in scores_to_kpi):
                scores_to_kpi[acronym] = []
                outcome[acronym] = []
            try:
                scores_to_kpi[acronym][i].append(
                    scores[acronym][result][i][1] / results[result]['baseline_score'])
            except:
                scores_to_kpi[acronym].append([])
                scores_to_kpi[acronym][i].append(
                    scores[acronym][result][i][1] / results[result]['baseline_score'])

    for i in range(1, max_iteration+1):
        for key in outcome.keys():
            outcome[key].append(mean(scores_to_kpi[key][i]))
            #outcome[key].append(mean(scores_to_kpi[key][i]) // 0.01 / 100)

    return outcome


def save_analysis(analysis, result_path, toy):

    max_iteration = 10 if toy else 100

    # with open(os.path.join(result_path, 'result_with_impact.csv'), 'w') as out:
    #     out.write(','.join(analysis.keys()) + '\n')

    # with open(os.path.join(result_path, 'result_with_impact.csv'), 'a') as out:
    #     for i in range(0, max_iteration):
    #         row = []
    #         for key in analysis.keys():
    #             row.append(str(analysis[key][i]))
    #         out.write(','.join(row) + '\n')

    x = np.linspace(0, max_iteration, num=max_iteration)

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    plt.plot(x, analysis['nb'], label='NB', linewidth=2.5, color='lightcoral')
    plt.plot(x, analysis['knn'], label='KNN',
             linewidth=2.5, color='darkturquoise')
    plt.plot(x, analysis['rf'], label='RF', linewidth=2.5, color='violet')
    plt.xlabel('Configurations visited')
    plt.ylabel('Ratio of predictive accuracy change')
    #plt.title("Optimization on bank-marketing data-set")
    plt.legend()
    plt.xlim(0, max_iteration)
    plt.ylim(0.8, 1.5)
    plt.axvline(x=max_iteration/2, color='#aaaaaa', linestyle='--')
    plt.grid(False)
    plt.tick_params(axis='both', which='both', length=5, color='#aaaaaa')
    if not toy:
        plt.xticks(np.linspace(0, max_iteration, int(max_iteration/10 + max_iteration/100)))
    fig = plt.gcf()
    fig.set_size_inches(12, 6, forward=True)
    fig.savefig(os.path.join(result_path, 'Figure2.pdf'))


def pipeline_impact(toy):

    logging.getLogger('matplotlib.font_manager').disabled = True
    if toy:
        path = os.path.join(RAW_RESULT_PATH, "toy")
        result_path = create_directory(ARTIFACTS_PATH, 'toy')
    else:
        path = os.path.join(RAW_RESULT_PATH, "paper")
        result_path = create_directory(ARTIFACTS_PATH, 'paper')
    input_path = os.path.join(path, "pipeline_impact")
    result_path = create_directory(result_path, 'pipeline_impact')
    filtered_data_sets = get_filtered_datasets(
        experiment='pipeline_impact', toy=toy)

    pipeline_algorithm_results = load_results(input_path, filtered_data_sets)

    pipeline_algorithm_analysis = perform_algorithm_pipeline_analysis(
        pipeline_algorithm_results, toy)

    save_analysis(pipeline_algorithm_analysis, result_path, toy)
