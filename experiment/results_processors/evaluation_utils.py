import collections
import os
import json
import string

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

algorithms = ['NaiveBayes', 'KNearestNeighbors', 'RandomForest']
benchmark_suite = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307,
                       300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501,
                       1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499,
                       40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978,
                       40670, 40701]
extended_benchmark_suite = [41145, 41156, 41157, 4541, 41158, 42742, 40498, 42734, 41162, 42733, 42732, 1596, 40981, 40685, 
                        4135, 41142, 41161, 41159, 41163, 41164, 41138, 41143, 41146, 41150, 40900, 41165, 41166, 41168, 41169, 
                        41147, 1111, 1169, 41167, 41144, 1515, 1457, 181]

def get_filtered_datasets():
    df = pd.read_csv('meta_features/simple-meta-features.csv')
    df = df.loc[df['did'].isin(benchmark_suite + extended_benchmark_suite + [10, 20, 26] + [15, 29, 1053, 1590])]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

def load_results_pipelines(input_path, filtered_data_sets):
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']

    for acronym in filtered_data_sets:
        algorithm = acronym.split('_')[0]
        data_set = acronym.split('_')[1]

        if acronym in results:
            if not(algorithm in results_map.keys()):
                results_map[algorithm] = {}
            results_map[algorithm][data_set] = []
            with open(os.path.join(input_path, acronym + '.json')) as json_file:
                data = json.load(json_file)
                for i in range(0, 24):
                    index = data['pipelines'][i]['index']
                    accuracy = data['pipelines'][i]['accuracy']
                    results_map[algorithm][data_set].append({'index': index, 'accuracy': accuracy})

    return results_map

def load_results_auto(input_path, filtered_data_sets):
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']

    for acronym in filtered_data_sets:
        algorithm = acronym.split('_')[0]
        data_set = acronym.split('_')[1]

        if not (algorithm in results_map.keys()):
            results_map[algorithm] = {}
        if acronym in results:
            with open(os.path.join(input_path, acronym + '.json')) as json_file:
                data = json.load(json_file)
                accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                try:
                    baseline = data['context']['baseline_score'] // 0.0001 / 100
                except:
                    baseline = 0
        else:
            accuracy = 0
            baseline = 0

        results_map[algorithm][data_set] = (accuracy, baseline)

    return results_map

def declare_winners(results_map):

    winners_map = {}
    for algorithm, value in results_map.items():
        winners_map[algorithm] = {}

        for dataset, pipelines in value.items():
            index_max_accuracy = -1
            for i in range(0, 24):
                if index_max_accuracy == -1:
                    if results_map[algorithm][dataset][i]['accuracy'] != 0:
                        index_max_accuracy = i
                else:
                    if results_map[algorithm][dataset][index_max_accuracy]['accuracy'] < results_map[algorithm][dataset][i]['accuracy']:
                        index_max_accuracy = i
            winners_map[algorithm][dataset] = index_max_accuracy

    return winners_map

def get_winners_accuracy(results_map):

    accuracy_map = {}
    for algorithm, value in results_map.items():
        accuracy_map[algorithm] = {}

        for dataset, pipelines in value.items():
            index_max_accuracy = -1
            for i in range(0, 24):
                if index_max_accuracy == -1:
                    if results_map[algorithm][dataset][i]['accuracy'] != 0:
                        index_max_accuracy = i
                else:
                    if results_map[algorithm][dataset][index_max_accuracy]['accuracy'] < results_map[algorithm][dataset][i]['accuracy']:
                        index_max_accuracy = i
            accuracy_map[algorithm][dataset] = results_map[algorithm][dataset][index_max_accuracy]['accuracy']

    return accuracy_map

def summarize_winners(winners_map):

    summary_map = {}
    for algorithm, value in winners_map.items():
        summary_map[algorithm] = {}
        for i in range(-1, 24):
            summary_map[algorithm][i] = 0

        for _, winner in value.items():
            summary_map[algorithm][winner] += 1

    return summary_map

def save_summary(summary_map, results_path, plots_path):
    if os.path.exists('summary.csv'):
        os.remove('summary.csv')
    total = {}
    algorithm_map = {'nb': 'NB', 'knn': 'KNN', 'rf': 'RF'}
    win = {}
    pipelines = []

    for algorithm, value in summary_map.items():
        pipelines = []
        with open(os.path.join(results_path, '{}.csv'.format(algorithm)), 'w') as out:
            out.write('pipeline,winners\n')
            winners_temp = []
            for pipeline, winner in value.items():
                out.write(str(pipeline) + ',' + str(winner) + '\n')
                pipelines.append(str(pipeline))
                winners_temp.append(int(winner))
                total[str(pipeline)] = winner if not(str(pipeline) in total.keys()) else total[str(pipeline)] + winner
            win[algorithm] = winners_temp[1:]
            pipelines = pipelines[1:]

    winners = {'pipelines': pipelines, 'nb': [e / 168 * 100 for e in win['nb']], 'knn': [e / 168 * 100 for e in win['knn']], 'rf': [e / 168 * 100 for e in win['rf']]}
    winners['total'] = [winners['nb'][j] +  winners['knn'][j] +  winners['rf'][j] for j in range(len(winners['knn']))]
    winners = pd.DataFrame.from_dict(winners)
    #winners = winn ers.sort_values(by=['total'], ascending=False)

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


    plt.bar([str(int(a) + 1) for a in winners['pipelines']], winners['nb'], label=algorithm_map['nb'], color="lightcoral")
    plt.bar([str(int(a) + 1) for a in winners['pipelines']], winners['knn'], bottom=winners['nb'], label=algorithm_map['knn'], color="darkturquoise")
    plt.bar([str(int(a) + 1) for a in winners['pipelines']], winners['rf'], bottom=winners['nb'] +  winners['knn'], label=algorithm_map['rf'], color="violet")

    plt.xlabel('Prototype ID', labelpad=10.0)
    plt.ylabel('Percentage of cases for which a prototype\nachieved the best performance', labelpad=10.0)
    plt.yticks(ticks=np.linspace(0, 20, 11), labels=['{}%'.format(int(x)) for x in np.linspace(0, 20, 11)])
    #plt.title('Comparison of the goodness of the prototypes')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    fig.savefig(os.path.join(plots_path, 'evaluation1.pdf'))

    plt.clf()



def save_comparison(results_pipelines, results_auto, result_path, plot_path):
    if os.path.exists('comparison.csv'):
        os.remove('comparison.csv')
    algorithm_map = {'nb': 'NaiveBayes', 'knn': 'KNearestNeighbors', 'rf': 'RandomForest'}
    plot_results = {}

    for algorithm, value in results_pipelines.items():
        plot_results[algorithm] = {}
        with open(os.path.join(result_path, '{}.csv'.format(algorithm)), 'w') as out:
            out.write('dataset,exhaustive,pseudo-exhaustive,baseline,score\n')
            for dataset, accuracy in value.items():
                score = 0 if (accuracy - results_auto[algorithm][dataset][1]) == 0 else (results_auto[algorithm][dataset][0] - results_auto[algorithm][dataset][1]) / (accuracy - results_auto[algorithm][dataset][1])
                out.write(str(dataset) + ',' + str(accuracy) + ',' + str(results_auto[algorithm][dataset][0]) + ',' +
                          str(results_auto[algorithm][dataset][1]) + ',' + str(score) + '\n')
                plot_results[algorithm][dataset] = score

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

    plt.axhline(y=1.0, color='#aaaaaa', linestyle='--')

    plt.boxplot([[value for value in plot_results['nb'].values() if value != 0],
                 [value for value in plot_results['knn'].values() if value != 0],
                 [value for value in plot_results['rf'].values() if value != 0]])


    #plt.xlabel('Algorithms', labelpad=15.0)
    plt.xticks([1, 2, 3], ['NB', 'KNN', 'RF'])
    plt.ylabel('Normalized distance', labelpad=15.0)
    plt.xlabel('Algorithms', labelpad=15.0)
    plt.yticks(np.linspace(0, 1.2, 7))
    plt.ylim(0.0, 1.25)
    #plt.title('Evaluation of the prototype building through the proposed precedence')
    #plt.tight_layout()
    plt.tight_layout(pad=0.2)
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(plot_path, 'evaluation2.pdf'))

    plt.clf()