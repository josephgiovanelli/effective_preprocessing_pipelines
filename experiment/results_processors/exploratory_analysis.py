from __future__ import print_function
from os import listdir
from os.path import isfile, join

import os
import json
import argparse
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import create_directory
from evaluation_utils import get_filtered_datasets, load_results_pipelines, load_results_auto

parser = argparse.ArgumentParser(
    description="Automated Machine Learning Workflow creation and configuration")
parser.add_argument("-toy", "--toy_example", action='store_true',
                    default=False, help="wether it is a toy example or not")
args = parser.parse_args()


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def transformation_analysis(input_auto, new_results_path, plots_path):
    # configure environment
    discretize_count, normalize_count = {}, {}
    results_map = pd.DataFrame()
    for algorithm in ["nb", "knn", "rf"]:
        discretize_count[algorithm], normalize_count[algorithm] = 0, 0
        df = pd.read_csv(os.path.join("results", "summary",
                         "evaluation2_3", algorithm + ".csv"))
        #df = df[(df["pa_percentage"] == 0.5)]
        ids = list(df["dataset"])

        files = [f for f in listdir(input_auto) if isfile(join(input_auto, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']

        for dataset in ids:
            acronym = algorithm + "_" + str(dataset)
            if acronym in results:
                with open(os.path.join(input_auto, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    pipeline = data['context']['best_config']['pipeline']
                    for transformation in ["encode", "features", "impute", "normalize", "discretize", "rebalance"]:
                        try:
                            results_map = results_map.append(pd.DataFrame({
                                "algorithm": [algorithm],
                                "dataset": [dataset],
                                "transformation": [pipeline[transformation][0].split("_")[0]],
                                "operator": [pipeline[transformation][0].split("_")[1]]
                            }), ignore_index=True)
                        except:
                            pass

    results_map.to_csv(os.path.join(
        new_results_path, "pp_pipeline_operator_study.csv"), index=False)

    result = results_map.groupby(
        ['algorithm', 'transformation', 'operator']).count()
    result_sum = result.groupby(['algorithm', 'transformation']).sum()
    for algorithm in result.index.get_level_values('algorithm').unique():
        for transformation in result.index.get_level_values('transformation').unique():
            for operator in result.index.get_level_values('operator').unique():
                try:
                    result.loc[algorithm, transformation,
                               operator] /= result_sum.loc[algorithm, transformation]
                except:
                    pass
    result = result.reset_index()
    result = result[result['operator'] != 'NoneType']
    result = result.set_index(['transformation', 'operator', 'algorithm'])
    # print(result)

    labels = ["NB", "KNN", "RF"]
    colors = ['mediumpurple', 'xkcd:dark grass green', 'xkcd:kermit green', 'xkcd:lime green', 'xkcd:light pea green', 'xkcd:dark coral', 'xkcd:salmon',
              'xkcd:sun yellow', 'xkcd:straw', 'xkcd:aqua green', 'xkcd:light aquamarine', 'xkcd:pumpkin orange', 'xkcd:apricot', 'xkcd:light peach']
    x = np.arange(len(labels))  # the label locations
    width = 0.125  # the width of the bars
    SMALL_SIZE = 8
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 21

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    column = -1
    i = 0
    cumulative = False
    last_transformation, last_operator = '', ''
    for transformation in ['encode', 'normalize', 'discretize', 'impute', 'rebalance', 'features']:
        for operator in result.index.get_level_values('operator').unique():
            try:
                result.loc[transformation, operator]
                flag = True
            except:
                flag = False
            if flag:
                curr_bar = []
                for algo in ["nb", "knn", "rf"]:
                    try:
                        c = result.loc[transformation, operator, algo].to_numpy().flatten()[
                            0]
                    except:
                        c = 0
                    curr_bar.append(c)
                if transformation != last_transformation or last_transformation == '':
                    column += 1
                    ax.bar((x * width * 8) + (width * (column - 1)) - 0.2, curr_bar, width,
                           color=colors[i], label=transformation[0].upper() + "  " + (" " if transformation[0].upper() == "I" else "") + operator)
                else:
                    if not(cumulative):
                        last_bar = []
                        for algo in ["nb", "knn", "rf"]:
                            try:
                                c = result.loc[last_transformation, last_operator, algo].to_numpy().flatten()[
                                    0]
                            except:
                                c = 0
                            last_bar.append(c)
                    ax.bar((x * width * 8) + (width * (column - 1)) - 0.2, curr_bar, width, bottom=last_bar,
                           color=colors[i], label=transformation[0].upper() + "  " + (" " if transformation[0].upper() == "I" else "") + operator)
                if transformation == last_transformation:
                    cumulative = True
                    last_bar = [curr_bar[0] + last_bar[0], curr_bar[1] +
                                last_bar[1], curr_bar[2] + last_bar[2]]
                else:
                    cumulative = False
                last_transformation, last_operator = transformation, operator
                i += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Usage')
    ax.set_xlabel('Algorithms')
    ax.set_yticks(np.linspace(0, 1, 11))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(),
                     loc='lower center', ncol=3, bbox_to_anchor=(0.55, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(plots_path, "Figure9.pdf"),
                bbox_extra_artists=(lgd, text), bbox_inches='tight')


def physical_pipelines_analysis(input_auto, new_results_path, plots_path):
    # configure environment
    results_map = pd.DataFrame()
    for algorithm in ["knn", "nb", "rf"]:

        df = pd.read_csv(os.path.join("results", "summary",
                         "evaluation2_3", algorithm + ".csv"))
        ids = list(df["dataset"])

        files = [f for f in listdir(input_auto) if isfile(join(input_auto, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']

        for dataset in ids:
            acronym = algorithm + "_" + str(dataset)
            if acronym in results:
                with open(os.path.join(input_auto, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                    pipeline = data['context']['best_config']['pipeline']
                    pipeline_conf = data['pipeline']
                    pipeline_conf = ''.join([a[0]
                                            for a in pipeline_conf]).upper()

                    encode_flag = "None" in pipeline["encode"][0]
                    features_flag = "None" in pipeline["features"][0]
                    impute_flag = "None" in pipeline["impute"][0]
                    try:
                        normalize_flag = "None" in pipeline["normalize"][0]
                    except:
                        normalize_flag = True
                    try:
                        discretize_flag = "None" in pipeline["discretize"][0]
                    except:
                        discretize_flag = True
                    rebalance_flag = "None" in pipeline["rebalance"][0]
                    if encode_flag or not(encode_flag):
                        pipeline_conf = pipeline_conf.replace("E", "")
                    if features_flag:
                        pipeline_conf = pipeline_conf.replace("F", "")
                    if impute_flag or not(impute_flag):
                        pipeline_conf = pipeline_conf.replace("I", "")
                    if normalize_flag:
                        pipeline_conf = pipeline_conf.replace("N", "")
                    if discretize_flag:
                        pipeline_conf = pipeline_conf.replace("D", "")
                    if rebalance_flag:
                        pipeline_conf = pipeline_conf.replace("R", "")

                    results_map = results_map.append(pd.DataFrame(
                        {"algorithm": [algorithm], "pipeline": [pipeline_conf]}), ignore_index=True)

    results_map.to_csv(os.path.join(
        new_results_path, "pp_pipeline_study2.csv"), index=False)
    results_map = results_map.pivot_table(
        index='algorithm', columns='pipeline', aggfunc=len, fill_value=0)
    results_map["sum"] = results_map.sum(axis=1)
    results_map = results_map.div(results_map["sum"], axis=0)
    results_map = results_map.drop(['sum'], axis=1)
    results_map = results_map.reset_index()
    results_map = results_map.set_index(["algorithm"])
    results_map = results_map.reindex(["nb", "knn", "rf"])
    results_map = results_map.reset_index()
    results_map.to_csv(os.path.join(
        new_results_path, "pp_pipeline_study2_pivoted.csv"), index=False)
    results_map = results_map.rename(columns={
        'DF': r'$D \to F$',
        'RF': r'$R \to F$',
        'D': r'$D$',
        'N': r'$N$',
        'R': r'$R$',
        'DFR': r'$D \to F \to R$',
        'DR': r'$D \to R$',
        'DRF': r'$D \to R \to F$',
        'NFR': r'$N \to F \to R$',
        'NRF': r'$N \to R \to F$',
        'RD': r'$R \to D$',
        'DF': r'$D \to F$',
        'DR': r'$D \to R$',
        'FR': r'$F \to R$',
        'NF': r'$N \to F$',
        'NR': r'$N \to R$',
        'RDF': r'$R \to D \to F$',
    })
    # print(results_map)
    labels = [x.upper() for x in results_map["algorithm"]]
    patterns = ['/', '\\', '-\\', '-', '+', 'x', 'o',
                '//', '\\\\', 'O.', '--', '++', 'xx', 'OO', '\\|']
    colors = ['skyblue', 'orange', 'lightgreen', 'tomato', 'mediumorchid', 'xkcd:medium brown', 'xkcd:pale pink', 'xkcd:greyish',
              'xkcd:aquamarine', 'xkcd:dodger blue', 'xkcd:sun yellow', 'xkcd:flat green', 'xkcd:red orange', 'xkcd:lighter purple', 'xkcd:silver']
    x = np.arange(len(labels))  # the label locations
    width = 0.125  # the width of the bars
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 21
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots()
    rects = {}
    columns = list(results_map.columns)[2:]
    for column in range(len(columns)):
        ax.bar((x * width * 20) + (width * (column)),
               results_map[columns[column]], width, label=columns[column], color=colors[column], hatch=patterns[column])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Usage')
    ax.set_xlabel('Algorithms')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(x*2.5+0.88)
    ax.set_xticklabels(labels)
    # ax.legend()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(),
                     loc='lower center', ncol=5, bbox_to_anchor=(0.55, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=ax.transAxes)
    fig.tight_layout()
    fig.set_size_inches(10, 5)
    fig.savefig(os.path.join(plots_path, "Figure8.pdf"),
                bbox_extra_artists=(lgd, text), bbox_inches='tight')


def prototypes_impact_analysis(evaluation1_results_path, evaluation2_3_pipeline_algorithm_results_path, evaluation2_3_results_path, plots_path):
    # configure environment

    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(
        ["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets(args.toy_example)]))]
    # print(filtered_data_sets)

    results_pipelines = load_results_pipelines(
        evaluation1_results_path, filtered_data_sets)
    # print(results_pipelines)
    results_auto = load_results_auto(
        evaluation2_3_pipeline_algorithm_results_path, filtered_data_sets)
    # print(results_auto)
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            if results_auto[algorithm][dataset][1] == 0:
                evaluation_3 = pd.read_csv(os.path.join(
                    evaluation2_3_results_path, "summary", "evaluation3", algorithm + ".csv"))
                evaluation_3 = evaluation_3.set_index(['dataset'])
                results_auto[algorithm][dataset] = (
                    results_auto[algorithm][dataset][0], evaluation_3.loc[int(dataset)]["baseline"])
    # print(results_auto)
    impacts = pd.DataFrame()
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            try:
                for elem in results_pipelines[algorithm][dataset]:
                    impacts = impacts.append({'algorithm': algorithm, 'dataset': dataset, 'index': int(
                        elem['index']) + 1, 'impact': elem['accuracy']-results_auto[algorithm][dataset][1]}, ignore_index=True)
            except:
                if not args.toy_example:
                    print(dataset)
    # print(impacts)
    knn = impacts[impacts["algorithm"] == "knn"]
    nb = impacts[impacts["algorithm"] == "nb"]
    rf = impacts[impacts["algorithm"] == "rf"]
    knn = knn.pivot(index='dataset', columns='index', values='impact')
    nb = nb.pivot(index='dataset', columns='index', values='impact')
    rf = rf.pivot(index='dataset', columns='index', values='impact')
    data = [nb, knn, rf]
    fig, ax = plt.subplots(1, 3, sharey=True, constrained_layout=True)
    for i in range(len(data)):
        ax[i].boxplot(data[i], showfliers=False)
        ax[i].set_title("NB" if i == 0 else ("KNN" if i == 1 else "RF"))
        ax[i].set_xlabel('Prototypes ID')
        ax[i].set_xticklabels(range(1, 25), fontsize=8)
    fig.text(0.0, 0.5, 'Impact over the baseline',
             va='center', rotation='vertical')
    fig.set_size_inches(12, 3)
    fig.savefig(os.path.join(plots_path, "Figure7.pdf"))


def main():
    results_path = "results"
    plots_path = "plots"
    if args.toy_example:
        results_path = os.path.join(results_path, "toy")
        plots_path = os.path.join(plots_path, "toy")
    else:
        results_path = os.path.join(results_path, "paper")
        plots_path = os.path.join(plots_path, "paper")
    evaluation2_3_results_path = os.path.join(results_path, "evaluation2_3")
    evaluation2_3_pipeline_algorithm_results_path = os.path.join(
        evaluation2_3_results_path, "pipeline_algorithm")
    evaluation1_results_path = os.path.join(results_path, "evaluation1")
    plots_path = create_directory(plots_path, "extension")
    new_results_path = create_directory(results_path, "extension")

    prototypes_impact_analysis(
        evaluation1_results_path, evaluation2_3_pipeline_algorithm_results_path, evaluation2_3_results_path, plots_path)
    transformation_analysis(
        evaluation2_3_pipeline_algorithm_results_path, new_results_path, plots_path)
    physical_pipelines_analysis(
        evaluation2_3_pipeline_algorithm_results_path, new_results_path, plots_path)


main()
