from __future__ import print_function

import argparse
import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt


from evaluation_utils import get_filtered_datasets, load_results_pipelines, load_results_auto
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
    plots_path = create_directory(plots_path, "extension")
    
    filtered_data_sets = ['_'.join(i) for i in list(itertools.product(["knn", "nb", "rf"], [str(integer) for integer in get_filtered_datasets()]))]
    #print(filtered_data_sets)

    results_pipelines = load_results_pipelines(evaluation1_results_path, filtered_data_sets)
    #print(results_pipelines)
    results_auto = load_results_auto(evaluation2_3_pipeline_algorithm_results_path, filtered_data_sets)
    #print(results_auto)
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            if results_auto[algorithm][dataset][1] == 0:
                evaluation_3 = pd.read_csv(os.path.join(evaluation2_3_results_path, "summary", "evaluation3", algorithm + ".csv"))
                evaluation_3 = evaluation_3.set_index(['dataset'])
                results_auto[algorithm][dataset] = (results_auto[algorithm][dataset][0], evaluation_3.loc[int(dataset)]["baseline"])
    #print(results_auto)
    impacts = pd.DataFrame()
    for algorithm in results_auto.keys():
        for dataset in results_auto[algorithm].keys():
            try:
                for elem in results_pipelines[algorithm][dataset]:
                    impacts = impacts.append({'algorithm': algorithm, 'dataset': dataset, 'index': int(elem['index']) + 1, 'impact': elem['accuracy']-results_auto[algorithm][dataset][1]}, ignore_index=True)
            except:
                print(dataset)
    #print(impacts)
    knn = impacts[impacts["algorithm"] == "knn"]
    nb = impacts[impacts["algorithm"] == "nb"]
    rf = impacts[impacts["algorithm"] == "rf"]
    knn = knn.pivot(index = 'dataset', columns='index', values = 'impact')
    nb = nb.pivot(index = 'dataset', columns='index', values = 'impact')
    rf = rf.pivot(index = 'dataset', columns='index', values = 'impact')
    data = [nb, knn, rf]
    fig, ax = plt.subplots(1, 3, sharey= True, constrained_layout=True)
    for i in range(len(data)):
        ax[i].boxplot(data[i], showfliers=False)
        ax[i].set_title("NB" if i == 0 else ("KNN" if i == 1 else "RF"))
        ax[i].set_xlabel('Prototypes ID')
        ax[i].set_xticklabels(range(1,25), fontsize=8)
    fig.text(0.0, 0.5, 'Impact over the baseline', va='center', rotation='vertical')
    fig.set_size_inches(12, 3)
    fig.savefig(os.path.join(plots_path, "prototypes_impact.pdf"))


main()