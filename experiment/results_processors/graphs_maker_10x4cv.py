import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
parser.add_argument("-toy", "--toy-example", nargs="?", type=bool, required=False, default=False, help="wether it is a toy example or not")
args = parser.parse_args()

def main():
    cv_file_name = 'summary_with_mean_.csv'
    pipeline_construction_path = 'results'
    plot_path = 'plots'
    if args.toy_example:
        pipeline_construction_path = os.path.join(pipeline_construction_path, "toy")
        plot_path = os.path.join(plot_path, "toy")
    pipeline_construction_path = os.path.join(pipeline_construction_path, 'pipeline_construction')
    plot_path = os.path.join(plot_path, 'pipeline_construction')

    fn_path = os.path.join(pipeline_construction_path, 'features_normalize')
    fn_cv_path = os.path.join(fn_path, 'summary', '10x4cv')
    fn_df = pd.read_csv(os.path.join(fn_cv_path, cv_file_name))
    fn_df['fn'] = fn_df['p']

    df_path = os.path.join(pipeline_construction_path, 'discretize_features')
    df_cv_path = os.path.join(df_path, 'summary', '10x4cv')
    df_df = pd.read_csv(os.path.join(df_cv_path, cv_file_name))
    df_df['df'] = df_df['p']

    df = pd.concat([fn_df['fn'], df_df['df']], axis=1)
    
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    fig, ax = plt.subplots()
    ax.boxplot(df, widths = 0.3)
    ax.axhline(y = 0.05, color = 'grey', linestyle = '--')
    ax.set_xticklabels([r'$F \rightarrow N$', r'$D \rightarrow F$'])
    ax.set_ylabel('Means of the p-values')
    ax.set_yticks([0., 0.05, 0.2, 0.4, 0.6, 0.8, 1.])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.set_size_inches(12, 6, forward=True)
    fig.savefig(os.path.join(plot_path, '10_times_4_folds_cv.pdf'))


main()