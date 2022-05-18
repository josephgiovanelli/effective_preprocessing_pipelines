import numpy as np
import openml
import os
import pandas as pd

from enum import Enum
class UndefinedOrders(Enum):
    features_rebalance = 1
    discretize_rebalance = 2

class DefinedOrders(Enum):
    first_second = 1
    second_first = 2

def load_metafeatures(id):
    df = pd.read_csv('results_processors/meta_features/extracted-meta-features.csv')
    df = df.loc[df['id'] == id]
    df = df.drop(['id'], axis=1)
    df = df.fillna("na")
    return df

#['impute encode normalize rebalance features', 'impute encode rebalance discretize features']
def build_pipeline(features_rebalance_order, discretize_rebalance_order):
    pipelines = []
    if features_rebalance_order == DefinedOrders.first_second:
        pipelines.append("impute encode normalize features rebalance")
    else:
        pipelines.append("impute encode normalize rebalance features")

    pipeline = "impute encode "

    if features_rebalance_order == DefinedOrders.first_second and discretize_rebalance_order == DefinedOrders.second_first:
        pipelines.append(pipeline + "rebalance discretize features")
        pipelines.append(pipeline + "discretize features rebalance")
    else:
        if features_rebalance_order == DefinedOrders.first_second and discretize_rebalance_order == DefinedOrders.first_second:
            pipeline += "discretize features rebalance"
        elif features_rebalance_order == DefinedOrders.second_first and discretize_rebalance_order == DefinedOrders.first_second:
            pipeline += "discretize rebalance features"
        elif features_rebalance_order == DefinedOrders.second_first and discretize_rebalance_order == DefinedOrders.second_first:
            pipeline += "rebalance discretize features"
        pipelines.append(pipeline)

    return pipelines

def framework_table_pipelines():
    from itertools import permutations

    all_perm = permutations(['impute', 'encode', 'normalize', 'discretize', 'features', 'rebalance'])
    valid_perm = []
    pipelines = []

    for i in list(all_perm):
        if i.index('encode') < i.index('normalize') and \
                i.index('discretize') > i.index('encode') and \
                i.index('impute') < i.index('encode') and \
                i.index('impute') < i.index('discretize') and \
                i.index('rebalance') > i.index('encode') and \
                i.index('rebalance') > i.index('impute') and \
                i.index('features') > i.index('encode') and \
                i.index('features') > i.index('impute'):
            valid_perm.append(i)

    for i in list(valid_perm):
        pipelines.append(' '.join(i))

    return pipelines



