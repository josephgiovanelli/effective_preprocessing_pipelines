from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_covtype
from .common import *


def load(name):
    loader = {
        'breast': breast_cancer,
        'iris': iris,
        'wine': wine,
        'digits': digits,
        'covtype': covtype,
        #'echr_article_1': echr.binary.get_dataset(article='1', flavors=[echr.Flavor.desc]).load
    }
    if name in loader:
        return loader[name]()
    else:
        print('Invalid dataset. Possible choices: {}'.format(
            ', '.join(loader.keys())
        ))
        exit(1)  # TODO: Throw exception

def breast_cancer():
    data = load_breast_cancer()
    return data.data, data.target

def iris():
    data = load_iris()
    return data.data, data.target

def wine():
    data = load_wine()
    return data.data, data.target

def digits():
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def covtype():
    data = fetch_covtype
    return data.data, data.target

def load_from_openml(id):
    import openml
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )
    return X, y, categorical_indicator

def load_from_csv(id, input_path = DATASETS_PATH):
    import pandas as pd
    import json
    df = pd.read_csv(os.path.join(input_path, f"{id}.csv"))
    with open(os.path.join(input_path, "categorical_indicators.json")) as f:
        categorical_indicators = json.load(f)
    categorical_indicator = categorical_indicators[str(id)]   
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    return X, y, categorical_indicator
