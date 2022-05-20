from sklearn.impute import SimpleImputer

from experiment.pipeline.PrototypeSingleton import PrototypeSingleton
from experiment.utils import scenarios, serializer, cli, datasets
from experiment import policies

import json
import os
import openml

from sklearn.model_selection import train_test_split

def load_dataset(id, args):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )
    if args.experiment == 'pipeline_construction' or (args.experiment == 'evaluation2_3' and args.mode == "algorithm"):
        X = SimpleImputer(strategy="constant").fit_transform(X)
    print(dataset.name)
    print(X, y)
    PrototypeSingleton.getInstance().setPipeline(args.pipeline)
    num_features = [i for i, x in enumerate(categorical_indicator) if x == False]
    cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    print("numeriche: " + str(len(num_features)) + " categoriche: " + str(len(cat_features)))
    PrototypeSingleton.getInstance().setFeatures(num_features, cat_features)
    PrototypeSingleton.getInstance().set_X_y(X, y)
    return X, y

def main(args):
    scenario = scenarios.load(args.scenario)
    scenario = cli.apply_scenario_customization(scenario, args.customize)
    config = scenarios.to_config(scenario, args)
    
    if args.experiment == 'evaluation2_3':
        if args.mode == "pipeline_algorithm":
            config['time'] /= args.num_pipelines
            config['step_pipeline'] /= args.num_pipelines
        else:
            if args.num_pipelines == 0:
                config['time'] = 400
            else:
                config['time'] = 240
                config['step_pipeline'] = 40

    print('SCENARIO:\n {}'.format(json.dumps(scenario, indent=4, sort_keys=True)))

    X, y = load_dataset(scenario['setup']['dataset'], args)

    policy = policies.initiate(scenario['setup']['policy'], config)
    policy.run(X, y)

    serializer.serialize_results(scenario, policy, args.result_path, args.pipeline)

if __name__ == "__main__":
    args = cli.parse_args()
    main(args)