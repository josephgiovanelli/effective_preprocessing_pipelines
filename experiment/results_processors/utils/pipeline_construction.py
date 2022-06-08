import collections
import logging
import os
import json
import warnings
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from yellowbrick.target import FeatureCorrelation
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from scipy import stats as s
from os import listdir
from os.path import isfile, join
from .common import *


def create_possible_categories(pipeline):
    first = pipeline[0][0].upper()
    second = pipeline[1][0].upper()
    first_or_second = first + 'o' + second
    first_second = first + second
    second_first = second + first
    draw = first_second + 'o' + second_first
    baseline = 'baseline'
    inconsistent = 'inconsistent'
    not_exec = 'not_exec'
    not_exec_once = 'not_exec_once'

    return {'first': first,
            'second': second,
            'first_or_second': first_or_second,
            'first_second': first_second,
            'second_first': second_first,
            'draw': draw,
            'baseline': baseline,
            'inconsistent': inconsistent,
            'not_exec': not_exec,
            'not_exec_once': not_exec_once}


def get_filtered_datasets(experiment, toy):
    if experiment == "pipeline_impact":
        return pipeline_impact_suite
    else:
        dataset_suit = list(dict.fromkeys(
            benchmark_suite + extended_benchmark_suite + [10, 20, 26]))
    df = pd.read_csv("meta_features/simple-meta-features.csv")
    df = df.loc[df['did'].isin(list(dict.fromkeys(dataset_suit)))]
    df = df.loc[df['NumberOfMissingValues'] /
                (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] /
                df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    if toy:
        df = df.loc[df['NumberOfInstances'] <= 2000]
        df = df.loc[df['NumberOfFeatures'] <= 10]
        df = df.sort_values(by=['NumberOfInstances', 'NumberOfFeatures'])
        df = df[:10]
    df = df['did']
    return df.values.flatten().tolist()


def merge_dict(list):
    ''' Merge dictionaries and keep values of common keys in list'''
    new_dict = {}
    for key, value in list[0].items():
        new_value = []
        for dict in list:
            new_value.append(dict[key])
        new_dict[key] = new_value
    return new_dict


def load_results(input_path, filtered_datasets):
    comparison = {}
    confs = [os.path.join(input_path, 'conf1'),
             os.path.join(input_path, 'conf2')]
    for path in confs:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']
        comparison[path] = {}
        for algorithm in algorithms:
            for dataset in filtered_datasets:
                acronym = ''.join(
                    [a for a in algorithm if a.isupper()]).lower()
                acronym += '_' + str(dataset)
                if acronym in results:
                    try:
                        with open(os.path.join(path, acronym + '.json')) as json_file:
                            data = json.load(json_file)
                            accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                            pipeline = str(data['context']['best_config']['pipeline']).replace(
                                ' ', '').replace(',', ' ')
                            num_iterations = data['context']['iteration'] + 1
                            best_iteration = data['context']['best_config']['iteration'] + 1
                            baseline_score = data['context']['baseline_score'] // 0.0001 / 100
                    except:
                        accuracy = 0
                        pipeline = ''
                        num_iterations = 0
                        best_iteration = 0
                        baseline_score = 0
                else:
                    accuracy = 0
                    pipeline = ''
                    num_iterations = 0
                    best_iteration = 0
                    baseline_score = 0

                comparison[path][acronym] = {}
                comparison[path][acronym]['accuracy'] = accuracy
                comparison[path][acronym]['baseline_score'] = baseline_score
                comparison[path][acronym]['num_iterations'] = num_iterations
                comparison[path][acronym]['best_iteration'] = best_iteration
                comparison[path][acronym]['pipeline'] = pipeline

    return dict(collections.OrderedDict(sorted(merge_dict([comparison[confs[0]], comparison[confs[1]]]).items())))


def load_algorithm_results(input_path, filtered_datasets):
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']
    for algorithm in algorithms:
        for dataset in filtered_datasets:
            acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
            acronym += '_' + str(dataset)
            if acronym in results:
                with open(os.path.join(input_path, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                    pipeline = str(data['context']['best_config']['pipeline']).replace(
                        ' ', '').replace(',', ' ')
                    num_iterations = data['context']['iteration'] + 1
                    best_iteration = data['context']['best_config']['iteration'] + 1
                    baseline_score = data['context']['baseline_score'] // 0.0001 / 100
            else:
                accuracy = 0
                pipeline = ''
                num_iterations = 0
                best_iteration = 0
                baseline_score = 0

            results_map[acronym] = {}
            results_map[acronym]['accuracy'] = accuracy
            results_map[acronym]['baseline_score'] = baseline_score
            results_map[acronym]['num_iterations'] = num_iterations
            results_map[acronym]['best_iteration'] = best_iteration
            results_map[acronym]['pipeline'] = pipeline

    return results_map


def save_simple_results(result_path, simple_results, filtered_datasets):
    def values_to_string(values):
        return [str(value).replace(',', '') for value in values]

    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists('{}.csv'.format(acronym)):
            os.remove('{}.csv'.format(acronym))
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'w') as out:
            first_element = simple_results[list(simple_results.keys())[0]]
            conf_keys = first_element['conf1'].keys()
            conf1_header = ','.join([a + '1' for a in conf_keys])
            conf2_header = ','.join([a + '2' for a in conf_keys])
            result_header = ','.join(first_element['result'].keys())
            header = ','.join([result_header, conf1_header, conf2_header])
            out.write('dataset,name,dimensions,' + header + '\n')

    df = pd.read_csv('meta_features/simple-meta-features.csv')
    df = df.loc[df['did'].isin(filtered_datasets)]

    for key, value in simple_results.items():
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]
        name = df.loc[df['did'] == int(data_set)]['name'].values.tolist()[0]
        dimensions = ' x '.join([str(int(a)) for a in df.loc[df['did'] == int(data_set)][
            ['NumberOfInstances', 'NumberOfFeatures']].values.flatten().tolist()])

        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'a') as out:
            results = ','.join(values_to_string(value['result'].values()))
            conf1 = ','.join(values_to_string(value['conf1'].values()))
            conf2 = ','.join(values_to_string(value['conf2'].values()))
            row = ','.join([data_set, name, dimensions, results, conf1, conf2])
            out.write(row + '\n')


def compose_pipeline(pipeline1, pipeline2, scheme):
    pipelines = {'pipeline1': [], 'pipeline2': []}
    parameters = {'pipeline1': [], 'pipeline2': []}
    for step in scheme:
        if pipeline1 != '':
            raw_pipeline1 = json.loads(pipeline1.replace('\'', '\"').replace(
                ' ', ',').replace('True', '1').replace('False', '0'))
            pipelines['pipeline1'].append(raw_pipeline1[step][0].split('_')[1])
            for param in raw_pipeline1[step][1]:
                parameters['pipeline1'].append(raw_pipeline1[step][1][param])
        if pipeline2 != '':
            raw_pipeline2 = json.loads(pipeline2.replace('\'', '\"').replace(
                ' ', ',').replace('True', '1').replace('False', '0'))
            pipelines['pipeline2'].append(raw_pipeline2[step][0].split('_')[1])
            for param in raw_pipeline2[step][1]:
                parameters['pipeline2'].append(raw_pipeline2[step][1][param])
    return pipelines, parameters


def have_same_steps(pipelines):
    pipeline1_has_first = not pipelines['pipeline1'][0].__contains__(
        'NoneType')
    pipeline1_has_second = not pipelines['pipeline1'][1].__contains__(
        'NoneType')
    pipeline2_has_first = not pipelines['pipeline2'][0].__contains__(
        'NoneType')
    pipeline2_has_second = not pipelines['pipeline2'][1].__contains__(
        'NoneType')
    both_just_first = pipeline1_has_first and not pipeline1_has_second and pipeline2_has_first and not pipeline2_has_second
    both_just_second = not pipeline1_has_first and pipeline1_has_second and not pipeline2_has_first and pipeline2_has_second
    both_baseline = not pipeline1_has_first and not pipeline1_has_second and not pipeline2_has_first and not pipeline2_has_second
    return both_just_first or both_just_second or both_baseline


def check_validity(pipelines, result, acc1, acc2):
    if pipelines['pipeline1'] == [] and pipelines['pipeline2'] == []:
        validity, problem = False, 'not_exec'
    elif pipelines['pipeline1'] == [] or pipelines['pipeline2'] == []:
        validity, problem = False, 'not_exec_once'
    else:
        if pipelines['pipeline1'].__contains__('NoneType') and pipelines['pipeline2'].__contains__('NoneType'):
            validity = result == 0
        elif pipelines['pipeline1'].__contains__('NoneType') and not(pipelines['pipeline2'].__contains__('NoneType')):
            validity = result == 0 or result == 2
        elif not(pipelines['pipeline1'].__contains__('NoneType')) and pipelines['pipeline2'].__contains__('NoneType'):
            validity = result == 0 or result == 1
        else:
            validity = True
        problem = '' if validity else 'inconsistent'

    if not(validity) and pipelines['pipeline1'] != [] and pipelines['pipeline2'] != []:
        if have_same_steps(pipelines):
            validity, problem, result = True, '', 0

    return validity, problem, result


def compute_result(result, pipelines, categories, baseline_scores, scores):
    if baseline_scores[0] != baseline_scores[1]:
        raise Exception('Baselines with different scores')

    # case a, b, c, e, i
    if result == 0 and (baseline_scores[0] == scores[0] or baseline_scores[1] == scores[1]):
        return 'baseline'
    # case d, o
    elif pipelines['pipeline1'].count('NoneType') == 2 or pipelines['pipeline2'].count('NoneType') == 2:
        if pipelines['pipeline1'].count('NoneType') == 2 and pipelines['pipeline2'].count('NoneType') == 0:
            if result == 2:
                return categories['second_first']
            else:
                raise Exception('pipeline2 is not winning. ' + str(pipelines) +
                                ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores))
        elif pipelines['pipeline1'].count('NoneType') == 0 and pipelines['pipeline2'].count('NoneType') == 2:
            if result == 1:
                return categories['first_second']
            else:
                raise Exception('pipeline1 is not winning. ' + str(pipelines) +
                                ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores))
        else:
            raise Exception('Baseline doesn\'t draw with a pipeline with just one operation. pipelines:' +
                            str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores))
    # case f, m, l, g
    elif pipelines['pipeline1'].count('NoneType') == 1 and pipelines['pipeline2'].count('NoneType') == 1:
        # case f
        if pipelines['pipeline1'][0] == 'NoneType' and pipelines['pipeline2'][0] == 'NoneType':
            if result == 0:
                return categories['second']
            else:
                raise Exception('pipelines is not drawing. ' + str(pipelines) +
                                ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores))
        # case m
        elif pipelines['pipeline1'][1] == 'NoneType' and pipelines['pipeline2'][1] == 'NoneType':
            if result == 0:
                return categories['first']
            else:
                raise Exception('pipelines is not drawing. ' + str(pipelines) + ' baseline_score ' +
                                str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
        # case g, l
        elif (pipelines['pipeline1'][0] == 'NoneType' and pipelines['pipeline2'][1] == 'NoneType') or (pipelines['pipeline1'][1] == 'NoneType' and pipelines['pipeline2'][0] == 'NoneType'):
            if result == 0:
                return categories['first_or_second']
            else:
                raise Exception('pipelines is not drawing. ' + str(pipelines) + ' baseline_score ' +
                                str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
    # case h, n
    elif pipelines['pipeline1'].count('NoneType') == 1:
        # case h
        if pipelines['pipeline1'][0] == 'NoneType':
            if result == 0:
                return categories['second']
            elif result == 2:
                return categories['second_first']
            else:
                raise Exception('pipeline2 is not winning. ' + str(pipelines) + ' baseline_score ' +
                                str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
        # case n
        elif pipelines['pipeline1'][1] == 'NoneType':
            if result == 0:
                return categories['first']
            elif result == 2:
                return categories['second_first']
            else:
                raise Exception('pipeline2 is not winning. ' + str(pipelines) + ' baseline_score ' +
                                str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
    # case p, q
    elif pipelines['pipeline2'].count('NoneType') == 1:
        # case p
        if pipelines['pipeline2'][0] == 'NoneType':
            if result == 0:
                return categories['second']
            elif result == 1:
                return categories['first_second']
            else:
                raise Exception('pipeline1 is not winning. ' + str(pipelines) + ' baseline_score ' +
                                str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
        # case q
        elif pipelines['pipeline2'][1] == 'NoneType':
            if result == 0:
                return categories['first']
            elif result == 1:
                return categories['first_second']
            else:
                raise Exception('pipeline1 is not winning. ' + str(pipelines) + ' baseline_score ' +
                                str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
    # case r
    elif pipelines['pipeline1'].count('NoneType') == 0 and pipelines['pipeline2'].count('NoneType') == 0:
        if result == 0:
            return categories['draw']
        elif result == 1:
            return categories['first_second']
        elif result == 2:
            return categories['second_first']
    else:
        raise Exception('This configuration matches nothing. ' + str(pipelines) +
                        ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')


def instantiate_results(grouped_by_dataset_result, grouped_by_algorithm_results, dataset, acronym, categories):
    if not (grouped_by_dataset_result.__contains__(dataset)):
        grouped_by_dataset_result[dataset] = {}

    if not (grouped_by_algorithm_results.__contains__(acronym)):
        grouped_by_algorithm_results[acronym] = {}
        for _, category in categories.items():
            grouped_by_algorithm_results[acronym][category] = 0


def get_winner(accuracy1, accuracy2):
    if accuracy1 > accuracy2:
        return 1
    elif accuracy1 == accuracy2:
        return 0
    elif accuracy1 < accuracy2:
        return 2
    else:
        raise ValueError('A very bad thing happened.')


def rich_simple_results(simple_results, pipeline_scheme, categories):
    for key, value in simple_results.items():
        first_configuration = value[0]
        second_configuration = value[1]
        pipelines, parameters = compose_pipeline(
            first_configuration['pipeline'], second_configuration['pipeline'], pipeline_scheme)

        try:
            winner = get_winner(
                first_configuration['accuracy'], second_configuration['accuracy'])
        except Exception as e:
            print(str(e))

        validity, label, winner = check_validity(
            pipelines, winner, first_configuration['accuracy'], second_configuration['accuracy'])

        if validity:
            try:
                baseline_scores = [
                    first_configuration['baseline_score'], second_configuration['baseline_score']]
                accuracies = [first_configuration['accuracy'],
                              second_configuration['accuracy']]
                label = compute_result(
                    winner, pipelines, categories, baseline_scores, accuracies)
            except Exception as e:
                print(str(e))
                label = categories['inconsistent']

        first_configuration['pipeline'] = pipelines['pipeline1']
        second_configuration['pipeline'] = pipelines['pipeline2']

        first_configuration['parameters'] = parameters['pipeline1']
        second_configuration['parameters'] = parameters['pipeline2']

        value.append({'winner': winner, 'validity': validity, 'label': label})
        simple_results[key] = {'conf1': value[0],
                               'conf2': value[1], 'result': value[2]}
    return simple_results


def aggregate_results(simple_results, categories):
    grouped_by_dataset_result = {}
    grouped_by_algorithm_results = {}

    for key, value in simple_results.items():
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]
        instantiate_results(grouped_by_dataset_result,
                            grouped_by_algorithm_results, data_set, acronym, categories)

        grouped_by_dataset_result[data_set][acronym] = value['result']['label']
        grouped_by_algorithm_results[acronym][value['result']['label']] += 1

    return grouped_by_algorithm_results, grouped_by_dataset_result


def compute_summary(grouped_by_algorithm_results, categories):
    summary = {}
    for _, category in categories.items():
        summary[category] = sum(x[category]
                                for x in grouped_by_algorithm_results.values())
    return summary


def save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, summary, no_algorithms=False):
    with open(os.path.join(result_path, 'summary.csv'), 'w') as out:
        out.write(',' + ','.join(summary.keys()) + '\n')
        if not(no_algorithms):
            for key, value in grouped_by_algorithm_results.items():
                row = key
                for k, v in value.items():
                    row += ',' + str(v)
                row += '\n'
                out.write(row)
        row = 'summary'
        for key, value in summary.items():
            row += ',' + str(value)
        out.write(row)


def save_details_grouped_by_dataset_result(result_path, details_grouped_by_dataset_result):
    for element in algorithms:
        header = False
        acronym = ''.join([a for a in element if a.isupper()]).lower()

        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'w') as out:

            for dataset, detail_results in details_grouped_by_dataset_result.items():
                if not(header):
                    out.write(
                        ',' + ','.join(detail_results[acronym].keys()) + '\n')
                    header = True
                results = ','.join(list(str(elem).replace(',', '')
                                                 .replace('[', '')
                                                 .replace(']', '')
                                                 .replace('\'', '') for elem in detail_results[acronym].values()))
                out.write(dataset + ',' + results + '\n')


def extract_results(input_path, filtered_data_sets, pipeline, categories):
    # load and format the results
    simple_results = load_results(input_path, filtered_data_sets)
    simple_results = rich_simple_results(simple_results, pipeline, categories)

    # summarize the results
    grouped_by_algorithm_results, grouped_by_data_set_result = aggregate_results(
        simple_results, categories)
    summary = compute_summary(grouped_by_algorithm_results, categories)

    return simple_results, grouped_by_algorithm_results, grouped_by_data_set_result, summary


def compute_summary_from_data_set_results(dataset_results, categories):
    summary = {algorithm: {category: 0 for _, category in categories.items()}
               for algorithm in ['knn', 'nb', 'rf']}
    for _, results in dataset_results.items():
        for algorithm, category in results.items():
            summary[algorithm][category] += 1
    return summary


def extract_results_10x4cv(input_path, filtered_data_sets, pipeline, categories, folds, repeat):
    from sklearn.model_selection import RepeatedKFold

    # load and format the results
    simple_results = load_results(input_path, filtered_data_sets)
    simple_results = rich_simple_results(simple_results, pipeline, categories)

    # summarize the results
    grouped_by_algorithm_results, grouped_by_data_set_result = aggregate_results(
        simple_results, categories)

    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeat, random_state=1)

    summaries = []
    datasets = list(grouped_by_data_set_result.keys())
    for train_index, test_index in rkf.split(datasets):
        train_dict = {datasets[your_key]: grouped_by_data_set_result[datasets[your_key]]
                      for your_key in train_index}
        test_dict = {datasets[your_key]: grouped_by_data_set_result[datasets[your_key]]
                     for your_key in test_index}
        train_per_algorithm = compute_summary_from_data_set_results(
            train_dict, categories)
        train_summary = compute_summary(train_per_algorithm, categories)
        train_per_algorithm['summary'] = train_summary
        test_per_algorithm = compute_summary_from_data_set_results(
            test_dict, categories)
        test_summary = compute_summary(test_per_algorithm, categories)
        test_per_algorithm['summary'] = test_summary
        summaries.append({'train': train_per_algorithm,
                         'test': test_per_algorithm})

    return summaries


def save_results(result_path, filtered_data_sets, simple_results, grouped_by_algorithm_results, summary):
    save_simple_results(result_path, simple_results, filtered_data_sets)
    save_grouped_by_algorithm_results(
        result_path, grouped_by_algorithm_results, summary)


def save_results_10x4cv(result_path, summaries):
    for batch in range(len(summaries)):
        for set_, results in summaries[batch].items():
            with open(os.path.join(result_path, str(batch + 1) + set_ + '.csv'), 'w') as out:
                out.write(',' + ','.join(results['summary'].keys()) + '\n')
                for key, value in results.items():
                    row = key
                    for k, v in value.items():
                        row += ',' + str(v)
                    row += '\n'
                    out.write(row)


def merge_results(pipeline_results, algorithm_results):
    comparison = {}
    summary = {}
    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        summary[acronym] = {'algorithm': 0, 'pipeline': 0, 'draw': 0}
        comparison[acronym] = {}

    for key, value in pipeline_results.items():
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]
        acc1 = pipeline_results[key]['conf1']['accuracy']
        acc2 = pipeline_results[key]['conf2']['accuracy']
        best_pipeline_result = pipeline_results[key]['conf' +
                                                     str(1 if acc1 > acc2 else 2)]

        if algorithm_results[key]['baseline_score'] != best_pipeline_result['baseline_score']:
            print('Different baseline scores: ' + str(key) + ' ' +
                  str(algorithm_results[key]['baseline_score']) + ' ' + str(best_pipeline_result['baseline_score']))

        comparison[acronym][data_set] = {'algorithm': algorithm_results[key]['accuracy'], 'pipeline': best_pipeline_result['accuracy'],
                                         'baseline': algorithm_results[key]['baseline_score'], 'choosen_pipeline': best_pipeline_result['pipeline']}
        winner = 'algorithm' if comparison[acronym][data_set]['algorithm'] > comparison[acronym][data_set]['pipeline'] else (
            'pipeline' if comparison[acronym][data_set]['algorithm'] < comparison[acronym][data_set]['pipeline'] else 'draw')
        summary[acronym][winner] += 1

    new_summary = {'algorithm': 0, 'pipeline': 0, 'draw': 0}
    for algorithm, results in summary.items():
        for category, result in summary[algorithm].items():
            new_summary[category] += summary[algorithm][category]

    summary['summary'] = new_summary

    return comparison, summary


def save_comparison(comparison, result_path):
    def values_to_string(values):
        return [str(value).replace(',', '') for value in values]

    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists('{}.csv'.format(acronym)):
            os.remove('{}.csv'.format(acronym))
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'w') as out:
            keys = comparison[acronym][list(
                comparison[acronym].keys())[0]].keys()
            header = ','.join(keys)
            out.write('dataset,' + header + '\n')
            for dataset, results in comparison[acronym].items():
                result_string = ','.join(values_to_string(results.values()))
                out.write(dataset + ',' + result_string + '\n')


def save_summary(summary, result_path):
    if os.path.exists('summary.csv'):
        os.remove('summary.csv')
    with open(os.path.join(result_path, 'summary.csv'), 'w') as out:
        keys = summary[list(summary.keys())[0]].keys()
        header = ','.join(keys)
        out.write(',' + header + '\n')
        for algorithm, results in summary.items():
            result_string = ','.join([str(elem) for elem in results.values()])
            out.write(algorithm + ',' + result_string + '\n')


def max_frequency(x):
    counter = 0
    num = x[0]

    for i in x:
        curr_frequency = x.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num, counter


def create_num_equal_elements_matrix(grouped_by_dataset_result):
    num_equal_elements_matrix = np.zeros((len(algorithms), len(algorithms)))

    for dataset, value in grouped_by_dataset_result.items():
        list_values = []
        for _, label in value.items():
            if label != 'inconsistent' and label != 'not_exec' and label != 'not_exec_once' and label != 'no_majority':
                list_values.append(label)
        if list_values:
            _, freq = max_frequency(list_values)
            num_equal_elements_matrix[len(list_values) - 1][freq - 1] += 1

    return num_equal_elements_matrix


def save_num_equal_elements_matrix(result_path, num_equal_elements_matrix):
    with open(os.path.join(result_path, 'num_equal_elements_matrix.csv'), 'w') as out:
        out.write('length,' + ','.join(str(i)
                  for i in range(1, len(algorithms) + 1)) + ',tot\n')
        for i in range(0, np.size(num_equal_elements_matrix, 0)):
            row = str(i + 1)
            sum = 0
            for j in range(0, np.size(num_equal_elements_matrix, 1)):
                value = int(num_equal_elements_matrix[i][j])
                sum += value
                row += ',' + str(value)
            row += ',' + str(sum) + '\n'
            out.write(row)


def create_hamming_matrix(X, y):
    def hamming_distance(s1, s2):
        assert len(s1) == len(s2)
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
    hamming_matrix = np.zeros((len(algorithms), len(algorithms)))
    value = np.zeros(len(algorithms))
    value[X[0][1]] = y[0] + 1
    for i in range(1, np.size(X, 0)):
        if X[i][0] == X[i-1][0]:
            value[X[i][1]] = y[i] + 1
        else:
            most_frequent = int(s.mode([x for x in value if x != 0])[0])
            weight = list(value).count(0)
            ideal = np.zeros(len(algorithms))
            for j in range(0, np.size(value, 0)):
                if value[j] != 0:
                    ideal[j] = most_frequent
            hamming_matrix[weight][hamming_distance(value, ideal)] += 1

            value = np.zeros(5)
            value[X[i][1]] = y[i] + 1
    return hamming_matrix


def get_results(grouped_by_dataset_result):
    data = []
    for dataset, value in grouped_by_dataset_result.items():
        for algorithm, result in value.items():
            if result != 'inconsistent' and result != 'not_exec' and result != 'no_majority':
                data.append([int(dataset), algorithm, result])

    df = pd.DataFrame(data)
    df.columns = ['dataset', 'algorithm', 'class']
    return df


def encode_data(data):
    numeric_features = data.select_dtypes(
        include=['int64', 'float64', 'int32', 'float32']).columns
    categorical_features = data.select_dtypes(include=['object']).columns
    reorder_features = list(numeric_features) + list(categorical_features)
    encoded = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('a', FunctionTransformer())]),
             numeric_features),
            ('cat', Pipeline(steps=[('b', OrdinalEncoder())]),
             categorical_features)
        ]).fit_transform(data)
    encoded = pd.DataFrame(encoded, columns=reorder_features)

    return encoded


def join_result_with_extended_meta_features(filtered_datasets, data):
    meta = pd.read_csv('meta_features/extended-meta-features.csv')
    meta = meta.loc[meta['id'].isin(filtered_datasets)]
    meta = meta.drop(columns=['name', 'runs'])

    join = pd.merge(data, meta, left_on='dataset', right_on='id')
    join = join.drop(columns=['id'])

    return join


def join_result_with_extracted_meta_features(data, impute):
    meta = pd.read_csv('meta_features/' + ('imputed-mean-' if impute else '') +
                       'extracted-meta-features.csv', index_col=False)

    join = pd.merge(meta, data, left_on='id', right_on='dataset')
    join = join.drop(columns=['id'])

    return join


def join_result_with_simple_meta_features(filtered_datasets, data):
    meta = pd.read_csv('meta_features/simple-meta-features.csv')
    meta = meta.loc[meta['did'].isin(filtered_datasets)]
    meta = meta.drop(columns=['version', 'status',
                     'format', 'uploader', 'row', 'name'])
    meta = meta.astype(int)

    join = pd.merge(data, meta, left_on='dataset', right_on='did')
    join = join.drop(columns=['did'])

    return join


def modify_class(data, categories, option):
    for key, value in categories.items():
        if option == 'group_all':
            if key == 'first_second' or key == 'second_first' or key == 'not_exec_once':
                data = data.replace(value, 'order')
            else:
                data = data.replace(value, 'no_order')
        if option == 'group_no_order':
            if key != 'first_second' and key != 'second_first' and key != 'not_exec_once':
                data = data.replace(value, 'no_order')
            if key == 'not_exec_once':
                data = data.drop(data.index[data['class'] == value].tolist())

    return data


def create_correlation_matrix(data):
    encoded = encode_data(data)

    kendall = encoded.corr(method='kendall')['class'].to_frame()
    pearson = encoded.corr(method='pearson')['class'].to_frame()
    spearman = encoded.corr(method='spearman')['class'].to_frame()
    kendall.columns = ['kendall']
    pearson.columns = ['pearson']
    spearman.columns = ['spearman']

    correlation_matrix = pd.concat(
        [kendall, pearson, spearman], axis=1, sort=False)

    X, y = encoded.drop(columns=['class']), encoded['class']
    visualizer = FeatureCorrelation(
        method='mutual_info-classification', labels=X.columns)
    visualizer.fit(X, y)

    correlation_matrix = correlation_matrix.drop('class', axis=0)
    correlation_matrix['mutual_info-classification'] = visualizer.scores_.tolist()

    return correlation_matrix


def save_data_frame(result_path, data_frame, index):
    data_frame.to_csv(result_path, index=index)


def save_correlation_matrix(result_path, name, correlation_matrix, group_no_order):
    save_data_frame(os.path.join(result_path, name +
                    ('_grouped' if group_no_order else '') + '.csv'), correlation_matrix, index=True)


def save_train_meta_learner(result_path, name, train_meta_learner, group_no_order):
    save_data_frame(os.path.join(result_path, name +
                    ('_grouped' if group_no_order else '') + '.csv'), train_meta_learner, index=False)


def chi2test(observed, distribution, prob=0.95):
    # the print after the first '->' are valid just if we comparing the observed frequencies with the uniform distribution
    table = [observed, distribution]
    stat, p, dof, expected = chi2_contingency(table)
    # print(table)
    # print(expected)
    # print()

    # interpret test-statistic
    # stat is high as much as the two distribution are different
    critical = chi2.ppf(prob, dof)
    statistic_test = abs(stat) >= critical
    # print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    # if statistic_test:
    #     print('reject H0 -> No similarity with uniform distribution -> there is a majority -> one of them is more frequent')
    # else:
    #     print('fail to reject H0 -> similarity found -> there is NOT a majority -> balanced frequencies')
    # print()

    # interpret p-value
    # p is high as much as the two distribution are similar (the frequencies are balanced)
    alpha = 1.0 - prob
    p_value = p <= alpha
    # print('significance=%.3f, p=%.3f' % (alpha, p))
    # if p_value:
    #     print('reject H0 -> No similarity with uniform distribution -> there is a majority -> one of them is more frequent')
    # else:
    #     print('fail to reject H0 -> similarity found -> there is NOT a majority -> balanced frequencies')
    # print()
    return critical // 0.0001 / 10000, stat // 0.0001 / 10000, statistic_test, alpha // 0.0001 / 10000, p // 0.0001 / 10000, p_value


def chi2tests(grouped_by_algorithm_results, summary, categories, uniform):
    def run_chi2test(observed, uniform, formatted_input):
        tot = sum(observed)
        observed.sort(reverse=True)

        if uniform:
            length = len(observed)
            uniform_frequency = tot / length
            distribution = [uniform_frequency] * length
        else:
            distribution = [tot * 0.9, tot * 0.1]

        critical, stat, statistic_test, alpha, p, p_value = chi2test(
            observed, distribution)

        formatted_output = {'critical': critical,
                            'stat': stat,
                            'statistic_test': statistic_test,
                            'alpha': alpha,
                            'p': p,
                            'p_value_test': p_value}

        formatted_input.update(formatted_output)
        return formatted_input

    grouped_by_algorithm_results['summary'] = summary
    test = {}
    order_test = {}
    not_order_test = {}
    for algorithm, values in grouped_by_algorithm_results.items():

        total = sum(a for a in values.values())
        not_valid = sum([values[categories['inconsistent']],
                        values[categories['not_exec']], values[categories['not_exec_once']]])
        valid = total - not_valid

        order = sum([values[categories['first_second']],
                    values[categories['second_first']]])
        not_order = valid - order

        formatted_input = {'valid': valid,
                           'order': order,
                           'not_order': not_order}

        test[algorithm] = run_chi2test(
            [order, not_order], uniform, formatted_input)

        first_second = values[categories['first_second']]
        second_first = values[categories['second_first']]

        formatted_input = {'order': order,
                           categories['first_second']: first_second,
                           categories['second_first']: second_first}

        order_test[algorithm] = run_chi2test(
            [first_second, second_first], uniform, formatted_input)

        if uniform:
            first = values[categories['first']]
            second = values[categories['second']]
            first_or_second = values[categories['first_or_second']]
            draw = values[categories['draw']]
            baseline = values[categories['baseline']]

            formatted_input = {'not_order': not_order,
                               categories['first']: first,
                               categories['second']: second,
                               categories['first_or_second']: first_or_second,
                               categories['draw']: draw,
                               categories['baseline']: baseline}

            not_order_test[algorithm] = run_chi2test(
                [first_second, second_first], uniform, formatted_input)

    return {'test': test, 'order_test': order_test, 'not_order_test': not_order_test}


def save_chi2tests(result_path, tests):
    def saver(collection, name):
        with open(name, 'w') as out:
            header = False
            for algorithm, values in collection.items():
                if not header:
                    out.write(',' + ','.join(values.keys()) + '\n')
                    header = True
                row = algorithm
                for _, value in values.items():
                    row += ',' + str(value)
                row += '\n'
                out.write(row)

    for key, value in tests.items():
        if value:
            saver(value, os.path.join(result_path, key + '.csv'))

def experiments_summarizer(pipeline, toy):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    path = "raw_results"
    if toy:
        path = os.path.join(path, "toy")
    input_path = os.path.join(path, "pipeline_construction", '_'.join(pipeline))
    result_path = create_directory(input_path, 'summary')
    categories = create_possible_categories(pipeline)
    filtered_data_sets = get_filtered_datasets(
        experiment='pipeline_construction', toy=toy)

    simple_results, grouped_by_algorithm_results, grouped_by_data_set_result, summary = extract_results(input_path,
                                                                                                        filtered_data_sets,
                                                                                                        pipeline,
                                                                                                        categories)

    save_results(create_directory(result_path, 'algorithms_summary'),
                 filtered_data_sets,
                 simple_results,
                 grouped_by_algorithm_results,
                 summary)

    # compute the chi square test
    for uniform in [True, False]:
        temp_result_path = create_directory(result_path, 'chi2tests')
        tests = chi2tests(grouped_by_algorithm_results,
                          summary, categories, uniform)
        if uniform:
            temp_result_path = create_directory(temp_result_path, 'uniform')
        else:
            temp_result_path = create_directory(temp_result_path, 'binary')
        save_chi2tests(temp_result_path, tests)

    # compute the matrix with the number of equal result per data set
    num_equal_elements_matrix = create_num_equal_elements_matrix(
        grouped_by_data_set_result)
    save_num_equal_elements_matrix(create_directory(
        result_path, 'correlations'), num_equal_elements_matrix)

    data = get_results(grouped_by_data_set_result)
    # create the correlation matrices
    for group_no_order in [True, False]:
        join = join_result_with_simple_meta_features(filtered_data_sets, data)
        if group_no_order:
            join = modify_class(join, categories, 'group_no_order')
        correlation_matrix = create_correlation_matrix(join)
        save_correlation_matrix(create_directory(result_path, 'correlations'), 'correlation_matrix', correlation_matrix,
                                group_no_order)

def experiments_summarizer_10x4cv(pipeline, toy):
    # configure environment
    path = "raw_results"
    if toy:
        path = os.path.join(path, "toy")
    input_path = os.path.join(path, "pipeline_construction", '_'.join(pipeline))
    result_path = create_directory(input_path, 'summary')
    result_path = create_directory(result_path, '10x4cv')
    categories = create_possible_categories(pipeline)
    filtered_data_sets = get_filtered_datasets(
        experiment='pipeline_construction', toy=toy)
    folds = 3 if toy else 4
    repeat = 10

    summaries = extract_results_10x4cv(
        input_path, filtered_data_sets, pipeline, categories, folds, repeat)

    save_results_10x4cv(create_directory(result_path, 'raw'), summaries)

    # if (not args.toy_example) or (args.toy_example and args.mode == 'features_rebalance' and args.mode == 'discretize_rebalance'):
    prob = 0.95
    a = 0.1 if categories['first_second'] == 'FN' else (
        0.9 if categories['first_second'] == 'DF' else 0.5)
    b = round(1 - a, 2)
    results = pd.DataFrame()
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    for batch in range(len(summaries)):
        total_train = summaries[batch]['train']['summary'][categories['first_second']
                                                           ] + summaries[batch]['train']['summary'][categories['second_first']]
        total_test = summaries[batch]['test']['summary'][categories['first_second']
                                                         ] + summaries[batch]['test']['summary'][categories['second_first']]
        train_frequencies = [summaries[batch]['train']['summary'][categories['first_second']],
                             summaries[batch]['train']['summary'][categories['second_first']]]
        test_frequencies = [summaries[batch]['test']['summary'][categories['first_second']],
                            summaries[batch]['test']['summary'][categories['second_first']]]
        critical, stat, statistic_test, alpha, p, p_value = chi2test(
            train_frequencies, test_frequencies, prob)
        critical_train, stat_train, statistic_test_train, alpha_train, p_train, p_value_train = chi2test(
            train_frequencies, [total_train * a, total_train * b], prob)
        critical_test, stat_test, statistic_test_test, alpha_test, p_test, p_value_test = chi2test(
            test_frequencies, [total_test * a, total_test * b], prob)
        results = results.append({'critical': critical, 'stat': stat, 'statistic_test': statistic_test,
                                 'alpha': alpha, 'p': p, 'p_value': p_value}, ignore_index=True)
        results_train = results_train.append({'critical': critical_train, 'stat': stat_train, 'statistic_test': statistic_test_train,
                                             'alpha': alpha_train, 'p': p_train, 'p_value': p_value_train}, ignore_index=True)
        results_test = results_test.append({'critical': critical_test, 'stat': stat_test, 'statistic_test': statistic_test_test,
                                           'alpha': alpha_test, 'p': p_test, 'p_value': p_value_test}, ignore_index=True)
    results.to_csv(os.path.join(result_path, 'summary.csv'), index=False)
    results_train.to_csv(os.path.join(
        result_path, 'summary_train.csv'), index=False)
    results_test.to_csv(os.path.join(
        result_path, 'summary_test.csv'), index=False)

    results = pd.DataFrame()
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    for repetition in range(repeat):
        batch_results = pd.DataFrame()
        batch_train_results = pd.DataFrame()
        batch_test_results = pd.DataFrame()
        for fold in range(folds):
            batch = repetition * folds + fold
            total_train = summaries[batch]['train']['summary'][categories['first_second']
                                                               ] + summaries[batch]['train']['summary'][categories['second_first']]
            total_test = summaries[batch]['test']['summary'][categories['first_second']
                                                             ] + summaries[batch]['test']['summary'][categories['second_first']]
            train_frequencies = [summaries[batch]['train']['summary'][categories['first_second']],
                                 summaries[batch]['train']['summary'][categories['second_first']]]
            test_frequencies = [summaries[batch]['test']['summary'][categories['first_second']],
                                summaries[batch]['test']['summary'][categories['second_first']]]
            critical, stat, statistic_test, alpha, p, p_value = chi2test(
                train_frequencies, test_frequencies, prob)
            critical_train, stat_train, statistic_test_train, alpha_train, p_train, p_value_train = chi2test(
                train_frequencies, [total_train * a, total_train * b], prob)
            critical_test, stat_test, statistic_test_test, alpha_test, p_test, p_value_test = chi2test(
                test_frequencies, [total_test * a, total_test * b], prob)
            batch_results = batch_results.append(
                {'critical': critical, 'stat': stat, 'statistic_test': statistic_test, 'alpha': alpha, 'p': p, 'p_value': p_value}, ignore_index=True)
            batch_train_results = batch_train_results.append(
                {'critical': critical_train, 'stat': stat_train, 'statistic_test': statistic_test_train, 'alpha': alpha_train, 'p': p_train, 'p_value': p_value_train}, ignore_index=True)
            batch_test_results = batch_test_results.append(
                {'critical': critical_test, 'stat': stat_test, 'statistic_test': statistic_test_test, 'alpha': alpha_test, 'p': p_test, 'p_value': p_value_test}, ignore_index=True)
        batch_results = batch_results.mean().round(3)
        batch_train_results = batch_train_results.mean().round(3)
        batch_test_results = batch_test_results.mean().round(3)
        results = results.append(batch_results, ignore_index=True)
        results_train = results_train.append(
            batch_train_results, ignore_index=True)
        results_test = results_test.append(
            batch_test_results, ignore_index=True)
    results.to_csv(os.path.join(
        result_path, 'summary_with_mean_.csv'), index=False)
    results_train.to_csv(os.path.join(
        result_path, 'summary_train_with_mean_.csv'), index=False)
    results_test.to_csv(os.path.join(
        result_path, 'summary_test_with_mean_.csv'), index=False)

def graph_maker(toy):
    logging.getLogger('matplotlib.font_manager').disabled = True
    input_path = "raw_results"
    result_path = "artifacts"
    if toy:
        input_path = os.path.join(input_path, "toy")
        result_path = os.path.join(result_path, "toy")
    input_path = os.path.join(input_path, "pipeline_construction")
    result_path = create_directory(result_path, "pipeline_construction")

    data = {}
    data[0] = {'title': r'$T_1$ = Feat. Eng., $T_2$ = Normalize', 'data': pd.read_csv(os.path.join(input_path, 'features_normalize', 'summary', 'algorithms_summary', 'summary.csv')).reindex([1, 0, 2, 3])}
    data[1] = {'title': r'$T_1$ = Discretize, $T_2$ = Feat. Eng.', 'data': pd.read_csv(os.path.join(input_path, 'discretize_features', 'summary', 'algorithms_summary', 'summary.csv')).reindex([1, 0, 2, 3])}
    data[2] = {'title': r'$T_1$ = Feat. Eng., $T_2$ = Rebalance', 'data': pd.read_csv(os.path.join(input_path, 'features_rebalance', 'summary', 'algorithms_summary', 'summary.csv')).reindex([1, 0, 2, 3])}
    data[3] = {'title': r'$T_1$ = Discretize, $T_2$ = Rebalance', 'data': pd.read_csv(os.path.join(input_path, 'discretize_rebalance', 'summary', 'algorithms_summary', 'summary.csv')).reindex([1, 0, 2, 3])}
    labels = [r'$T_1$', r'$T_2$', r'$T_1 \to T_2$', r'$T_2 \to T_1$', 'Baseline']
    colors = ['gold', 'mediumspringgreen', 'royalblue', 'sienna', 'mediumpurple', 'salmon']
    patterns = ["/", "\\", "o", "-", "x", ".", "O", "+", "*", "|"]

    SMALL_SIZE = 8
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(2, 2)
    n_groups = 3

    for i in range(0, 2):
        for j in range(0, 2):
            #fig2, ax = axs[i, j].subplots()
            data[i * 2 + j]['data'].columns = list(range(0, len(data[i * 2 + j]['data'].columns)))
            data[i * 2 + j]['data'] = data[i * 2 + j]['data'].drop(columns=[3, 6])
            index = np.arange(n_groups)
            bar_width = 0.4

            for k in range(1, 6):
                axs[i, j].bar((index * bar_width * 8) + (bar_width * (k - 1)), data[i * 2 + j]['data'].iloc[:-1, k], bar_width, label=labels[k - 1], color=colors[k - 1], hatch=patterns[k - 1])

            axs[i, j].set(ylabel='Number of wins')
            axs[i, j].set(xlabel='Algorithms')
            axs[i, j].set_title(data[i * 2 + j]['title'])
            axs[i, j].set_ylim([0, 40])
            plt.setp(axs, xticks=(index * bar_width * 8) + 0.8, xticklabels=['NB', 'KNN', 'RF'])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 8, bbox_to_anchor=(0.5, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=axs[1,1].transAxes)
    fig.set_size_inches(20, 10, forward=True)
    fig.tight_layout(h_pad=3.0, w_pad=4.0)
    fig.savefig(os.path.join(result_path, 'Figure4.pdf'), bbox_extra_artists=(lgd,text), bbox_inches='tight')

def graph_maker_10x4cv(toy):
    logging.getLogger('matplotlib.font_manager').disabled = True
    cv_file_name = 'summary_with_mean_.csv'
    pipeline_construction_path = 'raw_results'
    plot_path = 'artifacts'
    if toy:
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
    fig.savefig(os.path.join(plot_path, 'Figure5.pdf'))