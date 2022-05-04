import os
import signal
from functools import reduce

import yaml
from six import iteritems
import time
import subprocess
import datetime

from prettytable import PrettyTable
from tqdm import tqdm

import argparse

from results_processors.utils import create_directory

parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=False, help="step of the pipeline to execute")
parser.add_argument("-r", "--result_path", nargs="?", type=str, required=True, help="path where put the results")
parser.add_argument("-exp", "--experiment", nargs="?", type=str, required=True, help="type of the experiments")
parser.add_argument("-mode", "--mode", nargs="?", type=str, required=False, help="algorithm or algorithm_pipeline")
args = parser.parse_args()


scenario_path = create_directory("./", "scenarios")
scenario_path = create_directory(scenario_path, args.experiment)
if args.experiment == "preprocessing_impact":
    scenario_path = create_directory(scenario_path, args.mode
    )
result_path = './'
for directory in args.result_path.split("/"):
    result_path = create_directory(result_path , str(directory))
result_path = create_directory(result_path , args.mode)
GLOBAL_SEED = 42

def yes_or_no(question):
    while True:
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

# Gather list of scenarios
scenario_list = [p for p in os.listdir(scenario_path) if '.yaml' in p]
result_list = [p for p in os.listdir(result_path) if '.json' in p]
scenarios = {}

# Determine which one have no result files
for scenario in scenario_list:
    base_scenario = scenario.split('.yaml')[0]
    if scenario not in scenarios:
        scenarios[scenario] = {'results': None, 'path': scenario}
    for result in result_list:
        base_result = result.split('.json')[0]
        if base_result.__eq__(base_scenario):
            scenarios[scenario]['results'] = result
            #date = base_result.split(base_scenario + '_')[-1].replace('_', ' ')
            #scenarios[scenario]['results_date'] = date

# Calculate total amount of time
total_runtime = 0
for path, scenario in iteritems(scenarios):
    with open(os.path.join(scenario_path, path), 'r') as f:
        details = None
        try:
            details = yaml.safe_load(f)
        except Exception:
            details = None
            scenario['status'] = 'Invalid YAML'
        if details is not None:
            try:
                runtime = details['setup']['runtime']
                scenario['status'] = 'Ok'
                scenario['runtime'] = runtime
                if scenario['results'] is None:
                    total_runtime += runtime
            except:
                scenario['status'] = 'No runtime info'

# Display list of scenario to be run
invalid_scenarios = {k:v for k,v in iteritems(scenarios) if v['status'] != 'Ok'}
t_invalid = PrettyTable(['PATH', 'STATUS'])
t_invalid.align["PATH"] = "l"
for v in invalid_scenarios.values():
    t_invalid.add_row([v['path'], v['status']])

scenario_with_results = {k:v for k,v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is not None}
t_with_results = PrettyTable(['PATH', 'RUNTIME',  'STATUS', 'RESULTS'])
t_with_results.align["PATH"] = "l"
t_with_results.align["RESULTS"] = "l"
for v in scenario_with_results.values():
    t_with_results.add_row([v['path'], str(v['runtime']) + 's', v['status'], v['results']])

to_run = {k:v for k,v in iteritems(scenarios) if v['status'] == 'Ok' and v['results'] is None}
t_to_run = PrettyTable(['PATH', 'RUNTIME', 'STATUS'])
t_to_run.align["PATH"] = "l"
for v in to_run.values():
    t_to_run.add_row([v['path'], str(v['runtime']) + 's', v['status']])

print('# INVALID SCENARIOS')
print(t_invalid)

print
print('# SCENARIOS WITH AVAILABLE RESULTS')
print(t_with_results)

print
print('# SCENARIOS TO BE RUN')
print(t_to_run)
print('TOTAL RUNTIME: {} ({}s)'.format(datetime.timedelta(seconds=total_runtime), total_runtime))
print

print("The total runtime is {}.".format(datetime.timedelta(seconds=total_runtime)))
print

import psutil


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

with tqdm(total=total_runtime) as pbar:
    for info in to_run.values():
        base_scenario = info['path'].split('.yaml')[0]
        output = base_scenario.split('_')[0]
        pbar.set_description("Running scenario {}\n\r".format(info['path']))

        if args.experiment == "pipeline_construction":
            pipeline = args.pipeline
        else:
            if base_scenario.startswith("knn"):
                pipeline = ['impute', 'encode', 'normalize', 'rebalance', 'features']
            elif base_scenario.startswith("nb"):
                pipeline = ['impute', 'encode', 'normalize', 'features', 'rebalance']
            else:
                pipeline = ['impute', 'encode', 'normalize', 'rebalance', 'features']

        cmd = 'python ./main.py -s {} -c control.seed={} -p {} -r {} -exp {}'.format(
            os.path.join(scenario_path, info['path']),
            GLOBAL_SEED,
            reduce(lambda x, y: x + " " + y, pipeline),
            result_path,
            args.experiment)
        with open(os.path.join(result_path, '{}_stdout.txt'.format(base_scenario)), "a") as log_out:
            with open(os.path.join(result_path, '{}_stderr.txt'.format(base_scenario)), "a") as log_err:
                max_time = 1000
                try:
                    process = subprocess.Popen(cmd, shell=True, stdout=log_out, stderr=log_err)
                    process.wait(timeout = max_time)
                except:
                    kill(process.pid)
                    print("\n\n"+ base_scenario + " does not finish in " + str(max_time) + "\n\n" )

        pbar.update(info['runtime'])
