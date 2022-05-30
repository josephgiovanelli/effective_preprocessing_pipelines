from datetime import datetime
import json
import os

def serialize_results(scenario, result_path, policy=None, pipeline=None):
    results = {
        'scenario': scenario,
    }
    if policy and policy.context:
        results['context'] = policy.context
    else:
        results['context'] = f"The experiment did not finish in time."
    if pipeline:
        results['pipeline'] = pipeline
    path = os.path.join(result_path, '{}.json'.format(scenario['file_name']))
    with open(path, 'w') as outfile:
        json.dump(results, outfile, indent=4)