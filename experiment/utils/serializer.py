from datetime import datetime
import json
import os

def serialize_results(scenario, policy, result_path, pipeline=None):
    results = {
        'scenario': scenario,
        'context': policy.context
    }
    if pipeline:
        results['pipeline'] = pipeline
    path = os.path.join(result_path, '{}.json'.format(scenario['file_name']))
    with open(path, 'w') as outfile:
        json.dump(results, outfile, indent=4)