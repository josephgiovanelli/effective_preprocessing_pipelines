from .iterative import Iterative
from .split import Split
from .adaptive import Adaptive
from .joint import Joint

def initiate(name, config):
    policies = {
        'iterative': Iterative,
        'split': Split,
        'adaptive': Adaptive,
        'joint': Joint
    }
    if name in policies:
        return policies[name](config)
    else:
        print('Invalid dataset. Possible choices: {}'.format(
            ', '.join(policies.keys())
        ))
        exit(1)  # TODO: Throw exception