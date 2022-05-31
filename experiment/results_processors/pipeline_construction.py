from utils.common import *
from utils.pipeline_construction import experiments_summarizer, experiments_summarizer_10x4cv, graph_maker, graph_maker_10x4cv

args = parse_args()
experiments_summarizer('pipeline_construction', ['features', 'rebalance'], args.toy_example)
experiments_summarizer('pipeline_construction', ['discretize', 'features'], args.toy_example)
experiments_summarizer('pipeline_construction', ['features', 'normalize'], args.toy_example)
experiments_summarizer('pipeline_construction', ['discretize', 'rebalance'], args.toy_example)


experiments_summarizer_10x4cv('pipeline_construction', ['features', 'rebalance'], args.toy_example)
experiments_summarizer_10x4cv('pipeline_construction', ['discretize', 'features'], args.toy_example)
experiments_summarizer_10x4cv('pipeline_construction', ['features', 'normalize'], args.toy_example)
experiments_summarizer_10x4cv('pipeline_construction', ['discretize', 'rebalance'], args.toy_example)

graph_maker(args.toy_example)
graph_maker_10x4cv(args.toy_example)