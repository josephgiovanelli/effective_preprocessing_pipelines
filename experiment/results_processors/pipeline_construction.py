from utils.common import *
from utils.pipeline_construction import experiments_summarizer, experiments_summarizer_10x4cv, graph_maker, graph_maker_10x4cv

def main():
    args = parse_args()
    experiments_summarizer(pipeline=['features', 'rebalance'], toy=args.toy_example)
    experiments_summarizer(pipeline=['discretize', 'features'], toy=args.toy_example)
    experiments_summarizer(pipeline=['features', 'normalize'], toy=args.toy_example)
    experiments_summarizer(pipeline=['discretize', 'rebalance'], toy=args.toy_example)


    experiments_summarizer_10x4cv(pipeline=['features', 'rebalance'], toy=args.toy_example)
    experiments_summarizer_10x4cv(pipeline=['discretize', 'features'], toy=args.toy_example)
    experiments_summarizer_10x4cv(pipeline=['features', 'normalize'], toy=args.toy_example)
    experiments_summarizer_10x4cv(pipeline=['discretize', 'rebalance'], toy=args.toy_example)

    graph_maker(args.toy_example)
    graph_maker_10x4cv(args.toy_example)

main()