#!/bin/bash

# python experiment/scenario_generator.py -exp experiment/pipeline_construction -toy $1

# python experiment/experiments_launcher.py -exp pipeline_construction -mode features_rebalance -toy $1
# python experiment/experiments_launcher.py -exp pipeline_construction -mode rebalance_features -toy $1
# python experiment/results_processors/experiments_summarizer.py -exp pipeline_construction -mode features_rebalance -toy $1

# python experiment/experiments_launcher.py -exp pipeline_construction -mode discretize_features -toy $1
# python experiment/experiments_launcher.py -exp pipeline_construction -mode features_discretize -toy $1
# python experiment/results_processors/experiments_summarizer.py -exp pipeline_construction -mode discretize_features -toy $1

# python experiment/experiments_launcher.py -exp pipeline_construction -mode features_normalize -toy $1
# python experiment/experiments_launcher.py -exp pipeline_construction -mode normalize_features -toy $1
# python experiment/results_processors/experiments_summarizer.py  -exp pipeline_construction -mode features_normalize  -toy $1

# python experiment/experiments_launcher.py -exp pipeline_construction -mode discretize_rebalance -toy $1
# python experiment/experiments_launcher.py -exp pipeline_construction -mode rebalance_discretize -toy $1
# python experiment/results_processors/experiments_summarizer.py -exp pipeline_construction -mode discretize_rebalance -toy $1

# python experiment/results_processors/graphs_maker.py -toy $1

# python experiment/results_processors/experiments_summarizer_10x4cv.py -exp pipeline_construction -mode features_rebalance -toy $1
# python experiment/results_processors/experiments_summarizer_10x4cv.py -exp pipeline_construction -mode discretize_features -toy $1
# python experiment/results_processors/experiments_summarizer_10x4cv.py -exp pipeline_construction -mode features_normalize -toy $1
# python experiment/results_processors/experiments_summarizer_10x4cv.py -exp pipeline_construction -mode discretize_rebalance -toy $1

# python experiment/results_processors/graphs_maker_10x4cv.py -toy $1