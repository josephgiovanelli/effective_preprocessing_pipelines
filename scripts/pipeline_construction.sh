#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

# python experiment/scenario_generator.py -exp pipeline_construction "${params[@]}"

# python experiment/experiments_launcher.py -exp pipeline_construction -mode features_rebalance "${params[@]}"
# python experiment/experiments_launcher.py -exp pipeline_construction -mode rebalance_features "${params[@]}"
# python experiment/results_processors/experiments_summarizer.py -exp pipeline_construction -mode features_rebalance "${params[@]}"

# python experiment/experiments_launcher.py -exp pipeline_construction -mode discretize_features "${params[@]}"
# python experiment/experiments_launcher.py -exp pipeline_construction -mode features_discretize "${params[@]}"
# python experiment/results_processors/experiments_summarizer.py -exp pipeline_construction -mode discretize_features "${params[@]}"

# python experiment/experiments_launcher.py -exp pipeline_construction -mode features_normalize "${params[@]}"
# python experiment/experiments_launcher.py -exp pipeline_construction -mode normalize_features "${params[@]}"
# python experiment/results_processors/experiments_summarizer.py  -exp pipeline_construction -mode features_normalize  "${params[@]}"

python experiment/experiments_launcher.py -exp pipeline_construction -mode discretize_rebalance "${params[@]}"
# python experiment/experiments_launcher.py -exp pipeline_construction -mode rebalance_discretize "${params[@]}"
# python experiment/results_processors/experiments_summarizer.py -exp pipeline_construction -mode discretize_rebalance "${params[@]}"

# python experiment/results_processors/graphs_maker.py "${params[@]}"

# python experiment/results_processors/experiments_summarizer_10x4cv.py -exp pipeline_construction -mode features_rebalance "${params[@]}"
# python experiment/results_processors/experiments_summarizer_10x4cv.py -exp pipeline_construction -mode discretize_features "${params[@]}"
# python experiment/results_processors/experiments_summarizer_10x4cv.py -exp pipeline_construction -mode features_normalize "${params[@]}"
# python experiment/results_processors/experiments_summarizer_10x4cv.py -exp pipeline_construction -mode discretize_rebalance "${params[@]}"

# python experiment/results_processors/graphs_maker_10x4cv.py "${params[@]}"