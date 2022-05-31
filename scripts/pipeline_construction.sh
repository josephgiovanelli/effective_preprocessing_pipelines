#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

# SCENARIO GENERETOR
python experiment/scenario_generator.py -exp pipeline_construction "${params[@]}"

# EXPERIMENTS
## Features Rebalance
python experiment/experiments_launcher.py -exp pipeline_construction -mode features_rebalance "${params[@]}"
python experiment/experiments_launcher.py -exp pipeline_construction -mode rebalance_features "${params[@]}"
## Features Discretize
python experiment/experiments_launcher.py -exp pipeline_construction -mode discretize_features "${params[@]}"
python experiment/experiments_launcher.py -exp pipeline_construction -mode features_discretize "${params[@]}"
## Features Normalize
python experiment/experiments_launcher.py -exp pipeline_construction -mode features_normalize "${params[@]}"
python experiment/experiments_launcher.py -exp pipeline_construction -mode normalize_features "${params[@]}"
## Discretize Rebalance
python experiment/experiments_launcher.py -exp pipeline_construction -mode discretize_rebalance "${params[@]}"
python experiment/experiments_launcher.py -exp pipeline_construction -mode rebalance_discretize "${params[@]}"

# PLOTTING
python experiment/results_processors/pipeline_construction.py "${params[@]}"