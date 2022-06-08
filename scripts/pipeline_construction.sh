#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

echo ""
echo "### PIPELINE CONSTRUCTION ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios..."
python experiment/scenario_generator.py -exp pipeline_construction "${params[@]}"
echo -e "\tDone."

# EXPERIMENTS
echo "Running experiments..."
## Features Rebalance
echo -e "\tFeatures Rebalance"
python experiment/experiments_launcher.py -exp pipeline_construction -mode features_rebalance "${params[@]}"
echo -e "\tRebalance Features"
python experiment/experiments_launcher.py -exp pipeline_construction -mode rebalance_features "${params[@]}"
## Features Discretize
echo -e "\tDiscretize Features"
python experiment/experiments_launcher.py -exp pipeline_construction -mode discretize_features "${params[@]}"
echo -e "\tFeatures Discretize"
python experiment/experiments_launcher.py -exp pipeline_construction -mode features_discretize "${params[@]}"
## Features Normalize
echo -e "\tFeatures Normalize"
python experiment/experiments_launcher.py -exp pipeline_construction -mode features_normalize "${params[@]}"
echo -e "\tNormalize Features"
python experiment/experiments_launcher.py -exp pipeline_construction -mode normalize_features "${params[@]}"
## Discretize Rebalance
echo -e "\tDiscretize Rebalance"
python experiment/experiments_launcher.py -exp pipeline_construction -mode discretize_rebalance "${params[@]}"
echo -e "\tRebalance Discretize"
python experiment/experiments_launcher.py -exp pipeline_construction -mode rebalance_discretize "${params[@]}"

# PLOTTING
echo "Plotting..."
python experiment/results_processors/pipeline_construction.py "${params[@]}"
echo -e "\tDone."