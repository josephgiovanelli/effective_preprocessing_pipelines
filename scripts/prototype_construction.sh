#!/bin/bash

echo ""
echo "### PROTOTYPE CONSTRUCTION (PC) ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios..."
python experiment/scenario_generator.py -exp prototype_construction $1
echo -e "\tDone."

# EXPERIMENTS
echo "PC01. Run experiments with pairs of transformations"
## Features Rebalance
echo -e "\tFeatures - Rebalance"
python experiment/experiments_launcher.py -exp prototype_construction -mode features_rebalance $1
echo -e "\tRebalance - Features"
python experiment/experiments_launcher.py -exp prototype_construction -mode rebalance_features $1
## Features Discretize
echo -e "\tDiscretize - Features"
python experiment/experiments_launcher.py -exp prototype_construction -mode discretize_features $1
echo -e "\tFeatures - Discretize"
python experiment/experiments_launcher.py -exp prototype_construction -mode features_discretize $1
## Features Normalize
echo -e "\tFeatures - Normalize"
python experiment/experiments_launcher.py -exp prototype_construction -mode features_normalize $1
echo -e "\tNormalize - Features"
python experiment/experiments_launcher.py -exp prototype_construction -mode normalize_features $1
## Discretize Rebalance
echo -e "\tDiscretize - Rebalance"
python experiment/experiments_launcher.py -exp prototype_construction -mode discretize_rebalance $1
echo -e "\tRebalance - Discretize"
python experiment/experiments_launcher.py -exp prototype_construction -mode rebalance_discretize $1

# POST-PROCESSING
python experiment/results_processor.py -exp prototype_construction $1