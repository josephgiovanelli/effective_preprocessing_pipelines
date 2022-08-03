#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

echo ""
echo "### PIPELINE IMPACT ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios..."
python experiment/scenario_generator.py -exp pipeline_impact "${params[@]}"
echo -e "\tDone."

# EXPERIMENTS
echo "Running experiments..."
python experiment/experiments_launcher.py -exp pipeline_impact "${params[@]}"

# PLOTTING
echo "Plotting..."
python experiment/results_processor.py -exp pipeline_impact "${params[@]}"
echo -e "\tDone."