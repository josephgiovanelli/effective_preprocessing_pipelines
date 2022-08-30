#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

echo ""
echo "### EXPLORATORY ANALYSIS: Pipeline impact ###"
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

echo ""
echo "### EXPLORATORY ANALYSIS: Statistics and meta-learning ###"
echo ""

# EXPLORATORY ANALYSIS
echo "Performing the analysis..."
python experiment/results_processor.py -exp exploratory_analysis "${params[@]}"
echo -e "\tDone."