#!/bin/bash

echo ""
echo "### EXPLORATORY ANALYSIS (EA): Pipeline impact ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios..."
python experiment/scenario_generator.py -exp pipeline_impact $1
echo -e "\tDone."

# EXPERIMENTS
echo "Running experiments..."
python experiment/experiments_launcher.py -exp pipeline_impact $1

# PLOTTING
echo "Plotting..."
python experiment/results_processor.py -exp pipeline_impact $1
echo -e "\tDone."

echo ""
echo "### EXPLORATORY ANALYSIS (EA): Statistics and meta-learning ###"
echo ""

# EXPLORATORY ANALYSIS
echo "Performing the analysis..."
python experiment/results_processor.py -exp exploratory_analysis $1
echo -e "\tDone."