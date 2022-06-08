#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

echo ""
echo "### EXPLORATORY ANALYSIS ###"
echo ""

# EXPLORATORY ANALYSIS
echo "Performing the analysis..."
python experiment/results_processors/exploratory_analysis.py "${params[@]}"
echo -e "\tDone."