#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

echo ""
echo "### EXPLORATORY ANALYSIS ###"
echo ""

# EXPLORATORY ANALYSIS
echo "Performing the analysis..."
python experiment/results_processor.py -exp exploratory_analysis "${params[@]}"
echo -e "\tDone."