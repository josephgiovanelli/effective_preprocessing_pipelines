#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

# EXPLORATORY ANALYSIS
python experiment/results_processors/exploratory_analysis.py "${params[@]}"