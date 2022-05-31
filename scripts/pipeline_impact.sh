#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

# SCENARIO GENERETOR
python experiment/scenario_generator.py -exp pipeline_impact "${params[@]}"

# EXPERIMENTS
python experiment/experiments_launcher.py -exp pipeline_impact "${params[@]}"

# PLOTTING
python experiment/results_processors/pipeline_impact.py "${params[@]}"