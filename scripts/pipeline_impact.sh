#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

python experiment/scenario_generator.py -exp pipeline_impact "${params[@]}"

python experiment/experiments_launcher.py -exp pipeline_impact "${params[@]}"

python experiment/results_processors/pipeline_impact.py "${params[@]}"