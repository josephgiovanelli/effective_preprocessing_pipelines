#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)


# SCENARIO GENERETOR
python experiment/scenario_generator.py -exp evaluation1 "${params[@]}"
python experiment/scenario_generator.py -exp evaluation2_3 "${params[@]}"

# EXPERIMENTS
## Evaluation 1
python experiment/experiments_launcher.py -exp evaluation1 "${params[@]}"
## Evaluation 2 and 3
python experiment/experiments_launcher.py -exp evaluation2_3 -mode algorithm "${params[@]}"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode pipeline_algorithm "${params[@]}"

# PLOTTING
python experiment/results_processors/evaluation.py "${params[@]}"
