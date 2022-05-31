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
## Exploratory Analysis
python experiment/results_processors/exploratory_analysis.py "${params[@]}"
## Evaluation
python experiment/results_processors/evaluation.py "${params[@]}"
## Meta learning
# python experiment/results_processors/meta_learning_input_preparation.py "${params[@]}"

