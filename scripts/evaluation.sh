#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

# EVALUATION 1
python experiment/scenario_generator.py -exp evaluation1 "${params[@]}"
python experiment/experiments_launcher.py -exp evaluation1 "${params[@]}"
python experiment/results_processors/evaluation_results_extraction.py "${params[@]}"

# EVALUATION 2_3
python experiment/scenario_generator.py -exp evaluation2_3 "${params[@]}"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode algorithm "${params[@]}"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode pipeline_algorithm "${params[@]}"

python experiment/results_processors/evaluation_results_comparison.py "${params[@]}"
python experiment/results_processors/results_comparator.py "${params[@]}"

python experiment/results_processors/exploratory_analysis.py "${params[@]}"

# python experiment/results_processors/meta_learning_input_preparation.py "${params[@]}"

