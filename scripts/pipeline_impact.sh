#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

# python experiment/scenario_generator.py -exp pipeline_impact "${params[@]}"

# python experiment/experiments_launcher.py -exp pipeline_impact -mode algorithm "${params[@]}"
# python experiment/experiments_launcher.py -exp pipeline_impact -mode algorithm_pipeline "${params[@]}"

# python experiment/results_processors/pipeline_impact_experiments_summarizer.py -exp pipeline_impact "${params[@]}"