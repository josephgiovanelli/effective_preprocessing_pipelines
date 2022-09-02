#!/bin/bash
params=()
[[ $1 == true ]] && params+=(-toy)

echo ""
echo "### EVALUATION ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios"
python experiment/scenario_generator.py -exp evaluation1 "${params[@]}"
python experiment/scenario_generator.py -exp evaluation2_3 "${params[@]}"
echo -e "\tDone."

# EXPERIMENTS
echo "Running experiments"
echo -e "\tEvaluation 1"
## Evaluation 1
python experiment/experiments_launcher.py -exp evaluation1 "${params[@]}"
## Evaluation 2 and 3
echo -e "\tEvaluation 2 and 3"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode algorithm "${params[@]}"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode pipeline_algorithm "${params[@]}"

# PLOTTING
echo "Plotting"
python experiment/results_processor.py -exp evaluation "${params[@]}"
echo -e "\tDone."
