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
echo -e "\tExhaustive prototypes + ML algorithms"
## Evaluation 1
python experiment/experiments_launcher.py -exp evaluation1 "${params[@]}"
## Evaluation 2 and 3
echo -e "\tOnly ML algorithms"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode algorithm "${params[@]}"
echo -e "\tEffective prototypes + ML algorithms"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode pipeline_algorithm "${params[@]}"

# PLOTTING
echo "Plotting"
python experiment/results_processor.py -exp evaluation "${params[@]}"
echo -e "\tDone."
