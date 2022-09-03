#!/bin/bash

echo ""
echo "### EVALUATION ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios"
python experiment/scenario_generator.py -exp evaluation1 $1
python experiment/scenario_generator.py -exp evaluation2_3 $1
echo -e "\tDone."

# EXPERIMENTS
echo "Running experiments"
echo -e "\tExhaustive prototypes + ML algorithms"
## Evaluation 1
python experiment/experiments_launcher.py -exp evaluation1 $1
## Evaluation 2 and 3
echo -e "\tOnly ML algorithms"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode algorithm $1
echo -e "\tEffective prototypes + ML algorithms"
python experiment/experiments_launcher.py -exp evaluation2_3 -mode pipeline_algorithm $1

# PLOTTING
echo "Plotting"
python experiment/results_processor.py -exp evaluation $1
echo -e "\tDone."
