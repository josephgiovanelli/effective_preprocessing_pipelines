#!/bin/bash

echo ""
echo "### EXPERIMENTAL EVALUATION (EE) ###"
echo ""

# SCENARIO GENERETOR
echo "Creating scenarios..."
python experiment/scenario_generator.py -exp exhaustive_prototypes $1
python experiment/scenario_generator.py -exp custom_prototypes $1
echo -e "\tDone."

# EXPERIMENTS
echo "EE01. SMBO on ML algorithms"
python experiment/experiments_launcher.py -exp custom_prototypes -mode algorithm $1
echo "EE02-EE04. SMBO on effective pre-processing prototypes and ML algorithms"
python experiment/experiments_launcher.py -exp custom_prototypes -mode pipeline_algorithm $1
echo "EE05-EE07. SMBO on exhaustive pre-processing prototypes and ML algorithms"
python experiment/experiments_launcher.py -exp exhaustive_prototypes $1

# POST-PROCESSING
python experiment/results_processor.py -exp empirical_evaluation $1
