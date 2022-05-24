#!/bin/bash

# EVALUATION 1
# python scenario_generator.py -exp evaluation1 -toy $1
# python experiments_launcher.py -r results/evaluation1 -exp evaluation1
# python results_processors/evaluation_results_extraction.py

# EVALUATION 2_3
# python scenario_generator.py -exp evaluation2_3 -toy $1
# python experiments_launcher.py -mode algorithm -r results/evaluation2_3 -exp evaluation2_3
# python experiments_launcher.py -mode pipeline_algorithm -r results/evaluation2_3 -exp evaluation2_3

# python results_processors/results_comparator.py
# python results_processors/pp_pipeline_study.py
# python results_processors/pp_pipeline_study2.py
# python results_processors/meta_learning_input_preparation.py

# python results_processors/evaluation_results_comparison.py
# python results_processors/evaluation_prototypes_impact.py
