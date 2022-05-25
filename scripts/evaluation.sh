#!/bin/bash

# EVALUATION 1
# python experiment/scenario_generator.py -exp evaluation1 -toy $1
# python experiment/experiments_launcher.py -exp evaluation1 -toy $1
# python experiment/results_processors/evaluation_results_extraction.py -toy $1

# EVALUATION 2_3
# python experiment/scenario_generator.py -exp evaluation2_3 -toy $1
# python experiment/experiments_launcher.py -exp evaluation2_3 -mode algorithm -toy $1
# python experiment/experiments_launcher.py -exp evaluation2_3 -mode pipeline_algorithm -toy $1

# python experiment/results_processors/evaluation_results_comparison.py -toy $1
# python experiment/results_processors/results_comparator.py -toy $1

# python experiment/results_processors/pp_pipeline_study.py -toy $1
# python experiment/results_processors/pp_pipeline_study2.py -toy $1

# python experiment/results_processors/evaluation_prototypes_impact.py -toy $1

# python experiment/results_processors/meta_learning_input_preparation.py -toy $1

