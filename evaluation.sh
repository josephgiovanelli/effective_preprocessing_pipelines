#!/bin/bash

#python scenario_generator.py -exp evaluation1
#python experiments_launcher.py -r results/evaluation1 -exp evaluation1
python results_processors/evaluation_results_extraction.py
python results_processors/evaluation_results_comparison.py
python results_processors/evaluation_prototypes_impact.py