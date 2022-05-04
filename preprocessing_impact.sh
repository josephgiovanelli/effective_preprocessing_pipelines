#!/bin/bash

python scenario_generator.py -exp preprocessing_impact

python experiments_launcher.py -r results/preprocessing_impact -exp preprocessing_impact -mode algorithm
python experiments_launcher.py -r results/preprocessing_impact -exp preprocessing_impact -mode algorithm_pipeline

python results_processors/preprocessing_impact_experiments_summarizer.py -ip results/preprocessing_impact/algorithm_pipeline -ia results/preprocessing_impact/algorithm -o results/preprocessing_impact