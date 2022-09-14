#!/bin/bash

./scripts/exploratory_analysis.sh $1

./scripts/prototype_construction.sh $1

./scripts/experimental_evaluation.sh $1