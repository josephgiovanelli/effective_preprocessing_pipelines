#!/bin/bash

./scripts/exploratory_analysis.sh $1 $2

./scripts/prototype_construction.sh $1 $2

./scripts/experimental_evaluation.sh $1 $2