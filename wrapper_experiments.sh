#!/bin/bash

./scripts/pipeline_impact.sh $1

./scripts/pipeline_construction.sh $1

./scripts/evaluation.sh $1

./scripts/exploratory_analysis.sh $1