#!/bin/bash
TOY_EXAMPLE=true

./scripts/pipeline_construction.sh $TOY_EXAMPLE

./scripts/pipeline_impact.sh $TOY_EXAMPLE

./scripts/evaluation.sh $TOY_EXAMPLE

./scripts/meta_learning.sh $TOY_EXAMPLE