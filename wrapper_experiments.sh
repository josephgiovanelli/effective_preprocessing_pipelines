#!/bin/bash
TOY_EXAMPLE=True

./scripts/pipeline_construction.sh $TOY_EXAMPLE

./scripts/pipeline_impact.sh $TOY_EXAMPLE

./scripts/evaluation.sh $TOY_EXAMPLE