#!/bin/bash
[ "$(ls -A /home/autoprep))" ] || mv /home/dump /home/autoprep

sh /home/autoprep/scripts/exploratory_analysis.sh $1 $2

sh /home/autoprep/scripts/prototype_construction.sh $1 $2

sh /home/autoprep/scripts/experimental_evaluation.sh $1 $2